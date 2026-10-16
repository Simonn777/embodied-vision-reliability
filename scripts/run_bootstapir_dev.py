"""Development-only entry point. Run with a clean environment and timeout 900."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

from evr.bootstapir_dev import (UPSTREAM, compare, decode_development,
    download_object, sha256, stream, tree_bytes, verify_module_paths,
    verify_development, validation_status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["acquire", "decode", "measure"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimal", action="store_true", help="Fresh-process four-frame repetition only")
    parser.add_argument("--manifest", type=Path, help="Required verified decode manifest for measurement")
    parser.add_argument("--acquisition", type=Path, help="Required acquisition receipt for measurement")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    affinity = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, affinity)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "2":
            raise ValueError("All numerical-library thread variables must be 2")
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/0002-development.json"
    receipt = {"phase": args.phase, "status": "running", "python": platform.python_version(),
               "platform": platform.platform(), "affinity": affinity,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "config_sha256": sha256(config),
               "source_sha256": {p: sha256(root / p) for p in [
                   "src/evr/bootstapir_dev.py", "scripts/run_bootstapir_dev.py",
                   "protocols/0002-development-feasibility.md"]},
               "command": ["python", "scripts/run_bootstapir_dev.py", args.phase,
                           "--cache", "<cache>", "--output", "<output>"],
               "cpu_model": next(x.split(":", 1)[1].strip() for x in open("/proc/cpuinfo") if x.startswith("model name")),
               "meminfo_kib": {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith(("MemTotal:", "MemAvailable:"))},
               "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
               "address_space_limit_bytes": 6 * 1024**3,
               "disk_free_bytes": shutil.disk_usage(args.cache).free}
    try:
        if args.phase == "acquire":
            receipt["objects"] = {n: download_object(args.cache, n) for n in ["weights.pt", "rgb.zip"]}
            source = args.cache / "source"
            receipt["upstream_files"] = []
            for path in ["tapnet/torch/tapir_model.py", "tapnet/torch/nets.py", "tapnet/torch/utils.py", "LICENSE"]:
                url = f"https://raw.githubusercontent.com/google-deepmind/tapnet/{UPSTREAM}/{path}"
                body = urllib.request.urlopen(url, timeout=30).read(1024**2)
                target = source / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                receipt["upstream_files"].append({"path": path, "url": url, "bytes": len(body), "sha256": sha256(target)})
            for path in ["tapnet/__init__.py", "tapnet/torch/__init__.py"]:
                (source / path).write_text("")
        elif args.phase == "decode":
            # Validate official body again before any numeric deserialization.
            receipt["dataset"] = download_object(args.cache, "rgb.zip")
            decode_development(args.cache, args.output / "development-manifest.json")
        else:
            import numpy as np
            import torch
            receipt["actual_modules"] = verify_module_paths(args.cache / "source")
            if args.manifest is None or args.acquisition is None:
                raise ValueError("Measurement requires decode and acquisition receipts")
            acquired = json.loads(args.acquisition.read_text())
            receipt["acquisition_receipt_sha256"] = sha256(args.acquisition)
            for module, info in receipt["actual_modules"].items():
                recorded = next((f for f in acquired.get("upstream_files", [])
                                 if f["path"] == info["relative_path"]), None)
                if recorded is None or recorded["sha256"] != info["sha256"]:
                    raise ValueError("Actual import differs from acquired source")
            if torch.__version__ != "2.10.0+cpu":
                raise ValueError("Only patched pinned torch 2.10.0+cpu is approved")
            if torch.version.cuda is not None:
                raise ValueError("CUDA build forbidden")
            from tapnet.torch.tapir_model import TAPIR
            for module, info in receipt["actual_modules"].items():
                actual = Path(sys.modules[module].__file__).resolve()
                if actual != (args.cache / "source" / info["relative_path"]).resolve() or sha256(actual) != info["sha256"]:
                    raise ValueError("Imported module path/digest changed")
            torch.set_num_threads(2)
            torch.set_num_interop_threads(2)
            torch.manual_seed(20261016)
            np.random.seed(20261016)
            torch.use_deterministic_algorithms(True)
            receipt["packages"] = {p: importlib.metadata.version(p) for p in [
                "torch", "numpy", "dm-tree", "einshape", "absl-py", "attrs", "wrapt"]}
            receipt["torch_threads"] = [torch.get_num_threads(), torch.get_num_interop_threads()]
            receipt["checkpoint"] = download_object(args.cache, "weights.pt")
            model = TAPIR(pyramid_level=1, use_casual_conv=True)
            assert model.use_casual_conv is True
            state = torch.load(args.cache / "weights.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model.eval().requires_grad_(False)
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            receipt["parameter_count"] = sum(p.numel() for p in model.parameters())
            receipt["causal_convolution"] = model.use_casual_conv
            data, receipt["development_identity"] = verify_development(
                args.cache / "development-000.npz", args.manifest)
            receipt["command"].extend(["--manifest", "<decode-manifest>", "--acquisition", "<acquire-receipt>"])
            frames = data["video"][:16]
            points, occ = data["points"], data["occluded"]
            if frames.shape[1:] != (256, 256, 3) or len(frames) != 16:
                raise ValueError("Expected at least 16 frames of 256px RGB")
            candidates = np.flatnonzero(~occ[:, 0] & ~occ[:, 4])
            if len(candidates) < 2:
                raise ValueError("Need two predeclared first-index visible queries")
            ids = [int(x) for x in candidates[:2]]
            q0 = {"arrival": 0, "xy": points[ids[0], 0].tolist()}
            q1 = {"arrival": 0, "xy": points[ids[1], 0].tolist()}
            late = {"arrival": 4, "xy": points[ids[1], 4].tolist()}
            receipt["point_indices"] = ids
            receipt["queries"] = {"first": q0, "second": q1, "late_second": late}
            rows, outputs = [], {}
            def run(name, images, queries):
                begin = time.perf_counter()
                values, timings = stream(model, images, queries, chunk_size=8)
                wall = time.perf_counter() - begin
                outputs[name] = values
                for t, latency in enumerate(timings):
                    rows.append({"run": name, "frame": t, "seconds": latency})
                receipt.setdefault("runs", {})[name] = {"shape": list(values.shape),
                    "wall_seconds": wall, "first_frame_seconds": timings[0],
                    "subsequent_frame_median_seconds": float(np.median(timings[1:])),
                    "finite_values": int(np.isfinite(values).sum())}
                # Preserve partial evidence before the next stage.
                np.savez(args.output / "raw-outputs.npz", **outputs)
                (args.output / "progress.json").write_text(json.dumps(receipt, indent=2) + "\n")
                print(name, round(wall, 3), flush=True)
                return values
            run("cold_4", frames[:4], [q0])
            if args.minimal:
                receipt["minimal_only"] = True
                receipt["command"].append("--minimal")
                receipt["status"] = "completed"
                return 0
            base = run("single_16", frames, [q0])
            changed = frames.copy()
            changed[8:] = 255 - changed[8:]
            future = run("future_images_changed", changed, [q0])
            prefix = run("prefix_8", frames[:8], [q0])
            batch = run("batch_16", frames, [q0, q1])
            late_out = run("late_query_16", frames, [q0, late])
            cropped_late = run("independent_late_12", frames[4:], [{"arrival": 0, "xy": late["xy"]}])
            repeated = run("repeat_single_16", frames, [q0])
            receipt["checks"] = {"future_image_prefix": compare(base[:8], future[:8]),
                "truncated_prefix": compare(base[:8], prefix),
                "single_vs_batch": compare(base[:, 0], batch[:, 0]),
                "future_query_past": compare(base[:4, 0], late_out[:4, 0]),
                "existing_query_after_arrival": compare(base[:, 0], late_out[:, 0]),
                "late_query_vs_independent_crop": compare(late_out[4:, 1], cropped_late[:, 0]),
                "repeat": compare(base, repeated),
                "prequery_mask": {"passed": bool(np.isnan(late_out[:4, 1]).all())}}
            with (args.output / "latencies.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=["run", "frame", "seconds"])
                writer.writeheader(); writer.writerows(rows)
            receipt["all_checks_passed"] = all(v["passed"] for v in receipt["checks"].values())
            receipt["status"] = validation_status(receipt["checks"])
        if receipt["status"] == "running":
            receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["cache_bytes_at_end"] = tree_bytes(args.cache)
        (args.output / (args.phase + "-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
