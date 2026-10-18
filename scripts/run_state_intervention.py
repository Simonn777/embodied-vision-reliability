"""One cached-data, bounded CPU mechanism diagnostic; never downloads assets."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback


INPUTS = (
    "configs/0004-stage0.json", "protocols/0004-state-intervention-stage0.md",
    "scripts/run_state_intervention.py", "src/evr/state_intervention.py",
    "src/evr/state_intervention_metrics.py", "src/evr/bootstapir_dev.py",
    "src/evr/__init__.py", "results/0002-development/decode/development-manifest.json",
)
FIELDS = ("branch", "frame", "point_index", "shadow_start", "shadow_offset",
          "x", "y", "occlusion_logit", "expected_distance_logit")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tree_digest(value):
    """Fingerprint immutable feature trees or next context without saving tensors."""
    h = hashlib.sha256()
    def visit(x):
        if hasattr(x, "detach"):
            h.update(repr((tuple(x.shape), str(x.dtype), str(x.device))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(str(type(x).__name__).encode())
            for y in x:
                visit(y)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def timeout_handler(signum, frame):
    raise TimeoutError("bounded Stage 0 deadline reached")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = args.cache.resolve()
    # Never overwrite a prior attempt or manufacture a replacement marker.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {"status": "preflight", "experiment": "0004-stage0",
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": sys.argv, "python": platform.python_version(),
               "platform": platform.platform(), "seed": 20261018,
               "scientific_claim": "development mechanism diagnostic only"}
    controller = None
    try:
        freeze_path = root / "protocols/0004-stage0-freeze.json"
        if sha(freeze_path) != args.freeze_sha256:
            raise ValueError("Freeze receipt digest differs from reviewed publication")
        freeze = json.loads(freeze_path.read_text())
        if set(freeze["files"]) != set(INPUTS):
            raise ValueError("Freeze input set incomplete or unexpected")
        source_commit = freeze["source_commit"]
        if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
            raise ValueError("Expected full source commit")
        for path, digest in freeze["files"].items():
            committed = subprocess.check_output(["git", "show", f"{source_commit}:{path}"], cwd=root)
            if hashlib.sha256(committed).hexdigest() != digest or sha(root / path) != digest:
                raise ValueError("Frozen input differs: " + path)
        receipt.update(source_commit=source_commit, source_sha256=freeze["files"],
                       freeze_sha256=args.freeze_sha256,
                       checkout_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
        config = json.loads((root / "configs/0004-stage0.json").read_text())
        resource.setrlimit(resource.RLIMIT_AS, (config["address_space_bytes"],) * 2)
        affinity = sorted(os.sched_getaffinity(0))[:config["threads"]]
        os.sched_setaffinity(0, affinity)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            if os.environ.get(name) != "2":
                raise ValueError("Thread variable must be two: " + name)
        # The outer cloud supervisor also enforces 900 s with forced termination.
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.signal(signal.SIGTERM, timeout_handler)
        signal.alarm(config["hard_timeout_seconds"] - 10)
        for filename, key in (("weights.pt", "checkpoint_sha256"),
                              ("development-000.npz", "development_npz_sha256")):
            if not (cache / filename).is_file() or sha(cache / filename) != config[key]:
                raise ValueError("Existing cache missing or changed: " + filename)
        receipt["cache_objects"] = {name: {"bytes": (cache/name).stat().st_size, "sha256": sha(cache/name)}
                                    for name in ("weights.pt", "development-000.npz")}
        versions = {name: importlib.metadata.version(name) for name in config["packages"]}
        if versions != config["packages"]:
            raise ValueError("Pinned package versions changed: " + str(versions))
        receipt["packages"] = versions
        import numpy as np
        import torch
        from evr.bootstapir_dev import pixel_query, verify_development, verify_module_paths
        from evr.state_intervention import StateIntervention
        from evr.state_intervention_metrics import summarize
        receipt["upstream_modules"] = verify_module_paths(cache / "source")
        if torch.__version__ != "2.10.0+cpu" or torch.version.cuda is not None:
            raise ValueError("Only the verified CPU torch build is allowed")
        receipt["imports"] = {name: {"file": sys.modules[name].__file__, "sha256": sha(sys.modules[name].__file__)}
                              for name in ("numpy", "torch", "evr.state_intervention", "evr.state_intervention_metrics")}
        from tapnet.torch.tapir_model import TAPIR
        for module, info in receipt["upstream_modules"].items():
            actual = Path(sys.modules[module].__file__).resolve()
            if actual != cache / "source" / info["relative_path"] or sha(actual) != info["sha256"]:
                raise ValueError("Unexpected actual model import")
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        torch.manual_seed(config["seed"]); np.random.seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        receipt.update(affinity=affinity, address_space_bytes=config["address_space_bytes"],
                       torch_threads=[torch.get_num_threads(), torch.get_num_interop_threads()],
                       cpu_model=next(x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")),
                       cgroup_memory_max=Path("/sys/fs/cgroup/memory.max").read_text().strip())
        # A cache-wide exclusive marker protects this new operation across checkouts.
        marker = cache / "0004-stage0.started.json"
        with marker.open("x") as stream:
            json.dump({"started_utc": receipt["started_utc"], "source_commit": source_commit,
                       "freeze_sha256": args.freeze_sha256, "output": str(args.output.resolve()),
                       "pid": os.getpid()}, stream, indent=2)
        receipt["marker_sha256"] = sha(marker)
        receipt["status"] = "running"
        save(args.output / "receipt.json", receipt)
        data, receipt["development_identity"] = verify_development(cache / "development-000.npz",
            root / "results/0002-development/decode/development-manifest.json")
        frames, points, occluded = data["video"], data["points"], data["occluded"]
        if frames.shape != (config["frames"], 256, 256, 3):
            raise ValueError("Unexpected full development video shape")
        ids = np.flatnonzero(~occluded[:, 0])[:config["queries"]].tolist()
        if len(ids) != config["queries"]:
            raise ValueError("Not enough prescribed frame-zero queries")
        receipt["point_indices"] = ids
        model = TAPIR(pyramid_level=1, use_casual_conv=True)
        weights = torch.load(cache / "weights.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True); del weights
        model.eval().requires_grad_(False)
        receipt["parameter_count"] = sum(p.numel() for p in model.parameters())
        buffer_hash = tree_digest(dict(model.named_buffers()))
        query_features = None
        rows, timings, controls = [], [], []

        def validate(prediction, context):
            values = np.asarray(prediction["values"])
            if values.shape != (len(ids), 4) or not np.isfinite(values).all():
                raise ValueError("Prediction shape/nonfinite failure")
            for layer in context:
                for tensor in layer.values():
                    if not torch.isfinite(tensor).all():
                        raise ValueError("Nonfinite context")

        def check_equal(a, b, label):
            difference = float(np.max(np.abs(np.asarray(a["values"]) - np.asarray(b["values"]))))
            passed = difference <= config["clone_atol"] and a["next_context_sha256"] == b["next_context_sha256"]
            controls.append({"check": label, "frame": t, "max_abs_difference": difference,
                             "same_next_context": a["next_context_sha256"] == b["next_context_sha256"], "passed": passed})
            if not passed:
                raise ValueError("Matched branch control failed: " + label)

        def write_rows(writer, branch, frame, prediction, start=-1, offset=-1):
            for i, point in enumerate(ids):
                value = prediction["values"][i]
                row = dict(zip(FIELDS, (branch, frame, point, start, offset, *value)))
                writer.writerow(row); rows.append(row)

        with (args.output / "predictions.csv").open("x", newline="") as raw, torch.inference_mode():
            writer = csv.DictWriter(raw, fieldnames=FIELDS); writer.writeheader()
            for t, image in enumerate(frames):
                frame_started = time.perf_counter()
                tensor = torch.from_numpy(image.copy()).float()[None, None] / 255 * 2 - 1
                encoder_started = time.perf_counter()
                grids = model.get_feature_grids(tensor, is_training=False)
                encoder_seconds = time.perf_counter() - encoder_started
                if t == 0:
                    queries = np.stack([pixel_query(points[i, 0], 0, 256, 256) for i in ids])
                    query_features = model.get_query_features(tensor, is_training=False,
                        query_points=torch.from_numpy(queries)[None], feature_grids=grids)
                    query_hash = tree_digest(query_features)
                    initial = model.construct_initial_causal_state(len(ids), len(query_features.resolutions) - 1)
                    controller = StateIntervention(initial, total_frames=config["frames"],
                        shadow_starts=config["shadow_starts"], shadow_horizon=config["shadow_horizon"],
                        control_frames=config["control_frames"], validate_result=validate)
                features_hash = tree_digest(grids)
                call_seconds = []
                # This callback contains no GT, future image, query update or action selector.
                def infer_current(context):
                    call_started = time.perf_counter()
                    output = model.estimate_trajectories((256, 256), is_training=False,
                        feature_grids=grids, query_features=query_features,
                        query_points_in_video=None, query_chunk_size=config["query_chunk_size"],
                        causal_context=context, get_causal_context=True)
                    call_seconds.append(time.perf_counter() - call_started)
                    values = np.column_stack((output["tracks"][-1][0, :, 0].cpu().numpy(),
                        output["occlusion"][-1][0, :, 0].cpu().numpy(),
                        output["expected_dist"][-1][0, :, 0].cpu().numpy()))
                    context = output["causal_context"]
                    return {"values": values.tolist(), "next_context_sha256": tree_digest(context)}, context
                branches = controller.step(t, infer_current)
                # Raw numeric evidence is written even if a subsequent control fails.
                write_rows(writer, "keep", t, branches.keep)
                write_rows(writer, "zero", t, branches.zero)
                if branches.clone_control is not None:
                    write_rows(writer, "clone_control", t, branches.clone_control)
                for shadow in branches.shadows:
                    write_rows(writer, "shadow", t, shadow.prediction, shadow.start, shadow.offset)
                raw.flush()
                if branches.clone_control is not None:
                    check_equal(branches.keep, branches.clone_control, "same_state_clone")
                if t == 0:
                    check_equal(branches.keep, branches.zero, "initial_zero")
                for shadow in branches.shadows:
                    if shadow.offset == 0:
                        check_equal(branches.zero, shadow.prediction, "shadow_start_zero")
                if tree_digest(query_features) != query_hash or tree_digest(grids) != features_hash:
                    raise ValueError("Query identity or current frame features mutated")
                if tree_digest(dict(model.named_buffers())) != buffer_hash:
                    raise ValueError("Frozen model buffers mutated")
                timings.append({"frame": t, "encoder_seconds": encoder_seconds,
                                "refiner_seconds": call_seconds,
                                "whole_frame_seconds": time.perf_counter() - frame_started})
                receipt.update(completed_frames=t+1, counts=controller.counts, controls=controls)
                save(args.output / "timings.json", timings)
                save(args.output / "receipt.json", receipt)
                if t % 25 == 0:
                    print(json.dumps({"frame": t, "elapsed_seconds": time.perf_counter()-started}), flush=True)
        receipt["counts"] = controller.finish()
        receipt["predictions_sha256_before_scoring"] = sha(args.output / "predictions.csv")
        receipt["query_features_sha256"] = query_hash
        # Labels enter only this separate, after-inference descriptive summary.
        summary = summarize(rows, points, occluded, ids, thresholds=config["pixel_thresholds"],
                            shadow_starts=config["shadow_starts"], shadow_horizon=config["shadow_horizon"],
                            control_frames=config["control_frames"], control_atol=config["clone_atol"])
        save(args.output / "summary.json", summary)
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        signal.alarm(0)
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if controller is not None:
            receipt["counts"] = controller.counts
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["artifacts"] = {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "receipt.json"}
        save(args.output / "receipt.json", receipt)
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
