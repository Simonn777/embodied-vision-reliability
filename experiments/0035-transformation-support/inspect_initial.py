"""Read only the four acquired initial masks; no RGB, later labels or model."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run(path):
    spec = json.loads(Path(path).read_text())
    common = module("initial_common", spec["common_path"])
    admit = module("initial_admit", spec["admit_path"])
    require = common.require
    out = Path(spec["output"])
    require(not out.exists(), "fresh initial inspection required")
    out.mkdir(parents=True)
    began = time.perf_counter()
    record = dict(schema="evr0035.initial-inspection.v1", status="started", eligible=False,
                  spec_sha256=common.sha(path), source_sha256=spec["source_sha256"],
                  initial_only=True, noninitial_labels_read=0, model_calls=0, rows=[],
                  offline_attempts=[], compatibility_probes=[])
    def save():
        record["wall_seconds"] = time.perf_counter() - began
        common.save(out / "receipt.json", record)
    save()
    try:
        require(sys.flags.no_site == 1 and sys.version_info[:3] == (3, 12, 13)
                and str(Path(sys.executable).absolute()) == spec["qualified_python"], "qualified Python")
        for name, digest in spec["source_sha256"].items():
            require(common.sha(name) == digest, "source changed")
        fingerprints = {}
        require(spec["manifest"]["sha256"] == admit.MANIFEST_SHA256, "fixed initial manifest")
        manifest = admit.bound_json(spec["manifest"], fingerprints)
        acquisition, _ = admit.acquisition(spec["initial_acquisition"], spec["initial_outer"],
                                           "initial", manifest, fingerprints, spec["repo"])
        require(acquisition["schema"] == "evr0035.vost-acquisition.v1"
                and acquisition["status"] == "completed" and acquisition["phase"] == "initial"
                and acquisition["index_sha256"]["manifest"] == admit.MANIFEST_SHA256
                and len(acquisition["members"]) == 4, "complete initial acquisition")
        guard = common.module("initial_guard", spec["guard_source"])
        guard.install_guard(record["offline_attempts"], record["compatibility_probes"])
        import site
        site.main()
        import numpy as np
        from PIL import Image
        metrics = module("initial_metrics", spec["metrics_path"])
        root = Path(spec["initial_acquisition"]["path"]).parent
        by_sequence = {m["sequence"]: m for m in acquisition["members"]}
        require(set(by_sequence) == {n for n, _ in admit.COHORT}, "four complete initial IDs")
        total_calls, total_raw, peak_estimate = 0, 0, 0
        for name, count in admit.COHORT:
            item = by_sequence[name]
            require(item["phase"] == "initial" and item["frame_index"] == 0, "initial only")
            source = admit.relative(root, item["saved_relative"])
            require(source.stat().st_size == item["bytes"] and common.sha(source) == item["sha256"], "initial bytes")
            fingerprints[str(source)] = item["sha256"]
            with Image.open(source) as image:
                require(image.format == "PNG" and image.mode in ("L", "P"), "indexed PNG")
                array = np.array(image)
            require(array.ndim == 2 and array.dtype == np.uint8, "2D uint8 labels")
            ids = metrics.initial_object_ids(array)
            height, width = array.shape
            pixels, objects = height * width, len(ids)
            # Uncompressed float banks/native B and four uint8 maps, with
            # 16 KiB/frame metadata allowance and 1% ZIP worst-case allowance.
            raw = count * (objects * (3*256*256*4 + pixels*4) + 4*pixels + 16384)
            # Conservative planning estimate, not a measured memory guarantee:
            # full resized video, compact histories, model/workspace reserve,
            # multiple live original-scale render/compose buffers and labels.
            estimated_memory = count*3*1024*1024*4 + count*objects*1024*1024 + 2*1024**3 + pixels*(24*objects+64)
            row = dict(sequence=name, frames=count, shape=[height, width], object_ids=list(ids),
                       object_frames=count*objects, raw_output_bound=raw,
                       planning_peak_memory_bytes=estimated_memory,
                       original_void_pixels=int((array == 255).sum()))
            record["rows"].append(row)
            total_calls += row["object_frames"]
            total_raw += raw
            peak_estimate = max(peak_estimate, estimated_memory)
            del array
            gc.collect()
            save()
        output_cap = (total_raw * 101 + 99)//100 + 64*1024**2
        free = shutil.disk_usage(out).free
        checks = dict(native_object_frames_at_most_2000=total_calls <= 2000,
                      planning_memory_at_most_7GiB=peak_estimate <= 7*1024**3,
                      prediction_output_at_most_20GiB=output_cap <= 20*1024**3,
                      free_disk_covers_twice_output_plus_5GiB=free >= 2*output_cap+5*1024**3)
        record.update(checks=checks, expected_native_object_frames=total_calls,
                      prediction_output_cap_bytes=output_cap, free_disk_bytes=free,
                      planning_memory_peak_bytes=peak_estimate,
                      measured_inference_feasibility=False,
                      fingerprints=fingerprints)
        require(not record["offline_attempts"], "network attempt")
        require(all(common.sha(p) == digest for p, digest in fingerprints.items()), "initial inputs changed")
        require(all(common.sha(p) == digest for p, digest in spec["source_sha256"].items()), "source changed")
        record.update(status="completed", eligible=all(checks.values()))
    except BaseException as error:
        record.update(status="failed", eligible=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    value = run(parser.parse_args().spec)
    print(json.dumps({k: value[k] for k in ("status", "eligible", "wall_seconds")}), flush=True)
