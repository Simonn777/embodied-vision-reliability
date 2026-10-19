"""One-shot known-control cohort; never open labels or choose a winning branch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = HERE.parent / "0013-pose-headroom"
sys.path.insert(0, str(LEGACY))
import run_baseline as infrastructure
import evaluate_sealed as baseline_seal

NAMES = ("refine.py", "predict_pair.py", "run_controls.py", "evaluate_controls.py")


def sha(path):
    return infrastructure.worker.sha256(path)


def bindings():
    return {**{str((HERE / name).relative_to(REPO)): sha(HERE / name) for name in NAMES},
            **{str(path.relative_to(REPO)): sha(path) for path in
               (LEGACY / "predict_pair.py", LEGACY / "run_baseline.py", LEGACY / "evaluate_sealed.py")},
            "protocols/0014-pixel-noise-controls.md": sha(REPO / "protocols/0014-pixel-noise-controls.md")}


def run(data, manifest, original_seal, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = bindings()
    started = time.perf_counter()
    infrastructure.write_new(output_dir / "started.json", {
        "schema": "evr0014.operation.v1", "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sources, "pair_timeout_seconds": 90, "total_timeout_seconds": 1800,
        "command": "python experiments/0014-pixel-noise/run_controls.py --data <cache>/megadepth1500_dkm.h5 --manifest configs/0012-development-manifest.json --baseline-seal results/0013-baseline-calibration-compatibility/predictions/seal.json --output-dir <results>/predictions",
    })
    rows, error, mh, bh, metadata = [], None, None, None, {}
    try:
        metadata, old_seal, original_records, mh, bh = baseline_seal.verify_seal(manifest, original_seal)
        if Path(data).stat().st_size != metadata["dataset_bytes"] or sha(data) != metadata["dataset_sha256"]:
            raise ValueError("actual data does not match original sealed baseline")
        library = infrastructure.expected_library()
        infrastructure.write_new(output_dir / "integrity.json", {
            "dataset_sha256_verified": metadata["dataset_sha256"], "library": library,
            "baseline_seal_sha256": bh, "manifest_sha256": mh, "numeric_arrays_decoded": False})
        for index, (old_row, old_prediction) in enumerate(original_records):
            remaining = 1800 - (time.perf_counter() - started)
            if remaining <= 0:
                raise RuntimeError("total operation timeout")
            stem = "pair-%02d" % index
            target = output_dir / (stem + ".json")
            process = infrastructure.run_child([sys.executable, str(HERE / "predict_pair.py"),
                "--data", str(Path(data).resolve()), "--manifest", str(Path(manifest).resolve()),
                "--baseline-seal", str(Path(original_seal).resolve()),
                "--pair-index", str(index), "--output", str(target.resolve())],
                output_dir / (stem + ".log"), remaining)
            infrastructure.write_new(output_dir / (stem + ".process.json"), process)
            if process["exit_code"] != 0 or process["termination"] is not None:
                raise RuntimeError("continuation subprocess failed at pair %d" % index)
            record, digest = baseline_seal.load_hashed_json(target)
            expected = {"schema": "evr0014.prediction.v1", "status": "completed",
                        "pair_index": index, "key": old_row["key"],
                        "source_sha256": sources["experiments/0014-pixel-noise/predict_pair.py"],
                        "kernel_sha256": sources["experiments/0014-pixel-noise/refine.py"],
                        "baseline_seal_sha256": bh, "baseline_prediction_sha256": old_row["sha256"]}
            for name, value in expected.items():
                if record.get(name) != value:
                    raise ValueError("continuation record binding differs: " + name)
            if record.get("library") != library:
                raise ValueError("continuation library differs from pinned original")
            if record.get("calibration") != old_prediction["calibration"]:
                raise ValueError("continuation changed calibration")
            mask = old_prediction["returned_ransac_stats"]["inliers"]
            if (record.get("mask_sha256") != hashlib.sha256(bytes(mask)).hexdigest()
                    or record.get("kept_count") != sum(mask)):
                raise ValueError("continuation changed the original inlier mask")
            for mode in ("native", "N", "P"):
                estimate = record["estimates"][mode]
                baseline_seal.valid_rotation(estimate["R"])
                baseline_seal.unit_translation(estimate["t"])
            rows.append({"pair_index": index, "key": old_row["key"],
                         "path": target.name, "sha256": digest, "status": "completed"})
            print(json.dumps({"pair_index": index, "status": "completed",
                              "wall_seconds": process["wall_seconds"]}), flush=True)
        if bindings() != sources or sha(manifest) != mh or sha(original_seal) != bh:
            raise ValueError("a frozen input changed during the operation")
    except Exception as exc:
        message = str(exc)
        for path in (data, manifest, original_seal, output_dir):
            message = message.replace(str(Path(path).resolve()), "<local-path>")
        error = {"type": type(exc).__name__, "message": message}
    complete = len(rows) == 16 and error is None
    infrastructure.write_new(output_dir / "seal.json", {
        "schema": "evr0014.seal.v1", "complete": complete,
        "eligible_for_evaluation": complete, "source_sha256": sources,
        "manifest_sha256": mh, "baseline_seal_sha256": bh,
        "dataset_sha256_declared": metadata.get("dataset_sha256"),
        "rows": rows, "error": error, "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"complete": complete, "rows": len(rows), "error": error}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "baseline-seal", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.data, args.manifest, args.baseline_seal, args.output_dir))
