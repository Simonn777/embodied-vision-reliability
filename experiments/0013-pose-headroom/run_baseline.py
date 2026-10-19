"""Run the one-shot baseline cohort and seal predictions, without GT access."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import predict_pair as worker

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[1]
SOURCE_NAMES = ("predict_pair.py", "evaluate_sealed.py", "run_baseline.py")
PINNED_LIBRARY = {
    "version": "2.0.5", "module_version": "2.0.5",
    "binary_name": "_core.cpython-39-darwin.so", "binary_bytes": 2319704,
    "binary_sha256": "2e937f7a5289438710cbf00bc29fbc5583033b308beedd1da5f0634b527a4e26",
    "module_sha256": "983ff88910dcfa2cb95eb8b50bc8478d6b4153ca05c5b018928c14f07f3830af",
    "official_wheel_sha256": worker.WHEEL_SHA256,
}
PAIR_TIMEOUT = 90.0
TOTAL_TIMEOUT = 1800.0
RSS_LIMIT_KIB = 8 * 1024 * 1024


def write_new(path, value):
    encoded = json.dumps(worker.json_safe(value), indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def source_hashes():
    return {name: worker.sha256(SOURCE_DIR / name) for name in SOURCE_NAMES}


def expected_library():
    import poselib
    import h5py
    result = {**worker.library_metadata(poselib), "h5py_version": h5py.__version__}
    if any(result.get(k) != v for k, v in PINNED_LIBRARY.items()):
        raise ValueError("installed baseline differs from pinned official wheel")
    return result


def verify_record(record, identity, library):
    for name, expected in identity.items():
        if record.get(name) != expected:
            raise ValueError("worker binding mismatch: " + name)
    if record.get("settings") != worker.SETTINGS:
        raise ValueError("worker settings mismatch")
    if record.get("library") != library:
        raise ValueError("worker library mismatch")
    if record.get("status") == "completed":
        # Validate without importing the evaluator or opening a label handle.
        import numpy as np
        estimate = record.get("estimate", {})
        R = worker.finite_array(estimate.get("R"), (3, 3), "R")
        t = worker.finite_array(estimate.get("t"), (3,), "t")
        if (not np.any(t != 0) or not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-6)
                or abs(np.linalg.det(R) - 1) > 1e-6):
            raise ValueError("invalid completed pose")
    worker.json_safe(record)


def wrap_record(raw, identity, library, process):
    """Failed calibration/imports are never disguised as estimator coverage."""
    verify_record(raw, identity, library)
    status = raw.get("status")
    if status == "completed" and process["exit_code"] == 0 and not process["termination"]:
        final_status = "completed"
    elif (status == "error" and raw.get("failure_kind") == "estimator"
          and process["exit_code"] == 1 and not process["termination"]):
        final_status = "estimator_error"
    elif status == "started" and raw.get("phase") == "estimator":
        if process["termination"] == "pair_timeout":
            final_status = "timeout"
        elif process["termination"] is None and process["exit_code"] != 0:
            final_status = "estimator_error"
        else:
            raise ValueError("resource/global interruption is not evaluation eligible")
    else:
        raise ValueError("non-estimator failure or inconsistent worker exit")
    return {**raw, "status": final_status, "process_receipt": process}


def run_child(command, log_path, remaining_seconds):
    env = dict(os.environ)
    for name in worker.THREAD_VARIABLES:
        env[name] = "1"
    start = time.perf_counter()
    deadline = start + min(PAIR_TIMEOUT, remaining_seconds)
    termination, peak_rss, watchdog_error = None, 0, None
    with Path(log_path).open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            while process.poll() is None:
                if time.perf_counter() >= deadline:
                    termination = "pair_timeout" if remaining_seconds >= PAIR_TIMEOUT else "total_timeout"
                    break
                # macOS reports RSS in KiB. No grandchildren are created by this worker.
                result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                                        capture_output=True, text=True, timeout=2)
                value = result.stdout.strip()
                if value.isdigit():
                    peak_rss = max(peak_rss, int(value))
                    if peak_rss > RSS_LIMIT_KIB:
                        termination = "rss_limit"
                        break
                time.sleep(0.1)
        except Exception as exc:
            termination = "watchdog_error"
            watchdog_error = type(exc).__name__
        finally:
            # Also executes on KeyboardInterrupt: never abandon an unmonitored child.
            if process.poll() is None:
                process.kill()
            code = process.wait(timeout=5)
    return {"exit_code": code, "termination": termination,
            "wall_seconds": time.perf_counter() - start,
            "sampled_peak_rss_kib": peak_rss, "watchdog_error": watchdog_error,
            "rss_sampling": "ps approximately 0.1-second intervals, may miss peaks"}


def run(data, manifest_path, output_dir):
    # A pre-existing directory is an unconditional refusal, before any data read.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = source_hashes()
    protocol_hash = worker.sha256(REPO_ROOT / "protocols/0013-baseline-pose-headroom.md")
    write_new(output_dir / "started.json", {
        "schema": "evr0013.operation.v1", "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source, "protocol_sha256": protocol_hash,
        "pair_timeout_seconds": PAIR_TIMEOUT, "total_timeout_seconds": TOTAL_TIMEOUT,
        "rss_limit_kib": RSS_LIMIT_KIB, "settings": worker.SETTINGS,
        "command": "python experiments/0013-pose-headroom/run_baseline.py --data <cache>/megadepth1500_dkm.h5 --manifest configs/0012-development-manifest.json --output-dir <results>/predictions",
    })
    rows, error, manifest, manifest_hash = [], None, {}, None
    try:
        manifest, _ = worker.load_manifest(manifest_path, 0)
        manifest_hash = worker.sha256(manifest_path)
        if Path(data).stat().st_size != manifest["dataset_bytes"]:
            raise ValueError("dataset size differs")
        if worker.sha256(data) != manifest["dataset_sha256"]:
            raise ValueError("dataset content hash differs")
        library = expected_library()
        write_new(output_dir / "integrity.json", {
            "dataset_sha256_verified": manifest["dataset_sha256"],
            "dataset_bytes": manifest["dataset_bytes"], "manifest_sha256": manifest_hash,
            "library": library, "numeric_arrays_decoded": False,
        })
        for index, key in enumerate(manifest["selected_pair_keys"]):
            remaining = TOTAL_TIMEOUT - (time.perf_counter() - start)
            if remaining <= 0:
                raise RuntimeError("total operation timeout")
            stem = "pair-%02d" % index
            raw_path = output_dir / (stem + ".worker.json")
            final_path = output_dir / (stem + ".json")
            identity = {"schema": worker.SCHEMA, "pair_index": index, "key": key,
                        "manifest_sha256": manifest_hash,
                        "dataset_sha256_declared": manifest["dataset_sha256"],
                        "source_sha256": source["predict_pair.py"]}
            process = run_child([sys.executable, str(SOURCE_DIR / "predict_pair.py"),
                "--data", str(Path(data).resolve()), "--manifest", str(Path(manifest_path).resolve()),
                "--pair-index", str(index), "--output", str(raw_path.resolve())],
                output_dir / (stem + ".log"), remaining)
            process["raw_worker_path"] = raw_path.name
            process["raw_worker_sha256"] = worker.sha256(raw_path) if raw_path.exists() else None
            write_new(output_dir / (stem + ".process.json"), process)
            raw = json.loads(raw_path.read_text())
            final = wrap_record(raw, identity, library, process)
            write_new(final_path, final)
            rows.append({"pair_index": index, "key": key, "path": final_path.name,
                         "sha256": worker.sha256(final_path), "status": final["status"]})
            print(json.dumps({"pair_index": index, "status": final["status"],
                              "wall_seconds": process["wall_seconds"]}), flush=True)
        if source_hashes() != source or worker.sha256(manifest_path) != manifest_hash:
            raise ValueError("source or manifest changed during prediction")
        if worker.sha256(REPO_ROOT / "protocols/0013-baseline-pose-headroom.md") != protocol_hash:
            raise ValueError("protocol changed during prediction")
    except Exception as exc:
        # Error messages here contain labels, not input payloads or private paths.
        error = {"type": type(exc).__name__, "message": str(exc)}
        for private in (str(Path(data).resolve()), str(Path(manifest_path).resolve()), str(output_dir.resolve())):
            error["message"] = error["message"].replace(private, "<local-path>")
    complete = len(rows) == 16 and error is None
    seal = {"schema": "evr0013.seal.v1", "complete": complete,
            "eligible_for_evaluation": complete, "manifest_sha256": manifest_hash,
            "dataset_sha256_declared": manifest.get("dataset_sha256"),
            "source_sha256": source, "protocol_sha256": protocol_hash,
            "rows": rows, "error": error, "wall_seconds": time.perf_counter() - start}
    write_new(output_dir / "seal.json", seal)
    print(json.dumps({"complete": complete, "rows": len(rows), "error": error}), flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return run(args.data, args.manifest, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
