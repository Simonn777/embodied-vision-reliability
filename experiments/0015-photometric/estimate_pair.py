"""0015 geometric controls from sealed photometric coordinates; no image/GT IO.

The parent verifies the complete 16-pair photometric seal before invoking this
worker. This worker independently binds its one photometric record by SHA256,
checks the complete 0013 baseline seal, and calls its unchanged estimator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

SOURCE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SOURCE_DIR.parent / "0013-pose-headroom"
REPO_ROOT = SOURCE_DIR.parents[1]
PROTOCOL_PATH = REPO_ROOT / "protocols" / "0015-photometric-information-development.md"
SCHEMA = "evr0015.geometric.v1"
ARM_ORDER = ("subset_original", "subset_refined", "full_refined")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    return (_module("_evr0013_predict_for0015geometry", BASELINE_DIR / "predict_pair.py"),
            _module("_evr0013_seal_for0015geometry", BASELINE_DIR / "evaluate_sealed.py"))


def source_bindings(base):
    return {"source_sha256": base.sha256(SOURCE_DIR / "predict_pair.py"),
            "kernel_sha256": base.sha256(SOURCE_DIR / "refine_patch.py"),
            "selector_sha256": base.sha256(SOURCE_DIR / "select_queries.py"),
            "protocol_sha256": base.sha256(PROTOCOL_PATH)}


def expected_photometric_metadata(base, verifier, manifest, identity):
    """Reuse source/receipt validation only; never decode or open an image."""
    photo = _module("_evr0015_photo_metadata_for_geometry", SOURCE_DIR / "predict_pair.py")
    acq = _module("_evr0015_acq_metadata_for_geometry", SOURCE_DIR / "acquire_images.py")
    selector = _module("_evr0015_selector_metadata_for_geometry", SOURCE_DIR / "select_queries.py")
    kernel = _module("_evr0015_kernel_metadata_for_geometry", SOURCE_DIR / "refine_patch.py")
    _, binding = photo.image_binding(base, verifier, acq, manifest, identity["manifest_sha256"],
                                     photo.ACQUISITION_PATH, identity["key"])
    return {**binding, "settings": photo.SETTINGS, "kernel_settings": kernel.SETTINGS,
            "selector_settings": selector.SETTINGS}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def write_record(stream, record):
    encoded = json.dumps(record, allow_nan=False, sort_keys=True)+"\n"
    stream.seek(0)
    stream.write(encoded)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def verify_inputs(base, verifier, manifest_path, baseline_seal_path, photo_path,
                  photo_sha256, pair_index):
    manifest, seal, rows, manifest_sha, seal_sha = verifier.verify_seal(manifest_path, baseline_seal_path)
    # The complete baseline is checked before this selected record is used.
    validated_manifest, key = base.load_manifest(manifest_path, pair_index)
    _require(validated_manifest == manifest, "manifest loaders disagree")
    row, baseline = rows[pair_index]
    _require(baseline.get("status") == "completed" and baseline.get("settings") == base.SETTINGS,
             "baseline must be completed with exact 0013 settings")
    mask = baseline.get("returned_ransac_stats", {}).get("inliers")
    _require(isinstance(mask, list) and len(mask) == 10000 and all(type(v) is bool for v in mask),
             "baseline mask must be 10000 booleans")
    identity = {"pair_index": pair_index, "key": key, "manifest_sha256": manifest_sha,
                "baseline_seal_sha256": seal_sha, "baseline_prediction_sha256": row["sha256"],
                "dataset_sha256_declared": manifest["dataset_sha256"],
                "mask_sha256": hashlib.sha256(bytes(mask)).hexdigest(), "kept_count": sum(mask)}
    photo, observed_sha = verifier.load_hashed_json(photo_path)
    _require(observed_sha == photo_sha256, "photometric record SHA256 mismatch")
    _require(photo.get("schema") == "evr0015.photometric.v1" and photo.get("status") == "completed",
             "photometric record is not completed evr0015.photometric.v1")
    _require(all(photo.get(k) == v for k, v in identity.items()), "photometric baseline/manifest identity mismatch")
    bindings = source_bindings(base)
    _require(all(photo.get(k) == v for k, v in bindings.items()), "photometric source or protocol mismatch")
    metadata = expected_photometric_metadata(base, verifier, manifest, identity)
    _require(all(photo.get(k) == v for k, v in metadata.items()), "photometric image receipt or settings mismatch")
    _require(photo.get("library") == baseline.get("library"), "photometric/baseline library mismatch")
    _require(photo.get("calibration") == baseline.get("calibration"), "photometric/baseline calibration mismatch")
    base.json_safe(photo)  # All nested metadata must be finite, not only targets.
    total = photo.get("timing_seconds", {}).get("total")
    _require(type(total) in (int, float) and math.isfinite(total) and total >= 0,
             "photometric total timing must be finite and nonnegative")
    queries = photo.get("queries")
    indices = photo.get("selected_indices")
    _require(isinstance(queries, list) and isinstance(indices, list) and len(queries) == len(indices) <= 256,
             "invalid query/selected-index lists")
    _require(all(type(i) is int and 0 <= i < 10000 for i in indices)
             and len(indices) == len(set(indices)), "query indices must be unique original indices")
    _require(all(mask[i] for i in indices), "query outside sealed original inlier mask")
    _require(photo.get("selected_count") == len(indices), "selected count mismatch")
    admitted, changed, initialized = [], 0, 0
    for index, query in zip(indices, queries):
        _require(isinstance(query, dict) and type(query.get("index")) is int and query["index"] == index,
                 "query index/order mismatch")
        _require(type(query.get("admitted")) is bool and type(query.get("target_coordinate_changed")) is bool,
                 "query admission/change flags must be booleans")
        status = query.get("status")
        _require(status in ("initialization_failed", "fit_failed", "admitted"),
                 "unknown photometric query status")
        _require(query["admitted"] == (status == "admitted"), "query status/admission mismatch")
        initialization = query.get("initialization", {})
        _require(isinstance(initialization, dict) and initialization.get("index") == index,
                 "query initialization index mismatch")
        if status == "initialization_failed":
            _require(initialization.get("status") == "initialization_failed" and query.get("photometric") is None,
                     "failed initialization has inconsistent state")
        else:
            kernel = query.get("photometric")
            _require(initialization.get("status") == "initialized" and isinstance(kernel, dict)
                     and type(kernel.get("admitted")) is bool and kernel["admitted"] == query["admitted"],
                     "initialized query has inconsistent fit state")
        initialized += initialization.get("status") == "initialized"
        original = base.finite_array(query.get("original_target_cache"), (2,), "original target")
        source = base.finite_array(query.get("original_source_cache"), (2,), "original source")
        target = base.finite_array(query.get("target_cache"), (2,), "corrected target")
        delta = base.finite_array(query.get("delta_cache"), (2,), "applied delta")
        _require(base.np.array_equal(original+delta, target), "applied delta/target mismatch")
        is_changed = not base.np.array_equal(original, target)
        _require(query["target_coordinate_changed"] == is_changed, "target change flag mismatch")
        changed += is_changed
        if query["admitted"]:
            kernel = query.get("photometric")
            _require(isinstance(kernel, dict) and kernel.get("admitted") is True
                     and initialization.get("status") == "initialized", "admitted query lacks successful fit/init")
            _require(base.np.array_equal(base.finite_array(kernel.get("source"), (2,), "kernel source"), source-0.5)
                     and base.np.array_equal(base.finite_array(kernel.get("original_target"), (2,), "kernel original target"), original-0.5)
                     and base.np.array_equal(base.finite_array(kernel.get("final", {}).get("target"), (2,), "kernel final target")+0.5, target),
                     "kernel array/cache coordinate mapping mismatch")
            _require(kernel.get("original_A") == initialization.get("A0"), "kernel affine initializer mismatch")
            admitted.append(index)
        else:
            _require(not is_changed and base.np.array_equal(delta, base.np.zeros(2)),
                     "failed query changes target")
    _require(photo.get("admitted_count") == len(admitted)
             and photo.get("target_coordinate_changed_count") == changed
             and photo.get("initialized_count") == initialized, "photometric coverage counters mismatch")
    return manifest, baseline, photo, identity, bindings


def prepare_coordinates(base, photo, x1, x2):
    """Apply only admitted target corrections; never refresh an inlier mask."""
    refined = x2.copy()
    success = []
    for query in photo["queries"]:
        i = query["index"]
        _require(base.np.array_equal(x1[i], query["original_source_cache"])
                 and base.np.array_equal(x2[i], query["original_target_cache"]),
                 "photometric original coordinates disagree with cache")
        if query["admitted"]:
            success.append(i)
            refined[i] = query["target_cache"]
    indices = base.np.asarray(success, dtype=base.np.int64)
    return indices, refined


def _native_result(base, backend, x1, x2, calibration, original_indices):
    result = base.estimate_pair(backend, x1, x2, calibration)
    mask = result.get("returned_ransac_stats", {}).get("inliers")
    if not isinstance(mask, list) or len(mask) != len(original_indices) or any(type(v) is not bool for v in mask):
        raise base.PredictionFailure("output", "native inlier mask does not match arm input order", result.get("call_seconds"))
    result["returned_inlier_indices_original"] = [int(i) for i, keep in zip(original_indices, mask) if keep]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "baseline-seal", "photometric-record", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--photometric-sha256", required=True)
    parser.add_argument("--pair-index", required=True, type=int, choices=range(16))
    args = parser.parse_args(argv)
    try:
        stream = args.output.open("x+")
    except FileExistsError:
        return 2
    started = time.perf_counter()
    record = {"schema": SCHEMA, "status": "started", "phase": "binding",
              "pair_index": args.pair_index, "utc": datetime.now(timezone.utc).isoformat(),
              "arm_order": list(ARM_ORDER), "arms": {},
              "native_calls_started": 0, "native_calls_completed": 0, "failed_calls": 0}
    phase = "binding"
    interruption = None
    with stream:
        write_record(stream, record)
        try:
            base, verifier = load_dependencies()
            record["settings"] = base.SETTINGS
            record["source_sha256"] = base.sha256(__file__)
            record["photometric_sha256"] = args.photometric_sha256
            manifest, baseline, photo, identity, bindings = verify_inputs(base, verifier, args.manifest,
                args.baseline_seal, args.photometric_record, args.photometric_sha256, args.pair_index)
            record.update(identity)
            record["photometric_source_bindings"] = bindings
            record["image_acquisition_receipt_sha256"] = photo["image_acquisition_receipt_sha256"]
            record["baseline_worker_sha256"] = base.sha256(BASELINE_DIR / "predict_pair.py")
            record["full_original_reference"] = {"baseline_prediction_sha256": identity["baseline_prediction_sha256"],
                                                "estimate": baseline["estimate"], "rerun": False}
            _require(args.data.stat().st_size == manifest["dataset_bytes"], "dataset size differs from manifest")
            import poselib
            import h5py
            library = {**base.library_metadata(poselib), "h5py_version": h5py.__version__}
            _require(library == baseline["library"], "actual pinned library metadata differs from baseline")
            record["library"] = library
            record.update(phase="input")
            phase = "input"
            write_record(stream, record)
            with h5py.File(args.data, "r") as handle:
                x1, x2, calibration = base.read_pair(handle, identity["key"])
            _require(base.json_safe(calibration) == baseline["calibration"], "cache calibration differs from baseline")
            indices, full_refined = prepare_coordinates(base, photo, x1, x2)
            record.update(calibration=base.json_safe(calibration), selected_count=photo["selected_count"],
                          admitted_count=len(indices), target_coordinate_changed_count=photo["target_coordinate_changed_count"],
                          success_query_indices=indices.tolist(), photometric_queries=photo["queries"],
                          full_correspondence_count=len(x1))
            for arm in ARM_ORDER:
                if arm != "full_refined" and len(indices) < 8:
                    record["arms"][arm] = {"status": "unavailable", "reason": "fewer_than_eight_successful_queries",
                                           "input_count": len(indices), "native_calls": 0}
                    write_record(stream, record)
                    continue
                ids = indices if arm != "full_refined" else base.np.arange(10000)
                a = x1[ids]
                b = (x2 if arm == "subset_original" else full_refined)[ids]
                entry = {"status": "started", "input_count": len(ids), "native_calls": 1}
                record["arms"][arm] = entry
                record["native_calls_started"] += 1
                record.update(phase=arm)
                phase = arm
                write_record(stream, record)  # Durable before every native invocation.
                call_started = time.perf_counter()
                try:
                    result = _native_result(base, poselib, a, b, calibration, ids)
                    entry.update(result, status="completed")
                    record["native_calls_completed"] += 1
                except BaseException as error:
                    record["failed_calls"] += 1
                    elapsed = time.perf_counter()-call_started
                    entry.update(status="error", error_type=type(error).__name__, error=str(error),
                                 failure_kind=getattr(error, "kind", "estimator"),
                                 call_seconds=getattr(error, "call_seconds", None))
                    if entry["call_seconds"] is None:
                        entry["call_seconds"] = elapsed
                        entry["call_seconds_scope"] = "worker wall time covering estimate wrapper"
                    if not isinstance(error, Exception):
                        entry.update(interrupted=True, failure_kind="interrupted")
                        write_record(stream, record)
                        raise
                write_record(stream, record)
            _require(source_bindings(base) == bindings and base.sha256(__file__) == record["source_sha256"],
                     "source/protocol changed during estimation")
            _require(base.sha256(args.photometric_record) == args.photometric_sha256,
                     "photometric record changed during estimation")
            _require(base.sha256(args.manifest) == identity["manifest_sha256"]
                     and base.sha256(args.baseline_seal) == identity["baseline_seal_sha256"]
                     and base.sha256(BASELINE_DIR / "predict_pair.py") == record["baseline_worker_sha256"],
                     "baseline source/manifest/seal changed during estimation")
            record.update(status="completed" if not record["failed_calls"] else "error", phase="completed")
        except BaseException as error:
            message = str(error)
            for path in (args.data, args.manifest, args.baseline_seal, args.photometric_record, args.output):
                message = message.replace(str(path), "<input-or-output>")
            record.update(status="error", phase="error", failure_phase=phase,
                          error_type=type(error).__name__, error=message)
            if not isinstance(error, Exception):
                record["interrupted"] = True
                interruption = error
        record["timing_seconds"] = {"worker_elapsed": time.perf_counter()-started}
        write_record(stream, record)
    if interruption is not None:
        raise interruption
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
