"""0014 known-control continuations from one sealed baseline; no label access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

SCHEMA = "evr0014.prediction.v1"
SOURCE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SOURCE_DIR.parent / "0013-pose-headroom"
NATIVE_OPTIONS = {"loss_type": "CAUCHY", "loss_scale": 0.5,
    "max_iterations": 100, "gradient_tol": 1e-10, "step_tol": 1e-8,
    "initial_lambda": 1e-3, "min_lambda": 1e-10, "max_lambda": 1e10,
    "verbose": False}
SETTINGS = {"native_options": NATIVE_OPTIONS, "order": ["native", "N", "P"],
    "initialization": "same sealed original R/t independently for every arm",
    "mask": "same sealed pre-final RANSAC inlier mask; no recomputation",
    "coordinate_convention": "raw_pixels", "native_threads": 1,
    "numpy_calibration": "effective K from authored camera params; stored K only cross-checked",
    "objective": "dimensionless sum log1p((residual_pixels/0.5)^2)",
    "stationarity": "undamped local Gauss-Newton predicted decrease; not a global bound",
    "canonical_cauchy_objective_multiplier": 0.25}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    # Distinct names prevent either predict_pair module from replacing the other.
    base = load_module("_evr0013_predict_for0014", BASELINE_DIR / "predict_pair.py")
    seal = load_module("_evr0013_seal_for0014", BASELINE_DIR / "evaluate_sealed.py")
    kernel = load_module("_evr0014_refine_for_worker", SOURCE_DIR / "refine.py")
    return base, seal, kernel


def write_record(stream, record):
    encoded = json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
    stream.seek(0)
    stream.write(encoded)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def checked_pose(base, estimate):
    np = base.np
    R = base.finite_array(estimate.get("R"), (3, 3), "rotation")
    t = base.finite_array(estimate.get("t"), (3,), "translation")
    if (not np.any(t != 0) or not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-6)
            or abs(np.linalg.det(R) - 1.0) > 1e-6):
        raise ValueError("invalid pose; no automatic repair is permitted")
    return R, t


def selected_baseline(base, verifier, manifest_path, seal_path, index):
    manifest, seal, records, manifest_hash, seal_hash = verifier.verify_seal(manifest_path, seal_path)
    keys = manifest["selected_pair_keys"]
    if (manifest.get("schema") != "evr0012.development.v1"
            or manifest.get("coordinate_convention") != "raw_pixels"
            or manifest.get("development_scene") != "0015"
            or keys != sorted(set(keys)) or any(not k.startswith("0015-") for k in keys)):
        raise ValueError("baseline manifest allocation differs")
    row, baseline = records[index]
    if baseline.get("status") != "completed" or baseline.get("settings") != base.SETTINGS:
        raise ValueError("selected baseline is not a completed fixed-setting estimate")
    mask = baseline.get("returned_ransac_stats", {}).get("inliers")
    if (not isinstance(mask, list) or len(mask) != 10000
            or any(type(value) is not bool for value in mask) or sum(mask) < 6):
        raise ValueError("saved inlier mask must contain 10000 booleans and at least six inliers")
    checked_pose(base, baseline["estimate"])
    identity = {"key": keys[index], "manifest_sha256": manifest_hash,
        "baseline_seal_sha256": seal_hash, "baseline_prediction_sha256": row["sha256"],
        "dataset_sha256_declared": manifest["dataset_sha256"],
        "mask_sha256": hashlib.sha256(bytes(mask)).hexdigest(), "kept_count": sum(mask),
        "mask_encoding": "10000 uint8 values in original correspondence order"}
    return manifest, baseline, base.np.asarray(mask, dtype=bool), identity


def native_refine(backend, x1, x2, calibration, R, t):
    initial = backend.CameraPose()
    initial.R = R.copy()
    initial.t = t.copy()
    pose, stats = backend.refine_relative_pose(x1, x2, initial,
        calibration["camera1"], calibration["camera2"], dict(NATIVE_OPTIONS))
    return {"R": pose.R, "t": pose.t}, stats


def effective_intrinsics(base, calibration):
    result = []
    for name in ("camera1", "camera2"):
        fx, fy, cx, cy = calibration[name]["params"]
        result.append(base.np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]))
    return result


def local_gn_stationarity(np, residual, J, cost):
    """Undamped local proxy only; singular curvature is explicitly invalid."""
    try:
        if (not np.isfinite(cost) or not np.isfinite(residual).all()
                or not np.isfinite(J).all()):
            raise ValueError("nonfinite cost, residual or Jacobian")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            weight = 1.0 / (0.25 + residual * residual)
            g = 2.0 * np.einsum("ni,n->i", J, weight * residual, optimize=False)
            H = 2.0 * np.einsum("ni,nj->ij", J, weight[:, None] * J, optimize=False)
            if not np.isfinite(H).all() or not np.isfinite(g).all():
                raise ValueError("nonfinite local gradient or curvature")
            L = np.linalg.cholesky(H)
            z = np.linalg.solve(L, g)
            decrease = float(z @ z) / (2.0 * max(1.0, cost))
        if not np.isfinite(decrease) or decrease < 0:
            raise ValueError("invalid local predicted decrease")
        return {"valid": True, "relative_predicted_decrease": decrease}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {"valid": False, "relative_predicted_decrease": None,
                "reason": type(error).__name__ + ": " + str(error)}


def common_objectives(base, kernel, x1, x2, calibration, estimates):
    np = base.np
    K1, K2 = effective_intrinsics(base, calibration)
    x = np.column_stack(((x1 - K1[:2, 2]) / np.diag(K1)[:2], np.ones(len(x1))))
    y = np.column_stack(((x2 - K2[:2, 2]) / np.diag(K2)[:2], np.ones(len(x2))))
    costs, canonical, gradients, stationarity = {}, {}, {}, {}
    for name, estimate in estimates.items():
        R, t = checked_pose(base, estimate)
        costs[name], canonical[name], gradients[name], stationarity[name] = {}, {}, {}, {}
        for mode in ("N", "P"):
            residual, J = kernel.residuals_and_jacobian(x, y, R, t,
                np.diag(K1)[:2], np.diag(K2)[:2], mode=mode)
            cost = kernel.cauchy_cost(residual)
            gradient = 2.0 * np.einsum("ni,n->i", J,
                residual / (0.25 + residual * residual), optimize=False)
            costs[name][mode], canonical[name][mode] = cost, 0.25 * cost
            gradients[name][mode] = float(np.max(np.abs(gradient)))
            stationarity[name][mode] = local_gn_stationarity(np, residual, J, cost)
    return base.json_safe({"objectives": costs, "canonical_objectives": canonical,
                          "objective_gradients_infinity": gradients, "stationarity": stationarity})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "baseline-seal", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--pair-index", required=True, type=int, choices=range(16))
    args = parser.parse_args(argv)
    try:
        stream = args.output.open("x+")
    except FileExistsError:
        print(json.dumps({"status": "refused_existing_output", "pair_index": args.pair_index}))
        return 2
    started = time.perf_counter()
    record = {"schema": SCHEMA, "status": "started", "phase": "binding",
        "pair_index": args.pair_index, "utc": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS, "timing_seconds": {}}
    phase, failure_kind = "binding", "binding"
    with stream:
        # Persist before dependencies, manifests, baseline predictions or data are read.
        write_record(stream, record)
        try:
            base, verifier, kernel = load_dependencies()
            record.update(source_sha256=base.sha256(__file__),
                kernel_sha256=base.sha256(SOURCE_DIR / "refine.py"),
                kernel_settings=base.json_safe(kernel.SETTINGS))
            manifest, baseline, mask, identity = selected_baseline(base, verifier,
                args.manifest, args.baseline_seal, args.pair_index)
            record.update(identity)
            if args.data.stat().st_size != manifest["dataset_bytes"]:
                raise ValueError("dataset byte size differs from manifest")
            import h5py
            import poselib
            library = {**base.library_metadata(poselib), "h5py_version": h5py.__version__}
            if library != baseline.get("library"):
                raise ValueError("library differs from sealed baseline provenance")
            record["library"] = library
            phase, failure_kind = "input", "input"
            record["phase"] = phase
            write_record(stream, record)
            load_started = time.perf_counter()
            try:
                with h5py.File(args.data, "r") as handle:
                    x1, x2, calibration = base.read_pair(handle, record["key"])
                if base.json_safe(calibration) != baseline.get("calibration"):
                    raise ValueError("calibration differs from sealed baseline")
                x1, x2 = x1[mask].copy(), x2[mask].copy()
                R, t = checked_pose(base, baseline["estimate"])
                K1, K2 = effective_intrinsics(base, calibration)
            finally:
                record["timing_seconds"]["loading_and_validation"] = time.perf_counter() - load_started
            record.update(calibration=base.json_safe(calibration),
                effective_K1=base.json_safe(K1), effective_K2=base.json_safe(K2),
                original_estimate=base.json_safe(baseline["estimate"]), estimates={}, kernel_results={})
            for mode in SETTINGS["order"]:
                phase, failure_kind = mode, "estimator" if mode == "native" else "kernel"
                record["phase"] = phase
                write_record(stream, record)
                call_started = time.perf_counter()
                try:
                    if mode == "native":
                        estimate, stats = native_refine(poselib, x1.copy(), x2.copy(), calibration, R, t)
                        failure_kind = "output"
                        if not isinstance(stats, dict):
                            raise TypeError("native refinement statistics must be a dictionary")
                        native_stats = base.json_safe(stats)
                    else:
                        result = kernel.refine_pose(x1.copy(), x2.copy(), K1.copy(),
                            K2.copy(), R.copy(), t.copy(), mode=mode)
                        failure_kind = "output"
                        if (result.get("schema") != "evr0014.refinement.v1"
                                or result.get("mode") != mode or result.get("settings") != kernel.SETTINGS
                                or result.get("number_of_correspondences") != len(x1)):
                            raise ValueError("kernel result binding differs")
                        result = base.json_safe(result)
                        estimate = {key: result[key] for key in ("R", "t")}
                    checked_pose(base, estimate)
                    record["estimates"][mode] = base.json_safe(estimate)
                    if mode == "native":
                        record["native_stats"] = native_stats
                    else:
                        record["kernel_results"][mode] = result
                finally:
                    record["timing_seconds"][mode] = time.perf_counter() - call_started
            phase, failure_kind = "objectives", "output"
            record["phase"] = phase
            write_record(stream, record)
            objective_started = time.perf_counter()
            try:
                record.update(common_objectives(base, kernel, x1, x2, calibration,
                    {"original": record["original_estimate"], **record["estimates"]}))
            finally:
                record["timing_seconds"]["objectives"] = time.perf_counter() - objective_started
            record.update(status="completed", phase="completed")
        except Exception as error:
            message = str(error)
            for path in (args.data, args.manifest, args.baseline_seal, args.output):
                message = message.replace(str(path), "<local-path>")
            record.update(status="error", phase="error", failure_phase=phase,
                failure_kind=failure_kind, error_type=type(error).__name__, error=message)
        record["timing_seconds"]["worker_elapsed"] = time.perf_counter() - started
        write_record(stream, record)
    print(json.dumps({key: record[key] for key in ("status", "pair_index", "key", "failure_kind") if key in record}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
