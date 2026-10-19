"""0013: one fixed official baseline prediction; no ground-truth access.

The supervisor owns timeouts and the all-predictions seal. This worker creates
one exclusive JSON record and never retries an estimator or accesses labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

# Set before importing NumPy or any compiled estimator; the protocol is serial.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"

import numpy as np

SCHEMA = "evr0013.prediction.v1"
VERSION = "2.0.5"
WHEEL_SHA256 = "43a2275d8a336e1d6c41557d2c59366b0f8d1716ff14468602143f9af8bb2d59"
K_RTOL, K_ATOL = 1e-6, 1e-6
RANSAC_OPTIONS = {"min_iterations": 1000, "max_iterations": 100000,
    "dyn_num_trials_mult": 3.0, "success_prob": 0.9999,
    "max_epipolar_error": 1.0, "seed": 0, "progressive_sampling": False,
    "max_prosac_iterations": 100000, "real_focal_check": False}
SETTINGS = {"ransac_options": RANSAC_OPTIONS, "bundle_options": {},
    "initial_pose": None, "matches": 10000, "coordinate_convention": "raw_pixels",
    "camera_K_rtol": K_RTOL, "camera_K_atol": K_ATOL, "native_threads": 1,
    "rotation_validation_atol": 1e-6,
    "effective_final_bundle_defaults": {"loss_type": "CAUCHY", "loss_scale": 0.5,
        "max_iterations": 100, "gradient_tol": 1e-10, "step_tol": 1e-8,
        "initial_lambda": 1e-3, "min_lambda": 1e-10, "max_lambda": 1e10,
        "verbose": False}}


class PredictionFailure(Exception):
    def __init__(self, kind, message, call_seconds=None):
        super().__init__(message)
        self.kind = kind
        self.call_seconds = call_seconds


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """Strict recursive conversion: neither NaN nor infinity reaches JSON."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON dictionary key is not a string")
        return {key: json_safe(item) for key, item in value.items()}
    raise TypeError("unsupported JSON value type: " + type(value).__name__)


def write_record(stream, record):
    # Validate/serialize before replacing the prior started/error state.
    encoded = json.dumps(json_safe(record), allow_nan=False, sort_keys=True) + "\n"
    stream.seek(0)
    stream.write(encoded)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def load_manifest(path, pair_index):
    manifest = json.loads(Path(path).read_text())
    keys = manifest.get("selected_pair_keys", [])
    if (manifest.get("schema") != "evr0012.development.v1"
            or manifest.get("coordinate_convention") != "raw_pixels"
            or manifest.get("development_scene") != "0015"
            or len(keys) != 16 or keys != sorted(set(keys))
            or any(not key.startswith("0015-") for key in keys)
            or not 0 <= pair_index < 16):
        raise PredictionFailure("binding", "manifest allocation or pair index differs")
    return manifest, keys[pair_index]


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(name + " has invalid shape or nonfinite values")
    return np.ascontiguousarray(array)


def read_camera(group, K, name):
    model_values = np.asarray(group["model"][()])
    if model_values.shape != (1,):
        raise ValueError(name + " model must have shape (1,)")
    model = model_values[0]
    if isinstance(model, bytes):
        model = model.decode("utf-8")
    if model != "PINHOLE":
        raise ValueError(name + " model is not PINHOLE")
    camera = {"model": str(model)}
    for field in ("width", "height"):
        value = finite_array(group[field][()], (1,), name + "/" + field)[0]
        if value <= 0 or value != int(value):
            raise ValueError(name + " dimensions must be positive integers")
        camera[field] = int(value)
    p = finite_array(group["params"][()], (4,), name + "/params")
    if np.any(p[:2] <= 0):
        raise ValueError(name + " focal lengths must be positive")
    expected = np.array([[p[0], 0., p[2]], [0., p[1], p[3]], [0., 0., 1.]])
    if not np.allclose(K, expected, rtol=K_RTOL, atol=K_ATOL):
        raise ValueError(name + " disagrees with supplied intrinsics")
    camera["params"] = p.tolist()
    return camera, float(np.max(np.abs(K - expected)))


def read_pair(handle, key):
    """Explicit allowlist; never iterate HDF5 groups or request other fields."""
    group = handle[key]
    x1 = finite_array(group["x1"][()], (10000, 2), "x1")
    x2 = finite_array(group["x2"][()], (10000, 2), "x2")
    K1 = finite_array(group["K1"][()], (3, 3), "K1")
    K2 = finite_array(group["K2"][()], (3, 3), "K2")
    camera1, diff1 = read_camera(group["camera1"], K1, "camera1")
    camera2, diff2 = read_camera(group["camera2"], K2, "camera2")
    f1, f2 = (sum(camera["params"][:2]) / 2 for camera in (camera1, camera2))
    threshold = 0.5 * (1.0 / f1 + 1.0 / f2)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("normalized threshold is invalid")
    calibration = {"camera1": camera1, "camera2": camera2, "K1": K1, "K2": K2,
        "camera_K_max_abs_difference": [diff1, diff2],
        "normalized_epipolar_threshold": threshold,
        "normalized_cauchy_loss_scale": 0.5 * threshold}
    return x1, x2, calibration


def library_metadata(backend):
    version = importlib.metadata.version("poselib")
    if version != VERSION or backend.__version__ != VERSION:
        raise ValueError("PoseLib distribution/module version differs from 2.0.5")
    binary = Path(backend._core.__file__).resolve()
    initializer = Path(backend.__file__).resolve()
    return {"version": version, "module_version": backend.__version__,
        "module_path": "<runtime>/poselib/" + initializer.name,
        "module_sha256": sha256(initializer),
        "binary_path": "<runtime>/poselib/" + binary.name,
        "binary_name": binary.name, "binary_bytes": binary.stat().st_size,
        "binary_sha256": sha256(binary), "official_wheel_sha256": WHEEL_SHA256,
        "python_version": sys.version.split()[0], "numpy_version": np.__version__}


def estimate_pair(backend, x1, x2, calibration):
    started = time.perf_counter()
    try:
        pose, info = backend.estimate_relative_pose(x1, x2,
            calibration["camera1"], calibration["camera2"],
            dict(RANSAC_OPTIONS), {}, initial_pose=None)
    except Exception as error:
        raise PredictionFailure("estimator", type(error).__name__ + ": " + str(error),
                                time.perf_counter() - started) from error
    call_seconds = time.perf_counter() - started
    try:
        R = finite_array(pose.R, (3, 3), "estimated rotation")
        t = finite_array(pose.t, (3,), "estimated translation")
        if not np.any(t != 0):
            raise ValueError("estimated translation is zero")
        if (not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-6)
                or abs(np.linalg.det(R) - 1.0) > 1e-6):
            raise ValueError("estimated rotation is not proper orthogonal")
        if not isinstance(info, dict):
            raise TypeError("returned RANSAC info is not a dictionary")
        result = json_safe({"estimate": {"R": R, "t": t},
            "returned_ransac_stats": info,
            "returned_ransac_stats_semantics": "Library outputs before final Cauchy refinement; score/inlier_ratio may precede final internal LO.",
            "call_seconds": call_seconds})
    except Exception as error:
        raise PredictionFailure("output", type(error).__name__ + ": " + str(error), call_seconds) from error
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--pair-index", required=True, type=int, choices=range(16))
    args = parser.parse_args(argv)
    # Refuse an existing output before any library import or data read.
    try:
        stream = args.output.open("x+")
    except FileExistsError:
        print(json.dumps({"status": "refused_existing_output", "pair_index": args.pair_index}))
        return 2
    started = time.perf_counter()
    record = {"schema": SCHEMA, "status": "started", "pair_index": args.pair_index,
        "phase": "binding", "utc": datetime.now(timezone.utc).isoformat(), "settings": SETTINGS}
    phase = "binding"
    with stream:
        write_record(stream, record)
        try:
            record["source_sha256"] = sha256(__file__)
            record["manifest_sha256"] = sha256(args.manifest)
            manifest, key = load_manifest(args.manifest, args.pair_index)
            record.update(key=key, dataset_sha256_declared=manifest["dataset_sha256"])
            if args.data.stat().st_size != manifest["dataset_bytes"]:
                raise ValueError("dataset byte size differs from manifest")
            import h5py
            import poselib
            record["library"] = {**library_metadata(poselib), "h5py_version": h5py.__version__}
            phase = "input"
            record["phase"] = phase
            write_record(stream, record)
            load_started = time.perf_counter()
            try:
                with h5py.File(args.data, "r") as handle:
                    x1, x2, calibration = read_pair(handle, key)
            finally:
                record["timing_seconds"] = {"loading_and_validation": time.perf_counter() - load_started}
            record["calibration"] = json_safe(calibration)
            record["phase"] = "estimator"
            write_record(stream, record)
            phase = "estimator"
            result = estimate_pair(poselib, x1, x2, calibration)
            record["timing_seconds"]["estimator_call"] = result.pop("call_seconds")
            record.update(result, status="completed", phase="completed")
        except Exception as error:
            kind = error.kind if isinstance(error, PredictionFailure) else phase
            if isinstance(error, PredictionFailure) and error.call_seconds is not None:
                record.setdefault("timing_seconds", {})["estimator_call"] = error.call_seconds
            message = str(error)
            for path, replacement in ((args.data, "<data>"), (args.manifest, "<manifest>"),
                                      (args.output, "<output>")):
                message = message.replace(str(path), replacement)
            record.update(status="error", phase="error", failure_phase=phase, failure_kind=kind,
                          error_type=type(error).__name__, error=message)
        record.setdefault("timing_seconds", {})["worker_elapsed"] = time.perf_counter() - started
        write_record(stream, record)
    print(json.dumps({key: record[key] for key in ("status", "pair_index", "key", "failure_kind") if key in record}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
