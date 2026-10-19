"""0017 input contract and fixed, label-free diagnostics; no native calls.

Pixel MSAC is a capped sum at 2 px with a direction-only cheirality test.
Depth reprojection diagnostics preserve the native raw translation/scale gauge.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "configs/0017-development-manifest.json"
LIBRARIES = ROOT / "configs/0017-libraries.json"
PROTOCOL = ROOT / "protocols/0017-depth-information.md"
ASSET_PROTOCOL = ROOT / "protocols/0017-depth-assets.md"
ARMS = ("e_norm", "e_ts", "depth_scale", "depth_affine")
SCENES = ("florence_cathedral", "lincoln_memorial", "london_bridge",
          "milan_cathedral", "sagrada_familia")
SALT = "evr0017:development:2026-10-19\0"
ASSETS_SHA256 = "71b5dffb08a3afae0a708bc609e11cbeffda71cc64614e880ce5cbf2d7b5f909"
SCENE_BYTES = (10675000, 10678136, 10486136, 10636784, 10636488)
BASE_URL = "https://vision.maths.lth.se/viktor/posebench/relative/monodepth/"
SOURCE_FILES = ("common.py", "predict_pair.py", "run_diagnostic.py",
                "evaluate_saved.py", "acquire_and_allocate.py")
K_RTOL, K_ATOL = 1e-6, 1e-6


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: json_safe(v) for k, v in value.items()}
    raise TypeError("Unsupported JSON value")


def write_record(stream, value):
    encoded = json.dumps(json_safe(value), sort_keys=True, allow_nan=False) + "\n"
    stream.seek(0)
    stream.write(encoded)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def write_new(path, value):
    encoded = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def source_hashes():
    paths = [HERE / name for name in SOURCE_FILES] + [PROTOCOL, ASSET_PROTOCOL, LIBRARIES,
        ROOT / "experiments/0013-pose-headroom/evaluate_sealed.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def _is_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    if (m.get("schema") != "evr0017.development.v1"
            or m.get("selection_salt") != SALT
            or m.get("assets_sha256") != ASSETS_SHA256
            or m.get("coordinate_convention") != "authored_raw_pixels"
            or m.get("development_forever") != list(SCENES)):
        raise ValueError("Invalid 0017 allocation schema/salt/development scope")
    expected_sources = {str(p.relative_to(ROOT)): sha256(p)
        for p in (HERE / "acquire_and_allocate.py", ASSET_PROTOCOL)}
    if m.get("source_sha256") != expected_sources:
        raise ValueError("Allocation source bindings differ")
    scenes = m.get("scenes", [])
    if not isinstance(scenes, list) or [s.get("scene") for s in scenes] != list(SCENES):
        raise ValueError("Invalid scene inventory")
    expected = []
    for scene_number, s in enumerate(scenes):
        name = s["scene"]
        filename = name + "_roma_moge.h5"
        keys = s.get("root_keys", [])
        if (s.get("filename") != filename or not _is_hash(s.get("sha256"))
                or type(s.get("bytes")) is not int or s["bytes"] != SCENE_BYTES[scene_number]
                or s.get("url") != BASE_URL + filename
                or not isinstance(keys, list) or len(keys) < 16
                or any(not isinstance(k, str) or not k for k in keys)
                or keys != sorted(set(keys))):
            raise ValueError("Invalid scene keys/file binding")
        chosen = sorted(keys, key=lambda k: (
            hashlib.sha256((SALT + name + "\0" + k).encode("utf-8")).hexdigest(), k))[:16]
        if s.get("selected_pair_keys") != chosen:
            raise ValueError("Allocated pair order differs from salted selection")
        for key in chosen:
            expected.append({"pair_index": len(expected), "scene": name, "key": key,
                "filename": filename, "dataset_sha256": s["sha256"], "dataset_bytes": s["bytes"]})
    rows = m.get("selected_pairs")
    if rows != expected or any(type(r.get("pair_index")) is not int for r in rows):
        raise ValueError("Flat allocation is not the exact ordered 80-pair scene allocation")
    return m


def options(arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    opt = {"ransac": {"min_iterations": 100, "max_iterations": 1000,
        "dyn_num_trials_mult": 3.0, "success_prob": 0.9999, "seed": 0,
        "progressive_sampling": False, "max_prosac_iterations": 100000,
        "score_initial_model": False}, "bundle": {}}
    if arm.startswith("e_"):
        opt.update(max_error=2.0, tangent_sampson=arm == "e_ts", real_focal_check=False)
    else:
        opt.update(max_errors=[16.0, 2.0], estimate_shift=arm == "depth_affine", weight_sampson=1.0)
    return opt


def library_metadata(backend):
    folder = Path(backend.__file__).resolve().parent
    binary = Path(backend._core.__file__).resolve()
    return {"version": backend.__version__, "binary_name": binary.name,
        "binary_bytes": binary.stat().st_size, "binary_sha256": sha256(binary),
        "python_files": {p.name: sha256(p) for p in sorted(folder.glob("*.py"))}}


def verify_library(backend, expected=None):
    if expected is None:
        expected = json.loads(LIBRARIES.read_text())["current"]
    actual = library_metadata(backend)
    if actual != expected or actual["version"] != "3.0.0":
        raise ValueError("Installed library differs from the pinned current library")
    return actual


def finite_array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + " has invalid shape or nonfinite values")
    return np.ascontiguousarray(a)


def _camera(camera, K, name):
    if camera.get("model") != "PINHOLE":
        raise ValueError(name + " unsupported camera model")
    for field in ("width", "height"):
        v = camera.get(field)
        if type(v) is not int or v < 0:
            raise ValueError(name + " invalid authored dimensions")
    p = finite_array(camera.get("params"), (4,), name + " params")
    if np.any(p[:2] <= 0):
        raise ValueError(name + " focal lengths must be positive")
    expected = np.array([[p[0], 0., p[2]], [0., p[1], p[3]], [0., 0., 1.]])
    K = finite_array(K, (3, 3), name + " K")
    if not np.allclose(K, expected, rtol=K_RTOL, atol=K_ATOL):
        raise ValueError(name + " camera disagrees with supplied K")
    return p, expected, float(np.max(np.abs(K - expected)))


def read_camera(group, K, name):
    model = np.asarray(group["model"][()])
    if model.shape != (1,):
        raise ValueError(name + " model shape is not (1,)")
    model = model[0]
    if isinstance(model, bytes):
        model = model.decode("utf-8")
    camera = {"model": str(model)}
    for field in ("width", "height"):
        v = finite_array(group[field][()], (1,), name + " " + field)[0]
        if v < 0 or v != int(v):
            raise ValueError(name + " invalid authored dimensions")
        camera[field] = int(v)
    camera["params"] = finite_array(group["params"][()], (4,), name + " params").tolist()
    _, _, difference = _camera(camera, K, name)
    return camera, difference


def read_pair(handle, key):
    """Named allowlist only; never enumerate a group or request R/t/images."""
    group = handle[key]
    x1 = np.asarray(group["x1"][()], dtype=np.float64)
    if x1.ndim != 2 or x1.shape[1] != 2 or len(x1) == 0:
        raise ValueError("x1 must be nonempty Nx2")
    n = len(x1)
    x1 = finite_array(x1, (n, 2), "x1")
    x2 = finite_array(group["x2"][()], (n, 2), "x2")
    K1 = finite_array(group["K1"][()], (3, 3), "K1")
    K2 = finite_array(group["K2"][()], (3, 3), "K2")
    camera1, diff1 = read_camera(group["camera1"], K1, "camera1")
    camera2, diff2 = read_camera(group["camera2"], K2, "camera2")
    depth1 = finite_array(group["depth1"][()], (n,), "depth1")
    depth2 = finite_array(group["depth2"][()], (n,), "depth2")
    # No depth sign filter, reciprocal, offset, normalization, or point deletion.
    cal = {"camera1": camera1, "camera2": camera2, "K1": K1, "K2": K2,
           "camera_K_max_abs_difference": [diff1, diff2]}
    return {"x1": x1, "x2": x2, "depth1": depth1, "depth2": depth2,
        "calibration": cal, "input_diagnostics": {"count": n, "all_finite": True,
            "depth1_nonpositive_count": int(np.count_nonzero(depth1 <= 0)),
            "depth2_nonpositive_count": int(np.count_nonzero(depth2 <= 0)),
            "depth1_zero_count": int(np.count_nonzero(depth1 == 0)),
            "depth2_zero_count": int(np.count_nonzero(depth2 == 0)),
            "depth_convention": "unchanged author field interpreted as axial z proxy"}}


def validate_mask(mask, n):
    mask = np.asarray(mask)
    if mask.shape != (n,) or mask.dtype.kind not in "biuf" or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid returned inlier mask")
    return np.ascontiguousarray(mask, dtype=bool)


def _rotation(value):
    R = finite_array(value, (3, 3), "R")
    if (not np.allclose(np.einsum("ij,kj->ik", R, R, optimize=False), np.eye(3), rtol=0, atol=1e-6)
            or abs(np.linalg.det(R) - 1.) > 1e-6):
        raise ValueError("Rotation must be proper orthogonal")
    return R


def validate_pose(estimate):
    R = _rotation(estimate["R"])
    t = finite_array(estimate["t"], (3,), "t")
    scale = float(np.max(np.abs(t)))
    if scale == 0:
        raise ValueError("Translation is zero")
    unit = t / scale
    unit /= math.sqrt(float(np.einsum("i,i->", unit, unit, optimize=False)))
    return R, unit


def prepare_geometry(x1, x2, calibration):
    x1 = np.asarray(x1, dtype=np.float64)
    if x1.ndim != 2 or x1.shape[1] != 2 or len(x1) == 0:
        raise ValueError("Invalid x1 shape")
    n = len(x1)
    arrays = {}
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for i, x in enumerate((x1, x2), 1):
            x = finite_array(x, (n, 2), "x" + str(i))
            p, K, _ = _camera(calibration["camera" + str(i)], calibration["K" + str(i)], "camera" + str(i))
            h = np.column_stack(((x[:, 0] - p[2]) / p[0], (x[:, 1] - p[3]) / p[1], np.ones(n)))
            norm = np.hypot(np.hypot(h[:, 0], h[:, 1]), h[:, 2])
            invK = np.array([[1/p[0], 0., -p[2]/p[0]], [0., 1/p[1], -p[3]/p[1]], [0., 0., 1.]])
            arrays.update({"x"+str(i): x, "p"+str(i): np.column_stack((x, np.ones(n))),
                "h"+str(i): h, "b"+str(i): h/norm[:, None], "invK"+str(i): invK,
                "params"+str(i): p})
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite calibrated geometry")
    return arrays


def final_scores(estimate, g):
    """2-pixel raw-F Sampson cap; invalid/degenerate rows retain full cost."""
    R, t = validate_pose(estimate)
    n = len(g["p1"])
    if any(not np.isfinite(v).all() for v in g.values()):
        raise ValueError("Nonfinite geometry")
    tx = np.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        E = np.einsum("ij,jk->ik", tx, R, optimize=False)
        rd = np.einsum("ij,nj->ni", R, g["b1"], optimize=False)
        a = -np.einsum("ni,ni->n", rd, g["b2"], optimize=False)
        b1 = -np.einsum("ni,i->n", rd, t, optimize=False)
        b2 = np.einsum("ni,i->n", g["b2"], t, optimize=False)
        minimum = .01 * (1. - a*a)
        cheiral = (b1-a*b2 > minimum) & (b2-a*b1 > minimum)
        F = np.einsum("ij,jk,kl->il", g["invK2"].T, E, g["invK1"], optimize=False)
        l2 = np.einsum("ij,nj->ni", F, g["p1"], optimize=False)
        l1 = np.einsum("ji,nj->ni", F, g["p2"], optimize=False)
        residual = np.einsum("ni,ni->n", g["p2"], l2, optimize=False)
        denom = np.einsum("ni,ni->n", l1[:, :2], l1[:, :2], optimize=False) + np.einsum("ni,ni->n", l2[:, :2], l2[:, :2], optimize=False)
        finite = np.isfinite(residual) & np.isfinite(denom)
        usable = finite & (denom > 0)
        costs = np.ones(n)
        good = usable & cheiral
        ratio = np.abs(residual[good]) / np.sqrt(denom[good]) / 2.
        costs[good] = np.minimum(ratio, 1.)**2
        ratio_nonfinite = int((~np.isfinite(ratio)).sum())
    if not np.isfinite(costs).all():
        raise ValueError("Invalid final score arithmetic")
    return {"pixel_msac": float(costs.sum()), "pixel_threshold": 2.0,
        "pixel_inliers": int(np.count_nonzero(costs < 1.)), "count": n,
        "cheiral_count": int(cheiral.sum()), "degenerate_count": int((~usable).sum()),
        "nonfinite_arithmetic_count": int((~finite).sum()) + ratio_nonfinite}


def _direction(Z, params, target):
    """Nonfinite transformed/projection arithmetic is explicit saturated cost."""
    finite_Z = np.isfinite(Z).all(axis=1)
    positive = np.isfinite(Z[:, 2]) & (Z[:, 2] > 0)
    projectable = finite_Z & positive
    costs = np.ones(len(Z))
    valid = np.zeros(len(Z), dtype=bool)
    nonfinite_projection = np.zeros(len(Z), dtype=bool)
    ids = np.flatnonzero(projectable)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        projected = Z[ids, :2] / Z[ids, 2, None]
        projected = projected * params[:2] + params[2:]
        difference = projected - target[ids]
        finite = np.isfinite(projected).all(axis=1) & np.isfinite(difference).all(axis=1)
        distance = np.hypot(difference[:, 0], difference[:, 1])
        finite &= np.isfinite(distance)
        valid[ids[finite]] = True
        nonfinite_projection[ids[~finite]] = True
        costs[ids[finite]] = np.minimum(distance[finite] / 16., 1.)**2
    return {"cost": costs, "valid": valid, "nonfinite_transformed": ~finite_Z,
        "nonfinite_projection": nonfinite_projection, "positive_transformed_z": positive}


def depth_diagnostics(geometry, g, depth1, depth2, returned_mask):
    """Raw native gauge; invalid directions cost one in a fixed denominator.

    Unlike pose-angle scoring, raw t is never normalized. Finite negative/zero
    scale and corrected depths are reported, not discarded. Empty support is
    unavailable. Overflow is retained as explicit per-direction invalidity.
    """
    R = _rotation(geometry["R"])
    t = finite_array(geometry["t"], (3,), "raw t")
    nuisance = finite_array([geometry[k] for k in ("scale", "shift1", "shift2")], (3,), "depth nuisance")
    scale, shift1, shift2 = nuisance
    n = len(g["h1"])
    depth1 = finite_array(depth1, (n,), "depth1")
    depth2 = finite_array(depth2, (n,), "depth2")
    mask = validate_mask(returned_mask, n)
    if any(not np.isfinite(v).all() for v in g.values()):
        raise ValueError("Nonfinite geometry")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        c1, c2 = depth1 + shift1, depth2 + shift2
        z2 = scale * c2
        Z12 = np.einsum("ij,nj->ni", R, c1[:, None]*g["h1"], optimize=False) + t
        Z21 = np.einsum("ji,nj->ni", R, z2[:, None]*g["h2"] - t, optimize=False)
    directions = (_direction(Z12, g["params2"], g["x2"]), _direction(Z21, g["params1"], g["x1"]))
    quantities = {"corrected_source_depth": c1, "corrected_target_depth": c2,
        "scaled_target_depth": z2, "forward_transformed_z": Z12[:, 2], "reverse_transformed_z": Z21[:, 2]}
    nonpositive = {name: np.isfinite(v) & (v <= 0) for name, v in quantities.items()}
    nonfinite = {name: ~np.isfinite(v) for name, v in quantities.items()}

    def summarize(support):
        size = int(support.sum())
        result = {"denominator": size, "direction_denominator": 2*size,
            "status": "available" if size else "unavailable_empty_support",
            "nonpositive_counts": {k: int((v & support).sum()) for k, v in nonpositive.items()},
            "nonfinite_counts": {k: int((v & support).sum()) for k, v in nonfinite.items()},
            "nonpositive_union_count": int((np.logical_or.reduce(list(nonpositive.values())) & support).sum()),
            "nonfinite_union_count": int((np.logical_or.reduce(list(nonfinite.values())) & support).sum())}
        transformed = (Z12[:, 2], Z21[:, 2])
        result["nonpositive_transformed_direction_count"] = sum(int((np.isfinite(z) & (z <= 0) & support).sum()) for z in transformed)
        result["nonfinite_transformed_direction_count"] = sum(int((~np.isfinite(z) & support).sum()) for z in transformed)
        result["inactive_transformed_direction_count"] = result["nonpositive_transformed_direction_count"] + result["nonfinite_transformed_direction_count"]
        # Exact native predicate differs from valid geometry when z is +infinity.
        result["native_branch_positive_count"] = sum(int(((z > 0) & support).sum()) for z in transformed)
        result["native_branch_inactive_count"] = 2*size - result["native_branch_positive_count"]
        result["native_branch_positive_finite_z_count"] = sum(int(((z > 0) & np.isfinite(z) & support).sum()) for z in transformed)
        result["native_branch_positive_nonfinite_z_count"] = result["native_branch_positive_count"] - result["native_branch_positive_finite_z_count"]
        result["native_branch_inactive_finite_z_count"] = result["nonpositive_transformed_direction_count"]
        result["native_branch_inactive_nonfinite_z_count"] = result["native_branch_inactive_count"] - result["native_branch_inactive_finite_z_count"]
        total = 0.
        omitted = 0
        valid_cost = 0.
        for name, d in zip(("forward", "reverse"), directions):
            subtotal = float(d["cost"][support].sum())
            result[name] = {"cost_sum": subtotal, "valid_projection_count": int((d["valid"] & support).sum()),
                "transformed_z_positive_count": int((d["positive_transformed_z"] & support).sum()),
                "nonfinite_transformed_count": int((d["nonfinite_transformed"] & support).sum()),
                "nonfinite_projection_count": int((d["nonfinite_projection"] & support).sum())}
            total += subtotal
            omitted += int((~d["valid"] & support).sum())
            valid_cost += float(d["cost"][support & d["valid"]].sum())
        result.update(fixed_reprojection_cost=total/(2*size) if size else None,
            omission_same_denominator_cost=valid_cost/(2*size) if size else None,
            invalid_direction_fraction=omitted/(2*size) if size else None)
        return result

    max_t = float(np.max(np.abs(t)))
    norm_unit = 0. if max_t == 0 else math.sqrt(float(np.einsum("i,i->", t/max_t, t/max_t, optimize=False)))
    raw_norm = max_t*norm_unit
    return json_safe({"raw_translation_norm": raw_norm if math.isfinite(raw_norm) else None,
        "raw_translation_norm_overflow": not math.isfinite(raw_norm), "scale": float(scale),
        "shift1": float(shift1), "shift2": float(shift2), "pixel_threshold": 16.0,
        "all": summarize(np.ones(n, dtype=bool)), "returned_mask": summarize(mask),
        "returned_mask_semantics": "Pre-final-refinement epipolar support; not final depth-active support"})
