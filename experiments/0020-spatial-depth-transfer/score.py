"""Fixed 0020 scoring kernel. Pure arrays in, finite JSON out; no asset reader.

The only imported experiment module supplies frozen 0017 camera/pose validation.
pixel_costs is an explicit per-point port of its aggregate-only final_scores.
The four saved poses are never optimized or expanded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
COMMON_PATH = HERE.parent / "0017-depth-information/common.py"
DEPENDENCY_PATHS = (COMMON_PATH,)
_spec = importlib.util.spec_from_file_location("_evr0020_frozen0017", COMMON_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

ARMS = ("e_norm", "e_ts", "depth_scale", "depth_affine")
SELECTORS = ("pixel_msac", "global_affine_ls", "spatial_weight_only",
             "random_point_transfer", "spatial_transfer")
RANDOM_SALT = "evr0020-random-v1\0"


def _finite(value):
    return bool(np.isfinite(value).all())


def pixel_costs(estimate, g):
    """Return (N capped costs, metadata), ported from frozen 0017 final_scores.

    Same 2px raw-F Sampson and 0.01-unit-baseline cheirality, including the
    original operation order. Invalid arithmetic keeps full cost.
    """
    R, t = base.validate_pose(estimate)
    n = len(g["p1"])
    if any(not _finite(v) for v in g.values()):
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
        denom = (np.einsum("ni,ni->n", l1[:, :2], l1[:, :2], optimize=False)
                 + np.einsum("ni,ni->n", l2[:, :2], l2[:, :2], optimize=False))
        finite = np.isfinite(residual) & np.isfinite(denom)
        usable = finite & (denom > 0)
        costs = np.ones(n)
        good = usable & cheiral
        ratio = np.abs(residual[good]) / np.sqrt(denom[good]) / 2.
        costs[good] = np.minimum(ratio, 1.)**2
        ratio_nonfinite = int((~np.isfinite(ratio)).sum())
    if not _finite(costs):
        raise ValueError("Invalid final score arithmetic")
    return costs, {"pixel_msac": float(costs.sum()), "pixel_threshold": 2.0,
        "pixel_inliers": int(np.count_nonzero(costs < 1.)), "count": n,
        "cheiral_count": int(cheiral.sum()), "degenerate_count": int((~usable).sum()),
        "nonfinite_arithmetic_count": int((~finite).sum()) + ratio_nonfinite}


def partitions(key, x1):
    """Lower-median rectangles and one size-matched deterministic random fold."""
    n = len(x1)
    mid = (n-1)//2
    u0, v0 = float(np.sort(x1[:, 0])[mid]), float(np.sort(x1[:, 1])[mid])
    ids = 2*(x1[:, 0] > u0).astype(np.int64) + (x1[:, 1] > v0).astype(np.int64)
    counts = np.bincount(ids, minlength=4)
    meta = {"status": "completed" if np.all(counts > 0) else "unavailable",
            "reason": None if np.all(counts > 0) else "spatial_partition_unavailable",
            "u0": u0, "v0": v0, "counts": counts.tolist(), "ids": ids.tolist()}
    if np.any(counts == 0):
        return ids, None, None, meta, {"status": "unavailable",
            "reason": "spatial_partition_unavailable", "salt": RANDOM_SALT,
            "counts": counts.tolist(), "ids": None}
    weights = 1. / (4.*counts[ids])
    order = sorted(range(n), key=lambda i: (
        hashlib.sha256((RANDOM_SALT + key + "\0" + str(i)).encode("utf-8")).digest(), i))
    random_ids = np.empty(n, dtype=np.int64)
    start = 0
    for k, count in enumerate(counts):
        random_ids[np.asarray(order[start:start+int(count)], dtype=np.int64)] = k
        start += int(count)
    return ids, random_ids, weights, meta, {"status": "completed", "reason": None,
        "salt": RANDOM_SALT, "counts": counts.tolist(), "ids": random_ids.tolist()}


def normalize_depth(depth):
    """Population mean/RMS normalization; fail explicitly on unrepresentable math."""
    meta = {"status": "unavailable", "reason": None, "mean": None, "rms": None,
            "count": len(depth), "raw_nonpositive_count": int(np.count_nonzero(depth <= 0))}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            mean = float(np.mean(depth))
            if not math.isfinite(mean):
                raise FloatingPointError("Nonfinite mean")
            meta["mean"] = mean
            centered = depth - mean
            rms = math.sqrt(float(np.einsum("i,i->", centered, centered, optimize=False) / len(depth)))
            if not math.isfinite(rms):
                raise FloatingPointError("Nonfinite RMS")
            meta["rms"] = rms
            if rms == 0:
                meta["reason"] = "zero_depth_spread"
                return None, meta
            z = centered / rms
            if not _finite(z):
                raise FloatingPointError("Nonfinite normalized depth")
    except (FloatingPointError, OverflowError):
        meta["reason"] = "nonrepresentable_depth_normalization"
        return None, meta
    meta.update(status="completed", reason=None)
    return z, meta


def fit_calibration(R, t_unit, g, z1, z2, train, weights, unavailable_reason=None):
    """Four-parameter ordinary weighted 3D LS, with explicit admission metadata."""
    started = time.perf_counter()
    idx = np.flatnonzero(train)
    mass = float(math.fsum(float(weights[i]) for i in idx))
    out = {"status": "unavailable", "reason": None, "training_count": len(idx),
           "training_weight_mass": mass, "coefficients": None, "singular_values": None,
           "rank": None, "rank_threshold": None, "svd_attempted": False,
           "training_residual_mean_squared": None,
           "seconds": None}

    def fail(reason):
        out["reason"] = reason
        out["seconds"] = time.perf_counter() - started
        return out

    if unavailable_reason is not None:
        return fail(unavailable_reason)
    if len(idx)*3 < 4 or mass <= 0:
        return fail("insufficient_fit_support")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            q = weights[idx] / mass
            h1, h2 = g["h1"][idx], g["h2"][idx]
            rh1 = np.einsum("ij,nj->ni", R, h1, optimize=False)
            A = np.stack((z1[idx, None]*rh1, rh1, -z2[idx, None]*h2, -h2), axis=2)
            sqrtq = np.sqrt(q)
            Aw = (A*sqrtq[:, None, None]).reshape(-1, 4)
            bw = (-sqrtq[:, None]*t_unit[None, :]).reshape(-1)
            if not _finite(Aw) or not _finite(bw):
                raise FloatingPointError("Nonfinite design")
            out["svd_attempted"] = True
            U, singular, Vt = np.linalg.svd(Aw, full_matrices=False)
            if not all(_finite(v) for v in (U, singular, Vt)):
                raise FloatingPointError("Nonfinite SVD")
            out["singular_values"] = singular.tolist()
            threshold = float(np.finfo(np.float64).eps*max(len(bw), 4)*singular[0])
            if not math.isfinite(threshold):
                raise FloatingPointError("Nonfinite rank threshold")
            out["rank_threshold"] = threshold
            out["rank"] = int(np.count_nonzero(singular > threshold))
            if out["rank"] != 4:
                return fail("rank_deficient")
            projected = np.einsum("ij,i->j", U, bw, optimize=False)
            coeff = np.einsum("ji,j->i", Vt, projected/singular, optimize=False)
            if not _finite(coeff):
                raise FloatingPointError("Nonfinite coefficient")
            out["coefficients"] = coeff.tolist()
            residual = np.einsum("nij,j->ni", A, coeff, optimize=False) + t_unit
            error = float(np.einsum("n,ni,ni->", q, residual, residual, optimize=False))
            if not math.isfinite(error):
                raise FloatingPointError("Nonfinite training residual")
            out["training_residual_mean_squared"] = error
            if coeff[0] <= 0 or coeff[2] <= 0:
                return fail("nonpositive_fitted_slope")
    except np.linalg.LinAlgError:
        return fail("svd_failure")
    except (FloatingPointError, OverflowError):
        return fail("nonrepresentable_fit_arithmetic")
    out.update(status="completed", reason=None, seconds=time.perf_counter()-started)
    return out


def _direction(source_depth, transformed, params, target):
    n = len(source_depth)
    source_valid = np.isfinite(source_depth) & (source_depth > 0)
    finite_z = np.isfinite(transformed).all(axis=1)
    positive_z = np.isfinite(transformed[:, 2]) & (transformed[:, 2] > 0)
    projectable = source_valid & finite_z & positive_z
    costs = np.ones(n)
    valid = np.zeros(n, dtype=bool)
    invalid_projection = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(projectable)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        p = transformed[idx, :2]/transformed[idx, 2, None]
        p = p*params[:2] + params[2:]
        difference = p-target[idx]
        distance = np.hypot(difference[:, 0], difference[:, 1])
        finite = np.isfinite(p).all(axis=1) & np.isfinite(difference).all(axis=1) & np.isfinite(distance)
        valid[idx[finite]] = True
        invalid_projection[idx[~finite]] = True
        costs[idx[finite]] = np.minimum(distance[finite]/16., 1.)**2
    return costs, {"count": n, "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()), "source_invalid_count": int((~source_valid).sum()),
        "transformed_nonfinite_count": int((~finite_z).sum()),
        "transformed_nonpositive_or_nonfinite_z_count": int((~positive_z).sum()),
        "projection_nonfinite_count": int(invalid_projection.sum()),
        "capped_count": int(np.count_nonzero(costs == 1.))}


def validation_costs(fit, R, t_unit, g, z1, z2, indices):
    """Each validation match retains both directions, even after fit failure."""
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    if fit["status"] != "completed":
        return np.ones(n), {"status": "fit_unavailable", "reason": fit["reason"],
            "count": n, "direction_denominator": 2*n, "fit_failed_direction_count": 2*n,
            "forward": None, "reverse": None}
    coeff = np.asarray(fit["coefficients"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        c1 = coeff[0]*z1[indices]+coeff[1]
        c2 = coeff[2]*z2[indices]+coeff[3]
        P1 = c1[:, None]*g["h1"][indices]
        P2 = c2[:, None]*g["h2"][indices]
        Z12 = np.einsum("ij,nj->ni", R, P1, optimize=False) + t_unit
        Z21 = np.einsum("ji,nj->ni", R, P2-t_unit, optimize=False)
    forward, fm = _direction(c1, Z12, g["params2"], g["x2"][indices])
    reverse, rm = _direction(c2, Z21, g["params1"], g["x1"][indices])
    return (forward+reverse)*.5, {"status": "completed", "reason": None,
        "count": n, "direction_denominator": 2*n, "fit_failed_direction_count": 0,
        "forward": fm, "reverse": rm}


def _average(costs, weights):
    if not _finite(costs) or not _finite(weights) or np.any(costs < 0) or np.any(costs > 1):
        raise ValueError("Invalid bounded costs/weights")
    value = math.fsum(float(c)*float(w) for c, w in zip(costs, weights))
    if not math.isfinite(value):
        raise ValueError("Nonfinite bounded reduction")
    # Only endpoints can move by sum-roundoff; no score tie tolerance is used.
    return min(1., max(0., value))


def _calibrated_score(R, t_unit, g, z1, z2, support, weights, fold_ids, reason):
    n = len(support)
    costs = np.ones(n)
    records = []
    for k in (range(4) if fold_ids is not None else (None,)):
        validation = np.flatnonzero(fold_ids == k) if k is not None else np.arange(n)
        train = support & (fold_ids != k) if k is not None else support
        fit = fit_calibration(R, t_unit, g, z1, z2, train, weights, reason)
        local, metadata = validation_costs(fit, R, t_unit, g, z1, z2, validation)
        costs[validation] = local
        records.append({"fold": k, "validation_weight_mass": math.fsum(float(weights[i]) for i in validation),
                        "fit": fit, "validation": metadata})
    return _average(costs, weights), records


def score_pair(key, x1, x2, depth1, depth2, calibration, candidates):
    """Score the exact ordered four original 0017 completed records.

    Candidates require arm, status, estimate{R,t}, final_scores{pixel_msac,count}.
    Binding/provenance and scientific resource controls belong to the caller.
    No labels, files, native estimator or selection-dependent pose update.
    """
    started = time.perf_counter()
    if not isinstance(key, str) or not key:
        raise ValueError("Nonempty pair key required")
    if (not isinstance(candidates, (list, tuple)) or len(candidates) != 4
            or [c.get("arm") for c in candidates] != list(ARMS)):
        raise ValueError("The original ordered four-pose bank is required")
    g = base.prepare_geometry(x1, x2, calibration)
    n = len(g["x1"])
    depths = [base.finite_array(d, (n,), name) for d, name in ((depth1, "depth1"), (depth2, "depth2"))]
    # Validate the whole bank before fitting or producing any selected identity.
    bank = []
    for record in candidates:
        if record.get("status") != "completed":
            raise ValueError("All four original records must be completed")
        R, t = base.validate_pose(record["estimate"])
        sealed = record["final_scores"]
        expected = sealed.get("pixel_msac")
        if (type(sealed.get("count")) is not int or sealed["count"] != n
                or isinstance(expected, bool) or not isinstance(expected, (int, float))
                or not math.isfinite(expected) or not 0 <= expected <= n):
            raise ValueError("Invalid sealed geometric score/count")
        costs, metadata = pixel_costs(record["estimate"], g)
        if abs(metadata["pixel_msac"]-expected) > 1e-10*max(1, n):
            raise ValueError("Sealed geometric score mismatch")
        bank.append((R, t, costs, metadata))
    spatial, random, weights, smeta, rmeta = partitions(key, g["x1"])
    z1, m1 = normalize_depth(depths[0])
    z2, m2 = normalize_depth(depths[1])
    unavailable = "depth_normalization_unavailable" if z1 is None or z2 is None else None
    uniform = np.full(n, 1./n)
    outputs = []
    for arm, (R, t, geometry_costs, geometry_metadata) in zip(ARMS, bank):
        support = geometry_costs < 1.
        G = geometry_metadata["pixel_msac"]/n
        global_D, global_fits = _calibrated_score(R, t, g, z1, z2, support, uniform, None, unavailable)
        D = {"global_affine_ls": global_D}
        fits = {"global_affine_ls": global_fits}
        if weights is None:
            for name in SELECTORS[2:]:
                D[name] = 1.
                fits[name] = [{"status": "unavailable", "reason": "spatial_partition_unavailable"}]
        else:
            for name, ids in (("spatial_weight_only", None), ("random_point_transfer", random),
                              ("spatial_transfer", spatial)):
                D[name], fits[name] = _calibrated_score(R, t, g, z1, z2, support, weights, ids, unavailable)
        scores = {"pixel_msac": G, **{name: G+D[name] for name in SELECTORS[1:]}}
        outputs.append({"arm": arm, "unit_translation": t.tolist(), "geometry": geometry_metadata,
            "geometric_mean": G, "fit_support_count": int(support.sum()),
            "depth_scores": D, "scores": scores, "fits": fits})
    selectors = {}
    for name in SELECTORS:
        values = [row["scores"][name] for row in outputs]
        winner = min(range(4), key=lambda i: (values[i], i))
        selectors[name] = {"selected_arm": ARMS[winner], "selected_index": winner,
                           "score": values[winner], "scores": values}
    all_fits = [item["fit"] for row in outputs for group in row["fits"].values()
                for item in group if "fit" in item]
    counters = {"candidate_count": 4, "fit_calls": len(all_fits),
        "svd_calls": sum(int(f["svd_attempted"]) for f in all_fits),
        "fit_completed_count": sum(int(f["status"] == "completed") for f in all_fits),
        "fit_unavailable_count": sum(int(f["status"] != "completed") for f in all_fits)}
    return base.json_safe({"schema": "evr0020.scores.v1", "key": key, "count": n,
        "arms": list(ARMS), "normalization": {"depth1": m1, "depth2": m2},
        "spatial_partition": smeta, "random_partition": rmeta,
        "candidates": outputs, "selectors": selectors, "counters": counters,
        "seconds": time.perf_counter()-started})
