"""Known one-image normal/depth consistency controls; no data or pose access.

All offsets use depth divided by the whole positive model-valid depth median.
The prepared arrays are private numerical inputs; describe() and solve() return
finite JSON-compatible metadata. No correction is returned for unavailable fits.
"""
import math
import time

import numpy as np
from scipy.optimize import minimize_scalar


POLICY = {
    "grid_start": 2,
    "grid_stride": 4,
    "stencil_radius": 1,
    "maximum_stencil_depth_ratio": 1.1,
    "positive_depth_margin": 1e-6,
    "maximum_offset": 4.0,
    "angular_grid_count": 65,
    "angular_extra_zero_evaluation": True,
    "angular_flat_spread": 1e-12,
    "bounded_maxiter": 64,
    "bounded_xatol": 1e-8,
    "irls_updates": 5,
    "mad_multiplier": 1.4826,
    "huber_delta": 1.345,
}
EPS = np.finfo(np.float64).eps
METHODS = ("ANG", "LS", "IRLS")


def _array(value, shape, name, finite=False):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(name + " shape mismatch")
    if finite and not np.isfinite(a).all():
        raise ValueError(name + " must be finite")
    return a.copy()


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite arithmetic")
    return value


def _scalar(value):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError("offset must be a finite real scalar")
    a = np.asarray(value, dtype=np.float64)
    if a.shape != () or not np.isfinite(a):
        raise ValueError("offset must be a finite real scalar")
    return float(a)


def _unit_rows(a):
    """Scale before squaring; invalid/zero rows remain explicitly invalid."""
    finite = np.isfinite(a).all(axis=-1)
    safe = np.where(finite[..., None], a, 0.0)
    scale = np.max(np.abs(safe), axis=-1)
    valid = finite & (scale > 0.0)
    unit = np.zeros_like(a)
    if valid.any():
        scaled = safe[valid] / scale[valid, None]
        norm = np.sqrt(np.sum(scaled * scaled, axis=-1))
        unit[valid] = _finite(scaled / norm[:, None])
    return unit, valid


def _median_positive(values):
    """The ordinary median, avoiding overflow of the even-count midpoint."""
    n = len(values)
    v = np.partition(values.copy(), (n - 1) // 2 if n % 2 else (n // 2 - 1, n // 2))
    if n % 2:
        return float(v[n // 2])
    low, high = v[n // 2 - 1], v[n // 2]
    return float(low + (high - low) * 0.5)


def prepare(depth, normal, mask, K):
    """Prepare a fixed support without reading any external input.

    Require all nine stencil pixels to have valid positive depth and finite,
    nonzero normals. Central gradients use the two immediate axial neighbors.
    Invalid full-map locations are NaN in normalized_depth and zero in normals;
    valid_depth/valid_normal must accompany either map.
    """
    start = time.perf_counter()
    if np.ndim(depth) != 2:
        raise ValueError("depth must be H x W")
    h, w = np.shape(depth)
    if min(h, w) < 1:
        raise ValueError("empty image")
    d = _array(depth, (h, w), "depth")
    n = _array(normal, (h, w, 3), "normal")
    m = np.asarray(mask)
    if m.shape != (h, w) or m.dtype != np.bool_:
        raise ValueError("mask must be bool H x W")
    m = m.copy()
    camera = _array(K, (3, 3), "K", finite=True)
    if (not np.array_equal(camera[2], [0., 0., 1.]) or camera[1, 0] != 0.
            or camera[0, 0] <= 0. or camera[1, 1] <= 0.):
        raise ValueError("K must be an upper-triangular pixel pinhole calibration")
    valid = m & np.isfinite(d) & (d > 0.)
    unit, normal_valid = _unit_rows(n)
    counts = {
        "pixels": int(h * w),
        "model_mask_pixels": int(m.sum()),
        "model_mask_nonfinite_depth": int((m & ~np.isfinite(d)).sum()),
        "model_mask_nonpositive_finite_depth": int((m & np.isfinite(d) & (d <= 0.)).sum()),
        "positive_model_valid_pixels": int(valid.sum()),
        "positive_model_valid_nonzero_finite_normal_pixels": int((valid & normal_valid).sum()),
        "grid_samples": 0, "depth_stencil_valid": 0,
        "normal_stencil_valid": 0, "joint_stencil_valid": 0,
        "discontinuity_rejected": 0, "eligible_samples": 0,
    }
    out = {
        "schema": "evr0028.normal-offset-prepared.v1",
        "status": "unavailable", "reason": None, "shape": [int(h), int(w)],
        "median_raw": None, "domain": None, "counts": counts,
        "normalized_depth": np.full((h, w), np.nan, dtype=np.float64),
        "valid_depth": valid, "normalized_normal": unit,
        "valid_normal": normal_valid, "K": camera, "samples": None,
    }
    if not valid.any():
        out["reason"] = "no_positive_model_valid_depth"
        out["elapsed_seconds"] = time.perf_counter() - start
        return out
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        median = _median_positive(d[valid])
        nd = _finite(d[valid] / median)
        if not np.all(nd > 0.):
            raise FloatingPointError("normalized valid depth collapsed")
        out["median_raw"] = median
        out["normalized_depth"][valid] = nd
        lower = -float(np.min(nd)) + POLICY["positive_depth_margin"]
        out["domain"] = [lower, POLICY["maximum_offset"]]
        if not lower <= 0. <= POLICY["maximum_offset"]:
            out["reason"] = "zero_not_in_physical_domain"
            out["elapsed_seconds"] = time.perf_counter() - start
            return out

        yy, xx = np.meshgrid(np.arange(2, h - 2, 4, dtype=np.int64),
                             np.arange(2, w - 2, 4, dtype=np.int64), indexing="ij")
        y, x = yy.ravel(), xx.ravel()
        counts["grid_samples"] = int(len(x))
        depth_ok = np.ones(len(x), dtype=bool)
        normal_ok = np.ones(len(x), dtype=bool)
        low, high = np.full(len(x), np.inf), np.full(len(x), -np.inf)
        dn = out["normalized_depth"]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ok = valid[y + dy, x + dx]
                depth_ok &= ok
                normal_ok &= normal_valid[y + dy, x + dx]
                z = dn[y + dy, x + dx]
                low = np.minimum(low, np.where(ok, z, np.inf))
                high = np.maximum(high, np.where(ok, z, -np.inf))
        joint = depth_ok & normal_ok
        continuous = np.zeros(len(x), dtype=bool)
        continuous[joint] = high[joint] / low[joint] <= POLICY["maximum_stencil_depth_ratio"]
        keep = joint & continuous
        counts.update(depth_stencil_valid=int(depth_ok.sum()),
                      normal_stencil_valid=int(normal_ok.sum()),
                      joint_stencil_valid=int(joint.sum()),
                      discontinuity_rejected=int((joint & ~continuous).sum()),
                      eligible_samples=int(keep.sum()))
        if not keep.any():
            out["reason"] = "no_eligible_samples"
            out["elapsed_seconds"] = time.perf_counter() - start
            return out
        x, y = x[keep], y[keep]
        kinv = _finite(np.linalg.inv(camera))
        ex, ey = kinv[:, 0], kinv[:, 1]
        ray = _finite(x[:, None] * ex + y[:, None] * ey + kinv[:, 2])
        gradient = _finite(np.column_stack(
            ((dn[y, x + 1] - dn[y, x - 1]) * .5,
             (dn[y + 1, x] - dn[y - 1, x]) * .5)))
        depth_sample, normal_sample = dn[y, x].copy(), unit[y, x].copy()
        k = _finite(np.cross(ex, ey))
        ell = _finite(gradient[:, 1, None] * np.cross(ex, ray)
                      + gradient[:, 0, None] * np.cross(ray, ey))
        a = _finite(np.column_stack(
            (np.einsum("ni,i->n", normal_sample, ex, optimize=False),
             np.einsum("ni,i->n", normal_sample, ey, optimize=False))))
        dot = _finite(np.einsum("ni,ni->n", normal_sample, ray, optimize=False))
        response = _finite(-depth_sample[:, None] * a - dot[:, None] * gradient)
        out["samples"] = {
            "xy": np.column_stack((x, y)), "flat_indices": y * w + x,
            "d": depth_sample, "normal": normal_sample, "r": ray,
            "g": gradient, "k": k, "ell": ell, "ls_a": a, "ls_y": response,
        }
    out.update(status="completed", reason=None, elapsed_seconds=time.perf_counter() - start)
    return out


def describe(prepared):
    """Finite JSON metadata; numerical arrays remain separate."""
    return {k: prepared[k] for k in
            ("schema", "status", "reason", "shape", "median_raw", "domain", "counts", "elapsed_seconds")}


def angular_loss(prepared, b):
    """Mean 1-(unit(v) dot n)^2, with no output-dependent sample dropping."""
    b = _scalar(b)
    if prepared["status"] != "completed":
        raise ValueError("prepared support unavailable")
    s = prepared["samples"]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        v = _finite((s["d"] + b)[:, None] * s["k"] + s["ell"])
        unit, valid = _unit_rows(v)
        if not valid.all():
            raise FloatingPointError("collapsed angular tangent normal")
        dot = _finite(np.einsum("ni,ni->n", unit, s["normal"], optimize=False))
        if np.max(np.abs(dot)) > 1. + 64. * EPS:
            raise FloatingPointError("unit-normal dot outside rounding enclosure")
        # Only the roundoff excursion of a cosine is clipped, never b or depth.
        dot = np.clip(dot, -1., 1.)
        return float(_finite(np.mean(1. - dot * dot)))


def _ls(prepared, weights):
    s = prepared["samples"]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        aa = _finite(np.sum(s["ls_a"] * s["ls_a"], axis=1))
        ay = _finite(np.sum(s["ls_a"] * s["ls_y"], axis=1))
        denominator = float(_finite(np.sum(weights * aa)))
        numerator = float(_finite(np.sum(weights * ay)))
        if denominator == 0.:
            return None, numerator, denominator
        b = float(_finite(numerator / denominator))
        return b, numerator, denominator


def _physical(prepared, b):
    low, high = prepared["domain"]
    result = {"in_domain": bool(low <= b <= high),
              "original_valid_count": prepared["counts"]["positive_model_valid_pixels"],
              "positive_finite_count": 0, "minimum_corrected_depth": None}
    with np.errstate(over="raise", invalid="raise"):
        values = _finite(prepared["normalized_depth"][prepared["valid_depth"]] + b)
    result["positive_finite_count"] = int(np.sum(values > 0.))
    result["minimum_corrected_depth"] = float(np.min(values))
    result["admissible"] = bool(result["in_domain"] and np.all(values > 0.))
    return result


def _finish(prepared, out, b, status="completed", reason=None, final_loss=None):
    out["b"] = float(b)
    out["physical"] = _physical(prepared, b)
    if not out["physical"]["admissible"]:
        out.update(status="unavailable", reason="offset_out_of_domain_or_nonpositive")
        return out
    loss = angular_loss(prepared, b) if final_loss is None else final_loss
    out.update(status=status, reason=reason, admissible=True,
               correction_applied=bool(b != 0.), angular_loss=loss)
    return out


def _angular(prepared, out):
    low, high = prepared["domain"]
    history = out["history"]

    def objective(b, phase):
        entry = {"index": len(history), "phase": phase, "b": float(b),
                 "loss": None, "status": "unavailable", "reason": None}
        history.append(entry)
        try:
            value = angular_loss(prepared, b)
        except (FloatingPointError, OverflowError) as exc:
            entry["reason"] = type(exc).__name__ + ": " + str(exc)
            return math.inf
        entry.update(loss=value, status="completed")
        return value

    grid = np.linspace(low, high, POLICY["angular_grid_count"])
    for b in grid:
        objective(float(b), "grid")
    zero_loss = objective(0., "extra_zero")
    finite_grid = [r for r in history if r["status"] == "completed"]
    if len(finite_grid) != len(history):
        out.update(reason="nonfinite_angular_grid")
        return out
    spread = max(r["loss"] for r in history) - min(r["loss"] for r in history)
    out["grid_loss_spread"] = float(spread)
    out["zero_loss"] = zero_loss
    if spread <= POLICY["angular_flat_spread"]:
        out["selected_evaluation"] = len(history) - 1
        return _finish(prepared, out, 0., "uninformative", "flat_angular_loss", zero_loss)
    best = min(finite_grid, key=lambda r: (r["loss"], r["index"]))
    knots = sorted(set([float(b) for b in grid] + [0.]))
    j = knots.index(best["b"])
    bracket = [knots[max(0, j - 1)], knots[min(len(knots) - 1, j + 1)]]
    out["bounded_interval"] = bracket
    result = minimize_scalar(lambda b: objective(b, "bounded"), bounds=tuple(bracket),
                             method="bounded",
                             options={"maxiter": POLICY["bounded_maxiter"],
                                      "xatol": POLICY["bounded_xatol"]})
    finite_result = all(math.isfinite(float(v)) for v in (result.x, result.fun))
    out["optimizer"] = {"success": bool(result.success), "status": int(result.status),
                        "message": str(result.message), "nfev": int(result.nfev),
                        "x": float(result.x) if math.isfinite(float(result.x)) else None,
                        "fun": float(result.fun) if math.isfinite(float(result.fun)) else None}
    if not result.success or not finite_result or any(
            r["status"] != "completed" for r in history):
        out["reason"] = "bounded_optimizer_failed_or_nonfinite"
        return out
    selected = min(history, key=lambda r: (r["loss"], r["index"]))
    out["selected_evaluation"] = selected["index"]
    return _finish(prepared, out, selected["b"], final_loss=selected["loss"])


def _linear(prepared, out, robust):
    s = prepared["samples"]
    weights = np.ones(len(s["d"]), dtype=np.float64)
    b, numerator, denominator = _ls(prepared, weights)
    out["history"].append({"phase": "initial_ls", "b": b,
                           "numerator": numerator, "denominator": denominator})
    if b is None:
        out["reason"] = "zero_ls_information"
        return out
    if robust:
        for step in range(POLICY["irls_updates"]):
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                residual = _finite(s["ls_a"] * b - s["ls_y"])
                norms = _finite(np.hypot(residual[:, 0], residual[:, 1]))
                center = float(np.median(norms))
                scale = float(_finite(POLICY["mad_multiplier"] * np.median(np.abs(norms - center))))
                row = {"phase": "huber_update", "update": step + 1, "b_before": b,
                       "residual_norm_median": center, "scale": scale}
                out["history"].append(row)
                if scale == 0.:
                    row.update(status="stopped", reason="zero_robust_scale", b_after=b)
                    out["robust_stop"] = "zero_robust_scale"
                    break
                cutoff = float(_finite(POLICY["huber_delta"] * scale))
                weights = np.ones_like(norms)
                large = norms > cutoff
                weights[large] = cutoff / norms[large]
                next_b, numerator, denominator = _ls(prepared, weights)
                row.update(cutoff=cutoff, numerator=numerator, denominator=denominator,
                           downweighted_samples=int(large.sum()), b_after=next_b)
                if next_b is None:
                    row.update(status="unavailable", reason="zero_weighted_ls_information")
                    out["reason"] = "zero_weighted_ls_information"
                    return out
                row.update(status="completed", reason=None)
                b = next_b
    return _finish(prepared, out, b)


def solve(prepared, method="ANG"):
    """Fit exactly one declared control; no corrected-map fallback is implicit."""
    if method not in METHODS:
        raise ValueError("unknown method")
    start = time.perf_counter()
    out = {"schema": "evr0028.normal-offset.v1", "method": method,
           "status": "unavailable", "reason": None, "admissible": False,
           "correction_applied": False, "b": None, "angular_loss": None,
           "physical": None, "preparation": describe(prepared), "policy": dict(POLICY),
           "history": []}
    if prepared["status"] != "completed":
        out["reason"] = "preparation_unavailable: " + str(prepared["reason"])
    else:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                out = _angular(prepared, out) if method == "ANG" else _linear(prepared, out, method == "IRLS")
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            out.update(status="unavailable", reason="numerical_error: " + type(exc).__name__ + ": " + str(exc),
                       admissible=False, correction_applied=False)
    out["elapsed_seconds"] = time.perf_counter() - start
    return out


def corrected_depth(prepared, result):
    """Return normalized corrected depth, NaN off original validity, or None."""
    if not result["admissible"]:
        return None
    b = _scalar(result["b"])
    physical = _physical(prepared, b)
    if not physical["admissible"]:
        raise ValueError("result no longer satisfies physical domain")
    corrected = prepared["normalized_depth"].copy()
    with np.errstate(over="raise", invalid="raise"):
        corrected[prepared["valid_depth"]] = _finite(corrected[prepared["valid_depth"]] + b)
    return corrected
