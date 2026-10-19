"""Artificial-tested asymmetric photometric registration; no IO or pose solver.

All 961 pixels train the fit. Engineering settings below are not published
defaults. Parameters are [delta_x, delta_y, 15*A00, 15*A01, 15*A10,
15*A11, gain, bias]. The source query is fixed; only the target is updated.
"""
from __future__ import annotations

import math
import time
import numpy as np


SETTINGS = {
    "patch_width": 31, "maximum_batch": 64, "maximum_attempts": 30,
    "initial_lambda": 1e-3, "lambda_min": 1e-10, "lambda_max": 1e10,
    "diagonal_floor": 1e-12, "gradient_tolerance": 1e-8,
    "step_tolerance": 1e-6, "relative_cost_tolerance": 1e-10,
    "gain_min": 0.25, "gain_max": 4.0, "maximum_delta": 4.0,
    "relative_singular_min": 0.5, "relative_singular_max": 2.0,
    "maximum_normalized_condition": 1e8, "maximum_relative_gn_decrease": 1e-8,
}
_oy, _ox = np.meshgrid(np.arange(-15, 16, dtype=np.float64),
                        np.arange(-15, 16, dtype=np.float64), indexing="ij")
OFFSETS = np.stack((_ox.ravel(), _oy.ravel()), axis=1)
OFFSETS.setflags(write=False)


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite {name}")


def _image(value, name):
    image = np.asarray(value, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError(f"{name} must be a grayscale image at least 2 by 2")
    _finite(image, name)
    if np.any(image < 0) or np.any(image > 1):
        raise ValueError(f"{name} intensities must lie in [0,1]")
    return image


def _inside(image, xy):
    return (np.all(np.isfinite(xy), axis=(-2, -1)) &
            np.all((xy[..., 0] >= 0) & (xy[..., 0] <= image.shape[1]-1) &
                   (xy[..., 1] >= 0) & (xy[..., 1] <= image.shape[0]-1), axis=-1))


def _bilinear(image, xy):
    """Value and piecewise derivative of the identical bilinear interpolant.

    The last row/column use the adjacent interior cell, including its one-sided
    derivative. At other integer grid lines the right/bottom cell is used.
    """
    _finite(xy, "sample coordinates")
    if not np.all(_inside(image, xy)):
        raise ValueError("sample coordinates outside image")
    x, y = xy[..., 0], xy[..., 1]
    ix = np.minimum(np.floor(x).astype(np.int64), image.shape[1]-2)
    iy = np.minimum(np.floor(y).astype(np.int64), image.shape[0]-2)
    ax, ay = x-ix, y-iy
    z00, z10 = image[iy, ix], image[iy, ix+1]
    z01, z11 = image[iy+1, ix], image[iy+1, ix+1]
    value = (1-ay)*((1-ax)*z00+ax*z10)+ay*((1-ax)*z01+ax*z11)
    dx = (1-ay)*(z10-z00)+ay*(z11-z01)
    dy = (1-ax)*(z01-z00)+ax*(z11-z10)
    return value, np.stack((dx, dy), axis=-1)


def initial_parameters(A):
    A = np.asarray(A, dtype=np.float64)
    if A.shape[-2:] != (2, 2):
        raise ValueError("A must end in shape (2,2)")
    _finite(A, "A")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            theta = np.zeros(A.shape[:-2]+(8,), dtype=np.float64)
            theta[..., 2:6] = 15*A.reshape(A.shape[:-2]+(4,))
            theta[..., 6] = 1
        except FloatingPointError as exc:
            raise ValueError("overflow forming scaled affine parameters") from exc
    _finite(theta, "parameters")
    return theta


def _warp(q, theta):
    A = theta[:, 2:6].reshape(-1, 2, 2)/15
    return q[:, None, :] + theta[:, None, :2] + np.einsum(
        "bij,nj->bni", A, OFFSETS, optimize=False)


def _evaluate(image2, template, q, theta):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            target, gradient = _bilinear(image2, _warp(q, theta))
            residual = target-theta[:, 6, None]*template-theta[:, 7, None]
            J = np.empty(residual.shape+(8,), dtype=np.float64)
            J[..., :2] = gradient
            J[..., 2:4] = gradient[..., 0, None]*(OFFSETS[None, :, :]/15)
            J[..., 4:6] = gradient[..., 1, None]*(OFFSETS[None, :, :]/15)
            J[..., 6] = -template
            J[..., 7] = -1
            cost = 0.5*np.einsum("bn,bn->b", residual, residual, optimize=False)
        except FloatingPointError as exc:
            raise ValueError("nonfinite photometric arithmetic") from exc
    for value, name in ((residual, "residual"), (J, "Jacobian"), (cost, "cost")):
        _finite(value, name)
    return residual, J, cost


def residuals_and_jacobian(image1, image2, p, q, theta):
    """Single-patch unguarded-state evaluation for numerical checking.

    Images, finite arithmetic, and complete patch bounds remain strict. The
    optimizer's gain/displacement/affine constraints are separate.
    """
    image1, image2 = _image(image1, "image1"), _image(image2, "image2")
    p, q, theta = (np.asarray(v, dtype=np.float64) for v in (p, q, theta))
    if p.shape != (2,) or q.shape != (2,) or theta.shape != (8,):
        raise ValueError("expected p/q shape (2,) and theta shape (8,)")
    for value, name in ((p, "p"), (q, "q"), (theta, "theta")):
        _finite(value, name)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            template, _ = _bilinear(image1, p[None, None, :]+OFFSETS[None, :, :])
            r, J, _ = _evaluate(image2, template, q[None, :], theta[None, :])
        except FloatingPointError as exc:
            raise ValueError("nonfinite sampling arithmetic") from exc
    return r[0], J[0]


def _normal(r, J):
    # Avoid large BLAS matmuls: Accelerate can raise spurious NumPy FPE flags.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            H = np.einsum("bni,bnj->bij", J, J, optimize=False)
            g = np.einsum("bni,bn->bi", J, r, optimize=False)
        except FloatingPointError as exc:
            raise ValueError("nonfinite normal equations") from exc
    _finite(H, "normal matrix")
    _finite(g, "gradient")
    return H, g


def _guard(image2, q, theta, inverse_A0):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _finite(theta, "trial parameters")
            if not SETTINGS["gain_min"] <= theta[6] <= SETTINGS["gain_max"]:
                return "gain_guard"
            if float(np.hypot(theta[0], theta[1])) > SETTINGS["maximum_delta"]:
                return "displacement_guard"
            A = theta[2:6].reshape(2, 2)/15
            det = float(A[0, 0]*A[1, 1]-A[0, 1]*A[1, 0])
            if not math.isfinite(det) or det <= 0:
                return "affine_determinant_guard"
            relative = np.einsum("ij,jk->ik", A, inverse_A0, optimize=False)
            _finite(relative, "relative affine matrix")
            relative_det = float(relative[0, 0]*relative[1, 1]-relative[0, 1]*relative[1, 0])
            if not math.isfinite(relative_det) or relative_det <= 0:
                return "relative_affine_determinant_guard"
            sv = np.linalg.svd(relative, compute_uv=False)
            _finite(sv, "relative affine singular values")
            if sv.min() < SETTINGS["relative_singular_min"] or sv.max() > SETTINGS["relative_singular_max"]:
                return "relative_affine_guard"
            if not bool(_inside(image2, _warp(q[None, :], theta[None, :]))[0]):
                return "target_patch_bounds"
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return "invalid_trial_arithmetic"
    return None


def _diagnostic(r, J, cost):
    """Undamped local GN decrease, not a bound on remaining true improvement."""
    H, g = _normal(r[None, :], J[None, :, :])
    H, g = H[0], g[0]
    diag = np.diag(H)
    if np.any(diag <= 0):
        return {"valid": False, "reason": "nonpositive_normal_diagonal",
                "relative_gn_decrease": None, "normalized_condition": None}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scale = np.sqrt(diag)
            normalized = H/scale[:, None]/scale[None, :]
            L = np.linalg.cholesky(normalized)
            eigenvalues = np.linalg.eigvalsh(normalized)
            condition = float(eigenvalues[-1]/eigenvalues[0])
            z = np.linalg.solve(L, g/scale)
            ratio = float(np.einsum("i,i->", z, z, optimize=False)/(2*max(1.0, cost)))
            if not math.isfinite(condition) or not math.isfinite(ratio) or condition <= 0 or ratio < 0:
                raise ValueError("invalid final GN diagnostic")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return {"valid": False, "reason": "nonpositive_or_invalid_normal_matrix",
                "relative_gn_decrease": None, "normalized_condition": None}
    return {"valid": True, "reason": None, "relative_gn_decrease": ratio,
            "normalized_condition": condition}


def refine_batch(image1, image2, p, q, A0):
    """Return finite JSON-compatible records for 1..64 patches on one image pair.

    No state is silently replaced by the input. `admitted` alone authorizes a
    caller to use the returned target; a rejected finite state is diagnostic.
    Batch wall elapsed at finalization is not per-patch isolated compute time.
    """
    start = time.perf_counter()
    image1, image2 = _image(image1, "image1"), _image(image2, "image2")
    p, q, A0 = (np.asarray(v, dtype=np.float64) for v in (p, q, A0))
    if p.ndim != 2 or p.shape[1:] != (2,) or not 1 <= len(p) <= 64 or q.shape != p.shape or A0.shape != (len(p), 2, 2):
        raise ValueError("expected 1..64 p/q points and one 2x2 A0 per point")
    for value, name in ((p, "p"), (q, "q"), (A0, "A0")):
        _finite(value, name)
    B = len(p)
    theta = np.zeros((B, 8), dtype=np.float64)
    inverse = np.zeros_like(A0)
    template = np.zeros((B, 961), dtype=np.float64)
    residual = np.zeros((B, 961), dtype=np.float64)
    jacobian = np.zeros((B, 961, 8), dtype=np.float64)
    costs = np.zeros(B, dtype=np.float64)
    damping = np.full(B, SETTINGS["initial_lambda"])
    accepted = np.zeros(B, dtype=np.int64)
    active = np.zeros(B, dtype=bool)
    initialized = np.zeros(B, dtype=bool)
    records = []
    for i in range(B):
        rec = {"patch_index": i, "source": p[i].tolist(), "original_target": q[i].tolist(),
               "original_A": A0[i].tolist(), "initial_cost": None, "final_cost": None,
               "admitted": False, "changed": False, "status": "initialization_error", "reason": None,
               "termination": None, "accepted_steps": 0, "attempts": [], "final": None,
               "initial_diagnostic": None, "final_diagnostic": None,
               "batch_wall_elapsed_at_finalization": None}
        records.append(rec)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                theta[i] = initial_parameters(A0[i])
                inverse[i] = np.linalg.inv(A0[i])
                _finite(inverse[i], "inverse A0")
                reason = _guard(image2, q[i], theta[i], inverse[i])
                if reason:
                    raise ValueError(reason)
                template[i], _ = _bilinear(image1, p[i][None, :]+OFFSETS)
            active[i] = initialized[i] = True
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rec["reason"] = str(exc)
            rec["batch_wall_elapsed_at_finalization"] = time.perf_counter()-start
    good = np.flatnonzero(active)
    if len(good):
        residual[good], jacobian[good], costs[good] = _evaluate(image2, template[good], q[good], theta[good])
        for i in good:
            records[i]["initial_cost"] = float(costs[i])
            records[i]["initial_diagnostic"] = _diagnostic(residual[i], jacobian[i], float(costs[i]))

    for attempt in range(1, SETTINGS["maximum_attempts"]+1):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        Hs, gs = _normal(residual[ids], jacobian[ids])
        trial_ids, trial_states, trial_entries = [], [], []
        for k, i in enumerate(ids):
            rec, H, g = records[i], Hs[k], gs[k]
            gradient = float(np.max(np.abs(g)))
            entry = {"attempt": attempt, "cost_before": float(costs[i]),
                     "lambda_before": float(damping[i]), "gradient_infinity": gradient,
                     "step_norm": None, "trial_cost": None, "accepted": False,
                     "cost_after": float(costs[i]), "lambda_after": float(damping[i]),
                     "relative_cost_decrease": None, "status": None}
            rec["attempts"].append(entry)
            if gradient <= SETTINGS["gradient_tolerance"]:
                rec["termination"] = entry["status"] = "gradient_tolerance"
                active[i] = False
                continue
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    system = H.copy()
                    system.flat[::9] += damping[i]*np.maximum(np.diag(H), SETTINGS["diagonal_floor"])
                    step = np.linalg.solve(system, -g)
                    _finite(step, "LM step")
                    step_norm = float(np.sqrt(np.einsum("i,i->", step, step, optimize=False)))
                    if not math.isfinite(step_norm):
                        raise ValueError("nonfinite step norm")
                    entry["step_norm"] = step_norm
                    if step_norm <= SETTINGS["step_tolerance"]:
                        rec["termination"] = entry["status"] = "step_tolerance"
                        active[i] = False
                        continue
                    trial = theta[i]+step
                    reason = _guard(image2, q[i], trial, inverse[i])
                    if reason:
                        entry["status"] = reason
                    else:
                        trial_ids.append(i)
                        trial_states.append(trial)
                        trial_entries.append(entry)
                        continue
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                entry["status"] = "rejected_step_arithmetic"
                entry["error"] = str(exc)
            damping[i] = min(SETTINGS["lambda_max"], damping[i]*10)
            entry["lambda_after"] = float(damping[i])
        if trial_ids:
            ti = np.asarray(trial_ids, dtype=np.int64)
            ts = np.asarray(trial_states)
            tr, tj, tc = _evaluate(image2, template[ti], q[ti], ts)
            for k, i in enumerate(ti):
                entry = trial_entries[k]
                entry["trial_cost"] = float(tc[k])
                if tc[k] < costs[i]:
                    relative = float((costs[i]-tc[k])/costs[i])
                    theta[i], residual[i], jacobian[i], costs[i] = ts[k], tr[k], tj[k], tc[k]
                    accepted[i] += 1
                    damping[i] = max(SETTINGS["lambda_min"], damping[i]/10)
                    entry.update(accepted=True, status="accepted", cost_after=float(costs[i]),
                                 lambda_after=float(damping[i]), relative_cost_decrease=relative)
                    if relative <= SETTINGS["relative_cost_tolerance"]:
                        records[i]["termination"] = "relative_cost_tolerance"
                        active[i] = False
                else:
                    damping[i] = min(SETTINGS["lambda_max"], damping[i]*10)
                    entry.update(status="rejected_cost", lambda_after=float(damping[i]))
    for i in range(B):
        rec = records[i]
        if not initialized[i]:
            continue
        if active[i]:
            rec["termination"] = "iteration_exhaustion"
        diag = _diagnostic(residual[i], jacobian[i], float(costs[i]))
        rec.update(final_cost=float(costs[i]), accepted_steps=int(accepted[i]), final_diagnostic=diag,
                   final={"target": (q[i]+theta[i, :2]).tolist(), "delta": theta[i, :2].tolist(),
                          "A": (theta[i, 2:6].reshape(2, 2)/15).tolist(),
                          "gain": float(theta[i, 6]), "bias": float(theta[i, 7])})
        reasons = []
        if rec["termination"] == "iteration_exhaustion":
            reasons.append("iteration_exhaustion")
        if not diag["valid"]:
            reasons.append(diag["reason"])
        else:
            if diag["normalized_condition"] > SETTINGS["maximum_normalized_condition"]:
                reasons.append("normalized_condition")
            if diag["relative_gn_decrease"] > SETTINGS["maximum_relative_gn_decrease"]:
                reasons.append("relative_gn_decrease")
        rec["admitted"] = not reasons
        rec["changed"] = bool(np.any(theta[i] != initial_parameters(A0[i])))
        rec["status"] = "admitted" if not reasons else "not_admitted"
        rec["reason"] = reasons
        rec["batch_wall_elapsed_at_finalization"] = time.perf_counter()-start
    return {"schema": "evr0015.patch-refinement.v1", "settings": dict(SETTINGS),
            "conventions": {"cost": "0.5*sum(r^2), all 961 pixels",
                            "relative_gn_decrease": "g^T H^-1 g/(2*max(1,cost))",
                            "relative_affine": "A @ inv(A0)",
                            "timing": "batch wall time; patch elapsed is not isolated compute"},
            "patches": records, "timing_seconds": time.perf_counter()-start}
