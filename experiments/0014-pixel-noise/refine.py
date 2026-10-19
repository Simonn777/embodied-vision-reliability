"""Known Sampson-noise controls with one shared, fixed NumPy pose optimizer.

N uses average inverse focal scaling; P uses per-axis pixel gradient scaling.
Neither residual is exact reprojection error. This is an independent matched
optimizer control, not source-identical PoseLib refinement: increments are LEFT
SO(3), translation uses a normalized S2 retraction, and costs are in pixel units.
No data loading, ground truth, native estimator, or external solver is used here.
"""

from __future__ import annotations

import math
import time

import numpy as np


SETTINGS = {
    "loss": "cauchy",
    "loss_scale_pixels": 0.5,
    "initial_lambda": 1e-3,
    "minimum_lambda": 1e-10,
    "maximum_lambda": 1e10,
    "lambda_multiplier": 10.0,
    "diagonal_floor": 1e-12,
    "maximum_iterations_including_rejections": 100,
    "gradient_infinity_tolerance": 1e-8,
    "step_norm_tolerance": 1e-8,
    "relative_cost_tolerance": 1e-10,
}


def _finite_array(value, shape, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return arr.copy()


def _unit(value, name="translation"):
    arr = _finite_array(value, (3,), name)
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        raise ValueError(f"{name} must be nonzero")
    arr /= scale
    return arr / np.linalg.norm(arr)


def _pose(R, t):
    R = _finite_array(R, (3, 3), "rotation")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("rotation must be orthogonal")
    if not math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-6):
        raise ValueError("rotation must be proper")
    return R, _unit(t)


def skew(v):
    """Return [v]_x; v must be a finite three-vector."""
    x, y, z = _finite_array(v, (3,), "skew input")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(omega):
    """Rodrigues exponential with a small-angle series."""
    omega = _finite_array(omega, (3,), "rotation increment")
    theta = math.hypot(*omega)
    theta2 = theta * theta
    if not math.isfinite(theta2):
        raise ValueError("rotation increment is too large")
    if theta2 < 1e-12:
        A = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        B = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        A = math.sin(theta) / theta
        B = (1.0 - math.cos(theta)) / theta2
    K = skew(omega)
    result = np.eye(3) + A * K + B * (K @ K)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite SO(3) exponential")
    return result


def tangent_basis(t):
    """Deterministic orthonormal 3x2 basis of the unit translation tangent.

    The least-aligned Cartesian axis is selected, ties in x/y/z order. The
    basis is recomputed at each accepted/current pose, never differentiated.
    """
    t = _unit(t)
    axis = np.eye(3)[int(np.argmin(np.abs(t)))]
    b1 = _unit(np.cross(t, axis), "tangent basis")
    b2 = _unit(np.cross(t, b1), "tangent basis")
    return np.column_stack((b1, b2))


def retract_pose(R, t, delta):
    """Apply exp([delta_R]x) R and normalize(t+B delta_t)."""
    R, t = _pose(R, t)
    delta = _finite_array(delta, (5,), "pose increment")
    B = tangent_basis(t)
    return so3_exp(delta[:3]) @ R, _unit(t + B @ delta[3:])


def _points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{name} must be a nonempty Nx3 array")
    if not np.all(np.isfinite(points)) or not np.all(points[:, 2] == 1.0):
        raise ValueError(f"{name} must be finite normalized homogeneous points (z=1)")
    return points.copy()


def _metric_weights(focals1, focals2, mode):
    f1 = _finite_array(focals1, (2,), "focals1")
    f2 = _finite_array(focals2, (2,), "focals2")
    if np.any(f1 <= 0.0) or np.any(f2 <= 0.0):
        raise ValueError("focal lengths must be positive")
    if mode not in ("N", "P"):
        raise ValueError("mode must be N or P")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            # Halve before adding to avoid avoidable large-focal overflow.
            mean1, mean2 = float(np.sum(f1 / 2.0)), float(np.sum(f2 / 2.0))
            a = 0.5 / mean1 + 0.5 / mean2
            weights = (np.full((2, 2), a * a) if mode == "N"
                       else np.square(1.0 / np.stack((f1, f2))))
        except (FloatingPointError, ZeroDivisionError) as exc:
            raise ValueError("unrepresentable focal scaling") from exc
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("unrepresentable focal scaling")
    return weights


def _evaluate(x, y, R, t, weights, need_jacobian=True):
    """One shared analytic kernel. The caller has validated finite inputs."""
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            E = skew(t) @ R
            g1, g2 = y @ E, x @ E.T
            e = np.sum(y * g2, axis=1)
            v = (np.sum(g1[:, :2] ** 2 * weights[0], axis=1)
                 + np.sum(g2[:, :2] ** 2 * weights[1], axis=1))
            if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
                raise ValueError("collapsed or nonfinite Sampson denominator")
            denominator = np.sqrt(v)
            r = e / denominator
            if not np.all(np.isfinite(r)):
                raise ValueError("nonfinite residual")
            if not need_jacobian:
                return r, None
            B = tangent_basis(t)
            derivatives = [skew(t) @ skew(axis) @ R for axis in np.eye(3)]
            derivatives += [skew(B[:, j]) @ R for j in range(2)]
            J = np.empty((len(x), 5), dtype=np.float64)
            for j, dE in enumerate(derivatives):
                dg1, dg2 = y @ dE, x @ dE.T
                de = np.sum(y * dg2, axis=1)
                dv = 2.0 * (np.sum(g1[:, :2] * dg1[:, :2] * weights[0], axis=1)
                            + np.sum(g2[:, :2] * dg2[:, :2] * weights[1], axis=1))
                J[:, j] = de / denominator - 0.5 * r * (dv / v)
            if not np.all(np.isfinite(J)):
                raise ValueError("nonfinite residual Jacobian")
            return r, J
        except FloatingPointError as exc:
            raise ValueError("nonfinite or collapsed residual arithmetic") from exc


def residuals_and_jacobian(x, y, R, t, focals1, focals2, mode="N"):
    """Return signed pixel-unit Sampson residuals and a left-SO3/S2 Jacobian.

    x/y are calibrated homogeneous points with third coordinate exactly one;
    focals are [fx, fy]. The Jacobian columns are rotation x/y/z then the two
    columns of tangent_basis(t). Initial translation magnitude is discarded.
    """
    x, y = _points(x, "x"), _points(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    R, t = _pose(R, t)
    return _evaluate(x, y, R, t, _metric_weights(focals1, focals2, mode))


def cauchy_cost(residuals):
    """sum log(1+(r/0.5)^2), with no hidden normalization or row weights."""
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 1 or not np.all(np.isfinite(r)):
        raise ValueError("residuals must be a finite vector")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            cost = float(np.sum(np.log1p(np.square(r / 0.5))))
        except FloatingPointError as exc:
            raise ValueError("nonfinite Cauchy cost") from exc
    if not math.isfinite(cost):
        raise ValueError("nonfinite Cauchy cost")
    return cost


def _calibrate(pixels, K, name):
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels):
        raise ValueError(f"{name} pixels must be a nonempty Nx2 array")
    if not np.all(np.isfinite(pixels)):
        raise ValueError(f"{name} pixels must be finite")
    K = _finite_array(K, (3, 3), name + " intrinsics")
    canonical = np.array([[K[0, 0], 0.0, K[0, 2]],
                          [0.0, K[1, 1], K[1, 2]], [0.0, 0.0, 1.0]])
    if not np.array_equal(K, canonical) or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError("only finite positive-focal zero-skew PINHOLE intrinsics are supported")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            normalized = ((pixels - K[:2, 2]) / np.diag(K)[:2])
        except FloatingPointError as exc:
            raise ValueError("nonfinite calibrated points") from exc
    if not np.all(np.isfinite(normalized)):
        raise ValueError("nonfinite calibrated points")
    return np.column_stack((normalized, np.ones(len(pixels)))), np.diag(K)[:2].copy()


def refine_pose(x1_pixels, x2_pixels, K1, K2, R_initial, t_initial, mode="N"):
    """Refine a supplied pose with fixed Cauchy IRLS-LM; return a JSON-ready log.

    Every attempted solve, including a rejected/nonfinite trial, consumes an
    iteration. Strict objective decrease is required for acceptance. Relative
    decrease means (old-new)/old and is tested only after an accepted step.
    The gradient stop uses the actual objective gradient 2 J.T W r, whereas
    the IRLS solve cancels the common factor two. No cheirality mask is created.
    """
    started = time.perf_counter()
    x, f1 = _calibrate(x1_pixels, K1, "camera1")
    y, f2 = _calibrate(x2_pixels, K2, "camera2")
    if x.shape != y.shape:
        raise ValueError("correspondence arrays must have matching shapes")
    R, t = _pose(R_initial, t_initial)
    weights = _metric_weights(f1, f2, mode)
    r, J = _evaluate(x, y, R, t, weights)
    cost = initial_cost = cauchy_cost(r)
    damping = SETTINGS["initial_lambda"]
    history = []
    reason, converged = "maximum_iterations", False
    for iteration in range(SETTINGS["maximum_iterations_including_rejections"]):
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                robust_weights = 1.0 / (0.25 + r * r)
                H = J.T @ (robust_weights[:, None] * J)
                gradient = J.T @ (robust_weights * r)
            except FloatingPointError as exc:
                raise ValueError("nonfinite normal equations") from exc
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(gradient)):
            raise ValueError("nonfinite normal equations")
        gradient_inf = float(2.0 * np.max(np.abs(gradient)))
        if not math.isfinite(gradient_inf):
            raise ValueError("nonfinite objective gradient")
        entry = {"iteration": iteration, "cost_before": cost,
                 "gradient_infinity": gradient_inf, "lambda_before": damping,
                 "step_norm": None, "trial_cost": None, "accepted": False,
                 "cost_after": cost, "lambda_after": damping,
                 "relative_decrease": None}
        if gradient_inf <= SETTINGS["gradient_infinity_tolerance"]:
            entry["status"] = reason = "gradient_tolerance"
            history.append(entry)
            converged = True
            break
        diagonal = np.maximum(np.diag(H), SETTINGS["diagonal_floor"])
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                system = H + damping * np.diag(diagonal)
                delta = np.linalg.solve(system, -gradient)
            if not np.all(np.isfinite(delta)):
                raise ValueError("nonfinite step")
            step_norm = math.hypot(*delta)
            if not math.isfinite(step_norm):
                raise ValueError("nonfinite step norm")
            entry["step_norm"] = step_norm
            if step_norm <= SETTINGS["step_norm_tolerance"]:
                entry["status"] = reason = "step_tolerance"
                history.append(entry)
                converged = True
                break
            R_trial, t_trial = retract_pose(R, t, delta)
            trial_r, _ = _evaluate(x, y, R_trial, t_trial, weights, False)
            trial_cost = cauchy_cost(trial_r)
            entry["trial_cost"] = trial_cost
            if trial_cost < cost:
                # A finite trial cost is insufficient if its next Jacobian is
                # invalid: validate the accepted state's full kernel first.
                trial_r, trial_J = _evaluate(x, y, R_trial, t_trial, weights)
                relative = (cost - trial_cost) / cost
                R, t, r, J, cost = R_trial, t_trial, trial_r, trial_J, trial_cost
                damping = max(SETTINGS["minimum_lambda"], damping / 10.0)
                entry.update(accepted=True, status="accepted", cost_after=cost,
                             lambda_after=damping, relative_decrease=relative)
                history.append(entry)
                if relative <= SETTINGS["relative_cost_tolerance"]:
                    reason, converged = "relative_cost_tolerance", True
                    break
                continue
            entry["status"] = "rejected_cost"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            entry["status"] = "rejected_numerical"
            entry["numerical_reason"] = str(exc)
        damping = min(SETTINGS["maximum_lambda"], damping * 10.0)
        entry["lambda_after"] = damping
        history.append(entry)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            final_gradient = 2.0 * (J.T @ (r / (0.25 + r * r)))
        except FloatingPointError as exc:
            raise ValueError("nonfinite final objective gradient") from exc
    if not np.all(np.isfinite(final_gradient)):
        raise ValueError("nonfinite final objective gradient")
    elapsed = time.perf_counter() - started
    return {
        "schema": "evr0014.refinement.v1", "mode": mode,
        "settings": dict(SETTINGS), "number_of_correspondences": len(x),
        "R": R.tolist(), "t": t.tolist(),
        "initial_cost": initial_cost, "final_cost": cost,
        "final_gradient_infinity": float(np.max(np.abs(final_gradient))),
        "converged": converged, "termination_reason": reason,
        "iteration_count": len(history),
        "accepted_steps": sum(row["accepted"] for row in history),
        "iterations": history, "elapsed_seconds": elapsed,
    }
