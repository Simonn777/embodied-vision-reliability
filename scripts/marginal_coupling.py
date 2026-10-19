"""Prospective, NumPy-only marginal coupling; no model or data access.

Joint/cost arrays are indexed [y, x]; point coordinates are returned [x, y].
The native core emits its y logits first, despite internal variable names.
The probability floor below is an explicit prospective convention. It is not
bitwise parity with native softmax when the floor changes any probability.
Sinkhorn here is a standard entropic-transport baseline, not a new solver.
"""

from typing import NamedTuple, Optional

import numpy as np


PROBABILITY_FLOOR = 1e-12
LOGIT_MULTIPLIER = 0.5


class CouplingResult(NamedTuple):
    joint: Optional[np.ndarray]
    iterations: int
    row_error: float
    column_error: float
    residual: float
    converged: bool


def _array(value, ndim, name):
    original = np.asarray(value)
    if original.dtype.kind not in "fiu":
        raise TypeError(name + " must contain real numeric values")
    array = np.asarray(original, dtype=np.float64)
    if array.ndim != ndim or not all(array.shape):
        raise ValueError(name + " has an invalid shape")
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _probability(value, ndim, name):
    array = _array(value, ndim, name)
    if np.any(array <= 0) or np.any(array > 1):
        raise ValueError(name + " must lie in (0,1]; no implicit floor")
    total = np.sum(array, dtype=np.float64)
    if not np.isfinite(total) or not np.isclose(total, 1.0, rtol=0, atol=1e-12):
        raise ValueError(name + " must sum to one")
    return array


def probabilities_from_logits(logits):
    """One head: stable softmax(logits * .5), floor at 1e-12, renormalize."""
    values = _array(logits, 1, "logits") * LOGIT_MULTIPLIER
    # Multiplication before subtraction also bounds finite float64 differences.
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        probabilities = np.exp(values - np.max(values))
    probabilities /= probabilities.sum()
    probabilities = np.maximum(probabilities, PROBABILITY_FLOOR)
    return probabilities / probabilities.sum()


def native_marginals(core_logits):
    """Return (p_x, p_y) from one native 512-vector ordered [y256, x256]."""
    values = _array(core_logits, 1, "core_logits")
    if values.shape != (512,):
        raise ValueError("one native coordinate head must have 512 logits")
    p_y = probabilities_from_logits(values[:256])
    p_x = probabilities_from_logits(values[256:])
    return p_x, p_y


def normalize_joint(weights):
    """Explicitly normalize a positive grid field (including a warped view)."""
    values = _array(weights, 2, "weights")
    if np.any(values <= 0):
        raise ValueError("weights must be strictly positive")
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        scaled = values / np.max(values)
        joint = scaled / scaled.sum()
    if not np.isfinite(joint).all() or np.any(joint <= 0):
        raise ValueError("joint normalization lost finite positive support")
    return joint


def independent_joint(p_x, p_y):
    """Construct Q[y,x] = p_y[y] p_x[x] without changing either marginal."""
    x = _probability(p_x, 1, "p_x")
    y = _probability(p_y, 1, "p_y")
    joint = np.outer(y, x)
    if np.any(joint <= 0):
        raise ValueError("independent joint underflowed")
    return joint


def _joint_pair(first, second):
    first = _probability(first, 2, "first joint")
    second = _probability(second, 2, "second joint")
    if first.shape != second.shape:
        raise ValueError("joint shapes differ")
    return first, second


def poe_joint(first, second):
    """Normalized product of two normalized positive joint distributions."""
    first, second = _joint_pair(first, second)
    log_weights = np.log(first) + np.log(second)
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        weights = np.exp(log_weights - np.max(log_weights))
    return normalize_joint(weights)


def mixture_joint(first, second):
    """The fixed equal-weight arithmetic mixture, not a fitted mixture."""
    first, second = _joint_pair(first, second)
    return 0.5 * first + 0.5 * second


def _logsumexp(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def entropic_coupling(p_x, p_y, cost_yx, *, max_iterations=1000, tolerance=1e-7):
    """Minimize <C,Q> + KL(Q || outer(p_y,p_x)) with fixed marginals.

    Float64 log-domain Sinkhorn, unit initial scaling, row then column update.
    Entropy coefficient is exactly one. Residual is the maximum absolute error
    of any individual row or column sum. No retries or tolerance adjustment.
    A capped nonconverged result has joint=None and must not be decoded/scored.
    Nonfinite arithmetic/underflow raises ValueError rather than adding floors.
    """
    x = _probability(p_x, 1, "p_x")
    y = _probability(p_y, 1, "p_y")
    cost = _array(cost_yx, 2, "cost_yx")
    if cost.shape != (len(y), len(x)):
        raise ValueError("cost must have shape [len(p_y), len(p_x)]")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be a finite positive scalar")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive scalar")

    # Q0 is separable and is absorbed into the two scaling factors, without
    # changing the fixed-marginal optimization problem.
    log_kernel = -cost
    log_x, log_y = np.log(x), np.log(y)
    log_v = np.zeros(len(x), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for iteration in range(1, int(max_iterations) + 1):
                log_u = log_y - _logsumexp(log_kernel + log_v[None, :], 1)
                log_v = log_x - _logsumexp(log_kernel + log_u[:, None], 0)
                joint = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
                if not np.isfinite(joint).all() or np.any(joint <= 0):
                    raise ValueError("coupling lost finite positive support")
                row_error = float(np.max(np.abs(joint.sum(axis=1) - y)))
                column_error = float(np.max(np.abs(joint.sum(axis=0) - x)))
                residual = max(row_error, column_error)
                if residual <= tolerance:
                    return CouplingResult(joint, iteration, row_error, column_error, residual, True)
    except FloatingPointError as exc:
        raise ValueError("coupling arithmetic was not finite") from exc
    return CouplingResult(None, int(max_iterations), row_error, column_error, residual, False)


def joint_mean(joint):
    """Global [x,y] mean at bin centers; fixed marginals fix this quantity."""
    values = _probability(joint, 2, "joint")
    x = np.arange(values.shape[1], dtype=np.float64) + 0.5
    y = np.arange(values.shape[0], dtype=np.float64) + 0.5
    return np.array([values.sum(axis=0) @ x, values.sum(axis=1) @ y])


def mode_local_mean(joint, *, radius=20):
    """Joint mode, then mean in its inclusive rectangular +/-radius window.

    Exact mode ties use (x,y) lexicographic order, not array [y,x] order. There
    is no tolerance-based tie merging. Returned coordinates include +0.5 bin
    centers. For an independent joint this equals separate mode-local means.
    """
    values = _probability(joint, 2, "joint")
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    mode_y, mode_x = np.nonzero(values == values.max())
    chosen = np.lexsort((mode_y, mode_x))[0]
    x0, y0 = int(mode_x[chosen]), int(mode_y[chosen])
    left, right = max(0, x0 - radius), min(values.shape[1], x0 + radius + 1)
    top, bottom = max(0, y0 - radius), min(values.shape[0], y0 + radius + 1)
    window = values[top:bottom, left:right]
    local = window / window.sum()
    x = np.arange(left, right, dtype=np.float64) + 0.5
    y = np.arange(top, bottom, dtype=np.float64) + 0.5
    return np.array([local.sum(axis=0) @ x, local.sum(axis=1) @ y])
