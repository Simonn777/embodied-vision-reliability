"""Exploratory signed-moment/Cantelli addition; frozen 0012 code is unchanged.

For a uniformly chosen member of a cell, let R be its signed algebraic residual,
m=abs(E[R]), v=Var(R), and D bound every Sampson denominator from above. For
0<a<m, Cantelli gives P(abs(R)>=a) >= (m-a)^2/(v+(m-a)^2). Consequently

  sum min(R_i^2/d_i,t) >= n * min(a^2/D,t) * (m-a)^2/(v+(m-a)^2).

This is a deterministic finite-cell aggregate argument, not an independence or
sampling-confidence claim. It remains valid when d_i=0 costs t. No efficacy,
speed, novelty, or formal floating-point certification is asserted.
"""

from __future__ import annotations

import math
import numpy as np

from score_bounds import (
    Cell, Interval, _affine_square_range, _cap_ceiling, _enclose, _matrix,
    _threshold, cell_bounds,
)


def cantelli_lower_bound(cell: Cell, E, tau2: float) -> float:
    """Conservative numerical lower bound; failed arithmetic returns zero.

    A_i[8]=1, so M[:,8] already contains sum_i A_i. No new stored moments are
    needed. All square/difference estimates are enclosed *before* clipping.
    """
    E, tau2 = _matrix(E), _threshold(tau2)
    n = cell.n
    if n == 0 or not np.any(E):
        return 0.0
    maximum = n * tau2
    if not math.isfinite(maximum):
        return 0.0
    e, ae = E.ravel(), np.abs(E.ravel())
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean = float(e @ cell.moment[:, 8]) / n
        mean_scale = float(ae @ cell.absolute_moment[:, 8]) / n
        qsum = float(e @ cell.moment @ e)
        qscale = float(ae @ cell.absolute_moment @ ae)
    if not all(math.isfinite(v) for v in (mean, mean_scale, qsum, qscale)):
        return 0.0
    mean_lo, mean_hi = _enclose(mean, mean, mean_scale, n + 32)
    qsum_lo, qsum_hi = _enclose(qsum, qsum, qscale, n + 32)
    if not all(math.isfinite(v) for v in (mean_lo, mean_hi, qsum_lo, qsum_hi)):
        return 0.0
    if mean_lo <= 0 <= mean_hi or qsum_hi < 0:
        return 0.0
    m_lower = float(np.nextafter(min(abs(mean_lo), abs(mean_hi)), 0.0))
    if not math.isfinite(m_lower) or m_lower <= 0:
        return 0.0

    # v = Q/n - mean^2: use upper Q and LOWER squared magnitude, including
    # subtraction cancellation. A negative/NaN raw result is never reset to 0
    # and treated as proof of a deterministic residual.
    mean_square = m_lower * m_lower
    if not math.isfinite(mean_square):
        return 0.0
    mean_square_lo, _ = _enclose(mean_square, mean_square, mean_square)
    second_hi = float(np.nextafter(qsum_hi / n, np.inf))
    if not math.isfinite(mean_square_lo) or not math.isfinite(second_hi):
        return 0.0
    mean_square_lo = max(0.0, mean_square_lo)
    variance_raw = second_hi - mean_square_lo
    variance_scale = abs(second_hi) + mean_square_lo
    if not math.isfinite(variance_raw) or not math.isfinite(variance_scale):
        return 0.0
    _, variance_hi = _enclose(variance_raw, variance_raw, variance_scale, n + 32)
    if not math.isfinite(variance_hi) or variance_hi < 0:
        return 0.0

    squares = [_affine_square_range(row, cell.x_center, cell.x_halfwidth)
               for row in E[:2]]
    squares += [_affine_square_range(row, cell.y_center, cell.y_halfwidth)
                for row in E.T[:2]]
    if not all(math.isfinite(v) for pair in squares for v in pair):
        return 0.0
    try:
        denominator_sum = math.fsum(pair[1] for pair in squares)
    except OverflowError:
        return 0.0
    _, denominator_hi = _enclose(denominator_sum, denominator_sum, denominator_sum)
    if not math.isfinite(denominator_hi) or denominator_hi <= 0:
        return 0.0  # the existing bound handles identically zero denominators

    # Any threshold below m_lower is valid; a fixed threshold avoids a cubic
    # solver, parameter tuning, and optimization-rounding dependencies. Taking
    # the min is safe even if the cap product overflows (the other term is finite).
    cap_threshold = math.sqrt(tau2) * math.sqrt(denominator_hi)
    a = min(m_lower / 2.0, cap_threshold)
    if not math.isfinite(a) or not 0 < a < m_lower:
        return 0.0
    delta = m_lower - a
    # hypot avoids overflow in variance_hi + delta**2.
    radius = math.hypot(math.sqrt(variance_hi), delta)
    if not math.isfinite(radius) or radius <= 0:
        return 0.0
    probability = (delta / radius) ** 2
    probability_lo, _ = _enclose(probability, probability, probability)
    ratio = a / math.sqrt(denominator_hi)
    # Avoid squaring a large ratio when its capped value is already known.
    cost = tau2 if ratio >= math.sqrt(tau2) else ratio * ratio
    cost_lo, _ = _enclose(cost, cost, cost)
    if not math.isfinite(probability_lo) or not math.isfinite(cost_lo):
        return 0.0
    probability_lo, cost_lo = max(0.0, probability_lo), max(0.0, cost_lo)
    raw = n * probability_lo * cost_lo
    if not math.isfinite(raw):
        return 0.0
    lower, _ = _enclose(raw, raw, maximum, n + 32)
    return max(0.0, lower) if math.isfinite(lower) else 0.0


def bound(cell: Cell, E, tau2: float) -> Interval:
    """Intersect the new lower bound with the frozen moment/secant enclosure."""
    baseline = cell_bounds(cell, E, tau2)
    candidate = cantelli_lower_bound(cell, E, tau2)
    if not math.isfinite(candidate) or candidate > baseline.upper:
        return Interval(0.0, _cap_ceiling(cell.n, tau2))
    return Interval(max(baseline.lower, candidate), baseline.upper)
