"""Numerical prototype for hierarchical truncated Sampson-score bounds.

No model, dataset, or RANSAC loop is included. The score is the *sum* of
min((y.T @ E @ x)**2 / d, tau2), with homogeneous x/y, and
d = ||(E @ x)[:2]||**2 + ||(E.T @ y)[:2]||**2.
An exactly zero denominator contributes tau2, including the 0/0 case.

The real-arithmetic inequalities are accompanied by conservative numerical
padding and exact fallback. This implementation is NOT a floating-point proof
or a claim of speedup. E is rescaled by its largest absolute entry; this leaves
the mathematical score invariant and avoids trivial scale overflow/underflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np


EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).tiny


def _points(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be a finite N x 2 array")
    return value


def _matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("E must be a finite 3 x 3 matrix")
    scale = float(np.max(np.abs(value)))
    return value.copy() if scale == 0 else value / scale


def _threshold(tau2: float) -> float:
    tau2 = float(tau2)
    if not math.isfinite(tau2) or tau2 <= 0:
        raise ValueError("tau2 must be finite and strictly positive")
    return tau2


def _pad(scale: float, operations: int = 32) -> float:
    return 64.0 * EPS * (operations + 1) * max(abs(float(scale)), TINY)


def _enclose(lo: float, hi: float, scale: float, operations: int = 32):
    # Do not let a later max(0, NaN) turn failed arithmetic into a zero bound.
    # Callers either propagate this unbounded interval or use the full cap.
    if not all(math.isfinite(v) for v in (lo, hi, scale)):
        return -math.inf, math.inf
    margin = _pad(scale, operations)
    if not math.isfinite(margin):
        return -math.inf, math.inf
    return (float(np.nextafter(lo - margin, -np.inf)),
            float(np.nextafter(hi + margin, np.inf)))


def _cap_ceiling(n: int, tau2: float) -> float:
    # The mathematical cap is n*tau2; a reference reduction of n copies of
    # tau2 can round slightly above that product. Enclose both, not just n*tau2.
    return _enclose(n * tau2, n * tau2, n * tau2, n + 32)[1]


def exact_msac(E, xy1, xy2, tau2: float) -> float:
    """Reference full residual evaluation; 'exact' means no score approximation."""
    x, y = _points(xy1), _points(xy2)
    if len(x) != len(y):
        raise ValueError("point arrays must have equal lengths")
    E, tau2 = _matrix(E), _threshold(tau2)
    xh = np.column_stack((x, np.ones(len(x))))
    yh = np.column_stack((y, np.ones(len(y))))
    Ex, Ety = xh @ E.T, yh @ E
    numerator = np.sum(yh * Ex, axis=1) ** 2
    denominator = np.sum(Ex[:, :2] ** 2 + Ety[:, :2] ** 2, axis=1)
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise OverflowError("input magnitude exceeds reference arithmetic range")
    cost = np.full(len(x), tau2)
    np.divide(numerator, denominator, out=cost, where=denominator > 0)
    return float(np.sum(np.minimum(cost, tau2), dtype=np.float64))


@dataclass(frozen=True)
class Cell:
    indices: np.ndarray
    rows: np.ndarray  # A_i = vec(y_i x_i^T), row-major
    moment: np.ndarray  # A.T @ A
    absolute_moment: np.ndarray  # numerical cancellation guard only
    row_min: np.ndarray
    row_max: np.ndarray
    x_center: np.ndarray
    x_halfwidth: np.ndarray
    y_center: np.ndarray
    y_halfwidth: np.ndarray
    children: tuple["Cell", ...] = ()

    @property
    def n(self):
        return len(self.indices)


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    @property
    def width(self):
        return max(0.0, self.upper - self.lower)


@dataclass(frozen=True)
class BoundResult:
    interval: Interval
    refinements: int
    node_evaluations: int
    exact_residuals: int


@dataclass(frozen=True)
class RankingResult:
    order: tuple[int, ...]
    bounds: tuple[BoundResult, ...]
    exact_scores: dict[int, float]
    fallback_models: int
    exact_residuals: int


@dataclass(frozen=True)
class SelectionResult:
    index: int
    score: float
    interval: Interval
    fallback_models: int
    pruned_models: int
    exact_residuals: int
    node_evaluations: int
    refinements: int


def _affine_square_range(coeff, center, halfwidth):
    value = float(coeff[:2] @ center + coeff[2])
    radius = float(np.abs(coeff[:2]) @ halfwidth)
    scale = float(np.abs(coeff[:2]) @ np.abs(center) + abs(coeff[2]) + radius)
    if not all(math.isfinite(v) for v in (value, radius, scale)):
        return 0.0, math.inf
    lo, hi = _enclose(value - radius, value + radius, scale)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return 0.0, math.inf
    sqlo = 0.0 if lo <= 0 <= hi else min(lo * lo, hi * hi)
    sqhi = max(lo * lo, hi * hi)
    slo, shi = _enclose(sqlo, sqhi, sqhi)
    if not math.isfinite(slo) or not math.isfinite(shi):
        return 0.0, math.inf
    return max(0.0, slo), max(0.0, shi)


def cell_bounds(cell: Cell, E, tau2: float, *, secant: bool = True,
                mode: str = "moments") -> Interval:
    """Moment/row-box bounds; optional mixed-cluster concavity secant bound."""
    E, tau2 = _matrix(E), _threshold(tau2)
    if mode not in ("moments", "culling"):
        raise ValueError("mode must be 'moments' or 'culling'")
    n, maximum = cell.n, cell.n * tau2
    if not math.isfinite(maximum):
        raise OverflowError("N * tau2 exceeds float64 range")
    if not n:
        return Interval(0.0, 0.0)
    if not np.any(E):
        lo, hi = _enclose(maximum, maximum, maximum, n + 32)
        return Interval(max(0.0, lo), hi)
    e = E.ravel()
    if mode == "moments":
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            qsum = float(e @ cell.moment @ e)
            qscale = float(np.abs(e) @ cell.absolute_moment @ np.abs(e))
        # M itself can be finite while these reductions overflow/cancel to NaN.
        # Inspect RAW values before _enclose or any nonnegative clipping.
        if not math.isfinite(qsum) or not math.isfinite(qscale):
            return Interval(0.0, _cap_ceiling(n, tau2))
        qsum_lo, qsum_hi = _enclose(qsum, qsum, qscale, n + 32)
        if not math.isfinite(qsum_lo) or not math.isfinite(qsum_hi):
            return Interval(0.0, _cap_ceiling(n, tau2))
        qsum_lo, qsum_hi = max(0.0, qsum_lo), max(0.0, qsum_hi)
    else:
        qsum_lo = qsum_hi = 0.0

    # Each coefficient independently selects the corresponding row-box endpoint.
    terms_lo = np.minimum(e * cell.row_min, e * cell.row_max)
    terms_hi = np.maximum(e * cell.row_min, e * cell.row_max)
    nscale = float(np.sum(np.maximum(np.abs(terms_lo), np.abs(terms_hi))))
    nlo, nhi = _enclose(float(np.sum(terms_lo)), float(np.sum(terms_hi)), nscale)
    if not math.isfinite(nlo) or not math.isfinite(nhi):
        return Interval(0.0, _cap_ceiling(n, tau2))
    qlo = 0.0 if nlo <= 0 <= nhi else min(nlo * nlo, nhi * nhi)
    qhi = max(nlo * nlo, nhi * nhi)
    qlo, qhi = _enclose(qlo, qhi, qhi)
    if not math.isfinite(qlo) or not math.isfinite(qhi):
        return Interval(0.0, _cap_ceiling(n, tau2))
    qlo, qhi = max(0.0, qlo), max(0.0, qhi)

    squares = [_affine_square_range(row, cell.x_center, cell.x_halfwidth)
               for row in E[:2]]
    squares += [_affine_square_range(row, cell.y_center, cell.y_halfwidth)
                for row in E.T[:2]]
    if not all(math.isfinite(v) for square in squares for v in square):
        return Interval(0.0, _cap_ceiling(n, tau2))
    dlo, dhi = _enclose(math.fsum(s[0] for s in squares),
                        math.fsum(s[1] for s in squares),
                        math.fsum(s[1] for s in squares))
    if not math.isfinite(dlo) or not math.isfinite(dhi):
        return Interval(0.0, _cap_ceiling(n, tau2))
    dlo, dhi = max(0.0, dlo), max(0.0, dhi)
    values = [qsum_lo, qsum_hi, qlo, qhi, dlo, dhi]
    if not all(math.isfinite(v) for v in values):
        return Interval(0.0, _cap_ceiling(n, tau2))
    if dhi == 0:
        lo, hi = _enclose(maximum, maximum, maximum, n + 32)
        return Interval(max(0.0, lo), hi)
    if mode == "culling":
        # Spatial-only control: certify capped outliers; use no moment or
        # nonconstant inlier aggregate. Ambiguous cells must be refined/scored.
        threshold_hi = tau2 * dhi
        threshold_hi += _pad(threshold_hi)
        if qlo >= threshold_hi:
            lo, hi = _enclose(maximum, maximum, maximum, n + 32)
            return Interval(max(0.0, lo), hi)
        return Interval(0.0, _cap_ceiling(n, tau2))

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # Raw residual interval [l,u] and bounds on its (untruncated) sum.
        l = qlo / dhi
        u = qhi / dlo if dlo > 0 else math.inf
        sum_lo = max(qsum_lo / dhi, n * l)
        sum_hi = min(qsum_hi / dlo, n * u) if dlo > 0 else math.inf
        divisor = max(dhi, qhi / tau2)
        lower = max(qsum_lo / divisor if divisor > 0 else 0.0,
                    n * min(l, tau2))
        upper = min(sum_hi, maximum, n * min(u, tau2))
        if secant:
            if l >= tau2:
                lower = maximum
            elif u <= tau2:
                lower = max(lower, sum_lo)
            elif math.isfinite(u) and u > l:
                # min(z,tau2) is concave: its secant is a lower envelope.
                alpha = (tau2 - l) / (u - l)
                secant_lower = alpha * sum_lo + n * l * (1.0 - alpha)
                if not math.isfinite(secant_lower):
                    return Interval(0.0, _cap_ceiling(n, tau2))
                lower = max(lower, secant_lower)

    lower, upper = _enclose(lower, upper, maximum, n + 32)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return Interval(0.0, _cap_ceiling(n, tau2))
    lower, upper = max(0.0, lower), min(_cap_ceiling(n, tau2), upper)
    return Interval(lower, upper)


class MatchTree:
    """Median hierarchy in (x1,y1,x2,y2); preprocessing is explicit and reusable."""

    def __init__(self, xy1, xy2, leaf_size: int = 16):
        self.xy1, self.xy2 = _points(xy1).copy(), _points(xy2).copy()
        if len(self.xy1) != len(self.xy2):
            raise ValueError("point arrays must have equal lengths")
        if not isinstance(leaf_size, int) or leaf_size < 1:
            raise ValueError("leaf_size must be a positive integer")
        self.leaf_size = leaf_size
        xh = np.column_stack((self.xy1, np.ones(len(self.xy1))))
        yh = np.column_stack((self.xy2, np.ones(len(self.xy2))))
        rows = (yh[:, :, None] * xh[:, None, :]).reshape(-1, 9)
        self.root = self._build(np.arange(len(rows)), rows)

    def _build(self, indices, all_rows):
        rows = all_rows[indices]
        coords = np.column_stack((self.xy1[indices], self.xy2[indices]))
        if len(indices):
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            row_min, row_max = rows.min(axis=0), rows.max(axis=0)
        else:
            lo = hi = np.zeros(4)
            row_min = row_max = np.zeros(9)
        center, halfwidth = lo / 2 + hi / 2, (hi - lo) / 2
        # Include center/halfwidth arithmetic error in the affine intervals.
        halfwidth += 64 * EPS * np.maximum(np.maximum(np.abs(lo), np.abs(hi)), TINY)
        # Ignore stale BLAS floating flags, but inspect both actual outputs.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            moment, absolute = rows.T @ rows, np.abs(rows).T @ np.abs(rows)
        if not np.isfinite(moment).all() or not np.isfinite(absolute).all():
            raise OverflowError("input magnitude exceeds moment arithmetic range")
        children = ()
        if len(indices) > self.leaf_size:
            axis = int(np.argmax(hi - lo))
            order = np.argsort(coords[:, axis], kind="stable")
            middle = len(order) // 2
            children = (self._build(indices[order[:middle]], all_rows),
                        self._build(indices[order[middle:]], all_rows))
        return Cell(indices, rows, moment, absolute, row_min, row_max,
                    center[:2], halfwidth[:2], center[2:], halfwidth[2:], children)

    def partition(self, target_cells: int = 1) -> tuple[Cell, ...]:
        """Deterministic common starting partition; split largest cells first."""
        if not isinstance(target_cells, int) or target_cells < 1:
            raise ValueError("target_cells must be a positive integer")
        queue, done, counter = [(-self.root.n, 0, self.root)], [], 0
        while queue and len(queue) + len(done) < target_cells:
            item = heapq.heappop(queue)
            cell = item[2]
            if not cell.children:
                done.append(item)
                continue
            for child in cell.children:
                counter += 1
                heapq.heappush(queue, (-child.n, counter, child))
        return tuple(item[2] for item in sorted(queue + done, key=lambda v: v[1]))

    def score_bounds(self, E, tau2: float, *, max_refinements: int = 0,
                     exact_leaves: bool = True, secant: bool = True,
                     initial_cells: int = 1, mode: str = "moments") -> BoundResult:
        E, tau2 = _matrix(E), _threshold(tau2)
        if not isinstance(max_refinements, int) or max_refinements < 0:
            raise ValueError("max_refinements must be a nonnegative integer")
        start = self.partition(initial_cells)
        parts = {i: cell_bounds(cell, E, tau2, secant=secant, mode=mode)
                 for i, cell in enumerate(start)}
        initial = Interval(math.fsum(p.lower for p in parts.values()),
                           math.fsum(p.upper for p in parts.values()))
        initial_lo, initial_hi = _enclose(initial.lower, initial.upper,
                                        len(self.xy1) * tau2, len(start) + 32)
        initial = Interval(max(0.0, initial_lo), min(_cap_ceiling(len(self.xy1), tau2), initial_hi))
        queue = [(-parts[i].width, i, cell) for i, cell in enumerate(start)]
        heapq.heapify(queue)
        counter, refinements, nodes, residuals = len(start) - 1, 0, len(start), 0
        while queue and refinements < max_refinements:
            _, key, cell = heapq.heappop(queue)
            if parts[key].width <= 4 * _pad(cell.n * tau2, cell.n + 32):
                continue
            if not cell.children and not exact_leaves:
                continue
            if cell.children:
                del parts[key]
                for child in cell.children:
                    counter += 1
                    bound = cell_bounds(child, E, tau2, secant=secant, mode=mode)
                    parts[counter] = bound
                    heapq.heappush(queue, (-bound.width, counter, child))
                    nodes += 1
            else:
                score = exact_msac(E, self.xy1[cell.indices], self.xy2[cell.indices], tau2)
                lo, hi = _enclose(score, score, cell.n * tau2, cell.n + 32)
                parts[key] = Interval(max(0.0, lo), min(_cap_ceiling(cell.n, tau2), hi))
                residuals += cell.n
            refinements += 1
        lo, hi = math.fsum(p.lower for p in parts.values()), math.fsum(p.upper for p in parts.values())
        lo, hi = _enclose(lo, hi, len(self.xy1) * tau2, len(parts) + 32)
        # Intersect with root bounds so refinement cannot worsen the enclosure.
        lo, hi = max(initial.lower, lo), min(initial.upper, hi)
        if lo > hi:
            lo, hi = initial.lower, initial.upper
        return BoundResult(Interval(lo, hi), refinements, nodes, residuals)

    def best_hypothesis(self, matrices: Sequence[np.ndarray], tau2: float, *,
                        max_refinements: int = 0, initial_cells: int = 1,
                        mode: str = "moments", secant: bool = True) -> SelectionResult:
        """Exact-score incumbent, strict lower-bound pruning, exact overlap fallback."""
        tau2 = _threshold(tau2)
        matrices = tuple(_matrix(E) for E in matrices)
        if not matrices:
            raise ValueError("at least one hypothesis is required")
        coarse = [self.score_bounds(E, tau2, initial_cells=initial_cells,
                                    mode=mode, secant=secant) for E in matrices]
        incumbent = min(range(len(matrices)), key=lambda i: (coarse[i].interval.upper, i))
        best_score = exact_msac(matrices[incumbent], self.xy1, self.xy2, tau2)
        fallback, pruned, residuals = 1, 0, len(self.xy1)
        nodes, refinements = sum(b.node_evaluations for b in coarse), 0
        candidates = sorted((i for i in range(len(matrices)) if i != incumbent),
                            key=lambda i: (coarse[i].interval.lower, i))
        for i in candidates:
            bound = coarse[i]
            safe_upper = best_score + _pad(len(self.xy1) * tau2, len(self.xy1) + 32)
            if bound.interval.lower > safe_upper:
                pruned += 1
                continue
            if max_refinements:
                bound = self.score_bounds(matrices[i], tau2, initial_cells=initial_cells,
                                          max_refinements=max_refinements,
                                          mode=mode, secant=secant)
                nodes += bound.node_evaluations
                residuals += bound.exact_residuals
                refinements += bound.refinements
            if bound.interval.lower > safe_upper:
                pruned += 1
                continue
            value = exact_msac(matrices[i], self.xy1, self.xy2, tau2)
            fallback += 1
            residuals += len(self.xy1)
            if (value, i) < (best_score, incumbent):
                best_score, incumbent = value, i
        lo, hi = _enclose(best_score, best_score, len(self.xy1) * tau2, len(self.xy1) + 32)
        return SelectionResult(incumbent, best_score,
                               Interval(max(0.0, lo), min(_cap_ceiling(len(self.xy1), tau2), hi)),
                               fallback, pruned, residuals, nodes, refinements)

    def rank_hypotheses(self, matrices: Sequence[np.ndarray], tau2: float,
                        *, max_refinements: int = 0, secant: bool = True) -> RankingResult:
        """Order disjoint intervals directly, fully score each overlap component.

        Deterministic index tie breaking agrees with sorting (reference score,
        original index). The exact path is intentionally explicit in counters.
        """
        tau2 = _threshold(tau2)
        matrices = tuple(_matrix(E) for E in matrices)
        bounds = tuple(self.score_bounds(E, tau2, max_refinements=max_refinements,
                                         secant=secant) for E in matrices)
        remaining = sorted(range(len(matrices)), key=lambda i: (bounds[i].interval.lower, i))
        order, exact = [], {}
        k = 0
        while k < len(remaining):
            group = [remaining[k]]
            right = bounds[remaining[k]].interval.upper
            k += 1
            while k < len(remaining) and bounds[remaining[k]].interval.lower <= right:
                group.append(remaining[k])
                right = max(right, bounds[remaining[k]].interval.upper)
                k += 1
            if len(group) > 1:
                for i in group:
                    exact[i] = exact_msac(matrices[i], self.xy1, self.xy2, tau2)
                group.sort(key=lambda i: (exact[i], i))
            order.extend(group)
        residuals = sum(b.exact_residuals for b in bounds) + len(exact) * len(self.xy1)
        return RankingResult(tuple(order), bounds, exact, len(exact), residuals)
