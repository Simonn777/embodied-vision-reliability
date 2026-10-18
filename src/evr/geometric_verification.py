"""Original small NumPy geometry/scoring helpers; no GT or tail integration."""
from dataclasses import dataclass
import math

import numpy as np

CELL_SIDE = 8
COORDINATE_EXTENT = 512.0
HORIZON_RELATIVE_TOLERANCE = 1e-12


def _finite_real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(name + " must contain real numeric values (not bool/object/complex)")
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = array.astype(np.float64, copy=False)
    except FloatingPointError as exc:
        raise ValueError(name + " must be representable as finite float64") from exc
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _xy(value, name="xy"):
    xy = _finite_real(value, name)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(name + " must have shape (N, 2), in x,y order")
    return xy


def _tau(value):
    array = _finite_real(value, "tau")
    if array.ndim != 0 or float(array) <= 0:
        raise ValueError("tau must be a positive finite scalar")
    return float(array)


def project(H, xy):
    """Project x,y points; explicitly reject a sampled projective horizon.

    H is normalized by its largest absolute entry to remove arbitrary global
    scale. A denominator w is rejected when |w| <= 1e-12 * s, where
    s=|h20*x|+|h21*y|+|h22|. This is a relative cancellation check, including
    s=w=0. No determinant/rank/proposal-validity criterion is imposed here.
    """
    h = _finite_real(H, "H")
    points = _xy(xy)
    if h.shape != (3, 3):
        raise ValueError("H must have shape (3, 3)")
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        raise ValueError("H must not be all zero")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            h = h / scale
            homogeneous = points @ h[:, :2].T + h[:, 2]
            w = homogeneous[:, 2]
            denominator_scale = (np.abs(points) @ np.abs(h[2, :2]) + abs(h[2, 2]))
            if np.any(np.abs(w) <= HORIZON_RELATIVE_TOLERANCE * denominator_scale):
                raise ValueError("projection reaches or approaches a sampled projective horizon")
            projected = homogeneous[:, :2] / w[:, None]
    except FloatingPointError as exc:
        raise OverflowError("nonfinite float64 arithmetic during projection") from exc
    if not np.isfinite(projected).all():
        raise OverflowError("nonfinite projected coordinates")
    return projected


def matched_residuals(projected_x, y):
    """Euclidean pixel residual for each existing matched pair; no GT."""
    x, y = _xy(projected_x, "projected_x"), _xy(y, "y")
    if x.shape != y.shape:
        raise ValueError("matched arrays must have equal shape")
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference = x - y
            residuals = np.hypot(difference[:, 0], difference[:, 1])
    except FloatingPointError as exc:
        raise OverflowError("nonfinite matched-residual arithmetic") from exc
    if not np.isfinite(residuals).all():
        raise OverflowError("nonfinite matched residual")
    return residuals


def pairwise_radius_A(projected_x, y, tau):
    """A[i,j] = distance(projected_x[i], y[j]) <= tau (inclusive boundary).

    Allows rectangular input for arithmetic use; hypothesis scoring requires
    the square existing-match case. Materializes an N-by-M Boolean matrix.
    """
    x, y = _xy(projected_x, "projected_x"), _xy(y, "y")
    tau = _tau(tau)
    try:
        with np.errstate(over="raise", invalid="raise"):
            dx = x[:, None, 0] - y[None, :, 0]
            dy = x[:, None, 1] - y[None, :, 1]
            distances = np.hypot(dx, dy)
    except FloatingPointError as exc:
        raise OverflowError("nonfinite pairwise-distance arithmetic") from exc
    if not np.isfinite(distances).all():
        raise OverflowError("nonfinite pairwise distance")
    return distances <= tau


def marginal_cell_counts(xy):
    """Counts at each point in a fixed 8x8 grid on the original coordinates.

    For each axis: floor(coordinate / 64), then clip to [0, 7]. Exactly 512
    belongs to cell 7; finite out-of-range coordinates are clipped too. Every
    point occurrence counts, with no location deduplication. H is not an input.
    """
    points = _xy(xy)
    cells = np.clip(np.floor(points / (COORDINATE_EXTENT / CELL_SIDE)), 0, CELL_SIDE - 1)
    cells = cells.astype(np.int64)
    ids = cells[:, 1] * CELL_SIDE + cells[:, 0]
    counts = np.bincount(ids, minlength=CELL_SIDE * CELL_SIDE)
    return counts[ids]


@dataclass(frozen=True)
class GeometryScores:
    compatibility: np.ndarray
    matched_residuals: np.ndarray
    source_cell_counts: np.ndarray
    target_cell_counts: np.ndarray
    raw_count: int
    msac: float
    cell_weighted: float
    null_mean: float
    excess: float
    ratio: float


def score_hypothesis(H, x, y, tau):
    """Five fixed scoring primitives. No selector, tie-breaking, or GT access.

    raw_count=N; msac=sum(max(0,1-r_i**2/tau**2));
    cell_weighted=sum(I_i/sqrt(cs_i*ct_i)); null_mean=mu=sum(A)/n;
    excess=N-mu; ratio=N/mu if mu>0, otherwise 0.
    This cell score is neither GMS nor MAGSAC.
    """
    x, y = _xy(x, "x"), _xy(y, "y")
    if x.shape != y.shape or x.shape[0] == 0:
        raise ValueError("verification pairs must be equally sized and nonempty")
    tau = _tau(tau)
    projected = project(H, x)
    residuals = matched_residuals(projected, y)
    compatibility = pairwise_radius_A(projected, y, tau)
    inliers = residuals <= tau
    raw_count = int(np.count_nonzero(inliers))
    # Outliers contribute exactly zero; restricting division to inliers avoids
    # needless overflow of huge residual / tiny tau for a zero contribution.
    normalized = residuals[inliers] / tau
    msac = float(np.maximum(0.0, 1.0 - np.square(normalized)).sum())
    source_counts = marginal_cell_counts(x)
    target_counts = marginal_cell_counts(y)
    denominators = np.sqrt(source_counts.astype(np.float64) * target_counts.astype(np.float64))
    cell_weighted = float(np.sum(1.0 / denominators[inliers]))
    mu = float(np.count_nonzero(compatibility)) / len(x)
    excess = raw_count - mu
    ratio = raw_count / mu if mu > 0 else 0.0
    if not all(math.isfinite(v) for v in (msac, cell_weighted, mu, excess, ratio)):
        raise OverflowError("nonfinite derived score")
    return GeometryScores(compatibility, residuals, source_counts, target_counts,
                          raw_count, msac, cell_weighted, mu, excess, ratio)
