"""Independent fixed-grid GT evaluation; never imported by geometry scoring."""
import numpy as np

from .geometric_verification import project


def evaluation_grid():
    """81 fixed x,y locations: Cartesian product of 0,64,...,512.

    This is the continuous [0,512]^2 domain, not pixel centers 0,...,511.
    The grid is independent of matches, scores, candidate H, and GT H.
    """
    coordinates = np.linspace(0.0, 512.0, 9, dtype=np.float64)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def evaluate_transfer(H_est, H_gt):
    """Mean Euclidean distance between the two projected fixed grids, in pixels.

    This is a one-way homography-transfer metric, not symmetric transfer error
    or correspondence error. Invalid/horizon projections are explicitly raised.
    """
    grid = evaluation_grid()
    estimate = project(H_est, grid)
    truth = project(H_gt, grid)
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference = estimate - truth
            distances = np.hypot(difference[:, 0], difference[:, 1])
            result = float(distances.mean())
    except FloatingPointError as exc:
        raise OverflowError("nonfinite fixed-grid transfer arithmetic") from exc
    if not np.isfinite(result):
        raise OverflowError("nonfinite fixed-grid transfer error")
    return result
