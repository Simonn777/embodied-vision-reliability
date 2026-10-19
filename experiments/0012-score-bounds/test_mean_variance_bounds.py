"""Artificial-array checks only; no current development files are read."""

import numpy as np

from score_bounds import MatchTree, cell_bounds, exact_msac
from mean_variance_bounds import bound, cantelli_lower_bound


E_VERTICAL = np.array([[0.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def test_correlated_coordinates_defeat_row_box_but_not_signed_moments():
    x = np.array([[0.0, 0.0], [0.0, 100.0]])
    y = np.array([[0.0, 9.0], [0.0, 111.0]])
    cell = MatchTree(x, y).root
    baseline = cell_bounds(cell, E_VERTICAL, 1.0)
    enhanced = bound(cell, E_VERTICAL, 1.0)
    score = exact_msac(E_VERTICAL, x, y, 1.0)
    assert baseline.lower < 0.02
    assert enhanced.lower > 1.97
    assert enhanced.lower <= score <= enhanced.upper
    # First moment is already available from the homogeneous constant column.
    np.testing.assert_allclose(cell.moment[:, 8], cell.rows.sum(axis=0))


def test_symmetric_bimodal_signed_mean_is_vacuous_and_preserves_base():
    x = np.zeros((2, 2))
    y = np.array([[0.0, -10.0], [0.0, 10.0]])
    cell = MatchTree(x, y).root
    assert cantelli_lower_bound(cell, E_VERTICAL, 1.0) == 0.0
    assert bound(cell, E_VERTICAL, 1.0) == cell_bounds(cell, E_VERTICAL, 1.0)


def test_random_cells_and_near_constant_residual_containment():
    rng = np.random.default_rng(987321)
    for coordinate_scale in (1e-6, 1.0, 1000.0):
        for mode in ("independent", "correlated"):
            x, y = rng.normal(size=(2, 129, 2)) * coordinate_scale
            E = rng.normal(size=(3, 3))
            if mode == "correlated":
                y = x + np.array([0.0, 4.0]) + rng.normal(scale=1e-7, size=x.shape)
                E = E_VERTICAL
            tree = MatchTree(x, y, leaf_size=8)
            for cell in (tree.root, *tree.partition(11)):
                for tau2 in (1e-10, 1.0, 1000.0):
                    value = exact_msac(E, x[cell.indices], y[cell.indices], tau2)
                    enhanced = bound(cell, E, tau2)
                    assert enhanced.lower <= value <= enhanced.upper
                    assert enhanced.lower >= cell_bounds(cell, E, tau2).lower


def test_raw_overflow_and_zero_denominators_fall_back_safely():
    a = 1e77
    b = np.nextafter(np.nextafter(a, np.inf), np.inf)
    x, y = np.array([[a, a]]), np.array([[a, b]])
    E = np.array([[1.0, 1.0, 0], [-1.0, -1.0, 0], [0, 0, 0]])
    cell = MatchTree(x, y).root
    assert cantelli_lower_bound(cell, E, 1.0) == 0.0
    interval = bound(cell, E, 1.0)
    assert interval.lower <= exact_msac(E, x, y, 1.0) <= interval.upper
    zeros = np.zeros((5, 2))
    cell = MatchTree(zeros, zeros).root
    for E in (np.zeros((3, 3)), np.diag([0.0, 0.0, 1.0])):
        interval = bound(cell, E, 0.1)
        assert interval.lower <= exact_msac(E, zeros, zeros, 0.1) <= interval.upper


def test_cancellation_scale_sign_and_empty_cell():
    rng = np.random.default_rng(31)
    x = 1e8 + rng.normal(size=(40, 2))
    y = x + np.array([0.0, 4.0]) + rng.normal(scale=1e-7, size=x.shape)
    cell = MatchTree(x, y).root
    for scale in (1.0, -1.0, 1e-250, -1e250):
        E = scale * E_VERTICAL
        score = exact_msac(E, x, y, 1.0)
        interval = bound(cell, E, 1.0)
        assert interval.lower <= score <= interval.upper
    empty = np.zeros((0, 2))
    interval = bound(MatchTree(empty, empty).root, E_VERTICAL, 1.0)
    assert interval.lower == interval.upper == 0.0
