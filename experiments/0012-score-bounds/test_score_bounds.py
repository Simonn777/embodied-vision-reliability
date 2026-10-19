"""Synthetic numerical checks, not scientific efficacy experiments."""

import numpy as np
import pytest

from score_bounds import MatchTree, cell_bounds, exact_msac


def assert_contains(interval, value):
    # No test-side tolerance: the implementation's enclosure must contain the
    # independently reduced full reference score as a floating-point value.
    assert interval.lower <= value <= interval.upper, (interval, value)


def all_cells(cell):
    yield cell
    for child in cell.children:
        yield from all_cells(child)


def test_random_node_and_adaptive_containment():
    rng = np.random.default_rng(12491)
    for scale in (0.01, 1.0, 512.0):
        x, y = rng.normal(size=(2, 97, 2)) * scale
        tree = MatchTree(x, y, leaf_size=8)
        for _ in range(8):
            E = rng.normal(size=(3, 3))
            tau2 = float(10 ** rng.uniform(-4, 3))
            value = exact_msac(E, x, y, tau2)
            for cell in all_cells(tree.root):
                local = exact_msac(E, x[cell.indices], y[cell.indices], tau2)
                for mode in ("moments", "culling"):
                    assert_contains(cell_bounds(cell, E, tau2, mode=mode), local)
            for initial_cells in (1, 7):
                for budget in (0, 5, 100):
                    result = tree.score_bounds(E, tau2, initial_cells=initial_cells,
                                               max_refinements=budget)
                    assert_contains(result.interval, value)


@pytest.mark.parametrize("n", [0, 1, 5, 65])
def test_empty_duplicate_zero_matrix_and_zero_denominator(n):
    x, y = np.zeros((n, 2)), np.zeros((n, 2))
    tree = MatchTree(x, y, leaf_size=4)
    for E in (np.zeros((3, 3)), np.diag([0.0, 0.0, 1.0]),
              np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0]])):
        # At x=y=0 each denominator is zero, including zero numerators.
        score = exact_msac(E, x, y, 0.1)
        assert score == float(np.sum(np.full(n, 0.1)))
        for mode in ("moments", "culling"):
            assert_contains(tree.score_bounds(E, 0.1, initial_cells=8,
                                             max_refinements=64, mode=mode).interval, score)


def test_threshold_boundary_and_extreme_matrix_scales():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(80, 2))
    y = x.copy()
    near = np.array([-1e-10, -1e-14, 0.0, 1e-14, 1e-10])
    y[:, 1] += np.sqrt(2) * (1 + np.resize(near, len(x)))
    E = np.array([[0.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    tree = MatchTree(x, y, leaf_size=8)
    reference = exact_msac(E, x, y, 1.0)
    for multiplier in (1.0, -1.0, 1e-250, -1e250):
        Em = E * multiplier
        assert exact_msac(Em, x, y, 1.0) == reference
        assert_contains(tree.score_bounds(Em, 1.0, max_refinements=100).interval, reference)


def test_large_coordinate_cancellation_enclosure():
    rng = np.random.default_rng(551)
    x = np.full((128, 2), 1e8) + rng.normal(size=(128, 2))
    y = x + rng.normal(scale=1e-4, size=x.shape)
    E = np.array([[0.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    tree = MatchTree(x, y, leaf_size=8)
    reference = exact_msac(E, x, y, 1e-6)
    # e^T M e loses low-order information here; the absolute product-sum guard
    # must widen the enclosure rather than pretending max(qsum, 0) is accurate.
    for cell in all_cells(tree.root):
        assert_contains(cell_bounds(cell, E, 1e-6),
                        exact_msac(E, x[cell.indices], y[cell.indices], 1e-6))
    assert_contains(tree.score_bounds(E, 1e-6, max_refinements=1000).interval, reference)


def test_finite_moments_with_overflowing_quadratic_query_use_full_cap():
    # Independent review regression: stored M is finite, but e^T M e becomes
    # NaN and its absolute-product error scale becomes +inf. max(0, NaN) used
    # to hide that failure and incorrectly return an almost-zero upper bound.
    a = 1e77
    b = np.nextafter(np.nextafter(a, np.inf), np.inf)
    x, y = np.array([[a, a]]), np.array([[a, b]])
    E = np.array([[1.0, 1.0, 0], [-1.0, -1.0, 0], [0, 0, 0]])
    tree = MatchTree(x, y)
    assert np.isfinite(tree.root.moment).all()
    reference = exact_msac(E, x, y, 1.0)
    assert reference == 1.0
    for secant in (False, True):
        interval = cell_bounds(tree.root, E, 1.0, secant=secant)
        assert interval.lower == 0.0
        assert_contains(interval, reference)
    assert_contains(tree.score_bounds(E, 1.0, max_refinements=2).interval, reference)
    winner = tree.best_hypothesis([E, np.zeros((3, 3))], 1.0, max_refinements=2)
    assert winner.index == 0  # exact-score tie must retain original order


def test_secant_strictly_improves_mixed_cell_lower_bound():
    x = np.zeros((4, 2))
    y = np.column_stack((np.zeros(4), np.sqrt(2 * np.array([0.5, 0.7, 1.1, 1.5]))))
    E = np.array([[0.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    tree = MatchTree(x, y)
    base = cell_bounds(tree.root, E, 1.0, secant=False)
    enhanced = cell_bounds(tree.root, E, 1.0, secant=True)
    assert enhanced.lower > base.lower + 0.1
    assert_contains(enhanced, exact_msac(E, x, y, 1.0))


def test_common_partition_and_full_ranking_with_ties():
    rng = np.random.default_rng(99)
    x, y = rng.normal(size=(2, 200, 2))
    tree = MatchTree(x, y, leaf_size=16)
    cells = tree.partition(9)
    assert len(cells) == 9
    assert sorted(np.concatenate([c.indices for c in cells]).tolist()) == list(range(200))
    models = list(rng.normal(size=(12, 3, 3)))
    models += [models[3].copy(), np.zeros((3, 3)), models[3].copy()]
    scores = [exact_msac(E, x, y, 0.17) for E in models]
    expected = tuple(sorted(range(len(models)), key=lambda i: (scores[i], i)))
    ranking = tree.rank_hypotheses(models, 0.17, max_refinements=7)
    assert ranking.order == expected
    for mode in ("moments", "culling"):
        selected = tree.best_hypothesis(models, 0.17, initial_cells=9,
                                        max_refinements=16, mode=mode)
        assert selected.index == expected[0]
        assert selected.score == scores[expected[0]]
        assert selected.fallback_models + selected.pruned_models == len(models)


def test_pruning_and_equal_score_fallback():
    x = np.column_stack((np.linspace(0, 1, 96), np.zeros(96)))
    y = x.copy()
    good = np.array([[0.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    bad = np.diag([0.0, 0.0, 1.0])
    tree = MatchTree(x, y, leaf_size=8)
    selected = tree.best_hypothesis([good, bad, good.copy()], 1.0, initial_cells=8)
    assert selected.index == 0
    assert selected.fallback_models == 2  # equality is not pruned
    assert selected.pruned_models == 1
    assert selected.score == 0.0


def test_nonfinite_and_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        MatchTree([[np.nan, 0]], [[0, 0]])
    with pytest.raises(ValueError):
        exact_msac(np.eye(3), [[0, 0]], [[0, 0]], 0)
    with pytest.raises(ValueError):
        cell_bounds(MatchTree([[0, 0]], [[0, 0]]).root, np.eye(3), 1, mode="unknown")
