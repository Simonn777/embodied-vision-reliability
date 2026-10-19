"""Artificial-only arithmetic fixtures; no model, assets, or study seeds."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "marginal_coupling.py"
SPEC = importlib.util.spec_from_file_location("marginal_coupling_fixture", SOURCE)
coupling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coupling)


class ProbabilityTests(unittest.TestCase):
    def test_native_core_axes_are_y_then_x(self):
        logits = np.zeros(512)
        logits[7], logits[256 + 19] = 8, 10
        px, py = coupling.native_marginals(logits)
        self.assertEqual(int(px.argmax()), 19)
        self.assertEqual(int(py.argmax()), 7)
        self.assertEqual(coupling.independent_joint(px, py).shape, (256, 256))

    def test_multiplier_floor_and_shift_stability(self):
        np.testing.assert_allclose(coupling.probabilities_from_logits([0, 2]), [1 / (1 + np.e), np.e / (1 + np.e)], rtol=1e-15)
        np.testing.assert_array_equal(coupling.probabilities_from_logits([0, 2]), coupling.probabilities_from_logits([1000, 1002]))
        result = coupling.probabilities_from_logits([-1e308, 1e308])
        expected = np.array([1e-12, 1.0]) / (1 + 1e-12)
        np.testing.assert_array_equal(result, expected)

    def test_product_and_mixture_are_distinct(self):
        a = np.array([[0.1, 0.2], [0.3, 0.4]])
        b = np.array([[0.4, 0.3], [0.2, 0.1]])
        np.testing.assert_allclose(coupling.mixture_joint(a, b), np.full((2, 2), 0.25), atol=0, rtol=1e-15)
        np.testing.assert_allclose(coupling.poe_joint(a, b), [[.2, .3], [.3, .2]], atol=1e-15)
        np.testing.assert_allclose(coupling.normalize_joint(100 * a), a, atol=1e-15)

    def test_invalid_logits_and_probability_inputs(self):
        for logits in ([np.nan, 0], [np.inf, 0], [], [[0, 1]], [True, False], [1j, 0]):
            with self.subTest(logits=logits), self.assertRaises((TypeError, ValueError)):
                coupling.probabilities_from_logits(logits)
        with self.assertRaises(ValueError):
            coupling.native_marginals(np.zeros(510))
        for p in ([0, 1], [-1, 2], [.2, .2], [np.nan, .5], [np.inf, .5]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                coupling.independent_joint(p, [.5, .5])
        for weights in ([[0., 1.]], [[-1., 2.]], [[np.inf]], [[np.nan]]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                coupling.normalize_joint(weights)
        with self.assertRaises(ValueError):
            coupling.poe_joint(np.ones((1, 1)), np.full((2, 2), .25))


class CouplingTests(unittest.TestCase):
    def test_separable_axis_cost_cancels_for_nonuniform_rectangular_grid(self):
        px, py = np.array([.1, .2, .7]), np.array([.4, .6])
        cost = np.array([4., -3.])[:, None] + np.array([2., 7., -1.])[None, :]
        result = coupling.entropic_coupling(px, py, cost)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        np.testing.assert_allclose(result.joint, np.outer(py, px), atol=2e-15, rtol=0)

    def test_hand_three_by_three_recovers_higher_hit_mass_than_poe(self):
        # Exact analytical example has zeros; use the predeclared positive
        # floor for its projection inputs, not an implicit solver correction.
        truth = np.array([[.3, 0, 0], [0, .2, .2], [0, .2, .1]])
        px, py = truth.sum(axis=0), truth.sum(axis=1)
        pu = np.maximum(np.array([.3, 0, .2, .4, .1]), 1e-12)
        pv = np.maximum(np.array([0, .2, .6, .2, 0]), 1e-12)
        pu, pv = pu / pu.sum(), pv / pv.sum()
        yy, xx = np.indices((3, 3))
        kernel = pu[xx + yy] * pv[yy - xx + 2]
        original = coupling.independent_joint(px, py)
        poe = coupling.poe_joint(original, coupling.normalize_joint(kernel))
        result = coupling.entropic_coupling(px, py, -np.log(kernel), tolerance=1e-11)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.joint.sum(axis=0), px, atol=1e-11, rtol=0)
        np.testing.assert_allclose(result.joint.sum(axis=1), py, atol=1e-11, rtol=0)
        t = (6.4 - np.sqrt(35.2)) / 2
        expected = np.array([[.3, 0, 0], [0, t, .4 - t], [0, .4 - t, t - .1]])
        np.testing.assert_allclose(result.joint, expected, atol=2e-10, rtol=0)
        np.testing.assert_array_equal(coupling.mode_local_mean(poe, radius=0), [1.5, 1.5])
        np.testing.assert_array_equal(coupling.mode_local_mean(result.joint, radius=0), [.5, .5])
        self.assertGreater(truth[0, 0], truth[1, 1])

    def test_fixed_marginals_keep_global_mean_and_row_column_factors_cancel(self):
        px, py = np.array([.2, .5, .3]), np.array([.4, .6])
        cost = np.array([[0., 2., 1.], [3., .2, 4.]])
        first = coupling.entropic_coupling(px, py, cost, tolerance=1e-12)
        shifted = cost + np.array([5., -2.])[:, None] + np.array([1., -3., 7.])[None, :]
        second = coupling.entropic_coupling(px, py, shifted, tolerance=1e-12)
        self.assertTrue(first.converged and second.converged)
        np.testing.assert_allclose(first.joint, second.joint, atol=2e-12, rtol=0)
        np.testing.assert_allclose(coupling.joint_mean(first.joint), coupling.joint_mean(coupling.independent_joint(px, py)), atol=3e-12, rtol=0)

    def test_one_cycle_nonconvergence_is_explicit_and_not_decodable(self):
        result = coupling.entropic_coupling([.5, .5], [.9, .1], [[0., 8.], [8., 0.]], max_iterations=1, tolerance=1e-12)
        self.assertFalse(result.converged)
        self.assertIsNone(result.joint)
        self.assertEqual(result.iterations, 1)
        self.assertGreater(result.residual, 1e-12)
        self.assertEqual(result.residual, max(result.row_error, result.column_error))

    def test_input_arrays_are_not_mutated(self):
        px, py = np.array([.3, .7]), np.array([.6, .4])
        cost = np.array([[0., 1.], [2., 0.]])
        before = [a.copy() for a in (px, py, cost)]
        self.assertTrue(coupling.entropic_coupling(px, py, cost).converged)
        for current, previous in zip((px, py, cost), before):
            np.testing.assert_array_equal(current, previous)

    def test_invalid_solver_inputs(self):
        for kw in ({"max_iterations": 0}, {"max_iterations": True}, {"max_iterations": 1.5}, {"tolerance": 0}, {"tolerance": np.nan}, {"tolerance": True}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                coupling.entropic_coupling([.5, .5], [.5, .5], np.zeros((2, 2)), **kw)
        for cost in ([[np.inf, 0], [0, 0]], [[np.nan, 0], [0, 0]], np.zeros((2, 3))):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                coupling.entropic_coupling([.5, .5], [.5, .5], cost)


class ReadoutTests(unittest.TestCase):
    def test_independent_joint_matches_native_mode_local_formula(self):
        # Artificial logits with no active floor; direct native formula is an
        # independent oracle, including inclusive +/-20 and +.5 bin centers.
        index = np.arange(256, dtype=np.float64)
        ly = 3 * np.cos(index / 31) + 2 * np.sin(index / 7)
        lx = 4 * np.sin(index / 19) - 3 * np.cos(index / 11)
        px, py = coupling.native_marginals(np.r_[ly, lx])
        expected = []
        for logits in (lx, ly):
            weights = np.exp(.5 * logits - np.max(.5 * logits))
            weights /= weights.sum()
            weights *= np.abs(index - np.argmax(logits)) <= 20
            weights /= weights.sum()
            expected.append(float(weights @ index + .5))
        np.testing.assert_allclose(coupling.mode_local_mean(coupling.independent_joint(px, py)), expected, atol=6e-14, rtol=0)

    def test_exact_mode_ties_are_xy_lexicographic(self):
        joint = np.array([[.1, .4], [.4, .1]])
        np.testing.assert_array_equal(coupling.mode_local_mean(joint, radius=0), [.5, 1.5])

    def test_joint_mode_local_window_has_explicit_boundaries(self):
        joint = np.array([[.4, .2, .1], [.15, .1, .05]])
        expected = np.array([(.2 + .1) / .85 + .5, (.15 + .1) / .85 + .5])
        np.testing.assert_allclose(coupling.mode_local_mean(joint, radius=1), expected, atol=2e-15, rtol=0)
        np.testing.assert_allclose(coupling.mode_local_mean(joint), coupling.joint_mean(joint), atol=1e-15, rtol=0)

    def test_readout_rejects_invalid_joint_or_radius(self):
        for joint in ([[0, 1]], [[np.nan]], [[.5]]):
            with self.subTest(joint=joint), self.assertRaises(ValueError):
                coupling.mode_local_mean(joint)
        for radius in (-1, True, 1.2):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                coupling.mode_local_mean([[1.0]], radius=radius)


if __name__ == "__main__":
    unittest.main()
