"""Small mathematical fixtures only. No scientific data or model imports."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("independent_0007_scorer", Path(__file__).resolve().parents[1]/"scripts/antithetic_gradient_metrics.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture(cases=2, r=32, p=3):
    rng = np.random.default_rng(32)
    base = rng.normal(size=(cases, 2, r, p)).astype(np.float32)
    g = np.repeat(base[:, :, None], 4, axis=2)
    g[:, :, 2] *= 0.5
    return g, np.ones(g.shape[:-1], dtype=np.float64)


class MathematicsTests(unittest.TestCase):
    def test_trace_covariance_matches_explicit_coordinate_variances(self):
        x = np.array([[1., 9.], [3., 1.], [8., 2.], [4., -2.]])
        _, actual, norm = s._centered_gram(x)
        self.assertAlmostEqual(actual, sum(np.var(x[:, j], ddof=1) for j in range(2)))
        self.assertAlmostEqual(norm, np.linalg.norm(x.mean(0)))

    def test_translation_does_not_change_variance(self):
        x = np.array([[1., 2.], [2., 5.], [8., 3.]])
        np.testing.assert_allclose(s._centered_gram(x)[1], s._centered_gram(x + 1e10)[1], rtol=1e-12, atol=1e-12)

    def test_gram_resampling_matches_direct_repeated_rows(self):
        x = np.array([[1., 7., -1.], [2., 1., 8.], [8., -4., 9.], [3., 1., -2.]])
        selections = np.array([[0, 1, 2, 3], [2, 2, 0, 3], [1, 1, 1, 1], [3, 2, 3, 2]])
        counts = np.array([np.bincount(row, minlength=4) for row in selections])
        gram = s._centered_gram(x)[0]
        actual = s._bootstrap_variance(gram, counts)
        expected = np.array([np.var(x[row], axis=0, ddof=1).sum() for row in selections])
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_bad_count_sum_and_negative_counts_rejected(self):
        gram = np.eye(4)
        with self.assertRaises(ValueError):
            s._bootstrap_variance(gram, np.ones((1, 4)) * 2)
        with self.assertRaises(ValueError):
            s._bootstrap_variance(gram, np.array([[-1., 2., 2., 1.]]))

    def test_explicit_linear_quantile(self):
        self.assertEqual(s._linear_quantile([9, 1, 5, 3], 0), 1)
        self.assertEqual(s._linear_quantile([9, 1, 5, 3], 1), 9)
        self.assertEqual(s._linear_quantile([9, 1, 5, 3], 0.25), 2.5)
        self.assertAlmostEqual(s._linear_quantile([0, 10], s.UPPER_QUANTILE), 9.833333333333332)
        with self.assertRaises(ValueError):
            s._linear_quantile([1, np.nan], 0.5)

    def test_projection_unit_norm_and_fixed_generation(self):
        q = s._rademacher(7)
        self.assertEqual(q.shape, (8, 7))
        np.testing.assert_allclose(np.sum(q*q, axis=1), np.ones(8), atol=1e-15)
        np.testing.assert_array_equal(q, s._rademacher(7))
        self.assertTrue(np.all(np.isin(q * np.sqrt(7), [-1., 1.])))

    def test_geometric_case_average_not_ratio_of_pooled_variances(self):
        self.assertAlmostEqual(s._geomean_ratio([1, 100], [4, 25]), 1.)
        self.assertNotAlmostEqual(s._geomean_ratio([1, 100], [4, 25]), 101/29)

    def test_one_counts_matrix_per_case_used_for_both_regimes_and_all_methods(self):
        g, t = fixture()
        calls = []
        original = s._bootstrap_variance
        def wrapped(gram, counts):
            calls.append(counts)
            return original(gram, counts)
        with mock.patch.object(s, "_bootstrap_variance", side_effect=wrapped):
            s._analyze(g, t, bootstrap_draws=16)
        self.assertEqual(len(calls), 16)
        self.assertTrue(all(x is calls[0] for x in calls[:8]))
        self.assertTrue(all(x is calls[8] for x in calls[8:]))
        self.assertFalse(np.array_equal(calls[0], calls[8]))
        expected = s._resample_counts(32, 16, 2000000)
        np.testing.assert_array_equal(calls[0], expected)


class ContractAndGateTests(unittest.TestCase):
    def score(self, g, t):
        return s._analyze(g, t, bootstrap_draws=64)

    def test_public_contract_refuses_small_fixture(self):
        g, t = fixture()
        with self.assertRaisesRegex(ValueError, "production contract"):
            s.evaluate_gradients(g, t)

    def test_dtype_and_shape_contract(self):
        g, t = fixture()
        for badg, badt in ((g.astype(np.float64), t), (g, t[..., :-1]), (g, t.astype(int))):
            with self.assertRaises(ValueError):
                self.score(badg, badt)

    def test_nonfinite_and_nonpositive_timings_fail_before_scoring(self):
        g, t = fixture()
        for bad in [np.nan, np.inf, 0., -1.]:
            altered = t.copy()
            altered[0, 0, 0, 0] = bad
            with self.assertRaises(ValueError):
                self.score(g, altered)
        g[0, 0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.score(g, t)

    def test_zero_variance_stops_without_ratio_or_resample_redraw(self):
        g, t = fixture()
        g[:, :, 0] = 0
        out = self.score(g, t)
        self.assertEqual(out["status"], "invalid")
        self.assertEqual(out["decision"], "stop_invalid")
        self.assertNotIn("comparisons", out)
        json.dumps(out, allow_nan=False)

    def test_exact_scaling_has_known_ratios_and_all_gates_pass(self):
        g, t = fixture()
        out = self.score(g, t)
        self.assertEqual(out["status"], "analyzed")
        self.assertEqual(out["integrity"]["checks"], 2 * (3 + 6) * 8)
        self.assertTrue(out["gates"]["all_passed"])
        self.assertEqual(out["B_best_control_win_count"], 2)
        for row in out["comparisons"]:
            self.assertAlmostEqual(row["variance_ratio_geomean"], 0.25)
            self.assertAlmostEqual(row["bootstrap_upper"], 0.25)
            self.assertAlmostEqual(row["variance_times_step_time_sum_ratio_geomean"], 0.25)
        json.dumps(out, allow_nan=False)

    def test_A_samepatch_mean_shift_excluded_but_B_included(self):
        g, t = fixture()
        g[:, 0, 3] += 100
        out = self.score(g, t)
        self.assertTrue(out["integrity"]["all_passed"])
        self.assertTrue(all("samepatch" not in r["pair"] for r in out["integrity"]["records"] if r["regime"] == s.REGIMES[0]))
        g[:, 1, 3] += 100
        out = self.score(g, t)
        self.assertFalse(out["integrity"]["all_passed"])
        self.assertFalse(out["gates"]["all_passed"])

    def test_paired_standard_error_uses_differences(self):
        projected = np.zeros((1, 2, 4, 4, 8))
        base = np.array([-100., -3., 1., 102.])
        projected[:] = base[None, None, None, :, None]
        projected[:, 1, 2] += 0.01
        out = s._mean_integrity(projected)
        record = next(x for x in out["records"] if x["regime"] == s.REGIMES[1] and x["pair"] == ["independent", "mean"])
        self.assertTrue(all(x < 1e-12 for x in record["paired_standard_error"]))
        self.assertFalse(any(record["passed"]))

    def test_tolerance_scale_uses_projected_absolute_means(self):
        projected = np.full((1, 2, 4, 4, 8), 1e8)
        projected[:, 1, 2] += 5
        out = s._mean_integrity(projected)
        self.assertTrue(out["all_passed"])

    def test_time_gate_uses_all_B_sums_not_mean_case_ratios(self):
        g, t = fixture()
        t[0, 1] = 100
        t[1, 1] = 1
        t[0, 1, 2] = 101
        t[1, 1, 2] = 2
        out = self.score(g, t)
        self.assertAlmostEqual(out["B_total_mean_vs_independent_step_time_ratio"], 103 / 101)
        self.assertTrue(out["gates"]["B_total_step_time_vs_independent_at_most_110"])

    def test_time_gate_fails_even_with_variance_gain(self):
        g, t = fixture()
        t[:, 1, 2] = 1.2
        out = self.score(g, t)
        self.assertFalse(out["gates"]["B_total_step_time_vs_independent_at_most_110"])
        self.assertFalse(out["gates"]["all_passed"])

    def test_A_mechanism_gate_is_separate_from_B(self):
        g, t = fixture()
        g[:, 0, 2] = g[:, 0, 1]
        out = self.score(g, t)
        self.assertEqual(out["A_mean_vs_random_variance_geomean"], 1.)
        self.assertFalse(out["gates"]["A_mean_vs_random_variance_geomean_below_1"])
        self.assertTrue(out["gates"]["B_variance_geomean_at_most_090_each_control"])

    def test_best_control_ties_are_not_wins(self):
        g, t = fixture()
        g[:, 1, 3] = g[:, 1, 2]
        out = self.score(g, t)
        self.assertEqual(out["B_best_control_win_count"], 0)
        self.assertFalse(out["gates"]["B_best_control_case_wins"])

    def test_exactly_six_of_eight_wins_passes_wins_gate(self):
        g, t = fixture(cases=8, r=16)
        g[:2, 1, 3] = g[:2, 1, 2]
        out = s._analyze(g, t, bootstrap_draws=16, min_case_wins=6)
        self.assertEqual(out["B_best_control_win_count"], 6)
        self.assertTrue(out["gates"]["B_best_control_case_wins"])
        g[2, 1, 3] = g[2, 1, 2]
        out = s._analyze(g, t, bootstrap_draws=16, min_case_wins=6)
        self.assertEqual(out["B_best_control_win_count"], 5)
        self.assertFalse(out["gates"]["B_best_control_case_wins"])

    def test_large_finite_timings_do_not_overflow_cost_products(self):
        g, t = fixture()
        t *= 1e290
        out = self.score(g, t)
        self.assertTrue(out["gates"]["all_passed"])
        for row in out["comparisons"]:
            self.assertAlmostEqual(row["variance_times_step_time_sum_ratio_geomean"], 0.25)
        json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
