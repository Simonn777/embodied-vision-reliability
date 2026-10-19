"""Independent NumPy-only artificial checks; never opens scientific assets.

The scalar TS reference differentiates the bearing epipolar constraint in pixel
coordinates numerically. Cheirality is checked by a separate least-squares ray
intersection. These are mathematical parity tests, not bitwise PoseLib tests.
"""
import importlib.util
from pathlib import Path
import unittest
import warnings

import numpy as np

SPEC = importlib.util.spec_from_file_location("scores0016_under_test", Path(__file__).with_name("common.py"))
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def calibration(K1, K2, threshold=None):
    if threshold is None:
        threshold = .25 * sum(1.0 / K[i, i] for K in (K1, K2) for i in (0, 1))
    return {"K1": K1.tolist(), "K2": K2.tolist(), "normalized_epipolar_threshold": threshold}


def bearing(pixel, K):
    v = np.linalg.solve(K, np.r_[pixel, 1.0])
    return v / np.linalg.norm(v)


def pose(R=None, t=None):
    return {"R": np.eye(3) if R is None else R, "t": np.array([1., 0., 0.]) if t is None else t}


def artificial_scene():
    rng = np.random.default_rng(20261019)
    angle = .12
    R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                  [-np.sin(angle), 0., np.cos(angle)]])
    t = np.array([1., .07, -.04])
    t /= np.linalg.norm(t)
    X = rng.uniform([-3., -2., 4.], [4., 3., 12.], (21, 3))
    X[-3:] *= -1.
    K1 = np.array([[850., 0., 320.], [0., 570., 240.], [0., 0., 1.]])
    K2 = np.array([[1200., 0., 400.], [0., 1030., 300.], [0., 0., 1.]])
    Y = (R @ X.T).T + t
    x1 = (K1 @ X.T).T
    x2 = (K2 @ Y.T).T
    x1 = x1[:, :2] / x1[:, 2:]
    x2 = x2[:, :2] / x2[:, 2:]
    x2 += rng.normal(0., .3, x2.shape)
    x2[2, 1] += 8.
    return x1, x2, calibration(K1, K2), pose(R, t)


def scalar_reference(x1, x2, cal, est):
    K1, K2 = np.array(cal["K1"]), np.array(cal["K2"])
    R, t = np.array(est["R"]), np.array(est["t"], dtype=float)
    t /= np.linalg.norm(t)
    # Construct E by applying t cross (R e_j), avoiding the production skew code.
    E = np.column_stack([np.cross(t, R[:, j]) for j in range(3)])
    F = np.linalg.solve(K2.T, E) @ np.linalg.inv(K1)
    totals = {name: 0. for name in ("pixel_msac", "normalized_msac", "tangent_msac")}
    counts = {name: 0 for name in ("pixel_inliers", "normalized_inliers", "tangent_inliers")}
    cheiral_count = 0
    for p, q in zip(x1, x2):
        p3, q3 = np.r_[p, 1.], np.r_[q, 1.]
        v1, v2 = np.linalg.solve(K1, p3), np.linalg.solve(K2, q3)
        d1, d2 = bearing(p, K1), bearing(q, K2)
        matrix = np.column_stack((R @ d1, -d2))
        depths, _, rank, _ = np.linalg.lstsq(matrix, -t, rcond=None)
        cheiral = rank == 2 and bool(np.all(depths > .01))
        cheiral_count += cheiral
        for label, A, a, b, threshold in (
            ("pixel", F, p3, q3, 1.),
            ("normalized", E, v1, v2, cal["normalized_epipolar_threshold"]),
            ("tangent", E, d1, d2, 1.),
        ):
            c = float(b @ A @ a)
            if label != "tangent":
                gradient = np.r_[(A.T @ b)[:2], (A @ a)[:2]]
            else:
                values = np.r_[p, q]
                gradient = []
                for j in range(4):
                    plus, minus = values.copy(), values.copy()
                    plus[j] += 1e-3
                    minus[j] -= 1e-3
                    cp = bearing(plus[2:], K2) @ E @ bearing(plus[:2], K1)
                    cm = bearing(minus[2:], K2) @ E @ bearing(minus[:2], K1)
                    gradient.append((cp-cm) / .002)
                gradient = np.array(gradient)
            denominator = float(gradient @ gradient) * threshold ** 2
            cost = min(1., c*c/denominator) if cheiral and denominator > 0 else 1.
            totals[label+"_msac"] += cost
            counts[label+"_inliers"] += cost < 1.
    return {**totals, **counts, "cheiral_count": cheiral_count}


class ScoreTests(unittest.TestCase):
    def test_all_three_scores_against_independent_scalar_formulas(self):
        x1, x2, cal, est = artificial_scene()
        actual = common.final_scores(est, common.prepare_geometry(x1, x2, cal))
        expected = scalar_reference(x1, x2, cal, est)
        for name in expected:
            if name.endswith("msac"):
                self.assertAlmostEqual(actual[name], expected[name], delta=2e-7, msg=name)
            else:
                self.assertEqual(actual[name], expected[name], name)
        self.assertGreater(actual["cheiral_count"], 0)
        self.assertLess(actual["cheiral_count"], len(x1))

    def test_unit_bearing_jacobian_against_pixel_finite_differences(self):
        K = np.array([[870., 0., 311.], [0., 530., 229.], [0., 0., 1.]])
        pixels = np.array([[311., 229.], [-800., 1200.], [2000., -700.]])
        _, directions, jac = common.rays_and_jac(pixels, K)
        for i, pixel in enumerate(pixels):
            np.testing.assert_allclose(directions[i], bearing(pixel, K), rtol=0, atol=2e-15)
            for j in range(2):
                delta = np.zeros(2)
                delta[j] = 1e-3
                numerical = (bearing(pixel+delta, K)-bearing(pixel-delta, K)) / .002
                np.testing.assert_allclose(jac[i, :, j], numerical, rtol=2e-7, atol=1e-12)
                self.assertAlmostEqual(float(jac[i, :, j] @ directions[i]), 0., delta=1e-15)

    def test_pixel_sampson_and_tangent_sampson_are_distinct_off_manifold(self):
        K = np.diag([10., 10., 1.])
        p, q = np.array([[20., 10.]]), np.array([[22., 10.3]])
        cal = calibration(K, K)
        scores = common.final_scores(pose(), common.prepare_geometry(p, q, cal))
        expected = scalar_reference(p, q, cal, pose())
        self.assertEqual(scores["cheiral_count"], 1)
        self.assertAlmostEqual(scores["pixel_msac"], .045, delta=1e-13)
        self.assertAlmostEqual(scores["tangent_msac"], expected["tangent_msac"], delta=1e-8)
        self.assertGreater(abs(scores["pixel_msac"]-scores["tangent_msac"]), 1e-5)

    def test_anisotropic_normalized_score_is_not_pixel_score(self):
        x1, x2, cal, est = artificial_scene()
        scores = common.final_scores(est, common.prepare_geometry(x1, x2, cal))
        self.assertGreater(abs(scores["pixel_msac"]-scores["normalized_msac"]), .01)

    def test_exact_correspondences_cheirality_and_translation_sign(self):
        K = np.eye(3)
        p = np.array([[0., 0.], [.1, .2], [-.3, .1]])
        q = p + np.array([.2, 0.])
        g = common.prepare_geometry(p, q, calibration(K, K))
        forward, backward = common.final_scores(pose(), g), common.final_scores(pose(t=np.array([-1., 0., 0.])), g)
        self.assertEqual(forward["cheiral_count"], 3)
        self.assertEqual(backward["cheiral_count"], 0)
        for name in ("pixel_msac", "normalized_msac", "tangent_msac"):
            self.assertLess(forward[name], 1e-25)
            self.assertEqual(backward[name], 3.)

    def test_positive_depth_tolerance_is_after_translation_normalization(self):
        K = np.eye(3)
        p = np.zeros((2, 2))
        q = np.array([[1/.009, 0.], [1/.011, 0.]])
        g = common.prepare_geometry(p, q, calibration(K, K))
        scores = common.final_scores(pose(t=np.array([100., 0., 0.])), g)
        self.assertEqual(scores["cheiral_count"], 1)
        self.assertEqual(scores["pixel_msac"], 1.)
        self.assertEqual(scores["pixel_inliers"], 1)

    def test_zero_denominator_is_finite_saturated_cost(self):
        p = np.zeros((4, 2))
        g = common.prepare_geometry(p, p, calibration(np.eye(3), np.eye(3)))
        scores = common.final_scores(pose(t=np.array([0., 0., 1.])), g)
        self.assertEqual(scores["cheiral_count"], 0)
        for label in ("pixel", "normalized", "tangent"):
            self.assertEqual(scores[label+"_msac"], 4.)
            self.assertEqual(scores[label+"_inliers"], 0)

    def test_truncation_is_one_and_boundary_is_not_an_inlier(self):
        p = np.zeros((3, 2))
        q = np.column_stack((np.ones(3), np.sqrt(2.)*np.array([.5, 1., 2.])))
        scores = common.final_scores(pose(), common.prepare_geometry(p, q, calibration(np.eye(3), np.eye(3))))
        self.assertAlmostEqual(scores["pixel_msac"], 2.25, delta=1e-14)
        self.assertEqual(scores["pixel_inliers"], 1)

    def test_positive_translation_scale_invariance(self):
        x1, x2, cal, est = artificial_scene()
        g = common.prepare_geometry(x1, x2, cal)
        expected = common.final_scores(est, g)
        for scale in (1e-10, .1, 100., 1e100):
            actual = common.final_scores(pose(est["R"], scale*est["t"]), g)
            for name in actual:
                self.assertAlmostEqual(actual[name], expected[name], delta=2e-10)

    def test_invalid_pose_rejected(self):
        x1, x2, cal, est = artificial_scene()
        g = common.prepare_geometry(x1, x2, cal)
        invalid = [pose(t=np.zeros(3)), pose(t=np.array([np.nan, 0., 0.])),
                   pose(R=np.diag([1., 1., -1.])), pose(R=2*np.eye(3))]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                common.final_scores(item, g)

    def test_geometry_nonfinite_inputs_rejected(self):
        x1, x2, cal, _ = artificial_scene()
        for value in (np.nan, np.inf, -np.inf):
            bad = x1.copy()
            bad[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                common.prepare_geometry(bad, x2, cal)

    def test_invalid_calibration_or_threshold_rejected(self):
        x1, x2, cal, _ = artificial_scene()
        for focal in (0., -1., np.nan, np.inf):
            K1 = np.array(cal["K1"])
            K1[0, 0] = focal
            with self.subTest(focal=focal), self.assertRaises(ValueError):
                common.prepare_geometry(x1, x2, calibration(K1, np.array(cal["K2"]), .001))
        for threshold in (0., -1., np.nan, np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                common.prepare_geometry(x1, x2, {**cal, "normalized_epipolar_threshold": threshold})

    def test_extreme_finite_translation_is_safely_normalized(self):
        x1, x2, cal, est = artificial_scene()
        g = common.prepare_geometry(x1, x2, cal)
        expected = common.final_scores(pose(est["R"], np.ones(3)), g)
        for scale in (1e308, 1e-308):
            actual = common.final_scores(pose(est["R"], np.full(3, scale)), g)
            for name in actual:
                self.assertAlmostEqual(actual[name], expected[name], delta=2e-10)

    def test_finite_pixel_overflow_must_not_make_zero_bearings(self):
        x1, x2, cal, _ = artificial_scene()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            huge = x1.copy()
            huge[0] = 1e308
            with self.assertRaises((ValueError, FloatingPointError)):
                common.prepare_geometry(huge, x2, cal)

    def test_nonfinite_prepared_geometry_must_not_be_silently_saturated(self):
        x1, x2, cal, est = artificial_scene()
        original = common.prepare_geometry(x1, x2, cal)
        for field in ("p1", "v2", "d1", "j2", "invK1"):
            g = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in original.items()}
            g[field].flat[0] = np.nan
            with self.subTest(field=field), self.assertRaises(ValueError):
                common.final_scores(est, g)


if __name__ == "__main__":
    unittest.main()
