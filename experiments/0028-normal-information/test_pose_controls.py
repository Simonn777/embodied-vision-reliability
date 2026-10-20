"""Manufactured arithmetic qualification only; no data, model, or native API."""
import json
import unittest
from unittest.mock import patch

import numpy as np

import pose_controls as pc


def fixture():
    X = np.array([[-.8, -.5, 3.], [.7, -.3, 3.6], [-.5, .8, 4.2],
                  [.9, .6, 3.2], [.1, -.8, 4.8], [-.3, .2, 5.1],
                  [.4, .9, 5.5], [-.9, .1, 4.4]])
    angle = .16
    R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                  [-np.sin(angle), 0., np.cos(angle)]])
    t = np.array([.4, -.15, .25])
    Y = X @ R.T + t
    n1 = np.array([[.2, .3, -1.], [-.4, .2, -1.], [.3, -.2, -.8],
                   [.7, .1, -.6], [-.1, -.6, -.7], [.4, .5, -.8],
                   [-.6, .3, -.9], [.2, -.4, -.9]])
    n1 /= np.linalg.norm(n1, axis=1)[:, None]
    n2 = n1 @ R.T
    return dict(X=X, Y=Y, r1=X/X[:, 2:], r2=Y/Y[:, 2:],
                R=R, t=t, n1=n1, n2=n2,
                K1=np.array([[510., 3., 319.5], [0., 490., 239.5], [0., 0., 1.]]),
                K2=np.array([[470., -2., 300.5], [0., 530., 250.5], [0., 0., 1.]]))


class PoseControlTests(unittest.TestCase):
    def test_exact_scale_only_and_depth_gauge(self):
        f = fixture()
        target_scale = 1.7
        out = pc.solve_scale_only(f["r1"], f["r2"], f["X"][:, 2],
                                  f["Y"][:, 2]/target_scale)
        self.assertEqual(out["status"], "completed")
        np.testing.assert_allclose(out["R"], f["R"], atol=2e-12)
        np.testing.assert_allclose(out["t"], f["t"], atol=2e-12)
        self.assertAlmostEqual(out["a"], target_scale, places=12)
        self.assertLess(out["sum_squared_residual"], 1e-22)
        self.assertEqual(out["support_indices"], list(range(8)))
        other = pc.solve_scale_only(f["r1"], f["r2"], 3*f["X"][:, 2],
                                    2*f["Y"][:, 2]/target_scale)
        np.testing.assert_allclose(other["R"], f["R"], atol=2e-12)
        np.testing.assert_allclose(other["t"], 3*f["t"], atol=4e-12)
        self.assertAlmostEqual(other["a"], 1.5*target_scale, places=12)
        json.dumps(out, allow_nan=False)

    def test_reflection_is_not_returned_as_rotation(self):
        f = fixture()
        reflected = f["X"] * [-1., 1., 1.] + [.2, -.1, .4]
        out = pc.solve_scale_only(f["r1"], reflected/reflected[:, 2:],
                                  f["X"][:, 2], reflected[:, 2])
        self.assertEqual(out["status"], "completed")
        self.assertTrue(out["reflected_unconstrained"])
        self.assertAlmostEqual(np.linalg.det(out["R"]), 1., places=12)
        self.assertGreater(out["sum_squared_residual"], 1e-5)

    def test_fixed_support_rank_and_nonpositive_depth(self):
        X = np.array([[i*.1, .2, 3.] for i in range(5)])
        Y = X + [.3, -.1, .2]
        out = pc.solve_scale_only(X/X[:, 2:], Y/Y[:, 2:], X[:, 2], Y[:, 2])
        self.assertEqual(out["status"], "unavailable")
        f = fixture()
        depths = f["X"][:, 2].copy()
        depths[3] = 0.
        out = pc.solve_scale_only(f["r1"], f["r2"], depths, f["Y"][:, 2])
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(out["support_count"], 8)

    def test_normal_pose_recovers_rotation_and_directed_translation(self):
        f = fixture()
        out = pc.solve_normal_pose(f["r1"], f["r2"], f["n1"], f["n2"])
        self.assertEqual(out["status"], "completed")
        np.testing.assert_allclose(out["R"], f["R"], atol=2e-12)
        np.testing.assert_allclose(out["t"], f["t"]/np.linalg.norm(f["t"]), atol=2e-11)
        signs = out["cheirality"]
        self.assertEqual(sorted(h["positive_count"] for h in signs["hypotheses"]), [0, 8])
        self.assertEqual(signs["hypotheses"][signs["selected_index"]]["positive_count"], 8)
        # Arbitrary nonzero raw normal magnitudes do not change oriented alignment.
        other = pc.solve_normal_pose(f["r1"], f["r2"], 2*f["n1"], 3*f["n2"])
        np.testing.assert_allclose(other["R"], out["R"], atol=2e-12)

    def test_degenerate_normals_and_zero_baseline_rays(self):
        f = fixture()
        normals = np.tile([0., 0., -1.], (8, 1))
        out = pc.solve_normal_pose(f["r1"], f["r2"], normals, normals)
        self.assertEqual(out["status"], "unavailable")
        out = pc.solve_normal_pose(f["r1"], f["r1"], f["n1"], f["n1"])
        # Identity alignment can incur roundoff. Exact repeated observations
        # below exercise an exactly zero epipolar system deterministically.
        r = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        n = np.eye(3)
        exact = pc.solve_normal_pose(r, r, n, n)
        self.assertEqual(exact["status"], "unavailable")

    def test_translation_sign_tie_and_tiny_nonzero_input(self):
        r = np.tile([0., 0., 1.], (3, 1))
        signs = pc.translation_signs(np.eye(3), [1e-300, 0., 0.], r, r)
        self.assertTrue(signs["tie"])
        np.testing.assert_array_equal(signs["t"], [1., 0., 0.])
        self.assertEqual(signs["selected_index"], 0)
        self.assertEqual(signs["hypotheses"][0]["finite_rank2_count"], 0)

    def test_pixel_score_full_K_sign_invariance_and_exact_tie_order(self):
        f = fixture()
        x1 = (f["r1"] @ f["K1"].T)[:, :2]
        x2 = (f["r2"] @ f["K2"].T)[:, :2]
        score = pc.pixel_msac(f["R"], f["t"], x1, x2, f["K1"], f["K2"])
        self.assertEqual(score["status"], "completed")
        self.assertLess(score["score"], 1e-22)
        x2 = x2.copy()
        x2[0] += [70., -110.]
        positive = pc.pixel_msac(f["R"], f["t"], x1, x2, f["K1"], f["K2"])
        negative = pc.pixel_msac(f["R"], -f["t"], x1, x2, f["K1"], f["K2"])
        self.assertEqual(positive["score"], negative["score"])
        self.assertEqual(positive["costs"][0], 1.)
        candidates = [dict(status="completed", R=f["R"], t=f["t"]),
                      dict(status="unavailable"),
                      dict(status="completed", R=f["R"], t=-f["t"])]
        selected = pc.select_by_pixel_msac(candidates, x1, x2, f["K1"], f["K2"])
        self.assertEqual(selected["selected_index"], 0)
        self.assertEqual(len(selected["attempts"]), 3)

    def test_exact_joint_refinement_keeps_all_starts_and_both_signs(self):
        f = fixture()
        starts = [dict(label="normal", R=f["R"], t=f["t"]),
                  dict(label="E", R=f["R"], t=-f["t"])]
        result = pc.refine_joint(f["r1"], f["r2"], f["n1"], f["n2"],
                                 f["K1"], f["K2"], starts)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["attempts"]), 2)
        for attempt in result["attempts"]:
            self.assertEqual(attempt["status"], "completed")
            self.assertLess(attempt["sum_squared_residual"], 1e-20)
            np.testing.assert_allclose(attempt["R"], f["R"], atol=2e-10)
            np.testing.assert_allclose(attempt["t"], pc._unit(f["t"]), atol=2e-10)
            self.assertEqual(attempt["support_indices"], list(range(8)))
            self.assertEqual(len(attempt["candidates"]), 2)
            self.assertLessEqual(attempt["nfev"], 100)
            self.assertGreaterEqual(attempt["residual_calls"], attempt["nfev"])
        json.dumps(result, allow_nan=False)

    def test_stereographic_chart_covers_second_hemisphere(self):
        t = np.array([0., 0., 1.])
        R, direction = pc._chart(np.array([0., 0., 0., 3., 1.]), np.eye(3), t, pc._basis(t))
        self.assertLess(np.dot(direction, t), 0.)
        self.assertAlmostEqual(np.linalg.norm(direction), 1., places=12)
        np.testing.assert_array_equal(R, np.eye(3))

    def test_sample_normal_requires_all_four_neighbors(self):
        normal = np.tile([0., 0., -1.], (3, 3, 1))
        mask = np.ones((3, 3), dtype=bool)
        out = pc.sample_normal(normal, mask, [[.25, .5], [2., 1.]])
        self.assertEqual(out["usable_count"], 1)
        np.testing.assert_array_equal(out["samples"][0]["normal"], [0., 0., -1.])
        mask[1, 1] = False
        self.assertEqual(pc.sample_normal(normal, mask, [[0., 0.]])["usable_count"], 0)
        mask[:] = True
        normal[0, 0] = [0., 0., 1.]
        normal[0, 1] = [0., 0., 1.]
        out = pc.sample_normal(normal, mask, [[.5, .5]])
        self.assertEqual(out["samples"][0]["reason"], "interpolated_normal_collapse")
        normal[0, 0] = np.nan
        self.assertEqual(pc.sample_normal(normal, mask, [[.2, .3]])["usable_count"], 0)

    def test_numerical_unavailability_does_not_hide_programming_errors(self):
        f = fixture()
        bad = f["n1"].copy()
        bad[0, 0] = np.nan
        self.assertEqual(pc.solve_normal_pose(f["r1"], f["r2"], bad, f["n2"])["status"],
                         "unavailable")
        with self.assertRaises(ValueError):
            pc.solve_normal_pose(f["r1"][:, :2], f["r2"], f["n1"], f["n2"])
        with self.assertRaises(ValueError):
            pc.solve_scale_only(f["r1"].astype(complex), f["r2"], f["X"][:, 2], f["Y"][:, 2])
        starts = [dict(label="E", R=f["R"], t=f["t"])]
        with patch("scipy.optimize.least_squares", side_effect=ValueError("programming/API defect")):
            with self.assertRaisesRegex(ValueError, "programming/API defect"):
                pc.refine_joint(f["r1"], f["r2"], f["n1"], f["n2"], f["K1"], f["K2"], starts)
        bad_result = pc.refine_joint(f["r1"], f["r2"], bad, f["n2"], f["K1"], f["K2"], starts)
        self.assertEqual(bad_result["status"], "unavailable")
        self.assertEqual(bad_result["attempts"][0]["residual_calls"], 0)


if __name__ == "__main__":
    unittest.main()
