"""Independent scalar/finite-difference tests on the declared fixed fixtures."""
import unittest
import json
import numpy as np
import math_checks as m


class ManufacturedChecks(unittest.TestCase):
    def setUp(self):
        self.f = m.fixtures()

    def test_feature_formula_independent_scalar(self):
        f = self.f
        expected = []
        a0, a1, a2, b0, b1, b2 = f["theta"]
        for (u, v), d in zip(f["uv"], f["depth"]):
            expected.append((a0 + a1 * u + a2 * v) * d + b0 + b1 * u + b2 * v)
        np.testing.assert_allclose(m.phi(f["uv"], f["depth"]) @ f["theta"], expected,
                                   rtol=0, atol=2e-15)

    def test_projection_jacobian_finite_difference(self):
        p, h = np.array([1.2, -.7, 4.]), m.FD_STEP
        # Direct scalar quotients are independent of the projection helper.
        direct = lambda v: np.array([v[0] / v[2], v[1] / v[2]])
        numerical = np.column_stack([(direct(p + h * e) - direct(p - h * e)) / (2 * h)
                                     for e in np.eye(3)])
        print("FD_CHECK " + json.dumps({"case": "projection", "max_absolute_error": float(
            np.max(np.abs(m.projection_jacobian(p) - numerical)))}))
        np.testing.assert_allclose(m.projection_jacobian(p), numerical,
                                   rtol=m.FD_RTOL, atol=m.FD_ATOL)

    def test_full_analytic_jacobian_independent_finite_difference(self):
        f, h = self.f, m.FD_STEP
        uv, d, theta, R, t = (f[k] for k in ("uv", "depth", "theta", "R", "t"))
        # Independent axis-angle exponential using eigendecomposition-free Rodrigues.
        def rotate(w):
            angle = float(np.linalg.norm(w))
            if angle == 0:
                return np.eye(3)
            axis = w / angle
            cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                              [-axis[1], axis[0], 0.]])
            return (np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis)
                    + np.sin(angle) * cross)
        for unit in (True, False):
            basis = m.tangent_basis(t) if unit else np.eye(3)
            size = 11 if unit else 12
            def forward(delta):
                th = theta + delta[:6]
                rr = rotate(delta[6:9]) @ R
                tt = t + basis @ delta[9:]
                if unit:
                    tt /= np.linalg.norm(tt)
                values = []
                for (u, v), depth in zip(uv, d):
                    z = ((th[0] + th[1] * u + th[2] * v) * depth
                         + th[3] + th[4] * u + th[5] * v)
                    q = rr @ np.array([u * z, v * z, z]) + tt
                    values.extend((q[0] / q[2], q[1] / q[2]))
                return np.array(values)
            numerical = np.column_stack([(forward(h * e) - forward(-h * e)) / (2 * h)
                                         for e in np.eye(size)])
            analytic = m.reference_jacobian(uv, d, theta, R, t, unit)
            print("FD_CHECK " + json.dumps({"case": "unit_joint" if unit else "free_joint",
                                            "max_absolute_error": float(np.max(np.abs(analytic - numerical)))}))
            np.testing.assert_allclose(analytic, numerical, rtol=m.FD_RTOL, atol=m.FD_ATOL)

    def test_generic_local_rank_from_analytic_jacobian_only(self):
        result = m.evaluate()["generic"]
        for name, rank in (("phi_rank", 6), ("pose_rank", 5),
                           ("unit_joint_rank", 11), ("free_joint_rank", 11)):
            self.assertEqual(result[name]["rank"], rank)
            self.assertEqual(set(result[name]["ranks_at_fixed_relative_tolerances"].values()), {rank})

    def test_true_scale_gauge_but_not_additive_depth_gauge(self):
        g = m.evaluate()["gauge"]
        self.assertLess(g["free_scale_J_times_null_norm"], 1e-12)
        self.assertLess(g["finite_scale_projection_max_difference"], 1e-12)
        self.assertGreater(g["finite_shift_projection_max_difference"], 1e-4)
        self.assertGreater(g["additive_shift_after_pose_projection_norm"], 1e-4)

    def test_plane_null_survives_positive_depth_and_slope(self):
        p = m.evaluate()["plane"]
        self.assertEqual(p["phi_rank"]["rank"], 5)
        self.assertLessEqual(p["unit_joint_rank"]["rank"], 10)
        for key in ("Phi_times_null_norm", "J_times_known_null_norm",
                    "finite_epsilon_reference_projection_max_difference",
                    "svd_nullspace_reference_residual_norm"):
            self.assertLess(p[key], 1e-12)
        for key in ("reference_corrected_depth_min", "query_corrected_depth_min",
                    "slope_min_on_fixed_reference_and_query_support"):
            self.assertGreater(p[key], 0.)

    def test_reference_null_visible_only_as_query_depth_shape(self):
        q = m.evaluate()["query_depth_shape_only"]
        self.assertTrue(q["coordinates_disjoint_from_reference"])
        self.assertGreater(q["known_null_visible_norm"], .01)
        self.assertGreater(q["svd_reference_null_after_affine_quotient_norm"], .01)
        self.assertLess(q["quotient_annihilates_scale_norm"], 1e-12)
        self.assertLess(q["quotient_annihilates_shift_norm"], 1e-12)
        self.assertTrue(q["no_pose_fit_or_pose_error_computation"])

    def test_pure_rotation_has_no_depth_information(self):
        p = m.evaluate()["pure_rotation"]
        self.assertLess(p["theta_block_norm"], 1e-12)
        self.assertLess(p["finite_shift_projection_max_difference"], 1e-12)

    def test_nonfinite_and_invalid_projection_fail(self):
        for v in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                m.phi([[0., 0.]], [v])
        for p in ([1., 1., 0.], [1., 1., -1.], [float("nan"), 1., 1.]):
            with self.assertRaises(ValueError):
                m.projection_jacobian(p)


if __name__ == "__main__":
    unittest.main()
