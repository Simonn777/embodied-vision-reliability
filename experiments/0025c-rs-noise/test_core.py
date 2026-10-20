"""Artificial linear-algebra unit properties only; no cohort/noise generation."""
import unittest
from unittest import mock

import numpy as np

import core


class CoreTests(unittest.TestCase):
    def test_so3_and_jacobian_identity(self):
        Q, J = core.rotation_and_left_jacobian(np.zeros(3))
        np.testing.assert_array_equal(Q, np.eye(3))
        np.testing.assert_array_equal(J, np.eye(3))
        w = np.array([.07, -.02, .03])
        Q, J = core.rotation_and_left_jacobian(w)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(Q, np.eye(3) + core.skew(w) @ J, atol=1e-14)

    def test_column_equilibration_is_undone(self):
        A = np.diag([1e6, 1e-3, 4.])
        expected = np.array([2., -3., .4])
        solved = core._linear_svd(A, A @ expected, None)
        np.testing.assert_allclose(solved["solution"], expected, rtol=1e-14)
        np.testing.assert_allclose(solved["column_scales"], [1e-6, 1e3, .25])
        self.assertEqual(solved["rank"], 3)

    def test_wide_full_vt_retains_homogeneous_null_vector(self):
        A = np.column_stack([np.eye(8), np.arange(1., 9.)])
        solved = core._linear_svd(A, None, None)
        self.assertEqual(solved["Vt"].shape, (9, 9))
        self.assertEqual(solved["rank"], 8)
        np.testing.assert_allclose(A @ solved["solution"], 0., atol=2e-14)
        self.assertGreater(abs(solved["solution"][-1]), 0.)

    def test_weak_spectrum_is_separate_from_machine_rank(self):
        solved = core._linear_svd(np.array([[1., 1.], [0., 1e-10]]), np.array([2., 1e-10]), None)
        self.assertEqual(solved["rank"], 2)
        self.assertEqual(solved["weak_rank"], 1)

    def test_translation_null_projector_distinguishes_scale_from_direction(self):
        along = core._linear_svd(np.diag([0., 1., 1.]), None, None)
        across = core._linear_svd(np.diag([1., 0., 1.]), None, None)
        a = core._task_null(along, np.array([1., 0., 0.]))
        b = core._task_null(across, np.array([1., 0., 0.]))
        self.assertEqual(a["numerical"]["translation_direction_frobenius"], 0.)
        self.assertEqual(b["numerical"]["translation_direction_frobenius"], 1.)

    @staticmethod
    def simple_data():
        return dict(r1=np.array([[.1, -.2, 1.], [-.3, .2, 1.], [.2, .3, 1.]]),
                    r2=np.array([[.2, -.1, 1.], [-.2, .1, 1.], [.1, .4, 1.]]),
                    s1=np.zeros(3), s2=np.zeros(3), d1=np.ones(3), d2=np.ones(3),
                    R=np.eye(3), w1=np.zeros(3), w2=np.zeros(3))

    def test_build_signs_and_angular_rays_are_not_renormalized(self):
        data = self.simple_data()
        built = core.build(data, [1], angular_only=False)
        expected = np.column_stack([-np.eye(3), np.zeros((3, 6)),
                                    -data["r1"][1], data["r2"][1], data["r2"][1]])
        np.testing.assert_array_equal(built["A"], expected)
        np.testing.assert_array_equal(built["y"], data["r1"][1])
        data["s1"][0] = .1
        data["w1"][1] = 1.
        angular = core.build(data, [0], angular_only=True)
        self.assertEqual(angular["A"].shape, (3, 6))
        self.assertNotEqual(angular["geometry"]["rbar1"][0, 2], 1.)
        np.testing.assert_array_equal(angular["A"][:, 3], -angular["geometry"]["Rrbar1"][0])

    def test_affine_admission_checks_unfitted_pool_and_scale(self):
        data = self.simple_data()
        physical = np.array([1., 0., 0., .1, 1., .2])
        built = dict(A=np.eye(6), y=physical, indices=np.array([0]), geometry={"pool_size": 3})
        with mock.patch.object(core, "build", return_value=built):
            good = core.solve_affine(data, [0], angular_only=True)
            self.assertEqual(good["status"], "valid")
            data["d1"][-1] = -1.
            bad = core.solve_affine(data, [0], angular_only=True)
            self.assertEqual(bad["status"], "unavailable")
            self.assertEqual(bad["cheirality"]["positive_count"], 2)
            self.assertEqual(bad["cheirality"]["fitting_count"], 1)
            self.assertEqual(bad["cheirality"]["fitting_finite_count"], 1)
            self.assertEqual(bad["cheirality"]["fitting_positive_count"], 1)
            data["d1"][-1] = 1.
            built["y"] = np.array([1., 0., 0., .1, -1., 2.])
            scale_bad = core.solve_affine(data, [0], angular_only=True)
            self.assertEqual(scale_bad["cheirality"]["positive_count"], 3)
            self.assertFalse(scale_bad["cheirality"]["scale_positive"])
            self.assertEqual(scale_bad["status"], "unavailable")

    @staticmethod
    def sign_geometry():
        return dict(pool_size=3, Rrbar1=np.array([[0., 0., 1.]] * 3),
                    rbar2=np.array([[.25, 0., 1.], [.25, 0., 1.], [-.25, 0., 1.]]),
                    RB1=np.zeros((3, 3, 3)), B2=np.zeros((3, 3, 3)))

    def test_directed_majority_preserves_negative_minority(self):
        g = self.sign_geometry()
        x = np.array([.5, 0., 0., 0., 0., 0., 0., 0., 0.])
        result = core._triangulate_signs(g, x, None)
        self.assertEqual(result["selected_sign"], 1)
        self.assertEqual([h["positive_count"] for h in result["hypotheses"]], [2, 1])
        self.assertLess(result["hypotheses"][0]["points"][-1]["depth1"], 0.)
        reversed_result = core._triangulate_signs(g, -x, None)
        self.assertEqual(reversed_result["selected_sign"], -1)

    def test_majority_does_not_hide_rank_deficient_triangulation(self):
        g = self.sign_geometry()
        g["rbar2"][-1] = g["Rrbar1"][-1]
        x = np.array([.5, 0., 0., 0., 0., 0., 0., 0., 0.])
        result = core._triangulate_signs(g, x, None)
        self.assertEqual(result["hypotheses"][0]["positive_count"], 2)
        self.assertFalse(result["all_triangulations_valid"])
        self.assertIsNone(result["selected_sign"])

    def test_depth_free_builder_does_not_access_depth(self):
        data = self.simple_data()
        del data["d1"], data["d2"]
        built = core.build_depth_free(data)
        self.assertEqual(built["A"].shape, (3, 9))
        self.assertIsNone(built["y"])

    def test_bad_inputs_fail_and_are_not_coerced(self):
        data = self.simple_data()
        with self.assertRaises(ValueError):
            core.build(data, [0, 0])
        data["R"][0, 0] = -1.
        with self.assertRaises(ValueError):
            core.build(data)
        with self.assertRaises(FloatingPointError):
            core._linear_svd(np.array([[np.inf]]), np.ones(1), None)
        with self.assertRaises(FloatingPointError):
            core._mm(np.array([[1e308]]), np.array([[1e308]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
