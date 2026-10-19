"""Small algebraic qualification; no benchmark data or optimization."""
import unittest
import numpy as np
import covariance as c


def same_shape(a):
    return {"projected": a, "unprojected": a}


class CovarianceTests(unittest.TestCase):
    def test_schur_matches_nuisance_minimization_and_preserves_null(self):
        j = np.array([[1., 1., 0.], [0., 0., 2.], [2., 2., 1.], [1., 1., -1.]])
        info = c.marginal_information(j, [0, 2])
        self.assertEqual(info["rank"], 1)
        self.assertEqual(info["null"].shape, (2, 1))
        delta = np.array([2., 3.])
        nuisance = np.linalg.lstsq(j[:, [1]], -j[:, [0, 2]] @ delta, rcond=None)[0]
        cost = np.linalg.norm(j[:, [0, 2]] @ delta + j[:, [1]] @ nuisance) ** 2
        self.assertAlmostEqual(cost, np.linalg.norm(info["precision_factor"] @ delta) ** 2)
        np.testing.assert_allclose(info["precision_factor"] @ info["null"], 0., atol=1e-13)

    def test_all_information_absorbed_by_nuisance_is_flat(self):
        j = np.array([[1., 1.], [2., 2.], [3., 3.]])
        info = c.marginal_information(j, [0])
        self.assertEqual(info["rank"], 0)
        self.assertEqual(info["precision_factor"].shape, (0, 1))

    def test_trace_matched_directionality_and_psd(self):
        info = c.marginal_information(np.eye(3), [0, 1, 2])
        deltas = np.zeros((8, 3)); deltas[:, 0] = np.arange(8) - 3.5
        control = c.covariance_controls(info, deltas, same_shape(np.eye(3)))
        scalar = control["identifiable_inflation"]["scalar_deletion"]
        directional = control["identifiable_inflation"]["directional"]
        self.assertAlmostEqual(float(np.trace(scalar)), float(np.trace(directional)))
        self.assertFalse(np.allclose(scalar, directional))
        self.assertGreaterEqual(float(np.linalg.eigvalsh(directional).min()), 1.)
        for name, factor in control["precision_factors"].items():
            np.testing.assert_allclose(factor.T @ factor,
                                       np.linalg.inv(control["identifiable_inflation"][name]), atol=1e-12)

    def test_zero_instability_is_hessian_and_null_remains_flat(self):
        info = c.marginal_information(np.diag([2., 0., 3.]), [0, 1, 2])
        control = c.covariance_controls(info, np.zeros((8, 3)), same_shape(np.eye(3)))
        np.testing.assert_allclose(control["precision_factors"]["directional"].T @
                                   control["precision_factors"]["directional"], np.diag([4., 0., 9.]))
        self.assertEqual(control["shape_null_norm"], 1.)
        self.assertEqual(control["shape_excess_trace"], 0.)

    def test_affine_depth_variations_are_quotiented(self):
        rng = np.random.default_rng(91)
        phi = rng.normal(size=(20, 6)); phi[:, 0] = np.arange(20) + 1; phi[:, 3] = 1
        theta = np.array([1., 0., 0., 0., 0., 0.])
        shape = c.shape_derivative(phi, phi, theta, theta)["projected"]
        # Query scale and both constant-shift columns are in the declared quotient.
        np.testing.assert_allclose(shape[:, [8, 11, 14]], 0., atol=2e-14)
        np.testing.assert_allclose(shape[:, :6], 0.)
        self.assertGreater(np.linalg.norm(shape[:, 6]), 1.)

    def test_no_fold_or_failure_can_disappear(self):
        with self.assertRaises(ValueError): c.delete_view_variability(np.zeros((7, 17)))
        with self.assertRaises(ValueError): c.delete_view_variability(np.full((8, 17), np.nan))
        info = c.marginal_information(np.eye(2), [0, 1])
        factor, alpha = c.residual_inflation_factor(info, np.full(8, 256.))
        self.assertGreater(alpha, 1.)
        np.testing.assert_allclose(factor, info["precision_factor"] / np.sqrt(alpha))

    def test_affine_only_support_roundoff_does_not_create_shape_rank(self):
        # A physical row of pixels at constant raw depth. Every available
        # variation is affine in current corrected depth: no shape remains.
        u = np.linspace(-.7, .7, 9)
        phi = np.column_stack([np.ones(9), u, np.zeros(9), np.ones(9), u, np.zeros(9)])
        ta = np.array([1., .3, 0., .2, .4, 0.])
        tb = np.array([2., .2, 0., .3, .1, 0.])
        shape = c.shape_derivative(phi, phi, ta, tb)
        self.assertLess(np.linalg.norm(shape["projected"]), 1e-13)
        info = c.marginal_information(np.eye(17), np.arange(17))
        delta = np.zeros((8, 17)); delta[:, 6] = np.arange(8)-3.5
        control = c.covariance_controls(info, delta, shape)
        self.assertEqual(control["shape_rank"], 0)
        self.assertEqual(control["shape_excess_trace"], 0.)
        self.assertGreater(control["shape_rank_threshold"], 1e-8)
        np.testing.assert_array_equal(control["identifiable_inflation"]["directional"], np.eye(17))

    def test_log_scale_and_seventeen_column_quotient_against_differences(self):
        u = np.linspace(-.7, .7, 11); v = .4*np.cos(np.arange(11))
        d = 1.4+.2*np.sin(np.arange(11)*.7)
        phi = np.column_stack([d, u*d, v*d, np.ones(11), u, v])
        ta = np.array([1., .13, -.1, .2, .07, -.05])
        tb = np.array([1.7, -.07, .11, .3, .08, .1])
        shape = c.shape_derivative(phi, phi, ta, tb)
        def depths(x):
            a = ta.copy(); b = tb.copy()
            a[1:] += x[6:11]; b[0] *= np.exp(x[11]); b[1:] += x[12:17]
            return np.r_[np.einsum("ni,i->n", phi, a), np.einsum("ni,i->n", phi, b)]
        numerical = []
        for k in range(17):
            dx = np.zeros(17); dx[k] = 1e-6
            numerical.append((depths(dx)-depths(-dx))/(2e-6))
        numerical = np.column_stack(numerical)
        np.testing.assert_allclose(shape["unprojected"], numerical, atol=5e-10, rtol=2e-8)
        # Independently solve the two-column quotient by normal equations.
        projected = numerical.copy()
        for block, theta in [(slice(0, 11), ta), (slice(11, 22), tb)]:
            nuisance = np.column_stack([np.einsum("ni,i->n", phi, theta), np.ones(11)])
            coef = np.linalg.solve(np.einsum("ni,nj->ij", nuisance, nuisance),
                                   np.einsum("ni,nj->ij", nuisance, numerical[block]))
            projected[block] -= np.einsum("ni,ij->nj", nuisance, coef)
        np.testing.assert_allclose(shape["projected"], projected, atol=1e-9, rtol=2e-7)

    def test_invalid_and_empty_depth_support_is_not_information(self):
        phi = np.array([[1., .1, .2, 1., .1, .2], [np.nan]*6])
        theta = np.array([1., 0., 0., 0., 0., 0.])
        shape = c.shape_derivative(phi, np.empty((0, 6)), theta, theta,
                                   valid_anchor=np.array([True, False]), valid_query=np.empty(0, dtype=bool))
        self.assertEqual(shape["projected"].shape, (1, 17))
        self.assertTrue(np.isfinite(shape["projected"]).all())
        empty = c.shape_derivative(phi, phi, theta, theta,
                                   valid_anchor=np.zeros(2, dtype=bool), valid_query=np.zeros(2, dtype=bool))
        info = c.marginal_information(np.eye(17), np.arange(17))
        control = c.covariance_controls(info, np.ones((8, 17)), empty)
        self.assertEqual(control["shape_rank"], 0); self.assertEqual(control["shape_unprojected_scale"], 0.)
        np.testing.assert_array_equal(control["identifiable_inflation"]["directional"], np.eye(17))
        with self.assertRaises(ValueError):
            c.shape_derivative(phi, phi, theta, theta)  # invalid flag cannot be omitted

    def test_rectangular_jacobian_retains_complete_flat_nullspace(self):
        info = c.marginal_information(np.array([[1., 0., 2., 0.]]), [0, 1, 2, 3])
        self.assertEqual(info["rank"], 1); self.assertEqual(info["null"].shape, (4, 3))
        np.testing.assert_allclose(info["precision_factor"] @ info["null"], 0., atol=1e-14)


if __name__ == "__main__":
    unittest.main()
