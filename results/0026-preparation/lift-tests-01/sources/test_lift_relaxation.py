"""Artificial PSD quadratics only; no data, moments, models or pose fixtures.

Prepared before the first invocation. Finite-difference tolerances are fixed
here; a failed first run must be retained rather than tuning them to results.
"""
import math
import unittest

import numpy as np

import lift_relaxation as lr


def target_quadratic(target):
    """Gram construction: [1,x]^T H [1,x] == ||x-target||^2."""
    target = np.asarray(target, dtype=float)
    factor = np.column_stack((-target, np.eye(5)))
    return factor.T @ factor


class LiftRelaxationTests(unittest.TestCase):
    def test_barrier_gradient_and_hessian_by_centered_differences(self):
        v = np.array([1., -2., 3., .5, -1., 2.])
        H = np.diag(np.arange(1., 7.)) + .1*np.outer(v, v)
        x = np.array([.25, 1.3, 1.7, .2, 1.1])
        mu, step = .37, 1e-5
        value, gradient, hessian = lr.barrier_terms(H, x, mu)
        self.assertTrue(math.isfinite(value))
        fd_gradient, fd_hessian = np.empty(5), np.empty((5, 5))
        for j in range(5):
            delta = np.zeros(5)
            delta[j] = step
            plus = lr.barrier_terms(H, x+delta, mu)
            minus = lr.barrier_terms(H, x-delta, mu)
            fd_gradient[j] = (plus[0]-minus[0])/(2.*step)
            fd_hessian[:, j] = (plus[1]-minus[1])/(2.*step)
        np.testing.assert_allclose(gradient, fd_gradient, rtol=5e-7, atol=5e-7)
        np.testing.assert_allclose(hessian, fd_hessian, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(hessian, hessian.T, rtol=0., atol=1e-12)
        self.assertGreater(float(np.linalg.eigvalsh(hessian)[0]), 0.)

    def test_barrier_offdiagonal_cone_factor_at_identity(self):
        # With no objective, x=(0,1,1,0,1), mu=1 has an exact hand derivative.
        value, gradient, hessian = lr.barrier_terms(
            np.zeros((6, 6)), np.array([0., 1., 1., 0., 1.]), 1.)
        self.assertEqual(value, 0.)
        np.testing.assert_array_equal(gradient, [0., -1., -1., 0., -1.])
        np.testing.assert_array_equal(hessian, np.diag([2., 1., 1., 2., 1.]))

    def test_dual_sign_and_quarter_factor_against_direct_lagrangian(self):
        H = np.diag([2., 3., 4., 5., 6., 7.])
        multiplier = .7
        S = np.array([[1.2, .3], [.3, .9]])
        report = lr.dual_diagnostic(H, multiplier, S)
        self.assertEqual(report['status'], 'numerical_only')
        diagonal = np.array([3.7, 4., 5., 6., 7.])
        linear = np.array([0., -.7, -1.2, -.6, -.9])
        minimizer = -linear/(2.*diagonal)
        expected = 2.-.25*sum(float(l*l/q) for l, q in zip(linear, diagonal))
        b, B, A, C, D = minimizer
        direct = lr.objective(H, minimizer) + multiplier*(b*b-B)
        direct -= S[0, 0]*A+2.*S[0, 1]*C+S[1, 1]*D
        self.assertAlmostEqual(report['value'], expected, delta=2e-14)
        self.assertAlmostEqual(report['value'], direct, delta=2e-14)
        # Every interior feasible primal point has objective at least this bound.
        for x in ([0., 1., 1., 0., 1.], [.4, 2., 1., .3, 2.]):
            self.assertTrue(lr.interior(x))
            self.assertLessEqual(report['value'], lr.objective(H, x)+1e-13)

    def test_exact_physical_recovery_and_all_support_minima(self):
        # Parameters b=.25, a=2, c=-.5; both cones exactly rank one.
        x = np.array([.25, .0625, 4., -1., .25])
        out = lr.recover_physical(x, .1, .3)
        self.assertEqual(out['status'], 'feasible')
        np.testing.assert_array_equal(out['parameters'], [.25, 2., -.5])
        np.testing.assert_array_equal(out['lifted'], x)
        self.assertEqual(out['parabola_slack'], 0.)
        self.assertEqual(out['determinant_slack'], 0.)
        # Equality at zero is rejected, not clipped to a positive depth.
        for minima in ((-.25, .3), (.1, .25)):
            bad = lr.recover_physical(x, *minima)
            self.assertEqual(bad['status'], 'unavailable')
            self.assertEqual(bad['reason'], 'nonpositive_corrected_depth')

    def test_known_rank_one_target_convex_quadratic(self):
        # Hand-known physical zero minimum. This is not a geometric data test.
        target = np.array([0., 0., 1., 0., 0.])
        H = target_quadratic(target)
        out = lr.solve_relaxation(H, 1., 1.)
        self.assertEqual(out['status'], 'completed')
        self.assertEqual(len(out['history']), 9)
        self.assertTrue(all(item['converged'] for item in out['history']))
        self.assertEqual(out['physical']['status'], 'feasible')
        self.assertEqual(out['dual']['status'], 'numerical_only')
        self.assertLess(out['physical']['objective'], 1e-6)
        self.assertGreaterEqual(out['physical']['objective'], -1e-12)
        self.assertLessEqual(out['dual']['value'], 1e-10)
        self.assertGreaterEqual(out['physical']['numerical_gap'], -1e-10)
        self.assertLess(out['physical']['numerical_gap'], 1e-5)
        self.assertFalse(out['rigorous_certificate'])
        self.assertFalse(out['robust_estimator'])

    def test_interior_zero_target_does_not_become_physical_certificate(self):
        # The relaxation has exact zero at a strictly interior point, but this
        # point cannot obey B=b^2 and AD=C^2 simultaneously.
        target = np.array([0., 1., 1., 0., 1.])
        H = target_quadratic(target)
        out = lr.solve_relaxation(H, 1., 1.)
        self.assertEqual(out['status'], 'completed')
        self.assertLess(out['objective'], 1e-8)
        self.assertEqual(out['physical']['status'], 'feasible')
        self.assertGreater(out['physical']['parabola_slack'], .9)
        self.assertGreater(out['physical']['determinant_slack'], .9)
        self.assertGreater(out['physical']['objective'], 1.5)
        self.assertGreater(out['physical']['numerical_gap'], 1.5)
        self.assertFalse(out['rigorous_certificate'])

    def test_singular_and_ill_conditioned_duals_are_unavailable(self):
        for H in (np.zeros((6, 6)), np.diag([1., 1e-14, 1., 1., 1., 1.])):
            out = lr.dual_diagnostic(H, 0., np.zeros((2, 2)))
            self.assertEqual(out['status'], 'unavailable')
            self.assertEqual(out['reason'], 'dual_curvature_not_well_conditioned_pd')
            self.assertNotIn('value', out)

    def test_invalid_h_weights_cones_and_minima_fail_closed(self):
        invalid_H = [np.eye(5), np.full((6, 6), np.nan),
                     np.diag([-1., 1., 1., 1., 1., 1.])]
        nonsymmetric = np.eye(6)
        nonsymmetric[1, 2] = .1
        invalid_H.append(nonsymmetric)
        for H in invalid_H:
            with self.assertRaises(ValueError):
                lr.validate_hessian_matrix(H)
        for multiplier, S in ((-1., np.eye(2)), (math.nan, np.eye(2)),
                              (0., np.diag([1., -1.]))):
            with self.assertRaises(ValueError):
                lr.dual_diagnostic(np.eye(6), multiplier, S)
        with self.assertRaises(ValueError):
            lr.solve_relaxation(np.eye(6), math.nan, 1.)
        for x in ([0., 0., 1., 0., 1.], [0., 1., 1., 1., 1.]):
            self.assertFalse(lr.interior(x))
            with self.assertRaises(ValueError):
                lr.barrier_terms(np.eye(6), x, 1.)

    def test_finite_input_arithmetic_overflow_is_not_feasible(self):
        # Both cases were found by source inspection before any test execution.
        cone = np.array([0., 1., 1e200, 0., 1e200])
        self.assertFalse(lr.interior(cone))
        with self.assertRaises(ValueError):
            lr.barrier_terms(np.eye(6), cone, 1.)
        recovery = lr.recover_physical([0., 1., 1e-300, 1e200, 1.], 1., 1.)
        self.assertEqual(recovery['status'], 'unavailable')
        self.assertEqual(recovery['reason'], 'nonfinite_computed_recovery')
        with self.assertRaises(FloatingPointError):
            lr.objective(np.eye(6), np.array([1e200, 0., 0., 0., 0.]))

    def test_roundoff_negative_h_is_reported_without_projection(self):
        H = np.diag([-1e-14, 1., 1., 1., 1., 1.])
        checked, eigenvalues = lr.validate_hessian_matrix(H)
        np.testing.assert_array_equal(checked, H)
        self.assertEqual(float(eigenvalues[0]), -1e-14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
