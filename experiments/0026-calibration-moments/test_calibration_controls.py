"""Manufactured arithmetic only. This module does not run an optimizer."""
import unittest
import numpy as np
import moments
import calibration_controls as control


def fixture():
    rng = np.random.default_rng(260027)
    r1 = np.column_stack((rng.uniform(-.5, .5, (9, 2)), np.ones(9)))
    r2 = np.column_stack((rng.uniform(-.4, .6, (9, 2)), np.ones(9)))
    d1, d2 = rng.uniform(1., 4., 9), rng.uniform(.8, 3.7, 9)
    w = np.array((1., .3, 0., 2., .8, 1., .1, 0., .6))
    m1, m2 = min(d1[w > 0]), min(d2[w > 0])
    return (r1, r2, d1, d2, w), m1, m2


class TestCalibrationControls(unittest.TestCase):
    def test_seed_roundtrip_and_positive_support(self):
        data, m1, m2 = fixture()
        for seed in ((.2, 1.3, -.1), (-.5*m1, .7, -.2*m2)):
            q = control.physical_to_unconstrained(seed, m1, m2)
            actual = control.unconstrained_to_physical(q, m1, m2)
            np.testing.assert_allclose(actual, seed, atol=2e-15, rtol=2e-15)
            b, a, c = actual
            self.assertGreater(a, 0.)
            self.assertTrue(np.all(data[2][data[4] > 0]+b > 0.))
            self.assertTrue(np.all(a*data[3][data[4] > 0]+c > 0.))

    def test_G_log_gradient_central_difference(self):
        self.check_gradient("G")

    def test_F_envelope_log_gradient_central_difference(self):
        self.check_gradient("F")

    def check_gradient(self, objective):
        data, m1, m2 = fixture()
        model = moments.build_moments(*data)
        for seed in ((.2, 1.3, -.1), (-.3, .7, .8)):
            q = control.physical_to_unconstrained(seed, m1, m2)
            result = control.evaluate_control(model, q, m1, m2, objective)
            numerical = np.zeros(3)
            for j in range(3):
                dq = np.zeros(3); dq[j] = 1e-6
                upper = control.evaluate_control(model, q+dq, m1, m2, objective)["cost_raw"]
                lower = control.evaluate_control(model, q-dq, m1, m2, objective)["cost_raw"]
                numerical[j] = (upper-lower)/(2*dq[j])
            np.testing.assert_allclose(result["gradient"], numerical, rtol=2e-6, atol=2e-6)

    def test_affine_unit_gauge_cost_and_log_gradient(self):
        data, m1, m2 = fixture()
        r1, r2, d1, d2, weights = data
        seed = np.array((.2, 1.3, -.1)); b, a, c = seed
        s1, t1, s2, t2 = 2.3, -.7, .6, .2
        transformed = np.array((s1*b-t1, s1*a/s2, s1*c-s1*a*t2/s2))
        m1n, m2n = s1*m1+t1, s2*m2+t2
        q = control.physical_to_unconstrained(seed, m1, m2)
        qn = control.physical_to_unconstrained(transformed, m1n, m2n)
        np.testing.assert_allclose(qn, q+np.log((s1, s1/s2, s1)), atol=2e-15)
        model = moments.build_moments(*data)
        modeln = moments.build_moments(r1, r2, s1*d1+t1, s2*d2+t2, weights)
        for objective, exponent in (("G", 4), ("F", 2)):
            old = control.evaluate_control(model, q, m1, m2, objective)
            new = control.evaluate_control(modeln, qn, m1n, m2n, objective)
            np.testing.assert_allclose(new["cost_raw"], s1**exponent*old["cost_raw"], rtol=2e-12)
            np.testing.assert_allclose(new["gradient"], s1**exponent*old["gradient"], rtol=2e-12, atol=2e-10)

    def test_invalid_seed_and_overflow_are_not_repaired(self):
        data, m1, m2 = fixture()
        for seed in ((-m1, 1., 0.), (0., 0., 1.), (0., 1., -m2)):
            with self.assertRaises(ValueError):
                control.physical_to_unconstrained(seed, m1, m2)
        model = moments.build_moments(*data)
        for objective in ("G", "F"):
            with self.assertRaises(control.ControlArithmeticError):
                control.evaluate_control(model, np.array((1000., 0., 0.)), m1, m2, objective)
            with self.assertRaises(control.ControlArithmeticError):
                control.evaluate_control(model, np.array((-1000., 0., 0.)), m1, m2, objective)


if __name__ == "__main__":
    unittest.main(verbosity=2)
