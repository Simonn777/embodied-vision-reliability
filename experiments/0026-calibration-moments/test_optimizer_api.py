"""Manufactured optimizer interface smoke; no benchmark or accuracy admission.

Run only with the already installed SciPy runtime on PYTHONPATH. No acquisition,
real image/depth/pose input, robust sampling, parameter search or model inference.
"""
import types
import unittest
from unittest.mock import patch
import numpy as np
import moments
import calibration_controls as control


def manufactured():
    # Sixteen fixed, noncoplanar, positive-depth points, not random draws.
    X = np.array((
        (-1.0, -.7, 2.0), (.8, -.6, 2.3), (-.6, .9, 2.6), (1.1, .7, 3.0),
        (-1.2, .2, 3.4), (.4, -1.0, 3.8), (.9, .1, 4.2), (-.3, 1.1, 4.7),
        (.2, .4, 2.1), (-.8, -.1, 4.4), (1.2, -.8, 3.6), (-.4, -.9, 2.9),
        (.6, 1.0, 4.9), (-1.1, .8, 4.0), (.1, -.3, 3.2), (.7, .5, 2.7)
    ), dtype=np.float64)
    R = np.array(((.96, 0., .28), (0., 1., 0.), (-.28, 0., .96)))
    t = np.array((.3, -.12, .2))
    Y = np.einsum("ij,nj->ni", R, X, optimize=False) + t
    # Known affine values construct the artificial inputs only; the optimizer
    # receives no truth. The fixed seed below differs in all three parameters.
    b, a, c = .35, 1.25, -.2
    d1, d2 = X[:, 2]-b, (Y[:, 2]-c)/a
    r1, r2 = X/X[:, 2, None], Y/Y[:, 2, None]
    weights = np.array((1., .7, 1.2, 1., .6, 1.1, .8, 1.,
                        .9, 1., .75, 1.25, 1., .85, 1.15, .95))
    model = moments.build_moments(r1, r2, d1, d2, weights)
    return model, float(np.min(d1)), float(np.min(d2)), np.array((.1, 1.05, .1))


class TestOptimizerAPI(unittest.TestCase):
    def test_actual_scipy_both_objectives_decrease(self):
        import scipy
        self.assertEqual(scipy.__version__, "1.13.1")
        model, m1, m2, seed = manufactured()
        self.assertEqual(model.support_count, 16)
        q0 = control.physical_to_unconstrained(seed, m1, m2)
        for objective in ("G", "F"):
            with self.subTest(objective=objective):
                initial = control.evaluate_control(model, q0, m1, m2, objective)
                result = control.optimize_calibration(model, seed, m1, m2, objective)
                self.assertIn(result["status"], ("completed", "failed"))
                self.assertEqual(result["status"] == "completed", result["failure"] is None)
                report = result["optimizer"]
                self.assertIsNotNone(report)
                self.assertEqual(result["status"] == "completed", report["success"])
                self.assertEqual(result["method"], "L-BFGS-B")
                self.assertEqual(result["options"], control.OPTIONS)
                self.assertIsNone(result["bounds"])
                self.assertEqual(result["objective_evaluations"], report["nfev"])
                self.assertGreater(result["objective_evaluations"], 0)
                self.assertGreaterEqual(result["feasible_evaluations"], 1)
                self.assertLessEqual(result["feasible_evaluations"], result["objective_evaluations"])
                self.assertGreaterEqual(result["optimization_seconds"], 0.)
                self.assertTrue(np.isfinite(result["optimization_seconds"]))
                final = result["final_evaluation"]
                self.assertIsNotNone(final)
                self.assertLess(final["cost_raw"], initial["cost_raw"])
                for name in ("parameters", "coordinates", "gradient", "gradient_physical"):
                    self.assertTrue(np.isfinite(final[name]).all(), name)
                b, a, c = final["parameters"]
                self.assertGreater(a, 0.)
                self.assertGreater(b+m1, 0.)
                self.assertGreater(a*m2+c, 0.)
                if objective == "F":
                    self.assertTrue(np.isfinite(final["detail"]["R"]).all())
                    self.assertTrue(np.isfinite(final["detail"]["t"]).all())
                # Deliberately no truth-error or convergence-accuracy gate.

    def test_nonconverged_return_preserves_final_and_last_evaluation(self):
        model, m1, m2, seed = manufactured()
        calls = []

        def fake_minimize(function, q0, **kwargs):
            calls.append(kwargs)
            cost, gradient = function(q0)
            return types.SimpleNamespace(x=q0.copy(), jac=gradient.copy(), fun=cost,
                                         success=False, status=1,
                                         message="manufactured iteration limit",
                                         nit=0, nfev=1, njev=1)

        with patch("scipy.optimize.minimize", side_effect=fake_minimize):
            result = control.optimize_calibration(model, seed, m1, m2, "F")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], {"jac": True, "method": "L-BFGS-B",
                                    "bounds": None, "options": control.OPTIONS})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["type"], "optimizer_nonconvergence")
        self.assertFalse(result["optimizer"]["success"])
        self.assertEqual(result["objective_evaluations"], 1)
        self.assertEqual(result["feasible_evaluations"], 1)
        self.assertIsNotNone(result["last_feasible"])
        self.assertIsNotNone(result["final_evaluation"])
        np.testing.assert_allclose(result["final_evaluation"]["parameters"], seed, atol=1e-15)

    def test_overflow_trial_preserves_previous_feasible_without_retry(self):
        model, m1, m2, seed = manufactured()
        calls = []

        def fake_minimize(function, q0, **kwargs):
            calls.append(kwargs)
            function(q0)
            function(np.array((1000., 0., 0.)))
            self.fail("invalid numerical trial should have raised")

        with patch("scipy.optimize.minimize", side_effect=fake_minimize):
            result = control.optimize_calibration(model, seed, m1, m2, "G")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["type"], "ControlArithmeticError")
        self.assertIsNone(result["optimizer"])
        self.assertIsNone(result["final_evaluation"])
        self.assertEqual(result["objective_evaluations"], 2)
        self.assertEqual(result["feasible_evaluations"], 1)
        self.assertIsNotNone(result["last_feasible"])
        np.testing.assert_allclose(result["last_feasible"]["parameters"], seed, atol=1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
