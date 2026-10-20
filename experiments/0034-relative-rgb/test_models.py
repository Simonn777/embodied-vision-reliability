"""Small manufactured checks; no scientific inputs or native models."""

import json
import math
import unittest
from unittest import mock

import numpy as np

import models as m


def manual_model(name, coefficients):
    return {
        "schema": m.MODEL_SCHEMA, "status": "completed", "name": name,
        "family": "signed_gate" if name in m.GATED else "ordinary",
        "input_dim": m.DIMENSIONS[name], "coefficients": list(coefficients),
    }


class PointwiseModels(unittest.TestCase):
    def test_ordinary_hand_calculation_and_input_immutability(self):
        x = np.array([[1., -2., .5], [1., 3., -1.]])
        original = x.copy()
        coefficient = [.3, -.2, .4]
        result = m.predict_delta(manual_model("logit", coefficient), x)
        expected = np.array([4 * math.tanh(.3 + .4 + .2),
                             4 * math.tanh(.3 - .6 - .4)])
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(x, original)
        self.assertEqual(result.dtype, np.float64)

    def test_signed_polarity_exact_zero_and_evidence_bound(self):
        x = np.array([[1., 0., .2, .8]] * 5)
        e = np.array([-1., -.3, 0., .3, 1.])
        model = manual_model("signed_no_gram", [-4., 0., .5, -.25])
        actual = m.predict_delta(model, x, evidence=e)
        gate = 1 / (1 + math.exp(4.1))
        np.testing.assert_allclose(actual, 4 * e * gate, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(m.predict_delta(model, x, evidence=-e), -actual)
        self.assertEqual(actual[2], 0.0)
        self.assertTrue(np.all(abs(actual) <= 4 * abs(e)))
        saturated = manual_model("signed_no_gram", [1000., 0., 0., 0.])
        np.testing.assert_array_equal(m.predict_delta(saturated, x, evidence=e), 4 * e)

    def test_toy_fit_uses_unrenormalized_weights_and_reduces_objective(self):
        t = np.array([-1., -.5, -.2, .2, .5, 1.])
        x = np.column_stack((np.ones(6), t, t * t))
        y = (t > 0).astype(float)
        w = np.full(6, .4 / 6)
        model, trace = m.fit_model("logit", x, np.zeros(6), y, w)
        self.assertAlmostEqual(trace["closures"][0]["data_loss"], .4 * math.log(2), 13)
        self.assertLess(model["terminal"]["loss"], trace["closures"][0]["loss"] - .01)
        delta = m.predict_delta(model, x)
        expected_data = sum(float(wi) * (np.logaddexp(0., di) - yi * di)
                            for wi, di, yi in zip(w, delta, y))
        expected_penalty = 1e-3 * sum(float(wi) * di * di for wi, di in zip(w, delta))
        self.assertAlmostEqual(model["terminal"]["data_loss"], expected_data, 13)
        self.assertAlmostEqual(model["terminal"]["penalty_loss"], expected_penalty, 13)
        self.assertEqual(model["optimizer"]["func_evals"], len(trace["closures"]))
        self.assertFalse(model["optimizer"]["convergence_claimed"])
        json.dumps([model, trace], allow_nan=False)

    def test_terminal_autograd_matches_independent_finite_difference(self):
        t = np.array([-1., -.3, .4, .9])
        x = np.column_stack((np.ones(4), t, t * t))
        logits = np.array([.2, -.1, .4, -.2])
        target = np.array([0., 1., 0., 1.])
        weights = np.array([.1, .15, .2, .25])
        model, _ = m.fit_model("logit", x, logits, target, weights)
        coefficient = np.array(model["coefficients"])

        def objective(c):
            delta = 4 * np.tanh(x @ c)
            z = logits + delta
            return float(np.sum(weights * (np.logaddexp(0., z) - target * z
                                           + .001 * delta ** 2)))

        numeric = []
        for i in range(3):
            direction = np.zeros(3)
            direction[i] = 1e-5
            numeric.append((objective(coefficient + direction)
                            - objective(coefficient - direction)) / 2e-5)
        np.testing.assert_allclose(model["terminal"]["gradient"], numeric,
                                   rtol=1e-4, atol=1e-8)

    def test_gate_initial_loss_gradient_and_nonzero_initialization(self):
        import torch
        x = np.array([[1., .2, .3, .4], [1., .6, .2, .1]])
        e = np.array([.5, -.7])
        logits = np.array([.2, -.3])
        y = np.array([1., 0.])
        w = np.array([.2, .3])

        # One closure is enough to inspect the declared starting operator.
        def one_closure_then_error(optimizer, closure):
            closure()
            raise RuntimeError("manufactured optimizer interruption")

        with mock.patch.object(torch.optim.LBFGS, "step", one_closure_then_error):
            with self.assertRaises(m.FitError) as caught:
                m.fit_model("signed_no_gram", x, logits, y, w, evidence=e)
        error = caught.exception
        self.assertEqual(error.partial_model["initial_coefficients"], [-4., 0., 0., 0.])
        rho = 1 / (1 + math.exp(4))
        delta = 4 * e * rho
        z = logits + delta
        expected_loss = np.sum(w * (np.logaddexp(0., z) - y * z + .001 * delta ** 2))
        grad = x.T @ (w * (1 / (1 + np.exp(-z)) - y + .002 * delta)
                      * 4 * e * rho * (1 - rho))
        row = error.closure_trace["closures"][0]
        self.assertAlmostEqual(row["loss"], float(expected_loss), 13)
        self.assertAlmostEqual(row["gradient_norm"], float(np.linalg.norm(grad)), 13)
        self.assertEqual(error.reason_code, "optimizer_exception")
        self.assertEqual(error.partial_model["status"], "failed")
        json.dumps([error.partial_model, error.closure_trace], allow_nan=False)

    def test_no_training_weight_is_not_a_numerical_failure(self):
        for x, values in [(np.empty((0, 3)), np.empty(0)),
                          (np.array([[1., 0., 0.]]), np.zeros(1))]:
            with self.subTest(n=len(x)), self.assertRaises(m.FitError) as caught:
                m.fit_model("logit", x, values, values, values)
            self.assertEqual(caught.exception.reason_code, "no_positive_training_weight")
            self.assertEqual(caught.exception.closure_trace["closures"], [])

    def test_invalid_input_contracts_fail_before_optimization(self):
        x = np.array([[1., 0., 0.]])
        valid = dict(name="logit", x=x, native_logits=np.zeros(1),
                     target=np.zeros(1), weights=np.ones(1))
        changes = [
            {"x": np.ones((1, 2))}, {"x": np.zeros((1, 3))},
            {"native_logits": np.array([np.nan])}, {"target": np.array([255.])},
            {"weights": np.array([-1.])}, {"weights": np.array([1.1])},
            {"weights": np.ones(2)}, {"evidence": np.zeros(1)},
        ]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(m.FitError) as caught:
                m.fit_model(**(valid | change))
            self.assertEqual(caught.exception.reason_code, "invalid_input")
            self.assertEqual(caught.exception.closure_trace["closures"], [])
        for e in (None, np.array([1.01]), np.zeros(2)):
            with self.subTest(e=str(e)), self.assertRaises(m.FitError):
                m.fit_model("signed_no_gram", np.ones((1, 4)), [0.], [0.], [1.],
                            evidence=e)

    def test_nonfinite_gradient_preserves_finite_loss_and_failure(self):
        with self.assertRaises(m.FitError) as caught:
            m.fit_model("logit", np.array([[1., 1e308, 0.]]), [0.], [1.], [1.])
        error = caught.exception
        self.assertEqual(error.reason_code, "nonfinite_gradient")
        self.assertAlmostEqual(error.closure_trace["closures"][0]["loss"], math.log(2))
        self.assertEqual(error.partial_model["status"], "failed")
        json.dumps([error.partial_model, error.closure_trace], allow_nan=False)

    def test_predict_rejects_failed_wrong_dimension_and_overflow(self):
        base = manual_model("logit", [0., 0., 0.])
        for model in (base | {"status": "failed"},
                      base | {"coefficients": [0.]}, base | {"input_dim": 4}):
            with self.assertRaises(ValueError):
                m.predict_delta(model, [[1., 0., 0.]])
        with self.assertRaises(ValueError):
            m.predict_delta(manual_model("logit", [0., 2., 0.]), [[1., 1e308, 0.]])
        empty = m.predict_delta(base, np.empty((0, 3)))
        self.assertEqual(empty.shape, (0,))


if __name__ == "__main__":
    unittest.main()
