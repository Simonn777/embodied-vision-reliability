"""Small manufactured checks only; no file inputs or native model."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "d37_models", Path(__file__).with_name("models.py"))
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


def toy():
    x = np.zeros((64, 3), dtype=np.float32)
    x[:, 0] = np.tile(np.array([-1, 1], dtype=np.float32), 32)
    x[:, 1] = 0.25
    y = (x[:, 0] > 0).astype(np.float32)
    return x, y, np.full(64, 1 / 64, dtype=np.float32)


class ModelsTest(unittest.TestCase):
    def test_separable_fit_terminal_trace_serialization_and_immutability(self):
        x, y, w = toy()
        originals = [a.copy() for a in (x, y, w)]
        model, trace = models.fit(x, y, w)
        for original, actual in zip(originals, (x, y, w)):
            np.testing.assert_array_equal(original, actual)
        self.assertEqual(trace["completed_epochs"], 20)
        self.assertEqual(trace["steps"], 20)
        self.assertEqual(len(trace["epochs"]), 20)
        self.assertLess(trace["terminal_weighted_bce"], trace["initial_weighted_bce"])
        p = models.predict(model, x)
        self.assertEqual(p.dtype, np.float32)
        self.assertGreater(np.mean((p >= 0.5) == y), 0.9)
        restored = json.loads(json.dumps(model, allow_nan=False))
        np.testing.assert_array_equal(p, models.predict(restored, x))
        json.dumps(trace, allow_nan=False)

    def test_constant_keep_zero_active_zero_weights_and_empty(self):
        x, y, w = toy()
        cases = [(x, y, w, np.zeros(len(x), dtype=bool)),
                 (x, y, np.zeros_like(w), None),
                 (x[:0], y[:0], w[:0], None)]
        for inputs in cases:
            model, trace = models.fit(*inputs)
            self.assertEqual(model["kind"], "constant_keep")
            self.assertEqual(trace["reason"], "no_positive_eligible_weight")
            self.assertFalse(trace["actual_fit"])
            self.assertEqual(trace["steps"], 0)
            np.testing.assert_array_equal(models.predict(model, x), np.ones(64))

    def test_inactive_rows_do_not_influence_gradient_or_order(self):
        x, y, w = toy()
        active = np.arange(len(x)) < 32
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[~active] = 3
        changed_y[~active] = 1 - changed_y[~active]
        first, trace1 = models.fit(x, y, w, active)
        second, trace2 = models.fit(changed_x, changed_y, w, active)
        self.assertEqual(first["state_dict"], second["state_dict"])
        self.assertEqual(trace1["initial_state_dict"], trace2["initial_state_dict"])
        self.assertEqual([e["order_sha256"] for e in trace1["epochs"]],
                         [e["order_sha256"] for e in trace2["epochs"]])
        self.assertEqual(trace1["effective_float32_weight_sum"], 0.5)
        self.assertEqual(trace1["n"], 64)
        self.assertEqual(trace1["steps"], 20)

    def test_fractional_labels_are_not_thresholded(self):
        x = np.zeros((16, 2), dtype=np.float32)
        y = np.full(16, 0.25, dtype=np.float32)
        w = np.full(16, 1 / 16, dtype=np.float32)
        model, trace = models.fit(x, y, w)
        p = models.predict(model, x)
        hand = float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))
        self.assertAlmostEqual(hand, trace["terminal_weighted_bce"], places=6)
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_unscaled_weight_sum_is_preserved(self):
        x, y, w = toy()
        model, trace = models.fit(x, y, w / 4)
        p = models.predict(model, x)
        hand = float(np.sum(-(w / 4) * (
            y * np.log(p) + (1 - y) * np.log1p(-p))))
        self.assertEqual(trace["input_weight_sum"], 0.25)
        self.assertAlmostEqual(hand, trace["terminal_weighted_bce"], places=6)

    def test_roundoff_admission_is_bounded_recorded_and_not_rescaled(self):
        x = np.zeros((1, 2), dtype=np.float32)
        y = np.zeros(1, dtype=np.float32)
        w = np.array([np.nextafter(np.float32(1), np.float32(2))])
        original = w.copy()
        # Inspect the actual prepared weights, not merely the reported total.
        prepared = models._training_arrays(x, y, w, None)
        np.testing.assert_array_equal(prepared[2], original)
        _, trace = models.fit(x, y, w, np.zeros(1, dtype=bool))
        self.assertTrue(trace["weight_sum_roundoff_admitted"])
        self.assertEqual(trace["input_weight_sum"], float(w[0]))
        self.assertEqual(trace["input_weight_excess_over_one"],
                         float(np.finfo(np.float32).eps))
        self.assertEqual(trace["config"]["weight_sum_roundoff_tolerance"],
                         16 * np.finfo(np.float32).eps)
        self.assertFalse(trace["config"]["weight_renormalization"])
        np.testing.assert_array_equal(w, original)
        excessive = np.array([1 + 32 * np.finfo(np.float32).eps], dtype=np.float32)
        with self.assertRaises(models.FitError):
            models.fit(x, y, excessive, np.zeros(1, dtype=bool))

    def test_invalid_inputs_and_partial_fit_error_contract(self):
        x, y, w = toy()
        bad_x = x.copy()
        bad_x[0, 0] = np.nan
        invalid = [(bad_x, y, w, None), (x.astype(np.float64), y, w, None),
                   (x, y[:, None], w, None), (x, y + 2, w, None),
                   (x, y, -w, None), (x, y, w * 2, None),
                   (x, y, w, np.ones(len(x), dtype=np.int64))]
        for arguments in invalid:
            with self.assertRaises(models.FitError) as caught:
                models.fit(*arguments)
            self.assertEqual(caught.exception.code, "invalid_input")
            self.assertEqual(caught.exception.trace["status"], "failed")
            self.assertIsNone(caught.exception.model)
            json.dumps(caught.exception.trace, allow_nan=False)

    def test_serialized_network_formula_and_reject_partial_or_nonfinite(self):
        model = {
            "schema": "evr0037.mlp.v1", "kind": "mlp", "input_dim": 1,
            "config": dict(models.CONFIG), "terminal_epoch": 20,
            "constant_probability": None,
            "state_dict": {"0.weight": [[1.0]] + [[0.0]] * 31,
                           "0.bias": [0.0] * 32,
                           "2.weight": [[2.0] + [0.0] * 31],
                           "2.bias": [-1.0]},
        }
        x = np.array([[-1], [0], [1]], dtype=np.float32)
        expected = 1 / (1 + np.exp(-(2 * np.maximum(x[:, 0], 0) - 1)))
        import torch
        rng_before = torch.random.get_rng_state().clone()
        np.testing.assert_allclose(models.predict(model, x), expected, atol=1e-7)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        partial = dict(model, kind="partial_mlp")
        with self.assertRaises(ValueError):
            models.predict(partial, x)
        model["state_dict"]["2.bias"] = [float("nan")]
        with self.assertRaises(ValueError):
            models.predict(model, x)


if __name__ == "__main__":
    unittest.main()
