"""Only generated arrays: no images on disk, HDF5, models, or pose labels."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

_spec = importlib.util.spec_from_file_location("refine_patch", Path(__file__).with_name("refine_patch.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


def texture(x, y):
    return (0.45 + 0.11*np.sin(0.19*x+0.11*y) +
            0.09*np.cos(0.14*x-0.23*y) + 0.07*np.sin(0.31*x+0.27*y) +
            0.04*np.cos(0.08*x+0.17*y))


def artificial():
    y, x = np.indices((128, 128), dtype=np.float64)
    image1 = texture(x, y)
    A = np.array([[1.01, 0.012], [-0.008, 0.99]])
    c = np.array([0.3, -0.2])
    gain, bias = 1.12, -0.03
    inv = np.linalg.inv(A)
    source = np.einsum("ij,hwj->hwi", inv, np.stack((x-c[0], y-c[1]), axis=-1), optimize=False)
    image2 = gain*texture(source[..., 0], source[..., 1])+bias
    p = np.array([[50.23, 52.37], [64.19, 65.31], [77.27, 70.41]])
    truth = np.einsum("ij,bj->bi", A, p, optimize=False)+c
    q = truth-np.array([0.43, -0.36])
    return image1, image2, p, q, A, truth, gain, bias


class PatchTests(unittest.TestCase):
    def test_bilinear_derivative_is_same_interpolant(self):
        image = np.random.default_rng(10).uniform(size=(20, 23))
        xy = np.array([[[4.17, 6.28], [8.34, 9.41], [18.32, 15.67]]])
        _, derivative = rp._bilinear(image, xy)
        for j in range(2):
            d = np.zeros_like(xy)
            d[..., j] = 1e-6
            plus, _ = rp._bilinear(image, xy+d)
            minus, _ = rp._bilinear(image, xy-d)
            np.testing.assert_allclose((plus-minus)/2e-6, derivative[..., j], atol=1e-9, rtol=1e-8)

    def test_eight_parameter_jacobian(self):
        image1, image2, p, q, _, _, _, _ = artificial()
        theta = rp.initial_parameters(np.eye(2))
        theta[:2] = [0.13, 0.17]
        theta[6:] = [1.03, 0.01]
        r, J = rp.residuals_and_jacobian(image1, image2, p[0], q[0], theta)
        self.assertEqual(J.shape, (961, 8))
        for j in range(8):
            d = np.zeros(8)
            d[j] = 1e-6
            plus, _ = rp.residuals_and_jacobian(image1, image2, p[0], q[0], theta+d)
            minus, _ = rp.residuals_and_jacobian(image1, image2, p[0], q[0], theta-d)
            np.testing.assert_allclose((plus-minus)/2e-6, J[:, j], atol=4e-9, rtol=2e-6)
        self.assertTrue(np.isfinite(r).all())

    def test_textured_subpixel_affine_and_radiometry(self):
        i1, i2, p, q, A, truth, gain, bias = artificial()
        result = rp.refine_batch(i1, i2, p, q, np.repeat(np.eye(2)[None], len(p), axis=0))
        json.dumps(result, allow_nan=False)
        for index, record in enumerate(result["patches"]):
            self.assertTrue(record["admitted"], record)
            self.assertTrue(record["changed"])
            self.assertLess(record["final_cost"], record["initial_cost"]*0.01)
            np.testing.assert_allclose(record["final"]["target"], truth[index], atol=0.035)
            np.testing.assert_allclose(record["final"]["A"], A, atol=0.003)
            self.assertAlmostEqual(record["final"]["gain"], gain, delta=0.012)
            self.assertAlmostEqual(record["final"]["bias"], bias, delta=0.006)
            accepted_costs = [record["initial_cost"]]+[a["cost_after"] for a in record["attempts"] if a["accepted"]]
            self.assertTrue(all(a > b for a, b in zip(accepted_costs, accepted_costs[1:])))
            self.assertLessEqual(len(record["attempts"]), 30)
            self.assertLessEqual(record["final_diagnostic"]["relative_gn_decrease"], 1e-8)

    def test_batch_single_parity(self):
        i1, i2, p, q, _, _, _, _ = artificial()
        A0 = np.repeat(np.eye(2)[None], len(p), axis=0)
        batch = rp.refine_batch(i1, i2, p, q, A0)["patches"]
        for i in range(len(p)):
            single = rp.refine_batch(i1, i2, p[i:i+1], q[i:i+1], A0[i:i+1])["patches"][0]
            self.assertEqual(batch[i]["admitted"], single["admitted"])
            self.assertEqual(batch[i]["termination"], single["termination"])
            self.assertEqual(batch[i]["attempts"], single["attempts"])
            self.assertEqual(batch[i]["final"], single["final"])

    def test_zero_update_valid_patch_is_admitted(self):
        i1, _, p, _, _, _, _, _ = artificial()
        record = rp.refine_batch(i1, i1, p[:1], p[:1], np.eye(2)[None])["patches"][0]
        self.assertTrue(record["admitted"], record)
        self.assertFalse(record["changed"])
        self.assertEqual(record["accepted_steps"], 0)
        self.assertEqual(record["termination"], "gradient_tolerance")
        self.assertEqual(record["initial_cost"], 0.0)
        self.assertEqual(record["final_cost"], 0.0)
        self.assertTrue(record["initial_diagnostic"]["valid"])

    def test_flat_patch_not_admitted(self):
        image = np.full((80, 80), 0.4)
        record = rp.refine_batch(image, image, [[40, 40]], [[40, 40]], np.eye(2)[None])["patches"][0]
        self.assertFalse(record["admitted"])
        self.assertFalse(record["final_diagnostic"]["valid"])
        self.assertEqual(record["accepted_steps"], 0)
        json.dumps(record, allow_nan=False)

    def test_initialization_failures_preserve_other_patch(self):
        i1, i2, p, q, _, _, _, _ = artificial()
        p[1] = [2, 2]
        A0 = np.repeat(np.eye(2)[None], 3, axis=0)
        A0[2, 0, 0] = -1
        rows = rp.refine_batch(i1, i2, p, q, A0)["patches"]
        self.assertTrue(rows[0]["admitted"])
        self.assertEqual(rows[1]["status"], "initialization_error")
        self.assertEqual(rows[2]["status"], "initialization_error")
        self.assertIsNone(rows[1]["final"])
        self.assertEqual(rows[2]["original_A"], A0[2].tolist())
        json.dumps(rows, allow_nan=False)

    def test_all_state_guards(self):
        i1, _, p, _, _, _, _, _ = artificial()
        q, A0 = p[0], np.eye(2)
        base = rp.initial_parameters(A0)
        cases = []
        for j, value, reason in [(6, 4.01, "gain_guard"), (0, 4.01, "displacement_guard"),
                                  (2, -15, "affine_determinant_guard"),
                                  (2, 31, "relative_affine_guard")]:
            t = base.copy()
            t[j] = value
            cases.append((t, reason))
        for theta, reason in cases:
            self.assertEqual(rp._guard(i1, q, theta, A0), reason)
        self.assertEqual(rp._guard(i1, np.array([4, 4]), base, A0), "target_patch_bounds")

    def test_constraints_reject_trials_without_silent_output_fallback(self):
        y, x = np.indices((128, 128), dtype=np.float64)
        i1 = texture(x, y)
        i2 = texture(x-7, y)
        row = rp.refine_batch(i1, i2, [[64.2, 64.3]], [[64.2, 64.3]], np.eye(2)[None])["patches"][0]
        self.assertTrue(any(a["status"] == "displacement_guard" for a in row["attempts"]), row)
        self.assertLessEqual(np.linalg.norm(row["final"]["delta"]), 4)
        self.assertFalse(row["admitted"])
        self.assertNotEqual(row["final"]["delta"], [0, 0])
        json.dumps(row, allow_nan=False)

    def test_true_nonfinite_and_overflow_refused(self):
        i1, i2, p, q, _, _, _, _ = artificial()
        for bad in (np.nan, np.inf, 1e308, -0.1):
            corrupted = i1.copy()
            corrupted[0, 0] = bad
            with self.assertRaises(ValueError):
                rp.refine_batch(corrupted, i2, p[:1], q[:1], np.eye(2)[None])
        with self.assertRaises(ValueError):
            rp.initial_parameters(np.full((2, 2), 1e308))
        huge = np.full(8, 1e308)
        with self.assertRaises(ValueError):
            rp.residuals_and_jacobian(i1, i2, p[0], q[0], huge)

    def test_maximum_batch_and_input_limit(self):
        i1, _, p, _, _, _, _, _ = artificial()
        points = np.repeat(p[:1], 64, axis=0)
        A0 = np.repeat(np.eye(2)[None], 64, axis=0)
        rows = rp.refine_batch(i1, i1, points, points, A0)["patches"]
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(row["admitted"] and not row["changed"] for row in rows))
        with self.assertRaises(ValueError):
            rp.refine_batch(i1, i1, np.repeat(p[:1], 65, axis=0), np.repeat(p[:1], 65, axis=0), np.repeat(np.eye(2)[None], 65, axis=0))


if __name__ == "__main__":
    unittest.main()
