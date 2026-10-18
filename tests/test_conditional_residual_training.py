"""Own tiny fixtures only. No model assets/data downloads or scientific training."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import conditional_residual_training as core

HAS_TORCH = importlib.util.find_spec("torch") is not None


class NumpyInterfaceTests(unittest.TestCase):
    def test_valid_centers_order(self):
        image = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        prep = core._prepare_image(image, np.eye(49, dtype=np.float64))
        self.assertEqual(prep["patches"].shape, (14884, 49))
        np.testing.assert_array_equal(prep["patches"][0], image[:7, :7].reshape(49))
        self.assertEqual(prep["patches"][1, 24], image[3, 4])
        self.assertEqual(prep["patches"][-1, 24], image[124, 124])

    def test_residual_is_per_target_and_center_zero(self):
        patches = (np.arange(196, dtype=np.float32).reshape(4, 49) - 40) / 100
        gains = np.linspace(-0.2, 0.4, 49, dtype=np.float32)
        gains[24] = 1.0
        result = core._fixed_features(patches, gains, "residual")
        expected = patches - gains[None, :] * patches[:, 24, None]
        expected[:, 24] = 0
        np.testing.assert_array_equal(result, expected)
        changed = patches.copy()
        changed[2] += 10
        other = core._fixed_features(changed, gains, "residual")
        np.testing.assert_array_equal(result[[0, 1, 3]], other[[0, 1, 3]])
        self.assertTrue(np.all(result[:, 24] == 0))

    def test_white_noise_residual_equals_blindspot(self):
        patches = np.arange(196, dtype=np.float32).reshape(4, 49)
        gains = np.zeros(49, dtype=np.float32)
        gains[24] = 1
        np.testing.assert_array_equal(core._fixed_features(patches, gains, "residual"),
                                      core._fixed_features(patches, gains, "blindspot"))
        np.testing.assert_array_equal(core._fixed_features(patches, gains, "r2r_alpha05"), patches)
        np.testing.assert_array_equal(core._fixed_features(patches, gains, "safe_blindspot", gains != 0),
                                      core._fixed_features(patches, gains, "blindspot"))

    def test_safe_blindspot_uses_supplied_exact_covariance_mask(self):
        p = np.arange(1, 197, dtype=np.float32).reshape(4, 49)
        gains = np.zeros(49, dtype=np.float32)
        gains[24] = 1.0
        mask = np.zeros((7, 7), dtype=bool)
        mask[1:6, 1:6] = True
        result = core._fixed_features(p, gains, "safe_blindspot", mask.reshape(49))
        self.assertEqual(np.count_nonzero(result[0]), 24)
        self.assertTrue(np.all(result[:, mask.reshape(49)] == 0))
        np.testing.assert_array_equal(result[:, ~mask.reshape(49)], p[:, ~mask.reshape(49)])
        # A value that rounds to zero as a gain must still be masked when C says nonzero.
        c = np.eye(49, dtype=np.float64)
        c[0, 24] = c[24, 0] = 1e-80
        prepared = core._prepare_image(np.zeros((128, 128), dtype=np.float32), c)
        self.assertEqual(prepared["gains"][0], 0)
        self.assertTrue(prepared["correlated_mask"][0])
        result = core._fixed_features(p, prepared["gains"], "safe_blindspot", prepared["correlated_mask"])
        self.assertTrue(np.all(result[:, 0] == 0))

    def test_safe_blindspot_rejects_missing_or_wrong_mask(self):
        p = np.ones((4, 49), dtype=np.float32)
        g = np.zeros(49, dtype=np.float32)
        for mask in (None, np.zeros(49, dtype=bool), np.ones(49, dtype=np.float32)):
            with self.assertRaises(ValueError):
                core._fixed_features(p, g, "safe_blindspot", mask)

    def test_reject_bad_inputs_and_no_covariance_repair(self):
        good = np.zeros((128, 128), dtype=np.float32)
        c = np.eye(49, dtype=np.float64)
        for image in (good[:-1], good.astype(np.float64), np.full_like(good, np.nan)):
            with self.assertRaises(ValueError):
                core._prepare_image(image, c)
        for bad in (c.astype(np.float32), c[:48], np.zeros_like(c)):
            with self.assertRaises(ValueError):
                core._prepare_image(good, bad)
        c[0, 1] = 0.2
        with self.assertRaises(ValueError):
            core._prepare_image(good, c)

    def test_public_science_settings_cannot_be_shortened(self):
        for name, value in (("steps", 2), ("batch_size", 4), ("mc_samples", 2),
                            ("max_fit_seconds", 91), ("learning_rate", 0.1),
                            ("inference_chunk_size", 2), ("steps", True), ("steps", 500.0)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                core.fit_one(np.zeros((128, 128), dtype=np.float32), np.eye(49), "blindspot",
                             init_seed=1, batch_seed=2, augment_seed=3, infer_seed=4,
                             **{name: value})

    def test_seed_validation(self):
        for invalid in (True, -1, 2**63, 1.0):
            with self.assertRaises(ValueError):
                core._seed(invalid, "seed")

    def test_state_hash_order_invariance_and_value_binding(self):
        a = {"0.weight": np.ones((2, 3), dtype=np.float32),
             "0.bias": np.zeros(2, dtype=np.float32)}
        b = dict(reversed(list(a.items())))
        self.assertEqual(core.state_dict_sha256(a), core.state_dict_sha256(b))
        b["0.weight"] = b["0.weight"].copy()
        b["0.weight"][0, 0] = 2
        self.assertNotEqual(core.state_dict_sha256(a), core.state_dict_sha256(b))


@unittest.skipUnless(HAS_TORCH, "Torch is not installed; CPU tensor fixture checks skipped")
class TinyTorchInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = core._torch_runtime()

    def fixture(self, method):
        p = (np.arange(196, dtype=np.float32).reshape(4, 49) - 90) / 200
        gains = np.zeros(49, dtype=np.float32)
        gains[24] = 1
        f = core._fixed_features(p, gains, method, gains != 0)
        t = self.torch
        return t.from_numpy(p), t.from_numpy(f), t.eye(49, dtype=t.float32) * 0.1

    def fit_fixture(self, method):
        p, f, factor = self.fixture(method)
        return core._fit_patches(self.torch, p, f, factor, method, init_seed=11,
                                 batch_seed=12, augment_seed=13, steps=2, batch_size=4,
                                 learning_rate=1e-3, max_fit_seconds=90.0)

    def test_same_initialization_and_batch_stream(self):
        records = [self.fit_fixture(m)[1] for m in core.METHODS]
        self.assertEqual(len({r["initial_weight_sha256"] for r in records}), 1)
        self.assertEqual(len({r["batch_indices_sha256"] for r in records}), 1)
        self.assertTrue(all(r["completed_steps"] == 2 and r["parameter_count"] == 5313 for r in records))
        self.assertTrue(all(r["status"] == "completed" for r in records))
        self.assertEqual(records[2]["augmentation_draws_sha256"], records[3]["augmentation_draws_sha256"])
        self.assertEqual(records[2]["augmentation_standard_normal_draws"], 2 * 4 * 49)
        self.assertEqual(records[0]["weight_sha256"], records[1]["weight_sha256"])
        self.assertEqual(records[0]["weight_sha256"], records[4]["weight_sha256"])

    def test_r2r_pair_uses_same_noise_with_reciprocal_negative_target(self):
        t = self.torch
        captured_inputs, captured_targets = [], []
        original_new_model = core._new_model
        original_loss = t.nn.functional.mse_loss
        def capture_model(*args):
            model = original_new_model(*args)
            model.register_forward_pre_hook(lambda _, inputs: captured_inputs.append(inputs[0].detach().clone()))
            return model
        def capture_loss(prediction, target, **kwargs):
            captured_targets.append(target.detach().clone())
            return original_loss(prediction, target, **kwargs)
        with mock.patch.object(core, "_new_model", capture_model), \
             mock.patch.object(t.nn.functional, "mse_loss", capture_loss):
            self.fit_fixture("r2r_alpha05")
        p, _, factor = self.fixture("r2r_alpha05")
        indices = t.randint(4, (4,), generator=t.Generator().manual_seed(12))
        eps = t.randn((4, 49), generator=t.Generator().manual_seed(13))
        perturbation = eps @ factor.T
        t.testing.assert_close(captured_inputs[0], p[indices] + 0.5 * perturbation)
        t.testing.assert_close(captured_targets[0], p[indices, 24] - 2.0 * perturbation[:, 24])

    def test_local_repeatability(self):
        a = self.fit_fixture("r2r_alpha05")[1]
        b = self.fit_fixture("r2r_alpha05")[1]
        self.assertEqual(a["weight_sha256"], b["weight_sha256"])
        self.assertEqual(a["batch_indices_sha256"], b["batch_indices_sha256"])

    def test_timeout_never_completed(self):
        p, f, factor = self.fixture("blindspot")
        ticks = iter([0.0, 91.0, 92.0])
        _, record = core._fit_patches(self.torch, p, f, factor, "blindspot", init_seed=11,
                                     batch_seed=12, augment_seed=13, steps=2, batch_size=4,
                                     learning_rate=1e-3, max_fit_seconds=90.0,
                                     clock=lambda: next(ticks))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_reason"], "fit_time_cap")
        self.assertEqual(record["completed_steps"], 0)

    def test_overbudget_final_step_is_still_failed(self):
        p, f, factor = self.fixture("blindspot")
        ticks = iter([0.0, 0.0, 0.0, 0.0, 91.0, 92.0])
        _, record = core._fit_patches(self.torch, p, f, factor, "blindspot", init_seed=11,
                                     batch_seed=12, augment_seed=13, steps=2, batch_size=4,
                                     learning_rate=1e-3, max_fit_seconds=90.0,
                                     clock=lambda: next(ticks))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_reason"], "fit_time_cap")
        self.assertEqual(record["completed_steps"], 2)

    def test_r2r_mc50_and_direct_count_no_output_correction(self):
        t = self.torch
        class ConstantModel(t.nn.Module):
            def forward(self, x):
                return t.full((x.shape[0], 1), 0.25, dtype=t.float32, device="cpu")
        _, f, factor = self.fixture("r2r_alpha05")
        primary, direct, record = core._predict(t, ConstantModel(), f, factor, "r2r_alpha05",
                                               infer_seed=14, chunk_size=3, mc_samples=50)
        np.testing.assert_array_equal(primary, np.full(4, 0.25, dtype=np.float32))
        np.testing.assert_array_equal(direct, primary)
        self.assertEqual(record["primary_forward_calls"], 100)
        self.assertEqual(record["direct_forward_calls"], 2)
        self.assertEqual(record["standard_normal_draws"], 4 * 50 * 49)


if __name__ == "__main__":
    unittest.main(verbosity=2)
