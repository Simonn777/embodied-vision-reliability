"""Manufactured arithmetic tests only; no dataset or native-model loading."""

import unittest

import torch
from torch.nn import functional as F

import head


class HeadTests(unittest.TestCase):
    def make_head(self, *, zero_features=False, mean=None, std=None):
        torch.manual_seed(0)
        return head.NativeFeatureHead(
            torch.zeros(32) if mean is None else mean,
            torch.ones(32) if std is None else std,
            zero_features=zero_features,
        )

    def observations(self, batch=1):
        generator = torch.Generator().manual_seed(17)
        features = torch.randn(batch, 32, 256, 256, generator=generator)
        native = torch.randn(batch, 1, 256, 256, generator=generator)
        return features, native

    def test_initial_exact_identity_and_same_arm_initial_state(self):
        visual = self.make_head()
        zero = self.make_head(zero_features=True)
        self.assertEqual(sum(p.numel() for p in visual.parameters()), 1377)
        for key, value in visual.state_dict().items():
            self.assertTrue(torch.equal(value, zero.state_dict()[key]), key)
        features, native = self.observations(batch=2)
        self.assertTrue(torch.equal(visual(features, native), native))
        self.assertTrue(torch.equal(zero(features, native), native))
        self.assertEqual(visual(features, native).dtype, torch.float32)

    def test_head_gradients_and_detached_native_observations(self):
        model = self.make_head()
        features, native = self.observations()
        features.requires_grad_()
        native.requires_grad_()
        target = torch.zeros_like(native)
        loss = F.binary_cross_entropy_with_logits(model(features, native), target)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(native.grad)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        self.assertGreater(float(model.conv2.bias.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.conv2.weight.grad.abs().sum()), 0.0)
        # A zero final weight intentionally blocks the first layer at step zero.
        self.assertEqual(float(model.conv1.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(model.conv1.bias.grad.abs().sum()), 0.0)
        with torch.no_grad():
            model.conv2.weight.add_(-0.01 * model.conv2.weight.grad)
            model.conv2.bias.add_(-0.01 * model.conv2.bias.grad)
        model.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(features, native), target).backward()
        self.assertGreater(float(model.conv1.weight.grad.abs().sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(model.conv1.weight.grad).all()))
        self.assertIsNone(features.grad)
        self.assertIsNone(native.grad)

    def test_fixed_train_statistics_and_standard_deviation_floor(self):
        mean = torch.arange(32, dtype=torch.float32)
        std = torch.full((32,), 2.0)
        std[0] = 0.0
        model = self.make_head(mean=mean, std=std)
        features = mean.reshape(1, 32, 1, 1).expand(1, 32, 256, 256).clone()
        features[:, 1:] += 4.0
        features[:, 0] += 1e-6
        native = torch.full((1, 1, 256, 256), -2.0)
        seen = []
        handle = model.conv1.register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].detach().clone())
        )
        try:
            model(features, native)
        finally:
            handle.remove()
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0][:, 0], torch.ones(1, 256, 256)))
        self.assertTrue(torch.equal(seen[0][:, 1:32], torch.full((1, 31, 256, 256), 2.0)))
        self.assertTrue(torch.equal(seen[0][:, 32:], torch.sigmoid(native)))
        self.assertFalse(model.train_mean.requires_grad)
        self.assertFalse(model.train_std.requires_grad)

    def test_bounded_signed_correction(self):
        model = self.make_head()
        features, _ = self.observations()
        native = torch.zeros(1, 1, 256, 256)
        for sign in (-1.0, 1.0):
            with torch.no_grad():
                model.conv2.bias.fill_(sign * 100.0)
            output = model(features, native)
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertTrue(bool((output.abs() <= 4.0).all()))
            self.assertTrue(torch.equal(output, torch.full_like(output, sign * 4.0)))

    def test_zero_features_remain_independent_after_nonzero_final_weights(self):
        model = self.make_head(zero_features=True)
        features, native = self.observations()
        with torch.no_grad():
            model.conv2.weight.fill_(0.002)
            model.conv2.bias.fill_(0.1)
        first = model(features, native)
        second = model(features * 7.0 + 13.0, native)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, native))

    def test_original_resize_identity_and_independent_half_pixel_reference(self):
        model = self.make_head()
        features, native = self.observations()
        expected = F.interpolate(native, size=(321, 479), mode="bilinear", align_corners=False)
        actual = head.original_logits(model(features, native), (321, 479))
        self.assertTrue(torch.equal(actual, expected))
        # Affine coordinate ramp has an analytic bilinear interpolation value.
        yy = torch.arange(256, dtype=torch.float32).reshape(256, 1)
        xx = torch.arange(256, dtype=torch.float32).reshape(1, 256)
        ramp = (yy + 2.0 * xx).reshape(1, 1, 256, 256)
        resized = head.original_logits(ramp, (17, 23))[0, 0]
        oy = ((torch.arange(17, dtype=torch.float64) + 0.5) * 256.0 / 17.0 - 0.5)
        ox = ((torch.arange(23, dtype=torch.float64) + 0.5) * 256.0 / 23.0 - 0.5)
        reference = oy[:, None] + 2.0 * ox[None, :]
        # Float32 interpolation of values <=765: eight ulps bound this check.
        self.assertLessEqual(
            float((resized.double() - reference).abs().max()),
            8.0 * torch.finfo(torch.float32).eps * 765.0,
        )

    def test_invalid_statistics_and_observation_shapes(self):
        with self.assertRaises(ValueError):
            self.make_head(mean=torch.zeros(31))
        with self.assertRaises(ValueError):
            self.make_head(std=-torch.ones(32))
        with self.assertRaises(ValueError):
            self.make_head(mean=torch.full((32,), float("nan")))
        model = self.make_head()
        features, native = self.observations()
        with self.assertRaises(ValueError):
            model(features[:, :31], native)
        with self.assertRaises(TypeError):
            model(features.double(), native)
        with self.assertRaises(ValueError):
            model(features, native.expand(2, -1, -1, -1))
        with self.assertRaises(ValueError):
            head.original_logits(native, (0, 480))


if __name__ == "__main__":
    unittest.main()
