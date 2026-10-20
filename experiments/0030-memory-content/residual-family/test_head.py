"""Source-prepared manufactured head checks; no native model or real data."""
import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import family
import head as campaign_head


class CausalHeadChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(4)

    def inputs(self):
        t = self.torch
        memory = t.tensor([-1., 1., -1., 1.]).reshape(1, 1, 2, 2).repeat(1, 64, 1, 1)
        memory[:, 1] = 5.
        memory[:, 2] *= 2.
        context = family.prepare(memory, .1, torch=t)
        mask = t.tensor([[[[0., .5], [1., .7]]]])
        return context, mask

    def test_dynamic_source_import_has_no_dependencies_or_side_effects(self):
        path = Path(campaign_head.__file__)
        spec = importlib.util.spec_from_file_location("isolated_causal_head", path)
        module = importlib.util.module_from_spec(spec)
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "numpy", "family", "sam2"):
                raise AssertionError("dependency imported while loading head source")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            spec.loader.exec_module(module)
        self.assertEqual(module.configuration()["input_features"], 131)
        self.assertTrue(callable(module.predict_coefficients))

    def test_exact_feature_order_and_detached_causal_statistics(self):
        t = self.torch
        context, mask = self.inputs()
        features = campaign_head.extract_input(context, t.tensor(0., requires_grad=True),
            t.tensor(.75, requires_grad=True), mask.requires_grad_(True), torch=t)
        self.assertEqual(tuple(features.shape), (1, 131))
        self.assertFalse(features.requires_grad)
        self.assertAlmostEqual(features[0, 1].item(), t.tensor(6.).log().item(), places=6)
        self.assertAlmostEqual(features[0, 64].item(), t.tensor(2.).log().item(), places=6)
        self.assertEqual(features[0, 65].item(), 0.)
        self.assertTrue(t.equal(features[0, -3:], t.tensor([.5, .75, .5])))
        with self.assertRaises(TypeError):
            campaign_head.extract_input(context, 0., .75, mask, torch=t, future_mask=mask)

    def test_architecture_and_zero_initialization_gradient(self):
        t = self.torch
        context, mask = self.inputs()
        model = campaign_head.build_head(torch=t)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 33408)
        self.assertEqual([type(m).__name__ for m in model], ["Linear", "GELU", "Linear"])
        output = campaign_head.predict_coefficients(model, context, 0., .75, mask, torch=t, family=family)
        self.assertTrue(t.equal(output["coefficients"], t.zeros((64, 2))))
        output["coefficients"].sum().backward()
        self.assertTrue(all(t.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model[-1].bias.grad.abs().max().item(), 0.)
        self.assertEqual(model[0].weight.grad.abs().max().item(), 0.)
        self.assertTrue(t.equal(model[-1].bias.grad.reshape(64, 2)[1], t.zeros(2)))

    def test_channel_variation_scaling_and_inactive_coordinates(self):
        t = self.torch
        context, mask = self.inputs()
        model = campaign_head.build_head(torch=t)
        with t.no_grad():
            model[-1].bias.copy_(t.tensor([.25, .5]).repeat(64))
        output = campaign_head.predict_coefficients(model, context, -1., .7, mask, torch=t, family=family)
        expected = context["sigma"].float()[:, None] * .1 * t.tensor([[.25, .5]])
        self.assertTrue(t.allclose(output["coefficients"], expected, atol=2e-7, rtol=2e-6))
        self.assertTrue(t.equal(output["coefficients"][1], t.zeros(2)))
        self.assertTrue(t.allclose(output["coefficients"][2], 2*output["coefficients"][0]))

    def test_nonzero_output_obeys_shared_energy_bound(self):
        t = self.torch
        context, mask = self.inputs()
        model = campaign_head.build_head(torch=t)
        with t.no_grad():
            model[-1].bias.copy_(t.tensor([3., -4.]).repeat(64))
        output = campaign_head.predict_coefficients(model, context, 2., .8, mask, torch=t, family=family)
        report = family.describe(context, output["coefficients"], torch=t)
        self.assertLessEqual(report["actual_energy"], context["radius"] ** 2 * (1+2e-6))
        weights = t.arange(128).reshape(64, 2).float()
        (output["coefficients"] * weights).sum().backward()
        self.assertTrue(all(t.isfinite(p.grad).all() for p in model.parameters()))

    def test_input_rejection_and_large_finite_statistics(self):
        t = self.torch
        context, mask = self.inputs()
        for value in (float("nan"), float("inf"), complex(1, 2)):
            with self.assertRaises(ValueError):
                campaign_head.extract_input(context, value, .5, mask, torch=t)
        with self.assertRaises(ValueError):
            campaign_head.extract_input(context, 0., .5, mask + 1., torch=t)
        with self.assertRaises(ValueError):
            campaign_head.extract_input(context, 0., float("inf"), mask, torch=t)
        huge = t.tensor([-1e30, 1e30, -1e30, 1e30]).reshape(1, 1, 2, 2).repeat(1, 64, 1, 1)
        huge_context = family.prepare(huge, .1, torch=t)
        features = campaign_head.extract_input(huge_context, -1e30, .8, mask, torch=t)
        self.assertTrue(t.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
