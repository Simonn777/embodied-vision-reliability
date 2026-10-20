"""Manufactured arithmetic/autograd checks only; parent owns first execution."""
import unittest
import family


class ResidualFamilyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy
        import torch
        cls.np, cls.torch = numpy, torch
        torch.set_num_threads(4)

    def context(self, rho=0.1):
        t = self.torch
        memory = t.tensor([[[[1., 3.], [5., 7.]], [[6., 6.], [6., 6.]]]])
        return family.prepare(memory, rho, torch=t)

    def test_statistics_and_inactive_channel(self):
        t = self.torch
        memory = t.tensor([[[[1., 3.], [5., 7.]], [[6., 6.], [6., 6.]]]],
                          requires_grad=True)
        c = family.prepare(memory, .1, torch=t)
        self.assertEqual(c["active"].tolist(), [True, False])
        self.assertAlmostEqual(c["sigma"][0].item() ** 2, 5., places=12)
        self.assertFalse(any(c[k].requires_grad for k in ("basis", "mean", "sigma", "gram")))
        q = t.tensor([[.01, -.02], [3., 4.]])
        self.assertTrue(t.equal(family.expand(c, q, torch=t)[0, 1], t.zeros((2, 2))))
        self.assertTrue(t.equal(family.project(c, q, torch=t)[1], t.zeros(2)))

    def test_zero_projection_has_identity_active_gradient(self):
        t = self.torch
        c = self.context()
        q = t.zeros((2, 2), requires_grad=True)
        bounded = family.project(c, q, torch=t)
        bounded.sum().backward()
        self.assertTrue(t.isfinite(q.grad).all())
        self.assertTrue(t.equal(q.grad, t.tensor([[1., 1.], [0., 0.]])))
        self.assertTrue(t.equal(bounded, t.zeros_like(q)))

    def test_zero_radius_and_all_constant_are_safe(self):
        t = self.torch
        for c in (self.context(0), family.prepare(t.ones((1, 2, 2, 2)), .1, torch=t)):
            q = t.ones((2, 2), requires_grad=True)
            bounded = family.project(c, q, torch=t)
            bounded.sum().backward()
            self.assertTrue(t.equal(bounded, t.zeros_like(q)))
            self.assertTrue(t.equal(q.grad, t.zeros_like(q)))
            self.assertEqual(family.describe(c, bounded, torch=t)["actual_energy"], 0.)

    def test_projection_energy_and_saturated_gradient(self):
        t = self.torch
        c = self.context(.1)
        q = t.tensor([[3., 4.], [9., -2.]], requires_grad=True)
        bounded = family.project(c, q, torch=t)
        delta = family.expand(c, bounded, torch=t)
        info = family.describe(c, bounded, torch=t)
        self.assertLessEqual(info["actual_energy"], c["radius"] ** 2 * (1 + 2e-6))
        self.assertAlmostEqual(info["gram_energy"], c["radius"] ** 2, delta=1e-7)
        (delta * t.tensor([[[[1., -2.], [3., -4.]], [[0., 0.], [0., 0.]]]])).sum().backward()
        self.assertTrue(t.isfinite(q.grad).all())
        self.assertGreater(q.grad[0].abs().max().item(), 0)
        self.assertTrue(t.equal(q.grad[1], t.zeros(2)))

    def test_unbounded_target_matches_independent_numpy_lstsq(self):
        t, np = self.torch, self.np
        memory = t.tensor([[[[1., 2., 5.], [3., 8., 13.]],
                            [[-7., 1., 2.], [9., 4., -3.]],
                            [[4., 4., 4.], [4., 4., 4.]]]])
        target = t.tensor([[[[.2, -.5, .4], [.1, .8, -.2]],
                            [[.8, .3, -.1], [-.4, .7, .5]],
                            [[8., 3., 5.], [4., 2., 9.]]]])
        c = family.prepare(memory, 100, torch=t)
        answer = family.least_squares_target(c, target, torch=t)
        for channel in (0, 1):
            design = np.column_stack((c["basis"][channel].double().numpy(), np.ones(6)))
            expected = np.linalg.lstsq(design, target[0, channel].double().numpy().ravel(), rcond=None)[0]
            np.testing.assert_allclose(answer["coefficients"][channel].numpy(), expected, atol=2e-7, rtol=2e-6)
        self.assertTrue(t.equal(answer["coefficients"][2], t.zeros(2)))

    def test_constrained_ls_matches_measured_metric_formula(self):
        t, np = self.torch, self.np
        c = self.context(.02)
        target = t.tensor([[[[1., -3.], [5., 9.]], [[2., 4.], [3., 1.]]]])
        design = np.column_stack((c["basis"][0].double().numpy(), np.ones(4)))
        q = np.linalg.lstsq(design, target[0, 0].double().numpy().ravel(), rcond=None)[0]
        gram = design.T @ design / 4
        expected = q * min(1., c["radius"] / float(np.sqrt(q @ gram @ q)))
        answer = family.least_squares_target(c, target, torch=t)
        np.testing.assert_allclose(answer["coefficients"][0].numpy(), expected, atol=2e-7, rtol=2e-6)
        self.assertLessEqual(answer["diagnostics"]["actual_energy"], c["radius"] ** 2 * (1 + 2e-6))

    def test_family_reconstruction_and_target_detachment(self):
        t = self.torch
        c = self.context(10)
        coefficients = t.tensor([[.15, -.32], [0., 0.]])
        target = family.expand(c, coefficients, torch=t).requires_grad_(True)
        answer = family.least_squares_target(c, target, torch=t)
        self.assertTrue(t.allclose(answer["coefficients"], coefficients, atol=2e-7, rtol=2e-6))
        self.assertFalse(answer["coefficients"].requires_grad)
        info = answer["diagnostics"]
        self.assertAlmostEqual(info["actual_energy"], info["gram_energy"], delta=1e-7)

    def test_invalid_inputs_fail_explicitly(self):
        t = self.torch
        for rho in (-1., float("nan"), float("inf"), True, 1e-300):
            with self.assertRaises(ValueError):
                self.context(rho)
        with self.assertRaises(ValueError):
            family.prepare(t.full((1, 1, 2, 2), float("nan")), .1, torch=t)
        with self.assertRaises(ValueError):
            family.prepare(t.ones((2, 2)), .1, torch=t)
        c = self.context()
        with self.assertRaises(ValueError):
            family.project(c, t.ones((2, 3)), torch=t)
        with self.assertRaises(ValueError):
            family.least_squares_target(c, t.full((1, 2, 2, 2), float("inf")), torch=t)


if __name__ == "__main__":
    unittest.main()
