"""Artificial arrays only: no model, images, study seeds, or GT evaluation."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for name in ("marginal_coupling", "oblique_view_geometry", "oblique_fusion"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
fusion = sys.modules["oblique_fusion"]
coupling = sys.modules["marginal_coupling"]
geometry = sys.modules["oblique_view_geometry"]


def fixture():
    # Smooth, nonuniform and nonsymmetric; only these hand-authored values.
    centers = np.arange(256, dtype=np.float64) + .5
    logits = {}
    for view, (x, y) in zip(fusion.VIEWS, ((85, 101), (96, 112), (118, 94), (75, 125))):
        logits[view] = np.concatenate((-(centers - y) ** 2 / 2500, -(centers - x) ** 2 / 3200))
    native = {view: np.array([73.125 + i, 151.375 - i]) for i, view in enumerate(fusion.VIEWS)}
    return logits, native


def inverse_expected(xy, view):
    # Independent analytic inverse, including unequal x and y.
    angle, scale = {"original": (0, 1), "reference": (0, .7),
                    "oblique": (np.pi / 4, .7), "axis_control": (np.pi / 2, .7)}[view]
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]]) @ (xy - 128) / scale + 128


class FusionTests(unittest.TestCase):
    def test_all_methods_float64_and_exact_native_inverse_without_clipping(self):
        logits, native = fixture()
        native["original"] = np.array([-12.5, 270.75])
        native["reference"] = np.array([.1234567890123, 255.9876543210987])
        result = fusion.fuse(logits, native)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(tuple(result["methods"]), fusion.METHODS)
        self.assertEqual(len(result["methods"]), 11)
        for name, point in result["methods"].items():
            self.assertEqual(point.dtype, np.float64, name)
            self.assertEqual(point.shape, (2,))
            self.assertTrue(np.isfinite(point).all())
        for view in fusion.VIEWS:
            np.testing.assert_allclose(result["methods"][view + "_native"], inverse_expected(native[view], view), atol=6e-14, rtol=0)
        np.testing.assert_array_equal(result["methods"]["original_native"], native["original"])
        self.assertLess(result["methods"]["reference_native"][0], 0)
        self.assertGreater(result["methods"]["reference_native"][1], 256)

    def test_independent_native_analytical_readout_and_yx_mapping(self):
        logits, native = fixture()
        values = np.arange(256, dtype=np.float64)
        logits["reference"] = np.concatenate((-(values - 173) ** 2 / 2000, -(values - 61) ** 2 / 3000))
        probabilities = []
        for head in (logits["reference"][256:], logits["reference"][:256]):
            p = np.exp(.5 * head - (.5 * head).max())
            p /= p.sum()
            p = np.maximum(p, 1e-12)
            p /= p.sum()
            mode = int(p.argmax())
            indexes = np.arange(max(0, mode - 20), min(256, mode + 21))
            probabilities.append((p[indexes] @ (indexes + .5)) / p[indexes].sum())
        expected = inverse_expected(np.array(probabilities), "reference")
        result = fusion.fuse(logits, native)
        self.assertEqual(result["status"], "completed")
        np.testing.assert_allclose(result["methods"]["reference_independent"], expected, atol=6e-14, rtol=0)
        self.assertLess(expected[0], expected[1])

    def test_entropy_tie_prefers_reference_and_lower_oblique_uses_native(self):
        logits, native = fixture()
        logits["oblique"] = logits["reference"].copy()
        result = fusion.fuse(logits, native)
        self.assertEqual(result["diagnostics"]["entropy_selected_view"], "reference")
        np.testing.assert_array_equal(result["methods"]["entropy_select"], result["methods"]["reference_native"])
        logits["oblique"] *= 2
        result = fusion.fuse(logits, native)
        self.assertEqual(result["diagnostics"]["entropy_selected_view"], "oblique")
        np.testing.assert_array_equal(result["methods"]["entropy_select"], result["methods"]["oblique_native"])

    def test_coordinate_average_is_in_original_coordinates(self):
        logits, native = fixture()
        result = fusion.fuse(logits, native)
        expected = .5 * inverse_expected(native["reference"], "reference") + .5 * inverse_expected(native["oblique"], "oblique")
        np.testing.assert_allclose(result["methods"]["coordinate_average"], expected, atol=6e-14, rtol=0)

    def test_product_and_mixture_use_probability_fields_then_reference_inverse(self):
        logits, native = fixture()
        result = fusion.fuse(logits, native)
        views = geometry.fixed_view_affines()
        px, py = coupling.native_marginals(logits["reference"])
        q0 = np.outer(py, px)
        fields = geometry.map_marginal_probabilities(*coupling.native_marginals(logits["oblique"]), views["reference"], views["oblique"])
        likelihood = fields[0] * fields[1]
        product = q0 * likelihood
        product /= product.sum()
        mixture = .5 * q0 + .5 * likelihood / likelihood.sum()
        for method, joint in (("product_joint", product), ("mixture_joint", mixture)):
            expected = inverse_expected(coupling.mode_local_mean(joint), "reference")
            np.testing.assert_allclose(result["methods"][method], expected, atol=1e-13, rtol=0)

    def test_solver_inputs_mapping_and_fixed_limits(self):
        logits, native = fixture()
        actual = coupling.entropic_coupling
        with patch.object(coupling, "entropic_coupling", wraps=actual) as spy:
            result = fusion.fuse(logits, native)
        self.assertEqual(spy.call_count, 2)
        px, py = coupling.native_marginals(logits["reference"])
        for call, view in zip(spy.call_args_list, ("oblique", "axis_control")):
            np.testing.assert_array_equal(call.args[0], px)
            np.testing.assert_array_equal(call.args[1], py)
            transforms = geometry.fixed_view_affines()
            fx, fy = geometry.map_marginal_probabilities(*coupling.native_marginals(logits[view]), transforms["reference"], transforms[view])
            np.testing.assert_array_equal(call.args[2], -np.log(fx) - np.log(fy))
            self.assertEqual(call.kwargs, {"max_iterations": 1000, "tolerance": 1e-9})
        self.assertEqual(result["status"], "completed")

    def test_mean_bounds_and_separable_axis_joint(self):
        result = fusion.fuse(*fixture())
        self.assertTrue(result["diagnostics"]["axis_joint_check_passed"])
        self.assertLess(result["diagnostics"]["axis_joint_max_abs_difference"], 1e-14)
        for record in result["diagnostics"]["global_means"].values():
            self.assertTrue(record["mean_check_passed"])
            for frame in ("reference", "original"):
                self.assertTrue(np.all(np.abs(record[frame + "_delta_xy"]) <= np.array(record[frame + "_bound_xy"]) + 1e-15))
        for record in result["solvers"].values():
            self.assertTrue(record["converged"])
            self.assertLessEqual(record["residual"], 1e-9)

    def test_diagnostics_are_json_scalars_and_keep_distribution_changes(self):
        result = fusion.fuse(*fixture())
        json.dumps(result["diagnostics"], allow_nan=False)
        json.dumps(result["solvers"], allow_nan=False)
        records = result["diagnostics"]["joint_distributions"]
        self.assertEqual(set(records), {"reference_independent", "product_joint", "mixture_joint", "oblique_coupling", "axis_coupling"})
        self.assertEqual(records["reference_independent"]["l1_change_from_independent"], 0)
        self.assertEqual(records["reference_independent"]["kl_to_independent"], 0)
        for record in records.values():
            self.assertGreaterEqual(record["exact_mode_tie_count"], 1)
            self.assertAlmostEqual(record["total_mass"], 1., places=12)
        self.assertAlmostEqual(result["diagnostics"]["reference_native_parity_max_abs_difference_original"],
                               float(np.abs(result["methods"]["reference_native"] - result["methods"]["reference_independent"]).max()))

    def test_mean_gate_is_reference_units_not_inverse_warp_units(self):
        q0 = np.full((256, 256), 1 / 65536)
        q = q0.copy()
        q[50, 100] -= 4e-7
        q[150, 100] += 4e-7
        inverse = geometry.invert_affine(geometry.fixed_view_affines()["reference"])
        report = fusion._mean_diagnostic(q, q0, inverse)
        self.assertAlmostEqual(report["reference_max_abs_difference"], 4e-5, places=12)
        self.assertGreater(report["original_max_abs_difference"], 5e-5)
        self.assertTrue(report["mean_check_passed"])

    def test_nonconvergence_fails_entire_result_without_fallback(self):
        actual = coupling.entropic_coupling
        failed = coupling.CouplingResult(None, 1000, 2e-7, 1e-15, 2e-7, False)
        count = 0
        def first_fails(*args, **kwargs):
            nonlocal count
            count += 1
            return failed if count == 1 else actual(*args, **kwargs)
        with patch.object(coupling, "entropic_coupling", side_effect=first_fails):
            result = fusion.fuse(*fixture())
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["methods"])
        self.assertFalse(result["solvers"]["oblique_coupling"]["converged"])
        self.assertEqual(result["solvers"]["oblique_coupling"]["iterations"], 1000)
        self.assertTrue(result["solvers"]["axis_coupling"]["converged"])

    def test_numeric_solver_exception_is_explicit_failure(self):
        with patch.object(coupling, "entropic_coupling", side_effect=ValueError("fixture arithmetic failure")):
            result = fusion.fuse(*fixture())
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["methods"])
        self.assertEqual(len(result["failure_reasons"]), 2)
        self.assertIn("fixture arithmetic failure", result["solvers"]["axis_coupling"]["error"])

    def test_coordinate_tie_diagnostic_does_not_reject_valid_joint(self):
        # Uniform independent has all-bin exact ties. A marginal-preserving
        # 2x2 checkerboard perturbation changes the chosen mode far away, while
        # joint maxdiff is only 1e-16; do not make a coordinate agreement gate.
        logits = {view: np.zeros(512) for view in fusion.VIEWS}
        native = {view: np.array([128., 128.]) for view in fusion.VIEWS}
        q = np.full((256, 256), 1 / 65536)
        q[100, 100] += 1e-16
        q[100, 101] -= 1e-16
        q[101, 100] -= 1e-16
        q[101, 101] += 1e-16
        fixed = coupling.CouplingResult(q, 1, 0., 0., 0., True)
        with patch.object(coupling, "entropic_coupling", return_value=fixed):
            result = fusion.fuse(logits, native)
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["diagnostics"]["axis_coordinate_max_abs_difference_original"], 100)
        self.assertLess(result["diagnostics"]["axis_joint_max_abs_difference"], 1e-15)
        self.assertEqual(result["diagnostics"]["joint_distributions"]["reference_independent"]["exact_mode_tie_count"], 65536)
        self.assertEqual(result["diagnostics"]["joint_distributions"]["axis_coupling"]["exact_mode_tie_count"], 2)

    def test_invalid_axis_joint_is_not_accepted(self):
        logits, native = fixture()
        px, py = coupling.native_marginals(logits["reference"])
        bad = np.outer(py, px).copy()
        bad[90:92, 90:92] += np.array([[1, -1], [-1, 1]]) * 2e-6
        self.assertTrue(np.all(bad > 0))
        fixed = coupling.CouplingResult(bad, 1, 0., 0., 0., True)
        with patch.object(coupling, "entropic_coupling", return_value=fixed):
            result = fusion.fuse(logits, native)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["methods"])
        self.assertFalse(result["diagnostics"]["axis_joint_check_passed"])

    def test_incorrect_global_mean_fails_even_if_solver_claims_convergence(self):
        logits, native = fixture()
        px, py = coupling.native_marginals(logits["reference"])
        bad = np.outer(py, px).copy()
        bad[60, 100] -= 1e-6
        bad[180, 100] += 1e-6
        self.assertTrue(np.all(bad > 0))
        fixed = coupling.CouplingResult(bad, 1, 0., 0., 0., True)
        with patch.object(coupling, "entropic_coupling", return_value=fixed):
            result = fusion.fuse(logits, native)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["methods"])
        self.assertFalse(result["diagnostics"]["global_means"]["oblique_coupling"]["mean_check_passed"])
        self.assertAlmostEqual(result["diagnostics"]["global_means"]["oblique_coupling"]["original_max_abs_difference"], 120e-6 / .7, places=12)

    def test_input_and_output_mutation_isolation(self):
        logits, native = fixture()
        before_logits = {key: value.copy() for key, value in logits.items()}
        before_native = {key: value.copy() for key, value in native.items()}
        result = fusion.fuse(logits, native)
        result["methods"]["original_native"][0] = 999
        for view in fusion.VIEWS:
            np.testing.assert_array_equal(logits[view], before_logits[view])
            np.testing.assert_array_equal(native[view], before_native[view])

    def test_bad_shapes_keys_types_nonfinite_reject_before_solver(self):
        for which, view, bad in ((0, "original", np.zeros(511)), (0, "reference", np.zeros((512, 1))),
                                 (0, "oblique", np.full(512, np.nan)), (0, "axis_control", np.ones(512, dtype=bool)),
                                 (1, "original", np.array([1, np.inf])), (1, "reference", np.zeros(3)),
                                 (1, "oblique", np.array([1j, 2])), (1, "axis_control", np.array(["1", "2"]))):
            with self.subTest(which=which, view=view):
                inputs = list(fixture())
                inputs[which][view] = bad
                with patch.object(coupling, "entropic_coupling") as solver, self.assertRaises(ValueError):
                    fusion.fuse(*inputs)
                solver.assert_not_called()
        for which in (0, 1):
            inputs = list(fixture())
            del inputs[which]["original"]
            with self.assertRaises(ValueError):
                fusion.fuse(*inputs)


if __name__ == "__main__":
    unittest.main()
