"""Hand-authored saved-output fixtures only: no models, real images or labels."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/evaluate_oblique_saved.py"
SPEC = importlib.util.spec_from_file_location("oblique_saved_evaluation", SOURCE)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def dump(path, value):
    """Tests may replace/reseal their own synthetic fixtures, never study files."""
    path.write_text(json.dumps(value, allow_nan=False))


def diagnostic(identity=None):
    value = {"solvers": {m: {"converged": True, "iterations": 1,
                            "row_error": 0., "column_error": 0., "residual": 0.}
                         for m in ("oblique_coupling", "axis_coupling")},
             "global_means": {m: {"reference_max_abs_difference": 0.,
                                   "reference_delta_xy": [0., 0.],
                                   "reference_bound_xy": [0., 0.],
                                   "original_delta_xy": [0., 0.],
                                   "original_bound_xy": [0., 0.],
                                   "original_max_abs_difference": 0., "mean_check_passed": True}
                              for m in ("oblique_coupling", "axis_coupling")},
             "axis_joint_max_abs_difference": 0.,
             "axis_coordinate_max_abs_difference_original": 0.,
             "axis_joint_check_passed": True,
             "marginal_entropies": {"reference": 1., "oblique": 2.},
             "entropy_selected_view": "reference"}
    value.update(identity or {})
    return value


def fake_fusion(logits, native):
    # This is an interface fixture, not a surrogate for the transport method.
    assert set(logits) == set(native) == set(evaluation.VIEWS)
    d = diagnostic()
    return {"status": "completed", "methods": {
        m: np.array([0. if m == evaluation.CANDIDATE else 8., 0.])
        for m in evaluation.METHODS}, "solvers": d.pop("solvers"), "diagnostics": d}


def cases(error=8.):
    return [{"group_id": group, "frame_id": frame, "point_id": point,
             "occluded": False, "errors": {m: float(error) for m in evaluation.METHODS}}
            for group in evaluation.GROUPS for frame in evaluation.FRAMES for point in range(8)]


class ArithmeticTests(unittest.TestCase):
    def test_strict_thresholds_and_query_frame_excluded(self):
        rows = cases()
        for i, row in enumerate(rows):
            error = 0. if row["frame_id"] == 0 else [1., 2., 4., 8., 16.][(i % 128 - 8) % 5]
            row["errors"] = {m: error for m in evaluation.METHODS}
        result = evaluation.summarize_cases(rows)
        for group in evaluation.GROUPS:
            metric = result["groups"][group]["methods"][evaluation.CANDIDATE]
            self.assertEqual(metric["hit_counts"], [0, 24, 48, 72, 96])
            self.assertEqual(metric["eligible"], 120)
            self.assertEqual(metric["coverage"], 1.)
            self.assertAlmostEqual(metric["quality"], .4)

    def test_group_macro_not_pooled_rows(self):
        rows = cases(100.)
        for row in rows[:128]:
            row["occluded"] = not (row["frame_id"] == 16 and row["point_id"] == 0)
            row["errors"][evaluation.CANDIDATE] = 0.
        result = evaluation.summarize_cases(rows)
        self.assertEqual(result["groups"]["g-036"]["methods"][evaluation.CANDIDATE]["eligible"], 1)
        self.assertEqual(result["macro"][evaluation.CANDIDATE]["quality"], .5)
        self.assertNotEqual(result["macro"][evaluation.CANDIDATE]["quality"], 1 / 121)

    def test_empty_visible_group_refuses(self):
        rows = cases()
        for row in rows[:128]:
            row["occluded"] = True
        with self.assertRaisesRegex(ValueError, "eligible"):
            evaluation.summarize_cases(rows)

    def test_exact_five_percentage_point_boundary(self):
        rows = cases(1.)
        for group_index in range(2):
            for i in range(group_index * 128 + 8, group_index * 128 + 38):
                rows[i]["errors"][evaluation.CANDIDATE] = 0.
        result = evaluation.summarize_cases(rows)
        self.assertAlmostEqual(result["macro_quality_gain"], .05)
        self.assertTrue(result["quality_gates"]["macro_gain_at_least_0.05"])
        self.assertTrue(all(result["quality_gates"].values()))
        rows[8]["errors"][evaluation.CANDIDATE] = 1.
        self.assertFalse(evaluation.summarize_cases(rows)["quality_gates"]["macro_gain_at_least_0.05"])

    def test_axis_negative_control_is_a_strong_control_not_excluded(self):
        rows = cases(100.)
        for row in rows:
            row["errors"][evaluation.CANDIDATE] = 0.
            row["errors"]["axis_coupling"] = 0.
        result = evaluation.summarize_cases(rows)
        self.assertEqual(result["strongest_control"], "axis_coupling")
        self.assertFalse(result["quality_gates"]["macro_gain_at_least_0.05"])
        self.assertFalse(result["quality_gates"]["each_group_above_every_control"])

    def test_macro_winner_does_not_replace_per_group_strongest(self):
        rows = cases(100.)
        for row in rows:
            row["errors"][evaluation.CANDIDATE] = 1.
            row["errors"]["original_native"] = 0. if row["group_id"] == "g-036" else 100.
            row["errors"]["oblique_native"] = 100. if row["group_id"] == "g-036" else 0.
        result = evaluation.summarize_cases(rows)
        self.assertTrue(result["quality_gates"]["macro_gain_at_least_0.05"])
        self.assertFalse(result["quality_gates"]["each_group_above_every_control"])

    def test_quality_tie_uses_fixed_order_for_epe_control(self):
        rows = cases(100.)
        for i, row in enumerate(rows):
            row["errors"]["original_native"] = 2.1
            row["errors"]["reference_native"] = 3.9
            row["errors"][evaluation.CANDIDATE] = 0. if i % 128 < 68 else 5.8
        result = evaluation.summarize_cases(rows)
        self.assertEqual(result["strongest_control"], "original_native")
        self.assertTrue(result["quality_gates"]["macro_gain_at_least_0.05"])
        self.assertFalse(result["quality_gates"]["macro_epe_not_worse_than_best_quality_control"])

    def test_corrections_harms_use_strict_4px_and_eligible_only(self):
        rows = cases(100.)
        for offset in (0, 128):
            for j, (candidate, baseline) in enumerate(((3.99, 4.), (4., 3.99), (4., 4.), (1., 1.))):
                rows[offset + 8 + j]["errors"][evaluation.CANDIDATE] = candidate
                rows[offset + 8 + j]["errors"]["original_native"] = baseline
            rows[offset]["errors"][evaluation.CANDIDATE] = 0.  # Query frame never a correction.
            rows[offset + 12]["occluded"] = True
            rows[offset + 12]["errors"][evaluation.CANDIDATE] = 0.
        result = evaluation.summarize_cases(rows)
        for group in evaluation.GROUPS:
            self.assertEqual(result["groups"][group]["paired"]["original_native"],
                             {"corrections_4px": 1, "harms_4px": 1})


class SavedFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.preparation = self.root / "preparation"
        self.preparation.mkdir()
        self.derived = self.root / "derived"
        self.derived.mkdir()
        self.prep = {"status": "completed_preparation", "files": {}}
        self.truth = {}
        self.rows, self.diagnostics = [], []
        for group in evaluation.GROUPS:
            self.truth[group] = {"points": np.zeros((16, 8, 2), np.float64),
                                 "occluded": np.zeros((16, 8), np.bool_),
                                 "point_ids": np.arange(8, dtype=np.int64),
                                 "frame_ids": np.array(evaluation.FRAMES, np.int64)}
            for t, frame in enumerate(evaluation.FRAMES):
                for q in range(8):
                    identity = {"group_id": group, "observation_index": t, "frame_id": frame,
                                "point_index": q, "point_id": q}
                    self.rows.append(dict(identity, predictions={m: [0. if m == evaluation.CANDIDATE else 8., 0.]
                                                                  for m in evaluation.METHODS},
                                          visibility_logits={v: -1000. for v in evaluation.VIEWS}))
                    self.diagnostics.append(diagnostic(identity))
        self.save_truth()
        self.reseal()

    def save_truth(self):
        for group, arrays in self.truth.items():
            name = group + ".ground_truth.npz"
            np.savez(self.preparation / name, **arrays)
            self.prep["files"][name] = evaluation.record(self.preparation / name)
        dump(self.preparation / "receipt.json", self.prep)

    def reseal(self):
        dump(self.derived / "predictions.json", {"methods": list(evaluation.METHODS), "rows": self.rows})
        dump(self.derived / "diagnostics.json", self.diagnostics)
        dump(self.derived / "seal.json", {"status": "sealed", "rows": 256,
             "files": {name: evaluation.record(self.derived / name)
                       for name in ("predictions.json", "diagnostics.json")},
             "inputs": {"preparation_receipt": evaluation.record(self.preparation / "receipt.json")},
             "model_calls": 0, "GT_opened": False})

    def score(self, suffix=""):
        return evaluation.score(self.derived, self.preparation, self.root / ("score" + suffix))

    def test_complete_256_rows_all_methods_and_visibility_never_abstains(self):
        result = self.score()
        self.assertEqual(result["status"], "completed_evaluation")
        self.assertEqual(result["decision"], "GO_development_only")
        self.assertEqual(set(result["macro"]), set(evaluation.METHODS))
        self.assertEqual(len(json.loads((self.root / "score/case-errors.json").read_text())), 256)
        for group in evaluation.GROUPS:
            for metrics in result["groups"][group]["methods"].values():
                self.assertEqual(metrics["eligible"], 120)
                self.assertEqual(metrics["coverage"], 1.)
        self.assertTrue(all(result["numerical_gates"].values()))

    def test_occlusion_excludes_quality_but_reports_error(self):
        self.truth["g-036"]["occluded"][1, 0] = True
        self.rows[8]["predictions"][evaluation.CANDIDATE] = [1200., 0.]
        self.save_truth()
        self.reseal()
        result = self.score()
        metric = result["groups"]["g-036"]["methods"][evaluation.CANDIDATE]
        self.assertEqual(metric["quality"], 1.)
        self.assertEqual(metric["eligible"], 119)
        self.assertEqual(metric["occluded_epe"], 1200.)
        self.assertEqual(metric["all_point_epe"], 10.)

    def test_dynamic_numerical_gate_failure_cannot_go(self):
        original = copy.deepcopy(self.diagnostics)
        changes = (
            ("axis", lambda d: d.update(axis_joint_max_abs_difference=1.01e-7)),
            ("mean", lambda d: d["global_means"]["oblique_coupling"].update(reference_max_abs_difference=5.01e-5)),
            ("convergence", lambda d: d["solvers"]["axis_coupling"].update(converged=False)),
            ("residual", lambda d: d["solvers"]["oblique_coupling"].update(row_error=2e-9, residual=2e-9)),
        )
        for label, change in changes:
            self.diagnostics = copy.deepcopy(original)
            change(self.diagnostics[0])  # Query frame is still part of numerical checks.
            self.reseal()
            result = self.score(label)
            self.assertEqual(result["decision"], "NO_GO", label)
            self.assertFalse(all(result["numerical_gates"].values()), label)

    def test_numerical_boundaries_are_inclusive_without_boolean_trust(self):
        for d in self.diagnostics:
            d["axis_joint_max_abs_difference"] = 1e-7
            d["axis_joint_check_passed"] = False
            for m in ("oblique_coupling", "axis_coupling"):
                d["solvers"][m].update(iterations=1000, row_error=1e-9, residual=1e-9)
                d["global_means"][m]["reference_max_abs_difference"] = 5e-5
        self.reseal()
        self.assertTrue(all(self.score()["numerical_gates"].values()))

    def test_decoder_parity_uses_all_rows_and_original_pixels(self):
        self.rows[0]["predictions"]["reference_independent"][1] = .001001
        self.reseal()
        result = self.score()
        self.assertFalse(result["numerical_gates"]["reference_decoder_parity"])
        self.assertEqual(result["decision"], "NO_GO")

    def test_tamper_rejects_before_any_truth_load(self):
        for name in ("predictions.json", "diagnostics.json"):
            with self.subTest(name=name):
                self.reseal()
                path = self.derived / name
                path.write_text(path.read_text() + " ")
                with mock.patch.object(evaluation.np, "load", side_effect=AssertionError("GT read too early")) as load:
                    with self.assertRaisesRegex(ValueError, "binding"):
                        self.score(name)
                    load.assert_not_called()

    def test_negative_boolean_and_duplicate_indices_rejected(self):
        original_rows, original_diag = copy.deepcopy(self.rows), copy.deepcopy(self.diagnostics)
        for label, index, field, value in (("negative", 120, "observation_index", -1),
                                            ("bool", 0, "observation_index", False),
                                            ("negativepoint", 7, "point_index", -1)):
            self.rows, self.diagnostics = copy.deepcopy(original_rows), copy.deepcopy(original_diag)
            self.rows[index][field] = self.diagnostics[index][field] = value
            self.reseal()
            with self.assertRaisesRegex(ValueError, "indices"):
                self.score(label)
        self.rows, self.diagnostics = copy.deepcopy(original_rows), copy.deepcopy(original_diag)
        self.rows[1], self.diagnostics[1] = copy.deepcopy(self.rows[0]), copy.deepcopy(self.diagnostics[0])
        self.reseal()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.score("duplicate")

    def test_missing_row_and_missing_diagnostic_rejected(self):
        self.rows.pop()
        self.reseal()
        with self.assertRaisesRegex(ValueError, "row count"):
            self.score("row")
        self.rows.append(copy.deepcopy(self.rows[-1]))
        self.diagnostics.pop()
        self.reseal()
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            self.score("diagnostic")

    def test_exact_methods_and_xy_shape_required(self):
        original = copy.deepcopy(self.rows[0]["predictions"])
        for label, bad in (("missing", {k: v for k, v in original.items() if k != "axis_coupling"}),
                           ("extra", dict(original, surprise=[0., 0.])),
                           ("scalar", dict(original, oblique_coupling=0.)),
                           ("matrix", dict(original, oblique_coupling=[[0., 0.]]))):
            self.rows[0]["predictions"] = bad
            self.reseal()
            with self.assertRaisesRegex(ValueError, "schema"):
                self.score(label)

    def test_ground_truth_dtypes_and_ids_are_strict(self):
        original = copy.deepcopy(self.truth["g-036"])
        changes = (("points", np.float32), ("frame_ids", np.float64), ("point_ids", np.int32))
        for field, dtype in changes:
            self.truth["g-036"] = copy.deepcopy(original)
            self.truth["g-036"][field] = original[field].astype(dtype)
            self.save_truth()
            self.reseal()
            with self.assertRaisesRegex(ValueError, "GT schema"):
                self.score(field)


class DerivationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.preparation, self.prediction = self.root / "preparation", self.root / "prediction"
        self.preparation.mkdir()
        self.prediction.mkdir()
        prep = {"status": "completed_preparation", "files": {}}
        self.receipts = {}
        for group in evaluation.GROUPS:
            predictor = self.preparation / (group + ".predictor.npz")
            np.savez(predictor, video=np.zeros((16, 256, 256, 3), np.uint8),
                     query_xy=np.zeros((8, 2), np.float32), point_ids=np.arange(8, dtype=np.int64),
                     frame_ids=np.array(evaluation.FRAMES, np.int64))
            prep["files"][predictor.name] = evaluation.record(predictor)
            # Deliberately invalid GT bytes: derive must never open them.
            gt = self.preparation / (group + ".ground_truth.npz")
            gt.write_bytes(b"not a NumPy file and must not be opened during derive")
            prep["files"][gt.name] = evaluation.record(gt)
            folder = self.prediction / group
            folder.mkdir()
            receipt = {"status": "completed_predictions", "group_id": group,
                       "input_sha256": evaluation.sha(predictor), "forwards_attempted": 64, "frames": []}
            for view in evaluation.VIEWS:
                for t, frame in enumerate(evaluation.FRAMES):
                    path = folder / (view + "-%02d.npz" % t)
                    np.savez(path, coordinate_logits=np.zeros((8, 512), np.float32),
                             positions_xy=np.zeros((8, 2), np.float32),
                             visibility_logits=np.full(8, -1000., np.float32))
                    receipt["frames"].append({"view": view, "observation_index": t,
                                              "frame_index": frame, "output_file": path.name,
                                              "output_sha256": evaluation.sha(path)})
            self.receipts[group] = receipt
            dump(folder / "receipt.json", receipt)
        dump(self.preparation / "receipt.json", prep)

    def run_derive(self, name="derive", fusion=fake_fusion):
        return evaluation.derive(self.prediction, self.preparation, self.root / name, fusion=fusion)

    def test_complete_derive_never_opens_gt_and_retains_solver_records(self):
        real_open = Path.open
        opened = []
        def guarded(path, *args, **kwargs):
            if path.name.endswith(".ground_truth.npz"):
                raise AssertionError("derive opened GT")
            opened.append(path.name)
            return real_open(path, *args, **kwargs)
        fusion = mock.Mock(wraps=fake_fusion)
        with mock.patch.object(Path, "open", guarded):
            result = self.run_derive(fusion=fusion)
        self.assertEqual(result["rows"], 256)
        self.assertEqual(fusion.call_count, 256)
        self.assertFalse(result["GT_opened"])
        self.assertEqual(result["model_calls"], 0)
        diagnostics = evaluation.load_json(self.root / "derive/diagnostics.json")
        self.assertEqual(len(diagnostics), 256)
        self.assertTrue(all(set(d["solvers"]) == {"oblique_coupling", "axis_coupling"} for d in diagnostics))

    def test_raw_hash_tamper_or_missing_forward_refuses(self):
        group = evaluation.GROUPS[0]
        receipt = self.receipts[group]
        path = self.prediction / group / receipt["frames"][0]["output_file"]
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "changed raw"):
            self.run_derive("tamper")
        self.assertFalse((self.root / "tamper/seal.json").exists())
        receipt["frames"].pop()
        dump(self.prediction / group / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "64 forwards"):
            self.run_derive("missing")

    def test_duplicate_raw_filename_and_symlink_refused(self):
        group = evaluation.GROUPS[0]
        receipt = self.receipts[group]
        first = copy.deepcopy(receipt["frames"][0])
        receipt["frames"][0]["output_file"] = receipt["frames"][1]["output_file"]
        dump(self.prediction / group / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "unsafe or changed raw"):
            self.run_derive("duplicate-file")
        receipt["frames"][0] = first
        dump(self.prediction / group / "receipt.json", receipt)
        path = self.prediction / group / first["output_file"]
        path.unlink()
        path.symlink_to(receipt["frames"][1]["output_file"])
        with self.assertRaisesRegex(ValueError, "unsafe or changed raw"):
            self.run_derive("symlink")

    def test_numerical_failure_stops_without_seal_or_retry(self):
        def failed(logits, xy):
            result = fake_fusion(logits, xy)
            result["diagnostics"]["axis_joint_max_abs_difference"] = 1e-6
            return result
        with self.assertRaisesRegex(ValueError, "numerical"):
            self.run_derive(fusion=failed)
        self.assertTrue((self.root / "derive/started.json").is_file())
        self.assertTrue((self.root / "derive/failure.json").is_file())
        self.assertFalse((self.root / "derive/seal.json").exists())
        with self.assertRaises(FileExistsError):
            self.run_derive()


if __name__ == "__main__":
    unittest.main()
