"""Manufactured saved-bank tests, never a model/GT/dataset execution."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import evaluate
import metrics


class EvaluateTests(unittest.TestCase):
    def make_bank(self, directory, *, initial=False):
        objects, count, frame = 2, 1 if initial else 3, 0 if initial else 1
        shape = (3, 5)
        low = np.empty((objects, count, 256, 256), dtype=np.float32)
        quality = np.zeros((objects, count), dtype=np.float32)
        if initial:
            low[:, 0] = np.array([1, 2], dtype=np.float32)[:, None, None]
            selected = np.array([0, 0], dtype=np.int64)
        else:
            low[0] = np.array([1, 3, -1], dtype=np.float32)[:, None, None]
            low[1] = np.array([2, -2, 4], dtype=np.float32)[:, None, None]
            quality[:] = [[.2, .1, .1], [.1, .1, .3]]
            selected = np.array([0, 2], dtype=np.int64)
        alt = np.asarray([[i for i in range(count) if i != s] for s in selected],
                         dtype=np.int64).reshape(objects, count - 1)
        original = np.empty((objects, 3, *shape), dtype=np.float32)
        for obj in range(objects):
            original[obj, 0] = float(low[obj, selected[obj], 0, 0])
            for lane in range(2):
                original[obj, lane + 1] = original[obj, 0] if initial else low[obj, alt[obj, lane], 0, 0]
        controls = metrics.compose_controls(original, [1, 2])
        bank = dict(low_bank=low, native_original_logits=original[:, 0],
                    object_ids=np.array([1, 2], dtype=np.int64),
                    selected_slots=selected, alternative_slots=alt,
                    predicted_quality=quality, object_score_logits=np.ones(objects, dtype=np.float32),
                    labels=np.stack([controls[name] for name in metrics.CONTROLS]).astype(np.uint8))
        row = dict(shape=list(shape), object_ids=[1, 2])
        record = dict(frame=frame, bank_count=count, selected_low_exact=True,
            selected_original_exact=True, initial_diagnostic_only=initial,
            native_positive_pixels=(original[:, 0] > 0).sum(axis=(1, 2)).tolist(),
            **{key: bank[key].tolist() for key in
               ("selected_slots", "alternative_slots", "predicted_quality", "object_score_logits")})
        path = Path(directory) / "bank.npz"
        np.savez(path, **bank)
        return path, row, record, bank, original

    def reconstruct(self, path, row, record):
        return evaluate.reconstruct_frame(path, row, record, np=np, torch=torch, metrics=metrics)

    def test_rebuild_original_slot_order_with_competition(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row, record, bank, expected = self.make_bank(directory)
            actual, controls = self.reconstruct(path, row, record)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(controls["B"], np.full((3, 5), 2))
            for index, name in enumerate(metrics.CONTROLS):
                np.testing.assert_array_equal(controls[name], bank["labels"][index])

    def test_initial_single_bank_duplicates_only_for_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row, record, _, _ = self.make_bank(directory, initial=True)
            original, controls = self.reconstruct(path, row, record)
            np.testing.assert_array_equal(original[:, 0], original[:, 1])
            np.testing.assert_array_equal(original[:, 0], original[:, 2])
            for name in metrics.CONTROLS:
                np.testing.assert_array_equal(controls[name], controls["B"])

    def test_saved_map_and_native_parity_corruption_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row, record, bank, _ = self.make_bank(directory)
            changed = copy.deepcopy(bank)
            changed["labels"][0, 0, 0] = 0
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, "categorical composition"):
                self.reconstruct(path, row, record)
            changed = copy.deepcopy(bank)
            changed["native_original_logits"][0, 0, 0] += np.float32(.001)
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, "original-resolution native parity"):
                self.reconstruct(path, row, record)

    def test_absence_cannot_resurrect_positive_raw_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row, record, bank, _ = self.make_bank(directory)
            bank["object_score_logits"][0] = 0
            record["object_score_logits"][0] = 0
            np.savez(path, **bank)
            with self.assertRaisesRegex(ValueError, "absent raw masks"):
                self.reconstruct(path, row, record)

    def test_continuation_includes_strongest_union_and_three_clip_condition(self):
        methods = {name: dict(J=.2, Jtr=.2) for name in metrics.CONTROLS}
        methods.update(O=dict(J=.8, Jtr=.8), O_B=dict(J=.6, Jtr=.6))
        report = dict(methods=methods, oracle_decomposition={
            metric: dict(total_O_minus_B=.6, additional_support_O_minus_O_B=.2)
            for metric in ("J", "Jtr")},
            per_sequence=[dict(sequence=name, oracle_decomposition={
                "Jtr": dict(additional_support_O_minus_O_B=value)})
                for name, value in zip(("a", "b", "c", "d"), (.02, .02, .02, 0))])
        self.assertTrue(evaluate.continuation(report, metrics.CONTROLS)["numerical_passed"])
        report["methods"]["B+M1"].update(J=.79)
        answer = evaluate.continuation(report, metrics.CONTROLS)
        self.assertFalse(answer["numerical_passed"])
        self.assertFalse(answer["numerical_clauses"]["ceiling_over_best_real_J_at_least_3pp"])
        report["methods"]["B+M1"].update(J=.2)
        report["per_sequence"][2]["oracle_decomposition"]["Jtr"]["additional_support_O_minus_O_B"] = 0
        answer = evaluate.continuation(report, metrics.CONTROLS)
        self.assertFalse(answer["numerical_clauses"]["extra_Jtr_at_least_1pp_in_three_clips"])
        self.assertTrue(answer["clean_evaluation_outer_required"])

    def test_failed_all_four_admission_stops_before_guard_or_label_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            common_path = root / "common.py"
            common_path.write_text("# unused fake serializer module\n")
            guard = root / "repo/experiments/0022-frontend/qualify.py"
            guard.parent.mkdir(parents=True)
            guard.write_text("raise AssertionError('guard must not import')\n")
            here = Path(evaluate.__file__).resolve().parent
            paths = [here / name for name in ("evaluate.py", "metrics.py", "admit.py", "ADMISSION.md")]
            paths += [common_path, guard]
            spec = dict(source_sha256={str(path): evaluate.sha(path) for path in paths},
                        common_path=str(common_path), repo=str(root / "repo"),
                        metrics_path=str(here / "metrics.py"), guard_source=str(guard),
                        output=str(root / "out"))
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            out = root / "out"

            def start(path, kind):
                out.mkdir()
                return spec, out, {}, time.perf_counter()

            def save(path, record):
                Path(path).write_text(json.dumps(record))

            admitted = mock.Mock(side_effect=ValueError("missing fourth complete sequence"))
            modules = []

            def fake_module(name, path):
                modules.append(name)
                if name == "d35_evaluate_common":
                    return SimpleNamespace(start=start, save=save)
                if name == "d35_evaluation_admission":
                    return SimpleNamespace(evaluation_context=admitted)
                raise AssertionError("runtime/guard import reached after failed admission")

            with mock.patch.object(evaluate, "COMMON_SHA256", evaluate.sha(common_path)), \
                    mock.patch.object(evaluate, "module", side_effect=fake_module):
                with self.assertRaisesRegex(ValueError, "missing fourth"):
                    evaluate.run(spec_path)
            admitted.assert_called_once()
            self.assertEqual(modules, ["d35_evaluate_common", "d35_evaluation_admission"])
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["label_decode_attempts"], 0)
            self.assertEqual(receipt["model_calls"], 0)
            self.assertFalse((out / "seal.json").exists())


if __name__ == "__main__":
    unittest.main()
