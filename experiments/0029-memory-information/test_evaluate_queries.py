"""Small manufactured-mask tests; no GT/media/runtime/model I/O."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


core = load("evr0029_eval_core_test", HERE / "diagnostic_core.py")
e = load("evr0029_query_eval_test", HERE / "evaluate_queries.py")


class ForbiddenGT:
    def __iter__(self):
        raise AssertionError("GT accessed before complete prediction validation")

    def __getitem__(self, key):
        raise AssertionError("GT accessed before complete prediction validation")


def fixture(actionable=True):
    rows, truth = [], {}
    for sequence, frame in core.QUERY_IDENTITIES:
        target = np.zeros((4, 4), dtype=np.uint8)
        target[:2] = 1
        masks = {name: target == 1 for name in ("B", "D1", "D3", "M0", "M1", "M2", "P", "R")}
        witnesses = {"target_id": 1, "actionable": actionable, "current_shape": [4, 4],
                     "positive": [], "negative": []}
        if actionable:
            for label, name, y in ((1, "positive", 0), (0, "negative", 3)):
                witnesses[name] = [{"target_xy": [float(x), float(y)],
                                    "rounded_current_xy": [x, y], "label": label} for x in range(4)]
        selection = core.select_controls(sequence=sequence, object_id=1, frame=frame,
            actionable=actionable, losses={"B": 2., "D1": 1., "D3": 3.},
            predicted_ious={"B": .9, "D1": .8, "D3": .7},
            unsigned_changes={"B": 0., "D1": .2, "D3": .1},
            native_losses=[2., 2., 2.], prompt_loss=2., reset_loss=2.)
        rows.append({"sequence": sequence, "frame": frame, "target_id": 1,
                     "status": "completed", "actionable": actionable, "selection": selection,
                     "witnesses": witnesses, "candidates": masks,
                     "costs": {"seconds": {"wall": 1., "decoder": .2},
                               "counts": {"decoder": 1}}})
        truth[(sequence, frame)] = target
    return rows, truth


class EvaluationTests(unittest.TestCase):
    def test_identity_failure_and_technical_incompleteness_precede_gt(self):
        rows, _ = fixture()
        with self.assertRaisesRegex(ValueError, "48-query"):
            e.evaluate_queries(rows[:-1], ForbiddenGT(), core=core)
        rows[-1]["status"] = "failed"
        rows[-1]["reason"] = "manufactured technical failure"
        result = e.evaluate_queries(rows, ForbiddenGT(), core=core)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["gt_accessed"])
        self.assertIsNone(result["summary"])
        self.assertFalse(result["gate"]["pass"])
        self.assertEqual(len(result["rows"]), 48)

    def test_invalid_last_choice_cannot_touch_first_gt(self):
        rows, _ = fixture()
        rows[-1]["selection"]["choices"]["signed_deletion"] = "M2"
        with self.assertRaisesRegex(ValueError, "escaped"):
            e.evaluate_queries(rows, ForbiddenGT(), core=core)

    def test_round_half_up_precision_void_and_selected_harm(self):
        rows, annotations = fixture()
        first = rows[0]
        first["witnesses"]["positive"][0]["target_xy"] = [.5, 1.]
        first["witnesses"]["positive"][0]["rounded_current_xy"] = [1, 1]
        gt = annotations[(first["sequence"], first["frame"])]
        gt[1, 1] = 255  # Unknown, never a negative or a correct positive.
        gt[3, 0] = 1   # One transferred negative is wrong.
        first["candidates"]["D1"] = np.zeros((4, 4), dtype=bool)
        result = e.evaluate_queries(rows, annotations, core=core)
        row = result["rows"][0]
        self.assertEqual(row["witness_precision"]["positive"],
                         {"retained": 4, "known": 3, "void_unknown": 1,
                          "correct": 3, "incorrect": 0, "precision": 1.})
        self.assertEqual(row["witness_precision"]["negative"]["precision"], .75)
        self.assertEqual(row["loss_j_sign"]["selected"]["loss_gain_sign"], 1)
        self.assertEqual(row["loss_j_sign"]["selected"]["j_gain_sign"], -1)
        harm = result["diagnostics"]["pooled"]["harmful_signed_actions"]
        self.assertEqual(harm["count"], 1)
        self.assertGreater(harm["worst"]["j_drop"], .05)

    def test_gt_cannot_rerank_sealed_choices_or_enter_control_gate(self):
        rows, annotations = fixture()
        rows[0]["candidates"]["D1"] = np.zeros((4, 4), dtype=bool)
        before = copy.deepcopy([row["selection"] for row in rows])
        report = e.evaluate_queries(rows, annotations, core=core)
        self.assertEqual([row["selection"] for row in rows], before)
        self.assertEqual(report["rows"][0]["selected_branches"]["signed_deletion"], "D1")
        self.assertEqual(report["rows"][0]["candidate_j"]["M0"], 1.)
        self.assertNotIn("oracle", report["summary"]["best_pooled_control"])
        self.assertFalse(report["gate"]["pass"])
        self.assertEqual(report["diagnostics"]["pooled"]["costs"]["seconds"]["wall"]["sum"], 48.)
        self.assertAlmostEqual(report["diagnostics"]["pooled"]["costs"]["seconds"]["decoder"]["sum"], 9.6)
        self.assertNotIn("total_seconds", report)

    def test_nonactionable_full_population_retains_unrestricted_headroom(self):
        rows, annotations = fixture(actionable=False)
        for row in rows:
            row["candidates"]["B"] = np.zeros((4, 4), dtype=bool)
            del row["candidates"]["P"]
            del row["candidates"]["R"]
        result = e.evaluate_queries(rows, annotations, core=core)
        self.assertEqual(result["denominator"], 48)
        self.assertEqual(result["summary"]["pooled"]["signed_deletion"], 0.)
        self.assertEqual(result["summary"]["pooled"]["oracle_admissible"], 0.)
        self.assertEqual(result["summary"]["pooled"]["oracle_unrestricted"], 1.)
        native = result["diagnostics"]["pooled"]["native_bank"]
        self.assertEqual(native["admissible_mean_gain"], 0.)
        self.assertEqual(native["unrestricted_mean_gain"], 1.)
        self.assertEqual(result["diagnostics"]["pooled"]["method_choices"]
                         ["signed_deletion"]["selected_baseline"], 48)
        self.assertFalse(result["gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
