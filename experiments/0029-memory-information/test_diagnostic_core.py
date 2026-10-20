"""Manufactured arrays only; no model, image decoder, file payload or benchmark."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("diagnostic_arithmetic", Path(__file__).with_name("diagnostic_core.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def selection(sequence, frame, prefer="D1", actionable=True):
    losses = {"B": 1., "D1": .5 if prefer == "D1" else 1.2,
              "D3": .5 if prefer == "D3" else 1.2}
    return c.select_controls(sequence=sequence, object_id=1, frame=frame,
        actionable=actionable, losses=losses,
        predicted_ious={"B": .9, "D1": .8, "D3": .8},
        unsigned_changes={"B": 0., "D1": .5, "D3": .5},
        native_losses=[.8, .8], prompt_loss=.8, reset_loss=.8)


def rows_fixture():
    result = []
    for sequence, n in c.COHORT:
        for i, frame in enumerate(c.query_schedule(n)):
            prefer = "D1" if i % 2 == 0 else "D3"
            result.append({"sequence": sequence, "frame": frame, "status": "completed",
                "actionable": True, "selection": selection(sequence, frame, prefer),
                "candidate_j": {"B": .5, "D1": .75 if prefer == "D1" else .25,
                    "D3": .75 if prefer == "D3" else .25, "M0": .625, "M1": .625,
                    "P": .625, "R": .625}})
    return result


class DiagnosticArithmetic(unittest.TestCase):
    def test_schedule_exact_and_video_evidence_gate_not_half_queries(self):
        self.assertEqual(c.query_schedule(25), [4, 7, 10, 13, 15, 18, 21, 24])
        self.assertEqual(c.query_schedule(83), [4, 15, 26, 37, 49, 60, 71, 82])
        rows = rows_fixture()
        for row in rows:
            row["actionable"] = row["sequence"] in dict(c.COHORT[:4]) and row["frame"] == 4
        report = c.evidence_coverage(rows)
        self.assertEqual(report["actionable_queries"], 4)
        self.assertTrue(report["pass"])
        rows[0]["actionable"] = False
        self.assertFalse(c.evidence_coverage(rows)["pass"])

    def test_boundary_collision_rounding_void_and_confidence(self):
        labels = np.zeros((16, 16), dtype=np.uint8)
        labels[4:12, 4:12] = 2; labels[0, 0] = 255
        p0 = np.array([[5,5], [1,1], [6,6], [6,6], [4,4],
                       [0,0], [15.8,2], [7,7], [2,2]], float)
        pt = np.array([[2.1,2.1], [2.4,2.4], [6.1,6.1], [6.2,6.2],
                       [8,8], [9,9], [10,10], [15.8,2], [11,11]], float)
        ids = np.column_stack((np.arange(9), np.arange(9)))
        report = c.select_witnesses(labels, p0, pt, ids,
                                   [.9,.8,.5,.8,.7,.7,.7,.7,.6], (16,16))
        self.assertEqual(report["target_id"], 2)
        self.assertEqual(report["counts"]["opposite_collision_points"], 2)
        self.assertEqual(report["counts"]["same_label_discarded"], 1)
        self.assertEqual(report["counts"]["boundary_or_void"], 2)
        self.assertEqual(report["counts"]["invalid_coordinates"], 2)
        self.assertEqual([x["match_index"] for x in report["positive"]], [3])
        self.assertEqual([x["match_index"] for x in report["negative"]], [8])
        self.assertFalse(report["actionable"])
        with self.assertRaises(ValueError):
            c.select_witnesses(labels, p0, pt, ids, np.ones(9), (16,16), target_id=3)

    def test_row_major_round_robin_same_class_ties_and_caps(self):
        labels = np.zeros((16,16), dtype=np.uint8); labels[4:12,4:12] = 1
        points = []
        for ox, oy in ((1,1), (113,113), (17,1), (97,113)):
            points.extend([[ox+i % 10, oy+i // 10] for i in range(20)])
        source = np.array([[6.,6.]]*40 + [[1.,1.]]*40)
        ids = np.column_stack((np.arange(80), np.arange(80)))
        report = c.select_witnesses(labels, source, points, ids, np.ones(80), (128,128))
        self.assertTrue(report["actionable"])
        self.assertEqual(len(report["positive"]), 32)
        self.assertEqual(len(report["negative"]), 32)
        self.assertEqual([x["match_index"] for x in report["positive"][:6]], [0,20,1,21,2,22])
        self.assertEqual([x["match_index"] for x in report["negative"][:4]], [40,60,41,61])
        self.assertEqual(report["counts"]["positive_before_cap"], 40)

    def test_pixel_center_logits_borders_and_extreme_balanced_loss(self):
        logits = np.array([[0.,10.],[20.,30.]])
        np.testing.assert_array_equal(
            c.sample_logits(logits, [[0,0],[3,3],[1.5,1.5],[1,1]], (4,4)),
            [0.,30.,15.,7.5])
        with self.assertRaises(ValueError):
            c.sample_logits(logits, [[4,0]], (4,4))
        witnesses = {"actionable": True, "current_shape": [4,4],
            "positive": [{"target_xy":[0.,0.]}]*4,
            "negative": [{"target_xy":[3.,3.]}]*8}
        extreme = np.array([[1000.,0.],[0.,-1000.]])
        self.assertEqual(c.balanced_logloss(extreme, witnesses), 0.)
        self.assertEqual(c.balanced_logloss(-extreme, witnesses), 1000.)
        witnesses["actionable"] = False
        with self.assertRaises(ValueError):
            c.balanced_logloss(extreme, witnesses)

    def test_signed_margin_ties_control_order_and_noop(self):
        args = dict(sequence="dogs-scale", object_id=1, frame=4, actionable=True,
            losses={"B": 2e-6, "D1": 1e-6, "D3": 3e-6},
            predicted_ious={"B": .5, "D1": .5, "D3": .5},
            unsigned_changes={"B": 0., "D1": .5, "D3": .5},
            native_losses=[0.,0.], prompt_loss=2e-6, reset_loss=3e-6)
        report = c.select_controls(**args)
        self.assertEqual(report["choices"]["signed_deletion"], "B")
        self.assertEqual(report["choices"]["random_on_signed"], "B")
        self.assertEqual(report["choices"]["native_bank"], "M0")
        self.assertEqual(report["choices"]["max_predicted_iou"], "B")
        self.assertEqual(report["choices"]["max_unsigned_change"], "D1")
        args["losses"] = {"B": 1., "D1": .5, "D3": .5}
        report = c.select_controls(**args)
        self.assertEqual(report["choices"]["signed_deletion"], "D1")
        self.assertEqual(report["choices"]["random_on_signed"], report["random"]["branch"])
        args["actionable"] = False
        self.assertEqual(set(c.select_controls(**args)["choices"].values()), {"B"})

    def test_j_threshold_void_and_empty_masks(self):
        labels = np.array([[1,255,0],[0,0,0]], np.uint8)
        logits = np.array([[1.,9.,0.],[0.,-1.,0.]])
        self.assertEqual(c.foreground_j(logits, labels, 1), 1.)
        self.assertEqual(c.foreground_j(np.zeros((2,3)), np.zeros((2,3), np.uint8), 1), 1.)
        self.assertEqual(c.foreground_j(logits, np.zeros((2,3), np.uint8), 1), 0.)
        self.assertEqual(c.unsigned_change(np.zeros((2,3)), np.zeros((2,3))), 0.)

    def test_complete_aggregation_preserves_48_and_compares_per_video_controls(self):
        rows = rows_fixture(); report = c.summarize(rows)
        self.assertTrue(report["eligible"])
        self.assertEqual(len(report["rows"]), 48)
        self.assertEqual(report["pooled"]["signed_deletion"], .75)
        self.assertEqual(report["pooled"]["baseline"], .5)
        self.assertEqual(report["candidate_gain"], .25)
        self.assertEqual(report["pooled"]["ungated_D1"], .5)
        self.assertEqual(report["pooled"]["ungated_D3"], .5)
        for video in report["videos"].values():
            independent = max(video["means"][name] for name in c.CONTROLS)
            self.assertEqual(video["signed_margin"], .75 - independent)
        self.assertEqual(report["gate"]["pass"],
                         report["candidate_gain"] >= .02 and report["signed_gain"] >= .02
                         and report["positive_videos"] >= 4 and report["worst_video_margin"] >= -.02)
        # A strong direct-prompt control defeats the fixed signed-selection gate.
        for row in rows:
            row["candidate_j"]["P"] = .875
        self.assertFalse(c.summarize(rows)["gate"]["pooled_selection"])

    def test_failure_inclusive_ineligible_and_unavailable_oracle_no_drop(self):
        rows = rows_fixture()
        for row in rows:
            row["actionable"] = False
            row["selection"] = selection(row["sequence"], row["frame"], actionable=False)
        report = c.summarize(rows)
        self.assertEqual(report["pooled"]["signed_deletion"], .5)
        self.assertEqual(report["pooled"]["oracle_admissible"], .5)
        self.assertEqual(report["pooled"]["oracle_unrestricted"], .75)
        self.assertFalse(report["gate"]["pass"])
        rows[0] = {key: rows[0][key] for key in ("sequence","frame","actionable")}
        rows[0].update(status="failed", reason="manufactured technical failure")
        report = c.summarize(rows)
        self.assertFalse(report["eligible"]); self.assertIsNone(report["pooled"])
        self.assertEqual(len(report["rows"]), 48)
        with self.assertRaises(ValueError):
            c.summarize(rows[:-1])
        duplicate = copy.deepcopy(rows); duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            c.summarize(duplicate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
