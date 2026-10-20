"""Manufactured-only audit arithmetic checks; prepared, not executed."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("independent_frontend_audit",
                                            Path(__file__).with_name("audit_frontend_saved.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class WitnessChecks(unittest.TestCase):
    def call(self, labels, sources, targets, scores=None):
        sources, targets = np.asarray(sources, float), np.asarray(targets, float)
        ids = np.column_stack((np.arange(len(sources)), np.arange(len(sources))))
        scores = np.asarray(scores if scores is not None else [1.]*len(sources))
        return audit.independent_witnesses(labels, sources, targets, ids, scores,
                                          (64,64), "manufactured", 4)

    def test_complete_binary_labels_and_void(self):
        labels = np.zeros((9,9), np.uint8); labels[2:7,2:7] = 2; labels[0,8] = 255
        r = self.call(labels, [[4,4],[0,0],[2,2],[8,0]], [[1,1],[2,2],[3,3],[4,4]])
        self.assertEqual([p["match_index"] for p in r["positive"]], [0])
        self.assertEqual([p["match_index"] for p in r["negative"]], [1])
        self.assertEqual(r["counts"]["boundary_or_void"], 2)
        self.assertFalse(r["actionable"])

    def test_opposite_collision_drops_all_same_label_uses_score_then_index(self):
        labels = np.zeros((9,9), np.uint8); labels[2:7,2:7] = 1
        r = self.call(labels, [[4,4],[0,0],[3,3],[4,4],[5,5]],
                      [[2.1,2],[2.2,2],[9.1,9],[9.2,9],[9.3,9]], [.9,.2,.3,.8,.8])
        self.assertEqual(r["counts"]["opposite_collision_points"], 2)
        self.assertEqual(r["counts"]["same_label_discarded"], 2)
        self.assertEqual([p["match_index"] for p in r["positive"]], [3])
        self.assertEqual(r["negative"], [])

    def test_round_half_up_invalid_and_nonfinite(self):
        labels = np.zeros((9,9), np.uint8); labels[2:7,2:7] = 1
        r = self.call(labels, [[4,4],[float("nan"),0],[8.5,0],[0,0]],
                      [[63.5,0],[0,0],[1,1],[63.49,1.5]])
        self.assertEqual(r["counts"]["invalid_coordinates"], 3)
        self.assertEqual(r["negative"][0]["rounded_current_xy"], [63,2])

    def test_rowmajor_round_robin_and_cap(self):
        labels = np.zeros((9,9), np.uint8); labels[2:7,2:7] = 300 % 256
        # Four cells, ten distinct pixels each. Within each cell confidence descends.
        targets = [[8*cell + (j % 5), 1+j//5] for cell in range(4) for j in range(10)]
        r = self.call(labels, [[4,4]]*40, targets, [10-j for _ in range(4) for j in range(10)])
        self.assertEqual(len(r["positive"]), 32)
        self.assertEqual([p["match_index"] for p in r["positive"][:8]], [0,10,20,30,1,11,21,31])
        self.assertEqual(r["counts"]["positive_before_cap"], 40)

    def test_nonbyte_id_and_empty_bank(self):
        labels = np.zeros((9,9), np.uint16); labels[2:7,2:7] = 300
        empty = np.empty((0,2))
        r = audit.independent_witnesses(labels, empty, empty, empty.astype(int),
                                        np.empty(0), (64,64), "manufactured", 4)
        self.assertEqual(r["target_id"], 300)
        self.assertEqual(r["counts"]["matches"], 0)
        self.assertFalse(r["actionable"])


if __name__ == "__main__":
    unittest.main()
