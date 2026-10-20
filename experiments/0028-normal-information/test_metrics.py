"""Manufactured metric and admission-boundary checks; no files/data/models."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "evr0028_metrics_tested", Path(__file__).with_name("metrics.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def pose(degrees=0.):
    theta = math.radians(degrees)
    return {"R": copy.deepcopy(IDENTITY), "t": [math.cos(theta), math.sin(theta), 0.]}


def population():
    rows, truth = [], {}
    for scene in m.SCENES:
        for query in m.QUERIES:
            rows.append({
                "scene": scene, "query": query,
                "anchor": scene + "/anchor.jpg", "target": scene + "/" + query + ".jpg",
                "arms": {arm: {"status": "completed", **pose(0 if arm == "ANG" else 1)}
                         for arm in m.ARMS}})
            truth[(scene, query)] = pose()
    return rows, truth


def boundary_summaries():
    """Exact synthetic summaries isolate Boolean gate boundaries from acos."""
    pooled = {arm: {"auc_percent": {"5": 60., "10": 75., "20": 80.},
                    "genuine_completed": 24}
              for arm in m.ARMS}
    pooled["ANG"] = {"auc_percent": {"5": 62., "10": 75., "20": 79.5},
                     "genuine_completed": 20}
    scenes = {scene: {arm: {"auc_percent": {"5": 60., "10": 75., "20": 80.},
                            "genuine_completed": 8}
                      for arm in m.ARMS}
              for scene in m.SCENES}
    for scene, delta in zip(m.SCENES, (4., 4., -2.)):
        scenes[scene]["ANG"]["auc_percent"]["5"] += delta
    return pooled, scenes


class MetricsTests(unittest.TestCase):
    def test_directed_translation_and_rotation(self):
        negative = m.pose_error(pose(180), pose())
        self.assertEqual(negative["translation_degrees"], 180.)
        self.assertEqual(negative["max_degrees"], 180.)
        rotated = pose()
        rotated["R"] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        self.assertEqual(m.pose_error(rotated, pose())["rotation_degrees"], 90.)
        # Huge/small finite translation magnitude does not alter its direction.
        for value in (1e308, 1e-308):
            scaled = pose()
            scaled["t"] = [value, 0., 0.]
            self.assertEqual(m.pose_error(scaled, pose())["max_degrees"], 0.)

    def test_dms_exact_threshold_ties_and_trapezoids(self):
        # At5: only0 enters -> area5*.25. At10: recalls0,.25,.5,.75
        # at x0,0,5,5; final plateau. These are exact independent areas.
        result = m.dms_auc_percent([0., 5., 5., 180.])
        self.assertEqual(result["5"], 25.)
        self.assertEqual(result["10"], 56.25)
        self.assertEqual(result["20"], 65.625)
        self.assertEqual(m.dms_auc_percent([5.] )["5"], 0.)
        self.assertEqual(m.dms_auc_percent([5.] )["10"], 75.)
        self.assertEqual(m.dms_auc_percent([0.] )["5"], 100.)

    def test_all_rows_retained_and_strongest_tie_order(self):
        rows, truth = population()
        result = m.evaluate_records(rows, truth, comparator_valid=True)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["gate"]["strongest_control_auc5"], "E")
        self.assertEqual(len(result["rows"]), 24)
        self.assertEqual(result["arm_order"], list(m.ARMS))
        self.assertFalse(result["manuscript_admission"])
        for scene in m.SCENES:
            self.assertEqual(result["scenes"][scene]["ANG"]["denominator"], 8)
        json.dumps(result, allow_nan=False)

    def test_unavailable_stays_in_denominator_and_fallback_not_genuine(self):
        rows, truth = population()
        rows[0]["arms"]["ANG"] = {"status": "unavailable", "R": None, "t": None}
        rows[1]["arms"]["ANG"]["status"] = "fallback"
        result = m.evaluate_records(rows, truth, comparator_valid=True)
        self.assertEqual(result["rows"][0]["arms"]["ANG"]["max_degrees"], 180.)
        self.assertEqual(result["rows"][1]["arms"]["ANG"]["max_degrees"], 0.)
        self.assertEqual(result["pooled"]["ANG"]["denominator"], 24)
        self.assertEqual(result["pooled"]["ANG"]["genuine_completed"], 22)
        self.assertAlmostEqual(result["pooled"]["ANG"]["auc_percent"]["5"],
                               100 * 23 / 24, places=12)

    def test_invalid_pose_penalized_without_projection_or_omission(self):
        for bad in (
                {"R": [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], "t": [1., 0., 0.]},
                {"R": IDENTITY, "t": [0., 0., 0.]},
                {"R": IDENTITY, "t": [float("nan"), 0., 0.]},
                {"R": [[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]], "t": [1., 0., 0.]}):
            with self.subTest(bad=bad):
                rows, truth = population()
                rows[0]["arms"]["ANG"] = {"status": "completed", **bad}
                result = m.evaluate_records(rows, truth, comparator_valid=True)
                error = result["rows"][0]["arms"]["ANG"]
                self.assertEqual(error["status"], "completed")
                self.assertFalse(error["pose_valid"])
                self.assertEqual(error["max_degrees"], 180.)
                self.assertEqual(result["pooled"]["ANG"]["genuine_completed"], 23)
                json.dumps(result, allow_nan=False)

    def test_identity_barrier_rejects_duplicate_missing_and_wrong_query(self):
        for mutation in ("duplicate", "missing", "wrong_query", "wrong_scene",
                         "truth_missing", "target_duplicate", "arm_missing"):
            with self.subTest(mutation=mutation):
                rows, truth = population()
                if mutation == "duplicate":
                    rows[1] = copy.deepcopy(rows[0])
                elif mutation == "missing":
                    rows.pop()
                elif mutation == "wrong_query":
                    rows[0]["query"] = "B9"
                elif mutation == "wrong_scene":
                    rows[0]["scene"] = "unallocated"
                elif mutation == "truth_missing":
                    truth.pop((m.SCENES[0], "B1"))
                elif mutation == "target_duplicate":
                    rows[1]["target"] = rows[0]["target"]
                else:
                    rows[0]["arms"].pop("FULL")
                with self.assertRaises(ValueError):
                    m.evaluate_records(rows, truth, comparator_valid=True)

    def test_invalid_truth_is_not_a_metric_penalty(self):
        rows, truth = population()
        truth[(m.SCENES[0], "B1")]["t"] = [0., 0., 0.]
        with self.assertRaises(ValueError):
            m.evaluate_records(rows, truth, comparator_valid=True)

    def test_four_gate_exact_boundaries_pass(self):
        pooled, scenes = boundary_summaries()
        gate = m.evaluate_gate(pooled, scenes, comparator_valid=True)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["auc5_margin_pp"], 2.)
        self.assertEqual(gate["ang_genuine_completed"], 20)
        self.assertEqual(gate["positive_scene_count"], 2)
        self.assertEqual(gate["auc20_margin_pp"], -.5)
        self.assertEqual(min(gate["scene_auc5_margins_pp"].values()), -2.)
        self.assertFalse(gate["manuscript_admission"])

    def test_gate_failures_are_separate_and_strict(self):
        for change, clause in (
                ("comparator", "trustworthy_comparators"),
                ("gain", "auc5_gain_and_coverage"),
                ("coverage", "auc5_gain_and_coverage"),
                ("positive_scenes", "scene_consistency"),
                ("scene_loss", "scene_consistency"),
                ("auc20", "auc20_noninferiority")):
            with self.subTest(change=change):
                pooled, scenes = boundary_summaries()
                comparator = True
                if change == "comparator":
                    comparator = False
                elif change == "gain":
                    pooled["ANG"]["auc_percent"]["5"] = math.nextafter(62., -math.inf)
                elif change == "coverage":
                    pooled["ANG"]["genuine_completed"] = 19
                elif change == "positive_scenes":
                    scenes[m.SCENES[0]]["ANG"]["auc_percent"]["5"] = 60.
                elif change == "scene_loss":
                    scenes[m.SCENES[2]]["ANG"]["auc_percent"]["5"] = math.nextafter(58., -math.inf)
                else:
                    pooled["ANG"]["auc_percent"]["20"] = math.nextafter(79.5, -math.inf)
                gate = m.evaluate_gate(pooled, scenes, comparator_valid=comparator)
                self.assertFalse(gate["passed"])
                self.assertFalse(gate["clauses"][clause])
                self.assertEqual(sum(not v for v in gate["clauses"].values()), 1)

    def test_best_auc20_and_scene_reference_are_fixed_comparisons(self):
        pooled, scenes = boundary_summaries()
        pooled["AFFINE"]["auc_percent"]["5"] = 61.
        pooled["ANG"]["auc_percent"]["5"] = 63.
        pooled["FULL"]["auc_percent"]["20"] = 81.
        # Scene comparators are the strongest pooledAUC5 arm AFFINE, not each
        # scene's best arm (NORMAL is made stronger only inside each scene).
        for scene in m.SCENES:
            scenes[scene]["NORMAL"]["auc_percent"]["5"] = 100.
        gate = m.evaluate_gate(pooled, scenes, comparator_valid=True)
        self.assertEqual(gate["strongest_control_auc5"], "AFFINE")
        self.assertEqual(gate["strongest_control_auc20"], "FULL")
        self.assertTrue(gate["clauses"]["scene_consistency"])
        self.assertEqual(gate["auc20_margin_pp"], -1.5)
        self.assertFalse(gate["clauses"]["auc20_noninferiority"])

    def test_flag_must_be_boolean(self):
        rows, truth = population()
        with self.assertRaises(ValueError):
            m.evaluate_records(rows, truth, comparator_valid=1)


if __name__ == "__main__":
    unittest.main()
