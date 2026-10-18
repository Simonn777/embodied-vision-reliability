"""Analytic, generated fixtures; no stored labels or model imports."""
from copy import deepcopy
import unittest

import numpy as np

from evr.state_intervention_metrics import summarize


def fixture(points=1, frames=5, starts=(2,), horizon=2, controls=(0, 4)):
    gt = np.zeros((points, frames, 2), dtype=np.float64)
    occ = np.zeros((points, frames), dtype=bool)
    rows = []
    for frame in range(frames):
        branches = [("keep", -1, -1), ("zero", -1, -1)]
        if frame in controls:
            branches.append(("clone_control", -1, -1))
        branches.extend(("shadow", start, frame - start) for start in starts
                        if start <= frame < start + horizon)
        for branch, start, offset in branches:
            for point in range(points):
                rows.append(dict(branch=branch, frame=frame, point_index=point,
                                 shadow_start=start, shadow_offset=offset, x=0.0, y=0.0,
                                 occlusion_logit=-1000.0, expected_distance_logit=-1000.0))
    kwargs = dict(shadow_starts=starts, shadow_horizon=horizon, control_frames=controls)
    return rows, gt, occ, list(range(points)), [1, 2, 4, 8, 16], kwargs


def run(f):
    rows, gt, occ, ids, thresholds, kwargs = f
    return summarize(rows, gt, occ, ids, thresholds, **kwargs)


def alter(rows, branch, frame, **fields):
    for row in rows:
        if row["branch"] == branch and row["frame"] == frame:
            row.update(fields)


class StateInterventionMetricTests(unittest.TestCase):
    def test_threshold_boundary_recovery_harm_and_visibility_independent_coordinates(self):
        f = fixture()
        rows, _, occ, *_ = f
        occ[0, 3] = True
        # Keep visible correct at exactly 4; zero wrong but rejected: still a harm.
        alter(rows, "keep", 1, x=4.0)
        alter(rows, "zero", 1, x=5.0, occlusion_logit=1000.0)
        # Keep wrong, zero correct but rejected: still a coordinate recovery.
        alter(rows, "keep", 2, x=5.0)
        alter(rows, "zero", 2, x=4.0, occlusion_logit=1000.0)
        alter(rows, "shadow", 2, x=4.0, occlusion_logit=1000.0)
        # Occluded prediction visible: a false positive even at zero coordinate error.
        result = run(f)
        changes = result["paired_coordinate_changes"]
        self.assertEqual((changes["baseline_visible_errors"], changes["recoveries"], changes["harms"]),
                         (1, 1, 1))
        keep = result["branch_summaries"]["keep"]
        at4 = keep["thresholds"][2]
        self.assertEqual((at4["true_positive_count"], at4["false_positive_count"]), (2, 2))
        self.assertAlmostEqual(at4["jaccard"], 2 / 5)
        self.assertAlmostEqual(keep["visibility_accuracy"], 3 / 4)
        self.assertEqual(result["per_query"][0]["mean_epe_zero_minus_keep_px"], 0)

    def test_zero_denominators_are_null_and_occluded_false_positive_is_not_null(self):
        f = fixture()
        f[2][:, 1:] = True
        for row in f[0]:
            row["occlusion_logit"] = 1000.0
        result = run(f)
        keep = result["branch_summaries"]["keep"]
        self.assertIsNone(keep["mean_visible_epe_px"])
        self.assertIsNone(keep["thresholds"][2]["visible_coordinate_error_rate"])
        self.assertIsNone(keep["thresholds"][2]["jaccard"])
        self.assertEqual(keep["visibility_accuracy"], 1)
        self.assertIsNone(result["per_query"][0]["mean_epe_zero_minus_keep_px"])
        alter(f[0], "keep", 1, occlusion_logit=-1000.0)
        result = run(f)
        self.assertEqual(result["branch_summaries"]["keep"]["thresholds"][2]["jaccard"], 0)

    def test_visibility_uses_strict_half_and_stable_extreme_logits(self):
        f = fixture()
        alter(f[0], "keep", 1, occlusion_logit=0.0, expected_distance_logit=-1000.0)
        result = run(f)
        self.assertEqual(result["branch_summaries"]["keep"]["predicted_visible_count"], 3)
        self.assertEqual(result["branch_summaries"]["keep"]["thresholds"][2]["jaccard"], 3 / 4)

    def test_normalized_xy_uses_256_and_not_255(self):
        f = fixture()
        f[1][0, 1] = [0.5, 0.25]
        alter(f[0], "keep", 1, x=128.0, y=64.0)
        alter(f[0], "zero", 1, x=131.0, y=68.0)
        result = run(f)
        self.assertEqual(result["per_query"][0]["mean_keep_epe_px"], 0)
        self.assertEqual(result["per_query"][0]["mean_zero_epe_px"], 1.25)
        self.assertEqual(result["paired_coordinate_changes"]["harms"], 1)

    def test_shadow_loss_includes_visibility_and_all_eight_or_custom_frames(self):
        f = fixture(points=2)
        # Start frame reset is correct but keep is wrong: both queries gain one.
        alter(f[0], "keep", 2, x=8.0)
        # Next frame only point 1 incurs a shadow false negative.
        for row in f[0]:
            if row["branch"] == "shadow" and row["frame"] == 3 and row["point_index"] == 1:
                row["occlusion_logit"] = 1000.0
        result = run(f)
        episodes = result["shadow_episodes"]
        self.assertEqual([episode["loss_delta_count"] for episode in episodes], [1, 0])
        self.assertEqual(episodes[0]["frame_indices"], [2, 3])
        self.assertEqual(episodes[0]["keep_losses"], [1, 0])
        self.assertEqual(episodes[0]["shadow_losses"], [0, 0])
        self.assertEqual(episodes[1]["frame_loss_deltas"], [1, -1])
        self.assertEqual(result["shadow_aggregate"]["mean_return_keep_minus_shadow"], .25)
        self.assertEqual(result["shadow_aggregate"]["queries_with_beneficial_shadow"], [0])
        self.assertEqual(result["continuation"]["status"], "undersensitive")

    def test_fixed_continuation_gate_and_harms_retained(self):
        f = fixture(points=2, frames=12, starts=(2,), horizon=2, controls=(0,))
        for row in f[0]:
            if row["branch"] == "keep" and row["frame"] > 0:
                row["x"] = 8.0
        result = run(f)
        self.assertEqual(result["paired_coordinate_changes"]["baseline_visible_errors"], 22)
        self.assertEqual(result["paired_coordinate_changes"]["recoveries"], 22)
        self.assertEqual(result["shadow_aggregate"]["mean_return_keep_minus_shadow"], 1)
        self.assertEqual(result["continuation"]["status"], "continue_independent_development")
        # Exactly 20 baseline errors still permits continuation.
        alter(f[0], "keep", 11, x=0.0)
        self.assertEqual(run(f)["paired_coordinate_changes"]["baseline_visible_errors"], 20)
        self.assertEqual(run(f)["continuation"]["status"], "continue_independent_development")
        # Keep opportunities fixed, but one query has no beneficial shadow.
        for row in f[0]:
            if row["point_index"] == 1 and row["branch"] in ("zero", "shadow") and row["frame"] in (2, 3):
                row["x"] = 8.0
        self.assertEqual(run(f)["continuation"]["status"], "stop_this_candidate")

    def test_frame_zero_is_excluded_but_integrity_checked(self):
        f = fixture()
        for row in f[0]:
            if row["frame"] == 0:
                row["x"] = 100.0
        result = run(f)
        self.assertEqual(result["branch_summaries"]["keep"]["mean_visible_epe_px"], 0)
        self.assertEqual(result["branch_summaries"]["keep"]["post_query_rows"], 4)
        alter(f[0], "zero", 0, x=99)
        with self.assertRaisesRegex(ValueError, "frame_zero"):
            run(f)

    def test_missing_duplicate_wrong_offset_and_point_order_rejected(self):
        original = fixture(points=2)
        for kind in ("missing", "duplicate", "offset", "order"):
            f = deepcopy(original)
            if kind == "missing":
                f[0].pop()
            elif kind == "duplicate":
                f[0].append(deepcopy(f[0][0]))
            elif kind == "offset":
                next(row for row in f[0] if row["branch"] == "shadow")["shadow_offset"] = 7
            else:
                f[0][0], f[0][1] = f[0][1], f[0][0]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                run(f)

    def test_nonfinite_values_bad_types_and_control_disagreement_rejected(self):
        for key, value in (("x", float("nan")), ("y", float("inf")),
                           ("occlusion_logit", "-1"), ("point_index", True)):
            f = fixture()
            f[0][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run(f)
        for branch, frame, message in (("clone_control", 4, "clone_control"),
                                        ("shadow", 2, "shadow_start")):
            f = fixture()
            alter(f[0], branch, frame, x=.001)
            with self.assertRaisesRegex(ValueError, message):
                run(f)

    def test_original_point_ids_index_full_gt_and_missing_shadows_are_not_zero_return(self):
        f = fixture(points=2)
        # Keep only original point 1 and retain the full two-point GT.
        f[0][:] = [row for row in f[0] if row["point_index"] == 1]
        f[3][:] = [1]
        f[1][0, 1:] = 100.0  # Unselected point must not enter any error aggregate.
        result = run(f)
        self.assertEqual(result["per_query"][0]["point_index"], 1)
        self.assertEqual(result["per_query"][0]["mean_keep_epe_px"], 0)
        f[0][:] = [row for row in f[0] if row["branch"] != "shadow"]
        with self.assertRaisesRegex(ValueError, "missing"):
            run(f)

    def test_finite_rows_cannot_silently_produce_nonfinite_error_mean(self):
        f = fixture()
        for row in f[0]:
            row["x"] = 1e308
        with self.assertRaisesRegex(ValueError, "aggregate"):
            run(f)


if __name__ == "__main__":
    unittest.main()
