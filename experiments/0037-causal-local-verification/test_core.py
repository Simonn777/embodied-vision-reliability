"""Manufactured cell arithmetic only; no files, models or scientific inputs."""
import inspect
import unittest

import numpy as np

import core


def masks_fixture():
    truth = np.array([[1, 0, 255, 1], [0, 1, 0, 0]], np.uint8)
    native = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], np.uint8)
    m1, m2 = native.copy(), native.copy()
    m1[0, 1], m1[0, 3] = 1, 1
    m2[1, 0], m2[1, 3] = 1, 1
    controls = np.stack((native, m1, m2, m1 | m2))
    ids = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], np.int64)
    return truth, controls, ids


def evidence_fixture(shape):
    return {"fg": np.zeros(shape, bool), "bg": np.ones(shape, bool),
            "unknown": np.zeros(shape, bool), "valid": np.ones(shape, bool),
            "fb": np.full(shape, 2.0), "photo": np.full(shape, 0.2),
            "flow_dx": np.full(shape, 1024, np.float32),
            "flow_dy": np.full(shape, -256, np.float32)}


class CoreTests(unittest.TestCase):
    def test_literal_counts_void_and_native_extra_partition(self):
        truth, masks, ids = masks_fixture()
        rows = core.label_counts(truth, masks, ids)
        row = rows["author_background"]
        self.assertEqual(row["total_gt"], 3)
        np.testing.assert_array_equal(row["union_tp"], [2, 1])
        np.testing.assert_array_equal(row["union_fp"], [2, 2])
        np.testing.assert_array_equal(row["native_tp"], [2, 0])
        np.testing.assert_array_equal(row["native_fp"], [0, 1])
        np.testing.assert_array_equal(row["extra_tp"], [0, 1])
        np.testing.assert_array_equal(row["extra_fp"], [2, 1])
        np.testing.assert_array_equal(row["fixed_tp"], [2, 3, 2, 3])
        np.testing.assert_array_equal(row["fixed_fp"], [1, 2, 3, 4])
        np.testing.assert_array_equal(rows["void_ignore"]["fixed_fp"], [0, 1, 2, 3])

    def test_same_cell_native_and_added_actions_score_original_pixels(self):
        truth, masks, ids = masks_fixture()
        counts = core.label_counts(truth, masks, ids)["author_background"]
        result = core.score_actions(counts, np.array([True, False]), np.array([False, True]))
        self.assertEqual(result, {"J": 0.75, "tp": 3, "fp": 1,
                                 "removed_native_tp": 0, "removed_native_fp": 1,
                                 "removed_extra_tp": 0, "removed_extra_fp": 2})
        # A cell mixes foreground/background; its action is not a point-label proof.
        all_keep = np.ones(2, bool)
        self.assertEqual(core.score_actions(counts, all_keep, all_keep)["J"], 3 / 7)

    def test_empty_union_and_explicit_unused_cells(self):
        truth = np.zeros((2, 2), np.uint8)
        masks = np.zeros((4, 2, 2), np.uint8)
        ids = np.array([[0, 2], [0, 2]], np.int64)
        rows = core.label_counts(truth, masks, ids, n_cells=4)
        for row in rows.values():
            self.assertEqual(len(row["union_tp"]), 4)
            self.assertEqual(core.score_actions(row, np.ones(4, bool), np.ones(4, bool))["J"], 1)
        self.assertEqual(core.sample_indices(rows["author_background"]["union_tp"]).size, 0)
        truth[0, 0] = 1
        row = core.label_counts(truth, masks, ids, n_cells=4)["author_background"]
        self.assertEqual(core.score_actions(row, np.ones(4, bool), np.ones(4, bool))["J"], 0)

    def test_feature_units_partial_cells_input_immutability_and_no_truth_api(self):
        rgb = np.full((5, 6, 3), [64, 128, 192], np.uint8)
        logits = np.stack([np.full((10, 12), v, np.float32) for v in (0, 32, -64)])
        masks = np.ones((4, 10, 12), np.uint8)
        evidence = evidence_fixture((5, 6))
        before = [v.copy() for v in (rgb, logits, masks, *evidence.values())]
        X, ids, counts, metadata = core.build_features(rgb, evidence, logits, masks, 300)
        self.assertEqual(X.shape, (4, 27))
        self.assertEqual(X.dtype, np.float32)
        np.testing.assert_array_equal(counts, [64, 32, 16, 8])
        np.testing.assert_array_equal(ids[:8, :8], np.zeros((8, 8), np.int64))
        np.testing.assert_allclose(X[:, :3], np.tile(np.array([64, 128, 192]) / 255, (4, 1)), atol=1e-7)
        np.testing.assert_allclose(X[:, 3:6], 0, atol=1e-7)
        np.testing.assert_array_equal(X[:, 6:9], np.tile([0, 1, -1], (4, 1)))
        np.testing.assert_array_equal(X[:, 9:12], 0)
        np.testing.assert_array_equal(X[:, 12:16], 1)
        np.testing.assert_allclose(X[:, 16:18], [[1/3, .4], [5/6, .4], [1/3, .9], [5/6, .9]], atol=1e-7)
        np.testing.assert_allclose(X[:, 18:], np.tile([0, 1, 0, 1, .2, .2, 1, -.5, 1], (4, 1)), atol=1e-7)
        self.assertEqual(metadata["union_pixels"], 120)
        for old, actual in zip(before, (rgb, logits, masks, *evidence.values())):
            np.testing.assert_array_equal(old, actual)
        self.assertEqual(tuple(inspect.signature(core.build_features).parameters),
                         ("rgb_small", "evidence", "original_logits", "controls", "frame"))
        self.assertEqual(core.TRACK_COLUMNS, tuple(range(18, 27)))

    def test_fixed_thresholds_and_unknown_policy(self):
        X = np.zeros((3, 27), np.float32)
        X[:, 18:21] = [[.25, .75, 0], [.75, .25, 0], [0, 0, 1]]
        result = core.fixed_actions(X)
        np.testing.assert_array_equal(result["fixed_negative"][0], [False, True, True])
        np.testing.assert_array_equal(result["fixed_negative"][1], [False, True, True])
        np.testing.assert_array_equal(result["positive_required"][0], [True, True, True])
        np.testing.assert_array_equal(result["positive_required"][1], [False, True, False])

    def test_sampling_is_row_major_label_free_and_capped(self):
        counts = np.zeros(2050, np.int64)
        counts[::2] = 1
        sample = core.sample_indices(counts)
        self.assertEqual(len(sample), 512)
        self.assertEqual((int(sample[0]), int(sample[-1])), (0, 2048))
        self.assertTrue(np.all(np.diff(sample) > 0))
        self.assertTrue(np.all(sample % 2 == 0))
        np.testing.assert_array_equal(core.sample_indices(np.array([0, 2, 0, 9])), [1, 3])

    def test_invalid_counts_masks_features_do_not_silently_filter(self):
        truth, masks, ids = masks_fixture()
        broken = masks.copy()
        broken[3, 0, 0] = 0
        with self.assertRaises(ValueError):
            core.label_counts(truth, broken, ids)
        with self.assertRaises(ValueError):
            core.label_counts(truth.astype(float), masks, ids)
        X = np.zeros((1, 27), np.float32)
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            core.fixed_actions(X)
        counts = core.label_counts(truth, masks, ids)["author_background"]
        counts["extra_tp"][0] += 1
        with self.assertRaises(ValueError):
            core.score_actions(counts, np.ones(2, bool), np.ones(2, bool))


if __name__ == "__main__":
    unittest.main()
