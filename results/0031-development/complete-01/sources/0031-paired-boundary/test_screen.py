"""Manufactured geometry/arithmetic checks only; no cache/model/file reads."""
import unittest
import numpy as np
import screen as s


class GeometryTests(unittest.TestCase):
    def test_left_half_crossings_and_signed_localization(self):
        label = np.zeros((9, 12), dtype=np.uint8)
        label[:, :5] = 1
        rows = s.crossings(label == 1)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["axis"] == 0 and row["sign"] == 1 for row in rows))
        row = rows[4]
        self.assertEqual(row["center"], [4.5, 4.0])
        center, xy, valid = s.endpoints(row, 0, 2, label.shape)
        self.assertTrue(valid)
        np.testing.assert_array_equal(xy, [[2.5, 4], [6.5, 4]])
        self.assertEqual(s.label_pair(label, xy, 1), ([True, False], 0))
        metric = s.candidate_metric(row, dict(center=center.tolist(), offset_pixels=0),
                                    label, 1, 2, {(0, 1, 4): [4.5]})
        self.assertEqual(metric["joint_loss"], 0)
        self.assertEqual(metric["signed_error_pixels"], 0)
        moved, _, _ = s.endpoints(row, 2, 2, label.shape)
        metric = s.candidate_metric(row, dict(center=moved.tolist(), offset_pixels=2),
                                    label, 1, 2, {(0, 1, 4): [4.5]})
        self.assertEqual(metric["signed_error_pixels"], 2)
        self.assertEqual(metric["localization_loss"], .5)
        # Farther wide-endpoint straddling cannot erase the center displacement.
        self.assertGreater(metric["joint_loss"], 0)

    def test_border_is_not_clipped_and_fixed_native_fallback(self):
        prediction = np.zeros((8, 10), dtype=bool)
        prediction[:, :1] = True
        feature = np.ones((8, 10, 3), dtype=np.float64)
        bank = np.array([[[1., 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1.]]])
        ranked = s.rank_frame(prediction, feature, bank, [1, 0], 2)
        self.assertEqual(len(ranked["queries"]), 8)
        for q in ranked["queries"]:
            self.assertEqual(q["candidates"][0]["reason"], "out_of_image")
            self.assertLess(q["candidates"][0]["endpoints"][0][0], 0)
            self.assertTrue(all(x["index"] == 0 for x in q["selection"].values()))
            self.assertEqual(q["selection"]["paired"]["fallback_reason"], "native_candidate_unavailable")

    def test_feature_pixel_center_map_and_normalization(self):
        feature = np.array([[[1., 0], [0, 1]], [[-1., 0], [0, -1.]]])
        # Original 4x4 pixel centers .5,2.5 map exactly to 2x2 feature centers.
        xy = np.array([[.5, .5], [2.5, .5], [.5, 2.5]])
        got, valid = s.descriptors(feature, xy, (4, 4))
        np.testing.assert_array_equal(valid, [True, True, True])
        np.testing.assert_allclose(got, [[1, 0], [0, 1], [-1, 0]], atol=0, rtol=0)
        got2, _ = s.descriptors(13*feature, xy, (4, 4))
        np.testing.assert_array_equal(got, got2)


class ScoreTests(unittest.TestCase):
    def test_derangement_marginals_and_pair_order_invariance(self):
        bank = np.arange(30, dtype=float).reshape(5, 2, 3)+1
        bank /= np.linalg.norm(bank, axis=-1, keepdims=True)
        permutation = s.derangement(5, "manufactured")
        self.assertEqual(sorted(permutation), list(range(5)))
        self.assertTrue(all(i != j for i, j in enumerate(permutation)))
        self.assertEqual(permutation, s.derangement(5, "manufactured"))
        np.testing.assert_array_equal(np.sort(bank[:, 1], axis=0),
                                      np.sort(bank[permutation, 1], axis=0))
        current = np.array([[1., 0, 0], [0, 1., 0]])
        before, _ = s.scores(current, bank, permutation)
        reordered = bank[[4, 2, 0, 3, 1]]
        after, _ = s.scores(current, reordered, s.derangement(5, "irrelevant"))
        self.assertAlmostEqual(before["paired"], after["paired"], places=14)
        self.assertAlmostEqual(before["unpaired_max"], after["unpaired_max"], places=14)

    def test_concatenated_nearest_neighbor_and_reversed_sign(self):
        bank = np.array([[[1., 0, 0], [0, 1, 0]], [[0, 0, 1.], [-1., 0, 0]]])
        current = np.array([[.8, .6, 0], [0, .6, .8]])
        result, components = s.scores(current, bank, [1, 0])
        q = current.reshape(-1)
        forward_dist = np.sum((bank.reshape(2, -1)-q)**2, axis=1)
        reverse_dist = np.sum((bank[:, ::-1].reshape(2, -1)-q)**2, axis=1)
        self.assertAlmostEqual(result["paired"], (reverse_dist.min()-forward_dist.min())/2, places=14)
        reverse, _ = s.scores(current[::-1], bank, [1, 0])
        for arm in ("paired", "unpaired_max", "deranged"):
            self.assertAlmostEqual(result[arm], -reverse[arm], places=14)
        self.assertAlmostEqual(result["paired"], result["unpaired_max"]
                               - components["forward_gap"] + components["reverse_gap"], places=14)

    def test_joint_association_can_change_without_marginal_change(self):
        bank = np.array([[[1., 0, 0], [0, 1, 0]], [[0, 0, 1.], [-1., 0, 0]]])
        current = np.array([[1., 0, 0], [0, 1., 0]])
        result, _ = s.scores(current, bank, [1, 0])
        shuffled = bank.copy()
        shuffled[:, 1] = bank[[1, 0], 1]
        control, _ = s.scores(current, shuffled, [1, 0])
        self.assertEqual(result["unpaired_max"], control["unpaired_max"])
        self.assertNotEqual(result["paired"], control["paired"])
        self.assertEqual(result["deranged"], control["paired"])

    def test_absent_gt_crossing_worst_and_empty_event_retained(self):
        label = np.zeros((8, 10), dtype=np.uint8)
        query = dict(axis=0, sign=1, center=[4.5, 4])
        candidate = dict(center=[4.5, 4], offset_pixels=0)
        metric = s.candidate_metric(query, candidate, label, 1, 2, {})
        self.assertIsNone(metric["signed_error_pixels"])
        self.assertEqual(metric["joint_loss"], 1)
        self.assertEqual(metric["localization_loss"], 1)
        self.assertEqual(metric["endpoint_class"], "both_background")
        result = s.measure_frame(dict(queries=[]), label, 1, 2)
        self.assertEqual(result["crossing_count"], 0)
        self.assertIsNone(result["means"]["paired"]["joint_loss"])
        self.assertEqual(result["delta_from_native"]["paired"]["joint_loss"], 0)


if __name__ == "__main__":
    unittest.main()
