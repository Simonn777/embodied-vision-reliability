"""Manufactured tests only; no files, media decoding, models, or network."""

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "evr0037_features_tested", Path(__file__).with_name("features.py"),
)
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import cv2
        cls.np, cls.cv2 = np, cv2

    def scene(self):
        np = self.np
        yy, xx = np.indices((12, 20))
        rgb = np.stack((xx * 7, yy * 9, xx * 3 + yy * 4), axis=2).astype(np.uint8)
        state = np.full((12, 20), 2, dtype=np.uint8)
        state[:, 2:7] = 1
        state[3:5, 4:6] = 0
        return rgb, state

    def test_initial_void_and_class_erosions(self):
        np, cv2 = self.np, self.cv2
        rgb = np.zeros((16, 24, 3), dtype=np.uint8)
        initial = np.zeros((16, 24), dtype=np.uint8)
        initial[:, :10] = 1
        initial[5:11, 10:14] = 255
        snapshots = [x.copy() for x in (rgb, initial)]
        state, small, ev = features.initialize(rgb, initial)
        self.assertEqual(small.shape, (341, 512, 3))
        resized = cv2.resize(initial, (512, 341), interpolation=cv2.INTER_NEAREST)
        self.assertTrue(np.all(state[resized == 255] == 0))
        for y, x in ((150, 100), (150, 450), (0, 100), (150, 250)):
            expected = 0
            if 0 < y < 340 and 0 < x < 511:
                window = resized[y-1:y+2, x-1:x+2]
                if np.all(window == 1):
                    expected = 1
                elif np.all(window == 0):
                    expected = 2
            self.assertEqual(int(state[y, x]), expected)
        self.assertTrue(np.array_equal(ev["valid"], state != 0))
        self.assertGreater(ev["metadata"]["initial_void_small"], 0)
        for original, saved in zip((rgb, initial), snapshots):
            np.testing.assert_array_equal(original, saved)

    def test_exact_translation_moves_positive_and_negative_labels(self):
        np = self.np
        previous, state = self.scene()
        current = np.zeros_like(previous)
        current[:, 2:] = previous[:, :-2]
        forward = np.zeros((*state.shape, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0], backward[..., 0] = 2.0, -2.0
        inputs = (previous, current, state, forward, backward)
        copies = [x.copy() for x in inputs]
        result, returned_rgb, ev = features.propagate(*inputs)
        expected = np.zeros_like(state)
        expected[:, 2:] = state[:, :-2]
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(returned_rgb, current)
        self.assertEqual(int(result[1, 2]), 2)  # formerly target at current x=2
        self.assertEqual(int(result[1, 8]), 1)  # formerly background at x=8
        self.assertTrue(np.all(ev["fb"][:, 2:] == 0))
        self.assertTrue(np.all(ev["photo"][:, 2:] == 0))
        self.assertTrue(np.all(~ev["in_bounds"][:, :2]))
        for original, saved in zip(inputs, copies):
            np.testing.assert_array_equal(original, saved)

    def test_bad_consistency_photo_and_coordinates_become_unknown(self):
        np = self.np
        previous = np.full((12, 20, 3), 100, dtype=np.uint8)
        current = previous.copy()
        state = np.ones((12, 20), dtype=np.uint8)
        forward = np.zeros((12, 20, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[4, 4, 0] = 1.51
        current[5, 5] = 121
        current[6, 6] = 120  # equality at the specified threshold passes
        backward[0, 0, 0] = -0.25
        result, _, ev = features.propagate(previous, current, state, forward, backward)
        for y, x in ((4, 4), (5, 5), (0, 0)):
            self.assertEqual(int(result[y, x]), 0)
        self.assertEqual(int(result[6, 6]), 1)
        self.assertEqual(float(ev["photo"][6, 6]), 20.0 / 255.0)
        self.assertTrue(np.isfinite(ev["fb"]).all())
        self.assertTrue(np.isfinite(ev["photo"]).all())

    def test_unknown_cannot_resurrect_on_later_reliable_step(self):
        np = self.np
        rgb, state = self.scene()
        zeros = np.zeros((*state.shape, 2), dtype=np.float32)
        bad_forward = zeros.copy()
        bad_forward[7, 8, 0] = 2.0
        first, _, _ = features.propagate(rgb, rgb, state, bad_forward, zeros)
        second, _, ev = features.propagate(rgb, rgb, first, zeros, zeros)
        self.assertEqual(int(second[7, 8]), 0)
        self.assertTrue(ev["reliable"][7, 8])
        self.assertFalse(ev["valid"][7, 8])
        np.testing.assert_array_equal(second[first == 0], np.zeros(np.count_nonzero(first == 0)))

    def test_cells_match_scalar_geometry_and_means(self):
        np = self.np
        ids = features.cell_ids((9, 13), (7, 10))
        expected = np.empty((9, 13), dtype=np.int64)
        for y in range(9):
            for x in range(13):
                expected[y, x] = ((y * 7 // 9) // 4) * 3 + ((x * 10 // 13) // 4)
        np.testing.assert_array_equal(ids, expected)
        values = np.stack((ids % 2 == 0, ids % 2 == 1), axis=2)
        means, counts = features.cell_means(values, ids, n_cells=6)
        self.assertEqual(int(counts.sum()), 9 * 13)
        for cell in range(6):
            picked = values[ids == cell]
            self.assertEqual(int(counts[cell]), len(picked))
            np.testing.assert_array_equal(means[cell], picked.mean(axis=0))
        sparse_ids = features.cell_ids((3, 3), (9, 9))
        sparse_means, sparse_counts = features.cell_means(
            np.ones((3, 3)), sparse_ids, n_cells=features.cell_count((9, 9)),
        )
        self.assertTrue(np.any(sparse_counts == 0))
        self.assertTrue(np.all(sparse_means[sparse_counts == 0] == 0))

    def test_advance_uses_only_supplied_previous_labels(self):
        np = self.np
        original = np.full((12, 20, 3), 90, dtype=np.uint8)
        small = features.resize_rgb(original)
        state = np.full(small.shape[:2], 2, dtype=np.uint8)
        state[30:50, 30:50] = 1
        state[40, 40] = 0
        zeros = np.zeros((*state.shape, 2), dtype=np.float32)
        with mock.patch.object(features, "_estimate_flows", return_value=(zeros, zeros)) as estimate:
            result, _, ev = features.advance(small, original, state)
        estimate.assert_called_once()
        np.testing.assert_array_equal(result, state)
        self.assertEqual(ev["metadata"]["flow_estimation_calls"], 2)
        self.assertEqual(ev["metadata"]["phase"], "advance")

    def test_dis_pair_api_on_manufactured_constant_images(self):
        """API/finite-output qualification, not a claim of flow identifiability."""
        np = self.np
        original = np.full((17, 25, 3), 90, dtype=np.uint8)
        previous = features.resize_rgb(original)
        state = np.full(previous.shape[:2], 2, dtype=np.uint8)
        state[40:50, 40:50] = 0
        snapshots = [x.copy() for x in (previous, original, state)]
        result, small, evidence = features.advance(previous, original, state)
        self.assertEqual(result.shape, previous.shape[:2])
        self.assertEqual(small.shape, previous.shape)
        self.assertTrue(np.all((result == 0) | (result == 2)))
        self.assertEqual(evidence["metadata"]["flow_estimation_calls"], 2)
        for name in ("fb", "photo", "flow_dx", "flow_dy"):
            self.assertTrue(np.isfinite(evidence[name]).all())
        for value, snapshot in zip((previous, original, state), snapshots):
            np.testing.assert_array_equal(value, snapshot)

    def test_fixed_dis_public_configuration_and_validation(self):
        np, cv2 = self.np, self.cv2
        dis = features._make_dis(cv2)
        for suffix, key in features._DIS_PARAMETERS:
            self.assertEqual(getattr(dis, "get" + suffix)(), features.CONFIG[key])
        rgb, state = self.scene()
        flow = np.zeros((*state.shape, 2), dtype=np.float32)
        invalid = flow.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            features.propagate(rgb, rgb, state, invalid, flow)
        badstate = state.copy()
        badstate[0, 0] = 3
        with self.assertRaises(ValueError):
            features.propagate(rgb, rgb, badstate, flow, flow)
        with self.assertRaises(ValueError):
            features.resize_rgb(rgb.astype(np.float32))
        with self.assertRaises(ValueError):
            features.initialize(rgb, np.full(state.shape, 2, dtype=np.uint8))
        with self.assertRaises(ValueError):
            features.cell_means(np.full((2, 2), np.nan), np.zeros((2, 2), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
