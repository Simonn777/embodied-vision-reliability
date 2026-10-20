"""Manufactured geometry checks only; no runtime/model/cache import or data access."""
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

_loader = importlib.util.spec_from_file_location("crop_detail_geometry", Path(__file__).with_name("refine.py"))
refine = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(refine)


class CropGeometryTests(unittest.TestCase):
    def test_padding_pixel_extent_and_edt_tie(self):
        mask = np.zeros((12, 16), dtype=bool)
        mask[3:7, 4:10] = True
        value = refine.crop_geometry(mask, cv2)
        self.assertEqual(value["bounds_xyxy"], [3, 2, 11, 8])
        self.assertEqual(value["padding_xy"], [1, 1])
        self.assertEqual(value["point_original_xy"], [5, 4])
        self.assertEqual(value["point_crop_xy"], [2, 2])
        self.assertEqual(value["box_crop_xyxy"], [0, 0, 7, 5])
        self.assertEqual(value["edt_distance"], 2.0)

    def test_image_edge_and_external_background(self):
        mask = np.ones((4, 4), dtype=bool)
        value = refine.crop_geometry(mask, cv2)
        self.assertEqual(value["bounds_xyxy"], [0, 0, 4, 4])
        self.assertEqual(value["point_original_xy"], [1, 1])
        self.assertEqual(value["edt_distance"], 2.0)

    def test_empty_and_one_pixel(self):
        mask = np.zeros((7, 9), dtype=bool)
        self.assertIsNone(refine.crop_geometry(mask, cv2))
        mask[0, 8] = True
        value = refine.crop_geometry(mask, cv2)
        self.assertEqual(value["bounds_xyxy"], [7, 0, 9, 2])
        self.assertEqual(value["point_crop_xy"], [1, 0])
        self.assertEqual(value["edt_distance"], 1.0)

    def test_crop_inverse_changes_only_crop_and_keeps_empty_result(self):
        native = np.ones((7, 9), dtype=bool)
        patch = np.zeros((3, 4), dtype=bool)
        answer = refine.stitch(native, patch, [2, 1, 6, 4])
        expected = native.copy()
        expected[1:4, 2:6] = False
        np.testing.assert_array_equal(answer, expected)
        self.assertTrue(native.all())
        entirely_empty = refine.stitch(native, np.zeros_like(native), [0, 0, 9, 7])
        self.assertFalse(entirely_empty.any())
        with self.assertRaises(ValueError):
            refine.stitch(native, patch, [2, 1, 7, 4])

    def test_highest_piou_stable_tie_and_nonfinite_rejection(self):
        self.assertEqual(refine.select(np.array([0.4, 0.9, 0.9])), 1)
        self.assertEqual(refine.select(np.zeros(3)), 0)
        self.assertEqual(refine.select(np.array([-2.0, -1.0, -3.0])), 1)
        with self.assertRaises(ValueError):
            refine.select(np.array([0.1, np.nan, 0.2]))

    def test_fixed_complete_frame_plan(self):
        train = [refine.frame_plan(n, role) for _, n, role in refine.COHORT[:4]]
        dev = [refine.frame_plan(n, role) for _, n, role in refine.COHORT[4:]]
        self.assertEqual(sum(map(len, train)), 87)
        self.assertEqual(sum(map(len, dev)), 154)
        self.assertEqual(sum(2 * len(refine.anchors_for(n)) for _, n, _ in refine.COHORT[:4]), 96)
        for (_, n, role), frames in zip(refine.COHORT, train + dev):
            self.assertEqual(frames, sorted(set(frames)))
            self.assertTrue(all(0 <= f < n for f in frames))
            if role == "DEV":
                self.assertEqual(frames, list(range(n)))
            else:
                self.assertNotIn(0, frames)


if __name__ == "__main__":
    cv2.setNumThreads(0)
    unittest.main()
