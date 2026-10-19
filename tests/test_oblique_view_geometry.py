"""Hand-computed geometry and probability fixtures, never videos or model calls."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts" / "oblique_view_geometry.py"
SPEC = importlib.util.spec_from_file_location("oblique_geometry", PATH)
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


class CoordinateTests(unittest.TestCase):
    def test_centers_and_positive_rotation(self):
        grid = geometry.pixel_center_grid()
        np.testing.assert_array_equal(grid[7, 11], [11.5, 7.5])
        for name, transform in geometry.fixed_view_affines().items():
            with self.subTest(view=name):
                np.testing.assert_array_equal(geometry.map_points([128, 128], transform), [128, 128])
        point = geometry.map_points([138, 128], geometry.view_affine(90, 0.7))
        np.testing.assert_array_equal(point, [128, 135])
        self.assertEqual(point.dtype, np.float32)

    def test_45_degree_mapping(self):
        mapped = geometry.map_points([138, 128], geometry.view_affine(45, 0.7))
        np.testing.assert_allclose(mapped, [128 + 7 / np.sqrt(2)] * 2, atol=1e-5, rtol=0)

    def test_fixed_views_preserve_whole_original_footprint(self):
        corners = np.array([[0, 0], [256, 0], [256, 256], [0, 256]])
        for name, matrix in geometry.fixed_view_affines().items():
            with self.subTest(view=name):
                points = geometry.map_points(corners, matrix)
                self.assertTrue(np.all(points >= 0) and np.all(points <= 256))
        oblique = geometry.map_points(corners, geometry.view_affine(45, 0.7))
        expected_margin = 128 - 0.7 * 128 * np.sqrt(2)
        self.assertAlmostEqual(float(oblique.min()), expected_margin, places=5)

    def test_inverse_and_query_inputs_unchanged(self):
        points = np.array([[0.5, 255.5], [31.25, 88.75], [128, 128]], dtype=np.float32)
        before = points.copy()
        matrix = geometry.view_affine(45, 0.7)
        back = geometry.map_points(geometry.map_points(points, matrix), geometry.invert_affine(matrix))
        np.testing.assert_allclose(back, points, atol=2e-5, rtol=0)
        np.testing.assert_array_equal(points, before)

    def test_relative_view_cancels_absolute_scale(self):
        views = geometry.fixed_view_affines()
        grid = geometry.reference_to_view_grid(views["reference"], views["oblique"])
        x, y = 180.5, 140.5
        c = 1 / np.sqrt(2)
        expected = [128 + c * ((x - 128) - (y - 128)),
                    128 + c * ((x - 128) + (y - 128))]
        np.testing.assert_allclose(grid[140, 180], expected, atol=1e-12, rtol=0)


class RasterTests(unittest.TestCase):
    def test_identity_is_exact_and_not_a_shared_array(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
        frame[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
        frame[0, 255, 2] = 137
        original = frame.copy()
        warped = geometry.warp_rgb(frame, geometry.view_affine(0, 1))
        np.testing.assert_array_equal(warped, frame)
        np.testing.assert_array_equal(frame, original)
        self.assertFalse(np.shares_memory(frame, warped))

    def test_exact_right_angle_and_inverse_raster(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[40, 180] = [10, 20, 30]
        matrix = geometry.view_affine(90, 1)
        warped = geometry.warp_rgb(frame, matrix)
        np.testing.assert_array_equal(warped[180, 215], [10, 20, 30])
        self.assertEqual(np.count_nonzero(warped), 3)
        np.testing.assert_array_equal(geometry.warp_rgb(warped, geometry.invert_affine(matrix)), frame)

    def test_inverse_direction_and_four_tap_weights(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[30, 40] = [100, 20, 4]
        transform = np.eye(3)
        transform[:2, 2] = [0.5, 0.5]
        output = geometry.warp_rgb_float(frame, transform)
        for y, x in ((30, 40), (30, 41), (31, 40), (31, 41)):
            np.testing.assert_array_equal(output[y, x], [25, 5, 1])
        self.assertEqual(np.count_nonzero(output), 12)
        self.assertEqual(output.dtype, np.float32)

    def test_round_ties_to_even(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[20, 20] = [1, 3, 5]
        transform = np.eye(3)
        transform[0, 2] = 0.5
        np.testing.assert_array_equal(geometry.warp_rgb_float(frame, transform)[20, 20], [.5, 1.5, 2.5])
        np.testing.assert_array_equal(geometry.warp_rgb(frame, transform)[20, 20], [0, 2, 2])

    def test_black_padding_and_half_pixel_transition(self):
        frame = np.full((256, 256, 3), 100, dtype=np.uint8)
        shrink = geometry.warp_rgb(frame, geometry.view_affine(0, 0.7))
        np.testing.assert_array_equal(shrink[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(shrink[128, 128], [100, 100, 100])
        translate = np.eye(3)
        translate[0, 2] = 0.5
        np.testing.assert_array_equal(geometry.warp_rgb_float(frame, translate)[10, 0], [50, 50, 50])


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.px = np.arange(1, 257, dtype=np.float64)
        self.px /= self.px.sum()
        self.py = self.px[::-1].copy()
        self.views = geometry.fixed_view_affines()

    def test_identity_and_axis_order(self):
        mx, my = geometry.map_marginal_probabilities(self.px, self.py,
                                                   self.views["original"], self.views["original"])
        self.assertEqual(mx[7, 11], self.px[11])
        self.assertEqual(my[7, 11], self.py[7])

    def test_right_angle_separable_control(self):
        mx, my = geometry.map_marginal_probabilities(self.px, self.py,
                                                   self.views["reference"], self.views["axis_control"])
        np.testing.assert_allclose(mx, np.broadcast_to(self.px[::-1, None], (256, 256)), atol=1e-16, rtol=0)
        np.testing.assert_allclose(my, np.broadcast_to(self.py[None, :], (256, 256)), atol=1e-16, rtol=0)

    def test_mid_bin_linear_interpolation(self):
        translate = np.eye(3)
        translate[0, 2] = 0.5
        mx, _ = geometry.map_marginal_probabilities(self.px, self.py, np.eye(3), translate)
        self.assertAlmostEqual(mx[20, 10], (self.px[10] + self.px[11]) / 2, places=16)
        self.assertEqual(mx[20, 255], geometry.PROBABILITY_FLOOR)

    def test_oblique_outside_and_zero_probability_floor(self):
        spike = np.zeros(256)
        spike[128] = 1.0
        mx, my = geometry.map_marginal_probabilities(spike, spike,
                                                   self.views["reference"], self.views["oblique"])
        self.assertEqual(my[0, 0], 1e-12)
        self.assertGreaterEqual(mx.min(), 1e-12)
        self.assertGreaterEqual(my.min(), 1e-12)
        # Fields repeat 1D likelihoods over a 2D grid; no PDF renormalization.
        identity_x, _ = geometry.map_marginal_probabilities(spike, spike, np.eye(3), np.eye(3))
        self.assertGreater(identity_x.sum(), 255)
        self.assertEqual(spike[128], 1.0)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            geometry.map_marginal_probabilities(np.zeros(256), self.py, np.eye(3), np.eye(3))
        with self.assertRaises(ValueError):
            geometry.map_marginal_probabilities(self.px, self.py, np.eye(3), np.eye(3), floor=0)
        with self.assertRaises(ValueError):
            geometry.view_affine(0, 0)
        with self.assertRaises(ValueError):
            geometry.invert_affine(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            geometry.map_points([np.nan, 1], np.eye(3))
        with self.assertRaises(ValueError):
            geometry.warp_rgb(np.zeros((256, 256, 3), dtype=np.float32), np.eye(3))


if __name__ == "__main__":
    unittest.main()
