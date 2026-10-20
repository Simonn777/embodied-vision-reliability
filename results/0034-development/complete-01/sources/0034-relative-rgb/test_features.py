"""Manufactured tests only. No scientific files, model, or network access."""

import json
import math
import unittest
from unittest import mock

import numpy as np

import features as f


def fixture(height=49, width=73):
    y, x = np.mgrid[:height, :width]
    rgb = np.stack((
        0.125 + 0.25 * x / max(width - 1, 1),
        0.25 + 0.25 * y / max(height - 1, 1),
        0.25 + 0.125 * ((x + 2 * y) % 7) / 6,
    ), axis=2).astype(np.float32)
    native = x < width // 2
    logits = np.where(native, 1.5, -1.5).astype(np.float32)
    return rgb, logits, native, rgb.copy()


def brute_band(native):
    height, width = native.shape
    answer = np.zeros_like(native)
    for y in range(height):
        for x in range(width):
            opposite = np.argwhere(native != native[y, x])
            squared = (opposite[:, 0] - y)**2 + (opposite[:, 1] - x)**2
            answer[y, x] = squared.min() <= 64
    return answer


def brute_references(x, y, native, band):
    height, width = native.shape
    groups = {"fg": [], "bg": []}
    for ry in range(0, height, 4):
        for rx in range(0, width, 4):
            if band[ry, rx] or max(abs(rx - x), abs(ry - y)) > 32:
                continue
            key = "fg" if native[ry, rx] else "bg"
            groups[key].append(((rx - x)**2 + (ry - y)**2, ry * width + rx))
    return {key: [i for _, i in sorted(rows)[:32]] for key, rows in groups.items()}


class FeatureTests(unittest.TestCase):
    def test_exact_euclidean_band_against_nonstraight_bruteforce(self):
        yy, xx = np.mgrid[:21, :23]
        native = ((xx - 8)**2 + (yy - 10)**2 <= 25) | ((xx >= 18) & (yy >= 16))
        np.testing.assert_array_equal(f._opposite_band(native), brute_band(native))

    def test_grid_support_order_cap_and_no_border_clipping(self):
        _, _, native, _ = fixture(81, 101)
        band = brute_band(native)
        coords = np.array([[46, 40], [47, 41], [48, 42], [49, 43],
                           [50, 1], [51, 79]], dtype=np.int64)
        ids, counts = f._reference_indices(coords, native, band)
        self.assertTrue(any(counts["fg"] > 32))
        for row, (x, y) in enumerate(coords):
            expected = brute_references(int(x), int(y), native, band)
            for name in ("fg", "bg"):
                actual = ids[name][row][ids[name][row] >= 0].tolist()
                self.assertEqual(actual, expected[name])
                for flat in actual:
                    ry, rx = divmod(flat, native.shape[1])
                    self.assertEqual(rx % 4, 0)
                    self.assertEqual(ry % 4, 0)
                    self.assertFalse(band[ry, rx])
                    self.assertLessEqual(max(abs(rx - x), abs(ry - y)), 32)

    def test_construct_features_against_explicit_single_query_reference(self):
        rgb, logits, native, coarse = fixture()
        arrays, meta = f.construct(rgb, logits, native, coarse)
        self.assertGreater(len(arrays["coords"]), 2)
        band = brute_band(native)
        for row in (0, len(arrays["coords"]) // 2, len(arrays["coords"]) - 1):
            x, y = arrays["coords"][row]
            references = brute_references(int(x), int(y), native, band)
            center = rgb[y, x].astype(np.float64)
            pd, nd = [], []
            for name in ("fg", "bg"):
                colors = [rgb[flat // rgb.shape[1], flat % rgb.shape[1]].astype(np.float64)
                          for flat in references[name]]
                mean = np.array([math.fsum(float(z[c]) for z in colors) / len(colors)
                                 for c in range(3)])
                pd.append(math.fsum(float(z)**2 for z in center - mean) / 3)
                nd.append(min(math.fsum(float(z)**2 for z in center - q) / 3
                              for q in colors))
            stencil = [rgb[y, x].astype(np.float64)] + [
                rgb[y + dy, x + dx].astype(np.float64) for dx, dy in f.AXIAL_OFFSETS]
            differences = [z - center for z in stencil[1:]]
            gram = [math.fsum(float(a) * float(b) for a, b in zip(differences[i], differences[j])) / 3
                    for i, j in f.GRAM_PAIRS]
            expected_e = (nd[1] - nd[0]) / (nd[1] + nd[0] + 1e-6)
            np.testing.assert_allclose(arrays["proto"][row, 3:], pd, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(arrays["nn"][row, 3:], nd, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(arrays["ordinary"][row, 7:], gram, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(arrays["absolute_prefix"][row, 3:], np.concatenate(stencil),
                                       rtol=0, atol=0)
            self.assertAlmostEqual(float(arrays["e"][row]), expected_e, places=7)
        flat = arrays["coords"][:, 1] * rgb.shape[1] + arrays["coords"][:, 0]
        self.assertTrue(np.all(np.diff(flat) > 0))
        self.assertEqual(meta["valid_query_count"], len(flat))
        json.dumps(meta, allow_nan=False)
        np.testing.assert_array_equal(arrays["sym"], arrays["sym_coarse"])
        np.testing.assert_array_equal(arrays["e"], arrays["e_coarse"])

    def test_bilateral_inbounds_and_excludes_center(self):
        rgb, logits, native, coarse = fixture()
        arrays, _ = f.construct(rgb, logits, native, coarse)
        row = 0  # First eligible row is near the image border.
        x, y = (int(v) for v in arrays["coords"][row])
        total, numerator = 0.0, 0.0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                xx, yy = x + dx, y + dy
                if (dx == 0 and dy == 0) or not (0 <= xx < rgb.shape[1] and 0 <= yy < rgb.shape[0]):
                    continue
                distance = math.fsum((float(rgb[y, x, c]) - float(rgb[yy, xx, c]))**2
                                     for c in range(3))
                weight = math.exp(-(dx * dx + dy * dy) / 8 - distance / .02)
                p = 1 / (1 + math.exp(-float(logits[yy, xx])))
                total += weight
                numerator += weight * p
        expected = numerator / total - 1 / (1 + math.exp(-float(logits[y, x])))
        self.assertAlmostEqual(float(arrays["bilateral"][row, 3]), expected, places=7)

    def test_translation_gram_and_foreground_background_exchange(self):
        rgb, logits, native, coarse = fixture()
        original, _ = f.construct(rgb, logits, native, coarse)
        translated, _ = f.construct(rgb + np.float32(0.125), logits, native,
                                     coarse + np.float32(0.125))
        swapped, _ = f.construct(rgb, -logits, ~native, coarse)
        np.testing.assert_array_equal(original["coords"], translated["coords"])
        np.testing.assert_allclose(original["sym"][:, 4:], translated["sym"][:, 4:],
                                   rtol=2e-5, atol=2e-8)
        np.testing.assert_array_equal(original["coords"], swapped["coords"])
        np.testing.assert_array_equal(original["sym"], swapped["sym"])
        np.testing.assert_array_equal(original["e"], -swapped["e"])
        np.testing.assert_array_equal(original["nn"][:, 3], swapped["nn"][:, 4])

    def test_all_noop_cases_and_single_eligible_query_rule(self):
        rgb, logits, native, coarse = fixture(17, 23)
        for value, reason in ((False, "empty_native"), (True, "full_native")):
            arrays, meta = f.construct(rgb, logits, np.full_like(native, value), coarse)
            self.assertEqual(meta["native_noop_reason"], reason)
            self.assertEqual(arrays["coords"].shape, (0, 2))
        small = fixture(3, 5)
        arrays, meta = f.construct(*small)
        self.assertEqual(len(arrays["coords"]), 0)
        self.assertGreater(meta["failure_counts"]["either_support_lt4"], 0)
        # Isolate the global Q<2 rule with valid manufactured supports. No native
        # EDT or scientific image is involved in this deliberately narrow stub.
        only = np.zeros_like(native)
        only[8, 10] = True
        with mock.patch.object(f, "_opposite_band", return_value=only):
            arrays, meta = f.construct(rgb, logits, native, coarse)
        self.assertEqual(meta["eligible_before_minimum_count"], 1)
        self.assertEqual(meta["failure_counts"]["minimum_query_rule_discarded"], 1)
        self.assertEqual(len(arrays["coords"]), 0)

    def test_sampling_round_half_up_and_tile_independence(self):
        self.assertEqual(f._uniform_sample(0).size, 0)
        np.testing.assert_array_equal(f._uniform_sample(1), [0])
        np.testing.assert_array_equal(f._uniform_sample(1024), np.arange(1024))
        sampled = f._uniform_sample(2047)
        np.testing.assert_array_equal(sampled, np.arange(0, 2047, 2))
        sampled = f._uniform_sample(2048)
        expected = [(2 * k * 2047 + 1023) // 2046 for k in range(1024)]
        self.assertEqual(sampled.tolist(), expected)
        self.assertEqual((sampled[0], sampled[-1]), (0, 2047))
        observations = fixture(25, 49)
        first, first_meta = f.construct(*observations)
        with mock.patch.object(f, "TILE_SIZE", 7):
            second, second_meta = f.construct(*observations)
        self.assertEqual(first_meta, second_meta)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_invalid_inputs_fail_before_noop_and_extreme_logits_stay_finite(self):
        rgb, logits, native, coarse = fixture(17, 23)
        wrong = rgb.copy()
        wrong[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            f.construct(wrong, logits, np.zeros_like(native), coarse)
        with self.assertRaises(TypeError):
            f.construct(rgb.astype(np.float64), logits, native, coarse)
        with self.assertRaises(TypeError):
            f.construct(rgb, logits, native.astype(np.uint8), coarse)
        with self.assertRaises(ValueError):
            f.construct(rgb, logits[:-1], native, coarse)
        with self.assertRaises(ValueError):
            f.construct(rgb, logits, native, coarse + np.float32(1.0))
        extreme = np.where(native, np.finfo(np.float32).max,
                           -np.finfo(np.float32).max).astype(np.float32)
        arrays, meta = f.construct(rgb, extreme, native, coarse)
        self.assertGreater(meta["valid_query_count"], 1)
        for values in arrays.values():
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
