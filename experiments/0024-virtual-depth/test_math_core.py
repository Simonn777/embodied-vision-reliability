"""Manufactured fixtures only; no scientific inputs, images or model calls."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("virtual_depth_math", Path(__file__).with_name("math_core.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def manufactured(offset=.4, wrong_shape=False):
    xy, train = m.base_grid((128, 128))
    u, v = (xy[:, 0]-63.5)/90., (xy[:, 1]-63.5)/90.
    z = 2.+.3*np.sin(5*u)+.2*v*v+.1*u*v
    angle = .12
    c, s = np.cos(angle), np.sin(angle)
    h = np.array([c-s*u, c+s*u, c+s*v, c-s*v])
    base = z+offset
    scales, shifts = [1.2, .8, 1.1, 1.4], [.1, .2, -.05, .3]
    latent = base if wrong_shape else z
    views = [scales[j]*h[j]*latent+shifts[j] for j in range(4)]
    values = np.vstack((base, views))
    return values, h, np.ones(len(z), dtype=bool), train, base.copy()


def rows(candidate=.8, baseline=1., usable=True):
    return [{"scene": "s%d" % (i//9), "image_id": "image-%d" % i,
             "analysis": {"usable_offset": usable, "directions": {
                 name: {"candidate_error": candidate, "zero_offset_error": baseline}
                 for name in m.DIRECTIONS}}} for i in range(27)]


class GeometryTests(unittest.TestCase):
    def test_canonical_geometry_margin_rotation_and_remap(self):
        K = np.array([[550., 0., 317.2], [0., 525., 237.7], [0., 0., 1.]])
        g = m.geometry(K, (640, 480))
        self.assertEqual(g["size"], (512, 384))
        self.assertEqual(g["focal_bisections"], 60)
        self.assertTrue(0 < g["angle_radians"] <= np.deg2rad(8.))
        self.assertEqual(g["Kc"][0, 0], g["Kc"][1, 1])
        for R in g["rotations"]:
            xy = m.remap_coordinates(K, g["Kc"], R, g["size"])
            self.assertEqual(xy.shape, (384, 512, 2))
            self.assertTrue(((xy[..., 0] >= 2) & (xy[..., 0] <= 637)).all())
            self.assertTrue(((xy[..., 1] >= 2) & (xy[..., 1] <= 477)).all())
        # Independent homogeneous evaluation for an interior non-integer ray.
        output = np.array([101., 77., 1.])
        direct = K.dot(g["rotations"][1].T.dot(np.linalg.solve(g["Kc"], output)))
        xy = m.remap_coordinates(K, g["Kc"], g["rotations"][1], g["size"])
        np.testing.assert_allclose(xy[77, 101], direct[:2]/direct[2], atol=1e-12, rtol=0)

    def test_grid_parity_and_four_neighbor_validity(self):
        xy, train = m.base_grid((128, 96))
        self.assertEqual(int(train.sum()), 2048)
        np.testing.assert_array_equal(xy[0], [.5, .25])
        self.assertTrue(train[0])
        self.assertFalse(train[1])
        self.assertFalse(train[64])
        depth = np.arange(1., 17.).reshape(4, 4)
        mask = np.ones((4, 4), dtype=bool)
        # Even zero-weight neighbors must be valid (four-neighbor policy).
        mask[1, 1] = False
        values, good = m.sample(depth, mask, np.array([[0., 0.], [1.5, 2.5], [3., 1.], [np.nan, 0.]]))
        np.testing.assert_array_equal(good, [False, True, False, False])
        self.assertEqual(values[1], 12.5)
        self.assertEqual(values[0], 0.)
        depth[2, 2] = -1
        _, good = m.sample(depth, mask, np.array([[1.5, 2.5]]))
        self.assertFalse(good[0])

    def test_bad_camera_or_rotation_is_technical_failure(self):
        K = np.array([[100., 0., 1.], [0., 100., 63.5], [0., 0., 1.]])
        with self.assertRaises(ValueError):
            m.geometry(K, (128, 128))
        with self.assertRaises(ValueError):
            m.remap_coordinates(np.eye(3), np.eye(3), np.zeros((3, 3)), (128, 128))


class FitTests(unittest.TestCase):
    def test_known_nonplanar_shift_sign_and_crossaxis_prediction(self):
        data = manufactured(.4)
        result = m.analyze_samples(*data)
        self.assertTrue(result["usable_offset"])
        for axis in ("yaw", "pitch"):
            self.assertAlmostEqual(result["offsets"][axis]["coefficients"]["t0"], -.4, places=10)
            self.assertGreaterEqual(result["offsets"][axis]["singular_ratio"], m.RANK_RATIO)
        for name in m.DIRECTIONS:
            d = result["directions"][name]
            self.assertGreater(d["zero_offset_error"], 1e-4)
            self.assertLess(d["candidate_error"], 1e-11)
        json.dumps(result, allow_nan=False)

    def test_roll_and_constant_depth_degeneracy_refuse(self):
        values, h, common, train, pixels = manufactured()
        result = m.analyze_samples(values, np.ones_like(h), common, train, pixels)
        self.assertFalse(result["usable_offset"])
        self.assertTrue(all(v["reason"] == "rank_deficient" for v in result["offsets"].values()))
        for d in result["directions"].values():
            self.assertEqual(d["candidate_error"], d["zero_offset_error"])
        values[1:] = 1.5
        result = m.analyze_samples(values, h, common, train, pixels)
        self.assertFalse(result["usable_offset"])

    def test_negative_shared_scale_refuses(self):
        values, h, common, train, pixels = manufactured()
        values[1] = 10.-values[1]
        result = m.analyze_samples(values, h, common, train, pixels)
        self.assertFalse(result["usable_offset"])
        self.assertEqual(result["offsets"]["yaw"]["reason"], "nonpositive_scale")
        self.assertTrue(result["offsets"]["pitch"]["usable"])
        for d in result["directions"].values():
            self.assertTrue(d["fallback_to_baseline"])
            self.assertEqual(d["candidate_error"], d["zero_offset_error"])

    def test_boundary_uses_inverse_h_not_constant_base_prediction(self):
        h = np.array([.8, .9, 1.1, 1.2, 1.3, 1.4])
        dj = np.array([.6, .8, 1.2, 1.6, 1.8, 2.])
        y = 4./h-dj/h
        out = m.fit_nuisance(y, dj, h, 0.)
        self.assertEqual(out["boundary"], "negative_scale")
        self.assertEqual(out["scale"], 0.)
        v = 1./h
        expected = float(np.sum(v*y)/np.sum(v*v))
        self.assertAlmostEqual(out["shift"], expected, places=14)
        self.assertGreater(np.ptp(out["shift"]/h), .1)
        # Constant virtual depth also creates rank deficiency and same boundary.
        out = m.fit_nuisance(y, np.ones_like(h), h, .2)
        self.assertEqual(out["boundary"], "rank_deficient")
        self.assertAlmostEqual(out["shift"], float(np.sum(v*(y+.2))/np.sum(v*v)), places=14)

    def test_all_image_positivity_not_just_grid(self):
        values, h, common, train, pixels = manufactured(.4)
        # The fit grid alone is positive after correction. Unseen base pixels
        # remain part of the prespecified whole-image positivity guard.
        full = np.r_[pixels, np.full(100, .1)]
        result = m.analyze_samples(values, h, common, train, full)
        self.assertFalse(result["usable_offset"])
        self.assertEqual(result["offsets"]["yaw"]["reason"], "full_image_positivity")
        self.assertLess(result["offsets"]["yaw"]["positive_fraction"], .99)
        for d in result["directions"].values():
            self.assertEqual(d["candidate_error"], d["zero_offset_error"])

    def test_checkerboard_evaluation_does_not_fit_offset_or_nuisance(self):
        values, h, common, train, pixels = manufactured()
        before = m.analyze_samples(values, h, common, train, pixels)
        changed = values.copy()
        changed[:, ~train] += np.linspace(.1, .5, 5)[:, None]
        after = m.analyze_samples(changed, h, common, train, pixels)
        self.assertEqual(before["offsets"], after["offsets"])
        for name in m.DIRECTIONS:
            self.assertNotEqual(before["directions"][name]["candidate_error"], after["directions"][name]["candidate_error"])
            for a, b in zip(before["directions"][name]["views"], after["directions"][name]["views"]):
                self.assertEqual(a["zero_offset_fit"], b["zero_offset_fit"])
                self.assertEqual(a["candidate_fit"], b["candidate_fit"])

    def test_common_support_and_unavailable_failure_mass(self):
        values, h, common, train, pixels = manufactured()
        common[:300] = False
        values[:, :300] = np.nan
        result = m.analyze_samples(values, h, common, train, pixels)
        self.assertEqual(result["support"]["common"], 3796)
        for offset in result["offsets"].values():
            self.assertEqual(offset["training_rows"], 2*result["support"]["train"])
        for d in result["directions"].values():
            for view in d["views"]:
                self.assertEqual(view["zero_offset_fit"]["training_rows"], result["support"]["train"])
        common[:] = False
        result = m.analyze_samples(values, h, common, train, pixels)
        self.assertEqual(result["unavailable_reason"], "insufficient_common_support")
        self.assertEqual([result["directions"][x]["candidate_error"] for x in m.DIRECTIONS], [1., 1.])

    def test_wrong_equivariant_shape_is_unobservable_counterexample(self):
        # Base has an additive error .4 against the manufactured true z, but
        # all virtual predictors share that wrong shape equivariantly.
        result = m.analyze_samples(*manufactured(.4, wrong_shape=True))
        self.assertTrue(result["usable_offset"])
        for axis in ("yaw", "pitch"):
            self.assertAlmostEqual(result["offsets"][axis]["coefficients"]["t0"], 0., places=10)
        for d in result["directions"].values():
            self.assertLess(d["zero_offset_error"], 1e-11)
        # Good consistency therefore cannot establish correction of .4 error.

    def test_unexpected_solver_error_is_technical_not_fallback(self):
        values, h, _, train, _ = manufactured()
        with mock.patch.object(np.linalg, "svd", side_effect=ValueError("programming fault")):
            with self.assertRaisesRegex(ValueError, "programming fault"):
                m.fit_offset(values[0, train], values[1:3, train], h[:2, train])


class ImageAndAggregateTests(unittest.TestCase):
    def test_full_image_sampling_recovers_axial_shift_in_median_gauge(self):
        w, height, focal, angle = 128, 96, 110., .06
        K = np.array([[focal, 0., (w-1)/2], [0., focal, (height-1)/2], [0., 0., 1.]])
        c, s = np.cos(angle), np.sin(angle)
        yaw = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        pitch = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
        rotations = np.stack((np.eye(3), yaw, yaw.T, pitch, pitch.T))
        yy, xx = np.indices((height, w), dtype=np.float64)
        rays = np.stack(((xx-(w-1)/2)/focal, (yy-(height-1)/2)/focal, np.ones_like(xx)), axis=-1)
        depths = []
        for j, R in enumerate(rotations):
            original = np.einsum("ji,hwj->hwi", R, rays, optimize=False)
            u, v = original[..., 0]/original[..., 2], original[..., 1]/original[..., 2]
            z0 = 3.+.2*u+.3*v*v+.1*np.sin(6*u)+.07*np.sin(5*v)
            axial = z0/original[..., 2]
            depths.append([1., 1.2, .8, 1.1, 1.4][j]*axial+[.6, .1, .2, -.05, .3][j])
        depths = np.stack(depths)
        result = m.analyze_image(depths, np.ones_like(depths, dtype=bool), K, rotations)
        self.assertTrue(result["usable_offset"])
        expected = -.6/float(np.median(depths[0]))
        for fitted in result["offsets"].values():
            self.assertAlmostEqual(fitted["coefficients"]["t0"], expected, delta=.002)
        for d in result["directions"].values():
            self.assertLess(d["candidate_error"], d["zero_offset_error"]*.05)
        json.dumps(result, allow_nan=False)

    def test_image_masks_medians_scale_gauge_and_finite_json(self):
        size = (128, 96)
        K = np.array([[100., 0., 63.5], [0., 100., 47.5], [0., 0., 1.]])
        yy, xx = np.indices((96, 128))
        depth = 2.+.001*xx+.002*yy
        depths = np.stack([depth*(j+1) for j in range(5)])
        masks = np.ones(depths.shape, dtype=bool)
        masks[1, 10:20, 10:20] = False
        rotations = np.repeat(np.eye(3)[None], 5, axis=0)
        out = m.analyze_image(depths, masks, K, rotations)
        self.assertFalse(out["usable_offset"])
        self.assertEqual(out["positive_valid_pixel_counts"][1], 96*128-100)
        scaled = m.analyze_image(depths*np.arange(1, 6)[:, None, None], masks, K, rotations)
        self.assertEqual(out["support"], scaled["support"])
        for name in m.DIRECTIONS:
            self.assertAlmostEqual(out["directions"][name]["zero_offset_error"], scaled["directions"][name]["zero_offset_error"], places=12)
        masks[4] = False
        absent = m.analyze_image(depths, masks, K, rotations)
        self.assertEqual(absent["unavailable_reason"], "positive_valid_median_unavailable")
        self.assertEqual(absent["directions"]["yaw_to_pitch"]["candidate_error"], 1.)
        json.dumps(absent, allow_nan=False)

    def test_exact_27_54_denominator_and_coverage(self):
        population = rows()
        # 19 usable images, distribution 7/6/6; other eight exactly baseline.
        bad = [0, 1, 9, 10, 11, 18, 19, 20]
        for i in bad:
            population[i]["analysis"]["usable_offset"] = False
            for d in population[i]["analysis"]["directions"].values():
                d["candidate_error"] = d["zero_offset_error"]
        result = m.summarize(population)
        self.assertEqual(result["usable_offset_count"], 19)
        self.assertEqual(result["direction_count"], 54)
        self.assertAlmostEqual(result["candidate_mean"], (38*.8+16)/54.)
        self.assertTrue(result["pass"])
        for d in population[2]["analysis"]["directions"].values():
            d["candidate_error"] = d["zero_offset_error"]
        population[2]["analysis"]["usable_offset"] = False
        self.assertFalse(m.summarize(population)["gates"]["coverage"])

    def test_zero_baselines_scene_floor_and_finite_guard(self):
        result = m.summarize(rows(0., 0.))
        self.assertEqual(result["relative_gain"], 0.)
        self.assertFalse(result["pass"])
        self.assertTrue(result["gates"]["scene_floor"])
        result = m.summarize(rows(.1, 0.))
        self.assertIsNone(result["relative_gain"])
        self.assertFalse(result["gates"]["pooled_gain"])
        self.assertFalse(result["gates"]["scene_floor"])
        population = rows(.5, 1.)
        for row in population[:9]:
            for d in row["analysis"]["directions"].values():
                d["candidate_error"] = 1.021
        self.assertFalse(m.summarize(population)["gates"]["scene_floor"])
        population[0]["analysis"]["directions"]["yaw_to_pitch"]["candidate_error"] = np.nan
        with self.assertRaises(ValueError):
            m.summarize(population)
        with self.assertRaises(ValueError):
            m.summarize(rows()[:-1])
        population = rows()
        population[1]["image_id"] = population[0]["image_id"]
        with self.assertRaises(ValueError):
            m.summarize(population)

    def test_one_direction_cannot_be_dropped_and_bad_fallback_rejected(self):
        population = rows(1., 1.)
        population[0]["analysis"]["directions"]["yaw_to_pitch"]["candidate_error"] = 55.
        self.assertEqual(m.summarize(population)["candidate_mean"], 2.)
        population[0]["analysis"]["usable_offset"] = False
        with self.assertRaises(ValueError):
            m.summarize(population)


if __name__ == "__main__":
    unittest.main()
