"""Artificial unit checks: no published projector or benchmark fixture is run."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_evr0025c_test_fixtures", HERE / "fixtures.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class FakePublishedSource:
    """Small fake API: projector is affine metadata, never implicit geometry."""
    WIDTH, HEIGHT = 400, 300
    K = np.array([[400., 0., 200.], [0., 410., 150.], [0., 0., 1.]])
    POINTS = np.column_stack((np.linspace(-.2, .2, 8),
                              np.sin(np.arange(8)), np.linspace(3., 4., 8)))

    def __init__(self):
        self.project_calls = []
        self.exp_calls = []

    def truth(self, kind):
        return dict(R=rz(.3), t=np.array([.2, .1, .05]),
                    xi1=np.array([.2, .3, .4, .5, .6, .7]),
                    xi2=np.array([-.3, .4, .2, -.5, .7, .1]),
                    b1=.2, a2=1.1, b2=-.1, readout1=.04, readout2=.04)

    def rotation_and_left_jacobian(self, w):
        return rx(float(w[0])), np.eye(3)

    def exp_se3(self, xi, s):
        self.exp_calls.append((xi.copy(), s))
        return rz(s * xi[0]), s * xi[3:]

    def project_implicit(self, X, R, t, xi, readout):
        self.project_calls.append((X.copy(), R.copy(), t.copy(), xi.copy(), readout))
        pixel = np.array([200. + 15. * X[0] + 2. * t[0],
                          120. + 5. * X[1] + 3. * t[1]])
        ray = np.linalg.solve(self.K, np.r_[pixel, 1.])
        time = readout * (pixel[1] - 149.5) / 299.
        return dict(pixel=pixel, ray=ray, time=time, depth=2. + X[2], row_residual=0.)


def generated():
    fake = FakePublishedSource()
    with patch.object(f, "_load_source", return_value=fake):
        records = f.generate()
    return fake, records


class FixtureTests(unittest.TestCase):
    def test_exact_coverage_and_generation_only_once(self):
        fake, records = generated()
        expected = []
        for case in f.CASES:
            expected.append((case, "clean", -1))
            expected.extend((case, regime, i) for i in range(16) for regime in ("P", "S"))
        self.assertEqual([(r["case"], r["regime"], r["realization"]) for r in records], expected)
        self.assertEqual(len(records), 132)
        self.assertEqual(len(fake.project_calls), 64)
        self.assertEqual(set(records[0]["data"]), {"r1", "r2", "s1", "s2", "d1", "d2", "R", "w1", "w2"})

    def test_exact_serialized_random_stream_and_paired_inputs(self):
        _, records = generated()
        rng = np.random.Generator(np.random.PCG64(20261019))
        for offset in range(0, len(records), 33):
            self.assertIsNone(records[offset]["draws"])
            for i in range(16):
                p, s = records[offset + 1 + 2 * i:offset + 3 + 2 * i]
                before = deepcopy(rng.bit_generator.state)
                expected = (rng.standard_normal((2, 8, 2)), rng.standard_normal(2),
                            rng.standard_normal((2, 8)), rng.standard_normal((3, 3)))
                self.assertEqual(p["draws"]["rng_state_before"], before)
                self.assertEqual(p["draws"]["rng_state_after"], rng.bit_generator.state)
                for name, values in zip(("pixel_standard_normal", "coherent_standard_normal",
                                         "epsilon_standard_normal", "angular_axes_standard_normal"), expected):
                    np.testing.assert_array_equal(p["draws"][name], values)
                    np.testing.assert_array_equal(s["draws"][name], values)
                for name in ("r1", "r2", "s1", "s2", "d1", "d2"):
                    np.testing.assert_array_equal(p["data"][name], s["data"][name])

    def test_pixel_error_changes_ray_and_sensor_row_time(self):
        fake, records = generated()
        p = records[1]
        pixels = p["observed_pixels"]
        rays = np.array([p["data"]["r1"], p["data"]["r2"]])
        projected = np.einsum("ij,knj->kni", fake.K, rays)
        np.testing.assert_allclose(projected[:, :, :2], pixels, rtol=0., atol=1e-12)
        expected_times = .04 * (pixels[:, :, 1] - 149.5) / 299.
        np.testing.assert_array_equal(np.array([p["data"]["s1"], p["data"]["s2"]]), expected_times)
        self.assertTrue(np.any(expected_times != p["clean"]["times"]))
        self.assertEqual(len(fake.project_calls), 64)

    def test_depth_attached_to_fixed_point_and_noise_equation(self):
        _, records = generated()
        for p in (records[1], records[34], records[67], records[100]):
            clean, draw = p["clean"], p["draws"]
            uv = (clean["pixels"] + .5) / np.array([400., 300.])
            g = np.sin(2 * np.pi * uv[:, :, 0]) + np.cos(2 * np.pi * uv[:, :, 1])
            g -= g.mean(axis=1, keepdims=True)
            g /= np.sqrt(np.mean(g * g, axis=1))[:, None]
            expected_error = .03 / np.sqrt(2) * (g * draw["coherent_standard_normal"][:, None]
                                                  + draw["epsilon_standard_normal"])
            expected_depth = clean["raw_depths"] + clean["axial_depths"] / np.array([1., 1.1])[:, None] * expected_error
            np.testing.assert_allclose(np.array([p["data"]["d1"], p["data"]["d2"]]), expected_depth,
                                       rtol=0., atol=2e-15)
            np.testing.assert_array_equal(p["truth"]["z1"], clean["axial_depths"][0])

    def test_angular_stress_is_left_rotation_and_no_depth_change(self):
        fake, records = generated()
        p, s = records[1:3]
        axes = s["draws"]["angular_axes_unit"]
        expected_R = rx(np.deg2rad(.1) * axes[0, 0]) @ s["truth"]["R"]
        np.testing.assert_array_equal(s["data"]["R"], expected_R)
        np.testing.assert_array_equal(p["data"]["R"], p["truth"]["R"])
        np.testing.assert_allclose(s["data"]["w1"] - p["data"]["w1"], np.deg2rad(.5) * axes[1],
                                   rtol=0., atol=3e-17)
        self.assertFalse(np.array_equal(expected_R, s["truth"]["R"] @ rx(np.deg2rad(.1) * axes[0, 0])))

    def test_constant_trajectory_ties_middle_and_both_readouts(self):
        fake, records = generated()
        self.assertEqual(len(fake.exp_calls), 1)
        xi, duration = fake.exp_calls[0]
        self.assertEqual(duration, .12)
        c = next(r for r in records if r["case"] == "C" and r["regime"] == "clean")
        np.testing.assert_array_equal(c["truth"]["t"], .12 * xi[3:])
        np.testing.assert_array_equal(c["truth"]["R"], rz(.12 * xi[0]))
        np.testing.assert_array_equal(c["data"]["w1"], c["data"]["w2"])
        for call in fake.project_calls[32:48]:
            np.testing.assert_array_equal(call[3], xi)

    def test_zero_pattern_fails_without_resampling(self):
        fake = FakePublishedSource()
        fake.POINTS = np.tile(np.array([0., 0., 3.]), (8, 1))
        with patch.object(f, "_load_source", return_value=fake):
            with self.assertRaisesRegex(ValueError, "undefined fixed spatial"):
                f.generate()
        self.assertEqual(len(fake.project_calls), 16)

    def test_hash_mismatch_prevents_any_source_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "work"
            fixture_path = root / "0025c-feasibility" / "fixtures.py"
            source = root / "embodied-vision-reliability" / f.SOURCE_RELATIVE
            source.parent.mkdir(parents=True)
            source.write_text("raise RuntimeError('must never execute')\n")
            with patch.object(f, "__file__", str(fixture_path)), \
                    patch.object(f.importlib.util, "spec_from_file_location") as loader:
                with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                    f._load_source()
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
