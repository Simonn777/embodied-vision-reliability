"""Synthetic tests only: no acquired dataset or ground-truth file is opened."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import predict_pair as worker
np = worker.np


class GuardDataset:
    def __init__(self, value, path, reads):
        self.value, self.path, self.reads = value, path, reads

    def __getitem__(self, item):
        if item != ():
            raise AssertionError("unexpected selection")
        self.reads.append(self.path)
        return self.value


class GuardGroup:
    def __init__(self, values, path="", reads=None):
        self.values, self.path = values, path
        self.reads = [] if reads is None else reads

    def __getitem__(self, key):
        if key not in self.values:
            raise AssertionError("unapproved field: " + key)
        value, path = self.values[key], self.path + "/" + key
        if isinstance(value, dict):
            return GuardGroup(value, path, self.reads)
        return GuardDataset(value, path, self.reads)

    def __iter__(self):
        raise AssertionError("group enumeration is forbidden")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fixture():
    camera = {"model": np.array([b"PINHOLE"]), "width": np.array([640]),
              "height": np.array([480]), "params": np.array([800, 820, 320, 240], np.float32)}
    K = np.array([[800., 0., 320.], [0., 820., 240.], [0., 0., 1.]])
    x = np.arange(20000, dtype=float).reshape(10000, 2) / 100
    return {"x1": x, "x2": x + 1, "K1": K, "K2": K.copy(),
            "camera1": camera, "camera2": dict(camera)}


class PredictionTests(unittest.TestCase):
    def test_allowlist_and_raw_pixel_estimator_arguments(self):
        root = GuardGroup({"0015-synthetic": fixture()})
        x1, x2, calibration = worker.read_pair(root, "0015-synthetic")
        calls = []
        def estimate(a, b, c1, c2, options, bundle, initial_pose):
            calls.append((a.copy(), b.copy(), c1, c2, options, bundle, initial_pose))
            return SimpleNamespace(R=np.eye(3), t=np.array([1., 0., 0.])), {
                "iterations": np.int64(1001), "inliers": np.ones(len(a), dtype=bool),
                "nested": {"score": np.float64(0.25)}}
        result = worker.estimate_pair(SimpleNamespace(estimate_relative_pose=estimate), x1, x2, calibration)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], fixture()["x1"])
        np.testing.assert_array_equal(calls[0][1], fixture()["x2"])
        self.assertEqual(calls[0][4], worker.RANSAC_OPTIONS)
        self.assertEqual(calls[0][5], {})
        self.assertIsNone(calls[0][6])
        expected = {"/0015-synthetic/" + field for field in ("x1", "x2", "K1", "K2")}
        expected |= {"/0015-synthetic/" + camera + "/" + field
                     for camera in ("camera1", "camera2")
                     for field in ("model", "width", "height", "params")}
        self.assertEqual(set(root.reads), expected)
        self.assertEqual(len(root.reads), 12)
        self.assertAlmostEqual(calibration["normalized_epipolar_threshold"], 1 / 810)
        json.dumps(result, allow_nan=False)

    def test_calibration_and_nonfinite_input_fail_closed(self):
        for change in ("K", "focal", "dimension", "model", "point"):
            with self.subTest(change=change):
                data = fixture()
                if change == "K": data["K1"][0, 0] += 2
                if change == "focal": data["camera1"]["params"][0] = 0
                if change == "dimension": data["camera1"]["height"] = np.array([480.5])
                if change == "model": data["camera1"]["model"] = np.array([b"SIMPLE_PINHOLE"])
                if change == "point": data["x1"][0, 0] = np.nan
                with self.assertRaises(ValueError):
                    worker.read_pair(GuardGroup({"0015-synthetic": data}), "0015-synthetic")

    def test_zero_dimension_metadata_preserved_negative_rejected(self):
        data = fixture()
        for name in ("camera1", "camera2"):
            data[name]["width"] = np.array([0])
            data[name]["height"] = np.array([0])
        _, _, calibration = worker.read_pair(GuardGroup({"0015-synthetic": data}), "0015-synthetic")
        self.assertEqual(calibration["camera1"]["width"], 0)
        self.assertEqual(calibration["camera2"]["height"], 0)
        data["camera1"]["width"] = np.array([-1])
        with self.assertRaises(ValueError):
            worker.read_pair(GuardGroup({"0015-synthetic": data}), "0015-synthetic")

    def test_json_and_invalid_estimates_are_not_repaired(self):
        converted = worker.json_safe({"x": [np.int64(3), np.array([True]), np.float32(2)]})
        self.assertEqual(converted, {"x": [3, [True], 2.0]})
        with self.assertRaises(ValueError): worker.json_safe({"nested": [np.float64(np.nan)]})
        calibration = {"camera1": {}, "camera2": {}}
        for R, t, info in ((np.eye(3), [np.inf, 0, 0], {}),
                           (np.eye(3), [0, 0, 0], {}),
                           (2 * np.eye(3), [1, 0, 0], {}),
                           (np.eye(3), [1, 0, 0], {"score": np.nan})):
            backend = SimpleNamespace(estimate_relative_pose=lambda *a, **kw: (SimpleNamespace(R=R, t=np.array(t)), info))
            with self.assertRaises(worker.PredictionFailure) as caught:
                worker.estimate_pair(backend, np.zeros((5, 2)), np.zeros((5, 2)), calibration)
            self.assertEqual(caught.exception.kind, "output")
        def raises(*args, **kwargs): raise RuntimeError("synthetic native failure")
        with self.assertRaises(worker.PredictionFailure) as caught:
            worker.estimate_pair(SimpleNamespace(estimate_relative_pose=raises), None, None, calibration)
        self.assertEqual(caught.exception.kind, "estimator")

    def test_existing_output_and_binding_error_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction.json"
            output.write_text("do not change\n")
            args = ["--data", "missing-data", "--manifest", "missing-manifest", "--pair-index", "0", "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(args), 2)
            self.assertEqual(output.read_text(), "do not change\n")
            output.unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(args), 1)
            record = json.loads(output.read_text())
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["failure_kind"], "binding")
            before = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(args), 2)
            self.assertEqual(output.read_bytes(), before)

    def test_cli_fake_h5py_reads_only_selected_allowed_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, manifest, output = [directory / name for name in ("data.h5", "manifest.json", "out.json")]
            data.write_bytes(b"x")
            keys = ["0015-artificial-%02d" % index for index in range(16)]
            manifest.write_text(json.dumps({"schema": "evr0012.development.v1",
                "coordinate_convention": "raw_pixels", "development_scene": "0015",
                "selected_pair_keys": keys, "dataset_bytes": 1, "dataset_sha256": "0" * 64}))
            root, file_calls = GuardGroup({keys[7]: fixture()}), []
            def open_file(path, mode):
                file_calls.append((path, mode))
                return root
            fake_h5py = SimpleNamespace(File=open_file, __version__="synthetic")
            def estimate(*args, **kwargs):
                marker = json.loads(output.read_text())
                self.assertEqual((marker["status"], marker["phase"]), ("started", "estimator"))
                self.assertIn("calibration", marker)
                self.assertIn("loading_and_validation", marker["timing_seconds"])
                return SimpleNamespace(R=np.eye(3), t=np.array([2., 0., 0.])), {"iterations": np.int64(1001)}
            fake_estimator = SimpleNamespace(estimate_relative_pose=estimate)
            args = ["--data", str(data), "--manifest", str(manifest), "--pair-index", "7", "--output", str(output)]
            with patch.dict(sys.modules, {"h5py": fake_h5py, "poselib": fake_estimator}), \
                 patch.object(worker, "library_metadata", return_value={"version": "synthetic"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(args), 0)
            record = json.loads(output.read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["phase"], "completed")
            self.assertEqual(record["key"], keys[7])
            self.assertEqual(record["estimate"]["t"], [2., 0., 0.])  # no normalization
            self.assertEqual(file_calls, [(data, "r")])
            self.assertEqual(len(root.reads), 12)
            self.assertTrue(all(path.startswith("/" + keys[7] + "/") for path in root.reads))
            self.assertEqual(record["manifest_sha256"], worker.sha256(manifest))
            self.assertTrue(record["library"])

    def test_real_poselib_noncoplanar_artificial_fixture(self):
        import poselib
        self.assertEqual(poselib.__version__, worker.VERSION)
        rng = np.random.default_rng(20261019)
        points = rng.uniform([-1.8, -1.3, 4.0], [1.8, 1.3, 9.0], (100, 3))
        angle = 0.12
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                      [-np.sin(angle), 0., np.cos(angle)]])
        t = np.array([0.5, 0.1, 0.05])
        moved = points @ R.T + t
        camera = {"model": "PINHOLE", "width": 640, "height": 480,
                  "params": [800., 820., 320., 240.]}
        def project(p): return p[:, :2] / p[:, 2, None] * [800., 820.] + [320., 240.]
        result = worker.estimate_pair(poselib, project(points), project(moved), {"camera1": camera, "camera2": camera})
        zero_camera = {**camera, "width": 0, "height": 0}
        zero_result = worker.estimate_pair(poselib, project(points), project(moved),
            {"camera1": zero_camera, "camera2": zero_camera})
        self.assertEqual(zero_result["estimate"], result["estimate"])
        np.testing.assert_allclose(result["estimate"]["R"], R, rtol=0, atol=1e-6)
        estimated_t = np.asarray(result["estimate"]["t"])
        self.assertGreater(np.dot(estimated_t, t) / (np.linalg.norm(estimated_t) * np.linalg.norm(t)), 1 - 1e-10)
        self.assertGreaterEqual(result["returned_ransac_stats"]["iterations"], 1000)
        self.assertEqual(worker.library_metadata(poselib)["version"], "2.0.5")


if __name__ == "__main__":
    unittest.main()
