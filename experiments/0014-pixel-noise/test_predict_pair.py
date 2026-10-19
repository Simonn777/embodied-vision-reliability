"""Artificial-only worker integration tests; no acquired HDF5 is opened."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_test_evr0014_worker", HERE / "predict_pair.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
base, verifier, real_kernel = worker.load_dependencies()
np = base.np


class GuardDataset:
    def __init__(self, value, path, reads):
        self.value, self.path, self.reads = value, path, reads

    def __getitem__(self, selection):
        if selection != ():
            raise AssertionError("unexpected HDF5 selection")
        self.reads.append(self.path)
        return self.value


class GuardGroup:
    def __init__(self, values, path="", reads=None):
        self.values, self.path = values, path
        self.reads = [] if reads is None else reads

    def __getitem__(self, key):
        if key not in self.values:
            raise AssertionError("unapproved HDF5 field: " + key)
        value, path = self.values[key], self.path + "/" + key
        if isinstance(value, dict):
            return GuardGroup(value, path, self.reads)
        return GuardDataset(value, path, self.reads)

    def __iter__(self):
        raise AssertionError("HDF5 enumeration forbidden")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fixture():
    camera = {"model": np.array([b"PINHOLE"]), "width": np.array([0]),
              "height": np.array([0]), "params": np.array([800, 820, 320, 240], np.float32)}
    K = np.array([[800., 0., 320.], [0., 820., 240.], [0., 0., 1.]])
    K[0, 0] += 0.0001  # Allowed saved-K mismatch must not affect NumPy geometry.
    x = np.arange(20000, dtype=float).reshape(10000, 2) / 100
    return {"x1": x, "x2": x + 1, "K1": K, "K2": K.copy(),
            "camera1": camera, "camera2": dict(camera)}


def write_json(path, value):
    path.write_text(json.dumps(base.json_safe(value), sort_keys=True, allow_nan=False))
    return base.sha256(path)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = self.directory / "artificial.h5"
        self.data.write_bytes(b"synthetic placeholder, never decoded")
        self.manifest_path = self.directory / "manifest.json"
        self.seal_path = self.directory / "seal.json"
        self.output = self.directory / "out.json"
        self.keys = ["0015-artificial-%02d" % i for i in range(16)]
        self.key = self.keys[7]
        self.values = fixture()
        _, _, calibration = base.read_pair(GuardGroup({self.key: self.values}), self.key)
        self.calibration = base.json_safe(calibration)
        self.library = {"version": "synthetic", "h5py_version": "synthetic"}
        self.mask = [i % 3 == 0 for i in range(10000)]
        self.initial = {"R": np.eye(3).tolist(), "t": [2., 0.2, 0.1]}
        manifest = {"schema": "evr0012.development.v1", "coordinate_convention": "raw_pixels",
            "development_scene": "0015", "selected_pair_keys": self.keys,
            "dataset_bytes": self.data.stat().st_size, "dataset_sha256": base.sha256(self.data)}
        manifest_hash = write_json(self.manifest_path, manifest)
        source = {name: base.sha256(worker.BASELINE_DIR / name) for name in verifier.SOURCE_NAMES}
        rows = []
        for i, key in enumerate(self.keys):
            record = {"schema": "evr0013.prediction.v1", "status": "completed", "pair_index": i,
                "key": key, "manifest_sha256": manifest_hash,
                "dataset_sha256_declared": manifest["dataset_sha256"],
                "source_sha256": source["predict_pair.py"], "estimate": self.initial,
                "settings": base.SETTINGS, "library": self.library, "calibration": self.calibration,
                "returned_ransac_stats": {"inliers": self.mask}}
            filename = "pair-%02d.json" % i
            rows.append({"pair_index": i, "key": key, "path": filename, "status": "completed",
                         "sha256": write_json(self.directory / filename, record)})
        self.seal = {"schema": "evr0013.seal.v1", "complete": True, "eligible_for_evaluation": True,
            "manifest_sha256": manifest_hash, "dataset_sha256_declared": manifest["dataset_sha256"],
            "source_sha256": source, "protocol_sha256": base.sha256(
                verifier.REPO_ROOT / "protocols/0013-baseline-pose-headroom.md"), "rows": rows}
        write_json(self.seal_path, self.seal)
        self.root = GuardGroup({self.key: self.values})
        self.calls = []
        self.fail_mode = None

    def args(self):
        return ["--data", str(self.data), "--manifest", str(self.manifest_path),
                "--baseline-seal", str(self.seal_path), "--pair-index", "7",
                "--output", str(self.output)]

    def marker(self, expected):
        record = json.loads(self.output.read_text())
        self.assertEqual((record["status"], record["phase"]), ("started", expected))
        return record

    def fake_native(self, a, b, initial, camera1, camera2, options):
        self.marker("native")
        self.calls.append(("native", a.copy(), b.copy(), initial.R.copy(), initial.t.copy(), options))
        self.assertEqual(camera1, self.calibration["camera1"])
        self.assertEqual(camera2, self.calibration["camera2"])
        if self.fail_mode == "native":
            raise RuntimeError("artificial native failure")
        a[:] = 9999.  # A branch cannot contaminate later branch inputs.
        initial.t[:] = -9999.
        return SimpleNamespace(R=np.eye(3), t=np.array([1., 0., 0.])), {
            "iterations": np.int64(2), "cost": np.float64(0.2)}

    def fake_kernel(self, a, b, K1, K2, R, t, mode):
        self.marker(mode)
        self.calls.append((mode, a.copy(), b.copy(), R.copy(), t.copy(), K1.copy()))
        self.assertEqual(K1[0, 0], 800.)  # Authored camera value, not saved 800.0001.
        np.testing.assert_array_equal(K1, K2)
        result = {"schema": "evr0014.refinement.v1", "mode": mode,
            "settings": real_kernel.SETTINGS, "number_of_correspondences": len(a),
            "R": R, "t": t, "iterations": [{"iteration": np.int64(0), "accepted": True}],
            "initial_cost": np.float64(1), "final_cost": np.float64(.5),
            "converged": True, "termination_reason": "synthetic",
            "iteration_count": 1, "accepted_steps": 1, "elapsed_seconds": 0.001}
        if self.fail_mode == mode:
            result["t"] = [np.nan, 0., 0.]
        return result

    def invoke(self, dependencies=None):
        fake_kernel = SimpleNamespace(SETTINGS=real_kernel.SETTINGS, refine_pose=self.fake_kernel,
            residuals_and_jacobian=lambda x, y, R, t, f1, f2, mode: (
                np.full(len(x), 0.1), np.full((len(x), 5), 0.2)),
            cauchy_cost=real_kernel.cauchy_cost)
        fake_backend = SimpleNamespace(CameraPose=SimpleNamespace, refine_relative_pose=self.fake_native)
        def h5open(path, mode):
            self.marker("input")
            self.assertEqual((path, mode), (self.data, "r"))
            return self.root
        def deps():
            self.marker("binding")
            return (base, verifier, fake_kernel) if dependencies is None else dependencies()
        with patch.object(worker, "load_dependencies", side_effect=deps), \
             patch.object(base, "library_metadata", return_value={"version": "synthetic"}), \
             patch.dict(sys.modules, {"h5py": SimpleNamespace(File=h5open, __version__="synthetic"),
                                     "poselib": fake_backend}), \
             contextlib.redirect_stdout(io.StringIO()):
            status = worker.main(self.args())
        return status, json.loads(self.output.read_text())

    def replace_baseline(self, index, change):
        path = self.directory / self.seal["rows"][index]["path"]
        record = json.loads(path.read_text())
        change(record)
        self.seal["rows"][index]["sha256"] = write_json(path, record)
        write_json(self.seal_path, self.seal)

    def test_full_seal_allowlist_independent_initialization_and_effective_calibration(self):
        with patch.object(verifier, "verify_seal", wraps=verifier.verify_seal) as verify:
            status, record = self.invoke()
            self.assertEqual(verify.call_count, 1)
        self.assertEqual(status, 0)
        self.assertEqual(record["status"], "completed")
        self.assertEqual([call[0] for call in self.calls], ["native", "N", "P"])
        for call in self.calls:
            np.testing.assert_array_equal(call[1], self.values["x1"][self.mask])
            np.testing.assert_array_equal(call[2], self.values["x2"][self.mask])
            np.testing.assert_array_equal(call[3], self.initial["R"])
            np.testing.assert_array_equal(call[4], self.initial["t"])
        self.assertEqual(self.calls[0][5], worker.NATIVE_OPTIONS)
        self.assertEqual(record["kept_count"], sum(self.mask))
        self.assertEqual(record["mask_sha256"], hashlib.sha256(bytes(self.mask)).hexdigest())
        self.assertEqual(record["effective_K1"][0][0], 800.)
        self.assertNotEqual(record["calibration"]["K1"][0][0], record["effective_K1"][0][0])
        self.assertEqual(record["baseline_prediction_sha256"], self.seal["rows"][7]["sha256"])
        self.assertEqual(set(record["objectives"]), {"original", "native", "N", "P"})
        for name in record["objectives"]:
            for mode in ("N", "P"):
                self.assertIsInstance(record["objectives"][name][mode], float)
                self.assertAlmostEqual(record["canonical_objectives"][name][mode],
                                       record["objectives"][name][mode] * 0.25)
                self.assertAlmostEqual(record["objective_gradients_infinity"][name][mode],
                                       2 * 0.2 * sum(self.mask) * .1 / .26)
        expected = {"/" + self.key + "/" + name for name in ("x1", "x2", "K1", "K2")}
        expected |= {"/" + self.key + "/" + c + "/" + f for c in ("camera1", "camera2")
                     for f in ("model", "width", "height", "params")}
        self.assertEqual(set(self.root.reads), expected)
        self.assertEqual(len(self.root.reads), 12)
        self.assertEqual(set(record["kernel_results"]), {"N", "P"})

    def test_unselected_prediction_tamper_blocks_hdf5_and_all_calls(self):
        path = self.directory / "pair-15.json"
        path.write_text(path.read_text() + " ")
        status, record = self.invoke()
        self.assertEqual((status, record["failure_kind"]), (1, "binding"))
        self.assertIn("prediction hash mismatch", record["error"])
        self.assertEqual((self.root.reads, self.calls), ([], []))

    def test_invalid_mask_and_settings_block_before_hdf5(self):
        self.replace_baseline(7, lambda r: r["returned_ransac_stats"]["inliers"].__setitem__(0, 1))
        status, record = self.invoke()
        self.assertEqual((status, record["failure_kind"]), (1, "binding"))
        self.assertEqual(self.root.reads, [])
        self.output.unlink()  # Synthetic test isolation only, never scientific outputs.
        self.replace_baseline(7, lambda r: r.update(settings={}))
        status, record = self.invoke()
        self.assertEqual((status, record["failure_kind"]), (1, "binding"))
        self.assertEqual(self.root.reads, [])

    def test_existing_output_refuses_before_dependency_or_input_read(self):
        self.output.write_text("preserve\n")
        with patch.object(worker, "load_dependencies", side_effect=AssertionError("must not import")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(self.args()), 2)
        self.assertEqual(self.output.read_text(), "preserve\n")

    def test_native_exception_retains_time_and_stops_later_arms(self):
        self.fail_mode = "native"
        status, record = self.invoke()
        self.assertEqual((status, record["failure_kind"], record["failure_phase"]), (1, "estimator", "native"))
        self.assertGreaterEqual(record["timing_seconds"]["native"], 0)
        self.assertEqual([call[0] for call in self.calls], ["native"])
        self.assertEqual(record["estimates"], {})

    def test_nonfinite_kernel_output_preserves_prior_arm_and_failure(self):
        self.fail_mode = "N"
        status, record = self.invoke()
        self.assertEqual((status, record["failure_kind"], record["failure_phase"]), (1, "output", "N"))
        self.assertEqual(set(record["estimates"]), {"native"})
        self.assertEqual([call[0] for call in self.calls], ["native", "N"])
        self.assertGreaterEqual(record["timing_seconds"]["N"], 0)

    def test_local_gn_stationarity_positive_curvature_and_invalid_curvature(self):
        residual, J = np.full(5, 0.1), np.eye(5)
        cost = real_kernel.cauchy_cost(residual)
        result = worker.local_gn_stationarity(np, residual, J, cost)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["relative_predicted_decrease"],
                               5 * .01 / .26 / max(1., cost), places=14)
        exact = worker.local_gn_stationarity(np, np.zeros(5), J, 0.)
        self.assertEqual(exact, {"valid": True, "relative_predicted_decrease": 0.})
        for invalid_J in (np.zeros((5, 5)), np.full((5, 5), np.nan)):
            invalid = worker.local_gn_stationarity(np, residual, invalid_J, cost)
            self.assertFalse(invalid["valid"])
            self.assertIsNone(invalid["relative_predicted_decrease"])
            self.assertTrue(invalid["reason"])

    def test_real_native_and_kernel_on_artificial_noncoplanar_points(self):
        import poselib
        rng = np.random.default_rng(20261019)
        points = rng.uniform([-1.5, -1.1, 4.], [1.5, 1.1, 9.], (80, 3))
        true_R = real_kernel.so3_exp([.02, .08, -.01])
        true_t = np.array([.5, .1, .05])
        moved = points @ true_R.T + true_t
        camera = {"model": "PINHOLE", "width": 0, "height": 0,
                  "params": [800., 800., 320., 240.]}
        K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
        calibration = {"camera1": camera, "camera2": dict(camera), "K1": K, "K2": K}
        def project(p):
            return p[:, :2] / p[:, 2, None] * 800 + [320., 240.]
        x1, x2 = project(points), project(moved)
        R = real_kernel.so3_exp([.001, -.002, .001]) @ true_R
        t = true_t + [.005, -.003, .002]
        native, stats = worker.native_refine(poselib, x1, x2, calibration, R, t)
        N = real_kernel.refine_pose(x1, x2, K, K, R, t, mode="N")
        P = real_kernel.refine_pose(x1, x2, K, K, R, t, mode="P")
        costs = worker.common_objectives(base, real_kernel, x1, x2, calibration,
            {"original": {"R": R, "t": t}, "native": native, "N": N, "P": P})
        for name in ("native", "N", "P"):
            self.assertLess(costs["objectives"][name]["N"], costs["objectives"]["original"]["N"])
            self.assertAlmostEqual(costs["objectives"][name]["N"], costs["objectives"][name]["P"], places=9)
            worker.checked_pose(base, native if name == "native" else {"N": N, "P": P}[name])
        np.testing.assert_allclose(N["R"], P["R"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(N["t"], P["t"], atol=1e-12, rtol=0)
        self.assertAlmostEqual(stats["cost"] / (.5 / 800) ** 2,
                               costs["objectives"]["native"]["N"], delta=1e-7)


if __name__ == "__main__":
    unittest.main()
