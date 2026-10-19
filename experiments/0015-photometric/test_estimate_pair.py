"""Fake HDF5/backend tests; no actual images, acquired data, GT, or native fit."""
import contextlib
import copy
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

_spec = importlib.util.spec_from_file_location("estimate_pair", Path(__file__).with_name("estimate_pair.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)
BASE, VERIFIER = w.load_dependencies()
np = BASE.np


class Dataset:
    def __init__(self, value, path, reads):
        self.value, self.path, self.reads = value, path, reads
    def __getitem__(self, item):
        if item != ():
            raise AssertionError("non-whole dataset read")
        self.reads.append(self.path)
        return self.value


class Group:
    def __init__(self, values, reads, path=""):
        self.values, self.reads, self.path = values, reads, path
    def __getitem__(self, key):
        if key not in self.values or key in ("R", "t", "confidence", "depth"):
            raise AssertionError("forbidden field: "+key)
        value, path = self.values[key], self.path+"/"+key
        return Group(value, self.reads, path) if isinstance(value, dict) else Dataset(value, path, self.reads)
    def __iter__(self):
        raise AssertionError("group enumeration forbidden")
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


class Fixture:
    def __init__(self, directory, successes=10):
        self.directory = Path(directory)
        self.data = self.directory/"fake.h5"
        self.data.write_bytes(b"not an HDF5 file; never read")
        self.manifest_path, self.seal_path, self.photo_path, self.output = [self.directory/n for n in
            ("manifest.json", "baseline-seal.json", "photometric.json", "geometric.json")]
        self.keys = [f"0015-artificial-{i:02d}" for i in range(16)]
        self.index, self.key = 3, self.keys[3]
        self.manifest = {"schema": "evr0012.development.v1", "coordinate_convention": "raw_pixels",
                         "development_scene": "0015", "selected_pair_keys": self.keys,
                         "dataset_bytes": self.data.stat().st_size, "dataset_sha256": "d"*64}
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.seal_path.write_text(json.dumps({"synthetic_complete_cohort": self.keys}))
        self.reads, self.file_calls, self.calls, self.seal_calls = [], [], [], []
        a = np.arange(10000, dtype=float)
        self.x1 = np.stack((30+a % 100, 40+np.floor(a/100)), axis=1)
        self.x2 = self.x1+[1.5, 2.25]
        camera = {"model": np.array([b"PINHOLE"]), "width": np.array([640]), "height": np.array([480]),
                  "params": np.array([800., 820., 320., 240.])}
        K = np.array([[800., 0., 320.], [0., 820., 240.], [0., 0., 1.]])
        self.values = {"x1": self.x1, "x2": self.x2, "K1": K, "K2": K.copy(),
                       "camera1": camera, "camera2": copy.deepcopy(camera)}
        root = Group({self.key: self.values}, self.reads)
        _, _, calibration = BASE.read_pair(root, self.key)
        self.reads.clear()
        self.calibration = BASE.json_safe(calibration)
        self.library = {"version": "2.0.5", "binary_sha256": "a"*64, "h5py_version": "fake-h5py"}
        mask = [True]*9999+[False]
        self.baseline = {"status": "completed", "settings": copy.deepcopy(BASE.SETTINGS),
                         "returned_ransac_stats": {"inliers": mask}, "library": self.library,
                         "calibration": self.calibration, "estimate": {"R": np.eye(3).tolist(), "t": [1, 0, 0]}}
        self.row = {"sha256": "b"*64}
        self.bindings = {"source_sha256": "1"*64, "kernel_sha256": "2"*64,
                         "selector_sha256": "3"*64, "protocol_sha256": "4"*64}
        self.metadata = {"image_acquisition_receipt_sha256": "5"*64, "settings": {"synthetic": True},
                         "kernel_settings": {"synthetic": True}, "selector_settings": {"synthetic": True}}
        ids = [5, 17, 42, 63, 81, 107, 215, 318, 441, 557, 614, 799]
        queries = []
        for ordinal, index in enumerate(ids):
            admitted = ordinal < successes
            original = self.x2[index]
            delta = np.array([0.25, -0.125]) if admitted and ordinal != 0 else np.zeros(2)
            target = original+delta
            initialization = {"index": index, "status": "initialized", "A0": np.eye(2).tolist()}
            kernel = {"admitted": admitted, "source": (self.x1[index]-0.5).tolist(),
                      "original_target": (original-0.5).tolist(), "original_A": np.eye(2).tolist(),
                      "final": {"target": (target-0.5).tolist()}}
            queries.append({"index": index, "cell": ordinal, "initialization": initialization,
                            "original_source_cache": self.x1[index].tolist(), "original_target_cache": original.tolist(),
                            "target_cache": target.tolist(), "delta_cache": delta.tolist(),
                            "status": "admitted" if admitted else "fit_failed", "admitted": admitted,
                            "target_coordinate_changed": bool(np.any(delta != 0)), "photometric": kernel})
        self.photo = {"schema": "evr0015.photometric.v1", "status": "completed", "pair_index": self.index,
                      "key": self.key, "manifest_sha256": BASE.sha256(self.manifest_path),
                      "baseline_seal_sha256": BASE.sha256(self.seal_path), "baseline_prediction_sha256": self.row["sha256"],
                      "dataset_sha256_declared": "d"*64, "mask_sha256": hashlib.sha256(bytes(mask)).hexdigest(),
                      "kept_count": sum(mask), "library": self.library, "calibration": self.calibration,
                      "queries": queries, "selected_indices": ids, "selected_count": 12,
                      "initialized_count": 12, "admitted_count": successes,
                      "target_coordinate_changed_count": max(0, successes-1), "timing_seconds": {"total": 0.1},
                      **self.bindings, **self.metadata}
        self.save_photo()
        self.fail_arm, self.bad_mask, self.on_call = None, False, None

    def save_photo(self):
        self.photo_path.write_text(json.dumps(self.photo, allow_nan=False))
        self.photo_sha = BASE.sha256(self.photo_path)

    def verify_seal(self, manifest, seal):
        self.seal_calls.append((manifest, seal))
        return (self.manifest, {}, [(self.row, self.baseline)]*16,
                BASE.sha256(self.manifest_path), BASE.sha256(self.seal_path))

    def open_h5(self, path, mode):
        assert path == self.data and mode == "r"
        self.file_calls.append((path, mode))
        return Group({self.key: self.values}, self.reads)

    def native(self, x1, x2, c1, c2, options, bundle, initial_pose):
        marker = json.loads(self.output.read_text())
        arm = marker["phase"]
        assert marker["status"] == "started" and marker["arms"][arm]["status"] == "started"
        assert options == BASE.RANSAC_OPTIONS and bundle == {} and initial_pose is None
        self.calls.append((arm, x1.copy(), x2.copy(), c1, c2))
        if self.on_call:
            self.on_call(self, arm)
        if arm == self.fail_arm:
            raise RuntimeError("artificial native failure")
        count = len(x1)-1 if self.bad_mask else len(x1)
        return SimpleNamespace(R=np.eye(3), t=np.array([1., 0., 0.])), {"inliers": [True]*count, "iterations": 1001}

    def run(self, **extra_patches):
        args = ["--data", str(self.data), "--manifest", str(self.manifest_path),
                "--baseline-seal", str(self.seal_path), "--photometric-record", str(self.photo_path),
                "--photometric-sha256", self.photo_sha, "--pair-index", str(self.index), "--output", str(self.output)]
        verifier = SimpleNamespace(verify_seal=self.verify_seal, load_hashed_json=VERIFIER.load_hashed_json)
        fake_h5 = SimpleNamespace(File=self.open_h5, __version__="fake-h5py")
        fake_backend = SimpleNamespace(estimate_relative_pose=self.native)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(w, "load_dependencies", return_value=(BASE, verifier)))
            stack.enter_context(patch.object(w, "source_bindings", **extra_patches.get("bindings", {"return_value": self.bindings})))
            stack.enter_context(patch.object(w, "expected_photometric_metadata", return_value=self.metadata))
            stack.enter_context(patch.object(BASE, "library_metadata", return_value={k: v for k, v in self.library.items() if k != "h5py_version"}))
            stack.enter_context(patch.dict(sys.modules, {"h5py": fake_h5, "poselib": fake_backend,
                "PIL": SimpleNamespace(__getattr__=lambda *_: (_ for _ in ()).throw(AssertionError("image access forbidden")))}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = w.main(args)
        return result, json.loads(self.output.read_text())


class GeometricTests(unittest.TestCase):
    def test_three_calls_same_indices_full_replacement_and_allowlist(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            code, out = f.run()
            self.assertEqual(code, 0, out)
            self.assertEqual([c[0] for c in f.calls], list(w.ARM_ORDER))
            ids = f.photo["selected_indices"][:10]
            self.assertEqual(out["success_query_indices"], ids)
            self.assertEqual([len(c[1]) for c in f.calls], [10, 10, 10000])
            np.testing.assert_array_equal(f.calls[0][1], f.x1[ids])
            np.testing.assert_array_equal(f.calls[1][1], f.calls[0][1])
            np.testing.assert_array_equal(f.calls[0][2], f.x2[ids])
            expected = f.x2.copy()
            for q in f.photo["queries"]:
                if q["admitted"]:
                    expected[q["index"]] = q["target_cache"]
            np.testing.assert_array_equal(f.calls[1][2], expected[ids])
            np.testing.assert_array_equal(f.calls[2][2], expected)
            self.assertEqual(out["arms"]["subset_refined"]["returned_inlier_indices_original"], ids)
            self.assertEqual(len(out["photometric_queries"]), 12)
            self.assertFalse(out["full_original_reference"]["rerun"])
            self.assertEqual(out["settings"], BASE.SETTINGS)
            self.assertEqual(out["native_calls_started"], 3)
            self.assertEqual(len(f.seal_calls), 1)
            self.assertEqual(len(f.reads), 12)
            self.assertEqual({p.rsplit("/", 1)[-1] for p in f.reads}, {"x1", "x2", "K1", "K2", "model", "width", "height", "params"})

    def test_too_few_successes_and_zero_change_still_run_full_once(self):
        for count in (0, 7):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as d:
                f = Fixture(d, successes=count)
                code, out = f.run()
                self.assertEqual(code, 0, out)
                self.assertEqual(len(f.calls), 1)
                self.assertEqual(f.calls[0][0], "full_refined")
                self.assertEqual(out["arms"]["subset_original"]["status"], "unavailable")
                self.assertEqual(out["arms"]["subset_refined"]["status"], "unavailable")
                if count == 0:
                    np.testing.assert_array_equal(f.calls[0][2], f.x2)

    def test_each_native_failure_retained_no_retry(self):
        for arm in w.ARM_ORDER:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as d:
                f = Fixture(d)
                f.fail_arm = arm
                code, out = f.run()
                self.assertEqual(code, 1)
                self.assertEqual(len(f.calls), 3)
                self.assertEqual(out["failed_calls"], 1)
                self.assertEqual(out["native_calls_completed"], 2)
                self.assertEqual(out["arms"][arm]["status"], "error")

    def test_interruption_preserves_completed_and_active_arms_then_reraises(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            def interrupt(fixture, arm):
                if arm == "subset_refined":
                    raise KeyboardInterrupt("artificial interruption")
            f.on_call = interrupt
            with self.assertRaises(KeyboardInterrupt):
                f.run()
            out = json.loads(f.output.read_text())
            self.assertEqual(out["status"], "error")
            self.assertTrue(out["interrupted"])
            self.assertEqual(out["failure_phase"], "subset_refined")
            self.assertEqual(out["arms"]["subset_original"]["status"], "completed")
            interrupted = out["arms"]["subset_refined"]
            self.assertEqual(interrupted["status"], "error")
            self.assertTrue(interrupted["interrupted"])
            self.assertGreaterEqual(interrupted["call_seconds"], 0)
            self.assertGreaterEqual(out["timing_seconds"]["worker_elapsed"], interrupted["call_seconds"])
            self.assertEqual(out["native_calls_started"], 2)
            self.assertEqual(out["native_calls_completed"], 1)
            self.assertEqual(out["failed_calls"], 1)
            self.assertNotIn("full_refined", out["arms"])
            self.assertEqual(len(f.calls), 2)

    def test_record_bindings_fail_before_h5(self):
        for field in ("manifest_sha256", "baseline_seal_sha256", "baseline_prediction_sha256", "source_sha256",
                      "kernel_sha256", "selector_sha256", "protocol_sha256", "mask_sha256", "dataset_sha256_declared",
                      "image_acquisition_receipt_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                f = Fixture(d)
                f.photo[field] = "f"*64
                f.save_photo()
                code, out = f.run()
                self.assertEqual(code, 1)
                self.assertFalse(f.file_calls)
                self.assertFalse(f.calls)

    def test_photo_sha_and_library_calibration_bindings(self):
        for kind in ("sha", "library", "calibration", "settings"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                f = Fixture(d)
                if kind == "sha": f.photo_sha = "0"*64
                if kind == "library": f.photo["library"] = {"bad": True}; f.save_photo()
                if kind == "calibration": f.photo["calibration"] = {"bad": True}; f.save_photo()
                if kind == "settings": f.baseline["settings"]["ransac_options"]["seed"] = 1
                code, _ = f.run()
                self.assertEqual(code, 1)
                self.assertFalse(f.file_calls)

    def test_query_index_inlier_and_failed_mapping_protection(self):
        for kind in ("duplicate", "not_inlier", "changed_failure", "kernel_mapping", "delta", "counter"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                f = Fixture(d)
                if kind == "duplicate": f.photo["selected_indices"][1] = f.photo["selected_indices"][0]
                if kind == "not_inlier": f.photo["selected_indices"][0] = 9999
                if kind == "changed_failure": f.photo["queries"][-1]["target_cache"][0] += 0.2
                if kind == "kernel_mapping": f.photo["queries"][1]["photometric"]["final"]["target"][0] += 0.2
                if kind == "delta": f.photo["queries"][1]["delta_cache"][0] += 0.2
                if kind == "counter": f.photo["admitted_count"] -= 1
                f.save_photo()
                code, _ = f.run()
                self.assertEqual(code, 1)
                self.assertFalse(f.calls)
                self.assertFalse(f.file_calls)

    def test_cache_original_mismatch_blocks_native(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            f.values["x1"] = f.x1.copy()
            f.values["x1"][5, 0] += 1
            code, out = f.run()
            self.assertEqual(code, 1)
            self.assertEqual(out["failure_phase"], "input")
            self.assertEqual(len(f.file_calls), 1)
            self.assertFalse(f.calls)

    def test_incomplete_or_inconsistent_query_and_bad_timing_refused(self):
        for kind in ("not_attempted", "initialization", "fit", "missing_timing", "negative_timing"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                f = Fixture(d)
                if kind == "not_attempted": f.photo["queries"][-1]["status"] = "not_attempted"
                if kind == "initialization": f.photo["queries"][-1]["initialization"]["status"] = "initialization_failed"
                if kind == "fit": f.photo["queries"][-1]["photometric"]["admitted"] = True
                if kind == "missing_timing": del f.photo["timing_seconds"]
                if kind == "negative_timing": f.photo["timing_seconds"]["total"] = -1
                f.save_photo()
                code, _ = f.run()
                self.assertEqual(code, 1)
                self.assertFalse(f.file_calls)

    def test_native_inlier_order_shape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            f.bad_mask = True
            code, out = f.run()
            self.assertEqual(code, 1)
            self.assertEqual(out["failed_calls"], 3)
            self.assertTrue(all(v["failure_kind"] == "output" for v in out["arms"].values()))

    def test_post_call_source_or_photo_mutation_is_error(self):
        for kind in ("source", "photo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                f = Fixture(d)
                extras = {}
                if kind == "source":
                    extras["bindings"] = {"side_effect": [f.bindings, {**f.bindings, "kernel_sha256": "9"*64}]}
                else:
                    f.on_call = lambda fixture, arm: fixture.photo_path.write_text("{}") if arm == "full_refined" else None
                code, out = f.run(**extras)
                self.assertEqual(code, 1)
                self.assertEqual(len(f.calls), 3)
                self.assertEqual(out["native_calls_completed"], 3)

    def test_existing_output_refusal_and_dependency_failure_marker(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d)
            f.output.write_text("preserved")
            with patch.object(w, "load_dependencies", side_effect=AssertionError("should not import")):
                args = ["--data", str(f.data), "--manifest", str(f.manifest_path), "--baseline-seal", str(f.seal_path),
                        "--photometric-record", str(f.photo_path), "--photometric-sha256", f.photo_sha,
                        "--pair-index", "3", "--output", str(f.output)]
                self.assertEqual(w.main(args), 2)
                self.assertEqual(f.output.read_text(), "preserved")
                f.output.unlink()
                self.assertEqual(w.main(args), 1)
            out = json.loads(f.output.read_text())
            self.assertEqual(out["status"], "error")
            self.assertEqual(out["failure_phase"], "binding")


if __name__ == "__main__":
    unittest.main()
