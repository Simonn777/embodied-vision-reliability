"""Artificial-only worker tests: fake HDF5/kernel and generated JPEG bytes."""
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
import zlib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_test0015_photo_worker", HERE / "predict_pair.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
base, verifier, acq, real_selector, real_kernel = worker.load_dependencies()
np = base.np


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

    def __getitem__(self, name):
        if name not in self.values:
            raise AssertionError("unapproved HDF5 access: " + name)
        value, path = self.values[name], self.path + "/" + name
        if isinstance(value, dict):
            return GuardGroup(value, path, self.reads)
        return GuardDataset(value, path, self.reads)

    def __iter__(self):
        raise AssertionError("HDF5 enumeration prohibited")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def write_json(path, value):
    path.write_text(json.dumps(base.json_safe(value), sort_keys=True, allow_nan=False))
    return base.sha256(path)


class PhotometricWorkerTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "fake.h5"
        self.data.write_bytes(b"artificial HDF5 placeholder")
        self.manifest_path, self.seal_path = self.root / "manifest.json", self.root / "seal.json"
        self.archive_path, self.receipt_path = self.root / "archive.json", self.root / "receipt.json"
        self.protocol, self.output = self.root / "protocol.md", self.root / "output.json"
        self.protocol.write_text("Artificial protocol; no scientific arrays.\n")
        self.images = self.root / "private-images"
        (self.images / "images/0015").mkdir(parents=True)
        self.keys = ["0015-img%02d.jpg_0015-img%02d.jpg" % (i, 16+i % 7) for i in range(16)]
        self.key = self.keys[0]
        K = np.array([[80., 0, 32.], [0, 82., 32.], [0, 0, 1.]])
        camera = {"model": np.array([b"PINHOLE"]), "width": np.array([0]),
                  "height": np.array([0]), "params": np.array([80., 82., 32., 32.])}
        coordinates = np.full((10000, 2), 30.)
        coordinates[:, 0] += np.arange(10000) % 3
        self.values = {"x1": coordinates, "x2": coordinates+1., "K1": K, "K2": K.copy(),
                       "camera1": camera, "camera2": dict(camera)}
        self.h5 = GuardGroup({self.key: self.values})
        _, _, calibration = base.read_pair(GuardGroup({self.key: self.values}), self.key)
        self.calibration = base.json_safe(calibration)
        self.mask = [i < 300 for i in range(10000)]
        self.library = {"version": "fake", "h5py_version": "fake"}
        self.manifest = {"schema": "evr0012.development.v1", "coordinate_convention": "raw_pixels",
            "development_scene": "0015", "selected_pair_keys": self.keys,
            "dataset_bytes": self.data.stat().st_size, "dataset_sha256": base.sha256(self.data)}
        self.mh = write_json(self.manifest_path, self.manifest)
        sources = {name: base.sha256(worker.BASELINE_DIR/name) for name in verifier.SOURCE_NAMES}
        rows = []
        for i, key in enumerate(self.keys):
            record = {"schema": "evr0013.prediction.v1", "status": "completed", "pair_index": i,
                "key": key, "manifest_sha256": self.mh, "dataset_sha256_declared": self.manifest["dataset_sha256"],
                "source_sha256": sources["predict_pair.py"], "settings": base.SETTINGS,
                "library": self.library, "calibration": self.calibration,
                "estimate": {"R": np.eye(3).tolist(), "t": [1., 0., 0.]},
                "returned_ransac_stats": {"inliers": self.mask}}
            name = "pair-%02d.json" % i
            rows.append({"pair_index": i, "key": key, "path": name, "status": "completed",
                         "sha256": write_json(self.root/name, record)})
        self.seal = {"schema": "evr0013.seal.v1", "complete": True, "eligible_for_evaluation": True,
            "manifest_sha256": self.mh, "dataset_sha256_declared": self.manifest["dataset_sha256"],
            "source_sha256": sources, "protocol_sha256": base.sha256(
                verifier.REPO_ROOT/"protocols/0013-baseline-pose-headroom.md"), "rows": rows}
        write_json(self.seal_path, self.seal)
        # Generate a JPEG with orientation metadata: the worker must not transpose it.
        image = Image.fromarray(np.arange(64*48, dtype=np.uint8).reshape(48, 64))
        exif = Image.Exif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        self.jpeg = buffer.getvalue()
        entries = [{"filename": "images/0015/img%02d.jpg" % i,
            "local_header_offset": 1000*i, "compression_method": 8, "flags": 0,
            "compressed_bytes": 1, "uncompressed_bytes": len(self.jpeg),
            "crc32_hex": "%08x" % (zlib.crc32(self.jpeg) & 0xffffffff)} for i in range(23)]
        archive = {"schema": "evr0015.archive-metadata.v1", "url": acq.URL,
            "archive_bytes": acq.ARCHIVE_BYTES, "selected_entries": entries}
        self.ah = write_json(self.archive_path, archive)
        self.receipt = {"schema": acq.SCHEMA, "status": "completed", "stage": "completed",
            "selected_images": 23, "url": acq.URL, "archive_bytes": acq.ARCHIVE_BYTES,
            "manifest_sha256": self.mh, "expected_manifest_sha256": self.mh,
            "archive_manifest_sha256": self.ah, "expected_archive_manifest_sha256": self.ah,
            "source_sha256": base.sha256(HERE/"acquire_images.py"),
            "output_bytes": 23*len(self.jpeg), "images": [{**entry, **acq.jpeg_header(self.jpeg),
                "sha256": hashlib.sha256(self.jpeg).hexdigest()} for entry in entries]}
        write_json(self.receipt_path, self.receipt)
        # Deliberately only two files exist; no other image may be read.
        for i in (0, 16):
            (self.images/entries[i]["filename"]).write_bytes(self.jpeg)
        self.calls, self.mode = [], None

    def argv(self):
        return ["--data", str(self.data), "--manifest", str(self.manifest_path),
            "--baseline-seal", str(self.seal_path), "--images-root", str(self.images),
            "--image-receipt", str(self.receipt_path), "--pair-index", "0", "--output", str(self.output)]

    def fake_select(self, x1, x2, mask, width, height):
        self.assertEqual((width, height), (64, 48))
        self.assertEqual(json.loads(self.output.read_text())["phase"], "selection")
        self.assertTrue(np.array_equal(mask, self.mask))
        return [{"cell": i, "index": i, "status": "initialized", "A0": np.eye(2).tolist(),
                 "neighbor_count": 8, "neighbor_indices": list(range(100, 108))} for i in range(65)] + [
            {"cell": 66, "index": 66, "status": "initialization_failed", "reason": "too_few_neighbors"}]

    def fake_refine(self, im1, im2, p, q, A):
        marker = json.loads(self.output.read_text())
        self.assertEqual(marker["phase"], "photometric_batch_%03d" % len(self.calls))
        self.assertEqual(im1.shape, (48, 64))
        self.assertEqual(im1.dtype, np.float64)
        self.assertTrue(np.array_equal(p[0], self.values["x1"][len(self.calls)*64]-.5))
        self.assertTrue(np.array_equal(q[0], self.values["x2"][len(self.calls)*64]-.5))
        self.calls.append(len(p))
        if self.mode == "interrupt":
            raise KeyboardInterrupt("artificial interrupt")
        patches = []
        for i in range(len(p)):
            delta = np.array([.25, -.125]) if i > 1 else np.zeros(2)
            admitted = i != 1
            item = {"patch_index": i, "source": p[i].tolist(), "original_target": q[i].tolist(),
                "original_A": A[i].tolist(), "admitted": admitted, "changed": True,
                "status": "admitted" if admitted else "not_admitted", "reason": [] if admitted else ["flat"],
                "final": {"target": (q[i]+delta).tolist(), "delta": delta.tolist(),
                          "A": A[i].tolist(), "gain": 1.2, "bias": 0.1}, "attempts": []}
            patches.append(item)
        if self.mode == "nan":
            patches[0]["final"]["delta"][0] = float("nan")
        if self.mode == "pixels":
            patches[0]["residuals"] = [1., 2.]
        return {"schema": "evr0015.patch-refinement.v1", "settings": real_kernel.SETTINGS,
                "patches": patches, "timing_seconds": 0.01, "conventions": {}}

    def run_worker(self):
        fake_h5 = SimpleNamespace(File=lambda *args: self.h5, __version__="fake")
        fake_backend = SimpleNamespace()  # No estimator/refiner API exists.
        selector = SimpleNamespace(SETTINGS=real_selector.SETTINGS, select_and_initialize=self.fake_select)
        kernel = SimpleNamespace(SETTINGS=real_kernel.SETTINGS, refine_batch=self.fake_refine)
        def dependencies():
            self.assertEqual(json.loads(self.output.read_text())["phase"], "binding")
            return base, verifier, acq, selector, kernel

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(worker, "load_dependencies", dependencies))
            stack.enter_context(patch.object(worker, "ARCHIVE_PATH", self.archive_path))
            stack.enter_context(patch.object(worker, "ACQUISITION_PATH", self.receipt_path))
            stack.enter_context(patch.object(worker, "PROTOCOL_PATH", self.protocol))
            stack.enter_context(patch.object(acq, "EXPECTED_MANIFEST_SHA256", self.mh))
            stack.enter_context(patch.object(acq, "EXPECTED_ARCHIVE_MANIFEST_SHA256", self.ah))
            stack.enter_context(patch.object(base, "library_metadata", lambda backend: {"version": "fake"}))
            stack.enter_context(patch.dict(sys.modules, {"h5py": fake_h5, "poselib": fake_backend}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return worker.main(self.argv())

    def test_complete_allowlist_two_jpegs_offsets_batches_and_all_failures(self):
        self.assertEqual(self.run_worker(), 0)
        r = json.loads(self.output.read_text())
        self.assertEqual(r["schema"], worker.SCHEMA)
        self.assertEqual(self.calls, [64, 1])
        self.assertEqual((r["selected_count"], r["initialized_count"], r["admitted_count"]), (66, 65, 64))
        self.assertEqual(r["target_coordinate_changed_count"], 62)
        self.assertEqual(r["calibration"], self.calibration)
        self.assertEqual(r["mask_sha256"], hashlib.sha256(bytes(self.mask)).hexdigest())
        self.assertEqual(len(self.h5.reads), 12)
        self.assertTrue(all("/R" not in x and "/t" not in x for x in self.h5.reads))
        self.assertEqual(r["queries"][0]["status"], "admitted")
        self.assertFalse(r["queries"][0]["target_coordinate_changed"])
        self.assertEqual(r["queries"][1]["delta_cache"], [0., 0.])
        self.assertEqual(r["queries"][-1]["status"], "initialization_failed")
        self.assertEqual(r["queries"][2]["delta_cache"], [.25, -.125])
        self.assertNotIn(str(self.root), self.output.read_text())

    def test_all23_receipt_binding_fails_before_any_coordinate_or_image_read(self):
        self.receipt["images"][22]["crc32_hex"] = "12345678"
        write_json(self.receipt_path, self.receipt)
        self.assertEqual(self.run_worker(), 1)
        r = json.loads(self.output.read_text())
        self.assertEqual(r["failure_kind"], "binding")
        self.assertEqual(self.h5.reads, [])
        self.assertEqual(self.calls, [])

    def test_other_baseline_pair_tamper_rejected_before_data(self):
        (self.root/"pair-15.json").write_text("{}")
        self.assertEqual(self.run_worker(), 1)
        self.assertEqual(self.h5.reads, [])

    def test_selected_jpeg_bytes_rejected_before_decode_and_fit(self):
        (self.images/"images/0015/img00.jpg").write_bytes(b"x"*len(self.jpeg))
        self.assertEqual(self.run_worker(), 1)
        r = json.loads(self.output.read_text())
        self.assertEqual((r["phase"], r["failure_kind"]), ("images", "input"))
        self.assertEqual(self.calls, [])

    def test_nonfinite_kernel_output_preserves_all_original_rows(self):
        self.mode = "nan"
        self.assertEqual(self.run_worker(), 1)
        r = json.loads(self.output.read_text())
        self.assertEqual(r["failure_kind"], "photometric")
        self.assertEqual(len(r["queries"]), 66)
        self.assertTrue(all(x["delta_cache"] == [0., 0.] for x in r["queries"]))
        self.assertNotIn("NaN", self.output.read_text())

    def test_unexpected_patch_array_is_not_exported(self):
        self.mode = "pixels"
        self.assertEqual(self.run_worker(), 1)
        self.assertNotIn('"residuals"', self.output.read_text())

    def test_interruption_records_error_and_reraises(self):
        self.mode = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_worker()
        r = json.loads(self.output.read_text())
        self.assertEqual(r["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(r["status"], "error")
        self.assertIn("photometric_batch_000", r["timing_seconds"])

    def test_existing_output_refused_without_import_or_read(self):
        self.output.write_text("preserve")
        with patch.object(worker, "load_dependencies", side_effect=AssertionError("read")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(self.argv()), 2)
        self.assertEqual(self.output.read_text(), "preserve")

    def test_decode_has_no_resize_or_exif_transpose(self):
        entry = self.receipt["images"][0]
        result = worker.decode_image(base, acq, self.images, entry)
        from PIL import Image
        with Image.open(io.BytesIO(self.jpeg)) as image:
            wanted = np.asarray(image.convert("L"), dtype=np.float64)/255.
        self.assertEqual(result.shape, (48, 64))
        np.testing.assert_array_equal(result, wanted)

    def test_real_kernel_on_artificial_texture_preserves_admitted_zero_change(self):
        rng = np.random.default_rng(123)
        image = rng.uniform(.1, .9, (72, 72))
        x = np.array([[36.25, 36.75]])
        selector = SimpleNamespace(select_and_initialize=lambda *args: [{
            "cell": 0, "index": 0, "status": "initialized", "A0": np.eye(2).tolist()}])
        rows = worker.initialize_queries(base, selector, x, x, np.array([True]), 72, 72)
        result = worker.apply_batch(base, real_kernel, rows, [0], image, image)
        self.assertEqual(result["schema"], "evr0015.patch-refinement.v1")
        self.assertTrue(rows[0]["admitted"])
        self.assertFalse(rows[0]["target_coordinate_changed"])
        self.assertEqual(rows[0]["delta_cache"], [0., 0.])
        self.assertEqual(rows[0]["target_cache"], x[0].tolist())


if __name__ == "__main__":
    unittest.main()
