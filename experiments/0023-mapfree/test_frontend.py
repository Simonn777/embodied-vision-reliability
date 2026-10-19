"""Manufactured images and fake model outputs only; never loads Torch/weights."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zlib

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("_evr0023_frontend_test", Path(__file__).with_name("frontend.py"))
front = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = front
spec.loader.exec_module(front)
planner = front.module("_evr0023_plan_fixture", Path(__file__).with_name("propose_plan.py"))


class FakeBackend:
    np, Image = np, Image

    def __init__(self, fail_depth=None, interrupt=False, no_depth=False):
        self.depth_calls = self.feature_calls = self.match_calls = 0
        self.fail_depth, self.interrupt, self.no_depth = fail_depth, interrupt, no_depth

    def depth(self, image):
        self.depth_calls += 1
        if self.depth_calls == self.fail_depth:
            if self.interrupt:
                raise KeyboardInterrupt("artificial interruption")
            raise RuntimeError("artificial inference failure")
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        depth = (1 + xx + yy).astype(np.float32)
        mask = np.full((h, w), not self.no_depth, dtype=np.bool_)
        return dict(depth=depth, mask=mask, points=np.stack((xx, yy, depth), -1).astype(np.float32),
                    intrinsics=np.eye(3, dtype=np.float32), normal=np.ones((h, w, 3), dtype=np.float32))

    def features(self, image):
        self.feature_calls += 1
        h, w = image.shape[:2]
        return dict(keypoints=np.asarray([[[1.25, 2.5], [w - 1, 1]]], dtype=np.float32),
                    descriptors=np.ones((1, 2, 256), dtype=np.float32),
                    keypoint_scores=np.asarray([[1, .9]], dtype=np.float32),
                    image_size=np.asarray([[w, h]], dtype=np.float32))

    def match(self, feature0, feature1):
        self.match_calls += 1
        return dict(matches=np.asarray([[0, 0], [1, 1]], dtype=np.int64),
                    scores=np.asarray([.8, .7], dtype=np.float32),
                    matches0=np.asarray([[0, 1]], dtype=np.int64), matches1=np.asarray([[0, 1]], dtype=np.int64),
                    matching_scores0=np.asarray([[.8, .7]], dtype=np.float32),
                    matching_scores1=np.asarray([[.8, .7]], dtype=np.float32),
                    prune0=np.asarray([[9, 9]], dtype=np.int64), prune1=np.asarray([[9, 9]], dtype=np.int64),
                    stop=np.asarray(9, dtype=np.int64))


class Fixture:
    def __init__(self, root):
        self.root, self.inputs, self.output = root, root / "inputs", root / "outputs"
        self.inputs.mkdir(); self.output.mkdir()
        entries, contents = {}, {}
        yy, xx = np.mgrid[:8, :10]
        pixels = np.stack((xx * 20, yy * 30, (xx + yy) * 12), -1).astype(np.uint8)
        stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format="PNG")
        # .jpg names carry manufactured PNG bytes to test decoder format handling.
        for scene in planner.EXPECTED_SCENES:
            names = ["seq0/frame_00000.jpg"] + ["seq1/frame_%05d.jpg" % i for i in range(16)]
            lines = ["%s 8 9 4.5 3.5 10 8" % n for n in names]
            for name in names:
                contents["val/%s/%s" % (scene, name)] = stream.getvalue()
            contents["val/%s/intrinsics.txt" % scene] = ("\n".join(lines) + "\n").encode()
            # GT exists as *directory metadata only*, never as a local payload.
            entries["val/%s/poses.txt" % scene] = dict(filename="val/%s/poses.txt" % scene, flags=0,
                compression_method=8, crc32_hex="00000000", compressed_bytes=7, uncompressed_bytes=8, local_header_offset=0)
        for name, data in contents.items():
            entries[name] = dict(filename=name, flags=0, compression_method=8,
                                 crc32_hex="%08x" % (zlib.crc32(data) & 0xffffffff),
                                 compressed_bytes=len(data), uncompressed_bytes=len(data), local_header_offset=0)
            path = self.inputs / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        index = planner.ranges.Index(planner.ranges.URL, '"artificial"', 1000000, 900000,
                                     entries, {k: "0" * 64 for k in ("members", "directory", "receipt")})
        self.plan = planner.build_plan(index, "1" * 64)
        self.plan_path = root / "plan.json"
        front.write_json_new(self.plan_path, self.plan)
        members = [dict(entries[name], sha256=front.q.sha(self.inputs / name))
                   for name in self.plan["frontend_acquisition"]["member_names"]]
        self.receipt_path = root / "acquisition.json"
        budget = self.plan["frontend_acquisition"]
        requests = [dict(status=206, body_bytes=size) for row in members
                    for size in (30, len(row["filename"]), row["compressed_bytes"])]
        receipt = dict(schema=front.ranges.SCHEMA, status="completed", response_body_bytes_exact=True,
                       pose_gt_members_allowed=False, source_sha256=front.q.sha(front.HERE / "range_inputs.py"),
                       url=index.url, etag=index.etag, archive_bytes=index.archive_bytes,
                       index_sha256=index.fingerprints, selected_names=[m["filename"] for m in members],
                       limits=dict(front.RANGE_LIMITS, output_bytes=budget["uncompressed_bytes"],
                                   member_bytes=budget["maximum_member_uncompressed_bytes"]),
                       wall_seconds=.01, response_body_bytes=sum(row["body_bytes"] for row in requests),
                       output_bytes=sum(len(v) for v in contents.values()), requests=requests, members=members)
        front.write_json_new(self.receipt_path, receipt)
        self.args = SimpleNamespace(plan=self.plan_path, plan_sha256=front.q.sha(self.plan_path),
                                    input_receipt=self.receipt_path, input_receipt_sha256=front.q.sha(self.receipt_path),
                                    inputs=self.inputs)


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_failed_acquisition_outer_blocks_before_model_loading(self):
        args = SimpleNamespace(output_dir=self.root / "fresh-frontend", expected_commit="a" * 40,
                               frontend_freeze="b" * 40, plan_sha256="c" * 64,
                               input_receipt_sha256="d" * 64, protocol=self.root / "protocol.md")
        with mock.patch.object(front.acquisition.supervisor, "verify_frozen", return_value={}), \
             mock.patch.object(front, "qualification_args", return_value=object()), \
             mock.patch.object(front.q, "frozen_bindings", return_value={}), \
             mock.patch.object(front.q, "verify_files", return_value=({}, None, {}, {})), \
             mock.patch.object(front, "verify_qualification", return_value={}), \
             mock.patch.object(front.acquisition, "verify_completed_chain", side_effect=ValueError("failed outer")), \
             mock.patch.object(front, "verify_inputs") as inputs, mock.patch.object(front, "Backend") as backend:
            with self.assertRaisesRegex(ValueError, "failed outer"):
                front.run(args)
            backend.assert_not_called(); inputs.assert_not_called()
        record = front.q.read_json(args.output_dir / "receipt.json")
        self.assertEqual((record["status"], record["eligible"], record["failed_stage"]),
                         ("failed", False, "verification"))

    def test_author_intrinsics_by_name_skips_unselected_values_and_rejects_duplicates(self):
        text = "# frame_path fx fy cx cy frame_width frame_height\nseq1/frame_00003.jpg 2 3 4 5 8 6\nseq1/frame_00009.jpg unused values not parsed here .\n"
        parsed = front.parse_intrinsics(text, ["seq1/frame_00003.jpg"])
        self.assertEqual(parsed["seq1/frame_00003.jpg"]["width"], 8)
        for bad in (text + "seq1/frame_00003.jpg 2 3 4 5 8 6\n", "poses.txt 2 3 4 5 8 6\n",
                    "seq1/frame_00003.jpg nan 3 4 5 8 6\n", "seq1/frame_00003.jpg 2 3 4 5 8.5 6\n"):
            with self.assertRaises(ValueError):
                front.parse_intrinsics(bad, ["seq1/frame_00003.jpg"])

    def test_rounding_and_half_pixel_intrinsics_are_axis_specific(self):
        k = dict(width=1001, height=701, fx=1040., fy=1030., cx=500., cy=350.)
        original, resized, size, scales = front.resize_geometry(k, np)
        self.assertEqual(size, (512, 359))
        self.assertEqual(resized[0, 0], original[0, 0] * 512 / 1001)
        self.assertAlmostEqual(resized[1, 2], (350.5) * 359 / 701 - .5)
        self.assertNotEqual(scales[0], scales[1])
        k.update(width=10, height=8)
        original, resized, size, scales = front.resize_geometry(k, np)
        np.testing.assert_array_equal(original, resized)
        self.assertEqual((size, scales), ((10, 8), (1., 1.)))

    def test_bilinear_depth_four_neighbors_median_and_invalid_flags(self):
        depth = np.asarray([[2, 4], [6, 8]], dtype=np.float32)
        mask = np.ones((2, 2), dtype=np.bool_)
        xy = np.asarray([[.25, .5], [1, 0], [0, 0]], dtype=np.float32)
        out = front.sample_depth(depth, mask, xy, np)
        self.assertEqual(float(out["depth_positive_valid_median"]), 5.)
        self.assertEqual(out["depth_sample_raw"][0], 4.5)
        self.assertEqual(out["depth_sample_normalized"][0], .9)
        np.testing.assert_array_equal(out["depth_sample_valid"], [True, False, True])
        self.assertTrue(np.isnan(out["depth_sample_raw"][1]))
        mask[1, 1] = False
        self.assertFalse(front.sample_depth(depth, mask, xy, np)["depth_sample_valid"].any())
        out = front.sample_depth(depth, np.zeros_like(mask), xy, np)
        self.assertTrue(np.isnan(out["depth_positive_valid_median"]))
        self.assertFalse(out["depth_sample_valid"].any())

    def test_manufactured_image_dimensions_rgb_hash_and_cache_geometry(self):
        path = self.root / "manufactured.png"
        pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        Image.fromarray(pixels).save(path)
        calibration = dict(width=10, height=8, fx=8., fy=9., cx=4.5, cy=3.5)
        values, arrays, meta = front.prepare_image(path, calibration, np, Image)
        self.assertEqual(values.shape, (8, 10, 3))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(meta["original_rgb_uint8_sha256"], front.raw_sha(pixels))
        self.assertEqual(meta["resized_rgb_uint8_sha256"], meta["original_rgb_uint8_sha256"])
        np.testing.assert_array_equal(arrays["K_original"], arrays["K_resized"])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            front.prepare_image(path, dict(calibration, width=11), np, Image)

    def test_all54_hashes_verified_no_gt_path_and_tamper_block(self):
        box = Fixture(self.root)
        actual_open = Path.open
        opened = []
        def audited(path, *args, **kwargs):
            if "poses" in path.name:
                raise AssertionError("GT read attempted")
            opened.append(path)
            return actual_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", audited):
            plan, hashes = front.verify_inputs(box.args)
        self.assertEqual(len(hashes), 54)
        self.assertFalse(any("poses" in str(path) for path in opened))
        first = box.inputs / box.plan["frontend_acquisition"]["image_names"][0]
        data = first.read_bytes(); first.write_bytes(b"X" + data[1:])
        with self.assertRaisesRegex(ValueError, "member bytes"):
            front.verify_inputs(box.args)
        box.args.plan_sha256 = "f" * 64
        with self.assertRaisesRegex(ValueError, "plan hash"):
            front.verify_inputs(box.args)

    def test_complete_fake51image120edge_run_keeps_invalid_depth_observations(self):
        box = Fixture(self.root)
        backend, record = FakeBackend(), {}
        front.process(box.plan, box.inputs, box.output, backend, record, lambda: None)
        self.assertEqual((backend.depth_calls, backend.feature_calls, backend.match_calls), (51, 51, 120))
        self.assertTrue(all(row["status"] == "completed" for row in record["images"] + record["edges"]))
        self.assertEqual(len(record["images"]), 51)
        self.assertEqual(len(record["edges"]), 120)
        with np.load(box.output / "image-000.npz", allow_pickle=False) as image:
            self.assertEqual(image["xy"].shape, (2, 2))
            np.testing.assert_array_equal(image["depth_sample_valid"], [True, False])
            self.assertEqual(image["depth_sample_normalized"][0], image["depth_sample_raw"][0] / image["depth_positive_valid_median"])
            self.assertTrue({"depth", "points", "mask", "intrinsics", "normal", "descriptors"} <= set(image.files))
        with np.load(box.output / "edge-000.npz", allow_pickle=False) as edge:
            self.assertEqual(edge["xy0"].shape, (2, 2))
            self.assertEqual(edge["matches"].shape, (2, 2))
            self.assertEqual(edge["match_ids"].shape, (2, 2))
            self.assertEqual(edge["match_scores"].shape, (2,))
            self.assertFalse(edge["depth_valid0"][1])
            self.assertTrue(np.isnan(edge["depth0_raw"][1]))
            self.assertIn("prune0", edge.files)
        self.assertTrue(all(row["geometry_observations_removed_for_depth"] == 0 for row in record["edges"]))
        for row in record["images"] + record["edges"]:
            self.assertEqual(front.q.sha(box.output / row["cache"]["filename"]), row["cache"]["sha256"])
        json.dumps(record, allow_nan=False)

    def test_acquisition_limits_cannot_be_redefined_in_rehashed_receipt(self):
        box = Fixture(self.root)
        original = front.q.read_json(box.receipt_path)
        def check_changed(value):
            box.receipt_path.write_text(json.dumps(value))
            box.args.input_receipt_sha256 = front.q.sha(box.receipt_path)
            with self.assertRaises(ValueError):
                front.verify_inputs(box.args)
        changed = copy.deepcopy(original)
        changed["limits"]["body_bytes"] *= 2
        check_changed(changed)
        changed = copy.deepcopy(original)
        changed["response_body_bytes"] = box.plan["frontend_acquisition"]["response_body_upper_bound"] + 1
        check_changed(changed)
        changed = copy.deepcopy(original)
        changed["output_bytes"] -= 1
        check_changed(changed)
        changed = copy.deepcopy(original)
        changed["requests"].pop()
        check_changed(changed)

    def test_inference_error_preserves_image_and_all_edge_denominators(self):
        box = Fixture(self.root)
        backend, record = FakeBackend(fail_depth=1), {}
        front.process(box.plan, box.inputs, box.output, backend, record, lambda: None)
        self.assertEqual((backend.depth_calls, backend.feature_calls, backend.match_calls), (51, 50, 104))
        self.assertEqual(record["images"][0]["status"], "error")
        self.assertEqual(record["images"][0]["failure_stage"], "moge")
        self.assertEqual(sum(r["status"] == "error" for r in record["edges"]), 16)
        self.assertEqual(len(record["edges"]), 120)
        self.assertTrue((box.output / "image-000-partial.npz").is_file())

    def test_interrupt_records_active_attempt_and_never_retries(self):
        box = Fixture(self.root)
        backend, record = FakeBackend(fail_depth=1, interrupt=True), {}
        with self.assertRaises(KeyboardInterrupt):
            front.process(box.plan, box.inputs, box.output, backend, record, lambda: None)
        self.assertEqual(backend.depth_calls, 1)
        self.assertEqual(record["images"][0]["status"], "error")
        self.assertEqual(record["images"][1]["status"], "not_attempted")
        self.assertEqual(len(record["edges"]), 120)
        self.assertEqual(record["images"][0]["error"]["type"], "KeyboardInterrupt")

    def test_match_dense_compact_inconsistency_and_bad_valid_depth_rejected(self):
        backend = FakeBackend()
        image = np.zeros((8, 10, 3), dtype=np.float32)
        feature = backend.features(image)
        raw = backend.match(feature, feature)
        front.validate_matches(raw, 2, 2, np)
        raw["matches0"][0, 1] = -1
        with self.assertRaisesRegex(ValueError, "disagreement"):
            front.validate_matches(raw, 2, 2, np)
        depth = backend.depth(image); depth["depth"][2, 2] = np.inf
        with self.assertRaisesRegex(ValueError, "model-valid"):
            front.validate_depth(depth, (8, 10), np)

    def test_intrinsics_read_interruption_retains_not_attempted_plan_rows(self):
        box = Fixture(self.root)
        backend, record = FakeBackend(), {}
        with mock.patch.object(Path, "read_text", side_effect=KeyboardInterrupt("artificial input interruption")):
            with self.assertRaises(KeyboardInterrupt):
                front.process(box.plan, box.inputs, box.output, backend, record, lambda: None)
        self.assertEqual(len(record["images"]), 51)
        self.assertEqual(len(record["edges"]), 120)
        self.assertTrue(all(row["status"] == "not_attempted" for row in record["images"] + record["edges"]))
        self.assertEqual(backend.depth_calls, 0)

    def test_native_tensor_shaped_empty_matches_have_canonical_cache_shapes(self):
        raw = dict(matches=np.empty((1, 0, 2), dtype=np.int64), scores=np.empty((1, 0), dtype=np.float32),
                   matches0=np.empty((1, 0), dtype=np.int64), matches1=np.empty((1, 0), dtype=np.int64),
                   matching_scores0=np.empty((1, 0), dtype=np.float32),
                   matching_scores1=np.empty((1, 0), dtype=np.float32), stop=np.asarray(0, dtype=np.int64))
        image = dict(xy=np.empty((0, 2), dtype=np.float32), depth_sample_raw=np.empty(0),
                     depth_sample_normalized=np.empty(0), depth_sample_valid=np.empty(0, dtype=np.bool_))
        arrays = front.edge_arrays(raw, image, image, np)
        self.assertEqual(arrays["matches"].shape, (1, 0, 2))
        self.assertEqual(arrays["scores"].shape, (1, 0))
        self.assertEqual(arrays["match_ids"].shape, (0, 2))
        self.assertEqual(arrays["match_scores"].shape, (0,))
        self.assertEqual(arrays["xy0"].shape, (0, 2))
        self.assertEqual(arrays["depth_valid0"].shape, (0,))

    def test_completed_qualification_and_outer_log_chain(self):
        args = SimpleNamespace(frontend_freeze="a" * 40,
                               qualification_receipt=self.root / "qualification/receipt.json",
                               qualification_supervision=self.root / "supervision/receipt.json",
                               installation=self.root / "installation", installation_supervision=self.root / "installation-supervision.json",
                               assets_dir=self.root / "assets", acquisition_receipt=self.root / "assets/ledger.json")
        args.qualification_receipt.parent.mkdir(); args.qualification_supervision.parent.mkdir()
        bindings = {"artificial": True}
        frozen = {"experiments/0022-frontend/qualify.py": front.q.sha(front.q.__file__),
                  "experiments/0022-frontend/supervise.py": "1" * 64}
        previous = dict(schema="evr0022.qualification.v1", status="completed", eligible=True,
                        expected_commit=args.frontend_freeze, bindings=bindings, frozen_files=frozen,
                        offline_attempts=[], source_sha256=front.q.sha(front.q.__file__), wall_seconds=.2)
        front.write_json_new(args.qualification_receipt, previous)
        out = args.qualification_supervision.parent
        (out / "stdout.log").write_text(json.dumps(dict(status="completed", eligible=True, wall_seconds=.2)) + "\n")
        (out / "stderr.log").write_text("")
        command = ["$PYTHON", "$REPOSITORY/experiments/0022-frontend/qualify.py", "--expected-commit", args.frontend_freeze]
        for flag, path in {"--output-dir": args.qualification_receipt.parent, "--installation": args.installation,
                           "--installation-supervision": args.installation_supervision, "--assets-dir": args.assets_dir,
                           "--acquisition-receipt": args.acquisition_receipt}.items():
            command += [flag, str(path)]
        outer = dict(schema="evr0022.supervision.v1", operation="qualification", status="completed", exit_code=0,
                     expected_commit=args.frontend_freeze, termination=None, remaining_descendant_pids=[], cleanup_errors=[],
                     source_sha256=frozen, wall_seconds=.3, hard_timeout_seconds=1800,
                     peak_group_rss_kib=100, group_rss_limit_kib=8 * 1024**2,
                     stdout_sha256=front.q.sha(out / "stdout.log"), stderr_sha256=front.q.sha(out / "stderr.log"), command=command)
        front.write_json_new(args.qualification_supervision, outer)
        checked = front.verify_qualification(args, bindings, frozen)
        self.assertEqual(checked["qualification_receipt_sha256"], front.q.sha(args.qualification_receipt))
        for change in ({"status": "failed"}, {"eligible": False}, {"expected_commit": "b" * 40}, {"wall_seconds": .25}):
            args.qualification_receipt.write_text(json.dumps(dict(previous, **change)))
            with self.assertRaises(ValueError):
                front.verify_qualification(args, bindings, frozen)


if __name__ == "__main__":
    unittest.main()
