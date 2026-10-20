"""Artificial orchestration tests: no learned model, real image, cache, or GT."""
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


probe = module("_evr0024_run_artificial", "run.py")
core = module("_evr0024_core_artificial_runner", "math_core.py")


def artificial_plan():
    scenes = []
    for j in range(3):
        scene = "manufactured-%d" % j
        anchor = scene + "/seq0/frame_00000.jpg"
        queries = [scene + "/seq1/frame_%05d.jpg" % (2*i+1) for i in range(8)]
        scenes.append(dict(scene=scene, anchor=anchor, query_images=queries,
                           query_edges=[dict(source=anchor, target=q) for q in queries],
                           intrinsics_member=dict(filename=scene + "/intrinsics.txt"),
                           reference_images=object(), ground_truth=object()))
    return dict(scenes=scenes)


def analysis(candidate=.8, baseline=1., usable=True):
    return dict(usable_offset=usable, directions={
        direction: dict(candidate_error=candidate, zero_offset_error=baseline)
        for direction in core.DIRECTIONS})


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeImage:
    format = "JPEG"

    def __init__(self, raw):
        self.raw, self.size = raw, (raw.shape[1], raw.shape[0])

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    def getexif(self):
        return {}

    def convert(self, mode):
        if mode != "RGB":
            raise AssertionError("unexpected image conversion")
        return self.raw


def save_arrays(path, arrays, unused_numpy):
    # Only tiny manufactured arrays reach this helper.
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    return dict(filename=path.name, sha256=probe.sha(path), keys=sorted(arrays))


class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evr0024-run-fake-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def original_outer(self):
        folder = self.root / "old-supervision"
        folder.mkdir(exist_ok=True)
        for name in ("stdout", "stderr"):
            (folder / (name + ".log")).write_text("manufactured " + name)
        values = {name: "$PRIVATE/manufactured/" + name for name in probe.PATH_FLAGS}
        values.update(expected_commit=probe.INPUT_FREEZE, frontend_freeze=probe.MODEL_FREEZE,
                      plan_sha256=probe.PLAN_SHA, input_receipt_sha256=probe.INPUT_SHA)
        command = ["$PYTHON", "$REPOSITORY/experiments/0023-mapfree/frontend.py"]
        for name in probe.PATH_FLAGS + probe.TEXT_FLAGS:
            command.extend(["--" + name.replace("_", "-"), values[name]])
        outer = dict(status="completed", exit_code=0, expected_commit=probe.INPUT_FREEZE,
                     remaining_descendant_pids=[], cleanup_errors=[], command=command,
                     stdout_sha256=probe.sha(folder / "stdout.log"),
                     stderr_sha256=probe.sha(folder / "stderr.log"))
        path = folder / "receipt.json"
        return path, outer

    def test_metadata_population_excludes_references_and_keeps_order(self):
        plan = artificial_plan()
        rows = probe.selected_images(plan)
        expected = [(s["scene"], n) for s in plan["scenes"]
                    for n in [s["anchor"], *s["query_images"]]]
        self.assertEqual([(r["scene"], r["image_id"]) for r in rows], expected)
        self.assertEqual(len(rows), 27)
        for mutation in ("reference_target", "duplicate", "missing_place"):
            bad = copy.deepcopy(plan)
            if mutation == "reference_target":
                bad["scenes"][0]["query_edges"][0]["target"] = "reference.jpg"
            elif mutation == "duplicate":
                bad["scenes"][0]["query_images"][1] = bad["scenes"][0]["query_images"][0]
            else:
                bad["scenes"].pop()
            with self.assertRaises(ValueError):
                probe.selected_images(bad)

    def test_original_command_hash_freeze_flags_and_log_binding(self):
        path, outer = self.original_outer()
        path.write_text(json.dumps(outer))
        with patch.object(probe, "FRONTEND_OUTER_SHA", probe.sha(path)):
            args = probe.original_args(path)
            self.assertEqual(args.frontend_freeze, probe.MODEL_FREEZE)
            self.assertEqual(args.inputs, probe.ROOT.parent / "manufactured/inputs")
            (path.parent / "stdout.log").write_text("changed log")
            with self.assertRaisesRegex(ValueError, "log hash"):
                probe.original_args(path)
        path, outer = self.original_outer()
        for label in ("unknown", "duplicate", "freeze", "hash"):
            altered = copy.deepcopy(outer)
            if label == "unknown":
                altered["command"].extend(["--read-gt", "yes"])
            elif label == "duplicate":
                altered["command"].extend(["--inputs", "$PRIVATE/elsewhere"])
            elif label == "freeze":
                altered["command"][altered["command"].index("--frontend-freeze")+1] = "b"*40
            path.write_text(json.dumps(altered))
            expected = "0"*64 if label == "hash" else probe.sha(path)
            with patch.object(probe, "FRONTEND_OUTER_SHA", expected):
                with self.assertRaises(ValueError):
                    probe.original_args(path)

    def test_returned_intrinsics_use_half_pixel_center_and_relative_tolerance(self):
        size = (512, 384)
        K = np.array([[600., 0, 255.5], [0, 600., 191.5], [0, 0, 1.]])
        actual = np.array([[600/512, 0, .5], [0, 600/384, .5], [0, 0, 1.]], dtype=np.float32)
        actual[0, 0] *= np.float32(1.+8e-6)
        probe.check_intrinsics(actual, K, size, np)
        for wrong in ("unshifted_center", "focal", "zero_entry", "nonfinite"):
            bad = actual.copy()
            if wrong == "unshifted_center":
                bad[0, 2] = 255.5/512
            elif wrong == "focal":
                bad[1, 1] *= np.float32(1.+2e-5)
            elif wrong == "zero_entry":
                bad[0, 1] = 1e-12
            else:
                bad[0, 0] = np.nan
            with self.assertRaises(ValueError):
                probe.check_intrinsics(bad, K, size, np)

    def test_remap_checks_float64_and_float32_before_image_operation(self):
        cv = SimpleNamespace(remap=Mock(), INTER_LINEAR=1, BORDER_CONSTANT=0)
        raw = np.zeros((10, 12, 3), np.uint8)
        for xy in (np.array([[[2-1e-10, 2.]]]), np.array([[[9+1e-10, 2.]]]),
                   np.array([[[np.nan, 2.]]])):
            with self.assertRaises(ValueError):
                probe.remap_rgb(raw, xy, cv, np)
        # No giant allocation: this declared integer upper edge rounds OUTWARD in float32.
        giant_metadata_only = SimpleNamespace(shape=(10, 2**25+6, 3))
        with self.assertRaisesRegex(ValueError, "float32"):
            probe.remap_rgb(giant_metadata_only, np.array([[[float(2**25+3), 2.]]]), cv, np)
        cv.remap.assert_not_called()
        cv.remap.return_value = np.ones((1, 2, 3), np.uint8)*255
        normalized, encoded = probe.remap_rgb(raw, np.array([[[2., 2.], [9., 7.]]]), cv, np)
        np.testing.assert_array_equal(normalized, np.ones((1, 2, 3), np.float32))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(cv.remap.call_args.args[1].dtype, np.float32)

    def image_fixture(self, fail_call=None):
        raw = np.arange(16*20*3, dtype=np.uint16).reshape(16, 20, 3).astype(np.uint8)
        encoded = self.root / "manufactured-image.bin"
        encoded.write_bytes(b"artificial image token; fake decoder owns all pixels")
        output = self.root / "per-image"
        output.mkdir()
        calls = []

        def infer(image, **kwargs):
            calls.append(kwargs)
            if len(calls) == fail_call:
                raise RuntimeError("manufactured forward failure")
            h, w = image.shape[:2]
            f = w/(2*math.tan(math.radians(kwargs["fov_x"])/2))
            return {"depth": FakeTensor(np.ones((h, w), np.float32)*len(calls)),
                    "mask": FakeTensor(np.ones((h, w), bool)),
                    "intrinsics": FakeTensor(np.array([[f/w, 0., .5], [0., f/h, .5], [0., 0., 1.]]))}

        backend = SimpleNamespace(np=np, Image=SimpleNamespace(open=Mock(return_value=FakeImage(raw))),
            torch=SimpleNamespace(inference_mode=contextlib.nullcontext), tensor=lambda x: x,
            moge=SimpleNamespace(infer=infer))  # There is intentionally no SP or LG interface.
        q = SimpleNamespace(MOGE_OPTIONS=dict(num_tokens=1200, use_fp16=False,
            force_projection=True, apply_mask=False), save_arrays=save_arrays)
        frontend = SimpleNamespace(q=q, safe_input=lambda root, name: encoded,
            raw_sha=lambda a: hashlib.sha256(a.tobytes()).hexdigest(),
            validate_depth=Mock())
        cv = SimpleNamespace(INTER_LINEAR=1, BORDER_CONSTANT=0,
            remap=Mock(side_effect=lambda raw, x, y, *a, **k: np.zeros((*x.shape, 3), np.uint8)))
        row = dict(index=0, scene="manufactured", image_id="image.jpg")
        calibration = dict(width=20, height=16, fx=20., fy=20., cx=9.5, cy=7.5)
        record = dict(depth_forwards_attempted=0, depth_forwards_completed=0, feature_forwards=0, matcher_forwards=0)
        numerical = SimpleNamespace(geometry=core.geometry, remap_coordinates=core.remap_coordinates,
                                    analyze_image=Mock(return_value=analysis()))
        return row, calibration, output, backend, frontend, cv, record, calls, numerical

    def test_image_has_exactly_five_depth_forwards_and_canonical_options(self):
        row, calibration, output, backend, frontend, cv, record, calls, numerical = self.image_fixture()
        with patch.dict(sys.modules, {"cv2": cv}):
            result = probe.process_image(row, calibration, SimpleNamespace(inputs=self.root), output,
                backend, numerical, frontend, record, lambda: None)
        self.assertEqual((record["depth_forwards_attempted"], record["depth_forwards_completed"]), (5, 5))
        self.assertEqual((record["feature_forwards"], record["matcher_forwards"]), (0, 0))
        self.assertEqual([v["name"] for v in result["views"]], list(probe.VIEW_NAMES))
        self.assertEqual(len(calls), 5)
        self.assertEqual(len({v["fov_x"] for v in calls}), 1)
        self.assertTrue(all({k: v for k, v in call.items() if k != "fov_x"} == frontend.q.MOGE_OPTIONS for call in calls))
        with np.load(output / "image-000.npz", allow_pickle=False) as data:
            self.assertEqual(len(data.files), 15)
            np.testing.assert_array_equal(data["view_4_depth"], np.ones((16, 20))*5)
        self.assertEqual(json.loads((output / "image-000.json").read_text())["status"], "completed")

    def test_failed_third_forward_preserves_first_two_raw_predictions(self):
        row, calibration, output, backend, frontend, cv, record, calls, numerical = self.image_fixture(3)
        with patch.dict(sys.modules, {"cv2": cv}), self.assertRaisesRegex(RuntimeError, "manufactured forward"):
            probe.process_image(row, calibration, SimpleNamespace(inputs=self.root), output,
                backend, numerical, frontend, record, lambda: None)
        saved = json.loads((output / "image-000.json").read_text())
        self.assertEqual((record["depth_forwards_attempted"], record["depth_forwards_completed"]), (3, 2))
        self.assertEqual((saved["status"], saved["failed_phase"]), ("failed", "yaw_negative"))
        with np.load(output / "image-000-partial.npz", allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), {"view_%d_%s" % (i, name)
                for i in (0, 1) for name in ("depth", "mask", "intrinsics")})
        numerical.analyze_image.assert_not_called()
        self.assertFalse((output / "image-000.npz").exists())

    def cohort_fixture(self):
        plan = artificial_plan()
        for scene in plan["scenes"]:
            path = self.root / "inputs" / scene["intrinsics_member"]["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("manufactured selected intrinsics")
        outer = self.root / "original-outer.json"
        outer.write_text("manufactured outer receipt")
        old = SimpleNamespace(inputs=self.root / "inputs")
        q = SimpleNamespace(MOGE_OPTIONS={"num_tokens": 1200},
            frozen_bindings=Mock(return_value={}),
            verify_files=Mock(return_value=({}, {}, [], {"models": "bound"})),
            capture_warnings=lambda record: contextlib.nullcontext())
        frontend = SimpleNamespace(q=q, qualification_args=lambda a: SimpleNamespace(),
            verify_qualification=lambda *a: {},
            acquisition=SimpleNamespace(verify_completed_chain=Mock(return_value={"acquisition": "bound"})),
            verify_inputs=Mock(return_value=(plan, {"inputs": "bound"})),
            safe_input=lambda root, name: root / name,
            parse_intrinsics=lambda text, names: {name: {} for name in names}, Backend=Mock(return_value=object()))
        args = SimpleNamespace(expected_commit="a"*40, original_frontend_supervision=outer,
                               output_dir=self.root / "cohort")
        return args, old, frontend, plan

    def fake_process(self, row, calibration, old, output, backend, numerical, frontend, record, save,
                     candidate=.8, baseline=1.):
        result = dict(row, status="completed", analysis=analysis(candidate, baseline))
        save_arrays(output / ("image-%03d.npz" % row["index"]), {"manufactured": np.array([row["index"]])}, np)
        probe.write_new(output / ("image-%03d.json" % row["index"]), result)
        record["depth_forwards_attempted"] += 5
        record["depth_forwards_completed"] += 5
        return result

    @contextlib.contextmanager
    def cohort_patches(self, args, old, frontend, numerical):
        with patch.object(probe, "load_module", side_effect=[frontend, numerical]), \
                patch.object(probe, "verified_sources", return_value=({}, {})), \
                patch.object(probe, "original_args", return_value=old), \
                patch.object(probe, "FRONTEND_OUTER_SHA", probe.sha(args.original_frontend_supervision)):
            yield

    def test_source_or_input_fault_prevents_backend_and_retains_ineligible_receipt(self):
        for fault in ("source", "input"):
            args, old, frontend, _ = self.cohort_fixture()
            args.output_dir = self.root / ("fault-" + fault)
            if fault == "input":
                frontend.verify_inputs.side_effect = ValueError("manufactured input hash fault")
            with self.cohort_patches(args, old, frontend, object()):
                if fault == "source":
                    with patch.object(probe, "verified_sources", side_effect=ValueError("manufactured source fault")):
                        with self.assertRaises(ValueError):
                            probe.run(args)
                else:
                    with self.assertRaises(ValueError):
                        probe.run(args)
            frontend.Backend.assert_not_called()
            receipt = json.loads((args.output_dir / "receipt.json").read_text())
            self.assertEqual((receipt["status"], receipt["eligible"], receipt["depth_forwards_attempted"]),
                             ("failed", False, 0))

    def test_all_27_raw_caches_and_analyses_are_sealed_before_aggregate(self):
        args, old, frontend, _ = self.cohort_fixture()
        def summarize(rows):
            seal = json.loads((args.output_dir / "seal.json").read_text())
            self.assertEqual((len(rows), len(seal["outputs"])), (27, 54))
            self.assertEqual({p.suffix for p in args.output_dir.glob("image-*")}, {".npz", ".json"})
            for item in seal["outputs"]:
                self.assertEqual(probe.sha(args.output_dir / item["filename"]), item["sha256"])
            return core.summarize(rows)
        numerical = SimpleNamespace(summarize=Mock(side_effect=summarize))
        with self.cohort_patches(args, old, frontend, numerical), \
                patch.object(probe, "process_image", side_effect=self.fake_process) as process:
            result = probe.run(args)
        self.assertEqual(process.call_count, 27)
        self.assertEqual(result["depth_forwards_completed"], 135)
        self.assertEqual((result["status"], result["eligible"]), ("completed", True))
        self.assertEqual(numerical.summarize.call_count, 1)
        diagnostic = json.loads((args.output_dir / "diagnostic.json").read_text())
        self.assertTrue(diagnostic["summary"]["pass"])
        self.assertEqual(diagnostic["prediction_seal_sha256"], probe.sha(args.output_dir / "seal.json"))

    def test_sealed_raw_cache_tamper_prevents_aggregate(self):
        args, old, frontend, _ = self.cohort_fixture()
        numerical = SimpleNamespace(summarize=Mock())
        original = probe.seal_outputs
        def corrupted(output, rows, sources):
            result = original(output, rows, sources)
            with (output / "image-013.npz").open("ab") as stream:
                stream.write(b"modified after sealing")
            return result
        with self.cohort_patches(args, old, frontend, numerical), \
                patch.object(probe, "process_image", side_effect=self.fake_process), \
                patch.object(probe, "seal_outputs", side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, "sealed output changed"):
                probe.run(args)
        numerical.summarize.assert_not_called()
        receipt = json.loads((args.output_dir / "receipt.json").read_text())
        self.assertFalse(receipt["eligible"])
        self.assertTrue((args.output_dir / "seal.json").exists())
        self.assertFalse((args.output_dir / "diagnostic.json").exists())

    def test_missing_or_failed_image_cannot_be_read_as_full_cohort(self):
        output = self.root / "incomplete"
        output.mkdir()
        rows = [dict(index=i, scene="manufactured-%d" % (i//9), image_id="image-%d" % i) for i in range(27)]
        for row in rows:
            self.fake_process(row, {}, None, output, None, None, None,
                              dict(depth_forwards_attempted=0, depth_forwards_completed=0), lambda: None)
        failed = output / "image-005.json"
        content = json.loads(failed.read_text()); content["status"] = "failed"
        failed.write_text(json.dumps(content))
        seal = probe.seal_outputs(output, rows, {})
        with self.assertRaisesRegex(ValueError, "incomplete sealed"):
            probe.read_sealed_analyses(output, seal)
        (output / "image-007.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            probe.read_sealed_analyses(output, seal)

    def test_negative_baseline_error_cannot_yield_eligible_run(self):
        args, old, frontend, _ = self.cohort_fixture()
        numerical = SimpleNamespace(summarize=core.summarize)
        def invalid(*positional):
            return self.fake_process(*positional, candidate=.1, baseline=-1.)
        with self.cohort_patches(args, old, frontend, numerical), \
                patch.object(probe, "process_image", side_effect=invalid):
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                probe.run(args)
        receipt = json.loads((args.output_dir / "receipt.json").read_text())
        self.assertEqual((receipt["eligible"], receipt["status"]), (False, "failed"))
        self.assertFalse((args.output_dir / "diagnostic.json").exists())
        rows = [dict(scene="s%d" % (i//9), image_id=str(i), analysis=analysis(1.1, 1.)) for i in range(27)]
        self.assertFalse(core.summarize(rows)["pass"])


if __name__ == "__main__":
    unittest.main()
