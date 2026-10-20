"""Artificial orchestration/barrier tests; no historical cache, GT or model.

Only tiny manufactured NumPy arrays are serialized. Execution is delegated to
the parent's supervised qualification, not triggered by writing these tests.
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


c = module("_evr0028_frontend_contract_common", "common.py")
with patch.dict(sys.modules, {"common": c}):
    front = module("_evr0028_frontend_contract_frontend", "frontend.py")
COMMIT = "a" * 40


def manufactured_metadata():
    images, edges, layouts = [], [], []
    K = [[12., 0., 2.5], [0., 15., 1.5], [0., 0., 1.]]
    for scene_index, scene in enumerate(c.SCENES):
        anchor = "val/" + scene + "/seq0/frame_00000.jpg"
        paths = [anchor] + ["val/%s/seq1/frame_%05d.jpg" % (scene, j) for j in range(1, 17)]
        for j, name in enumerate(paths):
            metadata = {
                "authored_K_original": copy.deepcopy(K), "authored_K_resized": copy.deepcopy(K),
                "original_shape_hwc": [4, 6, 3], "resized_shape_hwc": [4, 6, 3],
                "original_encoded_sha256": "1" * 64, "resized_rgb_uint8_sha256": "2" * 64,
                "resized_rgb_float32_sha256": "3" * 64,
            }
            images.append(dict(image_index=17 * scene_index + j, filename=name, input=metadata))
        queries = [dict(source=anchor, target=paths[j], role="A_to_B", scene=scene)
                   for j in range(1, 17, 2)]
        layouts.append(dict(scene=scene, query_edges=queries))
        edges.extend(dict(source=anchor, target=paths[2], role="reference", scene=scene,
                          edge_index=40 * scene_index + j) for j in range(32))
        edges.extend(dict(edge, edge_index=40 * scene_index + 32 + j)
                     for j, edge in enumerate(queries))
    return dict(layouts=layouts, receipt=dict(images=images, edges=edges),
                file_sha256={"manufactured-cache": "f" * 64})


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.value, axis))


class FrontendContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evr0028-frontend-artificial-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_selection_27_preserves_receipt_order_excludes_24_references(self):
        verified = manufactured_metadata()
        images, edges = c.selected_layout(verified)
        self.assertEqual([r["image_index"] for r in images],
                         [17*s+j for s in range(3) for j in (0, 1, 3, 5, 7, 9, 11, 13, 15)])
        self.assertEqual([e["edge_index"] for e in edges],
                         [40*s+32+j for s in range(3) for j in range(8)])
        self.assertEqual((len(images), len(edges)), (27, 24))
        for mutation in ("reordered_edge", "missing_image", "scene_order"):
            bad = copy.deepcopy(verified)
            if mutation == "reordered_edge":
                bad["receipt"]["edges"][32], bad["receipt"]["edges"][33] = (
                    bad["receipt"]["edges"][33], bad["receipt"]["edges"][32])
            elif mutation == "missing_image":
                del bad["receipt"]["images"][0]
            else:
                bad["layouts"][0], bad["layouts"][1] = bad["layouts"][1], bad["layouts"][0]
            with self.assertRaises(ValueError):
                c.selected_layout(bad)

    def test_original_loader_returns_native_args_and_qualified_source_mapping(self):
        f, fa, frozen, outer = object(), object(), {"qualified": "f" * 64}, {"outer": True}
        receipt = {"plan_sha256": c.PLAN_SHA}
        old = SimpleNamespace(frontend_admission=Mock(return_value=(f, fa, receipt, outer, frozen)))
        verified = object()
        cache = SimpleNamespace(BINDING_FIELDS=("plan_sha256",), verify_cache=Mock(return_value=verified))
        def load(name, path):
            return old if path.name == "predict.py" else cache
        job = {"cache": str(self.root/"cache"), "frontend_outer": str(self.root/"outer.json"),
               "plan": str(self.root/"plan.json")}
        with patch.object(c, "module", side_effect=load), patch.object(c, "sha", return_value="b"*64):
            result = c.original_frontend(job)
        self.assertEqual(result, (old, f, fa, receipt, outer, frozen, cache, verified))
        called = old.frontend_admission.call_args.args[0]
        self.assertEqual(called.input_freeze, c.INPUT_FREEZE)
        self.assertEqual(called.expected_commit, c.INPUT_FREEZE)
        self.assertEqual(called.frontend_cache, self.root/"cache")
        self.assertFalse(hasattr(called, "evaluation"))
        self.assertEqual(cache.verify_cache.call_args.args[2],
                         {"plan_sha256": c.PLAN_SHA, "frontend_receipt_sha256": "b"*64})

    def test_loader_and_adapter_are_in_new_source_freeze(self):
        sources = c.source_hashes()  # Source bytes only; never scientific outputs.
        self.assertIn("experiments/0023-mapfree/predict.py", sources)
        self.assertIn("experiments/0023-mapfree/cache_inputs.py", sources)
        self.assertIn("experiments/0028-normal-information/frontend.py", sources)
        self.assertIn("experiments/0028-normal-information/point_postprocess.py", sources)

    def outer_fixture(self):
        repo = self.root/"repo"
        for relative in ("experiments/0027-affine-tangent/supervise.py",
                         "experiments/0025c-rs-noise/supervise.py"):
            path = repo/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# manufactured supervisor identity\n")
        output = self.root/"output"
        output.mkdir()
        result = output/"seal.json"
        result.write_text('{"manufactured": true}\n')
        operation = self.root/"operation"
        operation.mkdir()
        for name in ("stdout.log", "stderr.log"):
            (operation/name).write_text("manufactured " + name)
        job = self.root/"job.json"
        job.write_text('{"expected_commit": "'+COMMIT+'"}\n')
        command = ["$PYTHON", "-W", "error",
                   "$REPOSITORY/experiments/0028-normal-information/frontend.py",
                   "--expected-commit", COMMIT, "--job", "$PRIVATE/job.json",
                   "--output", "$PRIVATE/output"]
        receipt = dict(schema="evr0026.operation.v1", status="completed", exit_code=0,
                       termination=None, remaining_descendant_pids=[], cleanup_errors=[],
                       expected_commit=COMMIT, hard_timeout_seconds=1800, wall_seconds=1.,
                       group_rss_limit_kib=8388608, peak_group_rss_kib=1000,
                       thread_environment={k: "4" for k in c.THREAD_KEYS},
                       supervisor_sha256=c.sha(repo/"experiments/0027-affine-tangent/supervise.py"),
                       monitor_sha256=c.sha(repo/"experiments/0025c-rs-noise/supervise.py"),
                       log_sha256={n: c.sha(operation/n) for n in ("stdout.log", "stderr.log")},
                       output_sha256=c.sha(result), spec_sha256="c"*64, command=command)
        path = operation/"receipt.json"
        return repo, result, job, path, receipt

    def test_outer_failed_parent_log_tamper_and_result_tamper_reject(self):
        repo, result, job, path, receipt = self.outer_fixture()
        with patch.object(c, "ROOT", repo):
            c.write(path, receipt)
            c.verify_outer(path, result, COMMIT, "frontend.py", job)
            for change in ({"status": "failed"}, {"termination": "timeout"},
                           {"remaining_descendant_pids": [123]}, {"peak_group_rss_kib": 8388609}):
                c.write(path, dict(receipt, **change), replace=True)
                with self.assertRaises(ValueError):
                    c.verify_outer(path, result, COMMIT, "frontend.py", job)
            c.write(path, receipt, replace=True)
            (path.parent/"stderr.log").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "log changed"):
                c.verify_outer(path, result, COMMIT, "frontend.py", job)
            (path.parent/"stderr.log").write_text("manufactured stderr.log")
            result.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "result binding"):
                c.verify_outer(path, result, COMMIT, "frontend.py", job)

    def test_outer_script_position_duplicate_flags_and_job_linkage_reject(self):
        repo, result, job, path, receipt = self.outer_fixture()
        with patch.object(c, "ROOT", repo):
            for case in ("inert_script_argument", "duplicate", "job", "output", "commit"):
                bad = copy.deepcopy(receipt)
                command = bad["command"]
                if case == "inert_script_argument":
                    command[3], command[5] = "different-program.py", command[3]
                elif case == "duplicate":
                    command[6] = "--expected-commit"
                else:
                    flag = {"job": "--job", "output": "--output", "commit": "--expected-commit"}[case]
                    command[command.index(flag)+1] = "different"
                c.write(path, bad, replace=path.exists())
                with self.assertRaises(ValueError):
                    c.verify_outer(path, result, COMMIT, "frontend.py", job)

    def fake_run(self, *, control_error=False, changed_pixels=False, forward_error=False,
                 mutate_job=False, bad_source=False, startup_warning=False):
        verified = manufactured_metadata()
        by_name = {r["filename"]: r for r in verified["receipt"]["images"]}
        job = self.root/"job.json"
        c.write(job, dict(expected_commit=COMMIT, cache="manufactured-cache",
                          plan="manufactured-plan", frontend_outer="manufactured-outer"))
        args = SimpleNamespace(job=job, output=self.root/"new-output", expected_commit=COMMIT)
        calls = dict(load=0, decode=0, forward=0, postprocess=0, features=0, matches=0)
        fa = SimpleNamespace(inputs=self.root/"no-real-images")
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
                                     is_tensor=lambda x: isinstance(x, Tensor))
        class Backend:
            def __init__(self, *unused):
                calls["load"] += 1
                if startup_warning:
                    warnings.warn("manufactured import compatibility warning", UserWarning)
                self.np, self.torch, self.Image = np, fake_torch, object()
                self.moge = SimpleNamespace(forward=self.forward)
            def tensor(self, pixels):
                return Tensor(pixels.transpose(2, 0, 1).copy())
            def forward(self, pixels, num_tokens):
                calls["forward"] += 1
                if pixels.value.shape != (1, 3, 4, 6) or num_tokens != 1200:
                    raise AssertionError("raw forward input contract")
                if forward_error:
                    raise RuntimeError("manufactured forward failure")
                return {"points": Tensor(np.full((1, 4, 6, 3), 2., np.float32)),
                        "normal": Tensor(np.full((1, 4, 6, 3), -1., np.float32)),
                        "mask": Tensor(np.ones((1, 4, 6), np.float32)),
                        "metric_scale": Tensor(np.ones(1, np.float32))}
            def features(self, *unused):
                calls["features"] += 1
                raise AssertionError("features are forbidden")
            def match(self, *unused):
                calls["matches"] += 1
                raise AssertionError("matches are forbidden")
        def prepare(path, calibration, unused_np, unused_Image):
            calls["decode"] += 1
            name = path.relative_to(fa.inputs).as_posix()
            meta = copy.deepcopy(by_name[name]["input"])
            if changed_pixels:
                meta["resized_rgb_float32_sha256"] = "9"*64
            K = np.asarray(meta["authored_K_resized"], dtype=np.float64)
            return np.zeros((4, 6, 3), np.float32), {"K_original": K.copy(), "K_resized": K}, meta
        @contextlib.contextmanager
        def capture(record):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    yield
                finally:
                    record['warnings'].extend(dict(category=type(w.message).__name__, message=str(w.message))
                                              for w in caught)
        qualified = SimpleNamespace(
            qualification_args=lambda value: value,
            q=SimpleNamespace(verify_files=lambda *a: ({}, None, {}, {"manufactured_model": True}),
                              capture_warnings=capture),
            Backend=Backend, prepare_image=prepare, safe_input=lambda root, name: root/name)
        def postprocess(raw, K):
            calls["postprocess"] += 1
            # The raw cache must already be sealed against this mutation.
            raw["points"].value[:] = 99.
            if mutate_job and calls["postprocess"] == 1:
                with job.open("a") as stream:
                    stream.write(" ")
            result = {}
            for mode in ("A", "B", "C"):
                result[mode] = dict(
                    status="completed", mode=mode, depth=Tensor(np.full((4, 6), 3., np.float32)),
                    normal=np.full((4, 6, 3), -1., np.float32), mask=np.ones((4, 6), bool),
                    original_mask=np.ones((4, 6), bool), intrinsics=K.astype(np.float32),
                    K_pixels=K.astype(np.float32), recovery={"shift": 1.},
                    warnings=[], validity={"depth_all_finite": True})
            if control_error:
                result["C"].update(status="helper_error",
                    error={"type": "RuntimeError", "message": "manufactured C failure"},
                    warnings=[{"category": "RuntimeWarning", "message": "manufactured C warning"}])
            return result
        hashes = {"manufactured_source.py": "4"*64}
        original_read = c.read
        def restricted_read(path):
            if Path(path).resolve() != job.resolve():
                raise AssertionError("unexpected JSON/GT access: " + str(path))
            return original_read(path)
        def source(commit):
            if bad_source:
                raise ValueError("unfrozen manufactured source")
            return hashes
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(c, "verify_source", side_effect=source))
        stack.enter_context(patch.object(c, "source_hashes", return_value=hashes))
        stack.enter_context(patch.object(c, "read", side_effect=restricted_read))
        stack.enter_context(patch.object(c, "original_frontend", return_value=(
            object(), qualified, fa, verified["receipt"], {"outer": True},
            {"qualified": True}, object(), verified)))
        stack.enter_context(patch.dict(sys.modules, {"point_postprocess": SimpleNamespace(postprocess_all=postprocess)}))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack, args, calls

    def test_complete_27_raw_forward_seal_serialization_and_no_gt(self):
        stack, args, calls = self.fake_run()
        with stack:
            self.assertEqual(front.run(args), 0)
        receipt, seal = c.read(args.output/"receipt.json"), c.read(args.output/"seal.json")
        self.assertEqual(calls, dict(load=1, decode=27, forward=27, postprocess=27, features=0, matches=0))
        self.assertEqual(receipt["raw_forwards_completed"], 27)
        self.assertEqual((seal["image_count"], seal["query_count"], seal["raw_forward_count"]), (27, 24, 27))
        self.assertFalse(receipt["ground_truth_access"] or seal["ground_truth_access"])
        self.assertEqual(len(seal["files"]), 82)  # 27*(raw,processed,JSON)+receipt
        for name, digest in seal["files"].items():
            self.assertEqual(c.sha(args.output/name), digest)
        with np.load(args.output/"raw-000.npz", allow_pickle=False) as saved:
            self.assertTrue(np.all(saved["raw_points"] == 2.))
            self.assertEqual(set(saved.files), {"raw_points", "raw_normal", "raw_mask", "raw_metric_scale"})
        row = receipt["images"][0]
        self.assertEqual(set(row["controls"]), {"A", "B", "C"})
        self.assertNotIn("depth", row["controls"]["A"])  # arrays are not huge JSON fields
        self.assertEqual(row["processed"]["arrays"]["A_depth"]["shape"], [4, 6])

    def test_startup_warning_is_retained_before_forward(self):
        stack, args, calls = self.fake_run(startup_warning=True)
        with stack, warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(front.run(args), 0)
        receipt = c.read(args.output/'receipt.json')
        self.assertEqual(receipt['warnings'], [dict(category='UserWarning',
                         message='manufactured import compatibility warning')])
        self.assertEqual(calls['forward'], 27)

    def test_failed_camera_control_keeps_raw_warning_and_partial_arrays(self):
        stack, args, calls = self.fake_run(control_error=True)
        with stack, self.assertRaisesRegex(ValueError, "camera-control failure"):
            front.run(args)
        receipt = c.read(args.output/"receipt.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(calls["forward"], 1)
        self.assertFalse((args.output/"seal.json").exists())
        row = receipt["images"][0]
        self.assertTrue((args.output/row["raw"]["filename"]).exists())
        self.assertEqual(row["controls"]["C"]["error"]["message"], "manufactured C failure")
        self.assertEqual(row["controls"]["C"]["warnings"][0]["message"], "manufactured C warning")
        self.assertTrue((args.output/row["partial_processed"]["filename"]).exists())

    def test_changed_rgb_blocks_forward_and_source_failure_blocks_backend(self):
        stack, args, calls = self.fake_run(changed_pixels=True)
        with stack, self.assertRaisesRegex(ValueError, "preprocessing changed"):
            front.run(args)
        self.assertEqual(calls["forward"], 0)
        self.assertEqual(c.read(args.output/"receipt.json")["status"], "failed")
        # Separate fresh fake operation; source verification precedes Backend.
        other = self.root/"second"
        other.mkdir()
        self.root = other
        stack, args, calls = self.fake_run(bad_source=True)
        with stack, self.assertRaisesRegex(ValueError, "unfrozen"):
            front.run(args)
        self.assertEqual(calls["load"], 0)
        self.assertFalse(args.output.exists())

    def test_forward_failure_keeps_attempted_count_and_no_seal(self):
        stack, args, calls = self.fake_run(forward_error=True)
        with stack, self.assertRaisesRegex(RuntimeError, "manufactured forward failure"):
            front.run(args)
        receipt = c.read(args.output/"receipt.json")
        self.assertEqual((receipt["raw_forwards_attempted"], receipt["raw_forwards_completed"]), (1, 0))
        self.assertEqual(receipt["images"][0]["stage"], "raw_forward")
        self.assertFalse((args.output/"seal.json").exists())
        self.assertEqual(calls["postprocess"], 0)

    def test_job_change_cannot_receive_completed_seal(self):
        stack, args, calls = self.fake_run(mutate_job=True)
        with stack, self.assertRaisesRegex(ValueError, "job changed"):
            front.run(args)
        receipt = c.read(args.output/"receipt.json")
        self.assertEqual(calls["forward"], 27)
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse((args.output/"seal.json").exists())


if __name__ == "__main__":
    unittest.main()
