"""Invented-array predictor contracts; no models, native APIs or dataset reads.

The legacy sample_depth function is extracted verbatim from its source AST:
this tests the actual pure helper without importing its application loaders.
The cohort test replaces every data/provider/backend boundary with fakes.
"""
import ast
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

import pose_controls as pose
import predict as predictor


def legacy_sampler():
    path = predictor.c.ROOT / "experiments/0023-mapfree/frontend.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "sample_depth"]
    if len(nodes) != 1:
        raise AssertionError("legacy pure sampler is not uniquely defined")
    namespace = {"math": math, "require": predictor.c.require}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["sample_depth"]


class FakeOffset:
    def __init__(self):
        self.fits = {
            method: dict(status="completed", admissible=True, b=b)
            for method, b in (("LS", .25), ("IRLS", .5), ("ANG", .1))}

    def prepare(self, depth, normal, mask, K):
        return dict(median_raw=8., normalized=depth/8.)

    def describe(self, prepared):
        return dict(median_raw=prepared["median_raw"])

    def solve(self, prepared, method):
        return copy.deepcopy(self.fits[method])

    def corrected_depth(self, prepared, fit):
        return prepared["normalized"]+fit["b"] if fit["admissible"] else None


def image_fixture():
    K = np.array([[5., 0., 2.5], [0., 5., 2.5], [0., 0., 1.]])
    arrays = {"K_resized": K}
    for mode, depth in (("A", 2.), ("B", 4.), ("C", 8.)):
        arrays[mode+"_depth"] = np.full((6, 6), depth)
        arrays[mode+"_mask"] = np.ones((6, 6), bool)
    arrays["C_normal"] = np.tile([0., 0., 1.], (6, 6, 1))
    # All four normals are valid, but their interpolant cancels at (.5,.5).
    arrays["C_normal"][0:2, 1, 2] = -1.
    original = {"xy": np.array([[.5, .5], [2.5, .5], [4.5, 4.5]])}
    return dict(image_index=0, filename="manufactured.png"), arrays, original


class PredictorContractTests(unittest.TestCase):
    def setUp(self):
        self.sampler = legacy_sampler()
        self.front = SimpleNamespace(sample_depth=self.sampler)
        self.torch = SimpleNamespace(is_tensor=lambda value: False)
        self.post = SimpleNamespace(refine_with_normal=lambda d, n, K, provider:
                                    dict(status="completed", depth=d.copy()))
        self.offset = FakeOffset()

    def prepare(self, *, sampler_pose=pose):
        row, arrays, original = image_fixture()
        return predictor.prepare_image(row, arrays, original, self.front, self.post,
                                       object(), self.offset, sampler_pose, np, self.torch)

    def test_legacy_nan_depth_sampling_is_warning_free_and_preserves_queries(self):
        d = np.full((4, 4), np.nan)
        xy = np.array([[.5, .5], [1., 1.], [3., 3.]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            all_bad = self.sampler(d, np.ones((4, 4), bool), xy, np)
            d[:] = 2.
            d[0, 0] = np.nan
            mixed = self.sampler(d, np.ones((4, 4), bool), xy, np)
        self.assertEqual(all_bad["depth_sample_valid"].tolist(), [False]*3)
        self.assertTrue(np.isnan(all_bad["depth_sample_raw"]).all())
        self.assertEqual(mixed["depth_sample_valid"].tolist(), [False, True, False])
        self.assertEqual(mixed["depth_sample_raw"][1], 2.)
        self.assertEqual(len(mixed["depth_sample_raw"]), len(xy))

    def test_prepare_keeps_C_units_and_actual_normal_sampler_contract(self):
        sampled, meta = self.prepare()
        for mode in ("A", "B", "C"):
            np.testing.assert_array_equal(sampled["depth_"+mode], np.ones(3))
        for mode, expected in (("LS", 1.25), ("IRLS", 1.5), ("ANG", 1.1)):
            np.testing.assert_allclose(sampled["depth_"+mode], np.full(3, expected))
        # Using depth_sample_normalized would incorrectly produce all ones.
        self.assertEqual(sampled["original_valid"].tolist(), [True]*3)
        self.assertEqual(sampled["normal_valid"].tolist(), [False, True, True])
        self.assertTrue(np.isnan(sampled["normal"][0]).all())
        self.assertEqual(meta["normal_sampling"]["samples"][0]["reason"],
                         "interpolated_normal_collapse")
        self.assertEqual(meta["medians"], dict(A=2., B=4., C=8.))
        self.assertEqual(meta["corrections"]["LS"]["b"], .25)
        json.dumps(meta, allow_nan=False)

    def test_unavailable_correction_keeps_original_eligibility_and_NaN_samples(self):
        self.offset.fits["LS"] = dict(status="unavailable", admissible=False, b=None)
        self.offset.fits["ANG"] = dict(status="uninformative", admissible=True, b=0.)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sampled, meta = self.prepare()
        self.assertTrue(np.isnan(sampled["depth_LS"]).all())
        self.assertEqual(sampled["valid_LS"].tolist(), [False]*3)
        self.assertEqual(sampled["original_valid"].tolist(), [True]*3)
        np.testing.assert_array_equal(sampled["depth_ANG"], sampled["depth_C"])
        self.assertEqual(meta["corrections"]["ANG"]["status"], "uninformative")

    def test_FULL_numeric_failure_disables_comparator_without_losing_other_samples(self):
        self.post.refine_with_normal = lambda *a, **k: dict(
            status="numerical_failure", reason="manufactured invalid output")
        sampled, meta = self.prepare()
        self.assertFalse(meta["comparator_valid"])
        self.assertTrue(np.isnan(sampled["depth_FULL"]).all())
        self.assertEqual(sampled["valid_FULL"].tolist(), [False]*3)
        self.assertEqual(sampled["original_valid"].tolist(), [True]*3)
        self.assertEqual(meta["samples"]["FULL"], dict(valid_count=0, count=3))

    def test_helper_error_returns_preservable_failure_before_sampling(self):
        self.post.refine_with_normal = lambda *a, **k: dict(
            status="helper_error", error_type="AttributeError", error="manufactured helper fault")
        with patch.object(self.front, "sample_depth", side_effect=AssertionError("must not sample")):
            sampled, meta = self.prepare()
        self.assertIsNone(sampled)
        self.assertEqual(meta["full"]["error_type"], "AttributeError")
        self.assertEqual(set(meta["corrections"]), {"LS", "IRLS", "ANG"})
        json.dumps(meta, allow_nan=False)

    def test_wrong_normal_sampler_population_is_technical_failure(self):
        wrong = SimpleNamespace(sample_normal=lambda *a: dict(
            status="completed", query_count=3, samples=[]))
        with self.assertRaisesRegex(ValueError, "normal sampler schema"):
            self.prepare(sampler_pose=wrong)

    def test_fake_full_cohort_preserves_pair_units_selection_and_comparator_failure(self):
        images, edges = [], []
        for scene in predictor.c.SCENES:
            names = [scene+"/A0"]+[scene+"/B"+str(i) for i in range(1, 9)]
            image_start, edge_start = len(images), len(edges)
            images.extend(dict(image_index=image_start+i, filename=name)
                          for i, name in enumerate(names))
            edges.extend(dict(scene=scene, source=names[0], target=name,
                              edge_index=edge_start+i) for i, name in enumerate(names[1:]))
        K = np.array([[500., 0., 319.5], [0., 500., 239.5], [0., 0., 1.]])
        xy = np.array([[10.+i, 20.+2*i] for i in range(6)])
        ids = np.column_stack((np.arange(6), np.arange(6)))
        selected = np.array([4, 1, 2, 0])
        received, captured_arrays = [], {}
        backend, refiner = object(), object()
        sources, commit = {"manufactured.py": "1"*64}, "2"*40

        def read_arrays(root, row, fields):
            if "match_ids" in fields:
                return dict(match_ids=ids.copy(), match_scores=np.ones(6),
                            xy0=xy.copy(), xy1=xy.copy())
            return dict(xy=xy.copy(), K_original=K.copy(), K_resized=K.copy(),
                        image_size_original=np.array([640, 480]),
                        image_size_resized=np.array([640, 480]))

        def fake_prepare(row, arrays, original, *unused):
            sampled = dict(xy=xy.copy(), K=K.copy(), original_valid=np.ones(6, bool),
                           normal=np.tile([0., 0., 1.], (6, 1)), normal_valid=np.ones(6, bool))
            sampled["original_valid"][2] = False
            for mode in ("A", "B", "C", "LS", "IRLS", "ANG", "FULL"):
                sampled["depth_"+mode] = np.arange(6, dtype=float)+1.
            sampled["depth_LS"] += .25
            corrections = {
                mode: dict(status="completed", admissible=True, b=.25 if mode == "LS" else 0.)
                for mode in ("LS", "IRLS", "ANG")}
            return sampled, dict(image_index=row["image_index"], filename=row["filename"],
                                 corrections=corrections, comparator_valid=True, status="completed")

        def fake_process(pair, passed_backend, passed_refiner, emit):
            self.assertIs(passed_backend, backend)
            self.assertIs(passed_refiner, refiner)
            received.append(copy.deepcopy(pair))
            np.testing.assert_array_equal(pair["xy1"], xy[selected])
            np.testing.assert_array_equal(pair["depths1"]["LS"]-pair["depths1"]["C"],
                                          np.full(4, .25))
            self.assertEqual(pair["correction1"]["LS"]["b"], .25)
            self.assertEqual(pair["original_valid1"].tolist(), [True, True, False, True])
            self.assertEqual((pair["width1"], pair["height1"]), (640, 480))
            emit(dict(status="started", phase="manufactured_intent"))
            arms = {
                arm: dict(status="fallback" if arm == "ANG" else "completed",
                          R=np.eye(3).tolist(), t=[1., 0., 0.], detail={})
                for arm in predictor.c.ARMS}
            return dict(status="completed", arms=arms,
                        comparator_valid=len(received) != 7)

        def fake_save_arrays(path, arrays):
            captured_arrays[Path(path).name] = {k: v.copy() for k, v in arrays.items()}
            # The production file seal still hashes actual fake artifact bytes.
            Path(path).write_bytes(b"manufactured artifact placeholder")
            return dict(filename=Path(path).name, sha256=predictor.c.sha(path))

        cache = SimpleNamespace(read_arrays=read_arrays,
                                select_matches=lambda *a: selected.copy())
        post = SimpleNamespace(load_provider=lambda: object(),
                               with_window_compatibility=lambda x: x)
        modules = dict(normal_offset=SimpleNamespace(), point_postprocess=post,
                       pose_controls=pose, study_core=SimpleNamespace(process_pair=fake_process))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            job_path = base/"job.json"
            predictor.c.write(job_path, dict(expected_commit=commit, cache=str(base/"fake_cache"),
                                            raw_frontend=str(base/"fake_raw")))
            raw = dict(images=[dict(filename=r["filename"], processed={}) for r in images])
            output = base/"prediction"
            with ExitStack() as stack:
                stack.enter_context(patch.object(predictor.c, "verify_source", return_value=sources))
                stack.enter_context(patch.object(predictor.c, "source_hashes", return_value=sources))
                stack.enter_context(patch.object(predictor, "admit_frontend", return_value=(raw, {})))
                stack.enter_context(patch.object(predictor.c, "original_frontend",
                    return_value=(None, None, None, None, {}, None, cache, {"file_sha256": {}})))
                stack.enter_context(patch.object(predictor.c, "selected_layout", return_value=(images, edges)))
                stack.enter_context(patch.object(predictor, "runtime", return_value=(np, self.torch, backend, refiner)))
                stack.enter_context(patch.object(predictor.c, "load_arrays", return_value={"K_resized": K.copy()}))
                stack.enter_context(patch.object(predictor.c, "save_arrays", side_effect=fake_save_arrays))
                stack.enter_context(patch.object(predictor, "prepare_image", side_effect=fake_prepare))
                stack.enter_context(patch.dict(sys.modules, modules))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(predictor.run(SimpleNamespace(job=job_path, output=output,
                                                              expected_commit=commit)), 0)
            receipt, seal = predictor.c.read(output/"receipt.json"), predictor.c.read(output/"seal.json")
            self.assertEqual((len(receipt["images"]), len(receipt["queries"])), (27, 24))
            self.assertEqual(len(received), 24)
            self.assertEqual(len(captured_arrays), 51)
            self.assertFalse(receipt["comparator_valid"])
            self.assertFalse(seal["comparator_valid"])
            self.assertFalse(seal["ground_truth_access"])
            for index in range(24):
                result = predictor.c.read(output/("pair-%02d.json" % index))
                self.assertEqual(result["arms"]["ANG"]["status"], "fallback")
                self.assertEqual(result["original_match_indices"], selected.tolist())
                self.assertEqual(result["selected_match_ids"], ids[selected].tolist())
                self.assertEqual(result["query"], "B"+str(index % 8+1))
                self.assertEqual(set(result["arms"]), set(predictor.c.ARMS))


if __name__ == "__main__":
    unittest.main()
