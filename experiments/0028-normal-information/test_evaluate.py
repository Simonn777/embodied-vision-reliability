"""Fake JSON/opaque bytes only; never read a real cache, model, or truth file."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


common = load("evr0028_eval_test_common", "common.py")
metric_module = load("evr0028_eval_test_metrics", "metrics.py")
with patch.dict(sys.modules, {"common": common, "metrics": metric_module}):
    e = load("evr0028_eval_tested", "evaluate.py")
ROTATION = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
COMMIT = "a" * 40
SOURCES = {"artificial-source-only.py": "b" * 64}


class EvaluationBarrierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pred, self.raw = self.root / "predictions", self.root / "frontend"
        self.pred.mkdir(); self.raw.mkdir()
        self.job_path = self.root / "job.json"
        self.truth_path = self.root / "historical-artificial.json"
        self.events = []
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"common": common, "metrics": metric_module}).start()
        patch.object(common, "verify_source", side_effect=self.source_barrier).start()
        patch.object(common, "source_hashes", return_value=SOURCES).start()
        self.images, self.edges, self.identities = [], [], []
        truth_rows = []
        for si, scene in enumerate(common.SCENES):
            anchor = "val/" + scene + "/seq0/frame_00000.jpg"
            names = [anchor] + ["val/" + scene + "/seq1/frame_%05d.jpg" % n
                                for n in range(1, 9)]
            self.images.extend(dict(image_index=si * 9 + j, filename=name)
                               for j, name in enumerate(names))
            for j, target in enumerate(names[1:]):
                index = si * 8 + j
                edge = dict(scene=scene, source=anchor, target=target,
                            edge_index=index, role="A_to_B")
                self.edges.append(edge)
                identity = dict(scene=scene, query="B" + str(j + 1), anchor=anchor,
                                target=target, edge_index=index, pair_index=index)
                self.identities.append(identity)
                truth_rows.append(dict(**{k: identity[k] for k in e.IDENTITY},
                    ground_truth=dict(relative=[ROTATION, [1., 0., 0.]])))
        common.write(self.truth_path, dict(schema="evr0023.evaluation.v1",
                                          query_count=24, rows=truth_rows))
        self.truth_hash = common.sha(self.truth_path)
        patch.object(e, "GROUND_TRUTH_SHA256", self.truth_hash).start()
        self.old_binding = {"artificial_original_metadata": "only"}
        self.model_binding = {"artificial_model_metadata": "only"}
        self.native_binding = {"version": "artificial", "files": {}}
        self.input_hashes = {"not-a-real-array.npz": "c" * 64}
        native_library = self.root / "fake-native.bytes"
        native_receipt = self.root / "fake-native-receipt.json"
        native_library.write_bytes(b"not an executable library")
        common.write(native_receipt, {"artificial": True})
        self.job = dict(expected_commit=COMMIT, predictions=str(self.pred),
            predictions_outer=str(self.root / "prediction-outer" / "receipt.json"),
            raw_frontend=str(self.raw),
            raw_frontend_outer=str(self.root / "frontend-outer" / "receipt.json"),
            ground_truth_evaluation=str(self.truth_path),
            ground_truth_evaluation_sha256=self.truth_hash,
            native_runtime=str(self.root / "not-imported-runtime"),
            native_library=str(native_library), native_receipt=str(native_receipt),
            native_library_sha256=common.sha(native_library),
            native_receipt_sha256=common.sha(native_receipt))
        common.write(self.job_path, self.job)
        self.job_hash = common.sha(self.job_path)
        self.verified = dict(file_sha256=self.input_hashes,
            layouts=[dict(query_edges=[r for r in self.edges if r["scene"] == scene])
                     for scene in common.SCENES],
            receipt=dict(images=self.images, edges=self.edges))
        fake_old = types.SimpleNamespace(native_binding=lambda *args: self.native_binding)
        fake_frontend = types.SimpleNamespace(
            qualification_args=lambda fa: fa,
            q=types.SimpleNamespace(verify_files=lambda *args: (None, None, None, self.model_binding)))
        patch.object(common, "original_frontend",
            return_value=(fake_old, fake_frontend, None, None, self.old_binding,
                          {}, None, self.verified)).start()
        self.raw_record = dict(self.header("evr0028.frontend.v1"),
            raw_forwards_attempted=27, raw_forwards_completed=27, images=[],
            offline_attempts=[], wall_seconds=.25,
            selected_images=[r["filename"] for r in self.images],
            selected_edges=[{k: r[k] for k in ("scene", "source", "target", "edge_index")}
                            for r in self.edges],
            input_bindings=dict(frontend=self.old_binding, plan_sha256=common.PLAN_SHA,
                model=self.model_binding, cache_file_sha256=self.input_hashes))
        self.pred_record = dict(self.header("evr0028.prediction.v1"),
            images=[], queries=[], offline_attempts=[], wall_seconds=.5,
            old_frontend_binding=self.old_binding, source_input_files=self.input_hashes,
            native_binding=self.native_binding,
            runtime=dict(tc_library_sha256=self.job["native_library_sha256"]),
            comparator_valid=True)
        for original in self.images:
            index = original["image_index"]
            raw = dict(original, status="completed",
                raw=self.opaque(self.raw, "raw-%03d.npz" % index),
                processed=self.opaque(self.raw, "processed-%03d.npz" % index))
            image = dict(original, status="completed", comparator_valid=True,
                         full={"status": "completed"},
                         samples_file=self.opaque(self.pred, "samples-%03d.npz" % index))
            self.raw_record["images"].append(raw)
            self.pred_record["images"].append(image)
            common.write(self.raw / ("image-%03d.json" % index), raw)
            common.write(self.pred / ("image-%03d.json" % index), image)
        for identity in self.identities:
            index = identity["pair_index"]
            name = "pair-%02d.json" % index
            row = dict(identity, schema="evr0028.pair.v1", status="completed",
                original_match_indices=[], selected_match_ids=[], comparator_valid=True,
                arms={arm: dict(status="completed", R=ROTATION, t=[1., 0., 0.])
                      for arm in common.ARMS},
                inputs=self.opaque(self.pred, "pair-%02d-input.npz" % index))
            common.write(self.pred / name, row)
            common.write(self.pred / ("pair-%02d.partial.json" % index),
                         dict(identity=identity, progress={"status": "completed"}))
            self.pred_record["queries"].append(dict(identity, original_match_indices=[],
                selected_match_ids=[], filename=name, sha256=common.sha(self.pred / name)))
        self.reseal_raw()
        self.pred_record["raw_frontend_binding"] = dict(
            outer=common.verify_outer(self.job["raw_frontend_outer"], self.raw / "seal.json",
                                      COMMIT, "frontend.py", self.job_path),
            seal_sha256=common.sha(self.raw / "seal.json"))
        self.reseal_prediction()
        self.args = argparse.Namespace(job=self.job_path, expected_commit=COMMIT,
                                       output=self.root / "evaluation")
        original_read = common.read
        def watched_read(path):
            if Path(path) == self.truth_path:
                self.events.append("truth_decode")
            return original_read(path)
        patch.object(common, "read", side_effect=watched_read).start()
        real_admit = e.admit_predictions
        def watched_admit(*args):
            result = real_admit(*args)
            self.events.append("admitted")
            return result
        patch.object(e, "admit_predictions", side_effect=watched_admit).start()

    def source_barrier(self, commit):
        self.events.append("source")
        self.assertEqual(commit, COMMIT)
        return SOURCES

    def header(self, schema):
        return dict(schema=schema, status="completed", expected_commit=COMMIT,
                    source_sha256=SOURCES, job_sha256=self.job_hash, ground_truth_access=False)

    def opaque(self, root, name):
        path = root / name
        path.write_bytes(b"manufactured opaque bytes, never np.load")
        return dict(filename=name, sha256=common.sha(path),
                    bytes=path.stat().st_size, arrays={})

    def outer(self, path, script, result):
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        for name in ("stdout.log", "stderr.log"):
            (path.parent / name).write_text("artificial log only\n")
        receipt = dict(schema="evr0026.operation.v1", status="completed", exit_code=0,
            termination=None, remaining_descendant_pids=[], cleanup_errors=[],
            expected_commit=COMMIT, hard_timeout_seconds=1800, wall_seconds=1.,
            group_rss_limit_kib=8388608, peak_group_rss_kib=100,
            thread_environment={k: "4" for k in common.THREAD_KEYS},
            supervisor_sha256=common.sha(common.ROOT / "experiments/0027-affine-tangent/supervise.py"),
            monitor_sha256=common.sha(common.ROOT / "experiments/0025c-rs-noise/supervise.py"),
            log_sha256={name: common.sha(path.parent / name) for name in ("stdout.log", "stderr.log")},
            output_sha256=common.sha(result), spec_sha256="d" * 64,
            command=["$PYTHON", "-W", "error",
                "$REPOSITORY/experiments/0028-normal-information/" + script,
                "--job", common.display(self.job_path.resolve()),
                "--expected-commit", COMMIT, "--output", common.display(result.parent.resolve())])
        common.write(path, receipt, replace=path.exists())

    def reseal_raw(self):
        common.write(self.raw / "receipt.json", self.raw_record, replace=True)
        seal = dict(self.header("evr0028.frontend-seal.v1"), image_count=27,
            query_count=24, raw_forward_count=27, receipt_sha256=common.sha(self.raw / "receipt.json"),
            files={p.name: common.sha(p) for p in self.raw.iterdir() if p.name != "seal.json"})
        common.write(self.raw / "seal.json", seal, replace=True)
        self.outer(self.job["raw_frontend_outer"], "frontend.py", self.raw / "seal.json")

    def reseal_prediction(self):
        common.write(self.pred / "receipt.json", self.pred_record, replace=True)
        seal = dict(self.header("evr0028.prediction-seal.v1"), image_count=27,
            query_count=24, arms=list(common.ARMS), comparator_valid=self.pred_record["comparator_valid"],
            receipt_sha256=common.sha(self.pred / "receipt.json"),
            files={p.name: common.sha(p) for p in self.pred.iterdir() if p.name != "seal.json"})
        common.write(self.pred / "seal.json", seal, replace=True)
        self.outer(self.job["predictions_outer"], "predict.py", self.pred / "seal.json")

    def assert_blocked_before_truth(self):
        with self.assertRaises(ValueError):
            e.run(self.args)
        self.assertNotIn("truth_decode", self.events)
        receipt = common.read(self.args.output / "receipt.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["ground_truth_access"])
        self.assertFalse((self.args.output / "seal.json").exists())

    def test_complete_order_and_output_contract(self):
        self.assertEqual(e.run(self.args), 0)
        self.assertEqual(self.events[0], "source")
        self.assertLess(self.events.index("admitted"), self.events.index("truth_decode"))
        result = common.read(self.args.output / "evaluation.json")
        self.assertEqual(len(result["truth_rows"]), 24)
        self.assertEqual(len(result["rows"]), 24)
        self.assertFalse(result["gate"]["passed"])  # Identical controls cannot show +2pp.
        self.assertFalse(result["manuscript_admission"])
        self.assertFalse(result["timing_summary"]["matched_complete_pipeline_time_claim"])
        seal = common.read(self.args.output / "seal.json")
        self.assertEqual(set(seal["files"]), {"receipt.json", "evaluation.json"})
        for name, digest in seal["files"].items():
            self.assertEqual(common.sha(self.args.output / name), digest)

    def test_source_failure_precedes_job_or_output(self):
        patch.object(common, "verify_source", side_effect=ValueError("source mismatch")).start()
        with self.assertRaises(ValueError):
            e.run(self.args)
        self.assertFalse(self.args.output.exists())
        self.assertNotIn("truth_decode", self.events)

    def test_incomplete_inventory_blocks_truth(self):
        (self.pred / "pair-23.json").unlink()
        self.assert_blocked_before_truth()

    def test_failed_outer_blocks_truth(self):
        path = Path(self.job["predictions_outer"])
        receipt = common.read(path); receipt["status"] = "failed"
        common.write(path, receipt, replace=True)
        self.assert_blocked_before_truth()

    def test_sealed_wrong_identity_blocks_truth(self):
        path = self.pred / "pair-00.json"
        row = common.read(path); row["target"] = "other.jpg"
        common.write(path, row, replace=True)
        self.pred_record["queries"][0]["sha256"] = common.sha(path)
        self.reseal_prediction()
        self.assert_blocked_before_truth()

    def test_failed_frontend_blocks_truth(self):
        self.raw_record["status"] = "failed"
        self.reseal_raw()
        self.assert_blocked_before_truth()

    def test_changed_native_asset_blocks_truth(self):
        Path(self.job["native_library"]).write_bytes(b"changed fake library")
        self.assert_blocked_before_truth()

    def test_missing_arm_even_when_resealed_blocks_truth(self):
        path = self.pred / "pair-00.json"
        row = common.read(path); row["arms"].pop("FULL")
        common.write(path, row, replace=True)
        self.pred_record["queries"][0]["sha256"] = common.sha(path)
        self.reseal_prediction()
        self.assert_blocked_before_truth()

    def test_changed_truth_hash_is_never_decoded(self):
        self.truth_path.write_text('{"changed":true}\n')
        with self.assertRaises(ValueError):
            e.run(self.args)
        self.assertIn("admitted", self.events)
        self.assertNotIn("truth_decode", self.events)
        self.assertFalse((self.args.output / "seal.json").exists())

    def test_post_source_mutation_prevents_seal(self):
        patch.object(common, "source_hashes", return_value={"different": "e" * 64}).start()
        with self.assertRaises(ValueError):
            e.run(self.args)
        self.assertIn("truth_decode", self.events)
        self.assertFalse((self.args.output / "seal.json").exists())
        self.assertFalse((self.args.output / "evaluation.json").exists())

    def test_interruption_preserves_failed_barrier_receipt(self):
        patch.object(e, "admit_predictions", side_effect=KeyboardInterrupt()).start()
        with self.assertRaises(KeyboardInterrupt):
            e.run(self.args)
        receipt = common.read(self.args.output / "receipt.json")
        self.assertEqual(receipt["error_type"], "KeyboardInterrupt")
        self.assertFalse(receipt["ground_truth_access"])
        self.assertNotIn("truth_decode", self.events)

    def test_existing_output_refused(self):
        self.args.output.mkdir()
        (self.args.output / "sentinel").write_text("keep")
        with self.assertRaises(FileExistsError):
            e.run(self.args)
        self.assertEqual((self.args.output / "sentinel").read_text(), "keep")
        self.assertNotIn("truth_decode", self.events)


if __name__ == "__main__":
    unittest.main()
