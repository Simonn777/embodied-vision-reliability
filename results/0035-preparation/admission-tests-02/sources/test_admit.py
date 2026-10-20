"""Artificial proof-chain checks: opaque one-byte files, no arrays or media."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_test_vost_admit", HERE / "admit.py")
admit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = admit
spec.loader.exec_module(admit)


class FakeProof:
    def __init__(self, root):
        self.root = root
        source = root / "source.py"
        source.write_text("# inert manufactured source\n")
        self.source = source
        self.sources = {str(source): admit.sha(source)}
        members = []
        for name, count in admit.COHORT:
            for i in range(count):
                for kind in ("rgb", "mask"):
                    members.append(dict(filename=f"{kind}/{name}/{i:05d}", sequence=name, frame_index=i,
                                        phase="rgb" if kind == "rgb" else ("initial" if i == 0 else "metric"),
                                        uncompressed_bytes=1))
        self.manifest = {"clips": [dict(sequence=n, frames=c) for n, c in admit.COHORT], "members": members}
        self.proof = dict(schema="evr0035.prediction-proof.v1", source_sha256=self.sources)
        self.proof["manifest"] = self.write("manifest.json", self.manifest)
        self.manifest_sha = self.proof["manifest"]["sha256"]
        self.acq = {}
        for phase in ("initial", "rgb"):
            rows = []
            for row in members:
                if row["phase"] != phase:
                    continue
                file = root / phase / "members" / row["filename"]
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"I" if phase == "initial" else b"R")
                rows.append(dict(row, saved_relative=file.relative_to(root / phase).as_posix(),
                                 bytes=1, sha256=admit.sha(file)))
            self.acq[phase] = dict(schema="evr0035.vost-acquisition.v1", status="completed", phase=phase,
                                  index_sha256={"manifest": self.manifest_sha}, retry_count=0,
                                  hidden_retries=False, full_archive_fallback=False, media_decoded=False,
                                  members=rows, source_sha256={"source.py": admit.sha(source)})
        self.proof["initial_acquisition"] = self.write("initial/receipt.json", self.acq["initial"])
        self.proof["initial_outer"] = self.outer("initial-outer", self.proof["initial_acquisition"]["sha256"])
        self.inspection = dict(status="completed", eligible=True, initial_only=True, noninitial_labels_read=0,
                               model_calls=0, source_sha256=self.sources, prediction_output_cap_bytes=1000000,
                               fingerprints={self.proof["initial_acquisition"]["path"]:
                                             self.proof["initial_acquisition"]["sha256"]},
                               checks={k: True for k in ("native_object_frames_at_most_2000",
                                   "planning_memory_at_most_7GiB", "prediction_output_at_most_20GiB",
                                   "free_disk_covers_twice_output_plus_5GiB")},
                               rows=[dict(sequence=n, frames=c, object_ids=[1], shape=[2, 3])
                                     for n, c in admit.COHORT])
        self.proof["initial_inspection"] = self.write("inspection/receipt.json", self.inspection)
        self.proof["initial_inspection_outer"] = self.outer("inspection-outer", self.proof["initial_inspection"]["sha256"])
        self.acq["rgb"]["initial_admission"] = dict(
            receipt_sha256=self.proof["initial_acquisition"]["sha256"],
            inspection_sha256=self.proof["initial_inspection"]["sha256"],
            inspection_outer_sha256=self.proof["initial_inspection_outer"]["sha256"])
        self.proof["rgb_acquisition"] = self.write("rgb/receipt.json", self.acq["rgb"])
        self.proof["rgb_outer"] = self.outer("rgb-outer", self.proof["rgb_acquisition"]["sha256"])
        def input_desc(phase, row):
            return dict(path=str(root / phase / row["saved_relative"]), frame=row["frame_index"],
                        bytes=1, sha256=row["sha256"])
        index = {"sequences": []}
        for name, count in admit.COHORT:
            initial = next(r for r in self.acq["initial"]["members"] if r["sequence"] == name)
            rgb = [r for r in self.acq["rgb"]["members"] if r["sequence"] == name]
            index["sequences"].append(dict(sequence=name, initial=input_desc("initial", initial),
                                            rgb=[input_desc("rgb", r) for r in rgb]))
        index_desc = self.write("data-index.json", index)
        prediction_root = root / "predictions"
        prediction_root.mkdir()
        predict_spec = dict(source_sha256=self.sources, data_index=index_desc, repo=str(root),
                            output=str(prediction_root), max_output_bytes=1000000)
        self.proof["predict_spec"] = self.write("prediction-spec.json", predict_spec)
        self.receipt = dict(schema="evr0035.native-bank.v1", status="completed", scientific_eligible=True,
                            source_sha256=self.sources, completed_frames=446, completed_sequences=4,
                            noninitial_label_decodes=0, initial_label_decodes=[n for n, _ in admit.COHORT],
                            spec_sha256=self.proof["predict_spec"]["sha256"], data_index=index_desc,
                            output=str(prediction_root), rows=[], outputs=[], calls={})
        for name, count in admit.COHORT:
            outputs = []
            for i in range(count):
                file = prediction_root / f"{name}-{i:05d}.npz"
                file.write_bytes(b"P")  # Not an archive and never decoded.
                outputs.append(dict(sequence=name, frame=i, path=file.name, bytes=1, sha256=admit.sha(file)))
            self.receipt["rows"].append(dict(sequence=name, frames=count, object_ids=[1], shape=[2, 3],
                status="completed", completed_frames=count, predictions=outputs,
                frame_records=[dict(frame=i, selected_low_exact=True, selected_original_exact=True)
                               for i in range(count)]))
            self.receipt["outputs"].extend(outputs)
        self.receipt["calls"] = {kind + "_" + suffix: 446
                                for kind in ("image_encoder", "track_step", "mask_decoder", "memory_encoder")
                                for suffix in ("intent", "completed")}
        self.refresh_predictions()
        qualification = dict(status="completed", eligible=True, manufactured_only=True,
                             scientific_input_access=False, full_state_parity=True,
                             native_output_parities=[{"exact": True}] * 7, source_sha256=self.sources)
        self.proof["qualification_receipt"] = self.write("qualification/receipt.json", qualification)
        self.proof["qualification_outer"] = self.outer("qualification-outer", self.proof["qualification_receipt"]["sha256"])
        self.path = root / "proof.json"
        self.save()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))
        return dict(path=str(path), sha256=admit.sha(path))

    def outer(self, name, output_sha):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        for filename in ("stdout.log", "stderr.log"):
            (directory / filename).write_text("manufactured only\n")
        value = dict(schema="evr0029.exit-checked-operation.v1", status="completed", exit_code=0,
                     supervisor_sha256=admit.SUPERVISOR_SHA256, monitor_sha256=admit.MONITOR_SHA256,
                     thread_environment={key: "4" for key in admit.THREAD_KEYS},
                     hard_timeout_seconds=1800, group_rss_limit_kib=8 * 1024 * 1024,
                     termination=None, remaining_descendant_pids=[], cleanup_errors=[], wall_seconds=0.01,
                     peak_group_rss_kib=1024, output_sha256=output_sha,
                     log_sha256={f: admit.sha(directory / f) for f in ("stdout.log", "stderr.log")})
        return self.write(name + "/receipt.json", value)

    def refresh_predictions(self):
        self.proof["predict_receipt"] = self.write("predictions/receipt.json", self.receipt)
        seal = dict(schema="evr0035.native-bank-seal.v1", complete=True, source_sha256=self.sources,
                    receipt_sha256=self.proof["predict_receipt"]["sha256"], spec_sha256=self.receipt["spec_sha256"],
                    outputs=self.receipt["outputs"], completed_frames=446, completed_sequences=4,
                    data_index=self.receipt["data_index"])
        self.proof["predict_seal"] = self.write("predictions/seal.json", seal)
        self.proof["predict_outer"] = self.outer("prediction-outer", self.proof["predict_receipt"]["sha256"])

    def save(self):
        self.path.write_text(json.dumps(self.proof))

    def verify(self, **kwargs):
        with patch.object(admit, "MANIFEST_SHA256", self.manifest_sha):
            return admit.verify_prediction_proof(self.path, self.manifest_sha, **kwargs)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bundle = FakeProof(Path(self.temp.name).resolve())

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_four_clip_chain_and_opaque_hash_scope(self):
        result = self.bundle.verify()
        self.assertTrue(result["eligible"])
        self.assertTrue(result["outputs_verified"])
        self.assertEqual(result["object_frames"], 446)
        self.assertEqual(result["output_bytes"], 446)
        self.assertEqual(len(result["prediction_receipt"]["outputs"]), 446)
        admit.postcheck(result)

    def test_same_size_prediction_tamper_fails_full_hash_barrier(self):
        row = self.bundle.receipt["outputs"][220]
        (self.bundle.root / "predictions" / row["path"]).write_bytes(b"X")
        with self.assertRaisesRegex(ValueError, "prediction byte seal"):
            self.bundle.verify(check_outputs=True)
        self.assertFalse(self.bundle.verify(check_outputs=False)["outputs_verified"])

    def test_incomplete_prediction_and_wrong_native_call_count(self):
        self.bundle.receipt["completed_frames"] = 445
        self.bundle.refresh_predictions()
        self.bundle.save()
        with self.assertRaisesRegex(ValueError, "complete prediction receipt"):
            self.bundle.verify()
        self.bundle.receipt["completed_frames"] = 446
        self.bundle.receipt["calls"]["mask_decoder_completed"] = 445
        self.bundle.refresh_predictions()
        self.bundle.save()
        with self.assertRaisesRegex(ValueError, "native call accounting"):
            self.bundle.verify()

    def test_inspection_is_bound_to_same_initial_receipt(self):
        self.bundle.inspection["fingerprints"] = {}
        self.bundle.proof["initial_inspection"] = self.bundle.write("inspection/receipt.json", self.bundle.inspection)
        self.bundle.proof["initial_inspection_outer"] = self.bundle.outer(
            "inspection-outer", self.bundle.proof["initial_inspection"]["sha256"])
        self.bundle.save()
        with self.assertRaisesRegex(ValueError, "inspection belongs"):
            self.bundle.verify()

    def test_initial_bytes_and_source_changes_rejected(self):
        row = self.bundle.acq["initial"]["members"][0]
        file = self.bundle.root / "initial" / row["saved_relative"]
        file.write_bytes(b"X")
        with self.assertRaisesRegex(ValueError, "acquired member changed"):
            self.bundle.verify()
        file.write_bytes(b"I")
        self.bundle.source.write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "proof source changed"):
            self.bundle.verify()

    def test_outer_negative_resources_missing_logs_and_residual_rejected(self):
        desc = self.bundle.proof["predict_outer"]
        original = json.loads(Path(desc["path"]).read_text())
        for change in ({"wall_seconds": -1}, {"peak_group_rss_kib": -1}, {"log_sha256": {}},
                       {"termination": "descendants_remain"}, {"remaining_descendant_pids": [123]},
                       {"supervisor_sha256": "0" * 64}, {"monitor_sha256": "0" * 64},
                       {"thread_environment": {}}, {"hard_timeout_seconds": 0},
                       {"hard_timeout_seconds": 1801}, {"group_rss_limit_kib": 9 * 1024 * 1024}):
            new = dict(original, **change)
            replaced = self.bundle.write("prediction-outer/receipt.json", new)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "clean complete outer"):
                admit.clean_outer(replaced, original["output_sha256"], {})


if __name__ == "__main__":
    unittest.main()
