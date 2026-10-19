"""Artificial supervisor tests: no acquired HDF5 or estimator is used."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_baseline as runner
worker = runner.worker

SOURCES = {name: str(index + 1) * 64 for index, name in enumerate(runner.SOURCE_NAMES)}
LIBRARY = {**runner.PINNED_LIBRARY, "test_runtime": "artificial-only"}


def identity(index=0, key="0015-artificial-00", manifest_hash="a" * 64, data_hash="b" * 64):
    return {"schema": worker.SCHEMA, "pair_index": index, "key": key,
            "manifest_sha256": manifest_hash, "dataset_sha256_declared": data_hash,
            "source_sha256": SOURCES["predict_pair.py"]}


def prediction(binding=None, status="completed", phase="completed"):
    return {**(identity() if binding is None else binding), "status": status,
            "phase": phase, "settings": copy.deepcopy(worker.SETTINGS),
            "library": dict(LIBRARY),
            "estimate": {"R": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                         "t": [1., 0., 0.]}}


def process_receipt(exit_code=0, termination=None):
    return {"exit_code": exit_code, "termination": termination,
            "wall_seconds": 0.125, "sampled_peak_rss_kib": 1024,
            "watchdog_error": None, "rss_sampling": "artificial"}


class SupervisorTests(unittest.TestCase):
    def test_binding_settings_and_library_changes_are_rejected(self):
        binding = identity()
        runner.verify_record(prediction(), binding, LIBRARY)
        for field in binding:
            with self.subTest(binding=field):
                raw = prediction()
                raw[field] = "changed"
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    runner.verify_record(raw, binding, LIBRARY)
        for mutate in (lambda r: r["settings"]["ransac_options"].update(max_iterations=10000),
                       lambda r: r["library"].update(binary_sha256="0" * 64),
                       lambda r: r["library"].update(version="2.0.4")):
            raw = prediction()
            mutate(raw)
            with self.assertRaises(ValueError):
                runner.verify_record(raw, binding, LIBRARY)

    def test_only_native_phase_pair_timeout_is_eligible(self):
        receipt = process_receipt(-9, "pair_timeout")
        raw = prediction(status="started", phase="estimator")
        self.assertEqual(runner.wrap_record(raw, identity(), LIBRARY, receipt)["status"], "timeout")
        for phase in ("input", "binding"):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                runner.wrap_record(prediction(status="started", phase=phase), identity(), LIBRARY, receipt)
        for termination in ("total_timeout", "rss_limit", "watchdog_error"):
            with self.subTest(termination=termination), self.assertRaises(ValueError):
                runner.wrap_record(raw, identity(), LIBRARY, process_receipt(-9, termination))

    def test_crash_error_and_exit_consistency(self):
        native_crash = prediction(status="started", phase="estimator")
        self.assertEqual(runner.wrap_record(native_crash, identity(), LIBRARY,
                         process_receipt(-11))["status"], "estimator_error")
        native_error = prediction(status="error", phase="error")
        native_error["failure_kind"] = "estimator"
        self.assertEqual(runner.wrap_record(native_error, identity(), LIBRARY,
                         process_receipt(1))["status"], "estimator_error")
        for kind in ("input", "output", "binding"):
            raw = {**native_error, "failure_kind": kind}
            with self.subTest(failure_kind=kind), self.assertRaises(ValueError):
                runner.wrap_record(raw, identity(), LIBRARY, process_receipt(1))
        for raw, receipt in ((native_crash, process_receipt(0)),
                             (native_error, process_receipt(0)),
                             (prediction(), process_receipt(1)),
                             (prediction(), process_receipt(-9, "pair_timeout"))):
            with self.subTest(raw_status=raw["status"], receipt=receipt), self.assertRaises(ValueError):
                runner.wrap_record(raw, identity(), LIBRARY, receipt)

    def test_invalid_completed_pose_blocks_seal(self):
        for value in ([0., 0., 0.], [float("nan"), 0., 0.]):
            raw = prediction()
            raw["estimate"]["t"] = value
            with self.assertRaises(ValueError):
                runner.verify_record(raw, identity(), LIBRARY)
        raw = prediction()
        raw["estimate"]["R"][0][0] = 2.
        with self.assertRaises(ValueError):
            runner.verify_record(raw, identity(), LIBRARY)

    def test_existing_directory_refuses_before_any_input_or_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already-used"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("existing scientific evidence\n")
            with patch.object(runner, "source_hashes", side_effect=AssertionError("source read")), \
                 patch.object(runner.worker, "sha256", side_effect=AssertionError("file read")), \
                 patch.object(runner, "run_child", side_effect=AssertionError("child started")):
                with self.assertRaises(FileExistsError):
                    runner.run("does-not-exist.h5", "does-not-exist.json", output)
            self.assertEqual(sentinel.read_text(), "existing scientific evidence\n")
            self.assertEqual([p.name for p in output.iterdir()], ["keep.txt"])

    def test_watchdog_exception_kills_and_reaps_child(self):
        class FakeProcess:
            pid = 123456789
            def __init__(self): self.killed, self.reaped = False, False
            def poll(self): return -9 if self.killed else None
            def kill(self): self.killed = True
            def wait(self, timeout):
                if not self.killed: raise AssertionError("waiting on un-killed running child")
                self.reaped = True
                return -9
        for failure in (subprocess.TimeoutExpired(["/bin/ps"], 2), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                child = FakeProcess()
                with patch.object(runner.subprocess, "Popen", return_value=child) as launch, \
                     patch.object(runner.subprocess, "run", side_effect=failure):
                    if isinstance(failure, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            runner.run_child(["artificial-child"], Path(temporary) / "log", 1800)
                    else:
                        receipt = runner.run_child(["artificial-child"], Path(temporary) / "log", 1800)
                        self.assertEqual(receipt["termination"], "watchdog_error")
                        self.assertEqual(receipt["watchdog_error"], "TimeoutExpired")
                        self.assertEqual(receipt["exit_code"], -9)
                        with self.assertRaises(ValueError):
                            runner.wrap_record(prediction(status="started", phase="estimator"),
                                               identity(), LIBRARY, receipt)
                self.assertTrue(child.killed)
                self.assertTrue(child.reaped)
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                for variable in worker.THREAD_VARIABLES:
                    self.assertEqual(launch.call_args.kwargs["env"][variable], "1")


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.manifest_path = self.root / "artificial.h5", self.root / "manifest.json"
        self.data.write_bytes(b"artificial bytes: deliberately not an HDF5 dataset")
        self.keys = ["0015-artificial-%02d" % i for i in range(16)]
        self.manifest = {"schema": "evr0012.development.v1", "development_scene": "0015",
            "coordinate_convention": "raw_pixels", "selected_pair_keys": self.keys,
            "dataset_bytes": self.data.stat().st_size, "dataset_sha256": worker.sha256(self.data)}
        self.manifest_path.write_text(json.dumps(self.manifest))
        protocols = self.root / "protocols"
        protocols.mkdir()
        (protocols / "0013-baseline-pose-headroom.md").write_text("artificial prospective protocol\n")
        self.output, self.calls = self.root / "prediction-output", []

    def fake_child(self, command, log_path, remaining, critical_index=None):
        index = int(command[command.index("--pair-index") + 1])
        output = Path(command[command.index("--output") + 1])
        self.calls.append(index)
        self.assertGreater(remaining, 0)
        binding = identity(index, self.keys[index], worker.sha256(self.manifest_path), self.manifest["dataset_sha256"])
        raw, receipt = prediction(binding), process_receipt()
        if index == 3:
            raw.update(status="started", phase="estimator")
            receipt = process_receipt(-9, "pair_timeout")
        if index == 7:
            raw.update(status="error", phase="error", failure_kind="estimator")
            receipt = process_receipt(1)
        if index == critical_index:
            raw.update(status="error", phase="error", failure_kind="input")
            receipt = process_receipt(1)
        runner.write_new(output, raw)
        with Path(log_path).open("xb") as stream:
            stream.write(b"artificial child only\n")
        return receipt

    def execute(self, critical_index=None, changed_source=False):
        changed = {**SOURCES, "predict_pair.py": "9" * 64}
        with patch.object(runner, "REPO_ROOT", self.root), \
             patch.object(runner, "source_hashes", side_effect=[dict(SOURCES), changed] if changed_source else None,
                          return_value=dict(SOURCES)), \
             patch.object(runner, "expected_library", return_value=dict(LIBRARY)), \
             patch.object(runner, "run_child", side_effect=lambda c, p, r: self.fake_child(c, p, r, critical_index)), \
             contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.data, self.manifest_path, self.output)

    def test_full16_seal_binds_all_rows_and_preserves_coverage_failures(self):
        self.assertEqual(self.execute(), 0)
        seal = json.loads((self.output / "seal.json").read_text())
        self.assertTrue(seal["complete"])
        self.assertTrue(seal["eligible_for_evaluation"])
        self.assertIsNone(seal["error"])
        self.assertEqual(self.calls, list(range(16)))
        self.assertEqual([row["key"] for row in seal["rows"]], self.keys)
        self.assertEqual(seal["source_sha256"], SOURCES)
        self.assertEqual(seal["manifest_sha256"], worker.sha256(self.manifest_path))
        for index, row in enumerate(seal["rows"]):
            path = self.output / row["path"]
            self.assertEqual(row["pair_index"], index)
            self.assertEqual(row["sha256"], worker.sha256(path))
            final = json.loads(path.read_text())
            receipt = final["process_receipt"]
            raw_path = self.output / receipt["raw_worker_path"]
            self.assertEqual(receipt["raw_worker_sha256"], worker.sha256(raw_path))
            expected = "timeout" if index == 3 else "estimator_error" if index == 7 else "completed"
            self.assertEqual(row["status"], expected)
            self.assertEqual(final["status"], expected)
        before = (self.output / "seal.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual((self.output / "seal.json").read_bytes(), before)

    def test_critical_input_failure_stops_without_eligible_seal(self):
        self.assertEqual(self.execute(critical_index=4), 1)
        seal = json.loads((self.output / "seal.json").read_text())
        self.assertFalse(seal["complete"])
        self.assertFalse(seal["eligible_for_evaluation"])
        self.assertEqual(self.calls, [0, 1, 2, 3, 4])
        self.assertEqual(len(seal["rows"]), 4)
        self.assertEqual(seal["error"]["type"], "ValueError")
        self.assertTrue((self.output / "pair-04.worker.json").exists())
        self.assertTrue((self.output / "pair-04.process.json").exists())
        self.assertFalse((self.output / "pair-04.json").exists())
        self.assertFalse((self.output / "pair-05.worker.json").exists())

    def test_source_change_after16_predictions_invalidates_seal(self):
        self.assertEqual(self.execute(changed_source=True), 1)
        seal = json.loads((self.output / "seal.json").read_text())
        self.assertEqual(len(seal["rows"]), 16)
        self.assertFalse(seal["eligible_for_evaluation"])
        self.assertIn("source or manifest changed", seal["error"]["message"])


if __name__ == "__main__":
    unittest.main()
