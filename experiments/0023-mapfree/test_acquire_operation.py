"""Artificial acquisition-wrapper checks: no HTTP or scientific member access."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("evr0023_acquisition_test", Path(__file__).with_name("acquire_operation.py"))
op = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = op
spec.loader.exec_module(op)


def artificial_index():
    entries = {}
    for scene in op.planner.EXPECTED_SCENES:
        names = ["val/%s/seq1/frame_%05d.jpg" % (scene, i) for i in range(31)]
        names += ["val/%s/%s" % (scene, name) for name in ("seq0/frame_00000.jpg", "intrinsics.txt", "poses.txt")]
        for name in names:
            entries[name] = dict(filename=name, flags=0, compression_method=8, crc32_hex="00000000",
                                 compressed_bytes=13, uncompressed_bytes=29, local_header_offset=0)
    return op.ranges.Index(op.ranges.URL, '"artificial"', 1000000, 900000, entries,
                           {key: "0" * 64 for key in ("members", "directory", "receipt")})


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="evr0023-acquire-fake-")
        self.root = Path(self.tmp.name)
        self.index = artificial_index()
        self.plan = op.planner.build_plan(self.index, "1" * 64)
        self.plan["planning_source_sha256"] = {name: op.sha(op.HERE / name)
                                                for name in ("propose_plan.py", "range_inputs.py")}
        self.args = SimpleNamespace(expected_commit="a" * 40, plan=self.root / "plan.json",
                                    protocol=self.root / "protocol.md", members_index=self.root / "members.json",
                                    central_directory=self.root / "central.bin", index_receipt=self.root / "index.json",
                                    output=self.root / "operation")
        self.save_plan()
        self.bindings = {"experiments/0023-mapfree/range_inputs.py": op.sha(op.HERE / "range_inputs.py")}
        self.frozen = patch.object(op.supervisor, "verify_frozen", return_value=self.bindings)
        self.load = patch.object(op.ranges, "load_index", return_value=self.index)
        self.frozen_mock = self.frozen.start(); self.load_mock = self.load.start()
        self.addCleanup(self.load.stop); self.addCleanup(self.frozen.stop); self.addCleanup(self.tmp.cleanup)
        self.calls = []

    def save_plan(self):
        write(self.args.plan, self.plan)
        self.args.plan_sha256 = op.sha(self.args.plan)
        self.args.protocol.write_text("Artificial frozen adoption: " + self.args.plan_sha256 + "\n")

    def fake_acquire(self, index, names, output, receipt_path, limits):
        self.calls.append((list(names), vars(limits)))
        output.mkdir()
        rows = []
        for name in names:
            self.assertIn(op.ranges.frontend_kind(name), ("image", "intrinsics"))
            path = output / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * index.entries[name]["uncompressed_bytes"])
            rows.append(dict(index.entries[name], sha256=op.sha(path)))
        requests = [dict(status=206, body_bytes=n) for name in names for n in (30, len(name), 13)]
        result = dict(schema=op.ranges.SCHEMA, status="completed", selected_names=names,
                      limits=vars(limits), source_sha256=op.sha(op.HERE / "range_inputs.py"),
                      index_sha256=index.fingerprints, url=index.url, etag=index.etag,
                      archive_bytes=index.archive_bytes, pose_gt_members_allowed=False,
                      hidden_retries=False, full_zip_fallback=False, response_body_bytes_exact=True,
                      requests=requests,
                      response_body_bytes=sum(row["body_bytes"] for row in requests),
                      output_bytes=sum(row["uncompressed_bytes"] for row in rows),
                      members=rows, wall_seconds=0.001)
        write(receipt_path, result)
        return result

    def test_complete_exact54_fixed_caps_and_private_output_chain(self):
        with patch.object(op.ranges, "acquire", side_effect=self.fake_acquire):
            result = op.run(self.args)
        self.assertEqual((result["status"], result["eligible"]), ("completed", True))
        self.assertEqual(len(self.calls), 1)
        names, limits = self.calls[0]
        self.assertEqual(len(names), 54)
        self.assertFalse(any(name.endswith("poses.txt") for name in names))
        self.assertEqual(limits, dict(op.FIXED_LIMITS, output_bytes=54 * 29, member_bytes=29))
        self.assertEqual(result["acquisition_receipt_sha256"], op.sha(self.args.output / "acquisition.json"))
        self.assertEqual(self.frozen_mock.call_count, 2)
        self.assertEqual(self.load_mock.call_count, 2)
        self.assertTrue(result["sources_unchanged"])
        self.assertFalse(result["calibration_parsing"])
        self.assertEqual(json.loads((self.args.output / "operation.json").read_text()), result)

    def test_hash_protocol_and_frozen_source_fail_before_acquisition(self):
        with patch.object(op.ranges, "acquire") as acquire:
            self.args.plan_sha256 = "f" * 64
            with self.assertRaisesRegex(ValueError, "protocol"):
                op.run(self.args)
            acquire.assert_not_called()
        self.assertEqual(json.loads((self.args.output / "operation.json").read_text())["status"], "failed")
        self.args.output = self.root / "wrong-plan"
        self.args.protocol.write_text(self.args.plan_sha256)
        with patch.object(op.ranges, "acquire") as acquire:
            with self.assertRaisesRegex(ValueError, "private plan SHA"):
                op.run(self.args)
            acquire.assert_not_called()
        self.args.output = self.root / "bad-source"
        self.frozen_mock.side_effect = ValueError("source differs from reviewed commit")
        with patch.object(op.ranges, "acquire") as acquire:
            with self.assertRaisesRegex(ValueError, "source differs"):
                op.run(self.args)
            acquire.assert_not_called()

    def test_rehashed_changed_plan_cannot_add_gt_or_change_cohort(self):
        self.plan["frontend_acquisition"]["member_names"][-1] = "val/s00478/poses.txt"
        self.save_plan()
        with patch.object(op.ranges, "acquire") as acquire:
            with self.assertRaisesRegex(ValueError, "deterministic metadata proposal"):
                op.run(self.args)
            acquire.assert_not_called()

    def test_fresh_private_output_required(self):
        self.args.output.mkdir()
        with patch.object(op.ranges, "acquire") as acquire:
            with self.assertRaisesRegex(ValueError, "fresh"):
                op.run(self.args)
            self.args.output = op.ROOT / "must-not-create-acquisition-fixture"
            with self.assertRaisesRegex(ValueError, "outside public"):
                op.run(self.args)
            acquire.assert_not_called()

    def test_interruption_preserves_helper_and_failure_receipt_without_retry(self):
        def interrupted(index, names, output, receipt_path, limits):
            write(receipt_path, {"status": "error", "response_body_bytes": 7})
            raise KeyboardInterrupt("artificial interruption")
        with patch.object(op.ranges, "acquire", side_effect=interrupted) as acquire:
            with self.assertRaises(KeyboardInterrupt):
                op.run(self.args)
            self.assertEqual(acquire.call_count, 1)
        result = json.loads((self.args.output / "operation.json").read_text())
        self.assertEqual((result["status"], result["failure_stage"]), ("failed", "acquisition"))
        self.assertFalse(result["eligible"])
        self.assertEqual(result["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(result["acquisition_receipt_sha256"], op.sha(self.args.output / "acquisition.json"))

    def test_post_acquisition_binding_change_is_ineligible(self):
        self.frozen_mock.side_effect = [self.bindings, dict(self.bindings, changed="f" * 64)]
        with patch.object(op.ranges, "acquire", side_effect=self.fake_acquire):
            with self.assertRaisesRegex(ValueError, "bindings changed"):
                op.run(self.args)
        result = json.loads((self.args.output / "operation.json").read_text())
        self.assertEqual((result["status"], result["failure_stage"]), ("failed", "postcheck"))
        self.assertEqual(len(self.calls), 1)

    def test_output_and_ledger_tampering_fail(self):
        def tampered(index, names, output, receipt_path, limits):
            result = self.fake_acquire(index, names, output, receipt_path, limits)
            (output / names[0]).write_bytes(b"y" * 29)
            return result
        with patch.object(op.ranges, "acquire", side_effect=tampered):
            with self.assertRaisesRegex(ValueError, "hash/size"):
                op.run(self.args)
        self.args.output = self.root / "ledger-failure"
        def undercount(index, names, output, receipt_path, limits):
            result = self.fake_acquire(index, names, output, receipt_path, limits)
            result["response_body_bytes"] -= 1
            write(receipt_path, result)
            return result
        with patch.object(op.ranges, "acquire", side_effect=undercount):
            with self.assertRaisesRegex(ValueError, "response ledger"):
                op.run(self.args)

    def test_consumer_requires_outer_inner_command_sources_and_logs(self):
        keys = ["experiments/0023-mapfree/" + name for name in
                ("supervise.py", "acquire_operation.py", "range_inputs.py", "propose_plan.py")]
        self.bindings.update({key: op.sha(op.ROOT / key) for key in keys})
        with patch.object(op.ranges, "acquire", side_effect=self.fake_acquire):
            inner = op.run(self.args)
        outer_dir = self.root / "supervision"; outer_dir.mkdir()
        for kind, path in (("members", self.args.members_index), ("directory", self.args.central_directory),
                           ("receipt", self.args.index_receipt)):
            path.write_bytes(b"artificial metadata " + kind.encode())
            inner["index_sha256"][kind] = op.sha(path)
        write(self.args.output / "operation.json", inner)
        (outer_dir / "stdout.log").write_text(json.dumps({key: inner[key] for key in
            ("status", "eligible", "wall_seconds")}) + "\n")
        (outer_dir / "stderr.log").write_text("")
        command = ["$PYTHON", "$REPOSITORY/experiments/0023-mapfree/acquire_operation.py",
                   "--expected-commit", self.args.expected_commit, "--plan-sha256", self.args.plan_sha256]
        for flag, name in (("--protocol", "protocol"), ("--plan", "plan"), ("--output", "output"),
                           ("--members-index", "members_index"), ("--central-directory", "central_directory"),
                           ("--index-receipt", "index_receipt")):
            command += [flag, str(getattr(self.args, name))]
        outer = dict(schema="evr0023.supervision.v1", operation="acquisition", status="completed", exit_code=0,
                     expected_commit=self.args.expected_commit, source_sha256={key: self.bindings[key] for key in keys},
                     termination=None, remaining_descendant_pids=[], cleanup_errors=[], wall_seconds=.5,
                     hard_timeout_seconds=1800, peak_group_rss_kib=100, group_rss_limit_kib=8 * 1024**2,
                     thread_environment={key: "4" for key in op.supervisor.THREAD_KEYS}, command=command,
                     stdout_sha256=op.sha(outer_dir / "stdout.log"), stderr_sha256=op.sha(outer_dir / "stderr.log"))
        write(outer_dir / "receipt.json", outer)
        self.args.input_operation = self.args.output / "operation.json"
        self.args.input_supervision = outer_dir / "receipt.json"
        self.args.input_receipt = self.args.output / "acquisition.json"
        self.args.input_receipt_sha256 = op.sha(self.args.input_receipt)
        self.args.inputs = self.args.output / "inputs"
        result = op.verify_completed_chain(self.args)
        self.assertEqual(result["input_supervision_sha256"], op.sha(self.args.input_supervision))
        # Synthetic protocol is outside the repository; isolate this source
        # branch with the four real owned helper paths. The production call
        # also includes its frozen in-repository protocol.
        with patch.object(op, "source_paths", return_value=[op.ROOT / key for key in keys]), \
             patch.object(op.supervisor, "verify_frozen", side_effect=AssertionError("offline subprocess forbidden")):
            self.assertEqual(op.verify_completed_chain(self.args, verified_bindings=self.bindings), result)
            with self.assertRaisesRegex(ValueError, "previously verified acquisition source changed"):
                op.verify_completed_chain(self.args, verified_bindings=dict(self.bindings, **{keys[0]: "0" * 64}))
        for change in ({"status": "failed"}, {"exit_code": 1}, {"termination": "timeout"},
                       {"hard_timeout_seconds": 9000}, {"stdout_sha256": "0" * 64},
                       {"source_sha256": {}}, {"command": command + ["--expected-commit", "b" * 40]}):
            write(self.args.input_supervision, dict(outer, **change))
            with self.assertRaises(ValueError):
                op.verify_completed_chain(self.args)
        write(self.args.input_supervision, outer)
        inner["eligible"] = False
        write(self.args.input_operation, inner)
        with self.assertRaisesRegex(ValueError, "completed acquisition operation"):
            op.verify_completed_chain(self.args)


if __name__ == "__main__":
    unittest.main()
