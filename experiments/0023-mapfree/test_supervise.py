"""Fake processes only; no child process, network, payload, or model execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("evr0023_supervision_test", Path(__file__).with_name("supervise.py"))
sup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sup
spec.loader.exec_module(sup)


class FakeProcess:
    pid = 123456
    def __init__(self, code=0):
        self.returncode = code
    def poll(self):
        return self.returncode
    def wait(self, timeout):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="evr0023-supervise-fake-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(operation="acquisition", expected_commit="a" * 40,
                                    output_dir=self.root / "output", command_json="")
        self.set_command()

    def set_command(self, script="acquire_operation.py", commit=None):
        self.args.command_json = json.dumps([str(sup.ROOT / "experiments/0023-mapfree" / script),
                                            "--expected-commit", commit or self.args.expected_commit])

    def execute(self, process=None, rss=100, clock=None):
        process = process or FakeProcess()
        with patch.object(sup, "verify_frozen", return_value={}) as frozen, \
             patch.object(sup.subprocess, "Popen", return_value=process) as launch, \
             patch.object(sup, "snapshot", return_value=([(process.pid, 1, process.pid, rss)], {process.pid})), \
             patch.object(sup.os, "killpg") as kill, patch.object(sup.time, "sleep"), \
             patch.object(sup.time, "perf_counter", side_effect=clock or [0, .01, .02]):
            code = sup.run(self.args)
        record = json.loads((self.args.output_dir / "receipt.json").read_text())
        return code, record, frozen, launch, kill

    def test_original_resource_and_cleanup_logic_is_unchanged(self):
        original = (sup.ROOT / "experiments/0022-frontend/supervise.py").read_text()
        current = Path(sup.__file__).read_text()
        start = "    began = time.perf_counter()\n"
        end = "\ndef main():"
        self.assertEqual(original.split(start, 1)[1].split(end, 1)[0],
                         current.split(start, 1)[1].split(end, 1)[0])
        self.assertEqual((sup.TIMEOUT, sup.RSS_LIMIT_KIB), (1800, 8 * 1024**2))

    def test_success_receipt_logs_and_operation_source_subset(self):
        code, record, frozen, launch, kill = self.execute()
        self.assertEqual((code, record["schema"], record["status"]), (0, "evr0023.supervision.v1", "completed"))
        self.assertEqual(record["thread_environment"], {key: "4" for key in sup.THREAD_KEYS})
        paths = {p.name for p in frozen.call_args.args[1]}
        self.assertEqual(paths, {"supervise.py", "acquire_operation.py", "range_inputs.py", "propose_plan.py"})
        self.assertEqual(launch.call_count, 1); kill.assert_not_called()
        for name in ("stdout", "stderr"):
            self.assertEqual(record[name + "_sha256"], sup.sha256(self.args.output_dir / (name + ".log")))

    def test_all_four_operations_have_their_declared_helpers(self):
        for operation, script in sup.SCRIPTS.items():
            self.args.operation = operation; self.args.output_dir = self.root / operation
            self.set_command(script)
            code, _, frozen, _, _ = self.execute()
            self.assertEqual(code, 0)
            self.assertEqual({p.name for p in frozen.call_args.args[1]},
                             {"supervise.py", script, *sup.HELPERS[operation]})

    def test_wrong_script_or_child_commit_blocks_before_launch(self):
        for script, commit in (("frontend.py", None), ("acquire_operation.py", "b" * 40)):
            self.set_command(script, commit)
            with patch.object(sup.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    sup.run(self.args)
                launch.assert_not_called()
            self.assertFalse(self.args.output_dir.exists())

    def test_frozen_source_failure_and_existing_output_block(self):
        with patch.object(sup, "verify_frozen", side_effect=ValueError("source mismatch")), \
             patch.object(sup.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                sup.run(self.args)
            launch.assert_not_called()
        self.args.output_dir.mkdir()
        with patch.object(sup, "verify_frozen", return_value={}), patch.object(sup.subprocess, "Popen") as launch:
            with self.assertRaises(FileExistsError):
                sup.run(self.args)
            launch.assert_not_called()

    def test_rss_and_timeout_kill_are_recorded(self):
        code, record, _, _, kill = self.execute(FakeProcess(None), rss=sup.RSS_LIMIT_KIB + 1)
        self.assertEqual((code, record["termination"], record["status"]), (1, "group_rss_limit", "failed"))
        self.assertGreaterEqual(kill.call_count, 2)
        self.args.output_dir = self.root / "timeout"
        code, record, _, _, kill = self.execute(FakeProcess(None), clock=[0, 1801, 1802])
        self.assertEqual((code, record["termination"]), (1, "timeout"))
        self.assertGreaterEqual(kill.call_count, 2)

    def test_nonzero_child_is_not_completed(self):
        code, record, _, _, kill = self.execute(FakeProcess(7))
        self.assertEqual((code, record["status"], record["exit_code"]), (1, "failed", 7))
        self.assertGreaterEqual(kill.call_count, 2)


if __name__ == "__main__":
    unittest.main()
