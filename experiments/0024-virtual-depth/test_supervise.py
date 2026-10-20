"""Artificial processes and source bytes only; never import the scientific probe."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "evr0024_supervise_artificial", Path(__file__).with_name("supervise.py"))
sup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sup
spec.loader.exec_module(sup)


class FakeProcess:
    pid = 123456

    def __init__(self, code=0):
        self.returncode = code
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        self.returncode = -9 if self.returncode is None else self.returncode
        return self.returncode


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evr0024-supervision-fake-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(operation="probe", expected_commit="a" * 40,
                                    output_dir=self.root / "output", command_json="")
        self.set_command()

    def set_command(self, script="run.py", extra=()):
        self.args.command_json = json.dumps([
            str(sup.ROOT / "experiments/0024-virtual-depth" / script),
            "--expected-commit", self.args.expected_commit, *extra])

    def execute(self, process=None, rss=100, clock=None, sample=None, launch_error=None):
        process = process or FakeProcess()
        rows = [(process.pid, 1, process.pid, rss)]
        with patch.object(sup, "verify_frozen", return_value={"fake": "bound"}) as frozen, \
                patch.object(sup.subprocess, "Popen", return_value=process,
                             side_effect=launch_error) as launch, \
                patch.object(sup, "snapshot", side_effect=sample,
                             return_value=(rows, {process.pid})), \
                patch.object(sup.os, "killpg") as kill, \
                patch.object(sup.time, "sleep"), \
                patch.object(sup.time, "perf_counter", side_effect=clock or [0, .01, .02]):
            code = sup.run(self.args)
        receipt = json.loads((self.args.output_dir / "receipt.json").read_text())
        return code, receipt, frozen, launch, kill

    def test_inherited_resource_cleanup_and_helper_code_is_unchanged(self):
        original = (sup.ROOT / "experiments/0023-mapfree/supervise.py").read_text()
        current = Path(sup.__file__).read_text()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),
                         "98a83c732a88e2cfd46842c14353b3c0e73744d55969bbc7d8eacc866b4dec30")
        self.assertEqual(original.split("def sha256", 1)[1].split("def run", 1)[0],
                         current.split("def sha256", 1)[1].split("def run", 1)[0])
        start, end = "    began = time.perf_counter()\n", "\ndef main():"
        self.assertEqual(original.split(start, 1)[1].split(end, 1)[0],
                         current.split(start, 1)[1].split(end, 1)[0])
        self.assertEqual((sup.TIMEOUT, sup.RSS_LIMIT_KIB), (1800, 8 * 1024**2))

    def test_success_binds_math_and_exact_protocol_and_hashes_logs(self):
        code, receipt, frozen, launch, kill = self.execute()
        self.assertEqual((code, receipt["schema"], receipt["status"]),
                         (0, "evr0024.supervision.v1", "completed"))
        paths = set(frozen.call_args.args[1])
        self.assertEqual(paths, {Path(sup.__file__), sup.PROTOCOL,
                               Path(sup.__file__).with_name("run.py"),
                               Path(sup.__file__).with_name("math_core.py")})
        self.assertEqual(receipt["thread_environment"], {key: "4" for key in sup.THREAD_KEYS})
        self.assertEqual(launch.call_count, 1)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(launch.call_args.kwargs["env"]["PYTHONPATH"], "")
        kill.assert_not_called()
        for name in ("stdout", "stderr"):
            self.assertEqual(receipt[name + "_sha256"],
                             sup.sha256(self.args.output_dir / (name + ".log")))

    def test_public_command_and_exception_remove_local_path_prefixes(self):
        self.set_command(extra=("--private-asset", str(sup.ROOT.parent / "assets/private")))
        _, receipt, _, _, _ = self.execute(
            launch_error=OSError("cannot open " + str(sup.ROOT / "private-name")))
        shown = json.dumps({key: receipt[key] for key in ("command", "termination")})
        self.assertNotIn(str(sup.ROOT.parent), shown)
        self.assertNotIn(str(sys.executable), shown)
        self.assertIn("$PRIVATE/assets/private", shown)
        self.assertIn("$PYTHON", shown)
        self.assertIn("$REPOSITORY", shown)

    def test_wrong_script_missing_duplicate_or_different_commit_block_before_launch(self):
        bad = [
            [str(Path(sup.__file__).with_name("math_core.py")), "--expected-commit", "a" * 40],
            [str(Path(sup.__file__).with_name("run.py"))],
            [str(Path(sup.__file__).with_name("run.py")), "--expected-commit", "b" * 40],
            [str(Path(sup.__file__).with_name("run.py")), "--expected-commit", "a" * 40,
             "--expected-commit=" + "a" * 40],
        ]
        with patch.object(sup.subprocess, "Popen") as launch:
            for command in bad:
                self.args.command_json = json.dumps(command)
                with self.assertRaises(ValueError):
                    sup.run(self.args)
            launch.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_protocol_byte_change_rejected_by_real_frozen_verifier_before_launch(self):
        repository = (self.root / "repository").resolve()
        protocol = repository / "protocols/0024-virtual-rotation-falsifier.md"
        protocol.parent.mkdir(parents=True)
        protocol.write_bytes(b"tampered threshold\n")
        with patch.object(sup, "ROOT", repository), \
                patch.object(sup.subprocess, "check_output", side_effect=["a" * 40, b"frozen threshold\n"]), \
                patch.object(sup.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "source differs"):
                sup.verify_frozen("a" * 40, [protocol])
            launch.assert_not_called()

    def test_source_fault_or_used_output_never_launches(self):
        with patch.object(sup, "verify_frozen", side_effect=ValueError("source mismatch")), \
                patch.object(sup.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                sup.run(self.args)
            launch.assert_not_called()
        self.args.output_dir.mkdir()
        with patch.object(sup, "verify_frozen", return_value={}), \
                patch.object(sup.subprocess, "Popen") as launch:
            with self.assertRaises(FileExistsError):
                sup.run(self.args)
            launch.assert_not_called()

    def test_timeout_kills_retained_isolated_group_and_waits(self):
        process = FakeProcess(None)
        rows = [(process.pid, 1, process.pid, 10), (123457, process.pid, 123457, 20)]
        sample = [(rows, {process.pid, 123457}), (rows, {process.pid, 123457}), ([], set())]
        code, receipt, _, _, kill = self.execute(process, clock=[0, 1800, 1800.1], sample=sample)
        self.assertEqual((code, receipt["termination"], receipt["status"]), (1, "timeout", "failed"))
        self.assertIn(((process.pid, signal.SIGSTOP),), [(call.args,) for call in kill.call_args_list])
        killed = {call.args[0] for call in kill.call_args_list if call.args[1] == signal.SIGKILL}
        self.assertEqual(killed, {process.pid, 123457})
        self.assertEqual(process.waits, [5])
        self.assertEqual(receipt["remaining_descendant_pids"], [])

    def test_rss_nonzero_exit_and_late_timeout_cannot_complete(self):
        for label, process, rss, clock, expected in (
                ("rss", FakeProcess(None), sup.RSS_LIMIT_KIB + 1, [0, .1], "group_rss_limit"),
                ("exit", FakeProcess(7), 10, [0, .01, .02], None),
                ("late", FakeProcess(0), 10, [0, .01, 1800.01], "timeout_at_exit")):
            self.args.output_dir = self.root / label
            code, receipt, _, _, _ = self.execute(process, rss=rss, clock=clock)
            self.assertEqual((code, receipt["status"], receipt["termination"]), (1, "failed", expected))

    def test_successful_parent_with_live_descendant_is_failed_and_cleaned(self):
        process = FakeProcess(0)
        rows = [(123457, 1, 123457, 20)]
        sample = [(rows, {123457}), (rows, {123457}), ([], set())]
        code, receipt, _, _, kill = self.execute(process, sample=sample)
        self.assertEqual((code, receipt["termination"]), (1, "descendants_remain_after_parent_exit"))
        self.assertIn(123457, {call.args[0] for call in kill.call_args_list})

    def test_launch_exception_is_preserved_with_empty_log_hashes(self):
        code, receipt, _, _, kill = self.execute(launch_error=OSError("manufactured launch failure"), clock=[0, .1])
        self.assertEqual((code, receipt["status"], receipt["exit_code"]), (1, "failed", None))
        self.assertIn("manufactured launch failure", receipt["termination"])
        self.assertEqual(receipt["stdout_sha256"], hashlib.sha256(b"").hexdigest())
        kill.assert_not_called()

    def test_interrupt_saves_failed_receipt_then_reraises(self):
        process = FakeProcess(None)
        with patch.object(sup, "verify_frozen", return_value={}), \
                patch.object(sup.subprocess, "Popen", return_value=process), \
                patch.object(sup, "snapshot", side_effect=[KeyboardInterrupt(), ([], set()), ([], set())]), \
                patch.object(sup.os, "killpg") as kill, \
                patch.object(sup.time, "perf_counter", side_effect=[0, .1]):
            with self.assertRaises(KeyboardInterrupt):
                sup.run(self.args)
        receipt = json.loads((self.args.output_dir / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertTrue(receipt["termination"].startswith("KeyboardInterrupt"))
        self.assertEqual(process.waits, [5])
        self.assertGreaterEqual(kill.call_count, 2)

    def test_snapshot_keeps_reparented_known_group_and_finds_nested_groups(self):
        text = "100 1 100 20\n101 100 101 30\n102 101 101 40\n103 1 103 50\n200 1 200 99\n"
        with patch.object(sup.subprocess, "run", return_value=SimpleNamespace(stdout=text)), \
                patch.object(sup.os, "getpgrp", return_value=999):
            rows, groups = sup.snapshot(100, {103})
        self.assertEqual({row[0] for row in rows}, {100, 101, 102, 103})
        self.assertEqual(sum(row[3] for row in rows), 140)
        self.assertEqual(groups, {100, 101, 103})


if __name__ == "__main__":
    unittest.main()
