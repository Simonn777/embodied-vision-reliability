"""Two manufactured process cases; no ML/data/network."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
MONITOR_PATH = Path("experiments/0025c-rs-noise/supervise.py")
if "EVR_EXIT_TEST_REPO" in os.environ:
    ROOT = Path(os.environ["EVR_EXIT_TEST_REPO"]).resolve()
else:
    # Published layout: R/experiments/0029-memory-information. Private layout:
    # P/work/0029-preparation, alongside P/work/embodied-vision-reliability.
    choices = (HERE.parents[1], HERE.parent / "embodied-vision-reliability")
    ROOT = next((path.resolve() for path in choices if (path / MONITOR_PATH).is_file()), None)
if ROOT is None or not (ROOT / MONITOR_PATH).is_file():
    raise RuntimeError("expected unchanged repository monitor was not found")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


supervisor = module("new_exit_checked_supervisor", HERE / "supervise_exit_checked.py")


class ExitOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicit fresh artifact root: retain the actual process receipts,
        # logs and sampled race witness even if an assertion later fails.
        if "EVR_EXIT_TEST_OUTPUT" in os.environ:
            cls.artifacts = Path(os.environ["EVR_EXIT_TEST_OUTPUT"]).absolute()
        else:
            # Preserve artifacts outside R in local runs and CI alike.
            retained = Path(tempfile.mkdtemp(prefix="exit-supervisor-tests-", dir=ROOT.parent))
            cls.artifacts = retained / "process-cases"
        if cls.artifacts.resolve() == ROOT or ROOT in cls.artifacts.resolve().parents:
            raise ValueError("test artifacts must remain outside the repository")
        if cls.artifacts.exists() or cls.artifacts.is_symlink():
            raise FileExistsError(cls.artifacts)
        cls.artifacts.mkdir(parents=True, exist_ok=False)

    def setUp(self):
        self.directory = self.artifacts / self._testMethodName
        self.directory.mkdir(exist_ok=False)
        self.original = supervisor.load_monitor(ROOT)

    def spec(self, script, output):
        path = self.directory / (output + "-spec.json")
        path.write_text(json.dumps(dict(repo=str(ROOT), commit="0" * 40, timeout=4,
            environment={key: "4" for key in self.original.THREAD_KEYS},
            command=["/bin/sh", "-c", script], operation_output=str(self.directory / output))))
        return path

    def test_stale_pre_poll_descendant_does_not_become_post_exit_leak(self):
        fifo, pidfile = self.directory / "release.fifo", self.directory / "child.pid"
        os.mkfifo(fifo)
        # Fail closed instead of shell-escaping arbitrary externally supplied
        # paths. The prescribed private artifact path contains no quotes.
        self.assertNotIn("'", str(self.directory))
        self.assertNotIn('"', str(self.directory))
        script = "/bin/sh -c 'read item < \"%s\"' & echo $! > \"%s\"; wait $!; exit 0" % (fifo, pidfile)
        forced = []

        def paused_snapshot(parent, groups):
            rows, discovered = self.original.snapshot(parent, groups)
            if not forced and pidfile.exists():
                descendant = int(pidfile.read_text())
                if any(row[0] == descendant for row in rows):
                    # This snapshot is truthful when captured. Permit the
                    # grandchild AND parent to finish before returning it.
                    forced.append(dict(parent=parent, descendant=descendant,
                                       captured_rows=[list(row) for row in rows]))
                    descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                    try:
                        os.write(descriptor, b"release\n")
                    finally:
                        os.close(descriptor)
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline:
                        probe = subprocess.run(["/bin/ps", "-p", str(parent), "-o", "stat="],
                            capture_output=True, text=True, timeout=.5, check=False)
                        if probe.returncode == 1 or probe.stdout.strip().startswith("Z"):
                            forced[0]["parent_exit_observed_before_snapshot_return"] = True
                            break
                        time.sleep(.005)
                    else:
                        raise AssertionError("manufactured parent did not exit within one second")
            return rows, discovered

        monitor = SimpleNamespace(**{name: getattr(self.original, name)
            for name in ("THREAD_KEYS", "display", "sha256", "save")}, snapshot=paused_snapshot)
        record = None
        try:
            record = supervisor.run_spec(self.spec(script, "race"), monitor=monitor)
        finally:
            proof = dict(scope="Manufactured process history only; no historical cat-girl inference",
                         forced_snapshots=forced, returned_record=record)
            if record is not None and record["pre_poll_snapshot"] is not None:
                proof["old_predicate_would_fail"] = (
                    record["exit_code"] is not None and
                    any(r[0] != record["child_pid"] for r in
                        record["pre_poll_snapshot"]["rows"]))
            (self.directory / "forced-snapshot-proof.json").write_text(
                json.dumps(proof, indent=2, allow_nan=False) + "\n")
        self.assertEqual(len(forced), 1)
        self.assertTrue(forced[0]["parent_exit_observed_before_snapshot_return"])
        self.assertEqual(record["exit_code"], 0)
        self.assertTrue(any(r[0] != record["child_pid"] for r in record["pre_poll_snapshot"]["rows"]))
        # Exactly the frozen old line-18 predicate would classify this sample
        # as descendants_remain. This is a manufactured counterexample, not
        # proof that cat-girl contained this process history.
        self.assertEqual(record["post_exit_snapshot"]["rows"], [])
        self.assertEqual(record["status"], "completed")
        self.assertIsNone(record["termination"])
        self.assertEqual(record["signal_attempts"], [])
        self.assertLess(record["wall_seconds"], 4)

    def test_genuinely_live_post_exit_child_is_rejected_and_signalled(self):
        pidfile = self.directory / "leaked.pid"
        self.assertNotIn('"', str(self.directory))
        script = '/bin/sleep 3 & echo $! > "%s"; exit 0' % pidfile
        record = supervisor.run_spec(self.spec(script, "leak"))
        leaked = int(pidfile.read_text())
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["termination"], "descendants_remain_after_fresh_exit_sample")
        self.assertIn(leaked, [r[0] for r in record["post_exit_snapshot"]["rows"]])
        self.assertEqual(record["triggering_snapshot"], record["post_exit_snapshot"])
        self.assertTrue(any(r["signal"] == "SIGKILL" and r["result"] == "sent"
                            for r in record["signal_attempts"]))
        self.assertLess(record["wall_seconds"], 4)


if __name__ == "__main__":
    unittest.main()
