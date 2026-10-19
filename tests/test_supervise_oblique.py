"""Artificial supervisor files/processes only; no model, dataset or cloud call."""

from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "scripts/supervise_oblique.py"
SPEC = importlib.util.spec_from_file_location("oblique_supervisor_test", PATH)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)
FREEZE_SHA = "f" * 64
SOURCE_COMMIT = "1" * 40


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EnvironmentAndCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.base = self.root / "cache"
        self.output = self.base / "scratch/attempt"
        self.config = {"python": "/fixed/python", "asset_receipt_sha256": "a" * 64}
        path = self.repo / "protocols/0011-oblique-coupling-development.md"
        path.parent.mkdir(parents=True)
        path.write_text("Artificial frozen protocol; no scientific run.\n")

    def test_environment_is_explicit_whitelist_with_no_credential_or_home_inheritance(self):
        private = {"HOME": "/private/fake-home", "GH_TOKEN": "fake-github-token",
                   "OPENAI_API_KEY": "fake-key", "AWS_SECRET_ACCESS_KEY": "fake-secret",
                   "HTTPS_PROXY": "http://fake-proxy", "PYTHONPATH": "/untrusted/path",
                   "OMP_NUM_THREADS": "128", "UNRELATED_TEST_VARIABLE": "fake-private-value"}
        with patch.dict(os.environ, private, clear=True):
            env = supervisor.clean_environment(self.repo, self.base)
        expected_keys = {"PATH", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE",
                         "PYTHONNOUSERSITE", "PYTHONHASHSEED", "HF_HUB_OFFLINE",
                         "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES", "PYTHONPATH",
                         "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"}
        self.assertEqual(set(env), expected_keys)
        self.assertTrue(all(value not in json.dumps(env) for value in private.values()))
        self.assertNotIn("HOME", env)
        self.assertEqual(env["PATH"], "/usr/bin:/bin")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(env[name], "4")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [
            str(self.base / "tapnextpp-cache/overlay"),
            str(self.base / "bootstapir-cache/runtime-fixed"),
            str(self.base / "bootstapir-cache/runtime"),
            str(self.repo / "src"), str(self.repo / "scripts")])
        # Offline library flags do not establish OS-level network isolation.
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def prepared(self, *, status="completed_preparation"):
        prep = self.output / "prepared"
        prep.mkdir(parents=True)
        files = {}
        for group in ("g-036", "g-002"):
            path = prep / (group + ".predictor.npz")
            path.write_bytes(b"artificial command fixture; not an NPZ or video")
            files[path.name] = {"sha256": supervisor.sha(path)}
        json_file(prep / "receipt.json", {"status": status, "files": files})
        return prep

    def test_phase_commands_bind_exact_paths_hashes_and_no_shell(self):
        prep = self.prepared()
        command = supervisor.command_for("prepare", self.repo, self.base, self.output, self.config)
        self.assertEqual(command, ["/fixed/python", str(self.repo / "scripts/prepare_oblique_development.py"),
                                  "--archive", str(self.base / "bootstapir-cache/rgb.zip"),
                                  "--manifest", str(self.repo / "results/0002-preflight/identity/manifest.json"),
                                  "--output", str(prep)])
        for group in ("g-036", "g-002"):
            command = supervisor.command_for(group, self.repo, self.base, self.output, self.config)
            self.assertEqual(command[:2], ["/fixed/python", str(self.repo / "scripts/run_oblique_predictions.py")])
            options = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(options, {
                "--prepared-input": str(prep / (group + ".predictor.npz")),
                "--input-sha256": supervisor.sha(prep / (group + ".predictor.npz")),
                "--group-id": group, "--output": str(self.output / "predictions" / group),
                "--source-root": str(self.base / "tapnextpp-recovery/source"),
                "--checkpoint": str(self.base / "tapnextpp-recovery/weights/tapnextpp_ckpt.pt"),
                "--overlay-root": str(self.base / "tapnextpp-cache/overlay"),
                "--base-runtime-root": str(self.base / "bootstapir-cache/runtime-fixed"),
                "--asset-receipt": str(self.base / "scratch/0005-recovered-native-capability/run/receipt.json"),
                "--asset-receipt-sha256": "a" * 64,
                "--protocol-sha256": supervisor.sha(self.repo / "protocols/0011-oblique-coupling-development.md"),
            })
        for phase, input_dir, result_dir in (("derive", "predictions", "derived"),
                                             ("score", "derived", "evaluation")):
            self.assertEqual(supervisor.command_for(phase, self.repo, self.base, self.output, self.config),
                             ["/fixed/python", str(self.repo / "scripts/evaluate_oblique_saved.py"), phase,
                              "--input", str(self.output / input_dir), "--preparation", str(prep),
                              "--output", str(self.output / result_dir)])

    def test_preparation_failure_or_changed_predictor_prevents_command(self):
        prep = self.prepared(status="failed")
        with self.assertRaisesRegex(ValueError, "not completed"):
            supervisor.command_for("g-036", self.repo, self.base, self.output, self.config)
        receipt = supervisor.read(prep / "receipt.json")
        receipt["status"] = "completed_preparation"
        json_file(prep / "receipt.json", receipt)
        (prep / "g-036.predictor.npz").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed"):
            supervisor.command_for("g-036", self.repo, self.base, self.output, self.config)

    def test_machine_config_deadlines_and_native_pins_match_implementations(self):
        config = supervisor.read(REPO / "configs/0011-oblique-development.json")
        self.assertEqual(supervisor.PHASES, ("prepare", "g-036", "g-002", "derive", "score"))
        self.assertEqual(supervisor.LIMITS, (300, 900, 900, 1800, 1800))
        self.assertEqual(config["phase_seconds"], dict(zip(supervisor.PHASES, supervisor.LIMITS)))
        self.assertEqual(config["threads"], 4)
        self.assertEqual(config["address_space_bytes"], supervisor.MEMORY)
        self.assertEqual(config["new_downloads"], 0)
        spec = importlib.util.spec_from_file_location("oblique_native_pin_test", REPO / "scripts/run_oblique_predictions.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        self.assertEqual(config["source_sha256"], native.SOURCES)
        self.assertEqual(config["checkpoint_sha256"], native.WEIGHT_SHA)
        self.assertEqual(config["checkpoint_bytes"], native.WEIGHT_BYTES)
        self.assertEqual(config["packages"], native.PACKAGES)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        names = getattr(supervisor, "REQUIRED_FILES", ("scripts/artificial-source.py",))
        files = {}
        for name in names:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Artificial source-binding fixture.\n")
            files[name] = supervisor.sha(path)
        self.payload = {"source_commit": SOURCE_COMMIT, "files": files}
        self.freeze = self.repo / "protocols/0011-oblique-development-freeze.json"
        json_file(self.freeze, self.payload)

    def test_freeze_and_each_bound_file_are_verified(self):
        result = supervisor.verify_freeze(self.repo, supervisor.sha(self.freeze))
        self.assertEqual(result, self.payload)
        with self.assertRaisesRegex(ValueError, "freeze receipt differs"):
            supervisor.verify_freeze(self.repo, "0" * 64)
        path = self.repo / next(iter(self.payload["files"]))
        path.write_text("Modified after freezing.\n")
        with self.assertRaisesRegex(ValueError, "source differs"):
            supervisor.verify_freeze(self.repo, supervisor.sha(self.freeze))

    def test_symlink_bound_source_and_escape_are_rejected(self):
        path = self.repo / next(iter(self.payload["files"]))
        original = path.read_bytes()
        target = self.repo / "other-file"
        target.write_bytes(original)
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "source differs"):
            supervisor.verify_freeze(self.repo, supervisor.sha(self.freeze))
        self.payload["files"] = {"../outside-source.py": "0" * 64}
        json_file(self.freeze, self.payload)
        with self.assertRaises(ValueError):
            supervisor.verify_freeze(self.repo, supervisor.sha(self.freeze))

    def test_missing_required_source_and_invalid_commit_fail_before_launch(self):
        for files, commit in (({}, SOURCE_COMMIT),
                              ({k: v for k, v in self.payload["files"].items()
                                if k != supervisor.REQUIRED_FILES[0]}, SOURCE_COMMIT),
                              (self.payload["files"], None),
                              (self.payload["files"], "not-a-commit")):
            with self.subTest(commit=commit, file_count=len(files)):
                json_file(self.freeze, {"files": files, "source_commit": commit})
                with self.assertRaisesRegex(ValueError, "incomplete freeze"):
                    supervisor.verify_freeze(self.repo, supervisor.sha(self.freeze))


class MainGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.base = self.root / "base"
        self.output = self.base / "scratch/attempt"
        self.output.mkdir(parents=True)
        self.config = {"cloud_base": str(self.base), "output_relative": "scratch/attempt",
                       "python": "/unused/fake-python", "asset_receipt_sha256": "a" * 64}
        json_file(self.repo / "configs/0011-oblique-development.json", self.config)

    def invoke(self, phase, fake_supervise):
        with ExitStack() as stack:
            stack.enter_context(patch.object(supervisor, "__file__", str(self.repo / "scripts/supervise_oblique.py")))
            stack.enter_context(patch.object(supervisor, "verify_freeze", return_value={"source_commit": SOURCE_COMMIT}))
            stack.enter_context(patch.object(supervisor, "command_for", return_value=["/unused/fake-child"]))
            stack.enter_context(patch.object(supervisor, "supervise", side_effect=fake_supervise))
            stack.enter_context(patch.object(sys, "argv", ["supervisor", phase, "--freeze-sha256", FREEZE_SHA]))
            return supervisor.main()

    def prior(self, phase, *, exit_code=0):
        directory = self.output / "supervision" / phase
        record = {"supervisor_exit_code": exit_code, "child_exit_code": exit_code}
        json_file(directory / "execution.json", record)
        json_file(directory / "source-binding.json", {
            "freeze_sha256": FREEZE_SHA, "source_commit": SOURCE_COMMIT,
            "execution_sha256": supervisor.sha(directory / "execution.json")})
        return directory

    def test_prior_failure_prevents_next_phase_without_launch(self):
        self.prior("prepare", exit_code=124)
        with self.assertRaisesRegex(ValueError, "prior phase failed"):
            self.invoke("g-036", lambda *args, **kwargs: self.fail("must not launch child"))

    def test_prior_missing_receipt_prevents_launch(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.invoke("g-036", lambda *args, **kwargs: self.fail("must not launch child"))

    def test_existing_prepare_root_cannot_be_reused(self):
        sentinel = self.output / "existing-result"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.invoke("prepare", lambda *args, **kwargs: self.fail("must not launch child"))
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_prior_binding_hash_and_freeze_identity_gate_continuation(self):
        directory = self.prior("prepare")
        supervisor.verify_prior_phases(self.output, "g-036", FREEZE_SHA)
        execution = directory / "execution.json"
        record = supervisor.read(execution)
        record["new_unbound_field"] = "modified after completion"
        json_file(execution, record)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            self.invoke("g-036", lambda *args, **kwargs: self.fail("must not launch child"))
        directory = self.prior("prepare")
        binding = supervisor.read(directory / "source-binding.json")
        binding["freeze_sha256"] = "e" * 64
        json_file(directory / "source-binding.json", binding)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            supervisor.verify_prior_phases(self.output, "g-036", FREEZE_SHA)
        (directory / "source-binding.json").unlink()
        with self.assertRaises(FileNotFoundError):
            supervisor.verify_prior_phases(self.output, "g-036", FREEZE_SHA)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "control"
        self.usage = SimpleNamespace(ru_maxrss=1234, ru_utime=.1, ru_stime=.2)

    def simulate(self, waits, *, timeout=False, launch_failure=None, interrupt_after_reap=False):
        process = SimpleNamespace(pid=424242, returncode=None)
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "platform", "linux"))
            stack.enter_context(patch.object(os, "sched_getaffinity", return_value={4, 5, 6, 7, 8}, create=True))
            waited = stack.enter_context(patch.object(os, "wait4", side_effect=waits))
            killed = stack.enter_context(patch.object(os, "killpg"))
            stack.enter_context(patch.object(supervisor.subprocess, "Popen", side_effect=launch_failure,
                                             return_value=process))
            stack.enter_context(patch.object(supervisor.signal, "signal", return_value=0))
            mask_calls = 0

            def mask_signal(*args):
                nonlocal mask_calls
                mask_calls += 1
                if interrupt_after_reap and mask_calls == 4:
                    raise InterruptedError("pending signal after atomic reap")
                return set()

            stack.enter_context(patch.object(supervisor.signal, "pthread_sigmask", side_effect=mask_signal))
            stack.enter_context(patch.object(supervisor.time, "sleep"))
            # A large deterministic increment reaches any fixed deadline/grace.
            clock = iter([1000 + (400 if timeout else .01) * i for i in range(40)])
            stack.enter_context(patch.object(supervisor.time, "monotonic", side_effect=lambda: next(clock)))
            result = supervisor.supervise(["/fake/python", "-c", "pass"], self.output,
                                           cwd=self.root, env={"PATH": "/usr/bin:/bin"}, seconds=300)
            if interrupt_after_reap:
                self.assertEqual(waited.call_count, 1, "a reaped child must not be waited twice")
        return result, killed, process

    def test_success_receipt_wait4_resources_and_exclusive_logs(self):
        receipt, killed, process = self.simulate([(424242, 0, self.usage)])
        self.assertEqual(receipt["supervisor_exit_code"], 0)
        self.assertEqual(receipt["child_exit_code"], 0)
        self.assertEqual(receipt["cpu_affinity"], [4, 5, 6, 7])
        self.assertEqual(receipt["memory_limit_bytes"], 8 * 1024**3)
        self.assertEqual(receipt["max_rss_kib"], 1234)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(set(receipt["logs"]), {"stdout.log", "stderr.log"})
        self.assertTrue((self.output / "started.json").is_file())
        self.assertTrue((self.output / "execution.json").is_file())
        killed.assert_called_with(424242, supervisor.signal.SIGKILL)
        with self.assertRaises(FileExistsError):
            self.simulate([(424242, 0, self.usage)])

    def test_timeout_cleanup_keeps_partial_execution_receipt(self):
        receipt, killed, process = self.simulate(
            [(0, 0, None), (424242, 9, self.usage)], timeout=True)
        self.assertTrue(receipt["timed_out"])
        self.assertEqual(receipt["supervisor_exit_code"], 124)
        self.assertEqual(receipt["child_exit_code"], -9)
        self.assertEqual(process.returncode, -9)
        killed.assert_called_with(424242, supervisor.signal.SIGKILL)
        self.assertTrue((self.output / "execution.json").is_file())

    def test_interrupted_wait_and_launch_failure_are_recorded(self):
        receipt, _, _ = self.simulate([InterruptedError("artificial interrupt"), (424242, 9, self.usage)])
        self.assertEqual(receipt["supervisor_exit_code"], 125)
        self.assertIn("artificial interrupt", receipt["interrupted"])
        self.output = self.root / "launch-failure"
        receipt, killed, _ = self.simulate([], launch_failure=OSError("artificial exec failure"))
        self.assertEqual(receipt["supervisor_exit_code"], 125)
        self.assertEqual(receipt["launch_error"]["type"], "OSError")
        self.assertIsNone(receipt["child_exit_code"])
        killed.assert_not_called()

    def test_pending_signal_after_reap_retains_status_and_execution_receipt(self):
        receipt, _, process = self.simulate([(424242, 0, self.usage)], interrupt_after_reap=True)
        self.assertEqual(receipt["supervisor_exit_code"], 125)
        self.assertEqual(receipt["child_exit_code"], 0)
        self.assertEqual(process.returncode, 0)
        self.assertIn("pending signal", receipt["interrupted"])
        self.assertTrue((self.output / "execution.json").is_file())

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "sched_getaffinity")
                         and len(os.sched_getaffinity(0)) >= 4, "requires Linux and four permitted CPUs")
    def test_tiny_real_linux_child_checks_actual_limits_without_scientific_assets(self):
        code = (
            "import os,resource; "
            "assert len(os.sched_getaffinity(0))==4; "
            "assert resource.getrlimit(resource.RLIMIT_AS)==(8589934592,8589934592); "
            "assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0); "
            "assert 'HOME' not in os.environ; assert 'GH_TOKEN' not in os.environ; "
            "assert os.environ['OMP_NUM_THREADS']=='4'; print('artificial child only')"
        )
        receipt = supervisor.supervise([sys.executable, "-c", code], self.output,
                                      cwd=self.root, env=supervisor.clean_environment(self.root, self.root),
                                      seconds=300)
        self.assertEqual(receipt["supervisor_exit_code"], 0)
        self.assertIn("artificial child only", (self.output / "stdout.log").read_text())


if __name__ == "__main__":
    unittest.main()
