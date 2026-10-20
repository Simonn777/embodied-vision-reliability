"""Artificial admission tests only: no packages, native fixture, Git or network.

The two imported production modules have stdlib-only, inert module bodies.
All installed files below are manufactured text placeholders. Actual Git access
is replaced by a fixed byte snapshot. This file does not execute on import.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import site
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


q = load("evr0029_qualification_under_test", "qualify.py")
h = load("evr0029_installer_for_contract_tests", "install_runtime.py")
COMMIT = "a" * 40


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Chain:
    """A complete tiny on-disk chain; no asset/runtime file is real."""
    def __init__(self, directory):
        self.base = Path(directory).resolve()
        self.root = self.base / "repository"
        self.install = self.base / "installation"
        self.supervision = self.base / "supervision"
        self.root.mkdir()
        self.install.mkdir()
        self.supervision.mkdir()
        self.receipt_path = self.install / "install-receipt.json"
        self.outer_path = self.supervision / "receipt.json"
        self.spec_path = self.base / "install-spec.json"
        required = {h.SELF, *h.MANIFESTS, *h.OLD_MANIFESTS,
                    q.PROTOCOL, q.REVIEW}
        self.snapshot = {}
        for name in required | {q.SUPERVISOR, q.MONITOR}:
            body = (Path(h.__file__).read_bytes() if name == h.SELF
                    else ("manufactured frozen file: " + name + "\n").encode())
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            self.snapshot[name] = body
        for name in ("sources/lightglue", "weights", "venv/bin", "built-wheels"):
            (self.install / name).mkdir(parents=True)
        for name in ("offline_guard.py", "sources/lightglue/local.py",
                     "weights/reused-placeholder.txt", "venv/bin/python",
                     "built-wheels/manufactured.txt"):
            (self.install / name).write_text("inert artificial file: " + name)
        self.receipt = {
            "schema": "evr0029.install.v1", "status": "completed",
            "expected_commit": COMMIT, "model_invocations": 0,
            "scientific_input_access": False,
            "source_sha256": {n: hashlib.sha256(self.snapshot[n]).hexdigest()
                              for n in required},
            "build_review_sha256": h.sha(self.root / q.REVIEW),
            "installed_inventory": h.inventory(self.install),
        }
        flags = [
            "--repo-root", str(self.root),
            "--expected-commit", COMMIT,
            "--protocol", str(self.root / q.PROTOCOL),
            "--acquisition", str(self.base / "acquisition"),
            "--acquisition-supervision", str(self.base / "acquisition-outer.json"),
            "--acquisition-spec", str(self.base / "acquisition-spec.json"),
            "--prior-assets", str(self.base / "prior-assets"),
            "--build-review", str(self.root / q.REVIEW),
            "--destination", str(self.install),
            "--python", "/manufactured/python3.12",
        ]
        self.spec = {
            "commit": COMMIT, "repo": str(self.root),
            "operation_output": str(self.supervision),
            "result_file": str(self.receipt_path), "timeout": 1800,
            "environment": {k: "4" for k in h.THREAD_KEYS},
            "command": ["/manufactured/python3.12", "-B",
                        str(self.root / h.SELF), *flags],
        }
        for name in ("stdout.log", "stderr.log"):
            (self.supervision / name).write_text("manufactured " + name + "\n")
        self.outer = {
            "schema": "evr0026.operation.v1", "status": "completed",
            "exit_code": 0, "termination": None,
            "remaining_descendant_pids": [], "cleanup_errors": [],
            "expected_commit": COMMIT,
            "hard_timeout_seconds": 1800, "group_rss_limit_kib": 8388608,
            "wall_seconds": 1.0, "peak_group_rss_kib": 1024,
            "thread_environment": {k: "4" for k in h.THREAD_KEYS},
            "supervisor_sha256": hashlib.sha256(
                self.snapshot[q.SUPERVISOR]).hexdigest(),
            "monitor_sha256": hashlib.sha256(self.snapshot[q.MONITOR]).hexdigest(),
            "log_sha256": {n: h.sha(self.supervision / n)
                           for n in ("stdout.log", "stderr.log")},
        }
        self.persist()

    def persist(self, bind_command=True):
        dump(self.receipt_path, self.receipt)
        dump(self.spec_path, self.spec)
        self.outer["output_sha256"] = h.sha(self.receipt_path)
        self.outer["spec_sha256"] = h.sha(self.spec_path)
        if bind_command:
            self.outer["command"] = [
                x.replace(str(self.root), "$REPOSITORY")
                 .replace(str(self.root.parent), "$PRIVATE")
                for x in self.spec["command"]
            ]
            self.outer["command"][0] = "$PYTHON"
        dump(self.outer_path, self.outer)

    def git_bytes(self, root, commit, relative):
        if Path(root) != self.root or commit != COMMIT:
            raise AssertionError("unexpected fake Git identity")
        return self.snapshot[relative]

    def verify(self):
        with mock.patch.object(h, "git_bytes", side_effect=self.git_bytes):
            return q.installation_contract(
                h, self.root, COMMIT, self.install, self.outer_path, self.spec_path)


class InstallationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.chain = Chain(self.tmp.name)

    def test_clean_complete_chain_is_accepted(self):
        receipt, bindings = self.chain.verify()
        self.assertEqual(receipt, self.chain.receipt)
        self.assertEqual(bindings["inventory_entries"], 5)
        self.assertEqual(bindings["receipt_sha256"], h.sha(self.chain.receipt_path))

    def test_failed_or_unclean_outer_is_rejected_even_with_valid_hashes(self):
        original = copy.deepcopy(self.chain.outer)
        for patch in ({"status": "failed"}, {"exit_code": 1},
                      {"termination": "timeout"}, {"remaining_descendant_pids": [7]},
                      {"cleanup_errors": ["kill failed"]}):
            with self.subTest(patch=patch):
                self.chain.outer = {**original, **patch}
                self.chain.persist()
                with self.assertRaisesRegex(ValueError, "unclean"):
                    self.chain.verify()

    def test_modified_reused_file_and_unexpected_installed_file_rejected(self):
        path = self.chain.install / "weights/reused-placeholder.txt"
        original = path.read_bytes()
        path.write_bytes(b"changed reused artifact")
        with self.assertRaisesRegex(ValueError, "installed files changed"):
            self.chain.verify()
        path.write_bytes(original)
        (self.chain.install / "sources/lightglue/unexpected.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "installed files changed"):
            self.chain.verify()

    def test_source_bytes_change_rejected_before_inventory(self):
        (self.chain.root / q.PROTOCOL).write_text("changed frozen protocol\n")
        with self.assertRaisesRegex(ValueError, "source differs from freeze"):
            self.chain.verify()

    def test_receipt_cannot_choose_a_smaller_frozen_source_set(self):
        del self.chain.receipt["source_sha256"][q.PROTOCOL]
        self.chain.persist()  # Consistent outer hash does not replace the contract.
        with self.assertRaisesRegex(ValueError, "source set"):
            self.chain.verify()

    def test_missing_unknown_and_duplicate_command_flags_rejected(self):
        original = list(self.chain.spec["command"])
        position = original.index("--acquisition")
        variants = [
            original[:position] + original[position + 2:],
            original + ["--unreviewed-option", "yes"],
            original + ["--destination", str(self.chain.install)],
        ]
        for command in variants:
            with self.subTest(command=command):
                self.chain.spec["command"] = command
                self.chain.persist()
                with self.assertRaisesRegex(ValueError, "flag"):
                    self.chain.verify()

    def test_log_and_outer_command_tamper_rejected(self):
        log = self.chain.supervision / "stdout.log"
        original = log.read_bytes()
        log.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.chain.verify()
        log.write_bytes(original)
        self.chain.outer["command"][2] = "$REPOSITORY/unreviewed-installer.py"
        self.chain.persist(bind_command=False)
        with self.assertRaisesRegex(ValueError, "command identity"):
            self.chain.verify()

    def test_resource_limits_rejected_with_completed_outer(self):
        original = copy.deepcopy(self.chain.outer)
        variants = [
            {"wall_seconds": 1800.1}, {"peak_group_rss_kib": 8388609},
            {"hard_timeout_seconds": 3600},
            {"thread_environment": {k: "8" for k in h.THREAD_KEYS}},
        ]
        for patch in variants:
            with self.subTest(patch=patch):
                self.chain.outer = {**original, **patch}
                self.chain.persist()
                with self.assertRaisesRegex(ValueError, "resource limit"):
                    self.chain.verify()


class FlagProxy:
    def __init__(self, original, no_site):
        self.original, self.no_site = original, no_site

    def __getattr__(self, name):
        return getattr(self.original, name)


class QualificationOrchestrationTests(unittest.TestCase):
    def exercise(self, *, version=(3, 12, 13), no_site=1,
                 executable_ok=True, fixture_status="completed", fail_fixture=False):
        """No real site processing, guard, fixture, interpreter or Git call."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name).resolve()
        install, output = base / "installation", base / "qualification"
        calls = []
        helper = SimpleNamespace(
            THREAD_KEYS=h.THREAD_KEYS, MANIFESTS=h.MANIFESTS,
            OLD_MANIFESTS=h.OLD_MANIFESTS, require=h.require,
            frozen=lambda *args: {}, inventory=lambda path: {})

        def fixture_run(inputs, path, record):
            calls.append("fixture")
            self.assertEqual(path, output / "native-fixture")
            self.assertFalse(path.exists())
            path.mkdir()
            self.assertEqual(inputs["sam2_source"], install / "sources/sam2")
            record["native_fixture"] = {"status": fixture_status,
                                        "warnings": ["manufactured warning"]}
            if fail_fixture:
                raise ValueError("manufactured fixture failure")
            return record["native_fixture"]

        def module(name, path):
            if name == "evr0029_install":
                return helper
            if name == "evr0022_qualification_helpers":
                return SimpleNamespace(install_guard=lambda *args: calls.append("guard"))
            if name == "evr0029_native_fixture":
                return SimpleNamespace(run=fixture_run)
            raise AssertionError("unexpected import: " + name)

        args = SimpleNamespace(
            repo_root=str(base / "repository"), expected_commit=COMMIT,
            installation=str(install), installation_supervision=str(base / "outer.json"),
            installation_spec=str(base / "spec.json"), output_dir=str(output))
        flags = FlagProxy(sys.flags, no_site)
        old_path = list(sys.path)
        error = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(q, "module", side_effect=module))
            stack.enter_context(mock.patch.object(q, "installation_contract",
                return_value=({"installed_inventory": {}}, {"manufactured": True})))
            stack.enter_context(mock.patch.object(q.sys, "version_info", version))
            stack.enter_context(mock.patch.object(q.sys, "flags", flags))
            stack.enter_context(mock.patch.object(q.sys, "executable",
                str(install / "venv/bin/python") if executable_ok else "/wrong/python"))
            stack.enter_context(mock.patch.object(q.sys, "prefix", str(install / "venv")))
            stack.enter_context(mock.patch.object(q.sys, "dont_write_bytecode", True))
            stack.enter_context(mock.patch.dict(os.environ,
                {k: "4" for k in h.THREAD_KEYS}, clear=False))
            stack.enter_context(mock.patch.object(site, "main",
                side_effect=lambda: calls.append("site")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                q.run(args)
            except ValueError as exc:
                error = exc
            finally:
                sys.path[:] = old_path
        return calls, json.loads((output / "receipt.json").read_text()), error

    def test_wrong_interpreter_or_site_startup_rejected_before_guard_and_fixture(self):
        variants = ({"version": (3, 11, 9)}, {"no_site": 0},
                    {"executable_ok": False})
        for options in variants:
            with self.subTest(options=options):
                calls, receipt, error = self.exercise(**options)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(calls, [])
                self.assertEqual(receipt["status"], "failed")
                self.assertFalse(receipt["eligible"])

    def test_guard_precedes_site_and_fixture_uses_fresh_subdirectory(self):
        calls, receipt, error = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(calls, ["guard", "site", "fixture"])
        self.assertEqual(receipt["status"], "completed")
        self.assertTrue(receipt["eligible"])
        self.assertFalse(receipt["scientific_input_access"])

    def test_fixture_exception_preserves_warning_and_failed_receipt(self):
        calls, receipt, error = self.exercise(fail_fixture=True)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(calls, ["guard", "site", "fixture"])
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["eligible"])
        self.assertEqual(receipt["native_fixture"]["warnings"], ["manufactured warning"])
        self.assertEqual(receipt["error"], "manufactured fixture failure")

    def test_returned_incomplete_fixture_cannot_become_eligible(self):
        _, receipt, error = self.exercise(fixture_status="failed")
        self.assertIsInstance(error, ValueError)
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["eligible"])


if __name__ == "__main__":
    unittest.main()
