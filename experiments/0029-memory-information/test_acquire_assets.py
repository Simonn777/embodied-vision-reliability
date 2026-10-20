"""Offline wrapper checks only. No network or scientific payload."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "acquire_assets", Path(__file__).with_name("acquire_assets.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
COMMIT = "a" * 40

FAKE_HELPER = b"""
import json
from pathlib import Path
def validate(manifest):
    if manifest.get("invalid"):
        raise ValueError("manufactured invalid manifest")
    return 128
def acquire_all(manifest, root):
    root = Path(root)
    root.mkdir()
    result = {"status": "failed" if manifest.get("fail") else "completed",
              "actual_body_bytes": 17, "assets": [], "requests": []}
    (root / "ledger.json").write_text(json.dumps(result))
    if manifest.get("fail"):
        if manifest["fail"] == "KeyboardInterrupt":
            raise KeyboardInterrupt("manufactured interruption")
        raise RuntimeError("manufactured batch failure")
    return result
"""


class WrapperChecks(unittest.TestCase):
    def test_completed_batches_recheck_sources_and_account_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = argparse.Namespace(repo_root=str(root), expected_commit=COMMIT,
                                      outputdir=str(root / "output"))
            bound = {m.MANIFESTS[0]: b"{}", m.MANIFESTS[1]: b"{}",
                     m.HELPER: FAKE_HELPER, m.PROTOCOL: b"synthetic protocol"}
            with patch.object(m, "frozen_inputs", return_value=bound) as proof:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = m.run(args)
            self.assertEqual(proof.call_count, 2)
            self.assertEqual(proof.call_args_list[0], proof.call_args_list[1])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["actual_body_bytes"], 34)
            self.assertEqual(result["body_cap_bytes"], 256)
            self.assertEqual([b["status"] for b in result["batches"]], ["completed", "completed"])
            self.assertEqual(json.loads((root / "output/operation.json").read_text()), result)
            for batch in result["batches"]:
                raw = (root / "output" / batch["directory"] / "ledger.json").read_bytes()
                self.assertEqual(batch["ledger_sha256"], hashlib.sha256(raw).hexdigest())

    def test_invalid_second_manifest_prevents_both_acquisitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = argparse.Namespace(repo_root=str(root), expected_commit=COMMIT,
                                      outputdir=str(root / "output"))
            bound = {m.MANIFESTS[0]: b"{}", m.MANIFESTS[1]: b'{"invalid":true}',
                     m.HELPER: FAKE_HELPER, m.PROTOCOL: b"synthetic protocol"}
            with patch.object(m, "frozen_inputs", return_value=bound):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "invalid manifest"):
                    m.run(args)
            receipt = json.loads((root / "output/operation.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["batches"], [])
            self.assertTrue(all(not (root / "output" / d).exists() for d in m.DIRECTORIES))

    def test_source_mismatch_precedes_helper_execution_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            own = root / "acquire_assets.py"
            own.write_text("# synthetic wrapper source\n")
            for relative in (*m.MANIFESTS, m.PROTOCOL, m.HELPER):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("raise AssertionError('must not execute')\n")
            args = argparse.Namespace(repo_root=str(root), expected_commit=COMMIT,
                                      outputdir=str(root / "output"))
            def git(command, **kwargs):
                return COMMIT + "\n" if command[1] == "rev-parse" else b"different frozen bytes"
            with patch.object(m, "__file__", str(own)), patch.object(m.subprocess, "check_output", side_effect=git):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Frozen source mismatch"):
                    m.run(args)
            receipt = json.loads((root / "output/operation.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["error_type"], "ValueError")
            self.assertEqual(receipt["batches"], [])
            self.assertGreaterEqual(receipt["wall_seconds"], 0)

    def test_completed_and_failed_ledgers_survive_exception_or_interrupt(self):
        for error_type in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error_type.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                args = argparse.Namespace(repo_root=str(root), expected_commit=COMMIT,
                                          outputdir=str(root / "output"))
                bound = {
                    m.MANIFESTS[0]: b"{}",
                    m.MANIFESTS[1]: json.dumps({"fail": error_type.__name__}).encode(),
                    m.HELPER: FAKE_HELPER,
                    m.PROTOCOL: b"synthetic protocol",
                    "acquire_assets.py": b"synthetic wrapper",
                }
                with patch.object(m, "frozen_inputs", return_value=bound) as proof:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(error_type):
                        m.run(args)
                proof.assert_called_once()
                receipt = json.loads((root / "output/operation.json").read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["error_type"], error_type.__name__)
                self.assertEqual([b["status"] for b in receipt["batches"]], ["completed", "failed"])
                for batch in receipt["batches"]:
                    raw = (root / "output" / batch["directory"] / "ledger.json").read_bytes()
                    self.assertEqual(batch["ledger_sha256"], hashlib.sha256(raw).hexdigest())
                before = (root / "output/operation.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    m.run(args)
                self.assertEqual((root / "output/operation.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
