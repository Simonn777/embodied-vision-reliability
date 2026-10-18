"""Own tiny files/receipts only: no real dependencies, network, weights or data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovered_supervisor", ROOT / "scripts/run_tapnext_recovered_capability.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def raw(path, payload=b"own toy bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"bytes": len(payload), "sha256": m.digest(path)}


class RecoveryGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.repo, self.old, self.failed, self.recovery = [self.base / n for n in ("repo", "old", "failed", "recovery")]
        for p in (self.repo, self.old, self.failed, self.recovery):
            p.mkdir()
        self.assets = []
        for i in range(9):
            kind = "source" if i < 5 else ("wheel" if i < 8 else "checkpoint")
            rel = ("source/tapnet/p" + str(i) + ".py" if i < 5 else
                   "wheels/w" + str(i) + ".whl" if i < 8 else "weights/tapnextpp_ckpt.pt")
            owner = self.failed if i < 8 else self.recovery
            info = raw(owner / rel, ("own fixture " + str(i)).encode())
            self.assets.append({"id": "a" + str(i), "kind": kind, "relative_path": rel,
                                "url": "https://example.invalid/unused", **info})
        raw(self.failed / "source/preserved.partial", b"")
        overlay = {"toy.dist-info/RECORD": raw(self.failed / "overlay/toy.dist-info/RECORD")}
        self.old_hashes = {"runtime-fixed/toy.py": raw(self.old / "runtime-fixed/toy.py")["sha256"]}
        old_markers = {"prior-started.json": raw(self.old / "prior-started.json")["sha256"]}
        original = {"preparation": {"assets": self.assets, "budget": {"historical_ceiling_bytes": 100,
                    "metadata_allowance_bytes": 10, "limit_bytes": 10000}, "import_timeout_seconds": 120,
                    "runtime": {"old_init_sha256": self.old_hashes}},
                    "cpu_affinity": [0, 1, 2, 3], "address_space_bytes": 8 * 1024**3}
        orig_path = self.repo / m.ORIGINAL_INPUTS[0]; write(orig_path, original)
        helper = self.repo / "scripts/prepare_tapnext_assets.py"; raw(helper)
        raw(self.repo / "scripts/prepare_tapnext_recovery.py", b"# own inert preparer placeholder\n")
        failure = {"status": "failed", "stage": "dependencies", "config_sha256": m.digest(orig_path), "source_sha256": m.digest(helper)}
        write(self.failed / "preparation-dependencies.json", failure)
        self.prior_ledger = {"config_sha256": m.digest(orig_path), "source_sha256": m.digest(helper),
            "base_ceiling_bytes": 110, "planned_ceiling_bytes": 110 + sum(a["bytes"] + 1 for a in self.assets),
            "limit_bytes": 10000, "reservations": {a["id"]: a["bytes"] + 1 for a in self.assets[:-1]},
            "payload_bytes_observed": sum(a["bytes"] for a in self.assets[:-1]),
            "conservative_cumulative_ceiling_bytes": 110 + sum(a["bytes"] + 1 for a in self.assets[:-1])}
        write(self.failed / "acquisition-ledger.json", self.prior_ledger)
        refs = {}
        for name, value in (("failure_receipt", failure), ("cache_inventory", m.file_inventory(self.failed)),
                            ("overlay_inventory", {"files": overlay}),
                            ("failure_review", {"native": "not_started", "checkpoint": "not_started",
                                                "import_check": "not_started", "old_markers_preserved": old_markers})):
            path = self.repo / ("evidence/" + name + ".json"); write(path, value)
            refs[name] = {"path": path.relative_to(self.repo).as_posix(), "sha256": m.digest(path)}
        self.cfg = {"recovery": {"original_config": {"path": m.ORIGINAL_INPUTS[0], "sha256": m.digest(orig_path)},
            "frozen_preparer": {"path": "scripts/prepare_tapnext_assets.py", "sha256": m.digest(helper)}, "evidence": refs}}
        write(self.repo / m.CONFIG_PATH, self.cfg)
        cfg_sha = m.digest(self.repo / m.CONFIG_PATH)
        src_sha = m.digest(self.repo / "scripts/prepare_tapnext_recovery.py")
        paths = {"old_cache": str(self.old), "failed_cache": str(self.failed), "recovery_root": str(self.recovery),
                 "source_root": str(self.recovery / "source"), "overlay": str(self.failed / "overlay")}
        self.common = {"schema_version": 1, "status": "completed", "config_sha256": cfg_sha, "source_sha256": src_sha,
            "original_config_sha256": m.digest(orig_path), "frozen_preparer_sha256": m.digest(helper),
            "prior_failure_receipt_sha256": refs["failure_receipt"]["sha256"],
            "prior_cache_inventory_sha256": refs["cache_inventory"]["sha256"],
            "prior_overlay_inventory_sha256": refs["overlay_inventory"]["sha256"],
            "paths": paths, "preserved_failed_cache": True, "started_utc": "toy-start", "finished_utc": "toy-end",
            "old_preservation": {**self.old_hashes, **old_markers}, "affinity": [0, 1, 2, 3],
            "AS_bytes": 8 * 1024**3, "model_constructed": False}
        for asset in self.assets[:5]:
            raw(self.recovery / asset["relative_path"], (self.failed / asset["relative_path"]).read_bytes())
        manifests = {}
        for name, files in (("source", m.file_inventory(self.recovery / "source")), ("overlay", overlay)):
            path = self.recovery / (name + "-manifest.json")
            write(path, {"schema_version": 1, "config_sha256": cfg_sha, "source_sha256": src_sha, "files": files})
            manifests[name + "_manifest"] = {"relative_path": path.name, "sha256": m.digest(path)}
        write(self.recovery / "import-check.stdout.log", {"status": "passed", "no_model_constructed_or_forward_called": True})
        raw(self.recovery / "import-check.stderr.log", b"")
        child = {"exit_code": 0, "raw_wait_status": 0, "terminated_by_parent": False, "elapsed_seconds": 1,
                 "started_utc": "toy-start", "finished_utc": "toy-end", "command": ["inert-fixture"],
                 "stdout_sha256": m.digest(self.recovery / "import-check.stdout.log"),
                 "stderr_sha256": m.digest(self.recovery / "import-check.stderr.log")}
        write(self.recovery / "import-check.json", child)
        inherited = {**copy.deepcopy(self.prior_ledger), "config_sha256": cfg_sha, "source_sha256": src_sha,
            "prior_config_sha256": self.prior_ledger["config_sha256"], "prior_source_sha256": self.prior_ledger["source_sha256"],
            "inherited_ledger_sha256": m.digest(self.failed / "acquisition-ledger.json"), "new_payload_bytes_observed": 0}
        write(self.recovery / "acquisition-ledger-dependencies.json", inherited)
        self.dep = {**self.common, **manifests, "stage": "dependencies-recovery", "new_network_payload_bytes": 0,
            "checkpoint_started": False, "import_check": child,
            "assets": [{k: a[k] for k in ("id", "relative_path", "bytes", "sha256")} | {"reused_verified": True} for a in self.assets[:-1]],
            "ledger_snapshot": {"relative_path": "acquisition-ledger-dependencies.json",
                                "sha256": m.digest(self.recovery / "acquisition-ledger-dependencies.json")}}
        self.make_marker("dependencies", self.dep)
        write(self.recovery / "recovery-dependencies.json", self.dep)
        asset = self.assets[-1]
        record = {"asset": asset, "status": "downloaded_verified", "config_sha256": cfg_sha, "source_sha256": src_sha,
                  "sha256": asset["sha256"], "partial_bytes_stored": 0, "http_payload_attempts": 1,
                  "payload_bytes_observed": asset["bytes"], "observed_payload_sha256": asset["sha256"],
                  "partial_exists": False, "finished_utc": "toy-end"}
        record_path = self.recovery / "asset-records/a8.json"; write(record_path, record)
        ledger = copy.deepcopy(inherited)
        ledger["reservations"][asset["id"]] = asset["bytes"] + 1
        ledger["payload_bytes_observed"] += asset["bytes"]
        ledger["new_payload_bytes_observed"] = asset["bytes"]
        ledger["conservative_cumulative_ceiling_bytes"] = ledger["planned_ceiling_bytes"]
        for name in ("acquisition-ledger.json", "acquisition-ledger-checkpoint.json"):
            write(self.recovery / name, ledger)
        self.ckpt = {**self.common, **manifests, "stage": "checkpoint-recovery", "checkpoint_started": True,
            "new_network_payload_bytes": asset["bytes"], "checkpoint_asset": record,
            "checkpoint_asset_record_sha256": m.digest(record_path),
            "dependency_receipt_sha256": m.digest(self.recovery / "recovery-dependencies.json"),
            "ledger_snapshot": {"relative_path": "acquisition-ledger-checkpoint.json",
                                "sha256": m.digest(self.recovery / "acquisition-ledger-checkpoint.json")}}
        self.make_marker("checkpoint", self.ckpt)
        write(self.recovery / "recovery-checkpoint.json", self.ckpt)

    def make_marker(self, stage, receipt):
        path = self.recovery / ("markers/" + stage + ".started.json")
        write(path, {**self.common, "stage": stage + "-recovery", "status": "running"})
        receipt["marker_sha256"] = m.digest(path)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self):
        return m.verify_recovery_preparation(self.repo, self.old, self.failed, self.recovery)

    def update_dep(self):
        write(self.recovery / "recovery-dependencies.json", self.dep)
        self.ckpt["dependency_receipt_sha256"] = m.digest(self.recovery / "recovery-dependencies.json")
        write(self.recovery / "recovery-checkpoint.json", self.ckpt)

    def test_complete_independent_recovery_accepted_original_failure_preserved(self):
        before = m.file_inventory(self.failed)
        self.assertEqual(set(self.verify()), {"dependencies", "checkpoint"})
        self.assertEqual(before, m.file_inventory(self.failed))
        self.assertEqual(m.load_json(self.failed / "preparation-dependencies.json")["status"], "failed")

    def test_no_native_package_import(self):
        import sys
        before = {n for n in sys.modules if n.split('.')[0] in ('torch', 'torchvision', 'numpy', 'tapnet')}
        self.verify()
        self.assertEqual(before, {n for n in sys.modules if n.split('.')[0] in ('torch', 'torchvision', 'numpy', 'tapnet')})

    def test_failed_or_missing_recovery_stage_rejected(self):
        for value in ("failed", "running"):
            with self.subTest(value=value):
                self.ckpt["status"] = value; write(self.recovery / "recovery-checkpoint.json", self.ckpt)
                with self.assertRaises(ValueError): self.verify()
        (self.recovery / "recovery-checkpoint.json").unlink()
        with self.assertRaises(FileNotFoundError): self.verify()

    def test_prior_cache_any_extra_file_rejected(self):
        raw(self.failed / "extra")
        with self.assertRaisesRegex(ValueError, "failed cache"): self.verify()

    def test_clean_source_changed_or_extra_file_rejected(self):
        raw(self.recovery / "source/unbound.py")
        with self.assertRaisesRegex(ValueError, "source manifest"): self.verify()

    def test_symlink_source_rejected(self):
        (self.recovery / "source/evil").symlink_to(self.old)
        with self.assertRaises(ValueError): self.verify()

    def test_old_runtime_or_markers_rechecked(self):
        raw(self.old / "prior-started.json", b"changed")
        with self.assertRaisesRegex(ValueError, "Old runtime"): self.verify()

    def test_dependency_network_payload_rejected(self):
        self.dep["new_network_payload_bytes"] = 1; self.update_dep()
        with self.assertRaisesRegex(ValueError, "scope"): self.verify()

    def test_checkpoint_wrong_dependency_binding_rejected(self):
        self.ckpt["dependency_receipt_sha256"] = "0" * 64
        write(self.recovery / "recovery-checkpoint.json", self.ckpt)
        with self.assertRaisesRegex(ValueError, "dependency receipt"): self.verify()

    def test_checkpoint_payload_bytes_not_just_receipt(self):
        raw(self.recovery / self.assets[-1]["relative_path"], b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint bytes"): self.verify()

    def test_checkpoint_actual_partial_rejected_despite_completed_record(self):
        path = self.recovery / (self.assets[-1]["relative_path"] + ".partial")
        raw(path, b"")
        with self.assertRaisesRegex(ValueError, "partial still exists"): self.verify()
        path.unlink(); path.symlink_to(self.base / "absent")
        with self.assertRaisesRegex(ValueError, "partial still exists"): self.verify()

    def test_output_must_not_mutate_protected_inputs(self):
        for directory in (self.old, self.failed, self.recovery / "source", self.recovery / "weights", self.recovery / "markers"):
            with self.subTest(directory=str(directory)), self.assertRaisesRegex(ValueError, "protected cache"):
                m.validate_output(directory / "new-run", self.old, self.failed, self.recovery)
        m.validate_output(self.base / "new-output", self.old, self.failed, self.recovery)
        m.validate_output(self.recovery / "native-output", self.old, self.failed, self.recovery)

    def test_checkpoint_record_must_equal_embedded_record(self):
        self.ckpt["checkpoint_asset"]["http_payload_attempts"] = 2
        write(self.recovery / "recovery-checkpoint.json", self.ckpt)
        with self.assertRaisesRegex(ValueError, "first checkpoint"): self.verify()

    def test_ledger_cannot_reset_prior_payload_or_double_count(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                lp = self.recovery / "acquisition-ledger-checkpoint.json"
                ledger = m.load_json(lp); ledger["payload_bytes_observed"] += delta
                write(lp, ledger); write(self.recovery / "acquisition-ledger.json", ledger)
                self.ckpt["ledger_snapshot"]["sha256"] = m.digest(lp)
                write(self.recovery / "recovery-checkpoint.json", self.ckpt)
                with self.assertRaisesRegex(ValueError, "cumulative"): self.verify()
                ledger["payload_bytes_observed"] -= delta; write(lp, ledger)

    def test_stage_marker_required_and_bound(self):
        raw(self.recovery / "markers/checkpoint.started.json", b"{}")
        with self.assertRaisesRegex(ValueError, "stage marker"): self.verify()

    def test_import_timeout_and_failed_exit_rejected(self):
        for field, val in (("elapsed_seconds", 121), ("exit_code", 1), ("raw_wait_status", 256), ("elapsed_seconds", True)):
            with self.subTest(field=field):
                original = copy.deepcopy(self.dep["import_check"])
                self.dep["import_check"][field] = val
                write(self.recovery / "import-check.json", self.dep["import_check"]); self.update_dep()
                with self.assertRaisesRegex(ValueError, "child did not finish"): self.verify()
                self.dep["import_check"] = original

    def test_import_success_claim_requires_no_model_probe(self):
        write(self.recovery / "import-check.stdout.log", {"status": "passed", "no_model_constructed_or_forward_called": False})
        self.dep["import_check"]["stdout_sha256"] = m.digest(self.recovery / "import-check.stdout.log")
        write(self.recovery / "import-check.json", self.dep["import_check"]); self.update_dep()
        with self.assertRaisesRegex(ValueError, "import-only"): self.verify()

    def test_wrong_recovery_paths_rejected(self):
        self.ckpt["paths"] = {**self.ckpt["paths"], "overlay": str(self.recovery / "overlay")}
        write(self.recovery / "recovery-checkpoint.json", self.ckpt)
        with self.assertRaisesRegex(ValueError, "Completed bound"): self.verify()

    def test_native_markers_and_dangling_links_forbid_repeat(self):
        for p in (self.failed / "0005-native-started.json", self.recovery / m.NATIVE_MARKER):
            raw(p)
            with self.assertRaises(ValueError): m.absent_native_markers(self.failed, self.recovery)
            p.unlink(); p.symlink_to(self.base / "absent")
            with self.assertRaises(ValueError): m.absent_native_markers(self.failed, self.recovery)
            p.unlink()

    def test_child_requires_launch_record_before_adapter_import(self):
        argv = ["supervisor", "--child", "--failed-cache", str(self.failed),
                "--recovery-root", str(self.recovery), "--bootstapir-cache", str(self.old),
                "--output", str(self.base / "output"), "--freeze-sha256", "f" * 64]
        with patch.object(m, "__file__", str(self.repo / "scripts/supervisor.py")), \
             patch.object(m.sys, "argv", argv), \
             patch.object(m, "verify_freeze", return_value={"source_commit": "c" * 40}), \
             patch.object(m.importlib.util, "spec_from_file_location", side_effect=AssertionError("adapter import forbidden")):
            with self.assertRaises(FileNotFoundError): m.main()
            write(self.recovery / m.LAUNCH_MARKER, {"freeze_sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "matching exclusive"): m.main()

    def test_parent_launch_marker_forbids_second_attempt(self):
        argv = ["supervisor", "--failed-cache", str(self.failed), "--recovery-root", str(self.recovery),
                "--bootstapir-cache", str(self.old), "--output", str(self.base / "different-output"),
                "--freeze-sha256", "f" * 64]
        raw(self.recovery / m.LAUNCH_MARKER)
        with patch.object(m, "__file__", str(self.repo / "scripts/supervisor.py")), \
             patch.object(m.sys, "argv", argv), \
             patch.object(m, "verify_freeze", return_value={"source_commit": "c" * 40}), \
             patch.object(m.subprocess, "Popen", side_effect=AssertionError("no second child")):
            with self.assertRaisesRegex(ValueError, "supervisor already attempted"): m.main()
        self.assertFalse((self.base / "different-output").exists())

    def test_unsafe_relative_artifact_paths_rejected(self):
        for name in ("../escape", "/escape", ""):
            with self.subTest(name=name), self.assertRaises(ValueError): m.local(self.recovery, name)


class RecoveryFreezeTests(unittest.TestCase):
    def test_old_and_new_freezes_bound_to_current_index_and_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
            git("init"); git("config", "user.email", "fixture@example.invalid"); git("config", "user.name", "Fixture")
            for name in m.ORIGINAL_INPUTS:
                raw(root / name, b"own inert fixture\n")
            raw(root / "scripts/run_tapnext_capability.py", (ROOT / "scripts/run_tapnext_capability.py").read_bytes())
            git("add", "."); git("commit", "-m", "Original source fixture")
            original = {"source_commit": git("rev-parse", "HEAD"), "files": {n: m.digest(root / n) for n in m.ORIGINAL_INPUTS}}
            old_freeze = root / "protocols/0005-native-capability-freeze.json"; write(old_freeze, original)
            for name in m.INPUTS:
                if not (root / name).exists(): raw(root / name, b"own recovery fixture\n")
            git("add", "."); git("commit", "-m", "Recovery source fixture")
            write(root / m.FREEZE_PATH, {"source_commit": git("rev-parse", "HEAD"), "files": {n: m.digest(root / n) for n in m.INPUTS}})
            git("add", "."); git("commit", "-m", "Recovery freeze fixture")
            freeze_sha = m.digest(root / m.FREEZE_PATH)
            with patch.object(m, "ORIGINAL_FREEZE_SHA", m.digest(old_freeze)):
                self.assertEqual(len(m.verify_freeze(root, freeze_sha)["files"]), 16)
                path = root / m.CONFIG_PATH; path.write_text("changed")
                with self.assertRaises(ValueError): m.verify_freeze(root, freeze_sha)
                git("checkout", "--", m.CONFIG_PATH)
                path.write_text("changed index"); git("add", m.CONFIG_PATH); git("show", "HEAD:" + m.CONFIG_PATH)
                path.write_bytes(git("show", "HEAD:" + m.CONFIG_PATH).encode() + b"\n")
                with self.assertRaises(ValueError): m.verify_freeze(root, freeze_sha)


if __name__ == "__main__":
    unittest.main()
