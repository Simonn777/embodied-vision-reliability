"""Artificial-only contracts: no Torch, model, wheel, image or scientific-data loads."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_evr0022_qualifier_test", HERE / "qualify.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


class Sandbox:
    """All manifests, assets and installation contents are invented tiny files."""
    def __init__(self, root):
        self.root = root
        self.code = root / "experiments/0022-frontend"
        self.code.mkdir(parents=True)
        for name in ("qualify.py", "supervise.py"):
            (self.code / name).write_bytes((HERE / name).read_bytes())
        for name in ("acquire.py", "acquire_operation.py", "install_runtime.py"):
            (self.code / name).write_text("# artificial source binding\n")
        self.args = argparse.Namespace(
            installation=root / "fake-install", assets_dir=root / "fake-assets",
            acquisition_receipt=root / "fake-assets/ledger.json",
            asset_manifest=root / "configs/sources.json",
            wheel_manifest=root / "metadata/wheels.json",
            wheel_assets_manifest=root / "configs/wheels.json",
            protocol=root / "protocols/artificial.md", expected_commit="a" * 40,
            installation_supervision=root / "supervision/receipt.json",
            output_dir=root / "qualification")
        self.args.protocol.parent.mkdir(parents=True)
        self.args.protocol.write_text("Artificial protocol only\n")
        note = root / "docs/literature/0022-runtime-source-contract.md"
        note.parent.mkdir(parents=True); note.write_text("Artificial source note\n")
        self.args.assets_dir.mkdir()
        assets, layout, inventory, sizes, records = [], [], {}, {}, []
        for prefix, pin in q.PINS.items():
            name = prefix + "-source.py"
            content = ("# invented " + prefix + " package, never imported\n").encode()
            (self.args.assets_dir / name).write_bytes(content)
            assets.append(dict(id=name, filename=name, bytes=len(content), sha256=digest(content)))
            rel = prefix + "/__init__.py"
            layout.append(dict(filename=name, install_prefix=prefix, revision=pin, relative_path=rel))
            target = self.args.installation / "sources" / prefix / rel
            target.parent.mkdir(parents=True); target.write_bytes(content)
            key = str(target.relative_to(self.args.installation))
            inventory[key], sizes[key] = digest(content), len(content)
        for name in ("moge2-small-model.pt", "superpoint_v1.pth", "superpoint_lightglue.pth"):
            content = ("NOT A MODEL: " + name).encode()
            (self.args.assets_dir / name).write_bytes(content)
            assets.append(dict(id=name, filename=name, bytes=len(content), sha256=digest(content)))
        for a in assets:
            records.append(dict(id=a["id"], status="downloaded_verified", sha256=a["sha256"]))
        source_manifest = dict(assets=assets, source_layout=layout)
        dump(self.args.asset_manifest, source_manifest)
        wheel = dict(filename="fake-1.0-py3-none-any.whl", size=7,
                     sha256=digest(b"fixture"), url="https://example.invalid/fake.whl")
        wheels = dict(wheels=[dict(wheel, name="fake", version="1.0")], reuse_installed=[])
        wheel_assets = dict(assets=[dict(filename=wheel["filename"], bytes=wheel["size"],
                                      sha256=wheel["sha256"], url=wheel["url"])])
        dump(self.args.wheel_manifest, wheels); dump(self.args.wheel_assets_manifest, wheel_assets)
        dump(self.args.acquisition_receipt, dict(status="completed", assets=records,
                                                manifest_sha256=q.canonical(source_manifest)))
        runtime = self.args.installation / "runtime/fake.dist-info/METADATA"
        runtime.parent.mkdir(parents=True); runtime.write_text("Name: fake\nVersion: 1.0\n")
        key = str(runtime.relative_to(self.args.installation))
        inventory[key], sizes[key] = q.sha(runtime), runtime.stat().st_size
        files = [self.code / name for name in ("qualify.py", "supervise.py", "install_runtime.py", "acquire.py", "acquire_operation.py")]
        files += [self.args.protocol, self.args.asset_manifest, self.args.wheel_assets_manifest, self.args.wheel_manifest, note]
        self.frozen = {str(p.relative_to(root)): q.sha(p) for p in files}
        keys = [self.code / name for name in ("install_runtime.py", "acquire.py", "supervise.py")]
        keys += [self.args.protocol, self.args.asset_manifest, self.args.wheel_assets_manifest]
        installed = dict(schema="evr0022.install.v1", status="completed", expected_commit=self.args.expected_commit,
            source_sha256={str(p.relative_to(root)): q.sha(p) for p in keys},
            source_manifest_sha256=q.sha(self.args.asset_manifest), source_manifest_canonical_sha256=q.canonical(source_manifest),
            wheel_manifest_sha256=q.sha(self.args.wheel_assets_manifest), wheel_manifest_canonical_sha256=q.canonical(wheel_assets),
            source_acquisition_ledger_sha256=q.sha(self.args.acquisition_receipt),
            installed_files=inventory, installed_file_bytes=sizes)
        dump(self.args.installation / "install-receipt.json", installed)
        outer_dir = self.args.installation_supervision.parent
        outer_dir.mkdir(); (outer_dir / "stdout.log").write_text("fake install succeeded\n")
        (outer_dir / "stderr.log").write_text("")
        command = ["$PYTHON", "$REPOSITORY/experiments/0022-frontend/install_runtime.py"]
        for key, val in [("--destination", self.args.installation), ("--source-assets", self.args.assets_dir),
                         ("--source-manifest", self.args.asset_manifest), ("--wheel-manifest", self.args.wheel_assets_manifest),
                         ("--wheel-assets", root / "fake-wheel-assets"), ("--expected-commit", self.args.expected_commit)]:
            command += [key, str(val).replace(str(root), "$REPOSITORY")]
        outer = dict(schema="evr0022.supervision.v1", status="completed", operation="installation", exit_code=0,
            expected_commit=self.args.expected_commit, termination=None, cleanup_errors=[], remaining_descendant_pids=[],
            wall_seconds=.01, peak_group_rss_kib=100, hard_timeout_seconds=1800,
            group_rss_limit_kib=8 * 1024 * 1024, command=command,
            source_sha256={str((self.code / x).relative_to(root)): q.sha(self.code / x)
                           for x in ("supervise.py", "install_runtime.py")},
            stdout_sha256=q.sha(outer_dir / "stdout.log"), stderr_sha256=q.sha(outer_dir / "stderr.log"))
        dump(self.args.installation_supervision, outer)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.box = Sandbox(self.root)
        self.patch_root = mock.patch.object(q, "ROOT", self.root)
        self.patch_file = mock.patch.object(q, "__file__", str(self.box.code / "qualify.py"))
        # The artifact contract targets 3.9.6; CI may run its artificial verifier
        # under another interpreter. No actual package ABI is exercised here.
        self.patch_version = mock.patch.object(q.sys, "version_info", (3, 9, 6, "final", 0))
        self.patch_root.start(); self.patch_file.start(); self.patch_version.start()

    def tearDown(self):
        self.patch_version.stop(); self.patch_file.stop(); self.patch_root.stop(); self.temp.cleanup()

    def test_fake_artifact_chain_and_same_size_checkpoint_tamper(self):
        files, _, _, bindings = q.verify_files(self.box.args, self.box.frozen)
        self.assertEqual(len(files), 6)
        self.assertEqual(bindings["installation_supervision_sha256"], q.sha(self.box.args.installation_supervision))
        with mock.patch.object(q.sys, "version_info", (3, 11, 0, "final", 0)):
            with self.assertRaisesRegex(q.QualificationError, "python_version"):
                q.verify_files(self.box.args, self.box.frozen)
        p = files["moge2-small-model.pt"]; data = p.read_bytes()
        p.write_bytes(b"X" + data[1:])
        with self.assertRaisesRegex(q.QualificationError, "asset_hash"):
            q.verify_files(self.box.args, self.box.frozen)

    def test_runtime_tamper_and_unlisted_module_rejected(self):
        p = self.box.args.installation / "runtime/fake.dist-info/METADATA"
        old = p.read_bytes(); p.write_bytes(b"x" + old[1:])
        with self.assertRaisesRegex(q.QualificationError, "installed_file_changed"):
            q.verify_files(self.box.args, self.box.frozen)
        p.write_bytes(old)
        (self.box.args.installation / "runtime/injected.py").write_text("# never executed")
        with self.assertRaisesRegex(q.QualificationError, "unlisted_runtime_file"):
            q.verify_files(self.box.args, self.box.frozen)

    def test_source_inventory_cannot_be_rebound_to_other_installer(self):
        path = self.box.args.installation / "install-receipt.json"
        value = q.read_json(path)
        key = "experiments/0022-frontend/install_runtime.py"
        value["source_sha256"][key] = "b" * 64
        dump(path, value)
        with self.assertRaisesRegex(q.QualificationError, "installation_source_binding"):
            q.verify_files(self.box.args, self.box.frozen)

    def test_failed_outer_blocks_completed_inner_and_log_tamper(self):
        path = self.box.args.installation_supervision
        original = q.read_json(path)
        for change in ({"status": "failed"}, {"exit_code": 1}, {"wall_seconds": 1800},
                       {"peak_group_rss_kib": 8 * 1024 * 1024 + 1}, {"termination": "timeout"},
                       {"expected_commit": "b" * 40}, {"remaining_descendant_pids": [123]}):
            with self.subTest(change=change):
                dump(path, dict(original, **change))
                with self.assertRaises(q.QualificationError):
                    q.verify_files(self.box.args, self.box.frozen)
        dump(path, original)
        (path.parent / "stdout.log").write_text("changed log\n")
        with self.assertRaisesRegex(q.QualificationError, "installation_log_changed"):
            q.verify_files(self.box.args, self.box.frozen)

    def test_install_command_cannot_name_other_destination(self):
        path = self.box.args.installation_supervision
        value = q.read_json(path); ix = value["command"].index("--destination")
        value["command"][ix + 1] = "$REPOSITORY/another-install"
        dump(path, value)
        with self.assertRaisesRegex(q.QualificationError, "installation_child_input"):
            q.verify_files(self.box.args, self.box.frozen)

    def test_frozen_git_bytes_and_unresolvable_commit_rejected(self):
        calls = []
        def git(cmd, **kw):
            calls.append(cmd)
            if cmd[1] == "rev-parse":
                return self.box.args.expected_commit + "\n"
            path = cmd[2].split(":", 1)[1]
            return (self.root / path).read_bytes()
        with mock.patch("subprocess.check_output", side_effect=git):
            self.assertEqual(q.frozen_bindings(self.box.args), self.box.frozen)
        self.assertTrue(all(x[0] == "git" for x in calls))
        def tampered(cmd, **kw):
            result = git(cmd, **kw)
            return result + b"tamper" if cmd[1] == "show" and cmd[2].endswith("qualify.py") else result
        with mock.patch("subprocess.check_output", side_effect=tampered):
            with self.assertRaisesRegex(ValueError, "differs from reviewed"):
                q.frozen_bindings(self.box.args)
        with mock.patch("subprocess.check_output", return_value="b" * 40 + "\n"):
            with self.assertRaisesRegex(ValueError, "does not resolve"):
                q.frozen_bindings(self.box.args)

    def test_scoped_local_mapper_has_no_fallback_and_restores(self):
        model = types.SimpleNamespace(enable_pytorch_native_sdpa=mock.Mock())
        torch_threads = types.SimpleNamespace(get_num_threads=lambda: 4, get_num_interop_threads=lambda: 1)
        cv_threads = types.SimpleNamespace(setNumThreads=mock.Mock(), getNumThreads=lambda: 1)
        settings = q.configure_execution(model, torch_threads, cv_threads)
        model.enable_pytorch_native_sdpa.assert_called_once_with()
        cv_threads.setNumThreads.assert_called_once_with(1)
        self.assertEqual(settings, {"moge_native_sdpa_enabled": True, "torch_intraop_threads": 4,
                                    "torch_interop_threads": 1, "opencv_threads": 1})
        cv_threads.getNumThreads = lambda: 8
        with self.assertRaisesRegex(q.QualificationError, "execution_thread_configuration"):
            q.configure_execution(model, torch_threads, cv_threads)
        original = mock.Mock(side_effect=AssertionError("network helper must not run"))
        loader = mock.Mock(return_value={"fake": "state"})
        backend = types.SimpleNamespace(hub=types.SimpleNamespace(load_state_dict_from_url=original), load=loader)
        paths = {q.SP_URL: self.root / "sp.pth", q.LG_URL: self.root / "lg.pth"}
        calls = []
        with q.local_hub(backend, paths, calls):
            self.assertEqual(backend.hub.load_state_dict_from_url(q.SP_URL), {"fake": "state"})
            loader.assert_called_once_with(paths[q.SP_URL], map_location="cpu", weights_only=True)
            with self.assertRaisesRegex(q.QualificationError, "unapproved_hub_url"):
                backend.hub.load_state_dict_from_url(q.SP_URL + "?other=1")
            with self.assertRaisesRegex(q.QualificationError, "unexpected_hub_options"):
                backend.hub.load_state_dict_from_url(q.LG_URL, pretrained=True)
        self.assertIs(backend.hub.load_state_dict_from_url, original)
        original.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with q.local_hub(backend, paths, calls):
                raise RuntimeError("body failure")
        self.assertIs(backend.hub.load_state_dict_from_url, original)

    def test_manufactured_bytes_and_shift_are_deterministic(self):
        a, meta = q.fixture(np); b, other = q.fixture(np)
        self.assertEqual(meta, other)
        self.assertEqual([digest(x.tobytes()) for x in a], [digest(x.tobytes()) for x in b])
        self.assertEqual(a[0].shape, (384, 512, 3)); self.assertEqual(a[0].dtype, np.float32)
        np.testing.assert_array_equal(a[1][:, :12], 0)
        np.testing.assert_array_equal(a[1][:, 12:], a[0][:, :-12])
        np.testing.assert_array_equal(a[0][36:78, 46:88], np.broadcast_to(np.array([.95, .1, .2], np.float32), (42, 42, 3)))
        self.assertTrue(np.isfinite(a[0]).all()); self.assertGreaterEqual(a[0].min(), 0); self.assertLessEqual(a[0].max(), 1)
        rng = np.random.default_rng(0)
        pixel = (.55 * np.array([0., 0., 0.]) + .35 * rng.random(3) + .05).astype(np.float32)
        np.testing.assert_array_equal(a[0][0, 0], pixel)
        print("fixture_sha256=" + json.dumps([digest(x.tobytes()) for x in a]))

    def test_failed_receipt_is_retained_and_never_overwritten(self):
        with mock.patch.object(q, "frozen_bindings", return_value=self.box.frozen), \
             mock.patch.object(q, "verify_files", side_effect=q.QualificationError("invented_tamper")), \
             mock.patch.object(q, "install_guard") as guard:
            with self.assertRaisesRegex(q.QualificationError, "invented_tamper"):
                q.qualify(self.box.args)
            guard.assert_not_called()
        receipt = self.box.args.output_dir / "receipt.json"
        original = receipt.read_bytes(); value = json.loads(original)
        self.assertEqual(value["status"], "failed"); self.assertFalse(value["eligible"])
        self.assertEqual(value["failed_stage"], "verification"); self.assertGreaterEqual(value["wall_seconds"], 0)
        with self.assertRaisesRegex(q.QualificationError, "output_directory_exists"):
            q.qualify(self.box.args)
        self.assertEqual(receipt.read_bytes(), original)

    def test_warnings_are_retained_on_exception_and_legacy_collision_fails(self):
        record = {"warnings": []}
        with self.assertRaisesRegex(RuntimeError, "manufactured failure"):
            with q.capture_warnings(record):
                warnings.warn("manufactured compatibility warning", UserWarning)
                raise RuntimeError("manufactured failure")
        self.assertEqual(record["warnings"], [{"category": "UserWarning", "message": "manufactured compatibility warning"}])
        renamed = q.rename_lightglue({"self_attn.0.weight": 1, "cross_attn.1.bias": 2}, 2)
        self.assertEqual(renamed, {"transformers.0.self_attn.weight": 1, "transformers.1.cross_attn.bias": 2})
        with self.assertRaisesRegex(q.QualificationError, "legacy_key_collision"):
            q.rename_lightglue({"self_attn.0.weight": 1, "transformers.0.self_attn.weight": 2}, 2)

    def test_offline_guard_denies_real_python_entry_points_in_isolated_process(self):
        # Audit hooks cannot be removed, so never install one in the test runner.
        code = r'''
import importlib.util, json, pathlib, socket, subprocess, sys, io
spec=importlib.util.spec_from_file_location('q',sys.argv[1]);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
sentinel=pathlib.Path(sys.argv[2]);attempts=[];q.install_guard(attempts)
def denied(fn):
    try: fn()
    except q.QualificationError: return
    raise AssertionError('operation escaped guard')
denied(lambda:socket.getaddrinfo('blocked.invalid',443))
with socket.socket() as s: denied(lambda:s.connect(('127.0.0.1',9)))
denied(lambda:subprocess.Popen([sys.executable,'-c','import pathlib;pathlib.Path('+repr(str(sentinel))+').write_text("escaped")']))
sys.stdin=io.StringIO('yes\n');denied(lambda:input('blocked input: '))
assert not sentinel.exists()
assert [a['event'] for a in attempts]==['socket.getaddrinfo','socket.connect','subprocess.Popen','builtins.input'],attempts
assert 'torch' not in sys.modules
print(json.dumps(attempts))
'''
        outcome = subprocess.run([sys.executable, "-c", code, str(HERE / "qualify.py"), str(self.root / "escaped")],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertFalse((self.root / "escaped").exists())
        self.assertEqual(len(json.loads(outcome.stdout)), 4)

    def test_only_exact_optional_numpy_probe_is_absence_compatible(self):
        code = r'''
import importlib.util, json, subprocess, sys, types
spec=importlib.util.spec_from_file_location('q',sys.argv[1]);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
module=types.ModuleType('numpy.testing._private.utils');module.__file__='/artificial/numpy/testing/_private/utils.py'
module.__dict__['subprocess']=subprocess;sys.modules[module.__name__]=module
source="def check_support_sve():\n    try:\n        output=subprocess.run('lscpu',capture_output=True,text=True)\n        return 'sve' in output.stdout\n    except OSError:\n        return False\n"
exec(compile(source,module.__file__,'exec'),module.__dict__)
q.sys.platform='darwin';attempts=[];probes=[];q.install_guard(attempts,probes)
assert module.check_support_sve() is False
assert attempts==[] and len(probes)==1 and probes[0]['outcome']=='suppressed_optional_probe'
def denied(fn):
    try: fn()
    except q.QualificationError: return
    raise AssertionError('unexpected exception downgrade or subprocess execution')
denied(lambda:subprocess.run('lscpu',capture_output=True,text=True))
for altered in [source.replace("'lscpu'","['lscpu','--help']"), source.replace('text=True','text=True,check=True'),source.replace('text=True','text=True,env={}'),source.replace('check_support_sve','other_function')]:
    exec(compile(altered,module.__file__,'exec'),module.__dict__)
    denied(module.other_function if 'other_function' in altered else module.check_support_sve)
exec(compile(source,module.__file__,'exec'),module.__dict__);q.sys.platform='linux'
denied(module.check_support_sve)
assert len(attempts)==6 and len(probes)==1
assert all(row['event']=='subprocess.Popen' for row in attempts)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps({'denied':len(attempts),'compatibility':probes}))
'''
        outcome = subprocess.run([sys.executable, "-c", code, str(HERE / "qualify.py")],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertEqual(json.loads(outcome.stdout)["denied"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
