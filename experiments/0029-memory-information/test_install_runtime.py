"""Artificial-only installer admission/guard tests; never run installer.run()."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('evr0029_installer_test_subject', Path(__file__).with_name('install_runtime.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
COMMIT = 'a' * 40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def acq_fixture(base):
    base = base.resolve()
    root = base / 'repo'; root.mkdir(); acq = base / 'assets'; acq.mkdir(); outerdir = base / 'outer'; outerdir.mkdir()
    manifests = []
    for path, dirname, cap in zip(m.MANIFESTS, m.BATCHES, (190000000, 110000000)):
        manifest = {'body_cap_bytes': cap, 'assets': [{'id': dirname, 'filename': 'one.txt', 'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}]}
        write_json(root / path, manifest); manifests.append(manifest)
        ledger = {'schema': 1, 'status': 'completed', 'manifest_sha256': m.canonical(manifest), 'body_cap_bytes': cap, 'actual_body_bytes': 1, 'requests': [{'bytes_read': 1, 'status': 'complete'}], 'assets': [{'id': dirname, 'status': 'downloaded_verified', 'sha256': hashlib.sha256(b'x').hexdigest(), 'new_body_bytes': 1}]}
        write_json(acq / dirname / 'ledger.json', ledger)
    frozen = {rel: (root / rel).read_bytes() if rel in m.MANIFESTS else ('original ' + rel).encode() for rel in (*m.MANIFESTS, m.ACQ_PROTOCOL, m.HELPER, m.ACQUIRE, m.SUPERVISOR, m.MONITOR)}
    operation = {'schema': 'evr0029.asset-operation.v1', 'status': 'completed', 'source_commit': COMMIT, 'asset_or_dependency_only': True, 'benchmark_payload_access': False, 'bindings': {p: hashlib.sha256(frozen[p]).hexdigest() for p in (*m.MANIFESTS, m.ACQ_PROTOCOL, m.HELPER, m.ACQUIRE)}, 'batches': [{'manifest': path, 'directory': dirname, 'status': 'completed', 'ledger_sha256': m.sha(acq / dirname / 'ledger.json'), 'actual_body_bytes': 1} for path, dirname in zip(m.MANIFESTS, m.BATCHES)], 'actual_body_bytes': 2, 'body_cap_bytes': 300000000, 'wall_seconds': .1}
    write_json(acq / 'operation.json', operation)
    command = ['/opt/homebrew/bin/python3.12', '-B', str(root / m.ACQUIRE), '--repo-root', str(root), '--expected-commit', COMMIT, '--outputdir', str(acq)]
    spec = {'commit': COMMIT, 'repo': str(root), 'operation_output': str(outerdir), 'result_file': str(acq / 'operation.json'), 'timeout': 1800, 'environment': {k: '4' for k in m.THREAD_KEYS}, 'command': command}
    spec_path = base / 'spec.json'; write_json(spec_path, spec)
    (outerdir / 'stdout.log').write_text(json.dumps({k: operation[k] for k in ('status', 'wall_seconds', 'batches')}) + '\n'); (outerdir / 'stderr.log').write_bytes(b'')
    outer = {'schema': 'evr0026.operation.v1', 'status': 'completed', 'exit_code': 0, 'termination': None, 'remaining_descendant_pids': [], 'cleanup_errors': [], 'expected_commit': COMMIT, 'spec_sha256': m.sha(spec_path), 'output_sha256': m.sha(acq / 'operation.json'), 'supervisor_sha256': hashlib.sha256(frozen[m.SUPERVISOR]).hexdigest(), 'monitor_sha256': hashlib.sha256(frozen[m.MONITOR]).hexdigest(), 'hard_timeout_seconds': 1800, 'group_rss_limit_kib': 8388608, 'thread_environment': {k: '4' for k in m.THREAD_KEYS}, 'wall_seconds': .2, 'peak_group_rss_kib': 1000, 'command': [x.replace(str(root), '$REPOSITORY').replace(str(root.parent), '$PRIVATE') for x in command], 'log_sha256': {n: m.sha(outerdir / n) for n in ('stdout.log', 'stderr.log')}}
    outer_path = outerdir / 'receipt.json'; write_json(outer_path, outer)
    return root, acq, outer_path, spec_path, manifests, frozen


def sdist_fixture(base):
    assets = {}; rows = []
    for key in ('antlr4-python3-runtime', 'iopath'):
        name = key + '-fake'; path = base / (name + '.tar.gz'); content = b'# Synthetic setup; must never execute.\n'
        with tarfile.open(path, 'w:gz', format=tarfile.USTAR_FORMAT) as archive:
            directory = tarfile.TarInfo(name); directory.type = tarfile.DIRTYPE; archive.addfile(directory)
            member = tarfile.TarInfo(name + '/setup.py'); member.size = len(content); archive.addfile(member, io.BytesIO(content))
        asset = {'id': key, 'kind': 'sdist', 'filename': path.name, 'sha256': m.sha(path)}; assets[key] = (asset, path)
        rows.append({'id': key, 'approved': True, 'filename': path.name, 'sha256': asset['sha256'], 'archive_root': name, 'review_note': 'Artificial only; no setup execution.', 'build_hooks': [name + '/setup.py'], 'files': {name + '/setup.py': {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}}, 'directories': [name]})
    return {'schema': 'evr0029.sdist-build-review.v1', 'approved': True, 'network': 'disabled', 'build_tool_versions': m.TOOLS, 'sdists': rows}, assets


class InstallerAdmission(unittest.TestCase):
    def test_source_and_same_size_asset_changes_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = root / 'config.json'; source.write_bytes(b'original')
            with mock.patch.object(m, 'git_bytes', return_value=b'original'):
                self.assertEqual(len(m.frozen(root, COMMIT, ['config.json'])), 1)
                source.write_bytes(b'changed!')
                with self.assertRaisesRegex(ValueError, 'source differs'): m.frozen(root, COMMIT, ['config.json'])
            asset = {'id': 'fake', 'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}
            with self.assertRaisesRegex(ValueError, 'asset size/SHA'): m.verify_asset(source, asset)

    def test_completed_outer_and_explicit_minus_b_then_failure_block(self):
        with tempfile.TemporaryDirectory() as name:
            root, acq, outer, spec, manifests, frozen = acq_fixture(Path(name))
            with mock.patch.object(m, 'git_bytes', side_effect=lambda r, c, p: frozen[p]):
                self.assertEqual(m.verify_acquisition(root, acq, outer, spec, manifests)['commit'], COMMIT)
                record = m.read_json(outer); record['status'] = 'failed'; write_json(outer, record)
                with self.assertRaisesRegex(ValueError, 'unclean acquisition outer'): m.verify_acquisition(root, acq, outer, spec, manifests)

    def test_changed_provenance_or_log_block(self):
        with tempfile.TemporaryDirectory() as name:
            root, acq, outer, spec, manifests, frozen = acq_fixture(Path(name))
            with mock.patch.object(m, 'git_bytes', side_effect=lambda r, c, p: frozen[p]):
                (outer.parent / 'stderr.log').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'logs changed'): m.verify_acquisition(root, acq, outer, spec, manifests)
                (outer.parent / 'stderr.log').write_bytes(b'')
                record = m.read_json(acq / 'operation.json'); record['bindings'][m.ACQUIRE] = '0' * 64; write_json(acq / 'operation.json', record)
                with self.assertRaisesRegex(ValueError, 'unbound acquisition'): m.verify_acquisition(root, acq, outer, spec, manifests)

    def test_review_requires_positive_explicit_approval(self):
        with tempfile.TemporaryDirectory() as name:
            review, assets = sdist_fixture(Path(name))
            self.assertEqual(len(m.inspect_reviewed_sdists(review, assets)), 2)
            review['approved'] = False
            with self.assertRaisesRegex(ValueError, 'explicit sdist build approval'): m.inspect_reviewed_sdists(review, assets)

    def test_archive_content_change_or_symlink_is_not_approved(self):
        with tempfile.TemporaryDirectory() as name:
            review, assets = sdist_fixture(Path(name)); row = review['sdists'][0]; path = assets[row['id']][1]
            with tarfile.open(path, 'w:gz', format=tarfile.USTAR_FORMAT) as archive:
                directory = tarfile.TarInfo(row['archive_root']); directory.type = tarfile.DIRTYPE; archive.addfile(directory)
                member = tarfile.TarInfo(row['build_hooks'][0]); member.size = 7; archive.addfile(member, io.BytesIO(b'changed'))
            with self.assertRaisesRegex(ValueError, 'archive bytes changed'): m.inspect_reviewed_sdists(review, assets)
            with tarfile.open(path, 'w:gz', format=tarfile.USTAR_FORMAT) as archive:
                member = tarfile.TarInfo(row['build_hooks'][0]); member.type = tarfile.SYMTYPE; member.linkname = '/etc/passwd'; archive.addfile(member)
            with self.assertRaisesRegex(ValueError, 'member/type'): m.inspect_reviewed_sdists(review, assets)

    @unittest.skipUnless(sys.version_info[:3] == (3, 12, 13), 'real guard requires the pinned CPython 3.12.13; passed locally in install-tests-01')
    def test_network_denied_before_any_socket_exists(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); guard = root / 'guard.py'; guard.write_text(m.GUARD); probe = root / 'probe.py'
            probe.write_text("import socket\ntry:\n socket.socket()\nexcept PermissionError as error:\n print(str(error))\nelse:\n raise RuntimeError('guard failed before connect')\n")
            env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_', 'EVR_'))}; env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-S', str(guard), 'script', str(probe)], env=env, capture_output=True, text=True, timeout=15, check=False)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('offline guard: socket.__new__', result.stdout)

    def test_only_fixed_pep517_helper_gets_guarded_child(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); py = root / 'venv/bin/python'; guard = root / 'guard.py'; helper = root / 'venv/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py'; hooks = []; namespace = {'__file__': str(guard)}
            with mock.patch.object(sys, 'version_info', (3, 12, 13)), mock.patch.object(sys, 'argv', [str(guard), 'module', 'fake_entry']), mock.patch.object(sys, 'prefix', str(root / 'venv')), mock.patch.object(sys, 'executable', str(py)), mock.patch.object(sys, 'addaudithook', side_effect=hooks.append), mock.patch('site.main'), mock.patch('runpy.run_module'), mock.patch('subprocess.Popen', return_value='synthetic_process') as spawn:
                exec(m.GUARD, namespace)
                result = namespace['guarded_popen']([str(py), str(helper), 'build_wheel', str(root / 'control')], env={'PYTHONHOME': 'bad', 'PYTHONPATH': 'bad'})
                self.assertEqual(result, 'synthetic_process'); called = spawn.call_args
                self.assertEqual(called.args[0], [str(py), '-S', str(guard), 'script', str(helper), 'build_wheel', str(root / 'control')])
                self.assertEqual(called.kwargs['env']['PYTHONPATH'], ''); self.assertNotIn('PYTHONHOME', called.kwargs['env'])
                self.assertEqual(called.kwargs['env']['PIP_NO_INDEX'], '1')
                with self.assertRaises(PermissionError): namespace['guarded_popen']([str(py), '-c', 'unreviewed', 'extra'])
                with self.assertRaises(PermissionError): hooks[0]('subprocess.Popen', ('curl', ['curl', 'https://invalid.example']))

    def test_inventory_excludes_mutable_receipts_but_keeps_extra_installed_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for sub in ('sources', 'weights', 'venv', 'built-wheels'): (root / sub).mkdir()
            (root / 'offline_guard.py').write_text('guard')
            (root / 'venv/extra.txt').write_text('must be included')
            (root / 'install-receipt.json').write_text('started'); (root / 'command-stdout.log').write_text('log')
            first = m.inventory(root); (root / 'install-receipt.json').write_text('completed')
            self.assertEqual(first, m.inventory(root)); self.assertIn('venv/extra.txt', first)
            self.assertNotIn('install-receipt.json', first); self.assertNotIn('command-stdout.log', first)


if __name__ == '__main__':
    unittest.main(verbosity=2)
