"""Offline miniature Git/asset receipts; never import a runtime or run a model."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_tapnext_capability.py'
spec = importlib.util.spec_from_file_location('capability_supervisor_test', SOURCE)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fake-native-supervisor-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repository'
        self.cache = Path(self.temp.name) / 'assets'
        self.root.mkdir(); self.cache.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'Synthetic fixture')
        for name in supervisor.INPUTS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic frozen input\n')
        shutil.copy2(SOURCE, self.root / 'scripts/run_tapnext_capability.py')
        (self.root / 'scripts/tapnext_native_adapter.py').write_text(
            "def run_capability(**kwargs):\n"
            "    return {'status': 'failed', 'frames_completed': 0}\n")
        self.assets = []
        for index, kind in enumerate(['source'] * 5 + ['wheel'] * 3 + ['checkpoint']):
            directory = {'source': 'source', 'wheel': 'wheels', 'checkpoint': 'weights'}[kind]
            name = f'{directory}/synthetic-{index}.bin'
            path = self.cache / name; path.parent.mkdir(exist_ok=True)
            body = f'own generated {kind} {index}\n'.encode(); path.write_bytes(body)
            self.assets.append({'id': f'asset-{index}', 'kind': kind, 'relative_path': name,
                                'bytes': len(body), 'sha256': sha_bytes(body),
                                'url': f'https://example.invalid/never-requested-{index}'})
        self.prep = {'budget': {'historical_ceiling_bytes': 100, 'metadata_allowance_bytes': 20,
                                'limit_bytes': 10000}, 'assets': self.assets,
                     'install_timeout_seconds': 60, 'import_timeout_seconds': 60}
        self.write(self.root / supervisor.INPUTS[0], {'expected_packages': {}, 'preparation': self.prep})
        self.git('add', '.'); self.git('commit', '-qm', 'synthetic frozen inputs')
        self.source_commit = self.git('rev-parse', 'HEAD').strip()
        self.freeze = {'source_commit': self.source_commit,
                       'files': {n: supervisor.digest(self.root / n) for n in supervisor.INPUTS}}
        self.write(self.root / supervisor.FREEZE_PATH, self.freeze)
        self.git('add', supervisor.FREEZE_PATH); self.git('commit', '-qm', 'synthetic receipt only')
        self.freeze_sha = supervisor.digest(self.root / supervisor.FREEZE_PATH)
        self.binding = {'config_sha256': supervisor.digest(self.root / supervisor.INPUTS[0]),
                        'source_sha256': supervisor.digest(self.root / 'scripts/prepare_tapnext_assets.py')}
        self.make_preparation()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, stderr=subprocess.DEVNULL, text=True)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + '\n')

    def load(self, name):
        return json.loads((self.cache / name).read_text())

    def make_preparation(self):
        overlay = self.cache / 'overlay'; overlay.mkdir()
        record = overlay / 'fake-0.dist-info/RECORD'; record.parent.mkdir(); record.write_text('synthetic\n')
        manifest = {**self.binding, 'schema_version': 1, 'files': {
            'fake-0.dist-info/RECORD': {'bytes': record.stat().st_size, 'sha256': supervisor.digest(record)}}}
        self.write(self.cache / 'overlay-manifest.json', manifest)
        manifest_ref = {'relative_path': 'overlay-manifest.json',
                        'sha256': supervisor.digest(self.cache / 'overlay-manifest.json')}
        children = {}
        for prefix in ['install', 'import-check']:
            for stream in ['stdout', 'stderr']:
                (self.cache / f'{prefix}.{stream}.log').write_text(f'synthetic {prefix} {stream}\n')
            children[prefix] = {'command': ['fake-command-never-executed', prefix],
                                'started_utc': '2026-10-18T00:00:00+00:00',
                                'finished_utc': '2026-10-18T00:00:01+00:00',
                                'elapsed_seconds': 1.0, 'exit_code': 0, 'raw_wait_status': 0,
                                'peak_rss_kib': 1,
                                **{s + '_sha256': supervisor.digest(self.cache / f'{prefix}.{s}.log')
                                   for s in ['stdout', 'stderr']}}
            self.write(self.cache / f'{prefix}.json', children[prefix])
        reservations = {}; payload = 0
        for stage in ['dependencies', 'checkpoint']:
            records = []
            selected = [a for a in self.assets if (a['kind'] == 'checkpoint') == (stage == 'checkpoint')]
            for a in selected:
                # Exercise both real preparer statuses without doing any network I/O.
                downloaded = a['id'] != 'asset-0'
                rec = {**self.binding, 'asset': a,
                       'status': 'downloaded_verified' if downloaded else 'reused_verified',
                       'sha256': a['sha256'], 'partial_bytes_stored': 0,
                       'payload_bytes_observed': a['bytes'] if downloaded else 0,
                       'http_payload_attempts': int(downloaded),
                       'observed_payload_sha256': a['sha256'] if downloaded else sha_bytes(b''),
                       'started_utc': '2026-10-18T00:00:00+00:00',
                       'finished_utc': '2026-10-18T00:00:01+00:00'}
                records.append(rec); self.write(self.cache / f"asset-records/{a['id']}.json", rec)
                reservations[a['id']] = a['bytes'] + 1
                payload += rec['payload_bytes_observed']
            ledger = {**self.binding, 'base_ceiling_bytes': 120,
                      'planned_ceiling_bytes': 120 + sum(a['bytes'] + 1 for a in self.assets),
                      'limit_bytes': 10000, 'reservations': dict(reservations),
                      'payload_bytes_observed': payload,
                      'conservative_cumulative_ceiling_bytes': 120 + sum(reservations.values())}
            name = f'acquisition-ledger-{stage}.json'; self.write(self.cache / name, ledger)
            receipt = {**self.binding, 'schema_version': 1, 'stage': stage, 'status': 'completed',
                       'assets': records, 'overlay_manifest': manifest_ref,
                       'ledger_sha256': supervisor.digest(self.cache / name),
                       'ledger_snapshot': {'relative_path': name, 'sha256': supervisor.digest(self.cache / name)}}
            if stage == 'dependencies':
                receipt.update(installation=children['install'], import_check=children['import-check'])
            else:
                receipt['dependency_receipt_sha256'] = supervisor.digest(self.cache / 'preparation-dependencies.json')
            self.write(self.cache / f'preparation-{stage}.json', receipt)
        self.write(self.cache / 'acquisition-ledger.json', ledger)

    def change_receipt(self, stage, change):
        path = f'preparation-{stage}.json'; receipt = self.load(path)
        change(receipt); self.write(self.cache / path, receipt)
        if stage == 'dependencies':
            self.change_receipt('checkpoint', lambda r: r.update(
                dependency_receipt_sha256=supervisor.digest(self.cache / path)))

    def test_valid_exact_git_and_preparation_chain(self):
        self.assertEqual(supervisor.verify_freeze(self.root, self.freeze_sha), self.freeze)
        self.assertEqual(set(supervisor.verify_preparation(self.root, self.cache)), {'dependencies', 'checkpoint'})

    def test_each_worktree_and_index_input_mutation_rejected(self):
        for name in supervisor.INPUTS:
            original = (self.root / name).read_bytes()
            with self.subTest(name=name, surface='worktree'):
                (self.root / name).write_bytes(original + b'\nchanged\n')
                with self.assertRaises(ValueError): supervisor.verify_freeze(self.root, self.freeze_sha)
            with self.subTest(name=name, surface='index'):
                self.git('add', name); (self.root / name).write_bytes(original)
                with self.assertRaises(ValueError): supervisor.verify_freeze(self.root, self.freeze_sha)
            self.git('reset', '-q', 'HEAD', '--', name)

    def test_head_input_mutation_rejected_despite_restored_index_worktree(self):
        name = supervisor.INPUTS[1]; original = (self.root / name).read_bytes()
        (self.root / name).write_bytes(original + b'changed\n'); self.git('add', name); self.git('commit', '-qm', 'alter head')
        self.git('read-tree', self.source_commit); (self.root / name).write_bytes(original)
        # Restore the receipt entry in the index, so the scientific HEAD change is the failure.
        self.git('add', supervisor.FREEZE_PATH)
        with self.assertRaises(ValueError): supervisor.verify_freeze(self.root, self.freeze_sha)

    def test_canonical_receipt_bytes_rejected_in_current_and_index(self):
        p = self.root / supervisor.FREEZE_PATH; original = p.read_bytes(); p.write_bytes(original + b' ')
        with self.assertRaises(ValueError): supervisor.verify_freeze(self.root, self.freeze_sha)
        self.git('add', supervisor.FREEZE_PATH); p.write_bytes(original)
        with self.assertRaises(ValueError): supervisor.verify_freeze(self.root, self.freeze_sha)

    def test_freeze_map_missing_extra_and_source_mismatch_rejected(self):
        original = copy.deepcopy(self.freeze)
        for kind in ['missing', 'extra', 'wrong-source']:
            f = copy.deepcopy(original)
            if kind == 'missing': f['files'].pop(supervisor.INPUTS[0])
            elif kind == 'extra': f['files']['unexpected.py'] = '0' * 64
            else: f['source_commit'] = '0' * 40
            self.write(self.root / supervisor.FREEZE_PATH, f)
            with self.subTest(kind=kind), self.assertRaises((ValueError, subprocess.CalledProcessError)):
                supervisor.verify_freeze(self.root, supervisor.digest(self.root / supervisor.FREEZE_PATH))

    def test_valid_old_source_commit_cannot_bind_new_head_hashes(self):
        name = supervisor.INPUTS[1]
        (self.root / name).write_text('changed prospective protocol\n')
        self.git('add', name); self.git('commit', '-qm', 'new prospective source')
        self.freeze['files'][name] = supervisor.digest(self.root / name)
        self.write(self.root / supervisor.FREEZE_PATH, self.freeze)
        self.git('add', supervisor.FREEZE_PATH); self.git('commit', '-qm', 'claims stale source')
        # HEAD/index/current match the claimed new hashes, and source is a real
        # ancestor; only comparison with the actual source commit catches this.
        with self.assertRaises(ValueError):
            supervisor.verify_freeze(self.root, supervisor.digest(self.root / supervisor.FREEZE_PATH))

    def test_preparation_status_and_source_config_chain_rejected(self):
        original = self.load('preparation-checkpoint.json')
        changes = [('status', 'failed'), ('stage', 'dependencies'),
                   ('config_sha256', '0' * 64), ('source_sha256', '0' * 64),
                   ('dependency_receipt_sha256', '0' * 64)]
        for field, value in changes:
            receipt = copy.deepcopy(original); receipt[field] = value
            self.write(self.cache / 'preparation-checkpoint.json', receipt)
            with self.subTest(field=field), self.assertRaises(ValueError):
                supervisor.verify_preparation(self.root, self.cache)

    def test_minimal_completed_claim_no_longer_passes(self):
        for stage in ['dependencies', 'checkpoint']:
            self.change_receipt(stage, lambda r: r.pop('assets'))
        with self.assertRaises((ValueError, KeyError)): supervisor.verify_preparation(self.root, self.cache)

    def test_failed_or_rewritten_child_or_log_rejected(self):
        original = (self.cache / 'import-check.json').read_bytes()
        for field, val in [('exit_code', 1), ('raw_wait_status', 256), ('terminated_by_parent', True), ('elapsed_seconds', 61)]:
            child = json.loads(original); child[field] = val
            self.write(self.cache / 'import-check.json', child)
            self.change_receipt('dependencies', lambda r: r.update(import_check=child))
            with self.subTest(field=field), self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)
        (self.cache / 'import-check.json').write_bytes(original)
        self.change_receipt('dependencies', lambda r: r.update(import_check=json.loads(original)))
        (self.cache / 'import-check.stdout.log').write_text('changed after recorded exit')
        with self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)

    def test_asset_failure_or_replaced_payload_rejected(self):
        original = self.load('preparation-checkpoint.json')
        for kind in ['failed', 'missing', 'partial', 'unbound-record']:
            r = copy.deepcopy(original)
            if kind == 'failed': r['assets'][0]['status'] = 'failed'
            elif kind == 'missing': r['assets'] = []
            elif kind == 'partial': r['assets'][0]['partial_bytes_stored'] = 1
            else: r['assets'][0]['finished_utc'] = 'changed'
            self.write(self.cache / 'preparation-checkpoint.json', r)
            with self.subTest(kind=kind), self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)
        self.write(self.cache / 'preparation-checkpoint.json', original)
        (self.cache / self.assets[-1]['relative_path']).write_text('same-path different bytes')
        with self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)

    def test_ledger_payload_reservation_budget_and_current_mutations_rejected(self):
        name = 'acquisition-ledger-checkpoint.json'; original = self.load(name)
        for field in ['payload_bytes_observed', 'base_ceiling_bytes', 'planned_ceiling_bytes', 'limit_bytes', 'conservative_cumulative_ceiling_bytes', 'reservations']:
            ledger = copy.deepcopy(original)
            ledger[field] = {} if field == 'reservations' else ledger[field] + 1
            self.write(self.cache / name, ledger)
            self.change_receipt('checkpoint', lambda r: r.update(
                ledger_sha256=supervisor.digest(self.cache / name),
                ledger_snapshot={'relative_path': name, 'sha256': supervisor.digest(self.cache / name)}))
            with self.subTest(field=field), self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)
        self.write(self.cache / name, original)
        self.change_receipt('checkpoint', lambda r: r.update(
            ledger_sha256=supervisor.digest(self.cache / name),
            ledger_snapshot={'relative_path': name, 'sha256': supervisor.digest(self.cache / name)}))
        (self.cache / 'acquisition-ledger.json').write_text('{}')
        with self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)

    def test_overlay_added_changed_and_symlink_rejected(self):
        path = self.cache / 'overlay/fake-0.dist-info/RECORD'; original = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)
        path.write_bytes(original); extra = self.cache / 'overlay/extra.py'; extra.write_text('x')
        with self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)
        extra.unlink(); extra.symlink_to(path)
        with self.assertRaises(ValueError): supervisor.verify_preparation(self.root, self.cache)

    def command(self, *options):
        return [sys.executable, str(self.root / 'scripts/run_tapnext_capability.py'),
                '--asset-cache', str(self.cache), '--bootstapir-cache', str(Path(self.temp.name) / 'absent-old-cache'),
                '--output', str(Path(self.temp.name) / 'output'), '--freeze-sha256', self.freeze_sha, *options]

    def test_verify_only_no_assets_output_runtime_or_adapter_access(self):
        shutil.rmtree(self.cache)
        proc = subprocess.run(self.command('--verify-only'), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['files'], 7)
        self.assertFalse((Path(self.temp.name) / 'output').exists())

    def test_fake_failed_child_has_actual_nonzero_exit(self):
        # Fake frozen adapter has no dependencies or model; only exercises CLI status mapping.
        proc = subprocess.run(self.command('--child'), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['status'], 'failed')

    def test_existing_marker_blocks_before_platform_or_child(self):
        (self.cache / '0005-native-started.json').write_text('existing attempt')
        proc = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Existing scientific attempt marker', proc.stderr)
        self.assertFalse((Path(self.temp.name) / 'output').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
