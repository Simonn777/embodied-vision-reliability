"""Artificial-only allocation and binding tests. No scientific files are opened."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load('_evr0016_test_bindings_common', HERE / 'common.py')
with patch.dict(sys.modules, {'common': common}):
    allocate = load('_evr0016_test_bindings_allocate', HERE / 'allocate.py')
    worker = load('_evr0016_test_bindings_worker', HERE / 'predict_pair.py')
with patch.dict(sys.modules, {'common': common, 'predict_pair': worker}):
    supervisor = load('_evr0016_test_bindings_supervisor', HERE / 'run_diagnostic.py')


class BindingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'artificial-repo'
        self.root.mkdir()
        self.here = self.root / 'experiments/0016-search-headroom'
        self.here.mkdir(parents=True)
        (self.root / 'configs').mkdir()
        (self.root / 'protocols').mkdir()
        self.data = self.root / 'opaque-artificial.h5'
        self.data.write_bytes(b'This is an artificial opaque byte fixture, not HDF5.')
        self.dev = [f'0015-artificial-{i:04d}' for i in range(600)]
        self.keys = self.dev + [f'0022-unread-{i:04d}' for i in range(900)]
        self.old_path = self.root / 'configs/0012-development-manifest.json'
        self.old = {'selected_pair_keys': self.dev[:16],
                    'dataset_sha256': common.sha256(self.data),
                    'dataset_bytes': self.data.stat().st_size}
        self.write(self.old_path, self.old)
        self.protocol = self.root / 'protocols/0016-search-headroom.md'
        self.protocol.write_text('Artificial protocol fixture only.\n')
        self.libraries = self.root / 'configs/0016-libraries.json'
        self.library_values = {'old': {'version': '2.0.5', 'binary_sha256': 'a' * 64},
                               'current': {'version': '3.0.0', 'binary_sha256': 'b' * 64}}
        self.write(self.libraries, self.library_values)
        for filename in common.SOURCE_FILES:
            (self.here / filename).write_text('# Artificial source binding only\n')
        legacy = self.root / 'experiments/0013-pose-headroom'
        legacy.mkdir()
        for filename in ('predict_pair.py', 'run_baseline.py', 'evaluate_sealed.py'):
            (legacy / filename).write_text('# Artificial legacy binding only\n')
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (('ROOT', self.root), ('HERE', self.here),
                            ('PROTOCOL', self.protocol), ('LIBRARIES', self.libraries)):
            self.stack.enter_context(patch.object(common, name, value))
        self.manifest_path = self.root / 'configs/0016-development-manifest.json'
        self.manifest = self.manifest_value()
        self.write(self.manifest_path, self.manifest)

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False))

    def manifest_value(self):
        return {'schema': 'evr0016.development.v1',
                'dataset_sha256': self.old['dataset_sha256'],
                'dataset_bytes': self.old['dataset_bytes'],
                'development_scene': '0015', 'development_scene_pairs': 600,
                'selected_pair_keys': allocate.select(self.keys, self.old['selected_pair_keys']),
                'selection_salt': common.SALT,
                'selection': '64 lowest salted SHA256 from584 previously unread pair keys',
                'excluded_manifest_sha256': common.sha256(self.old_path),
                'root_name_inventory_sha256': hashlib.sha256('\n'.join(sorted(self.keys)).encode()).hexdigest(),
                'held_out_scenes': ['0022'], 'coordinate_convention': 'raw_pixels',
                'access': 'Artificial names only; no numerical arrays.'}

    def cohort(self):
        output = self.root / 'predictions'
        output.mkdir()
        source = common.source_hashes()
        manifest_hash = common.sha256(self.manifest_path)
        rows = []
        for backend in common.BACKENDS:
            for i, key in enumerate(self.manifest['selected_pair_keys']):
                name = f'{backend}-{i:02d}.json'
                record = {'schema': 'evr0016.prediction.v1', 'status': 'completed',
                          'backend': backend, 'pair_index': i, 'key': key,
                          'manifest_sha256': manifest_hash, 'source_sha256': source,
                          'library': self.library_values['old' if backend == 'old_norm' else 'current'],
                          'candidates': [{'seed': seed, 'options': common.options(backend, seed),
                                          'status': 'estimator_error', 'error': 'Artificial receipt'}
                                         for seed in common.SEEDS]}
                self.write(output / name, record)
                process_path = output / (name + '.process.json')
                self.write(process_path, {'exit_code': 0, 'termination': None})
                rows.append({'backend': backend, 'pair_index': i, 'key': key,
                             'filename': name, 'sha256': common.sha256(output / name),
                             'process_sha256': common.sha256(process_path)})
        seal = {'status': 'completed', 'manifest_sha256': manifest_hash,
                'source_sha256': source, 'rows': rows}
        seal_path = output / 'seal.json'
        self.write(seal_path, seal)
        return seal_path, seal

    def test_selection_is_exact_deterministic_and_excludes_old16(self):
        chosen = allocate.select(self.keys, self.old['selected_pair_keys'])
        expected = sorted(self.dev[16:], key=lambda key: hashlib.sha256(
            (common.SALT + key).encode()).hexdigest())[:64]
        self.assertEqual(chosen, expected)
        self.assertEqual(chosen, allocate.select(list(reversed(self.keys)), list(reversed(self.dev[:16]))))
        self.assertEqual(len(set(chosen)), 64)
        self.assertFalse(set(chosen) & set(self.dev[:16]))
        self.assertTrue(all(key.startswith('0015-') for key in chosen))

    def test_selection_rejects_wrong_inventory_and_exclusions(self):
        cases = [(self.keys[1:], self.dev[:16]),
                 (self.keys + [self.dev[0]], self.dev[:16]),
                 (self.keys, self.dev[:15]),
                 (self.keys, self.dev[:15] + ['0015-not-present']),
                 (self.keys, self.dev[:15] + [self.dev[0]])]
        for keys, excluded in cases:
            with self.subTest(keys=len(keys), excluded=excluded[-1]):
                with self.assertRaises(ValueError):
                    allocate.select(keys, excluded)

    def test_allocator_fake_h5_reads_root_names_only_and_never_overwrites(self):
        events = []
        parent = self

        class FakeH5:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def keys(self):
                events.append('root_names')
                return list(parent.keys)

            def __getitem__(self, key):
                raise AssertionError('Numeric or group access forbidden: ' + str(key))

        def open_fake(path, mode):
            self.assertEqual(Path(path), self.data)
            self.assertEqual(mode, 'r')
            events.append('open')
            return FakeH5()

        output = self.root / 'allocated.json'
        argv = ['allocate.py', '--data', str(self.data), '--output', str(output)]
        with patch.dict(sys.modules, {'h5py': types.SimpleNamespace(File=open_fake)}), \
                patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            allocate.main()
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                allocate.main()
        self.assertEqual(events, ['open', 'root_names'])
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(common.load_manifest(output)['selected_pair_keys'], self.manifest['selected_pair_keys'])

    def test_allocator_integrity_failure_occurs_before_fake_h5_open(self):
        self.data.write_bytes(b'Wrong same-purpose artificial bytes')
        open_mock = unittest.mock.Mock(side_effect=AssertionError('HDF5 must not be opened'))
        argv = ['allocate.py', '--data', str(self.data), '--output', str(self.root / 'new.json')]
        with patch.dict(sys.modules, {'h5py': types.SimpleNamespace(File=open_mock)}), patch.object(sys, 'argv', argv):
            with self.assertRaises(ValueError):
                allocate.main()
        open_mock.assert_not_called()
        self.assertFalse((self.root / 'new.json').exists())

    def test_manifest_rejects_key_order_duplicates_exposure_and_dataset_tampering(self):
        mutations = []
        m = copy.deepcopy(self.manifest); m['selected_pair_keys'].reverse(); mutations.append(m)
        m = copy.deepcopy(self.manifest); m['selected_pair_keys'][0] = m['selected_pair_keys'][1]; mutations.append(m)
        m = copy.deepcopy(self.manifest); m['selected_pair_keys'][0] = self.dev[0]; mutations.append(m)
        for field, value in (('schema', 'wrong'), ('selection_salt', 'different'),
                             ('dataset_sha256', '0' * 64), ('dataset_bytes', 1)):
            m = copy.deepcopy(self.manifest); m[field] = value; mutations.append(m)
        for i, m in enumerate(mutations):
            with self.subTest(i=i):
                self.write(self.manifest_path, m)
                with self.assertRaises(ValueError):
                    common.load_manifest(self.manifest_path)

    def test_manifest_rejects_development_and_exclusion_metadata_tampering(self):
        for field, value in (('development_scene', '0022'), ('development_scene_pairs', 599),
                             ('coordinate_convention', 'normalized'), ('held_out_scenes', []),
                             ('excluded_manifest_sha256', '0' * 64),
                             ('root_name_inventory_sha256', 'not-a-digest')):
            with self.subTest(field=field):
                m = copy.deepcopy(self.manifest); m[field] = value
                self.write(self.manifest_path, m)
                with self.assertRaises(ValueError):
                    common.load_manifest(self.manifest_path)

    def test_options_have_correct_legacy_and_nested_current_key_paths(self):
        original = copy.deepcopy(common.base.RANSAC_OPTIONS)
        for seed in common.SEEDS:
            legacy = common.options('old_norm', seed)
            self.assertEqual(legacy['seed'], seed)
            self.assertEqual(legacy['max_epipolar_error'], 1.0)
            self.assertNotIn('ransac', legacy)
            self.assertNotIn('tangent_sampson', legacy)
            for backend in ('current_norm', 'current_ts'):
                opt = common.options(backend, seed)
                self.assertEqual(opt['max_error'], 1.0)
                self.assertIs(opt['tangent_sampson'], backend == 'current_ts')
                self.assertEqual(opt['ransac']['seed'], seed)
                self.assertEqual(opt['ransac']['min_iterations'], 1000)
                self.assertEqual(opt['ransac']['max_iterations'], 100000)
                self.assertEqual(opt['bundle'], {})
                for key in ('seed', 'min_iterations', 'max_iterations', 'max_epipolar_error'):
                    self.assertNotIn(key, opt)
        common.options('current_norm', 0)['ransac']['seed'] = 999
        self.assertEqual(common.options('current_norm', 0)['ransac']['seed'], 0)
        self.assertEqual(common.base.RANSAC_OPTIONS, original)

    def test_unknown_backend_and_noninteger_seed_are_rejected(self):
        for backend, seed in [('missing', 0), ('OLD_NORM', 0), ('old_norm', -1),
                              ('old_norm', 4), ('old_norm', '0'), ('current_ts', True),
                              ('current_norm', 1.0)]:
            with self.subTest(backend=backend, seed=seed):
                with self.assertRaises(ValueError):
                    common.options(backend, seed)

    def test_invoke_spy_verifies_legacy_and_current_argument_positions(self):
        x1, x2 = object(), object()
        cal = {'camera1': {'artificial': 1}, 'camera2': {'artificial': 2}}
        expected = object()
        fake = types.SimpleNamespace(estimate_relative_pose=unittest.mock.Mock(return_value=expected))
        for backend in common.BACKENDS:
            for seed in common.SEEDS:
                with self.subTest(backend=backend, seed=seed):
                    fake.estimate_relative_pose.reset_mock()
                    self.assertIs(worker.invoke(fake, x1, x2, cal, backend, seed), expected)
                    args, kwargs = fake.estimate_relative_pose.call_args
                    self.assertEqual(args[:4], (x1, x2, cal['camera1'], cal['camera2']))
                    self.assertEqual(args[4], common.options(backend, seed))
                    self.assertEqual(kwargs, {'initial_pose': None})
                    self.assertEqual(len(args), 6 if backend == 'old_norm' else 5)
                    if backend == 'old_norm':
                        self.assertEqual(args[5], {})

    def test_worker_native_exceptions_preserve_all_seeds_and_charged_work(self):
        calls, reads = [], []
        library = copy.deepcopy(self.library_values['current'])
        fake_native = types.SimpleNamespace()

        def fail_native(*args, **kwargs):
            calls.append((copy.deepcopy(args[4]), kwargs))
            raise RuntimeError('Artificial ordinary estimator error')

        fake_native.estimate_relative_pose = fail_native

        class FakeH5:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def read_pair(handle, key):
            reads.append(key)
            x = common.np.zeros((10000, 2))
            K = common.np.diag([100., 100., 1.])
            return x, x.copy(), {'K1': K, 'K2': K, 'camera1': {}, 'camera2': {},
                                'normalized_epipolar_threshold': .01}

        output = self.root / 'worker-error-candidates.json'
        with patch.dict(sys.modules, {'poselib': fake_native,
                                      'h5py': types.SimpleNamespace(File=lambda *_: FakeH5())}), \
                patch.object(worker, 'library_metadata', return_value=library), \
                patch.object(common.base, 'read_pair', side_effect=read_pair):
            self.assertEqual(worker.run(self.data, self.manifest_path, output, 'current_ts', 0), 0)
            with self.assertRaises(FileExistsError):
                worker.run(self.data, self.manifest_path, output, 'current_ts', 0)
        record = json.loads(output.read_text())
        self.assertEqual(reads, [self.manifest['selected_pair_keys'][0]])
        self.assertEqual(record['status'], 'completed')
        self.assertEqual([c['seed'] for c in record['candidates']], [0, 1, 2, 3])
        self.assertEqual([c['status'] for c in record['candidates']], ['estimator_error'] * 4)
        self.assertEqual([opt['ransac']['seed'] for opt, _ in calls], [0, 1, 2, 3])
        charged = record['input_seconds']
        for candidate in record['candidates']:
            self.assertNotIn('estimate', candidate)
            charged += candidate['call_seconds'] + candidate['score_seconds']
            self.assertAlmostEqual(candidate['charged_end_seconds'], charged, places=12)
        self.assertAlmostEqual(record['charged_total_seconds'], charged, places=12)

    def test_worker_library_mismatch_precedes_any_fake_h5_access(self):
        open_mock = unittest.mock.Mock(side_effect=AssertionError('Must stop before HDF5'))
        output = self.root / 'wrong-library.json'
        with patch.dict(sys.modules, {'poselib': types.SimpleNamespace(),
                                      'h5py': types.SimpleNamespace(File=open_mock)}), \
                patch.object(worker, 'library_metadata', return_value={'version': 'wrong'}):
            self.assertEqual(worker.run(self.data, self.manifest_path, output, 'old_norm', 0), 1)
        open_mock.assert_not_called()
        record = json.loads(output.read_text())
        self.assertEqual(record['status'], 'error')
        self.assertEqual(record['phase'], 'binding')
        self.assertEqual(record['candidates'], [])

    def fake_child_record(self, command):
        args = dict(zip(command[2::2], command[3::2]))
        backend, index = args['--backend'], int(args['--pair-index'])
        record = {'schema': 'evr0016.prediction.v1', 'status': 'completed',
                  'backend': backend, 'pair_index': index,
                  'key': self.manifest['selected_pair_keys'][index],
                  'manifest_sha256': common.sha256(self.manifest_path),
                  'source_sha256': common.source_hashes(),
                  'library': self.library_values['old' if backend == 'old_norm' else 'current'],
                  'candidates': [{'seed': seed, 'options': common.options(backend, seed),
                                  'status': 'estimator_error', 'error': 'Artificial only'}
                                 for seed in common.SEEDS]}
        return Path(args['--output']), record

    def test_supervisor_rotates_all192_mock_children_and_binds_processes(self):
        calls = []

        def fake_child(command, log, remaining):
            path, record = self.fake_child_record(command)
            calls.append((record['pair_index'], record['backend'], os.environ['PYTHONPATH']))
            self.assertGreater(remaining, 0)
            self.assertLessEqual(remaining, 1800)
            self.write(path, record)
            return {'exit_code': 0, 'termination': None}

        output = self.root / 'supervised-complete'
        with patch.object(supervisor.prior, 'run_child', side_effect=fake_child), \
                patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(supervisor.run(self.data, self.manifest_path, output,
                                           Path('/artificial/old'), Path('/artificial/current')), 0)
        expected = [(i, backend) for i in range(64)
                    for backend in common.BACKENDS[i % 3:] + common.BACKENDS[:i % 3]]
        self.assertEqual([(i, backend) for i, backend, _ in calls], expected)
        for _, backend, path in calls:
            first = '/artificial/old' if backend == 'old_norm' else '/artificial/current'
            self.assertEqual(path, os.pathsep.join([first, '/artificial/old']))
        _, seal, records = common.verify_seal(self.manifest_path, output / 'seal.json')
        self.assertEqual(seal['status'], 'completed')
        self.assertEqual(len(records), 192)

    def test_supervisor_process_failure_or_bad_record_stops_before_second_child(self):
        for mode in ('exit', 'timeout', 'watchdog', 'binding', 'interrupt'):
            with self.subTest(mode=mode):
                output = self.root / ('supervised-failure-' + mode)
                calls = []

                def fake_child(command, log, remaining):
                    calls.append(command)
                    if mode == 'interrupt':
                        raise KeyboardInterrupt('Artificial interrupt after child cleanup')
                    path, record = self.fake_child_record(command)
                    if mode == 'binding':
                        record['source_sha256'] = {}
                    self.write(path, record)
                    return {'exit_code': 1 if mode == 'exit' else 0,
                            'termination': {'timeout': 'pair_timeout',
                                            'watchdog': 'watchdog_error'}.get(mode)}

                with patch.object(supervisor.prior, 'run_child', side_effect=fake_child), \
                        patch.dict(os.environ), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(supervisor.run(self.data, self.manifest_path, output,
                                                   Path('/artificial/old'), Path('/artificial/current')), 1)
                self.assertEqual(len(calls), 1)
                seal = json.loads((output / 'seal.json').read_text())
                self.assertEqual(seal['status'], 'ineligible')
                self.assertEqual(seal['rows'], [])
                self.assertTrue(seal['error'])
                with self.assertRaises(ValueError):
                    common.verify_seal(self.manifest_path, output / 'seal.json')

    def test_source_bindings_detect_change_and_include_all_production_sources(self):
        first = common.source_hashes()
        for filename in common.SOURCE_FILES:
            self.assertIn('experiments/0016-search-headroom/' + filename, first)
        self.assertIn('protocols/0016-search-headroom.md', first)
        self.assertIn('configs/0016-libraries.json', first)
        self.protocol.write_text('Changed artificial protocol')
        second = common.source_hashes()
        self.assertNotEqual(first, second)
        changed = [name for name in first if first[name] != second[name]]
        self.assertEqual(changed, ['protocols/0016-search-headroom.md'])

    def test_full192_cohort_binding_and_source_change(self):
        path, seal = self.cohort()
        manifest, parsed, records = common.verify_seal(self.manifest_path, path)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(parsed, seal)
        self.assertEqual(len(records), 192)
        self.protocol.write_text('Changed after prediction')
        with self.assertRaises(ValueError):
            common.verify_seal(self.manifest_path, path)

    def test_seal_rejects_missing_duplicate_and_reidentified_rows(self):
        path, original = self.cohort()
        variants = []
        s = copy.deepcopy(original); s['rows'].pop(); variants.append(s)
        s = copy.deepcopy(original); s['rows'][-1] = s['rows'][0]; variants.append(s)
        s = copy.deepcopy(original); s['rows'][0]['backend'] = 'not-admitted'; variants.append(s)
        s = copy.deepcopy(original); s['status'] = 'incomplete'; variants.append(s)
        s = copy.deepcopy(original); s['manifest_sha256'] = '0' * 64; variants.append(s)
        for i, seal in enumerate(variants):
            with self.subTest(i=i):
                self.write(path, seal)
                with self.assertRaises(ValueError):
                    common.verify_seal(self.manifest_path, path)

    def test_record_tampering_and_rehashed_identity_input_tampering(self):
        path, original = self.cohort()
        target = path.parent / original['rows'][0]['filename']
        good = json.loads(target.read_text())
        target.write_text(target.read_text() + ' ')
        with self.assertRaises(ValueError):
            common.verify_seal(self.manifest_path, path)
        for field, value in (('pair_index', 1), ('key', '0015-incorrect'),
                             ('status', 'started'), ('manifest_sha256', '0' * 64),
                             ('source_sha256', {})):
            with self.subTest(field=field):
                record = copy.deepcopy(good); record[field] = value
                self.write(target, record)
                seal = copy.deepcopy(original)
                seal['rows'][0]['sha256'] = common.sha256(target)
                self.write(path, seal)
                with self.assertRaises(ValueError):
                    common.verify_seal(self.manifest_path, path)

    def test_seal_rejects_unsafe_relative_path_and_same_bytes_symlink(self):
        path, original = self.cohort()
        for name in ('../outside.json', '/absolute.json', 'sub/file.json'):
            with self.subTest(name=name):
                seal = copy.deepcopy(original); seal['rows'][0]['filename'] = name
                self.write(path, seal)
                with self.assertRaises(ValueError):
                    common.verify_seal(self.manifest_path, path)
        self.write(path, original)
        target = path.parent / original['rows'][0]['filename']
        outside = self.root / 'outside.json'
        outside.write_bytes(target.read_bytes())
        target.unlink(); target.symlink_to(outside)
        with self.assertRaises(ValueError):
            common.verify_seal(self.manifest_path, path)

    def test_rehashed_record_cannot_change_library_seed_or_options(self):
        path, original = self.cohort()
        target = path.parent / original['rows'][0]['filename']
        good = json.loads(target.read_text())
        variants = []
        r = copy.deepcopy(good); r['library']['version'] = 'wrong'; variants.append(r)
        r = copy.deepcopy(good); r['candidates'].pop(); variants.append(r)
        r = copy.deepcopy(good); r['candidates'][0]['seed'] = True; variants.append(r)
        r = copy.deepcopy(good); r['candidates'][1]['seed'] = 0; variants.append(r)
        r = copy.deepcopy(good); r['candidates'][0]['options']['max_iterations'] = 1; variants.append(r)
        for i, record in enumerate(variants):
            with self.subTest(i=i):
                self.write(target, record)
                seal = copy.deepcopy(original); seal['rows'][0]['sha256'] = common.sha256(target)
                self.write(path, seal)
                with self.assertRaises(ValueError):
                    common.verify_seal(self.manifest_path, path)

    def test_process_hash_failure_status_and_symlink_are_rejected(self):
        path, original = self.cohort()
        process = path.parent / (original['rows'][0]['filename'] + '.process.json')
        process.write_text(process.read_text() + ' ')
        with self.assertRaises(ValueError):
            common.verify_seal(self.manifest_path, path)
        for receipt in ({'exit_code': 1, 'termination': None},
                        {'exit_code': 0, 'termination': 'watchdog_error'}):
            with self.subTest(receipt=receipt):
                self.write(process, receipt)
                seal = copy.deepcopy(original); seal['rows'][0]['process_sha256'] = common.sha256(process)
                self.write(path, seal)
                with self.assertRaises(ValueError):
                    common.verify_seal(self.manifest_path, path)
        self.write(process, {'exit_code': 0, 'termination': None})
        seal = copy.deepcopy(original); seal['rows'][0]['process_sha256'] = common.sha256(process)
        self.write(path, seal)
        outside = self.root / 'outside-process.json'
        outside.write_bytes(process.read_bytes())
        process.unlink(); process.symlink_to(outside)
        with self.assertRaises(ValueError):
            common.verify_seal(self.manifest_path, path)

    def test_write_new_is_exclusive_and_rejects_nonfinite_before_creating(self):
        output = self.root / 'exclusive.json'
        common.write_new(output, {'value': 1})
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            common.write_new(output, {'value': 2})
        self.assertEqual(output.read_bytes(), before)
        invalid = self.root / 'nonfinite.json'
        with self.assertRaises((ValueError, TypeError)):
            common.write_new(invalid, {'value': float('nan')})
        self.assertFalse(invalid.exists())


if __name__ == '__main__':
    unittest.main()
