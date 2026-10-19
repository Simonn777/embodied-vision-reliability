"""End-to-end saved-record/GT barrier tests; fake bytes and fake HDF5 only."""
import contextlib
import copy
import hashlib
import importlib.util
import json
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
    spec.loader.exec_module(module)
    return module


c = load('evr0017_integration_common', HERE / 'common.py')
with patch.dict(sys.modules, {'common': c}):
    w = load('evr0017_integration_worker', HERE / 'predict_pair.py')
with patch.dict(sys.modules, {'common': c, 'predict_pair': w}):
    runner = load('evr0017_integration_runner', HERE / 'run_diagnostic.py')
ev = load('evr0017_integration_evaluator', HERE / 'evaluate_saved.py')
I = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return c.sha256(path)


class Integration(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.here = self.root / 'experiments/0017-depth-information'
        self.data = self.root / 'opaque-cache'
        self.data.mkdir()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (('ROOT', self.root), ('HERE', self.here),
                            ('PROTOCOL', self.root / 'protocols/0017-depth-information.md'),
                            ('ASSET_PROTOCOL', self.root / 'protocols/0017-depth-assets.md'),
                            ('LIBRARIES', self.root / 'configs/0017-libraries.json')):
            self.stack.enter_context(patch.object(c, name, value))
        source_files = [self.here / name for name in c.SOURCE_FILES] + [c.PROTOCOL, c.ASSET_PROTOCOL,
            self.root / 'experiments/0013-pose-headroom/evaluate_sealed.py']
        for path in source_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Artificial source declaration only\n')
        self.library = {'version': '3.0.0', 'binary_sha256': 'a' * 64}
        write(c.LIBRARIES, {'current': self.library})
        scenes, selected = [], []
        for scene in c.SCENES:
            filename = scene + '_roma_moge.h5'
            data = self.data / filename
            data.write_bytes(b'FAKE HDF5, NEVER OPEN AS REAL HDF5')
            keys = ['pair-%02d' % i for i in range(20)]
            chosen = sorted(keys, key=lambda k: (hashlib.sha256((c.SALT + scene + '\0' + k).encode()).hexdigest(), k))[:16]
            scene_row = {'scene': scene, 'filename': filename, 'url': c.BASE_URL + filename,
                         'sha256': c.sha256(data), 'bytes': data.stat().st_size,
                         'root_keys': sorted(keys), 'selected_pair_keys': chosen}
            scenes.append(scene_row)
            for key in chosen:
                selected.append({'pair_index': len(selected), 'scene': scene, 'key': key, 'filename': filename,
                                 'dataset_sha256': scene_row['sha256'], 'dataset_bytes': scene_row['bytes']})
        self.stack.enter_context(patch.object(c, 'SCENE_BYTES', tuple(s['bytes'] for s in scenes)))
        self.manifest = {'schema': 'evr0017.development.v1', 'selection_salt': c.SALT,
                         'assets_sha256': c.ASSETS_SHA256, 'coordinate_convention': 'authored_raw_pixels',
                         'development_forever': list(c.SCENES), 'scenes': scenes, 'selected_pairs': selected,
                         'source_sha256': {str(p.relative_to(self.root)): c.sha256(p)
                             for p in (self.here / 'acquire_and_allocate.py', c.ASSET_PROTOCOL)}}
        self.mp = self.root / 'configs/0017-development-manifest.json'
        self.mh = write(self.mp, self.manifest)
        self.sources = c.source_hashes()
        self.folder = self.root / 'predictions'
        self.folder.mkdir()
        self.sp = self.folder / 'seal.json'
        self.output = self.root / 'evaluation.json'
        self.events = []
        self.truth_direction = [1., 0., 0.]
        self.make_seal()
        self.stack.enter_context(patch.dict(sys.modules, {'common': c, 'run_diagnostic': runner,
            'h5py': types.SimpleNamespace(File=self.fake_h5)}))

    def make_seal(self):
        assets = {s['filename']: {'bytes': s['bytes'], 'sha256': s['sha256']} for s in self.manifest['scenes']}
        ih = write(self.folder / 'integrity.json', {'manifest_sha256': self.mh, 'source_sha256': self.sources,
                     'library': self.library, 'verified_assets': assets})
        rows = []
        for pair in self.manifest['selected_pairs']:
            for order, arm in enumerate(c.ARMS):
                raw = {**runner.identity(pair, arm, self.mh, self.sources, self.library),
                    'status': 'completed', 'phase': 'complete',
                    'input_arrays': {name: {'shape': [8, 2] if name.startswith('x') else [8],
                        'dtype': 'float64', 'sha256': 'c' * 64} for name in ('x1', 'x2', 'depth1', 'depth2')},
                    'calibration': {'synthetic': True}, 'input_diagnostics': {'count': 8, 'all_finite': True},
                    'estimate': {'R': I, 't': [1., 0., 0.]},
                    'mask_length': 8, 'kept_count': 8, 'native_inliers_packbits_big_hex': 'ff',
                    'mask_sha256': hashlib.sha256(bytes([1] * 8)).hexdigest(),
                    'final_scores': {'pixel_msac': float(order + 1)},
                    'timings': {'input_seconds': .1, 'call_seconds': .2, 'score_seconds': .1,
                                'charged_total_seconds': .5},
                    'geometry': None, 'depth_diagnostics': None}
                if arm.startswith('depth_'):
                    raw['geometry'] = {'R': I, 't': [1., 0., 0.], 'scale': 1., 'shift1': 0., 'shift2': 0.}
                    raw['depth_diagnostics'] = {'returned_mask': {'denominator': 8, 'native_branch_inactive_count': 0}}
                process = {'exit_code': 0, 'termination': None, 'wall_seconds': .75,
                           'sampled_peak_rss_kib': 100, 'watchdog_error': None}
                index = pair['pair_index']
                if index == 0 or (index == 2 and arm == 'e_ts') or (index == 1 and arm == 'depth_scale'):
                    for field in ('estimate', 'geometry', 'depth_diagnostics', 'final_scores', 'mask_length',
                                  'kept_count', 'native_inliers_packbits_big_hex', 'mask_sha256'):
                        raw.pop(field, None)
                    raw.update(status='error', phase='error', failure_kind='estimator', error='Artificial failure')
                    raw['timings'].pop('score_seconds')
                    process['exit_code'] = 1
                    if index == 1:
                        raw.update(status='started', phase='estimator')
                        raw['timings'] = {'input_seconds': .1}
                        process.update(exit_code=-9, termination='pair_timeout', wall_seconds=90.)
                    elif index == 2:
                        raw.update(status='started', phase='estimator')
                        raw['timings'] = {'input_seconds': .1}
                        process['exit_code'] = -11
                stem = 'pair-%02d-%s' % (index, arm)
                log = self.folder / (stem + '.log')
                log.write_bytes(b'Artificial child log only\n')
                process.update(log_path=log.name, log_sha256=c.sha256(log))
                raw_path = self.folder / (stem + '.worker.json')
                rh = write(raw_path, raw)
                process.update(raw_worker_path=raw_path.name, raw_worker_sha256=rh)
                pp = self.folder / (stem + '.process.json')
                ph = write(pp, process)
                final = runner.wrap_record(raw, runner.identity(pair, arm, self.mh, self.sources, self.library), process)
                fp = self.folder / (stem + '.json')
                fh = write(fp, final)
                rows.append({'pair_index': index, 'scene': pair['scene'], 'key': pair['key'], 'arm': arm,
                    'path': fp.name, 'sha256': fh, 'raw_path': raw_path.name, 'raw_sha256': rh,
                    'process_path': pp.name, 'process_sha256': ph, 'status': final['status']})
        self.seal = {'schema': 'evr0017.seal.v1', 'status': 'completed', 'manifest_sha256': self.mh,
            'source_sha256': self.sources, 'library': self.library, 'verified_assets': assets,
            'integrity_sha256': ih, 'rows': rows}
        write(self.sp, self.seal)

    def fake_h5(self, path, mode):
        self.assertEqual(mode, 'r')
        scene = next(s for s in self.manifest['scenes'] if s['filename'] == Path(path).name)
        self.events.append(('open', scene['scene']))
        parent = self

        class Field:
            def __init__(self, key, field): self.key, self.field = key, field
            def __getitem__(self, sl):
                parent.assertEqual(sl, slice(None))
                parent.events.append(('value', scene['scene'], self.key, self.field))
                return c.np.asarray(I if self.field == 'R' else parent.truth_direction)

        class Pair:
            def __init__(self, key): self.key = key
            def __getitem__(self, name):
                parent.assertIn(name, ('R', 't'))
                return Field(self.key, name)

        class File:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __getitem__(self, key):
                parent.assertIn(key, scene['selected_pair_keys'])
                return Pair(key)
        return File()

    def evaluate(self):
        return ev.evaluate(self.data, self.mp, self.sp, self.output)

    def test_complete_sealed_mixed_coverage_retains_all80_and_exact_gt_allowlist(self):
        original_prepare = ev.prepare_records
        def prepare(*args):
            self.assertFalse(self.events)
            value = original_prepare(*args)
            self.events.append(('selections_fixed',))
            return value
        with patch.object(ev, 'prepare_records', side_effect=prepare):
            result = self.evaluate()
        self.assertEqual(self.events[0], ('selections_fixed',))
        self.assertEqual(sum(e[0] == 'open' for e in self.events), 5)
        self.assertEqual(sum(e[0] == 'value' for e in self.events), 160)
        for arm, successes in (('e_norm', 79), ('e_ts', 78), ('depth_scale', 78), ('depth_affine', 79)):
            summary = result['summaries'][arm]
            self.assertEqual(summary['pair_count'], 80)
            self.assertEqual(summary['status_counts']['completed'], successes)
            self.assertAlmostEqual(summary['auc_fraction']['5'], successes / 80)
            self.assertEqual(summary['success_only_auc_fraction']['5'], 1.)
        self.assertEqual(result['rows'][0]['choices']['pixel_msac_external']['penalized_max_degrees'], 180.)
        self.assertAlmostEqual(result['external_selector_charged_seconds'], 319 * .75 + 90.)
        self.assertTrue(all(v['e_norm']['pair_count'] == 16 for v in result['scene_summaries'].values()))
        self.assertTrue(all(v['e_norm']['pair_count'] == 64 for v in result['leave_one_scene_out_descriptive'].values()))

    def test_bad_seal_and_source_fail_before_any_fake_gt_open(self):
        for mutation in ('incomplete', 'duplicate', 'source', 'rawbytes', 'processbytes'):
            with self.subTest(mutation=mutation):
                seal_bytes = self.sp.read_bytes()
                source = self.here / 'predict_pair.py'; source_bytes = source.read_bytes()
                raw = self.folder / self.seal['rows'][0]['raw_path']; raw_bytes = raw.read_bytes()
                proc = self.folder / self.seal['rows'][0]['process_path']; proc_bytes = proc.read_bytes()
                value = copy.deepcopy(self.seal)
                if mutation == 'incomplete': value['rows'].pop()
                elif mutation == 'duplicate': value['rows'][-1] = value['rows'][0]
                elif mutation == 'source': source.write_text('Changed artificial source')
                elif mutation == 'rawbytes': raw.write_bytes(raw_bytes + b' ')
                else: proc.write_bytes(proc_bytes + b' ')
                write(self.sp, value)
                with self.assertRaises(ValueError): self.evaluate()
                self.assertEqual(self.events, [])
                self.assertFalse(self.output.exists())
                self.sp.write_bytes(seal_bytes); source.write_bytes(source_bytes)
                raw.write_bytes(raw_bytes); proc.write_bytes(proc_bytes)

    def test_cache_byte_or_manifest_change_fails_before_any_fake_gt_open(self):
        path = self.data / self.manifest['scenes'][-1]['filename']
        before = path.read_bytes()
        path.write_bytes(b'x' * len(before))
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.events, [])
        self.assertFalse(self.output.exists())
        path.write_bytes(before)
        manifest = copy.deepcopy(self.manifest)
        manifest['selected_pairs'][0]['dataset_sha256'] = '0' * 64
        write(self.mp, manifest)
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.events, [])

    def test_rehashed_final_record_cannot_hide_different_input(self):
        row = self.seal['rows'][7]
        fp = self.folder / row['path']
        record = json.loads(fp.read_text())
        record['input_arrays']['depth1']['sha256'] = 'e' * 64
        row['sha256'] = write(fp, record)
        write(self.sp, self.seal)
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.events, [])

    def test_gt_change_never_reselects_deployed_arm(self):
        self.truth_direction = [-1., 0., 0.]
        result = self.evaluate()
        self.assertTrue(all(r['choices']['pixel_msac_external']['arm'] == 'e_norm' for r in result['rows'][1:]))
        self.assertEqual(result['summaries']['e_norm']['auc_fraction']['5'], 0.)


if __name__ == '__main__':
    unittest.main()
