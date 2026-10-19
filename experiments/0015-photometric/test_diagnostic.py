"""Artificial cohort supervision/evaluation tests; no scientific assets or labels."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


runner = load('_test0015_diagnostic_runner', HERE/'run_diagnostic.py')
with patch.dict(sys.modules, {'run_diagnostic': runner}):
    evaluator = load('_test0015_diagnostic_evaluator', HERE/'evaluate_diagnostic.py')


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest, self.baseline = self.root/'manifest.json', self.root/'baseline-seal.json'
        self.data, self.receipt = self.root/'artificial.h5', self.root/'receipt.json'
        self.manifest.write_text('{"artificial":true}')
        self.baseline.write_text('{"artificial_seal":true}')
        self.receipt.write_text('{"artificial_receipt":true}')
        self.data.write_bytes(b'Only artificial opaque bytes; never HDF5 decoded')
        self.output = self.root/'operation'
        self.mh, self.bh = runner.sha(self.manifest), runner.sha(self.baseline)
        self.metadata = {'dataset_bytes': self.data.stat().st_size, 'dataset_sha256': runner.sha(self.data)}
        self.sources = {'experiments/0015-photometric/estimate_pair.py': 'a'*64,
                        'experiments/0013-pose-headroom/predict_pair.py': 'b'*64}
        self.photo_sources = {'artificial_source': 'c'*64}
        self.pose = {'R': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], 't': [1., 0., 0.]}
        self.originals, self.photos = [], []
        for i in range(16):
            old_row = {'pair_index': i, 'key': '0015-artificial-%02d' % i,
                       'sha256': '%064x' % (i+1)}
            old = {'library': {'version': 'artificial'}, 'calibration': {'camera': 'artificial'},
                   'estimate': copy.deepcopy(self.pose), 'call_seconds': .123}
            self.originals.append((old_row, old))
            self.photos.append({'schema': 'evr0015.photometric.v1', 'status': 'completed',
                'pair_index': i, 'key': old_row['key'],
                'dataset_sha256_declared': self.metadata['dataset_sha256'],
                'mask_sha256': 'd'*64, 'kept_count': 10000, 'selected_count': 9,
                'initialized_count': 9, 'admitted_count': 9, 'target_coordinate_changed_count': 9,
                'queries': [{'index': k, 'admitted': True} for k in range(9)],
                'timing_seconds': {'total': .01}})
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, 'bindings', lambda: self.sources.copy()))
        self.stack.enter_context(patch.object(runner, 'RECEIPT', self.receipt))
        self.stack.enter_context(patch.object(runner.metrics, 'verify_seal', self.verify_baseline))
        self.stack.enter_context(patch.object(runner.geometry, 'source_bindings', lambda base: self.photo_sources))
        self.stack.enter_context(patch.object(runner.geometry, 'verify_inputs', self.verify_photo))
        self.stack.enter_context(patch.object(runner.infrastructure, 'expected_library', lambda: {'fake': True}))
        self.calls, self.fail_at, self.tamper_last = [], None, False

    def verify_baseline(self, manifest, baseline):
        self.assertEqual(Path(manifest), self.manifest)
        self.assertEqual(Path(baseline), self.baseline)
        return self.metadata, {}, self.originals, self.mh, self.bh

    def verify_photo(self, base, metrics, manifest, baseline, path, digest, index):
        record, actual = metrics.load_hashed_json(path)
        self.assertEqual(actual, digest)
        if record.get('status') != 'completed' or record.get('pair_index') != index:
            raise ValueError('artificial photo status or index invalid')
        return self.metadata, self.originals[index][1], record, None, None

    def write(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
        return runner.sha(path)

    def geometry_record(self, photo, index, digest):
        old_row, old = self.originals[index]
        n = photo['admitted_count']
        ids = [x['index'] for x in photo['queries'] if x['admitted']]
        arms, calls = {}, 0
        for name in runner.geometry.ARM_ORDER:
            count = 10000 if name == 'full_refined' else n
            if name != 'full_refined' and n < 8:
                arms[name] = {'status': 'unavailable', 'reason': 'fewer_than_eight_successful_queries',
                              'input_count': n, 'native_calls': 0}
            else:
                calls += 1
                arms[name] = {'status': 'completed', 'input_count': count, 'native_calls': 1,
                    'estimate': copy.deepcopy(self.pose), 'call_seconds': .001,
                    'returned_ransac_stats': {'inliers': [False]*count},
                    'returned_inlier_indices_original': []}
        return {'schema': 'evr0015.geometric.v1', 'status': 'completed', 'pair_index': index,
            'key': old_row['key'], 'manifest_sha256': self.mh, 'baseline_seal_sha256': self.bh,
            'baseline_prediction_sha256': old_row['sha256'],
            'dataset_sha256_declared': photo['dataset_sha256_declared'],
            'mask_sha256': photo['mask_sha256'], 'kept_count': photo['kept_count'],
            'photometric_sha256': digest, 'library': old['library'], 'calibration': old['calibration'],
            'settings': runner.base.SETTINGS, 'source_sha256': self.sources[
                'experiments/0015-photometric/estimate_pair.py'], 'baseline_worker_sha256': self.sources[
                'experiments/0013-pose-headroom/predict_pair.py'],
            'photometric_source_bindings': self.photo_sources, 'selected_count': photo['selected_count'],
            'admitted_count': n, 'target_coordinate_changed_count': photo['target_coordinate_changed_count'],
            'photometric_queries': photo['queries'], 'full_correspondence_count': 10000,
            'full_original_reference': {'baseline_prediction_sha256': old_row['sha256'],
                'estimate': old['estimate'], 'rerun': False},
            'arm_order': list(runner.geometry.ARM_ORDER), 'failed_calls': 0,
            'success_query_indices': ids, 'arms': arms, 'native_calls_started': calls,
            'native_calls_completed': calls, 'timing_seconds': {'worker_elapsed': .005}}

    def make_phase(self, phase, photo_path=None):
        directory = self.root/phase
        directory.mkdir(exist_ok=True)
        rows = []
        for i, photo in enumerate(self.photos):
            if phase == 'photometric':
                record = photo
            else:
                photo_digest = runner.sha(self.root/'photometric'/('pair-%02d.json' % i))
                record = self.geometry_record(photo, i, photo_digest)
            name = 'pair-%02d.json' % i
            digest = self.write(directory/name, record)
            process = {'exit_code': 0, 'termination': None, 'watchdog_error': None,
                       'raw_worker_sha256': digest}
            pname = 'pair-%02d.process.json' % i
            pdigest = self.write(directory/pname, process)
            rows.append({'pair_index': i, 'key': self.originals[i][0]['key'], 'status': 'completed',
                         'path': name, 'sha256': digest, 'process_path': pname, 'process_sha256': pdigest})
        previous = runner.sha(photo_path) if photo_path else None
        seal = {**runner.phase_identity(phase, self.sources, self.mh, self.bh,
                    self.metadata['dataset_sha256'], previous),
                'complete': True, 'eligible_for_next_phase': True, 'error': None,
                'rows': rows, 'wall_seconds_since_operation_start': .1}
        path = directory/'seal.json'
        self.write(path, seal)
        return path, seal

    def fake_child(self, command, log, remaining):
        phase = 'photometric' if Path(command[1]).name == 'predict_pair.py' else 'geometric'
        index = int(command[command.index('--pair-index')+1])
        target = Path(command[command.index('--output')+1])
        self.calls.append((phase, index))
        self.assertGreater(remaining, 0)
        if phase == 'geometric':
            # A genuine all16 verifier runs in the fake child at the crossing.
            photo_seal = target.parents[1]/'photometric/seal.json'
            _, _, photos = runner.verify_phase(self.manifest, self.baseline, photo_seal, 'photometric')
            self.assertEqual(len(photos), 16)
            self.assertEqual(self.calls[:16], [('photometric', i) for i in range(16)])
            photo_path = Path(command[command.index('--photometric-record')+1])
            photo_digest = command[command.index('--photometric-sha256')+1]
            photo = json.loads(photo_path.read_text())
            record = self.geometry_record(photo, index, photo_digest)
        else:
            record = self.photos[index]
        self.write(target, record)
        if self.tamper_last and phase == 'photometric' and index == 15:
            (target.parent/'pair-00.json').write_text('{}')
        return {'exit_code': 1 if self.fail_at == (phase, index) else 0,
                'termination': None, 'watchdog_error': None, 'wall_seconds': .001}

    def run_fake(self):
        with patch.object(runner.infrastructure, 'run_child', self.fake_child), contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.data, self.manifest, self.baseline, self.root/'no-real-images',
                              self.receipt, self.output)

    def test_full32_fake_children_cross_barrier_only_after_complete16(self):
        self.assertEqual(self.run_fake(), 0)
        self.assertEqual(self.calls, [(p, i) for p in runner.PHASES for i in range(16)])
        operation = json.loads((self.output/'operation.json').read_text())
        self.assertTrue(operation['eligible_for_evaluation'])
        self.assertFalse(operation['labels_accessed'])
        with self.assertRaises(FileExistsError):
            self.run_fake()
        self.assertEqual(len(self.calls), 32)

    def test_failed_photo_stops_before_pose_and_preserves_ineligible_seal(self):
        self.fail_at = ('photometric', 3)
        self.assertEqual(self.run_fake(), 1)
        self.assertEqual(self.calls, [('photometric', i) for i in range(4)])
        seal = json.loads((self.output/'photometric/seal.json').read_text())
        self.assertFalse(seal['complete'])
        self.assertEqual(len(seal['rows']), 3)
        self.assertFalse((self.output/'geometric').exists())

    def test_photo_tampered_after_child_success_blocks_phase_crossing(self):
        self.tamper_last = True
        self.assertEqual(self.run_fake(), 1)
        self.assertEqual(self.calls, [('photometric', i) for i in range(16)])
        self.assertFalse((self.output/'geometric').exists())
        self.assertFalse(json.loads((self.output/'operation.json').read_text())['eligible_for_evaluation'])

    def test_phase_requires16_and_safe_immutable_records_and_processes(self):
        path, seal = self.make_phase('photometric')
        runner.verify_phase(self.manifest, self.baseline, path, 'photometric')
        for mutation, pattern in [('missing', 'all16'), ('unsafe', 'unsafe'),
                                  ('hash', 'hash mismatch'), ('process', 'process failed'),
                                  ('source', 'binding differs')]:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(seal)
                if mutation == 'missing':
                    changed['rows'].pop()
                elif mutation == 'unsafe':
                    changed['rows'][0]['path'] = '../outside.json'
                elif mutation == 'hash':
                    changed['rows'][0]['sha256'] = '0'*64
                elif mutation == 'process':
                    changed['rows'][0]['process_sha256'] = '0'*64
                else:
                    changed['source_sha256'] = {}
                self.write(path, changed)
                with self.assertRaisesRegex(ValueError, pattern):
                    runner.verify_phase(self.manifest, self.baseline, path, 'photometric')
        self.write(path, seal)
        linked = path.parent/'linked.json'
        linked.symlink_to(path.parent/'pair-00.json')
        seal['rows'][0]['path'] = linked.name
        self.write(path, seal)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            runner.verify_phase(self.manifest, self.baseline, path, 'photometric')

    def test_unavailable_subset_is_coverage_but_native_failure_is_not(self):
        photo = copy.deepcopy(self.photos[0])
        photo.update(admitted_count=7, target_coordinate_changed_count=7)
        for q in photo['queries'][7:]:
            q['admitted'] = False
        record = self.geometry_record(photo, 0, 'e'*64)
        old_row, old = self.originals[0]
        runner.validate_geometry(record, photo, old, old_row, 0, 'e'*64, self.mh, self.bh, self.sources)
        self.assertEqual(record['native_calls_completed'], 1)
        for mutation in ('full_failure', 'subset_fake_call', 'count', 'mask', 'inlier_map'):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(record)
                if mutation == 'full_failure':
                    bad['arms']['full_refined']['status'] = 'estimator_error'
                elif mutation == 'subset_fake_call':
                    bad['arms']['subset_refined']['native_calls'] = 1
                elif mutation == 'count':
                    bad['native_calls_started'] = 2
                elif mutation == 'mask':
                    bad['arms']['full_refined']['returned_ransac_stats']['inliers'].pop()
                else:
                    bad['arms']['full_refined']['returned_inlier_indices_original'] = [0]
                with self.assertRaises(ValueError):
                    runner.validate_geometry(bad, photo, old, old_row, 0, 'e'*64,
                                             self.mh, self.bh, self.sources)

    def test_native_phase_failure_makes_cohort_ineligible(self):
        self.fail_at = ('geometric', 1)
        self.assertEqual(self.run_fake(), 1)
        self.assertEqual(len(self.calls), 18)
        seal = json.loads((self.output/'geometric/seal.json').read_text())
        self.assertFalse(seal['eligible_for_next_phase'])
        self.assertEqual(len(seal['rows']), 1)

    def test_label_file_not_read_when_either_phase_verification_fails(self):
        truth = self.root/'forbidden-labels.json'
        truth.write_text('must not be read')
        for fail_call in (1, 2):
            with self.subTest(fail_call=fail_call):
                calls = []

                def verify(*args):
                    calls.append(args[3])
                    if len(calls) == fail_call:
                        raise ValueError('incomplete artificial phase')
                    return {}, 'x'*64, []

                with patch.object(runner, 'verify_phase', verify), patch.object(
                        evaluator.metrics, 'load_hashed_json', side_effect=AssertionError('label read')):
                    with self.assertRaisesRegex(ValueError, 'incomplete artificial'):
                        evaluator.evaluate(self.manifest, self.baseline, self.root/'photo',
                            self.root/'pose', truth, self.root/'score.json')

    def test_evaluation_penalizes_unavailable_subsets_in_all16(self):
        self.photos[0].update(admitted_count=7, target_coordinate_changed_count=7)
        for q in self.photos[0]['queries'][7:]:
            q['admitted'] = False
        photo_path, _ = self.make_phase('photometric')
        pose_path, _ = self.make_phase('geometric', photo_path)
        truth_path = self.root/'artificial-labels.json'
        truth = {'seal_sha256': self.bh, 'rows': [{'pair_index': i, 'key': row['key'],
            'ground_truth': {'R_original': self.pose['R'], 't_original': self.pose['t']},
            'errors': {'max_degrees': 0.}} for i, (row, _) in enumerate(self.originals)]}
        digest = self.write(truth_path, truth)
        with patch.object(evaluator, 'BASELINE_EVALUATION_SHA', digest):
            result = evaluator.evaluate(self.manifest, self.baseline, photo_path, pose_path,
                                        truth_path, self.root/'score.json')
        self.assertEqual(len(result['rows']), 16)
        self.assertEqual(result['coverage']['subset_unavailable_pairs'], 1)
        self.assertEqual(result['rows'][0]['errors']['subset_original']['max_degrees'], 180.)
        self.assertEqual(result['rows'][0]['errors']['subset_refined']['max_degrees'], 180.)
        self.assertEqual(result['auc']['subset_refined']['auc_percent']['5'], 93.75)
        self.assertEqual(result['rows'][0]['full_original_native_call_seconds'], .123)
        self.assertFalse(result['expansion_gate']['passed'])

    def test_decision_exact_prospective_thresholds_and_every_guard(self):
        auc = {mode: {'auc_percent': {'5': 80., '20': 95.}} for mode in evaluator.MODES}
        auc['full_refined']['auc_percent'] = {'5': 80.25, '20': 94.5}
        auc['subset_refined']['auc_percent']['5'] = 82.
        rows = [{'admitted_count': 8 if i < 12 else 7, 'errors': {
            'full_original': {'max_degrees': 1.}, 'full_refined': {'max_degrees': .9 if i < 10 else 1.}}}
            for i in range(16)]
        self.assertTrue(evaluator.decision(auc, rows)['passed'])
        for case in ('full_gain', 'subset_gain', 'loss20', 'coverage', 'usable', 'improvements'):
            with self.subTest(case=case):
                a, r = copy.deepcopy(auc), copy.deepcopy(rows)
                if case == 'full_gain':
                    a['full_refined']['auc_percent']['5'] = 80.249999
                elif case == 'subset_gain':
                    a['subset_refined']['auc_percent']['5'] = 81.999999
                elif case == 'loss20':
                    a['full_refined']['auc_percent']['20'] = 94.499999
                elif case == 'coverage':
                    r.pop()
                elif case == 'usable':
                    r[11]['admitted_count'] = 7
                else:
                    r[9]['errors']['full_refined']['max_degrees'] = 1.-1e-10
                self.assertFalse(evaluator.decision(a, r)['passed'])


if __name__ == '__main__':
    unittest.main()
