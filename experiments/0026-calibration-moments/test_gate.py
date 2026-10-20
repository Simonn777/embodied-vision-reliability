"""Manufactured saved records only; no scientific inputs, native calls or fits."""
import copy
from contextlib import redirect_stdout
import io as text_io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate_saved as evaluator


SLOTS = ('continued_tc', 'G_neutral', 'G_incumbent', 'G_lift',
         'F_neutral', 'F_incumbent', 'F_lift')
SOURCES = {'manufactured.py': 'f' * 64}
NATIVE = {'library_sha256': 'a' * 64, 'receipt_sha256': 'b' * 64}


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def pose(angle):
    a = math.radians(angle)
    return {'R': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            't': [math.cos(a), math.sin(a), 0.]}


def identity(index):
    return {'pair_index': index, 'scene': 'scene-' + str(index // 16),
            'key': 'pair-' + str(index), 'filename': 'manufactured.h5',
            'dataset_bytes': 1, 'dataset_sha256': 'c' * 64}


def select(record):
    bank = {'incumbent': record['incumbent'], **record['attempts']}
    record['selectors'] = {}
    for name, members in evaluator.SELECTORS.items():
        eligible = [key for key in members if bank[key]['status'] == 'completed']
        # Enumerating the explicit declared order is an independent tie rule.
        winner = eligible[0]
        for key in eligible[1:]:
            if bank[key]['final_scores']['pixel_msac'] < bank[winner]['final_scores']['pixel_msac']:
                winner = key
        record['selectors'][name] = {
            'members': list(members), 'selected_attempt': winner,
            'estimate': copy.deepcopy(bank[winner]['estimate']),
            'score': bank[winner]['final_scores']['pixel_msac'],
            'charged_seconds': .01,
        }
    return record


def record(index, primary_error=.5):
    def candidate(score, error):
        return {'status': 'completed', 'estimate': pose(error),
                'geometry': {**pose(error), 'scale': 1., 'shift1': 0., 'shift2': 0.},
                'final_scores': {'pixel_msac': score}}
    out = {'status': 'completed', 'pair_index': index, 'identity': identity(index),
           'source_sha256': SOURCES, 'prior_baseline_sha256': 'd' * 64,
           'native_binding': NATIVE,
           'worker_timings': {'binding_seconds': .01, 'input_seconds': .01, 'total_seconds': .1},
           'incumbent': candidate(10., 1.5),
           'attempts': {name: candidate(11., 1.5) for name in SLOTS}}
    out['attempts']['G_lift'] = candidate(9., primary_error)
    return select(out)


def fake_metric(R, t, gt_R, gt_t):
    assert R == gt_R
    assert gt_t == [1., 0., 0.]
    return {'rotation_degrees': 0.,
            'translation_degrees': math.degrees(math.atan2(abs(t[1]), t[0])),
            'max_degrees': math.degrees(math.atan2(abs(t[1]), t[0]))}, {}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sealed(self):
        folder = self.root / 'predictions'
        folder.mkdir()
        job = {'schema': 'evr0026.job.v1', 'commit': '1' * 40,
               'source_sha256': SOURCES, 'native': NATIVE,
               'selected_pairs': [identity(i) for i in range(80)],
               'prior': {'baselines': {str(i): 'd' * 64 for i in range(80)},
                         'original_seal': 'e' * 64, 'TC_seal': 'f' * 64,
                         'legacy_sources': {}}}
        write(folder / 'job.json', job)
        rows = []
        for i in range(80):
            stem = 'pair-%02d' % i
            write(folder / (stem + '.json'), record(i))
            write(folder / (stem + '.process.json'), {
                'exit_code': 0, 'termination': None, 'wall_seconds': .2,
                'sampled_peak_rss_kib': 100})
            write(folder / (stem + '.partial.json'), {'status': 'started', 'pair_index': i})
            (folder / (stem + '.log')).write_text('manufactured worker completed\n')
            row = {'pair_index': i}
            for key, suffix, hashkey in (
                    ('path', '.json', 'sha256'), ('process', '.process.json', 'process_sha256'),
                    ('partial', '.partial.json', 'partial_sha256'), ('log', '.log', 'log_sha256')):
                row[key] = stem + suffix
                row[hashkey] = evaluator.io.digest(folder / row[key])
            rows.append(row)
        seal = {'schema': 'evr0026.seal.v1', 'status': 'completed',
                'source_commit': job['commit'], 'source_sha256': SOURCES,
                'job_sha256': evaluator.io.digest(folder / 'job.json'), 'rows': rows}
        write(folder / 'seal.json', seal)
        outer = {'schema': 'evr0026.outer.v1', 'status': 'completed', 'exit_code': 0,
                 'error': None, 'completed_pairs': 80, 'source_commit': job['commit'],
                 'wall_seconds': 20., 'operation_limit_seconds': 1800, 'pair_limit_seconds': 20,
                 'seal_sha256': evaluator.io.digest(folder / 'seal.json')}
        write(folder / 'outer-receipt.json', outer)
        return folder, job, seal, outer

    def test_exact_auc_keeps_failure_mass(self):
        self.assertEqual(evaluator.auc([0., 1., 5., 180.], 5), 45.)
        self.assertEqual(evaluator.auc([0., 0., 0., 0.], 5), 100.)
        self.assertEqual(evaluator.auc([5., 180.], 5), 0.)
        for errors in ([], [float('nan')], [-1.], [181.]):
            with self.subTest(errors=errors), self.assertRaises(ValueError):
                evaluator.auc(errors, 5)

    def test_selection_ties_and_failed_proposals(self):
        r = record(0)
        for value in [r['incumbent'], *r['attempts'].values()]:
            value['final_scores']['pixel_msac'] = 3.
        select(r)
        evaluator.audit_selection(r)
        self.assertTrue(all(v['selected_attempt'] == 'incumbent' for v in r['selectors'].values()))
        r['incumbent']['final_scores']['pixel_msac'] = 4.
        select(r)
        evaluator.audit_selection(r)
        self.assertEqual(r['selectors']['F_all']['selected_attempt'], 'continued_tc')
        r['attempts']['continued_tc'].update(status='failed', estimate=None, final_scores=None)
        select(r)
        evaluator.audit_selection(r)
        self.assertEqual(r['selectors']['G_local']['selected_attempt'], 'G_neutral')
        self.assertEqual(r['selectors']['F_all']['selected_attempt'], 'F_neutral')
        r['selectors']['G_local']['selected_attempt'] = 'G_incumbent'
        with self.assertRaisesRegex(ValueError, 'selection mismatch'):
            evaluator.audit_selection(r)

    def test_complete_80_seal_and_missing_last_pair(self):
        folder, job, seal, outer = self.sealed()
        with patch.object(evaluator.io, 'source_hashes', return_value=SOURCES):
            self.assertEqual(len(evaluator.verify_predictions(folder)[1]), 80)
            seal['rows'].pop()
            write(folder / 'seal.json', seal)
            outer['seal_sha256'] = evaluator.io.digest(folder / 'seal.json')
            write(folder / 'outer-receipt.json', outer)
            self.assert_blocks_labels(folder)

    def assert_blocks_labels(self, folder):
        original = evaluator.io.digest
        def watched(path):
            if Path(path).name == 'evaluation.json':
                self.fail('label file was accessed before evidence rejection')
            return original(path)
        args = SimpleNamespace(repo=self.root, predictions=folder, output=self.root / 'evaluation-output.json')
        with patch.object(evaluator.io, 'digest', side_effect=watched), \
             patch.object(evaluator.io, 'module', side_effect=AssertionError('metric import before barrier')):
            with self.assertRaises(ValueError):
                evaluator.evaluate(args)
        self.assertFalse(args.output.exists())

    def test_changed_pair_bytes_block_labels(self):
        folder, _, _, _ = self.sealed()
        with (folder / 'pair-79.json').open('a') as stream:
            stream.write(' ')
        with patch.object(evaluator.io, 'source_hashes', return_value=SOURCES):
            self.assert_blocks_labels(folder)

    def test_rehashed_bad_selection_blocks_labels(self):
        folder, _, seal, outer = self.sealed()
        path = folder / seal['rows'][-1]['path']
        r = json.loads(path.read_text())
        r['selectors']['G_lift']['selected_attempt'] = 'incumbent'
        write(path, r)
        seal['rows'][-1]['sha256'] = evaluator.io.digest(path)
        write(folder / 'seal.json', seal)
        outer['seal_sha256'] = evaluator.io.digest(folder / 'seal.json')
        write(folder / 'outer-receipt.json', outer)
        with patch.object(evaluator.io, 'source_hashes', return_value=SOURCES):
            self.assert_blocks_labels(folder)

    def test_legacy_metric_source_change_blocks_labels(self):
        folder, job, seal, outer = self.sealed()
        metric = self.root / 'manufactured-metric.py'
        metric.write_text('# changed artificial source\n')
        job['prior']['legacy_sources'] = {metric.name: '0' * 64}
        write(folder / 'job.json', job)
        seal['job_sha256'] = evaluator.io.digest(folder / 'job.json')
        write(folder / 'seal.json', seal)
        outer['seal_sha256'] = evaluator.io.digest(folder / 'seal.json')
        write(folder / 'outer-receipt.json', outer)
        with patch.object(evaluator.io, 'source_hashes', return_value=SOURCES):
            self.assert_blocks_labels(folder)

    def evaluation_fixture(self, primary_by_scene):
        records = [record(i, primary_by_scene[i // 16]) for i in range(80)]
        job = {'selected_pairs': [identity(i) for i in range(80)],
               'prior': {'original_seal': 'e' * 64, 'TC_seal': 'f' * 64, 'legacy_sources': {}}}
        binding = {'seal_sha256': 'a' * 64, 'outer_sha256': 'b' * 64}
        labelpath = self.root / 'results/0017-depth/evaluation.json'
        write(labelpath, {'rows': [{**identity(i), 'ground_truth': {
            'R_original': pose(0)['R'], 't_original': [1., 0., 0.]}} for i in range(80)]})
        def reference(repo, index, arm, seal, expected):
            if arm == 'depth_affine':
                return {**identity(index), **copy.deepcopy(records[index]['incumbent'])}
            angle = 1.25 if arm in ('e_norm', 'e_ts') else 1.5
            return {**identity(index), 'status': 'completed', 'estimate': pose(angle)}
        args = SimpleNamespace(repo=self.root, predictions=self.root / 'unused', output=self.root / 'answer.json')
        with patch.object(evaluator, 'verify_predictions', return_value=(job, records, binding)), \
             patch.object(evaluator, 'LABEL_SHA', evaluator.io.digest(labelpath)), \
             patch.object(evaluator, 'saved_reference', side_effect=reference), \
             patch.object(evaluator.io, 'module', return_value=SimpleNamespace(pose_errors=fake_metric)), \
             redirect_stdout(text_io.StringIO()):
            self.assertEqual(evaluator.evaluate(args), 0)
        return json.loads(args.output.read_text())

    def test_complete_80_metrics_and_best_control_fixed_tie(self):
        out = self.evaluation_fixture([.5] * 5)
        self.assertEqual(len(out['rows']), 80)
        self.assertTrue(all(v['count'] == 80 for v in out['pooled'].values()))
        self.assertTrue(all(v['G_lift']['count'] == 16 for v in out['scenes'].values()))
        self.assertAlmostEqual(out['pooled']['G_lift']['auc_percent']['5'], 90., places=10)
        self.assertAlmostEqual(out['decision']['auc5_gain_pp'], 15., places=10)
        self.assertEqual(out['decision']['best_control'], 'historical_e_norm')
        self.assertTrue(out['decision']['development_gate_passed'])
        self.assertFalse(out['decision']['main_goal_complete'])
        self.assertFalse(out['decision']['manuscript_admitted'])

    def test_one_bad_scene_cannot_be_hidden_by_pooled_gain(self):
        out = self.evaluation_fixture([1.35, .5, .5, .5, .5])
        clauses = out['decision']['clauses']
        self.assertTrue(clauses['auc5_gain_at_least2pp'])
        self.assertTrue(clauses['positive_in_at_least3_scenes'])
        self.assertFalse(clauses['no_scene_loses_more_than1pp'])
        self.assertFalse(out['decision']['development_gate_passed'])


if __name__ == '__main__':
    unittest.main()
