"""Tiny manufactured checks; no production metrics, media, GT or network."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location('independent_query_audit_test', Path(__file__).with_name('audit_queries_saved.py'))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def witness(actionable=True):
    rows = {}
    for name, y, label in (('positive', 0, 1), ('negative', 1, 0)):
        rows[name] = [dict(target_xy=[float(x), float(y)], rounded_current_xy=[x, y], label=label) for x in range(4)]
    if not actionable:
        rows['positive'] = rows['positive'][:3]
    return dict(target_id=1, current_shape=[2, 4], actionable=actionable, **rows)


def invented_rows():
    rows, arrays, labels = [], {}, {}
    for sequence, frame in a.IDENTITIES:
        identity = sequence, frame
        w = witness()
        base = np.full((2, 4), -2., dtype=np.float32)
        good = np.array([[2.]*4, [-2.]*4], dtype=np.float32)
        values = {'B': base, 'D1': good, 'D3': base.copy(), 'M0': base.copy(),
                  'M1': base.copy(), 'M2': base.copy(), 'P': base.copy(), 'R': base.copy()}
        b_loss = math.log1p(math.exp(-2.))+1.
        good_loss = math.log1p(math.exp(-2.))
        candidates = {name: dict(loss=good_loss if name == 'D1' else b_loss, predicted_iou=.5)
                      for name in values}
        for name, value in zip(a.BANK, (0., 1., 0.)):
            candidates[name]['unsigned_change'] = value
        row = dict(sequence=sequence, frame=frame, target_id=1, status='completed', actionable=True,
                   witnesses=w, candidates=candidates, costs=dict(seconds={'probe': .25}, counts={'calls': 3}))
        row['selection'] = a.choose(row, {n: x['loss'] for n, x in candidates.items()},
                                   {'B': 0., 'D1': 1., 'D3': 0.}, {n: .5 for n in values})
        rows.append(row)
        arrays[identity] = values
        labels[identity] = np.array([[1]*4, [0]*4], dtype=np.uint8)
    return rows, arrays, labels


class AuditArithmetic(unittest.TestCase):
    def test_jaccard_void_and_empty_conventions(self):
        gt = np.array([[1, 255], [0, 0]], dtype=np.uint8)
        self.assertEqual(a.foreground_j(np.array([[1, 1], [1, 0]]), gt, 1), .5)
        self.assertEqual(a.foreground_j(np.zeros((2, 2)), np.full((2, 2), 255, dtype=np.uint8), 1), 1.)
        self.assertEqual(a.foreground_j(np.ones((2, 2)), np.zeros((2, 2), dtype=np.uint8), 1), 0.)

    def test_interpolation_balanced_loss_and_unknown_precision(self):
        image = np.array([[0., 2.], [4., 6.]])
        self.assertEqual(a.bilinear_one(image, [.5, .5], [2, 2]), 3.)
        self.assertEqual(a.bilinear_one(image, [0., 0.], [4, 4]), 0.)
        self.assertAlmostEqual(a.witness_loss(np.zeros((2, 4)), witness()), math.log(2.))
        gt = np.array([[1, 255, 0, 1], [0, 1, 255, 0]], dtype=np.uint8)
        result = a.precision(witness(), gt, 1)
        for name in ('positive', 'negative'):
            self.assertEqual(result[name]['known'], 3)
            self.assertEqual(result[name]['void_unknown'], 1)
            self.assertEqual(result[name]['correct'], 2)
            self.assertAlmostEqual(result[name]['precision'], 2/3)

    def test_independent_selection_ties_margin_and_nonactionable(self):
        row = dict(sequence='dogs-scale', frame=4, target_id=1, actionable=True)
        losses = {name: 2. for name in ('B', 'D1', 'D3', 'M0', 'M1', 'M2', 'P', 'R')}
        changes = dict(B=0., D1=.4, D3=.4)
        quality = dict(B=.2, D1=.3, D3=.3)
        selection = a.choose(row, losses, changes, quality)
        self.assertEqual(selection['choices']['signed_deletion'], 'B')
        self.assertEqual(selection['choices']['max_predicted_iou'], 'D1')
        self.assertEqual(selection['choices']['max_unsigned_change'], 'D1')
        losses.update(D1=1., D3=1., M0=.3, M1=.2, M2=.2, P=.1, R=3.)
        selection = a.choose(row, losses, changes, quality)
        self.assertEqual(selection['choices']['signed_deletion'], 'D1')
        self.assertEqual(selection['choices']['native_bank'], 'M1')
        self.assertEqual(selection['choices']['prompt_signed'], 'P')
        self.assertEqual(selection['choices']['reset_signed'], 'B')
        row['actionable'] = False
        self.assertEqual(set(a.choose(row, {}, {}, {})['choices'].values()), {'B'})

    def test_all48_arithmetic_and_reject_choices_before_gt_index(self):
        rows, arrays, labels = invented_rows()
        result = a.arithmetic(rows, arrays, labels, a.Checks())
        self.assertEqual(result['summary']['pooled']['signed_deletion'], 1.)
        self.assertEqual(result['summary']['pooled']['baseline'], 0.)
        self.assertEqual(result['summary']['signed_actions'], 48)
        self.assertFalse(result['gate']['pooled_selection'])  # always_D1 has equal performance
        self.assertEqual(result['diagnostics']['pooled']['costs']['seconds']['probe']['sum'], 12.)
        class UnreadableLabels(dict):
            def __getitem__(self, key):
                raise AssertionError('GT accessed before sealed choice rejection')
        invalid = copy.deepcopy(rows)
        invalid[-1]['selection']['choices']['native_bank'] = 'M1'
        with self.assertRaisesRegex(ValueError, 'audit mismatches'):
            a.arithmetic(invalid, arrays, UnreadableLabels(labels), a.Checks())
        with self.assertRaisesRegex(ValueError, '48-query'):
            a.arithmetic(rows[:-1], arrays, UnreadableLabels(labels), a.Checks())

    def test_near_tie_replays_verified_saved_scalar_without_replacing_choice(self):
        rows, arrays, labels = invented_rows()
        row = rows[0]
        identity = row['sequence'], row['frame']
        arrays[identity]['D3'] = arrays[identity]['D1'].copy()
        row['candidates']['D3']['loss'] = row['candidates']['D1']['loss'] - 1e-12
        row['candidates']['D3']['unsigned_change'] = 1.
        row['selection']['choices']['signed_deletion'] = 'D3'
        row['selection']['bank_losses']['D3'] = row['candidates']['D3']['loss']
        # An independent reduction sees an exact tie; the verified saved
        # scalars select D3. The audit must report, not replace, this choice.
        _, choices, _, detail = a.numerical_preflight(rows, arrays, a.Checks())
        self.assertEqual(choices[identity]['choices']['signed_deletion'], 'D3')
        self.assertEqual(detail[0]['recomputed_choices']['signed_deletion'], 'D1')
        self.assertEqual(detail[0]['choice_differences']['signed_deletion'],
                         {'recomputed': 'D1', 'verified_saved': 'D3'})
        # A tiny saved loss gain can also be nonzero while the independent
        # computation is exactly zero. Descriptive signs replay saved scalars.
        row = rows[1]
        identity = row['sequence'], row['frame']
        arrays[identity]['D1'] = arrays[identity]['B'].copy()
        row['candidates']['D1'].update(loss=row['candidates']['B']['loss']-1e-12, unsigned_change=0.)
        row['selection']['choices'].update(signed_deletion='B', max_unsigned_change='B', random_on_signed='B')
        row['selection']['signed_action'] = False
        row['selection']['bank_losses']['D1'] = row['candidates']['D1']['loss']
        _, _, _, detail = a.numerical_preflight(rows, arrays, a.Checks())
        self.assertEqual(detail[1]['descriptive_sign_differences']['D1'],
                         {'recomputed': 0, 'verified_saved': 1})
        report = a.arithmetic(rows, arrays, labels, a.Checks())
        self.assertEqual(report['rows'][1]['loss_j_sign']['D1']['loss_gain_sign'], 1)

    def test_equal_video_means_oracle_exclusion_and_all_gate_clauses(self):
        rows = []
        for name, frame in a.IDENTITIES:
            scores = {key: .3 for key in a.METHODS}
            scores.update(signed_deletion=.5, baseline=.2, ungated_D1=.99, ungated_D3=.99,
                          oracle_admissible=.8, oracle_unrestricted=1.)
            rows.append(dict(sequence=name, frame=frame, status='completed', actionable=True, j=scores,
                             selected_branches={key: 'D1' for key in a.METHODS}))
        cov = a.coverage(rows)
        report = a.aggregate(rows, cov)
        self.assertTrue(report['gate']['pass'])
        self.assertAlmostEqual(report['signed_gain'], .2)
        self.assertAlmostEqual(report['candidate_gain'], .6)
        self.assertEqual(report['positive_videos'], 6)
        self.assertNotIn(report['best_pooled_control'], ('oracle_admissible', 'ungated_D1'))
        for row in rows[:8]:
            row['j']['signed_deletion'] = .1
        report = a.aggregate(rows, cov)
        self.assertAlmostEqual(report['pooled']['signed_deletion'], (5*.5+.1)/6)
        self.assertFalse(report['gate']['no_video_below_minus_002'])
        self.assertFalse(report['gate']['pass'])
        for row in rows:
            row['j']['oracle_admissible'] = .21
        self.assertFalse(a.aggregate(rows, cov)['gate']['candidate_information'])
        for row in rows[:24]:
            row['j']['signed_deletion'] = .3
        self.assertFalse(a.aggregate(rows, cov)['gate']['four_positive_videos'])
        bad_coverage = dict(cov, **{'pass': False})
        self.assertFalse(a.aggregate(rows, bad_coverage)['gate']['coverage'])

    def test_failed_admission_persists_without_numerical_or_gt_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'repo'
            root.mkdir()
            output = Path(tmp)/'audit'
            args = SimpleNamespace(repo_root=str(root), output_dir=str(output), expected_commit='a'*40)
            with mock.patch.object(a, 'admission', side_effect=ValueError('unclean evaluation outer')):
                with mock.patch.object(a, 'load_arrays', side_effect=AssertionError('arrays touched')):
                    with self.assertRaisesRegex(ValueError, 'unclean evaluation outer'):
                        a.run(args)
            record = json.loads((output/'receipt.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['query_masks_decoded'], 0)
            self.assertFalse(record['eligible'])
            self.assertNotIn('gt_access_attempted', record)
            self.assertEqual(record['model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
