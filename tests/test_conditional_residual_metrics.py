import copy
import json
import math
import unittest

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from conditional_residual_metrics import aggregate_cases, score_pair


GROUPS, SEEDS = ('g0', 'g1', 'g2', 'g3'), (11, 29)
BASELINES = ('blindspot', 'safe_blindspot', 'r2r_alpha05')
CLEAN = np.full((8, 8), 0.5, dtype=np.float64)


def record(group, condition, seed, method, pred, clean=CLEAN):
    return dict(group=group, condition=condition, seed=seed, method=method,
                score=score_pair(clean, np.asarray(pred, dtype=np.float32)))


def cases():
    result = []
    errors = dict(residual=.11, blindspot=.15, safe_blindspot=.175, r2r_alpha05=.125)
    for g in GROUPS:
        for c in ('white', 'colored'):
            for s in SEEDS:
                for m in ('residual',) + BASELINES:
                    e = .0625 if c == 'white' and m in ('residual', 'blindspot', 'safe_blindspot') else errors[m]
                    result.append(record(g, c, s, m, CLEAN + e))
    return result


def aggregate(rows):
    return aggregate_cases(rows, groups=GROUPS, seeds=SEEDS, candidate='residual',
                           baselines=BASELINES, white_reference='blindspot', expected_shape=(8, 8))


def replace(rows, *, group=None, condition='colored', method='residual', pred):
    for row in rows:
        if row['condition'] == condition and row['method'] == method and (group is None or row['group'] == group):
            row['score'] = score_pair(CLEAN, np.asarray(pred, dtype=np.float32))


class PairTests(unittest.TestCase):
    def test_analytic_uniform_bias_no_clipping(self):
        score = score_pair(np.zeros((8, 8)), np.full((8, 8), -.5, dtype=np.float32))
        self.assertEqual(score['sse'], 16)
        self.assertEqual(score['mse'], .25)
        self.assertAlmostEqual(score['psnr_db'], 10 * math.log10(4))
        self.assertEqual(score['first_difference']['combined']['mse'], 0)
        self.assertEqual(sum(b['sse'] for b in score['blocks']), 16)

    def test_exact_horizontal_ramp_and_directional_denominators(self):
        pred = np.broadcast_to(np.arange(8, dtype=np.float32) / 8, (8, 8)).copy()
        score = score_pair(np.zeros((8, 8)), pred)
        self.assertEqual(score['sse'], 17.5)
        self.assertEqual(score['first_difference']['horizontal'], {'sse': .875, 'count': 56, 'mse': 1 / 64})
        self.assertEqual(score['first_difference']['vertical'], {'sse': 0., 'count': 56, 'mse': 0.})
        self.assertEqual(score['first_difference']['combined']['mse'], 1 / 128)
        transpose = score_pair(np.zeros((8, 8)), pred.T)
        self.assertEqual(transpose['first_difference']['vertical'], score['first_difference']['horizontal'])

    def test_122_rectangular_blocks_cover_once(self):
        pred = np.zeros((122, 122), dtype=np.float32)
        pred[0, 0], pred[15, 16], pred[16, 15], pred[121, 121] = 1, 2, 3, 4
        score = score_pair(np.zeros((122, 122)), pred)
        self.assertEqual(score['sse'], 30)
        self.assertEqual(score['count'], 14884)
        self.assertEqual(len(score['blocks']), 64)
        self.assertEqual(sum(b['count'] for b in score['blocks']), 14884)
        coverage = np.zeros((122, 122), dtype=np.int8)
        for b in score['blocks']:
            coverage[b['row_start']:b['row_stop'], b['col_start']:b['col_stop']] += 1
        np.testing.assert_array_equal(coverage, np.ones_like(coverage))
        self.assertEqual([score['blocks'][i]['sse'] for i in (0, 1, 8, 63)], [1, 4, 9, 16])
        self.assertEqual(score['blocks'][0]['count'], 256)
        self.assertEqual(score['blocks'][-1]['count'], 225)

    def test_zero_error_is_null_psnr_and_json_finite(self):
        score = score_pair(CLEAN, CLEAN.astype(np.float32))
        self.assertEqual(score['mse'], 0)
        self.assertIsNone(score['psnr_db'])
        json.dumps(score, allow_nan=False)

    def test_finite_inputs_cannot_produce_json_infinity(self):
        clean = np.zeros((8, 8))
        clean[0, 0] = 1e154
        with self.assertRaisesRegex(ValueError, "nonfinite combined"):
            score_pair(clean, np.zeros((8, 8), dtype=np.float32))

    def test_invalid_shape_dtype_and_nonfinite_rejected(self):
        for clean, pred in [(CLEAN, np.ones((7, 8), dtype=np.float32)),
                            (CLEAN, np.ones((8, 8), dtype=np.float64)),
                            (np.ones((8, 8), dtype=np.int32), CLEAN.astype(np.float32)),
                            (CLEAN, np.full((8, 8), np.nan, dtype=np.float32)),
                            (np.full((8, 8), np.inf), CLEAN.astype(np.float32))]:
            with self.subTest(shape=pred.shape, dtype=str(pred.dtype)):
                with self.assertRaises(ValueError):
                    score_pair(clean, pred)


class AggregateTests(unittest.TestCase):
    def test_complete_grid_pass_keeps_all_cases_and_oracle_choice(self):
        result = aggregate(cases())
        self.assertEqual(result['status'], 'continue_mechanism_exploration')
        self.assertEqual(len(result['cases']), 64)
        self.assertTrue(all(p['oracle_best_baseline'] == 'r2r_alpha05' for p in result['colored_pairs']))
        self.assertTrue(all(result['gates'].values()))
        self.assertFalse(result['paper_success'])
        self.assertTrue(result['development_only'])
        json.dumps(result, allow_nan=False)

    def test_prospective_defaults_and_safe_white_diagnostic(self):
        result = aggregate_cases(cases(), groups=GROUPS, seeds=SEEDS, expected_shape=(8, 8))
        self.assertEqual(result['methods'], ['residual', 'blindspot', 'safe_blindspot', 'r2r_alpha05'])
        self.assertTrue(all(p['safe_blindspot_bitwise_equal_to_reference'] for p in result['white_pairs']))
        self.assertTrue(all(p['safe_blindspot_gain_db'] == 0 for p in result['white_pairs']))

    def test_oracle_choice_is_per_pair_with_declared_tie_order(self):
        rows = cases()
        # At g0, both tie at exactly the same error; declared blindspot wins.
        replace(rows, group='g0', method='blindspot', pred=CLEAN + .125)
        # At g1, identity is genuinely best. Other groups still select r2r.
        replace(rows, group='g1', method='safe_blindspot', pred=CLEAN + .1)
        result = aggregate(rows)
        choices = {g: {p['oracle_best_baseline'] for p in result['colored_pairs'] if p['group'] == g} for g in GROUPS}
        self.assertEqual(choices, {'g0': {'blindspot'}, 'g1': {'safe_blindspot'}, 'g2': {'r2r_alpha05'}, 'g3': {'r2r_alpha05'}})

    def test_missing_duplicate_and_unexpected_cases_rejected(self):
        valid = cases()
        unexpected = copy.deepcopy(valid)
        unexpected[0]['condition'] = 'other'
        for rows in (valid[:-1], valid + [valid[0]], unexpected):
            with self.assertRaises(ValueError):
                aggregate(rows)

    def test_clean_hash_and_block_tampering_rejected(self):
        rows = cases()
        rows[0]['score'] = score_pair(CLEAN + .125, CLEAN.astype(np.float32))
        with self.assertRaisesRegex(ValueError, 'clean image'):
            aggregate(rows)
        rows = cases()
        rows[0]['score']['blocks'][0]['row_stop'] += 1
        with self.assertRaisesRegex(ValueError, 'block'):
            aggregate(rows)

    def test_white_same_psnr_different_predictions_is_failure(self):
        rows = cases()
        replace(rows, condition='white', pred=CLEAN - .0625)
        result = aggregate(rows)
        self.assertTrue(result['gates']['white_psnr_harm_zero'])
        self.assertFalse(result['gates']['white_predictions_bitwise_equal'])
        self.assertEqual(result['status'], 'stop_this_candidate')

    def test_safe_white_same_psnr_different_prediction_fails_gate(self):
        rows = cases()
        replace(rows, condition='white', method='safe_blindspot', pred=CLEAN - .0625)
        result = aggregate(rows)
        self.assertTrue(all(p['safe_blindspot_gain_db'] == 0 for p in result['white_pairs']))
        self.assertTrue(result['gates']['white_predictions_bitwise_equal'])
        self.assertFalse(result['gates']['white_safe_predictions_bitwise_equal'])
        self.assertEqual(result['status'], 'stop_this_candidate')

    def test_group_harm_floor_blocks_good_mean(self):
        rows = cases()
        replace(rows, group='g3', pred=CLEAN + .13)
        result = aggregate(rows)
        self.assertGreater(result['mean_colored_gain_db'], .3)
        self.assertTrue(result['gates']['colored_at_least3_group_means_gt0'])
        self.assertFalse(result['gates']['colored_each_group_mean_ge_minus0_2_db'])
        self.assertEqual(result['status'], 'stop_this_candidate')

    def test_positive_gains_below_point3_do_not_pass(self):
        rows = cases()
        replace(rows, pred=CLEAN + .123)
        result = aggregate(rows)
        self.assertGreater(result['mean_colored_gain_db'], 0)
        self.assertLess(result['mean_colored_gain_db'], .3)
        self.assertFalse(result['gates']['colored_pair_mean_ge_0_3_db'])
        self.assertEqual(result['status'], 'stop_this_candidate')

    def test_two_positive_groups_do_not_pass_despite_positive_mean(self):
        rows = cases()
        for group in ('g0', 'g1'):
            replace(rows, group=group, pred=CLEAN + .05)
        for group in ('g2', 'g3'):
            replace(rows, group=group, pred=CLEAN + .125)
        result = aggregate(rows)
        self.assertGreater(result['mean_colored_gain_db'], .3)
        self.assertFalse(result['gates']['colored_at_least3_group_means_gt0'])

    def test_first_difference_guard_uses_same_psnr_comparator(self):
        rows = cases()
        checker = ((np.indices((8, 8)).sum(axis=0) % 2) * 2 - 1) * .001
        replace(rows, pred=CLEAN + .1 + checker)
        result = aggregate(rows)
        self.assertGreater(result['mean_colored_gain_db'], .3)
        self.assertEqual(result['mean_corresponding_baseline_first_difference_mse'], 0)
        self.assertFalse(result['gates']['colored_first_difference_le_1_05'])
        self.assertEqual(result['status'], 'stop_this_candidate')

    def test_one_sided_zero_error_does_not_emit_infinity_or_claim_pass(self):
        rows = cases()
        replace(rows, group='g0', pred=CLEAN)
        result = aggregate(rows)
        self.assertEqual(result['status'], 'not_evaluable')
        self.assertIsNone(result['mean_colored_gain_db'])
        self.assertIsNone(result['group_mean_colored_gain_db']['g0'])
        self.assertIsNotNone(result['group_mean_colored_gain_db']['g1'])
        json.dumps(result, allow_nan=False)

    def test_both_perfect_white_predictions_are_zero_harm(self):
        rows = cases()
        for method in ('residual', 'blindspot', 'safe_blindspot'):
            replace(rows, condition='white', method=method, pred=CLEAN)
        result = aggregate(rows)
        self.assertTrue(result['gates']['white_psnr_harm_zero'])
        self.assertTrue(all(p['gain_db'] == 0 for p in result['white_pairs']))
        self.assertEqual(result['status'], 'continue_mechanism_exploration')


if __name__ == '__main__':
    unittest.main()
