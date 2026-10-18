"""Artificial arrays/summaries only. No dataset, model, cache or network use."""
import copy
import hashlib
import json
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np
# Supports ordinary repository unittest discovery after copying into tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import antithetic_training_metrics as m


def summary(mse=1., fd=1.):
    """Internally consistent analytic 8x8 summary; no claim of a saved image."""
    base = m.score_pair(np.zeros((8,8),dtype=np.float32),np.ones((8,8),dtype=np.float32))
    base.update(sse=64*mse,mse=mse,psnr_db=None if mse==0 else -10*math.log10(mse))
    for block in base['blocks']:block['sse']=mse
    for part in base['first_difference'].values():
        part.update(sse=part['count']*fd,mse=fd)
    base['prediction_sha256_f32le']=hashlib.sha256(repr((mse,fd)).encode()).hexdigest()
    return base


def grid():
    rows=[]
    for endpoint in m.ENDPOINTS:
        for group in m.GROUPS:
            for seed in m.SEEDS:
                for method in m.METHODS:
                    candidate=method==m.CANDIDATE
                    rows.append(dict(group=group,seed=seed,method=method,endpoint=endpoint,
                        status='completed',inference_mc_samples=50,
                        updates=500 if endpoint=='updates500' else 2000,
                        training_seconds=1. if endpoint=='updates500' else 5.01,
                        pipeline_seconds=7.,score=summary(.5 if candidate else 1.,.5 if candidate else 1.)))
    return rows


def select(rows, **values):
    return [r for r in rows if all(r[k]==v for k,v in values.items())]


class AggregateTests(unittest.TestCase):
    def score(self, rows):return m.aggregate_cases(rows,expected_shape=(8,8))

    def test_complete_success_and_no_population_or_paper_claim(self):
        out=self.score(grid())
        self.assertEqual(out['status'],'continue_to_novelty_review_only')
        self.assertTrue(out['gates']['all_passed'])
        self.assertFalse(out['paper_success'])
        self.assertEqual((out['case_count'],out['trajectory_count']),(64,32))
        self.assertAlmostEqual(out['endpoints']['updates500']['mean_gain_db'],10*math.log10(2))
        json.dumps(out,allow_nan=False)

    def test_real_artificial_arrays_unclipped_pooled_metric(self):
        clean=np.zeros((8,8),np.float32)
        pred=np.tile(np.array([-2.,2.],np.float32),(8,4))
        score=m.score_pair(clean,pred)
        self.assertEqual(score['mse'],4.)
        self.assertAlmostEqual(score['psnr_db'],-10*math.log10(4.))
        self.assertEqual(score['first_difference']['horizontal']['sse'],56*16)
        self.assertEqual(score['first_difference']['combined']['mse'],8.)
        self.assertEqual(sum(b['sse'] for b in score['blocks']),256.)

    def test_canonical_output_independent_of_input_order_and_no_mutation(self):
        rows=grid();before=copy.deepcopy(rows);first=self.score(rows)
        self.assertEqual(rows,before)
        random.Random(17).shuffle(rows)
        self.assertEqual(first,self.score(rows))
        rows[0]['score']['mse']=99
        self.assertNotEqual(first['cases'][0]['score']['mse'],99)

    def test_best_control_and_fixed_tie_order(self):
        rows=grid()
        out=self.score(rows)
        self.assertTrue(all(p['oracle_best_control']=='independent' for e in out['endpoints'].values() for p in e['pairs']))
        for row in rows:
            if row['method'] in ('random_antithetic','same_patch'):row['score']=summary(.25,.25)
        out=self.score(rows)
        self.assertEqual(out['status'],'stop_this_candidate')
        self.assertTrue(all(p['oracle_best_control']=='random_antithetic' for e in out['endpoints'].values() for p in e['pairs']))

    def test_two_endpoints_cannot_be_averaged_to_rescue_failure(self):
        rows=grid()
        for row in select(rows,method=m.CANDIDATE,endpoint='wall5s'):row['score']=summary(2.,2.)
        out=self.score(rows)
        self.assertTrue(out['endpoints']['updates500']['gates']['mean_gain_ge_0_5_db'])
        self.assertFalse(out['endpoints']['wall5s']['gates']['mean_gain_ge_0_5_db'])
        self.assertEqual(out['status'],'stop_this_candidate')

    def test_finite_gain_threshold_and_group_harm_guard(self):
        rows=grid()
        for row in select(rows,method=m.CANDIDATE):
            row['score']=summary(10**(-.5/10),.5)
            row['score']['psnr_db']=.5  # exact analytic boundary, within validator precision
        self.assertTrue(self.score(rows)['gates']['all_passed'])
        for row in select(rows,method=m.CANDIDATE):row['score']=summary(10**(-.499/10),.5)
        self.assertFalse(self.score(rows)['gates']['both_endpoint_quality_gates_pass'])
        rows=grid()
        for row in select(rows,method=m.CANDIDATE,group='d4'):row['score']=summary(10**(.201/10),.5)
        out=self.score(rows)['endpoints']['wall5s']
        self.assertTrue(out['gates']['mean_gain_ge_0_5_db'])
        self.assertTrue(out['gates']['at_least3_group_means_gt0'])
        self.assertFalse(out['gates']['each_group_mean_ge_minus0_2_db'])

    def test_three_positive_groups_required_even_with_large_mean(self):
        rows=grid()
        for row in select(rows,method=m.CANDIDATE):
            row['score']=summary(.01 if row['group'] in ('d1','d2') else 1.,.5)
        out=self.score(rows)['endpoints']['updates500']
        self.assertTrue(out['gates']['mean_gain_ge_0_5_db'])
        self.assertFalse(out['gates']['at_least3_group_means_gt0'])

    def test_pooled_fd_means_not_mean_of_ratios(self):
        rows=grid()
        for row in rows:
            cand=row['method']==m.CANDIDATE
            row['score']=summary(.5 if cand else 1., (2. if cand else 1.) if row['seed']==0 else (50. if cand else 100.))
        out=self.score(rows)
        self.assertTrue(out['gates']['all_passed'])  # mean ratios1.25 would wrongly fail.
        for row in rows:
            cand=row['method']==m.CANDIDATE
            row['score']=summary(.5 if cand else 1., (110. if cand else 100.) if row['seed']==0 else (.1 if cand else 1.))
        out=self.score(rows)
        self.assertFalse(out['endpoints']['wall5s']['gates']['mean_pooled_first_difference_le_1_05'])  # mean ratios.6 would wrongly pass.

    def test_fd_boundary_zero_and_selected_control(self):
        rows=grid()
        for row in select(rows,method=m.CANDIDATE):row['score']=summary(.5,1.05)
        self.assertTrue(self.score(rows)['gates']['all_passed'])
        for row in rows:row['score']=summary(.5 if row['method']==m.CANDIDATE else 1.,0.)
        self.assertTrue(self.score(rows)['gates']['all_passed'])
        for row in select(rows,method=m.CANDIDATE):row['score']=summary(.5,.001)
        self.assertFalse(self.score(rows)['gates']['all_passed'])
        rows=grid()
        for row in select(rows,method='random_antithetic'):row['score']=summary(.8,.01)
        self.assertFalse(self.score(rows)['endpoints']['wall5s']['gates']['mean_pooled_first_difference_le_1_05'])

    def test_cost_is_ratio_of_totals_and_exact_boundary(self):
        rows=grid()
        for row in select(rows,endpoint='wall5s'):
            if row['method']=='independent':row['pipeline_seconds']=6. if row['seed']==0 else 60.
            elif row['method']==m.CANDIDATE:row['pipeline_seconds']=12. if row['seed']==0 else 54.
        out=self.score(rows)
        self.assertEqual(out['wall5s_candidate_vs_independent_pipeline_ratio'],1.)
        self.assertTrue(out['gates']['all_passed'])
        for row in select(rows,endpoint='wall5s'):
            if row['method']=='independent':row['pipeline_seconds']=10.
            elif row['method']==m.CANDIDATE:row['pipeline_seconds']=11.
        self.assertTrue(self.score(rows)['gates']['wall5s_pipeline_ratio_le_1_10'])
        for row in select(rows,endpoint='wall5s',method=m.CANDIDATE):row['pipeline_seconds']=11.001
        self.assertFalse(self.score(rows)['gates']['all_passed'])

    def test_cost_gate_only_wall_endpoint(self):
        rows=grid()
        for row in select(rows,endpoint='updates500',method=m.CANDIDATE):row['pipeline_seconds']=100.
        self.assertTrue(self.score(rows)['gates']['all_passed'])

    def test_missing_duplicate_and_unexpected_grid_rejected(self):
        for rows in (grid()[:-1],grid()+[copy.deepcopy(grid()[0])]):
            with self.assertRaises(ValueError):self.score(rows)
        for field,value in [('group','d5'),('seed',False),('seed',2),('method','oracle'),('endpoint','best_epoch')]:
            rows=grid();rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.score(rows)

    def test_status_mc_count_and_updates_fail_closed(self):
        for field,value in [('status','failed'),('inference_mc_samples',49),('inference_mc_samples',True),('updates',499),('updates',500.),('updates',True)]:
            rows=grid();rows[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.score(rows)

    def test_time_contract_inclusive_bounds_and_nonfinite(self):
        for boundary in (5.,5.05):
            rows=grid()
            for row in select(rows,endpoint='wall5s'):row['training_seconds']=boundary
            self.assertTrue(self.score(rows)['gates']['all_passed'])
        for value in (4.999,5.05000000001,0.,float('nan'),float('inf'),True):
            rows=grid();select(rows,endpoint='wall5s')[0]['training_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.score(rows)
        for field,value in [('pipeline_seconds',0),('pipeline_seconds',float('nan')),('pipeline_seconds',float('inf')),('pipeline_seconds',True),('pipeline_seconds',.1)]:
            rows=grid();rows[0][field]=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.score(rows)

    def test_clean_identity_consistent_per_group_across_all_16_rows(self):
        rows=grid();rows[-1]['score']['clean_sha256_f64le']='f'*64
        with self.assertRaisesRegex(ValueError,'clean substrate'):self.score(rows)
        rows=grid()
        for row in rows:row['score']['clean_sha256_f64le']=hashlib.sha256(row['group'].encode()).hexdigest()
        self.assertTrue(self.score(rows)['gates']['all_passed'])

    def test_score_validation_and_malformed_rows(self):
        mutations=[lambda r:r['score'].update(sse=2.),lambda r:r['score'].update(mse=float('nan')),
                   lambda r:r['score'].update(clipped=True),lambda r:r['score'].update(data_range=True),
                   lambda r:r['score'].update(psnr_db=float('inf')),lambda r:r.pop('score'),
                   lambda r:r.update(score=None),lambda r:r['score'].pop('first_difference'),
                   lambda r:r['score'].update(first_difference=[]),lambda r:r['score'].update(shape=[8.,8]),
                   lambda r:r['score']['blocks'][0].update(row=False),lambda r:r['score']['blocks'][0].update(sse=True)]
        for change in mutations:
            rows=grid();change(rows[0])
            with self.subTest(change=change),self.assertRaises(ValueError):self.score(rows)
        with self.assertRaises(ValueError):m.aggregate_cases(grid())  # not122x122
        for shape in ((7,8),(8,8,8),(True,8),None):
            with self.subTest(shape=shape),self.assertRaises(ValueError):m.aggregate_cases(grid(),shape)

    def test_zero_error_psnr_never_manufactures_a_finite_gain(self):
        for perfect in (m.CANDIDATE,'independent','all'):
            rows=grid()
            for row in rows:
                if perfect=='all' or row['method']==perfect:row['score']=summary(0.,0.)
            out=self.score(rows)
            self.assertEqual(out['status'],'not_evaluable')
            self.assertFalse(out['gates']['all_passed'])
            self.assertFalse(out['paper_success'])
            json.dumps(out,allow_nan=False)

    def test_finite_individual_costs_with_overflowing_total_rejected(self):
        rows=grid()
        for row in rows:row['pipeline_seconds']=1e308
        with self.assertRaisesRegex(ValueError,'accumulation'):self.score(rows)


if __name__=='__main__':unittest.main()
