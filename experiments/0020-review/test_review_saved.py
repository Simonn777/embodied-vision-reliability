import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('review0020',Path(__file__).with_name('review_saved.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def fixture():
    n=8;ids=[0,1,2,3]*2;random=r.random_ids('artificial',[2]*4)
    def fits(method):
        labels=[None] if method in ('global_affine_ls','spatial_weight_only') else list(range(4))
        return [{'fold':k,'validation_weight_mass':1. if k is None else .25,
            'fit':{'status':'unavailable','reason':'depth_normalization_unavailable','training_count':8 if k is None else 6,
                'training_weight_mass':1. if k is None else .75,'svd_attempted':False,'singular_values':None},
            'validation':{'status':'fit_unavailable','reason':'depth_normalization_unavailable','count':8 if k is None else 2,
                'direction_denominator':16 if k is None else 4,'fit_failed_direction_count':16 if k is None else 4,
                'forward':None,'reverse':None}} for k in labels]
    candidates=[{'status':'completed','arm':a,'input_arrays':{'x1':{'shape':[8,2]}},
        'estimate':{'t':[1.,0.,0.]},'final_scores':{'pixel_msac':2.}} for a in r.ARMS]
    scores={'schema':'evr0020.scores.v1','key':'artificial','count':8,'arms':list(r.ARMS),
        'spatial_partition':{'status':'completed','counts':[2]*4,'ids':ids},
        'random_partition':{'status':'completed','counts':[2]*4,'salt':r.SALT,'ids':random},
        'normalization':{a:{'count':8,'status':'unavailable','reason':'zero_depth_spread','mean':1.,'rms':0.} for a in ('depth1','depth2')},
        'candidates':[{'arm':a,'geometry':{'count':8,'pixel_msac':2.,'pixel_inliers':8},'geometric_mean':.25,
            'unit_translation':[1.,0.,0.],'fit_support_count':8,'fits':{s:fits(s) for s in r.SELECTORS[1:]},
            'depth_scores':{s:1. for s in r.SELECTORS[1:]},'scores':{s:.25 if s=='pixel_msac' else 1.25 for s in r.SELECTORS}}
            for a in r.ARMS],
        'selectors':{s:{'selected_arm':r.ARMS[0],'selected_index':0,'score':.25 if s=='pixel_msac' else 1.25,
            'scores':[.25 if s=='pixel_msac' else 1.25]*4} for s in r.SELECTORS},
        'counters':{'candidate_count':4,'fit_calls':40,'svd_calls':0,'fit_completed_count':0,'fit_unavailable_count':40}}
    return scores,candidates


def outcome(error):
    return {'status':'completed','error':{'max_degrees':error},'penalized_max_degrees':error,'penalized_max_signfolded_degrees':error}


class Tests(unittest.TestCase):
    def test_all_failed_fullmass_and_exact_ties(self):
        s,c=fixture();choice,count,fail,bounds=r.check_scores(s,c,'artificial')
        self.assertEqual(set(choice.values()),{'e_norm'});self.assertEqual(count['fit_calls'],40)
        self.assertEqual(fail,{'depth_normalization_unavailable':40})
        self.assertTrue(all(b['forced_cost_lower_bound']==1. for b in bounds))
        s['selectors']['spatial_transfer'].update(selected_index=1,selected_arm='e_ts')
        with self.assertRaises(ValueError):r.check_scores(s,c,'artificial')

    def test_fullmass_random_partition_and_counter_tamper(self):
        for mutate in (
            lambda s:s['candidates'][0]['depth_scores'].update(spatial_transfer=.9),
            lambda s:s['candidates'][0]['fits']['spatial_transfer'][0].update(validation_weight_mass=.2),
            lambda s:s['candidates'][0]['fits']['spatial_transfer'][0]['fit'].update(training_count=5),
            lambda s:s['counters'].update(fit_calls=39),
            lambda s:s['random_partition']['ids'].__setitem__(0,(s['random_partition']['ids'][0]+1)%4)):
            s,c=fixture();mutate(s)
            with self.assertRaises(ValueError):r.check_scores(s,c,'artificial')

    def test_successful_fit_forced_direction_lower_bound(self):
        s,c=fixture();f=s['candidates'][0]['fits']['global_affine_ls'][0]
        f['fit'].update(status='completed',reason=None,svd_attempted=True,singular_values=[4.,3.,2.,1.],
            rank_threshold=2.220446049250313e-16*24*4,rank=4,coefficients=[1.,1.,1.,1.])
        direction={'count':8,'valid_count':4,'invalid_count':4,'source_invalid_count':4,
            'transformed_nonfinite_count':0,'transformed_nonpositive_or_nonfinite_z_count':0,
            'projection_nonfinite_count':0,'capped_count':4}
        f['validation'].update(status='completed',reason=None,fit_failed_direction_count=0,forward=direction,reverse=copy.deepcopy(direction))
        s['counters'].update(svd_calls=1,fit_completed_count=1,fit_unavailable_count=39)
        r.check_scores(s,c,'artificial')
        s['candidates'][0]['depth_scores']['global_affine_ls']=.49
        with self.assertRaisesRegex(ValueError,'undercharges'):r.check_scores(s,c,'artificial')

    def test_empty_partition_fullcost_and_only_four_fits(self):
        s,c=fixture();s['spatial_partition'].update(status='unavailable',reason='spatial_partition_unavailable',counts=[8,0,0,0],ids=[0]*8)
        s['random_partition'].update(status='unavailable',reason='spatial_partition_unavailable',counts=[8,0,0,0],ids=None)
        for row in s['candidates']:
            for method in r.SELECTORS[2:]:row['fits'][method]=[{'status':'unavailable','reason':'spatial_partition_unavailable'}]
        s['counters'].update(fit_calls=4,fit_unavailable_count=4)
        self.assertEqual(r.check_scores(s,c,'artificial')[1]['fit_calls'],4)

    def test_all80_eleven_methods_stronger_author_comparator_and_gate(self):
        ids=[];records=[];reference={'rows':[]};author={'rows':[]}
        for i in range(80):
            ident={'pair_index':i,'scene':r.u.SCENES[i//16],'key':str(i)};ids.append(ident)
            original={a:outcome(1. if j==i%4 else 10.) for j,a in enumerate(r.ARMS)}
            reference['rows'].append({**ident,'outcomes':original})
            author['rows'].append({**ident,'outcomes':{**original,**{a:outcome(3.) for a in r.AUTHOR}}})
            selectors={s:{'selected_arm':r.ARMS[i%4] if s=='spatial_transfer' else r.ARMS[0],
                'scores':[.2,.3,.4,.5]} for s in r.SELECTORS}
            records.append({**ident,'scores':{'selectors':selectors},'timings':{'charged_total_seconds':1.},'candidate_generation_seconds':4.})
        result=r.metrics(ids,records,reference,author)
        self.assertEqual(len(result['summaries']),11)
        self.assertTrue(result['development_gate']['passes'])
        self.assertEqual(result['development_gate']['best_auc5_comparator'],r.AUTHOR[0])
        self.assertEqual(len(result['development_gate']['positive_scenes_against_same_comparator']),5)
        self.assertEqual(result['historical_all_four_candidates_seconds'],320.)
        self.assertEqual(result['charged_scoring_seconds'],80.)
        for record in records:record['scores']['selectors']['spatial_transfer']['selected_arm']='e_norm'
        self.assertFalse(r.metrics(ids,records,reference,author)['development_gate']['passes'])
        with self.assertRaises(ValueError):r.metrics(ids,records[:-1],reference,author)


if __name__=='__main__':unittest.main()
