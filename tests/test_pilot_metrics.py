import math
import numpy as np
import pytest
from evr.pilot_metrics import (CAP, calibrate, matched, operational, prediction_scores,
    rows_from_arrays, original_tie, expand_draw, paired_bootstrap, evaluate)


def fixture(group='a',index=0):
    raw=np.zeros((6,1,4),np.float64);raw[:,:,0:2]=[128,64]
    raw[2,0,0]=132  # exact threshold: NOT a failure
    raw[3,0,0]=133  # strict >4 failure
    raw[4,0,2]=np.inf  # coordinates valid, logits invalid
    raw[5,0,0]=np.nan
    truth=np.broadcast_to([.5,.25],(1,6,2)).copy();occ=np.zeros((1,6),bool)
    return rows_from_arrays(index,str(index),group,raw,[0],truth,occ)


def test_coordinate_endpoint_and_distinct_validity():
    rows=fixture()
    assert rows[2]['error_px']==4 and rows[2]['failure'] is False
    assert rows[3]['error_px']==5 and rows[3]['failure'] is True
    assert rows[4]['coordinate_valid'] and not rows[4]['logit_valid'] and not rows[4]['available']
    assert rows[5]['error_px'] is None and not rows[5]['coordinate_valid']
    assert not rows[0]['opportunity']


def test_stable_native_and_nonfinite_logits_never_sigmoid_valid():
    score=prediction_scores([0,1,2,3],np.zeros((4,2)),[0,1e308,np.inf,np.nan],[0,1e308,0,0])
    assert score['scores']['native'][0]==pytest.approx(2*math.log(2))
    assert score['scores']['native'][1]==CAP
    assert np.isnan(score['scores']['native'][2:]).all()
    assert score['scores']['visibility'][0]==pytest.approx(math.log(2))
    assert score['scores']['local'][0]==pytest.approx(math.log(2))


def test_history_uses_invalid_logit_coordinates_and_real_gaps():
    s=prediction_scores([0,1,2,5],[[0,0],[8,0],[12,0],[24,0]],[-1,np.nan,-1,-1],[-1,-1,-1,-1])
    assert not s['available'][1]
    assert s['residual_px'][2]==4  # invalid-logit row DID enter history
    assert s['residual_px'][3]==0  # dt=3, no fabricated intermediate frames
    assert s['scores']['augmented'][2]-s['scores']['native'][2]==pytest.approx(.5)
    changed=prediction_scores([0,1,2],[[0,0],[8,0],[12,0]],[-1,np.nan,-1],[-1,-1,-1])
    np.testing.assert_array_equal(s['scores']['native'][:3],changed['scores']['native'])


def test_no_clipping_of_finite_outside_coordinates():
    s=prediction_scores([0,1,2],[[0,0],[300,0],[600,0]],[-1]*3,[-1]*3)
    assert s['available'].all() and s['residual_px'][2]==0


def test_unreachable_calibration_uses_max_observed_and_matched_null():
    rows=fixture();c=calibrate(rows,'native')
    assert c['unreachable'] and c['accepted_calibration']==3
    assert c['calibration_risk']==pytest.approx(1/3)
    order=sorted((r for r in rows if r['opportunity'] and r['available']),key=lambda r:(r['scores']['native'],r['tie']))
    assert c['threshold']==[order[-1]['scores']['native'],order[-1]['tie']]
    m=matched(rows,'native');assert m['target_accepts']==4 and m['risk'] is None and m['coverage'] is None
    assert operational(rows,'native',{'threshold':None})['risk'] is None
    assert calibrate([],'native')['threshold'] is None


def test_ties_do_not_depend_on_method_or_outcome():
    assert original_tie(1,'hash',0,4)==original_tie(1,'hash',0,4)
    assert original_tie(1,'hash',0,4)!=original_tie(2,'hash',0,4)
    rows=fixture();before=[r['tie'] for r in rows]
    for r in rows:r['failure']=not bool(r['failure'])
    assert [r['tie'] for r in rows]==before


def test_bootstrap_retains_multiplicity_and_recomputes_target():
    a=fixture('a',0)[1:3];b=fixture('b',1)[1:4]
    draw=expand_draw(a+b,['a','a','b'])
    assert len(draw)==7 and matched(draw,'native')['target_accepts']==6
    assert draw[0] is draw[2]  # repeated source row is deliberately counted twice
    thresholds={m:calibrate(a+b,m) for m in ['native','augmented']}
    boot=paired_bootstrap(a+b,['a','b','empty'],thresholds,repeats=100,seed=1)
    assert boot['allocated_groups']==['a','b','empty']
    assert boot['intervals']['matched_risk_difference']['defined_replicates']<100


def test_end_to_end_fixture_and_zero_opportunity_group():
    rows=fixture();result=evaluate(rows,rows,['a','empty'],repeats=2000)
    assert result['primary_opportunities']==5
    assert result['coordinate_missing_rate']==pytest.approx(.2)
    assert result['joint_score_unavailable_rate']==pytest.approx(.4)
    assert result['estimator_failure_including_missing']==pytest.approx(.4)
    assert result['accept_all_coordinates']['risk']==pytest.approx(.25)
    # occ=0 predicts not visible at strict .5 threshold; all GT visible => zero.
    assert result['visibility_accuracy']==0
    assert result['bootstrap']['intervals']['matched_risk_difference']['ci95'] is None
    assert evaluate([],[],['empty'],repeats=10)['matched']['native']['risk'] is None


def test_native_augmented_share_availability_under_invalid_logit():
    rows=fixture()
    assert matched(rows,'native')['available']==matched(rows,'augmented')['available']==3


def test_bootstrap_matches_separate_brute_force_with_repeated_clusters():
    import copy
    a=fixture('a',0)[1:4];b=fixture('b',1)[1:4]
    for j,r in enumerate(a+b):
        r['scores']['native']=float(j%3);r['scores']['augmented']=float(2-j%3)
        r['failure']=j%2==0
    thresholds={m:calibrate(a+b,m) for m in ['native','augmented']};saved=copy.deepcopy(thresholds)
    actual=paired_bootstrap(a+b,['a','b'],thresholds,repeats=100,seed=42)
    rng=np.random.default_rng(42);diff=[]
    for _ in range(100):
        indices=rng.integers(0,2,2);expanded=[r for i in indices for r in (a if i==0 else b)]
        m=math.ceil(.8*len(expanded));risks={}
        for method in ['native','augmented']:
            chosen=sorted(expanded,key=lambda r:(r['scores'][method],r['tie']))[:m]
            risks[method]=sum(r['failure'] for r in chosen)/m
        diff.append(risks['augmented']-risks['native'])
    assert actual['intervals']['matched_risk_difference']['ci95']==np.percentile(diff,[2.5,97.5]).tolist()
    assert actual['intervals']['matched_risk_difference']['defined_replicates']==100
    assert thresholds==saved


def test_complete_fixture_prediction_to_group_bootstrap_pipeline():
    import runpy
    pipeline=runpy.run_path('scripts/pilot_fixture_check.py')['run_fixture']
    result=pipeline()
    assert result['all_pipeline_stages_executed']
    assert result['result']['matched']['native']['reachable']
    assert result['result']['bootstrap']['allocated_groups']==['eval','empty']


def test_frozen_config_constants_match_executable_scores():
    import json
    c=json.loads(open('configs/0002-exploratory-v1.json').read())
    assert c['failure_threshold_px_strict_gt']==4
    assert c['matched_coverage_target']==c['calibration_coverage_target']==.8
    assert c['bootstrap_seed']==20261018 and c['bootstrap_repeats']==2000
    assert c['native_probability_floor']==1e-8
    assert c['practical_risk_difference_target']==-.02
    s=prediction_scores([0,1,2,3],[[0,0],[0,0],[8,0],[1000,0]],[-1]*4,[-1]*4)
    delta=s['scores']['augmented']-s['scores']['native']
    assert delta[2]==pytest.approx(8/c['motion_residual_scale_px']*c['motion_weight'])
    assert delta[3]==pytest.approx(c['motion_residual_cap']*c['motion_weight'])


def test_visibility_invalid_is_incorrect_even_when_occluded():
    rows=fixture();rows[4]['visible']=False;rows[4]['opportunity']=False
    # No finite occ head at row4; cannot be counted as correctly occluded.
    result=evaluate(rows,rows,['a'],repeats=2)
    assert result['visibility_accuracy']==0
