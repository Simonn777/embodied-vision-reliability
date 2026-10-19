"""Evaluate the complete 0019 known-control cohort against saved 0017 labels."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT/'results/0017-depth/evaluation.json'
REFERENCE_SHA256 = '66f3f693fa7473aba5371fe473c4f8c9df39a3dd711a32941f28f3e0380eb151'
ARMS = ('depth_scale','depth_affine')


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


old=module('evr0019_reference_metrics',HERE.parent/'0017-depth-information/evaluate_saved.py')


def harm(before,after):
    return after-before>=1. or (before<5. and after>=5.)


def compute(records, identities, reference):
    old.require(len(records)==160 and len(identities)==80,'Incomplete cohort')
    old.require(len(reference['rows'])==80,'Incomplete reference')
    bank={}
    for record in records:
        key=(record['pair_index'],record['arm'])
        old.require(key not in bank and key[1] in ARMS and 0<=key[0]<80,'Bad/duplicate identity')
        bank[key]=record
    rows=[]
    for identity,ref in zip(identities,reference['rows']):
        i=identity['pair_index']
        old.require(all(ref[k]==identity[k] for k in ('pair_index','scene','key')),'Reference identity mismatch')
        outcomes=dict(ref['outcomes']);timings={}
        for arm in ARMS:
            r=bank[(i,arm)]
            old.require(all(r[k]==identity[k] for k in ('pair_index','scene','key')),'Prediction identity mismatch')
            old.require(r['status'] in ('completed','estimator_error','timeout'),'Ineligible record')
            error=None
            if r['status']=='completed':
                gt=ref['ground_truth']
                error,_=old.metrics.pose_errors(r['estimate']['R'],r['estimate']['t'],gt['R_original'],gt['t_original'])
                folded=min(error['translation_degrees'],180.-error['translation_degrees'])
                error.update(translation_signfolded_degrees=folded,max_signfolded_degrees=max(error['rotation_degrees'],folded))
            name=arm+'_truncated_cauchy'
            outcomes[name]={'status':r['status'],'error':error,
                'penalized_max_degrees':180. if error is None else error['max_degrees'],
                'penalized_max_signfolded_degrees':180. if error is None else error['max_signfolded_degrees']}
            timings[name]=r['timings']
        comparisons={}
        for arm in ARMS:
            candidate=arm+'_truncated_cauchy';after=outcomes[candidate]['penalized_max_degrees']
            for baseline in (arm,'e_norm'):
                before=outcomes[baseline]['penalized_max_degrees']
                comparisons[candidate+'_vs_'+baseline]={'error_delta_degrees':after-before,
                    'material_harm':harm(before,after),'material_benefit':harm(after,before),
                    'damaged_5deg':before<5.<=after,'rescued_5deg':after<5.<=before}
        rows.append({**{k:identity[k] for k in ('pair_index','scene','key')},'outcomes':outcomes,
                     'comparisons':comparisons,'new_timings_by_arm':timings})
    names=(*old.ARMS,*(a+'_truncated_cauchy' for a in ARMS))
    def summaries(items):
        return {name:old.summarize([r['outcomes'][name] for r in items]) for name in names}
    totals=summaries(rows)
    comparisons={}
    for arm in ARMS:
        candidate=arm+'_truncated_cauchy'
        for baseline in (arm,'e_norm'):
            name=candidate+'_vs_'+baseline
            comparisons[name]={'auc_delta_percentage_points':{t:100*(totals[candidate]['auc_fraction'][t]-totals[baseline]['auc_fraction'][t]) for t in ('5','10','20')},
                **{k:sum(r['comparisons'][name][k] for r in rows) for k in ('material_harm','material_benefit','damaged_5deg','rescued_5deg')}}
    return {'rows':rows,'summaries':totals,'comparisons':comparisons,
        'scene_summaries':{s:summaries([r for r in rows if r['scene']==s]) for s in old.SCENES},
        'new_charged_seconds_by_arm':{a+'_truncated_cauchy':sum(r['new_timings_by_arm'][a+'_truncated_cauchy']['charged_total_seconds'] for r in rows) for a in ARMS}}


def evaluate(manifest_path,baseline_seal,seal_path,output_path):
    import common as c
    import run_diagnostic as runner
    out=Path(output_path)
    old.require(not out.exists(),'Output already exists')
    started=time.perf_counter()
    # Full verification before decoding the saved reference labels.
    manifest,seal,records=runner.verify_seal(manifest_path,baseline_seal,seal_path)
    sources=c.source_hashes();mh=c.sha256(manifest_path);sh=c.sha256(seal_path)
    old.require(c.sha256(REFERENCE)==REFERENCE_SHA256,'Reference evaluation changed')
    reference=json.loads(REFERENCE.read_text())
    old.require(reference['status']=='completed' and reference['manifest_sha256']==mh,'Reference binding mismatch')
    old.require(reference['seal_sha256']==c.sha256(baseline_seal),'Reference prediction seal mismatch')
    result=compute(records,manifest['selected_pairs'],reference)
    runner.verify_seal(manifest_path,baseline_seal,seal_path)
    old.require(c.source_hashes()==sources and c.sha256(manifest_path)==mh and c.sha256(seal_path)==sh
                and c.sha256(REFERENCE)==REFERENCE_SHA256,'Bindings changed during evaluation')
    output={'schema':'evr0019.evaluation.v1','status':'completed','pair_count':80,
        'manifest_sha256':mh,'seal_sha256':sh,'source_sha256':sources,'reference_evaluation_sha256':REFERENCE_SHA256,
        'evaluation_seconds':time.perf_counter()-started,'new_method_improvement_established':False,'manuscript_admitted':False,
        'scope':'Known author-loss controls; five development scenes. Historical timings do not establish matched-time efficacy. Frontend inference cost unavailable.',**result}
    c.write_new(out,output)
    print(json.dumps({'status':'completed','comparisons':result['comparisons']}),flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','baseline-seal','seal','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();evaluate(a.manifest,a.baseline_seal,a.seal,a.output)
