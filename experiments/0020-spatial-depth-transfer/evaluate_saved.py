"""Score sealed candidate choices, including both stronger fixed 0019 controls."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CONFIG=ROOT/'configs/0020-comparison-bindings.json'
ARMS=('e_norm','e_ts','depth_scale','depth_affine')
SELECTORS=('pixel_msac','global_affine_ls','spatial_weight_only','random_point_transfer','spatial_transfer')
AUTHOR_ARMS=('depth_scale_truncated_cauchy','depth_affine_truncated_cauchy')
COMPARATORS=(*SELECTORS[:4],*ARMS,*AUTHOR_ARMS)
spec=importlib.util.spec_from_file_location('evr0020_reference_metrics',HERE.parent/'0017-depth-information/evaluate_saved.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def material(before,after):
    return after-before>=1. or (before<5.<=after)


def verify_outer(seal_path,c,expected_commit=None):
    folder=Path(seal_path).parent.parent
    receipt,log=folder/'prediction-process.json',folder/'prediction.log'
    old.require(not receipt.is_symlink() and not log.is_symlink(),'Unsafe outer evidence')
    r=json.loads(receipt.read_text())
    old.require(r['schema']=='evr0020.outer-operation.v1' and r['mode']=='prediction'
                and r['status']=='completed' and r['exit_code']==0 and r['termination'] is None
                and r['remaining_process_groups_after_cleanup']==[],'Incomplete outer prediction operation')
    for key,limit in (('wall_seconds',600.),('peak_group_rss_kib',8*1024*1024)):
        old.require(type(r[key]) in (int,float) and math.isfinite(r[key]) and 0<=r[key]<=limit,'Outer resource limit failed')
    old.require(r['hard_timeout_seconds']==600 and r['group_rss_limit_kib']==8*1024*1024,'Outer resource configuration changed')
    old.require(r['helper_sha256']==c.sha256(HERE/'run_operation.py') and r['log_sha256']==c.sha256(log),'Outer helper/log changed')
    if expected_commit is None:
        expected_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old.require(r['source_commit']==expected_commit,'Outer source commit differs from current freeze')
    return {'receipt_sha256':c.sha256(receipt),'log_sha256':c.sha256(log),'source_commit':r['source_commit']}


def choice(selector):
    values=selector['scores']
    old.require(len(values)==4 and all(type(v) in (float,int) and math.isfinite(v) and 0.<=v<=2. for v in values),'Bad selector scores')
    index=min(range(4),key=lambda i:(values[i],i))
    old.require(selector['selected_index']==index and selector['selected_arm']==ARMS[index],'Selection not score minimum')
    return ARMS[index]


def compute(records,identities,reference,author):
    old.require(len(records)==len(identities)==len(reference['rows'])==len(author['rows'])==80,'Incomplete all80 comparison')
    old.require([r['pair_index'] for r in records]==list(range(80)),'Score order/coverage differs')
    rows=[]
    for record,identity,ref,auth in zip(records,identities,reference['rows'],author['rows']):
        for item in (record,ref,auth):
            old.require(all(item[k]==identity[k] for k in ('pair_index','scene','key')),'Reference/score identity mismatch')
        old.require(record['status']=='completed','Technical failure cannot be scored')
        outcomes={a:ref['outcomes'][a] for a in ARMS}
        for a in ARMS:old.require(auth['outcomes'][a]==outcomes[a],'Original outcomes differ between references')
        outcomes.update({a:auth['outcomes'][a] for a in AUTHOR_ARMS})
        selectors=record['scores']['selectors']
        old.require(set(selectors)==set(SELECTORS),'Missing or additional selector')
        selected={s:choice(selectors[s]) for s in SELECTORS}
        outcomes.update({s:dict(outcomes[selected[s]]) for s in SELECTORS})
        for value in outcomes.values():
            for key in ('penalized_max_degrees','penalized_max_signfolded_degrees'):
                old.require(math.isfinite(value[key]) and 0.<=value[key]<=180.,'Invalid saved metric')
        rows.append({**{k:identity[k] for k in ('pair_index','scene','key')},'outcomes':outcomes,
            'selected_arms':selected,'score_margins':{s:sorted(selectors[s]['scores'])[1]-min(selectors[s]['scores']) for s in SELECTORS},
            'score_timings':record['timings'],'candidate_generation_seconds':record['candidate_generation_seconds']})
    names=(*SELECTORS,*ARMS,*AUTHOR_ARMS)
    def summarize(items):return {a:old.summarize([r['outcomes'][a] for r in items]) for a in names}
    summaries=summarize(rows)
    scenes={s:summarize([r for r in rows if r['scene']==s]) for s in old.SCENES}
    best=max(COMPARATORS,key=lambda a:(summaries[a]['auc_fraction']['5'],-COMPARATORS.index(a)))
    best20=max(summaries[a]['auc_fraction']['20'] for a in COMPARATORS)
    gain=summaries['spatial_transfer']['auc_fraction']['5']-summaries[best]['auc_fraction']['5']
    a20=summaries['spatial_transfer']['auc_fraction']['20']
    positive=[s for s in old.SCENES if scenes[s]['spatial_transfer']['auc_fraction']['5']>scenes[s][best]['auc_fraction']['5']]
    conditions={'auc5_at_least_2pp':gain>=.02,'at_least_two_positive_scenes':len(positive)>=2,'auc20_loss_at_most_half_pp':a20>=best20-.005,'full80_coverage':True}
    comparisons={}
    for a in COMPARATORS:
        pairs=[]
        for row in rows:
            before=row['outcomes'][a]['penalized_max_degrees'];after=row['outcomes']['spatial_transfer']['penalized_max_degrees']
            pairs.append({'pair_index':row['pair_index'],'scene':row['scene'],'error_delta_degrees':after-before,
                'material_harm':material(before,after),'material_benefit':material(after,before),
                'strict_better':after<before,'strict_worse':after>before,'tie':after==before})
        comparisons[a]={'auc_delta_percentage_points':{t:100*(summaries['spatial_transfer']['auc_fraction'][t]-summaries[a]['auc_fraction'][t]) for t in ('5','10','20')},
            'counts':{k:sum(p[k] for p in pairs) for k in ('material_harm','material_benefit','strict_better','strict_worse','tie')},'pairs':pairs}
    return {'rows':rows,'summaries':summaries,'scene_summaries':scenes,'comparisons':comparisons,
        'selected_arm_counts':{s:{a:sum(r['selected_arms'][s]==a for r in rows) for a in ARMS} for s in SELECTORS},
        'development_gate':{'best_auc5_comparator':best,'auc5_gain_percentage_points':100*gain,
            'positive_scenes_against_same_comparator':positive,'auc20_delta_vs_best_comparator_percentage_points':100*(a20-best20),
            'conditions':conditions,'passes':all(conditions.values()),'benchmark_comparator_is_not_deployable_selector':True},
        'charged_scoring_seconds':sum(r['score_timings']['charged_total_seconds'] for r in rows),
        'historical_all_four_candidates_seconds':sum(r['candidate_generation_seconds'] for r in rows)}


def evaluate(manifest_path,baseline_seal,seal_path,output_path):
    import common as c
    import run_diagnostic as runner
    old.require(not Path(output_path).exists(),'Output already exists')
    start=time.perf_counter()
    manifest,seal,records=runner.verify_seal(manifest_path,baseline_seal,seal_path)
    outer=verify_outer(seal_path,c)
    sources=c.source_hashes();mh=c.sha256(manifest_path);sh=c.sha256(seal_path)
    config=json.loads(CONFIG.read_text());refs={}
    for key in ('baseline_evaluation','author_loss_evaluation'):
        item=config[key];path=ROOT/item['path']
        old.require(path.resolve().is_relative_to(ROOT) and not path.is_symlink(),'Unsafe comparison reference')
        old.require(c.sha256(path)==item['sha256'],'Comparison reference changed')
        refs[key]=json.loads(path.read_text())
        old.require(refs[key]['status']=='completed' and refs[key]['manifest_sha256']==mh,'Ineligible comparison reference')
    old.require(refs['baseline_evaluation']['seal_sha256']==c.sha256(baseline_seal),'Original reference seal differs')
    result=compute(records,manifest['selected_pairs'],refs['baseline_evaluation'],refs['author_loss_evaluation'])
    runner.verify_seal(manifest_path,baseline_seal,seal_path)
    old.require(verify_outer(seal_path,c)==outer,'Outer evidence changed during evaluation')
    old.require(c.source_hashes()==sources and c.sha256(manifest_path)==mh and c.sha256(seal_path)==sh,'Changed source/seal during evaluation')
    for item in config.values():old.require(c.sha256(ROOT/item['path'])==item['sha256'],'Changed reference during evaluation')
    output={'schema':'evr0020.evaluation.v1','status':'completed','manifest_sha256':mh,'seal_sha256':sh,
        'source_sha256':sources,'comparison_bindings':config,'prediction_outer_binding':outer,'evaluation_seconds':time.perf_counter()-start,
        'scope':'Development only. All candidate poses saw the full pair; omitted groups apply only to nuisance calibration. No held-out, independent-pose-validation, matched-time or causal-mechanism claim.',
        'manuscript_admitted':False,**result}
    c.write_new(output_path,output)
    print(json.dumps({'status':'completed','development_gate':result['development_gate']}),flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','baseline-seal','seal','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();evaluate(a.manifest,a.baseline_seal,a.seal,a.output)
