#!/usr/bin/env python3
"""Saved-only 0020 evidence/selection/metric audit; never reruns depth fitting."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent0017',HERE.parent/'0017-review/review_saved.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
ARMS=u.ARMS
SELECTORS=('pixel_msac','global_affine_ls','spatial_weight_only','random_point_transfer','spatial_transfer')
AUTHOR=('depth_scale_truncated_cauchy','depth_affine_truncated_cauchy')
COMPARATORS=(*SELECTORS[:4],*ARMS,*AUTHOR)
FREEZE='bec376568729c3a158cd9f1279cd00d3e5f1de76'
SALT='evr0020-random-v1\0'
MASS_TOL=1e-12
AUC_TOL=1e-10


def close(a,b,label,tol=MASS_TOL):
    u.need(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
           and abs(a-b)<=tol,label)


def integer(x,lo,hi,label):
    u.need(type(x) is int and lo<=x<=hi,label)


def random_ids(key,counts):
    order=sorted(range(sum(counts)),key=lambda i:(hashlib.sha256((SALT+key+'\0'+str(i)).encode()).digest(),i))
    result=[None]*len(order);offset=0
    for k,n in enumerate(counts):
        for i in order[offset:offset+n]:result[i]=k
        offset+=n
    return result


def check_scores(scores,candidates,key):
    """Check saved identities and forced-cost bounds; successful residuals stay inputs."""
    u.finite_tree(scores);n=candidates[0]['input_arrays']['x1']['shape'][0]
    integer(n,1,10**8,'Bad match count')
    u.need(scores['schema']=='evr0020.scores.v1' and scores['key']==key and scores['count']==n
           and scores['arms']==list(ARMS),'Score identity')
    sp,rp=scores['spatial_partition'],scores['random_partition'];ids=sp['ids'];counts=sp['counts']
    u.need(len(ids)==n and len(counts)==4,'Partition length')
    for k in ids:integer(k,0,3,'Partition ID')
    u.need(counts==[ids.count(k) for k in range(4)],'Partition counts')
    available=all(counts)
    u.need(sp['status']==('completed' if available else 'unavailable'),'Partition status')
    if available:
        w=[1/(4*counts[k]) for k in ids]
        u.need(rp['status']=='completed' and rp['counts']==counts and rp['salt']==SALT
               and rp['ids']==random_ids(key,counts),'Random fold derivation')
    else:
        w=None
        u.need(sp['reason']=='spatial_partition_unavailable' and rp['status']=='unavailable'
               and rp['ids'] is None and rp['counts']==counts and rp['salt']==SALT,'Unavailable partition')
    for norm in scores['normalization'].values():
        u.need(norm['count']==n and norm['status'] in ('completed','unavailable'),'Normalization status')
        if norm['status']=='completed':u.need(norm['rms']>0 and norm['reason'] is None,'Normalization scale')
    u.need([r['arm'] for r in scores['candidates']]==list(ARMS),'Candidate order')
    counters=Counter(candidate_count=4,fit_calls=0,svd_calls=0,fit_completed_count=0,fit_unavailable_count=0)
    failures=Counter();expected_values={s:[] for s in SELECTORS};forced=[]
    for row,candidate in zip(scores['candidates'],candidates):
        u.need(candidate['status']=='completed' and row['geometry']['count']==n,'Candidate/geometry status')
        close(row['geometry']['pixel_msac'],candidate['final_scores']['pixel_msac'],'Frozen geometry sum',1e-10*max(1,n))
        G=row['geometry']['pixel_msac']/n;u.need(0<=G<=1 and row['geometric_mean']==G,'Geometric mean')
        unit,_=u.unit(candidate['estimate']['t']);u.compare(row['unit_translation'],unit,'Translation gauge',2e-12)
        support=row['fit_support_count'];integer(support,0,n,'Fit support count')
        u.need(support==row['geometry']['pixel_inliers'],'Support/geometry count')
        u.need(set(row['fits'])==set(row['depth_scores'])==set(SELECTORS[1:]),'Calibration control coverage')
        weighted_global_mass=None
        for method in SELECTORS[1:]:
            D=row['depth_scores'][method];u.need(type(D) in (int,float) and 0<=D<=1,'Depth score range')
            records=row['fits'][method]
            if not available and method!='global_affine_ls':
                u.need(records==[{'status':'unavailable','reason':'spatial_partition_unavailable'}] and D==1,'Unavailable full-cost placeholder')
                continue
            weights=[1/n]*n if method=='global_affine_ls' else w
            folds=None if method in ('global_affine_ls','spatial_weight_only') else (ids if method=='spatial_transfer' else rp['ids'])
            labels=[None] if folds is None else list(range(4))
            u.need(len(records)==len(labels),'Fit coverage')
            forced_mass=0.;all_forced=True;train_counts=[];train_masses=[]
            for label,record in zip(labels,records):
                u.need(type(record['fold']) is type(label) and record['fold']==label,'Fit order')
                vi=list(range(n)) if folds is None else [i for i in range(n) if folds[i]==label]
                ti=list(range(n)) if folds is None else [i for i in range(n) if folds[i]!=label]
                vw=[weights[i] for i in vi];mass=math.fsum(vw)
                close(record['validation_weight_mass'],mass,'Validation mass')
                fit,val=record['fit'],record['validation'];tc=fit['training_count'];tm=fit['training_weight_mass']
                integer(tc,0,min(support,len(ti)),'Training support count');train_counts.append(tc);train_masses.append(tm)
                possible=sorted(weights[i] for i in ti)
                u.need(math.fsum(possible[:tc])-MASS_TOL<=tm<=math.fsum(possible[len(possible)-tc:] if tc else [])+MASS_TOL,'Training support mass bounds')
                if folds is None:u.need(tc==support,'Global training count')
                if method=='global_affine_ls':close(tm,support/n,'Uniform support mass')
                if method=='spatial_weight_only':weighted_global_mass=tm
                integer(val['count'],len(vi),len(vi),'Validation count')
                integer(val['direction_denominator'],2*len(vi),2*len(vi),'Fixed directional denominator')
                u.need(type(fit['svd_attempted']) is bool,'SVD attempt type');counters['fit_calls']+=1;counters['svd_calls']+=fit['svd_attempted']
                if fit['singular_values'] is not None:
                    singular=fit['singular_values'];u.need(len(singular)==4 and singular==sorted(singular,reverse=True) and min(singular)>=0,'SVD values')
                    threshold=2.220446049250313e-16*max(3*tc,4)*singular[0]
                    close(fit['rank_threshold'],threshold,'Rank threshold')
                    u.need(fit['rank']==sum(s>threshold for s in singular),'Numerical rank')
                if fit['status']=='unavailable':
                    counters['fit_unavailable_count']+=1;failures[fit['reason']]+=1
                    u.need(val['status']=='fit_unavailable' and val['reason']==fit['reason']
                           and val['fit_failed_direction_count']==2*len(vi) and val['forward'] is None and val['reverse'] is None,'Failed fit lost validation mass')
                    forced_mass+=mass
                else:
                    counters['fit_completed_count']+=1
                    u.need(fit['status']=='completed' and fit['reason'] is None and fit['svd_attempted']
                           and fit['rank']==4 and len(fit['coefficients'])==4 and fit['coefficients'][0]>0
                           and fit['coefficients'][2]>0,'Completed fit admission')
                    u.need(val['status']=='completed' and val['fit_failed_direction_count']==0,'Completed validation status')
                    ordered=sorted(vw)
                    for direction in ('forward','reverse'):
                        d=val[direction]
                        for k in ('count','valid_count','invalid_count','source_invalid_count','transformed_nonfinite_count','transformed_nonpositive_or_nonfinite_z_count','projection_nonfinite_count','capped_count'):
                            integer(d[k],0,len(vi),'Directional count')
                        u.need(d['count']==len(vi) and d['valid_count']+d['invalid_count']==len(vi)
                               and d['capped_count']>=d['invalid_count'],'Directional partition/cap')
                        u.need(all(d[k]<=d['invalid_count'] for k in ('source_invalid_count','transformed_nonfinite_count','transformed_nonpositive_or_nonfinite_z_count','projection_nonfinite_count')),'Invalid reason count')
                        forced_mass+=.5*math.fsum(ordered[:d['capped_count']])
                        all_forced &= d['capped_count']==len(vi)
            if folds is not None:
                u.need(sum(support-x for x in train_counts)==support,'Omitted support counts do not partition support')
                for k,tc in enumerate(train_counts):integer(support-tc,0,counts[k],'Omitted support size')
                close(math.fsum(train_masses),3*weighted_global_mass,'Cross-fit training mass conservation')
                if method=='spatial_transfer':
                    for k,tm in enumerate(train_masses):close(weighted_global_mass-tm,(support-train_counts[k])/(4*counts[k]),'Spatial omitted support mass')
            u.need(D+MASS_TOL>=forced_mass,'Depth score undercharges forced full-cost mass')
            if all_forced:close(D,1.,'All-failed/capped calibration must retain full mass')
            forced.append({'arm':row['arm'],'selector':method,'forced_cost_lower_bound':forced_mass,'saved_depth_cost':D})
        values={'pixel_msac':G,**{s:G+row['depth_scores'][s] for s in SELECTORS[1:]}}
        u.need(row['scores']==values,'Candidate G+D identity')
        for s in SELECTORS:expected_values[s].append(values[s])
    u.need(scores['counters']==dict(counters) and counters['fit_calls']<=40,'Actual work counters')
    u.need(set(scores['selectors'])==set(SELECTORS),'Selector coverage');choices={}
    for name,values in expected_values.items():
        saved=scores['selectors'][name];index=min(range(4),key=lambda i:(values[i],i))
        u.need(type(saved['selected_index']) is int and saved=={'selected_arm':ARMS[index],'selected_index':index,'score':values[index],'scores':values},'Exact selector/tie identity')
        choices[name]=ARMS[index]
    return choices,dict(counters),dict(failures),forced


def metrics(identities,records,reference,author):
    u.need(len(records)==len(identities)==len(reference['rows'])==len(author['rows'])==80,'All80 metric coverage')
    names=(*SELECTORS,*ARMS,*AUTHOR);rows=[]
    for identity,record,ref,auth in zip(identities,records,reference['rows'],author['rows']):
        for item in (record,ref,auth):u.need(all(item[k]==identity[k] for k in ('pair_index','scene','key')),'Metric identity')
        outcomes={a:ref['outcomes'][a] for a in ARMS}
        u.need(all(auth['outcomes'][a]==outcomes[a] for a in ARMS),'Original reference mismatch')
        outcomes.update({a:auth['outcomes'][a] for a in AUTHOR})
        choices={s:record['scores']['selectors'][s]['selected_arm'] for s in SELECTORS}
        outcomes.update({s:outcomes[choices[s]] for s in SELECTORS})
        rows.append({**{k:identity[k] for k in ('pair_index','scene','key')},'outcomes':outcomes,'selected_arms':choices,
            'score_margins':{s:sorted(record['scores']['selectors'][s]['scores'])[1]-min(record['scores']['selectors'][s]['scores']) for s in SELECTORS},
            'score_timings':record['timings'],'candidate_generation_seconds':record['candidate_generation_seconds']})
    def summarize(subset):return {name:u.summary([r['outcomes'][name] for r in subset]) for name in names}
    full=summarize(rows);scenes={s:summarize([r for r in rows if r['scene']==s]) for s in u.SCENES}
    u.need(all(scenes[s]['spatial_transfer']['pair_count']==16 for s in u.SCENES),'Scene denominators')
    best=min(COMPARATORS,key=lambda a:(-full[a]['auc_fraction']['5'],COMPARATORS.index(a)))
    gain=full['spatial_transfer']['auc_fraction']['5']-full[best]['auc_fraction']['5']
    best20=max(full[a]['auc_fraction']['20'] for a in COMPARATORS);d20=full['spatial_transfer']['auc_fraction']['20']-best20
    positive=[s for s in u.SCENES if scenes[s]['spatial_transfer']['auc_fraction']['5']>scenes[s][best]['auc_fraction']['5']]
    conditions={'auc5_at_least_2pp':gain>=.02,'at_least_two_positive_scenes':len(positive)>=2,
        'auc20_loss_at_most_half_pp':full['spatial_transfer']['auc_fraction']['20']>=best20-.005,'full80_coverage':True}
    comparisons={}
    for a in COMPARATORS:
        pairs=[]
        for row in rows:
            before=row['outcomes'][a]['penalized_max_degrees'];after=row['outcomes']['spatial_transfer']['penalized_max_degrees']
            pairs.append({'pair_index':row['pair_index'],'scene':row['scene'],'error_delta_degrees':after-before,
                'material_harm':after-before>=1 or before<5<=after,'material_benefit':before-after>=1 or after<5<=before,
                'strict_better':after<before,'strict_worse':after>before,'tie':after==before})
        comparisons[a]={'auc_delta_percentage_points':{t:100*(full['spatial_transfer']['auc_fraction'][t]-full[a]['auc_fraction'][t]) for t in ('5','10','20')},
            'counts':{k:sum(p[k] for p in pairs) for k in ('material_harm','material_benefit','strict_better','strict_worse','tie')},'pairs':pairs}
    return {'rows':rows,'summaries':full,'scene_summaries':scenes,'comparisons':comparisons,
        'selected_arm_counts':{s:{a:sum(r['selected_arms'][s]==a for r in rows) for a in ARMS} for s in SELECTORS},
        'development_gate':{'best_auc5_comparator':best,'auc5_gain_percentage_points':100*gain,'positive_scenes_against_same_comparator':positive,
            'auc20_delta_vs_best_comparator_percentage_points':100*d20,'conditions':conditions,'passes':all(conditions.values()),'benchmark_comparator_is_not_deployable_selector':True},
        'charged_scoring_seconds':math.fsum(r['score_timings']['charged_total_seconds'] for r in rows),
        'historical_all_four_candidates_seconds':math.fsum(r['candidate_generation_seconds'] for r in rows)}


def audit(repo):
    repo=Path(repo);folder=repo/'results/0020-spatial-depth-transfer';pred=folder/'predictions'
    mp=repo/'configs/0017-development-manifest.json';bs=repo/'results/0017-depth/predictions/seal.json'
    be=repo/'results/0017-depth/evaluation.json';m,base_seal,reference,bank,_=u.verify_files(repo,mp,bs,be)
    seal=u.load(pred/'seal.json');evaluation=u.load(folder/'evaluation.json');config=u.load(repo/'configs/0020-comparison-bindings.json')
    paths=set(u.SOURCE_PATHS)|{'configs/0020-comparison-bindings.json','protocols/0020-spatial-depth-transfer.md','protocols/0019-author-loss.md'}
    for number in ('0019-author-loss','0020-spatial-depth-transfer'):
        paths.update('experiments/'+number+'/'+p for p in ('common.py','predict_pair.py','run_diagnostic.py','evaluate_saved.py','run_operation.py'))
    paths.add('experiments/0020-spatial-depth-transfer/score.py');sources={p:u.digest(repo/p) for p in paths}
    hashes={str(mp.relative_to(repo)):u.digest(mp),str(bs.relative_to(repo)):u.digest(bs),str(be.relative_to(repo)):u.digest(be)}
    def bound(path):
        path=Path(path);sha=u.digest(path);hashes[str(path.relative_to(repo))]=sha;return sha
    u.need(seal['schema']=='evr0020.seal.v1' and seal['status']=='completed' and seal['error'] is None,'Eligible seal')
    u.need(seal['source_sha256']==sources and seal['manifest_sha256']==u.digest(mp) and seal['library']==base_seal['library'],'Seal source/library')
    oldrows={(r['pair_index'],r['arm']):r for r in base_seal['rows']}
    binding={'seal_sha256':u.digest(bs),'predictions':{f'{i}:{a}':oldrows[i,a]['sha256'] for i in range(80) for a in ARMS}}
    u.need(seal['baseline_binding']==binding and seal['verified_assets']==base_seal['verified_assets'],'Original bank/assets binding')
    integrity=pred/'integrity.json'
    u.need(bound(integrity)==seal['integrity_sha256'] and u.load(integrity)=={'manifest_sha256':u.digest(mp),'source_sha256':sources,'library':base_seal['library'],'baseline_binding':binding,'verified_assets':base_seal['verified_assets']},'Integrity receipt')
    u.need([r['pair_index'] for r in seal['rows']]==list(range(80)) and 0<=seal['wall_seconds']<=600,'All80/resource seal')
    records=[];totals=Counter();failure_reasons=Counter();forced=[];nonempty=[]
    for row in seal['rows']:
        i=row['pair_index'];identity=m['selected_pairs'][i];candidates=[bank[i,a] for a in ARMS]
        files={k:u.safe(pred,row[k]) for k in ('path','raw_path','process_path')}
        for k,h in (('path','sha256'),('raw_path','raw_sha256'),('process_path','process_sha256')):u.need(bound(files[k])==row[h],'Record hash')
        raw,final,process=(u.load(files[k]) for k in ('raw_path','path','process_path'));log=u.safe(pred,process['log_path'])
        u.need(bound(log)==process['log_sha256'] and process['raw_worker_path']==row['raw_path'] and process['raw_worker_sha256']==row['raw_sha256'],'Process/log link')
        if log.stat().st_size:nonempty.append(log.name)
        expected={k:identity[k] for k in ('pair_index','scene','key','filename')}
        expected.update(schema='evr0020.prediction.v1',manifest_sha256=u.digest(mp),source_sha256=sources,library=base_seal['library'],
            dataset_sha256_declared=identity['dataset_sha256'],dataset_sha256_verified=identity['dataset_sha256'],dataset_bytes=identity['dataset_bytes'],
            baseline_seal_sha256=u.digest(bs),candidate_prediction_sha256={a:oldrows[i,a]['sha256'] for a in ARMS},candidate_order=list(ARMS),max_fits=40)
        u.need(all(raw.get(k)==v for k,v in expected.items()) and raw['status']==row['status']=='completed','Worker identity/status')
        for candidate in candidates:
            u.need(all(raw[k]==candidate[k] for k in ('input_arrays','calibration','input_diagnostics')),'Original input equality')
        u.need(process['exit_code']==0 and process['termination'] is None and process['watchdog_error'] is None
               and 0<=process['wall_seconds']<=30 and 0<=process['sampled_peak_rss_kib']<=8388608,'Worker process/resource')
        timing={**raw['timings'],'charged_total_seconds':process['wall_seconds']}
        u.need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in timing.values()),'Timing values')
        components=math.fsum(timing[k] for k in ('binding_seconds','input_seconds','score_seconds','output_seconds'))
        u.need(components<=timing['worker_total_seconds']+1e-6<=process['wall_seconds']+2e-6,'Charged nesting')
        u.need(final=={**raw,'timings':timing,'process_receipt':process},'Final normalization')
        close(raw['candidate_generation_seconds'],math.fsum(c['timings']['charged_total_seconds'] for c in candidates),'Full candidate-bank cost',1e-10)
        _,count,fail,bounds=check_scores(raw['scores'],candidates,identity['key']);totals.update(count);failure_reasons.update(fail)
        forced.extend({'pair_index':i,**b} for b in bounds);records.append(final)
    u.need(totals['fit_calls']==seal['fit_attempts'] and totals['fit_calls']<=3200,'Cohort fit cap')
    outer={}
    for mode in ('prediction','evaluation'):
        path=folder/(mode+'-process.json');p=u.load(path);log=folder/(mode+'.log')
        u.need(p['schema']=='evr0020.outer-operation.v1' and p['mode']==mode and p['status']=='completed' and p['exit_code']==0
               and p['termination'] is None and p['remaining_process_groups_after_cleanup']==[],'Outer operation')
        u.need(p['source_commit']==FREEZE and p['helper_sha256']==sources['experiments/0020-spatial-depth-transfer/run_operation.py']
               and p['log_sha256']==bound(log),'Outer freeze/log')
        u.need(0<=p['wall_seconds']<=p['hard_timeout_seconds']==600 and 0<=p['peak_group_rss_kib']<=p['group_rss_limit_kib']==8388608,'Outer resources')
        bound(path);outer[mode]=p
    refs={}
    for key,item in config.items():
        path=repo/item['path'];u.need(path.resolve().is_relative_to(repo.resolve()) and not path.is_symlink(),'Unsafe reference')
        u.need(bound(path)==item['sha256'],'Comparison reference hash');refs[key]=u.load(path)
        u.need(refs[key]['status']=='completed' and refs[key]['manifest_sha256']==u.digest(mp),'Comparison eligibility')
    u.need(refs['baseline_evaluation']==reference,'Baseline comparison content')
    result=metrics(m['selected_pairs'],records,reference,refs['author_loss_evaluation'])
    u.need(evaluation['schema']=='evr0020.evaluation.v1' and evaluation['status']=='completed' and evaluation['source_sha256']==sources
           and evaluation['manifest_sha256']==u.digest(mp) and evaluation['seal_sha256']==bound(pred/'seal.json')
           and evaluation['comparison_bindings']==config and evaluation['manuscript_admitted'] is False,'Evaluation binding/scope')
    u.need(evaluation['prediction_outer_binding']=={'receipt_sha256':u.digest(folder/'prediction-process.json'),'log_sha256':u.digest(folder/'prediction.log'),'source_commit':FREEZE},'Evaluation outer binding')
    for key,value in result.items():u.compare(evaluation[key],value,'Independent '+key,100*AUC_TOL if key in ('comparisons','development_gate') else AUC_TOL)
    bound(folder/'evaluation.json')
    for name,sha in sources.items():u.need(u.digest(repo/name)==sha,'Source changed during audit')
    for name,sha in hashes.items():u.need(u.digest(repo/name)==sha,'Saved input changed during audit')
    return {'schema':'evr0020.independent-review.v1','status':'passed','freeze_commit':FREEZE,'pair_count':80,
        'review_source_sha256':u.digest(__file__),'independent_helper_sha256':u.digest(HERE.parent/'0017-review/review_saved.py'),
        'mass_tolerance':MASS_TOL,'auc_fraction_tolerance':AUC_TOL,'counters':dict(totals),'failure_reasons':dict(failure_reasons),
        'nonempty_worker_logs':nonempty,'forced_cost_bounds':forced,'outer_processes':outer,
        'bindings':{'sources':sources,'saved_files':hashes},
        'limitations':['Saved-only evidence and arithmetic audit; no private coordinates, HDF5, native estimator or production evaluator imported.',
            'Successful-fit depth costs remain saved inputs; their residuals and fitted coefficients are not independently rerun.',
            'Spatial IDs are checked internally but their original coordinate medians cannot be reconstructed without private arrays.',
            'Failed-fit full mass and forced capped-direction lower bounds are reconstructed from saved IDs/weights/counts.',
            'All candidate poses saw the full input; five scenes are permanently development, not held-out evaluation.'],**result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x+') as f:
        f.write('{"status":"started"}\n');f.flush()
        try:report=audit(a.repo)
        except BaseException as exc:
            f.seek(0);json.dump({'status':'failed','error':type(exc).__name__+': '+str(exc),'review_source_sha256':u.digest(__file__)},f,indent=2);f.write('\n');f.truncate();f.flush();raise
        f.seek(0);json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.truncate()
    print(json.dumps({'status':'passed','pair_count':80,'fit_calls':report['counters']['fit_calls'],'development_gate':report['development_gate']}))
