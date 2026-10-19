#!/usr/bin/env python3
"""Saved-only failed-diagnosis audit; never open labels, HDF5 or native code."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent
HELPER=HERE.parent/'0017-review/review_saved.py'
spec=importlib.util.spec_from_file_location('independent0017_helpers',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
need,load,digest,safe=h.need,h.load,h.digest,h.safe
ARMS=('depth_scale','depth_affine')
EXTRA_SOURCES={'experiments/0018-depth-trace/'+p for p in (
    'common.py','predict_pair.py','run_diagnostic.py','diagnose_trace.py','instrument_source.py',
    'build_instrumented.py','trace0018.h.in','trace_pybind.cc.in','qualify_native.py','qualify_observer.cpp','run_operation.py')}
EXTRA_SOURCES|={'protocols/0018-depth-lo-trace.md','configs/0018-libraries.json'}


def raw_equal(a,b,path='raw',floats=None):
    if type(b) is float:
        need(type(a) in (int,float) and math.isfinite(a) and math.isfinite(b),'Nonfinite '+path)
        need(abs(a-b)<=1e-12*max(1.,abs(a),abs(b)),'Raw parity '+path)
        if floats is not None:floats.append((a,b))
    elif isinstance(b,dict):
        need(isinstance(a,dict) and a.keys()==b.keys(),'Raw keys '+path)
        for k in b:raw_equal(a[k],b[k],path+'.'+k,floats)
    elif isinstance(b,list):
        need(isinstance(a,list) and len(a)==len(b),'Raw shape '+path)
        for i,(x,y) in enumerate(zip(a,b)):raw_equal(x,y,path+str(i),floats)
    else:need(type(a) is type(b) and a==b,'Raw discrete '+path)


def nonfinite_paths(value,prefix=''):
    if value is None:return [prefix]
    if isinstance(value,dict):return [v for k,x in value.items() for v in nonfinite_paths(x,prefix+'.'+k)]
    if isinstance(value,list):return [v for i,x in enumerate(value) for v in nonfinite_paths(x,prefix+'['+str(i)+']')]
    if type(value) in (int,float) and not math.isfinite(value):return [prefix]
    return []


def graph_only(trace):
    """Follow only explicit identifiers: no pose values or labels are consumed."""
    parents={e['lo_id']:(e['parent_kind'],e['parent_id']) for e in trace['events'] if e['kind']=='lo_begin'}
    returned=[e for e in trace['events'] if e['kind']=='pipeline' and e['phase']=='returned']
    need(len(returned)==1,'Returned event missing')
    source=(returned[0]['source_kind'],returned[0]['source_id']);ancestry=[]
    while source[0]=='lo':
        need(source[1] in parents and source[1] not in ancestry,'Broken/cyclic graph')
        ancestry.append(source[1]);source=parents[source[1]]
    need(source[0] in ('proposal','none'),'Unrecognized lineage root')
    return ancestry


def required_nonfinites(trace):
    ancestry=graph_only(trace);bad=[];counts=Counter();other=Counter()
    for index,event in enumerate(trace['events']):
        kind=event['kind'];required=kind in ('initial','exit') or (kind=='attempt' and event['accepted'])
        statebad=nonfinite_paths(event.get('state',{}),'state')
        costbad=nonfinite_paths(event.get('native_cost',{}),'native_cost')
        if required:
            counts['required_states']+=1
            if statebad or costbad:
                counts['nonfinite_required_states']+=1
                bad.append({'event_index':index,'kind':kind,'lo_id':event['lo_id'],'stage':event['stage'],
                    'state_paths':statebad,'cost_paths':costbad,'on_final_ancestry':event['lo_id'] in ancestry})
            elif not event['finite']:counts['auxiliary_nonfinite_only']+=1
        elif kind=='attempt' and not event['accepted']:
            counts['rejected_trials']+=1
            if statebad or costbad:counts['nonfinite_rejected_trials']+=1
        if statebad:other['nonfinite_'+kind+'_states']+=1
    return {'final_ancestry_lo_ids':ancestry,'counts':dict(counts),'all_event_nonfinite_state_counts':dict(other),'nonfinite_required_states':bad}


def audit(repo,manifest_path,baseline_seal,result_dir):
    repo,manifest_path,baseline_seal,result_dir=map(Path,(repo,manifest_path,baseline_seal,result_dir))
    manifest=load(manifest_path);bs=load(baseline_seal);config=load(repo/'configs/0018-libraries.json')
    mh,bh=digest(manifest_path),digest(baseline_seal)
    base_sources={p:digest(repo/p) for p in h.SOURCE_PATHS}
    sources={**base_sources,**{p:digest(repo/p) for p in EXTRA_SOURCES}}
    need(bs['status']=='completed' and bs['source_sha256']==base_sources and bs['manifest_sha256']==mh,'Original0017 seal')
    need(config['baseline']==bs['library'] and config['baseline']['binary_sha256']!=config['instrumented']['binary_sha256'],'Library identities')
    need(len(bs['rows'])==320 and len(manifest['selected_pairs'])==80,'Baseline/cohort coverage')
    original={};baseline_hashes={};hashes={str(manifest_path):mh,str(baseline_seal):bh}
    for row in bs['rows']:
        need(row['status']=='completed','Original coverage failure')
        paths={key:safe(baseline_seal.parent,row[key]) for key in ('path','raw_path','process_path')}
        for key,sha in (('path','sha256'),('raw_path','raw_sha256'),('process_path','process_sha256')):
            need(digest(paths[key])==row[sha],'Original0017 file changed');hashes[str(paths[key])]=row[sha]
        process=load(paths['process_path']);need(process['exit_code']==0 and process['termination'] is None,'Original process failure')
        log=safe(baseline_seal.parent,process['log_path']);need(digest(log)==process['log_sha256'],'Original log changed');hashes[str(log)]=process['log_sha256']
        if row['arm'] in ARMS:
            item=(row['pair_index'],row['arm']);need(item not in original,'Duplicate baseline item')
            original[item]=load(paths['path']);baseline_hashes[f'{item[0]}:{item[1]}']=row['sha256']
    need(len(original)==160,'Depth baseline Cartesian product')
    folder=result_dir/'predictions';sp=folder/'seal.json';seal=load(sp);hashes[str(sp)]=digest(sp)
    binding={'seal_sha256':bh,'predictions':baseline_hashes}
    need(seal['schema']=='evr0018.seal.v1' and seal['status']=='completed' and seal['error'] is None and seal['all_160_parity_passed'] is True,'Replay eligibility')
    need(seal['manifest_sha256']==mh and seal['source_sha256']==sources and seal['baseline_binding']==binding and seal['library']==config['instrumented'],'Replay seal bindings')
    assets={r['filename']:{'bytes':r['dataset_bytes'],'sha256':r['dataset_sha256']} for r in manifest['selected_pairs']}
    need(seal['verified_assets']==assets,'Asset metadata')
    ip=folder/'integrity.json';need(digest(ip)==seal['integrity_sha256'],'Integrity hash');hashes[str(ip)]=digest(ip)
    need(load(ip)=={'manifest_sha256':mh,'source_sha256':sources,'library':config['instrumented'],'baseline_binding':binding,'verified_assets':assets},'Integrity content')
    schedule=[(i,a) for i in range(80) for a in ARMS[i%2:]+ARMS[:i%2]]
    need([(r['pair_index'],r['arm']) for r in seal['rows']]==schedule,'160 replay order')
    byte_count=0;float_count=bitwise=0;cases=[];total_counts=Counter();all_nonfinite=Counter();badcases=[];outer=native=0.;warnings=[]
    for row in seal['rows']:
        item=(row['pair_index'],row['arm']);ident=manifest['selected_pairs'][item[0]]
        paths={key:safe(folder,row[key]) for key in ('path','raw_path','process_path')}
        for key,sha in (('path','sha256'),('raw_path','raw_sha256'),('process_path','process_sha256')):
            need(digest(paths[key])==row[sha],'Replay file changed');hashes[str(paths[key])]=row[sha]
        final,raw,process=load(paths['path']),load(paths['raw_path']),load(paths['process_path'])
        need(raw['status']=='completed' and row['status']=='completed' and process['exit_code']==0 and process['termination'] is None,'Replay process failure')
        need(process['raw_worker_path']==row['raw_path'] and process['raw_worker_sha256']==row['raw_sha256'],'Process/raw linkage')
        log=safe(folder,process['log_path']);need(digest(log)==process['log_sha256'],'Replay log hash');hashes[str(log)]=process['log_sha256']
        if log.read_text().strip():warnings.append({'pair_index':item[0],'arm':item[1],'log_path':log.name,'nonempty':True})
        for k in ('pair_index','scene','key','filename','dataset_bytes'):need(raw[k]==ident[k],'Replay identity '+k)
        need(raw['arm']==item[1] and raw['manifest_sha256']==mh and raw['source_sha256']==sources and raw['library']==config['instrumented'],'Worker bindings')
        need(raw['baseline_seal_sha256']==bh and raw['baseline_prediction_sha256']==baseline_hashes[f'{item[0]}:{item[1]}'],'Worker baseline binding')
        need(raw['dataset_sha256_declared']==raw['dataset_sha256_verified']==ident['dataset_sha256'],'Worker asset hash declaration')
        old=original[item]
        for k in ('input_arrays','calibration','input_diagnostics','options','native_inliers_packbits_big_hex','mask_sha256','mask_length','kept_count'):
            need(raw[k]==old[k],'Exact raw baseline parity '+k)
        floats=[]
        for k in ('estimate','geometry','returned_ransac_stats'):raw_equal(raw[k],old[k],k,floats)
        equal=sum(struct.pack('!d',a)==struct.pack('!d',b) for a,b in floats);float_count+=len(floats);bitwise+=equal
        parity={'passed':True,'float_relative_max_tolerance':1e-12,'discrete_and_mask_exact':True,'raw_translation_gauge_preserved':True,
            'floating_components':len(floats),'bitwise_equal_float_components':equal,'all_float_components_bitwise_equal':equal==len(floats),
            'max_normalized_float_error':max((abs(a-b)/max(1.,abs(a),abs(b)) for a,b in floats),default=0.)}
        need(raw['parity']==parity,'Parity receipt')
        timings={**raw['timings'],'charged_total_seconds':process['wall_seconds']}
        need(final=={**raw,'timings':timings,'process_receipt':process},'Final normalization')
        need(all(type(v) in (int,float) and v>=0 for v in timings.values()),'Timings')
        need(math.fsum(timings[k] for k in ('binding_seconds','input_seconds','call_seconds','score_seconds','trace_save_seconds'))<=timings['worker_total_seconds']+1e-6<=timings['charged_total_seconds']+2e-6,'Timing containment')
        outer+=timings['charged_total_seconds'];native+=timings['call_seconds']
        tp=safe(folder,raw['trace_path']);need(digest(tp)==raw['trace_sha256']==row['trace_sha256'],'Trace hash');hashes[str(tp)]=raw['trace_sha256']
        need(tp.stat().st_size==raw['trace_bytes']==row['trace_bytes']==raw['trace_required_bytes'],'Trace bytes')
        need(raw['trace_byte_budget']==min(64*1024*1024,512*1024*1024-byte_count) and raw['trace_bytes']<=raw['trace_byte_budget'],'Trace byte cap')
        byte_count+=raw['trace_bytes'];env=load(tp)
        expected={k:raw[k] for k in ('pair_index','scene','key','arm','source_sha256','manifest_sha256','library','baseline_seal_sha256','baseline_prediction_sha256','input_arrays','calibration','dataset_sha256_verified')}
        need(env['schema']=='evr0018.trace.v1' and env['bindings']==expected,'Trace envelope')
        tr=env['native'];need(tr['schema']=='evr0018.native-trace.v1' and tr['complete'] is True and tr['dropped_events']==0 and tr['event_count']==len(tr['events'])==raw['trace_event_count'] and tr['capacity']==50000,'Native trace coverage')
        found=required_nonfinites(tr);case={**{k:raw[k] for k in ('pair_index','scene','key','arm')},**found}
        cases.append(case);total_counts.update(found['counts']);all_nonfinite.update(found['all_event_nonfinite_state_counts'])
        if found['nonfinite_required_states']:badcases.append(case)
        returned=[e for e in tr['events'] if e['kind']=='pipeline' and e['phase']=='returned'][0]
        need(not nonfinite_paths(returned['state']),'Nonfinite final state')
        for k in ('R','t','scale','shift1','shift2'):raw_equal(returned['state'][k],raw['geometry'][k])
    need(byte_count==seal['trace_bytes'],'Cohort bytes')
    diag=result_dir/'diagnosis';operation=load(diag/'operation.json');hashes[str(diag/'operation.json')]=digest(diag/'operation.json')
    need(operation['status']=='ineligible' and operation['failed_phase']=='reconstruction','Frozen failure phase')
    need(not (diag/'evaluation.json').exists() and not (diag/'reconstruction-seal.json').exists() and not list(diag.glob('*.labels.json')),'Unexpected label/evaluation output')
    partial=sorted(diag.glob('*.reconstruction.json'));need(len(partial)==operation['completed_reconstructions']==88,'Partial reconstruction coverage')
    identities=[]
    for path in partial:
        r=load(path);identities.append((r['pair_index'],r['arm']));hashes[str(path)]=digest(path)
    need(set(identities)=={(i,a) for i in range(44) for a in ARMS},'Successful prefix differs')
    need(any(c['pair_index']==44 and c['arm']=='depth_scale' for c in badcases),'Failure case not independently found')
    for path,sha in hashes.items():need(digest(path)==sha,'Saved artifact changed during review')
    for path,sha in sources.items():need(digest(repo/path)==sha,'Frozen source changed during review')
    return {'schema':'evr0018.failure-review.v1','status':'passed','scientific_diagnostic_status':'ineligible',
        'review_source_sha256':digest(__file__),'review_helper_sha256':digest(HELPER),
        'bindings':{'manifest_sha256':mh,'baseline_seal_sha256':bh,'trace_seal_sha256':digest(sp),'source_sha256':sources},
        'pair_count':80,'replay_count':160,'all_raw_state_stat_mask_parity_passed':True,'floating_components':float_count,'bitwise_equal_components':bitwise,
        'trace_bytes':byte_count,'required_state_counts':dict(total_counts),'all_event_nonfinite_state_counts':dict(all_nonfinite),
        'nonfinite_required_case_count':len(badcases),'nonfinite_required_cases':badcases,
        'nonfinite_required_on_final_ancestry':sum(e['on_final_ancestry'] for c in badcases for e in c['nonfinite_required_states']),
        'partial_reconstruction_count':88,'diagnosis_receipt':operation,'label_artifacts_present':False,'worker_nonempty_logs':warnings,
        'timings':{'replay_outer_process_sum_seconds':outer,'instrumented_native_sum_seconds':native},
        'scope':'Saved JSON/source/log hashes and graph-only ancestry; no private HDF5, baseline evaluation/labels, scientific module or native call.',
        'conclusion':'The frozen complete-state barrier failed. Final-output parity does not make invalid initial states admissible. No label/gate result or method gain is produced.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','manifest','baseline-seal','result-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report=audit(a.repo,a.manifest,a.baseline_seal,a.result_dir)
    with a.output.open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','replay_count','nonfinite_required_case_count','nonfinite_required_on_final_ancestry','required_state_counts')}))


if __name__=='__main__':main()
