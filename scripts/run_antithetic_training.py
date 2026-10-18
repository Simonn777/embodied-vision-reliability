#!/usr/bin/env python3
"""0008 cache-only one-shot training supervisor; no scientific call at import."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FREEZE='protocols/0008-training-freeze.json'
REQUIRED=(
 'configs/0008-training.json','protocols/0008-antithetic-training.md',
 'scripts/antithetic_training_core.py','scripts/antithetic_training_inference.py',
 'scripts/antithetic_training_metrics.py','scripts/run_antithetic_training.py',
 'scripts/check_antithetic_training.py',
 'scripts/run_antithetic_gradient_screen.py','scripts/conditional_residual_metrics.py',
 'tests/test_antithetic_training_core.py','tests/test_antithetic_training_inference.py',
 'tests/test_antithetic_training_metrics.py','tests/test_antithetic_training_supervision.py',
)
METHODS=('independent','random_antithetic','mean_antithetic','same_patch')
ENDPOINTS=('updates500','wall5s')
# Reuse only already tested process/resource/I/O primitives; no 0007 run is called.
sys.path.insert(0,str(ROOT/'scripts'))
import run_antithetic_gradient_screen as control
read,write,digest,utc=control.read,control.write,control.digest,control.utc


def verify_freeze(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise RuntimeError('full canonical freeze commit required')
    f=read(ROOT/FREEZE)
    if subprocess.check_output(['git','show',commit+':'+FREEZE],cwd=ROOT)!=(ROOT/FREEZE).read_bytes():raise RuntimeError('canonical freeze differs')
    if set(f['files'])!=set(REQUIRED) or not re.fullmatch('[0-9a-f]{40}',f['source_commit']):raise RuntimeError('unexpected freeze inputs')
    for name,h in f['files'].items():
        if digest(ROOT/name)!=h or hashlib.sha256(subprocess.check_output(['git','show',f['source_commit']+':'+name],cwd=ROOT)).hexdigest()!=h:raise RuntimeError('source differs: '+name)
    return f


def bound_module(name,frozen):
    relative='scripts/'+name+'.py';path=ROOT/relative
    if relative not in REQUIRED or digest(path)!=frozen['files'][relative]:raise RuntimeError('module source is not frozen')
    # Explicit file loading defeats same-name files in the external runtime root.
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=path.resolve():raise RuntimeError('module path differs')
    return module


def expected_command(args,item,method):
    return [sys.executable,str(Path(__file__).resolve()),'case','--cache',str(args.cache),'--output',str(args.output),'--runtime-root',str(args.runtime_root),'--freeze-commit',args.freeze_commit,'--case-index',str(item['index']),'--method',method]


def array_hash(array):
    header=json.dumps({'shape':list(array.shape),'dtype':array.dtype.str},sort_keys=True,separators=(',',':')).encode('ascii')
    return hashlib.sha256(header+b'\0'+array.tobytes(order='C')).hexdigest()


def saved_state_hash(np,path):
    expected={'0.weight':(64,49),'0.bias':(64,),'2.weight':(32,64),'2.bias':(32,),'4.weight':(1,32),'4.bias':(1,)}
    h=hashlib.sha256(b'evr-mlp-state-v1\0')
    with np.load(path,allow_pickle=False) as state:
        if set(state.files)!=set(expected):raise RuntimeError('saved checkpoint tensor names differ')
        for name in sorted(expected):
            a=state[name]
            if a.shape!=expected[name] or a.dtype!=np.float32 or not np.isfinite(a).all():raise RuntimeError('invalid saved checkpoint')
            header=json.dumps({'name':name,'shape':list(a.shape),'dtype':'<f4'},sort_keys=True,separators=(',',':')).encode('ascii')
            payload=a.astype('<f4',copy=False).tobytes(order='C')
            h.update(len(header).to_bytes(8,'little'));h.update(header)
            h.update(len(payload).to_bytes(8,'little'));h.update(payload)
    return h.hexdigest()


def cached_file(cache,relative,expected):
    path=cache/relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(cache.resolve()):raise RuntimeError('unsafe or missing cached file')
    if digest(path)!=expected:raise RuntimeError('cached hash differs: '+relative)
    return path


def check_inputs(cache,cfg):
    return [{'relative':c['input_relative'],'sha256':digest(cached_file(cache,c['input_relative'],c['input_sha256']))} for c in cfg['cases']]


def numpy_runtime(root):
    sys.path.insert(0,str(root.resolve()))
    import numpy as np
    if np.__version__!='2.0.2' or not Path(np.__file__).resolve().is_relative_to(root.resolve()):raise RuntimeError('cached NumPy differs')
    return np


def case_name(item,method):return item['name']+'-'+method


def operation_order(cfg):
    return [(c,METHODS[(c['index']+j)%4]) for c in cfg['cases'] for j in range(4)]


def check_start(args,frozen):
    s=read(args.output/'study.started.json')
    if s['freeze_commit']!=args.freeze_commit or s['source_commit']!=frozen['source_commit'] or s['freeze_sha256']!=digest(ROOT/FREEZE):raise RuntimeError('mismatched study start')
    return s


def run_case(args,cfg,frozen):
    whole=time.monotonic();affinity=control.set_limits()
    check_start(args,frozen)
    np,env=control.runtime(args.runtime_root)
    core=bound_module('antithetic_training_core',frozen)
    inference=bound_module('antithetic_training_inference',frozen)
    item=cfg['cases'][args.case_index];method=args.method
    folder=args.output/'cases'/case_name(item,method);folder.mkdir()
    write(folder/'started.json',{'utc':utc(),'case':item,'method':method,'freeze_commit':args.freeze_commit,'source_commit':frozen['source_commit']})
    common=time.monotonic()
    inp=cached_file(args.cache,item['input_relative'],item['input_sha256'])
    with np.load(inp,allow_pickle=False) as data:
        if set(data.files)!={'noisy','covariance'}:raise RuntimeError('input members differ')
        noisy,covariance=data['noisy'],data['covariance']
    input_seconds=time.monotonic()-common
    trained=core.fit_trajectory(noisy,covariance,method,case_index=item['index'])
    record={'schema':'0008-trajectory-v1','status':trained['status'],'identity':item,'method':method,
        'source_commit':frozen['source_commit'],'freeze_commit':args.freeze_commit,'freeze_sha256':digest(ROOT/FREEZE),
        'source_hashes':{k:v for k,v in frozen['files'].items() if k.startswith('scripts/')},
        'environment':env,'cpu_affinity':affinity,'input_seconds':input_seconds,'fit':trained['receipt'],'endpoints':{},'files':{}}
    # Persist available locked/failure states before inference; no retry on failure.
    for endpoint,data in trained['endpoints'].items():
        began=time.monotonic();path=folder/(endpoint+'-weights.npz')
        np.savez(path,**data['state_dict_np'])
        record['files'][path.name]=digest(path)
        record['endpoints'][endpoint]={'training':data['receipt'],'weights_write_seconds':time.monotonic()-began}
    if trained['status']!='completed':
        record['pipeline_wall_seconds']=time.monotonic()-whole
        write(folder/'receipt.json',record)
        raise RuntimeError('failed trajectory; no inference or retry')
    if set(trained['endpoints'])!=set(ENDPOINTS):raise RuntimeError('missing endpoint')
    for endpoint in ENDPOINTS:
        began=time.monotonic();path=folder/(endpoint+'-weights.npz')
        with np.load(path,allow_pickle=False) as stored:state={k:stored[k] for k in stored.files}
        if core.state_dict_sha256(state)!=trained['endpoints'][endpoint]['receipt']['state_dict_sha256']:raise RuntimeError('checkpoint changed on disk')
        predicted=inference.predict_endpoint(noisy,covariance,state,case_index=item['index'])
        pred_path=folder/(endpoint+'-prediction.npy')
        if predicted['status']!='completed':
            record['status']='failed';record['inference_failure']=predicted['receipt'];record['pipeline_wall_seconds']=time.monotonic()-whole
            write(folder/'receipt.json',record)
            raise RuntimeError('inference failed; preserve without retry')
        np.save(pred_path,predicted['prediction'],allow_pickle=False)
        record['files'][pred_path.name]=digest(pred_path)
        row=record['endpoints'][endpoint]
        row['inference']=predicted['receipt'];row['inference_source_sha256']=frozen['files']['scripts/antithetic_training_inference.py']
        row['restore_infer_save_seconds']=time.monotonic()-began
        row['algorithm_pipeline_seconds']=input_seconds+trained['receipt']['shared_setup_seconds']+row['training']['training_elapsed_seconds']+row['weights_write_seconds']+row['restore_infer_save_seconds']
    if digest(inp)!=item['input_sha256']:raise RuntimeError('input modified')
    record['pipeline_wall_seconds']=time.monotonic()-whole
    write(folder/'receipt.json',record)
    print(json.dumps({'case':case_name(item,method),'status':record['status'],'receipt_sha256':digest(folder/'receipt.json')}),flush=True)


def run_study(args,cfg,frozen):
    control.set_limits();began=time.monotonic()
    inputs=check_inputs(args.cache,cfg)
    prior=ROOT/'results/0007-gradient-screen/study/scores.json'
    if digest(prior)!=cfg['prior_screen_score_sha256'] or read(prior)['decision']!='pass_local_mechanism_screen_only':raise RuntimeError('prior mechanism gate not passed')
    args.output.mkdir()
    write(args.output/'study.started.json',{'utc':utc(),'source_commit':frozen['source_commit'],'freeze_commit':args.freeze_commit,'freeze_sha256':digest(ROOT/FREEZE),'inputs':inputs,'order':[case_name(c,m) for c,m in operation_order(cfg)]})
    (args.output/'cases').mkdir();(args.output/'control').mkdir();rows=[]
    for item,method in operation_order(cfg):
        remaining=900-(time.monotonic()-began)
        if remaining<=0:raise RuntimeError('study deadline')
        name=case_name(item,method)
        command=expected_command(args,item,method)
        row=control.supervise(command,args.output/'control'/name,min(120,remaining),receipt_path=args.output/'cases'/name/'receipt.json')
        rows.append({'case':name,**row})
        if row['actual_exit']!=0 or row['receipt_sha256'] is None:
            write(args.output/'study.failed.json',{'utc':utc(),'reason':'trajectory failed, incomplete or timed out','cases':rows})
            raise RuntimeError('study stopped; preserve all evidence, no retry or selected score')
    write(args.output/'study.completed.json',{'utc':utc(),'wall_seconds':time.monotonic()-began,'source_commit':frozen['source_commit'],'freeze_commit':args.freeze_commit,'cases':rows,'inputs_after':check_inputs(args.cache,cfg)})
    print(json.dumps({'status':'32 trajectories complete; 64 endpoint predictions locked','wall_seconds':time.monotonic()-began}),flush=True)


def allocated_outer_cost(ex,record,endpoint):
    """Full observed child wall minus only measured work needed by the other endpoint.

    Runtime loading, source verification, shared setup, I/O/checks and shutdown
    remain charged. This is an allocation of a shared trajectory, not a second run.
    """
    import math
    numeric=[ex['wall_seconds'],record['input_seconds'],record['fit']['shared_setup_seconds'],record['fit']['training_wall_seconds']]
    numeric += [v[key] for v in record['endpoints'].values() for key in ('weights_write_seconds','restore_infer_save_seconds','algorithm_pipeline_seconds')]
    numeric += [v['training'][key] for v in record['endpoints'].values() for key in ('update_completion_elapsed_seconds','training_elapsed_seconds')]
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in numeric):raise RuntimeError('invalid timing component')
    if any(not v['training']['update_completion_elapsed_seconds']<=v['training']['training_elapsed_seconds']<=record['fit']['training_wall_seconds'] for v in record['endpoints'].values()):raise RuntimeError('invalid nested endpoint clocks')
    current=record['endpoints'][endpoint]
    extra_training=record['fit']['training_wall_seconds']-current['training']['training_elapsed_seconds']
    if not math.isfinite(extra_training) or extra_training<0:raise RuntimeError('negative extra training time')
    other_io=sum(v['weights_write_seconds']+v['restore_infer_save_seconds'] for k,v in record['endpoints'].items() if k!=endpoint)
    value=ex['wall_seconds']-extra_training-other_io
    if not math.isfinite(value) or value<current['algorithm_pipeline_seconds']:raise RuntimeError('inconsistent outer cost allocation')
    return value


def validate_evidence(args,cfg,frozen,np):
    start=check_start(args,frozen);complete=read(args.output/'study.completed.json')
    expected=operation_order(cfg);names=[case_name(c,m) for c,m in expected]
    if start['order']!=names or [x['case'] for x in complete['cases']]!=names or complete['source_commit']!=frozen['source_commit'] or complete['freeze_commit']!=args.freeze_commit or start['inputs']!=complete['inputs_after'] or check_inputs(args.cache,cfg)!=start['inputs']:raise RuntimeError('study manifest differs')
    records=[];initial={};fixed_streams={}
    for (item,method),summary in zip(expected,complete['cases']):
        name=case_name(item,method);folder=args.output/'cases'/name;prefix=args.output/'control'/name
        ex=read(prefix.with_suffix('.execution.json'));r=read(folder/'receipt.json')
        if summary!={'case':name,**ex} or ex['actual_exit']!=0 or ex['raw_wait_status']!=0 or ex['hard_timeout'] or ex['receipt_sha256']!=digest(folder/'receipt.json'):raise RuntimeError('unsuccessful or unbound execution')
        for suffix,key in [('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256'),('.command.private.json','command_file_sha256')]:
            if digest(prefix.with_suffix(suffix))!=ex[key]:raise RuntimeError('command/log changed')
        command=read(prefix.with_suffix('.command.private.json'))
        if command!={'argv':expected_command(args,item,method),'cwd':str(ROOT)}:raise RuntimeError('actual argv/cwd differ from frozen recipe')
        if r['status']!='completed' or r['identity']!=item or r['method']!=method or r['source_commit']!=frozen['source_commit'] or r['freeze_commit']!=args.freeze_commit or r['freeze_sha256']!=digest(ROOT/FREEZE):raise RuntimeError('trajectory binding differs')
        if r['source_hashes']!={k:v for k,v in frozen['files'].items() if k.startswith('scripts/')}:raise RuntimeError('source bindings differ')
        fit=r['fit']
        if fit['status']!='completed' or fit['failure'] is not None or fit['parameter_count']!=5313 or fit['initialization_seed']!=620000+100*(item['index']//2)+10*(item['index']%2)+1 or fit['clean_reference_accessed'] or fit['inference_calls']!=0 or fit['numeric_threads']!=4 or fit['interop_threads']!=1:raise RuntimeError('fit configuration mismatch')
        if fit['architecture']!=[49,64,32,1] or fit['batch_size']!=512 or fit['alpha']!=.5 or fit['requested_updates']!=500 or fit['wall_target_seconds']!=5 or fit['allowed_wall_overrun_seconds']!=.05 or fit['torch_version']!='2.10.0+cpu' or fit['numpy_version']!='2.0.2' or not fit['deterministic_algorithms']:raise RuntimeError('training settings/runtime differ')
        if fit['optimizer']!={'name':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'amsgrad':False,'foreach':False,'initial_state_entries':0}:raise RuntimeError('optimizer differs')
        if fit['completed_updates']!=fit['optimizer_steps_executed'] or not 500<=fit['completed_updates']<=100000 or fit['completed_updates']!=fit['drawn_batches']:raise RuntimeError('update accounting differs')
        if set(r['endpoints'])!=set(ENDPOINTS) or set(r['files'])!={e+s for e in ENDPOINTS for s in ('-weights.npz','-prediction.npy')}:raise RuntimeError('missing/unexpected outputs')
        for filename,h in r['files'].items():
            if digest(folder/filename)!=h:raise RuntimeError('output bytes changed')
        c=item['index'];initial.setdefault(c,fit['initial_weight_sha256'])
        if initial[c]!=fit['initial_weight_sha256']:raise RuntimeError('initialization differs')
        for endpoint,row in r['endpoints'].items():
            t=row['training'];inf=row['inference']
            if t!=fit['endpoint_receipts'][endpoint] or t['valid'] is not True or t['step']<=0 or (endpoint=='updates500' and t['step']!=500):raise RuntimeError('invalid endpoint')
            if endpoint=='wall5s' and not (5<=t['update_completion_elapsed_seconds']<=t['training_elapsed_seconds']<=5.05):raise RuntimeError('wall endpoint exceeds freeze')
            if endpoint=='updates500':
                hashes={k:t['stream_sha256'][k] for k in ('base_indices','gaussian_prefix','step_seed_tuple')}
                fixed_streams.setdefault(c,hashes)
                if fixed_streams[c]!=hashes:raise RuntimeError('500-step shared streams differ')
            pred=np.load(folder/(endpoint+'-prediction.npy'),allow_pickle=False)
            if pred.dtype!=np.float32 or pred.shape!=(122,122) or not np.isfinite(pred).all():raise RuntimeError('invalid prediction')
            if row['algorithm_pipeline_seconds'] < t['training_elapsed_seconds'] or not all(__import__('math').isfinite(row[k]) and row[k]>0 for k in ('algorithm_pipeline_seconds','restore_infer_save_seconds','weights_write_seconds')):raise RuntimeError('invalid timing')
            content=saved_state_hash(np,folder/(endpoint+'-weights.npz'))
            if content!=t['state_dict_sha256'] or any(inf[k]!=content for k in ('weight_content_sha256','weight_before_sha256','weight_after_sha256')):raise RuntimeError('checkpoint content binding differs')
            if (inf['status']!='completed' or inf['failure'] is not None or not inf['finite_outputs'] or not inf['weight_unchanged']
                or inf['forward_calls']!=750 or inf['forward_calls_attempted']!=750 or inf['standard_normal_draws']!=36465800
                or inf['mc_samples']!=50 or inf['chunk_size']!=1024 or inf['optimizer_steps']!=0 or inf['direct_forward_calls']!=0
                or inf['clipping']!='none' or inf['interpolation']!='none' or inf['accumulation_dtype']!='float64'
                or inf['alpha']!=.5 or inf['architecture']!=[49,64,32,1] or inf['output_shape']!=[122,122]
                or inf['prediction_sha256']!=array_hash(pred) or inf['clean_reference_accessed']
                or inf['case_index']!=c or inf['numeric_threads']!=4 or inf['interop_threads']!=1
                or inf['torch_version']!='2.10.0+cpu' or inf['numpy_version']!='2.0.2'
                or inf['inference_seed']!=int.from_bytes(hashlib.sha256(f'evr0008|{c}|inference|0'.encode('ascii')).digest()[:8],'little')&((1<<63)-1)):
                raise RuntimeError('inference identity or count mismatch')
            if any(inf[k]!=fit[k] for k in ('patches_sha256','covariance_sha256','factor_sha256')) or inf['noisy_sha256']!=fit['extra_setup']['noisy_sha256']:raise RuntimeError('fit/inference inputs differ')
            if row['inference_source_sha256']!=frozen['files']['scripts/antithetic_training_inference.py']:raise RuntimeError('inference source differs')
            expected_cost=r['input_seconds']+fit['shared_setup_seconds']+t['training_elapsed_seconds']+row['weights_write_seconds']+row['restore_infer_save_seconds']
            if not __import__('math').isclose(row['algorithm_pipeline_seconds'],expected_cost,rel_tol=1e-12,abs_tol=1e-12):raise RuntimeError('algorithm cost formula differs')
            row['allocated_outer_pipeline_seconds']=allocated_outer_cost(ex,r,endpoint)
            # Inference source supplies identity/counts; no new model call here.
            records.append((item,method,endpoint,r,row,folder))
    if len(records)!=64:raise RuntimeError('incomplete 64 endpoints')
    return records


def score_saved(args,cfg,frozen):
    control.set_limits();np=numpy_runtime(args.runtime_root)
    records=validate_evidence(args,cfg,frozen,np)
    write(args.output/'score.started.json',{'utc':utc(),'freeze_commit':args.freeze_commit,'locked_endpoint_count':len(records)})
    bound_module('conditional_residual_metrics',frozen)
    metrics=bound_module('antithetic_training_metrics',frozen)
    refs={};rows=[];bindings=[]
    # References are first opened only after every prediction has been validated.
    for item,method,endpoint,r,row,folder in records:
        if item['group'] not in refs:
            path=cached_file(args.cache,item['reference_relative'],item['reference_sha256'])
            with np.load(path,allow_pickle=False) as data:
                if set(data.files)!={'clean'}:raise RuntimeError('reference members differ')
                clean=data['clean']
            if clean.dtype!=np.float32 or clean.shape!=(128,128) or not np.isfinite(clean).all():raise RuntimeError('invalid reference')
            refs[item['group']]=clean[3:-3,3:-3]
        prediction=np.load(folder/(endpoint+'-prediction.npy'),allow_pickle=False)
        rows.append({'group':item['group'],'seed':item['seed'],'method':method,'endpoint':endpoint,'status':'completed','inference_mc_samples':50,'score':metrics.score_pair(refs[item['group']],prediction),'pipeline_seconds':row['allocated_outer_pipeline_seconds'],'training_seconds':row['training']['training_elapsed_seconds'],'updates':row['training']['step']})
        bindings.append({'case':case_name(item,method),'endpoint':endpoint,'receipt_sha256':digest(folder/'receipt.json'),'reference_sha256':item['reference_sha256'],'prediction_file_sha256':digest(folder/(endpoint+'-prediction.npy'))})
    result=metrics.aggregate_cases(rows)
    result['bindings']={'source_commit':frozen['source_commit'],'freeze_commit':args.freeze_commit,'freeze_sha256':digest(ROOT/FREEZE),'endpoints':bindings}
    write(args.output/'scores.json',result)
    print(json.dumps({'status':result['status'],'scores_sha256':digest(args.output/'scores.json')}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('verify','run','case','audit','score'))
    for name in ('cache','output','runtime-root'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--freeze-commit',required=True);ap.add_argument('--case-index',type=int,choices=range(8));ap.add_argument('--method',choices=METHODS)
    args=ap.parse_args();args.cache=args.cache.resolve();args.output=args.output.resolve();args.runtime_root=args.runtime_root.resolve()
    if args.output!=args.cache.parent/'antithetic-training-0008':ap.error('require unique sibling output antithetic-training-0008')
    frozen=verify_freeze(args.freeze_commit);cfg=read(ROOT/'configs/0008-training.json')
    if args.mode=='verify':print(json.dumps({'status':'verified','inputs':check_inputs(args.cache,cfg),'source_commit':frozen['source_commit']}));return
    if args.mode=='case' and (args.case_index is None or args.method is None):ap.error('case/method required')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,control.interrupted)
    signal.setitimer(signal.ITIMER_REAL,900 if args.mode=='run' else 120)
    try:
        if args.mode=='audit':
            control.set_limits();np=numpy_runtime(args.runtime_root)
            records=validate_evidence(args,cfg,frozen,np)
            print(json.dumps({'status':'PASS','trajectories':32,'endpoints':len(records),'references_opened':False}),flush=True)
        else:{'run':run_study,'case':run_case,'score':score_saved}[args.mode](args,cfg,frozen)
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
