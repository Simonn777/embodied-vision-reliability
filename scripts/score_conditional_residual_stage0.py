#!/usr/bin/env python3
"""Independent saved-array scorer. Never imports or calls the training module."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import re
import time
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="4"
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FROZEN_PATHS = ('configs/0006-stage0.json', 'configs/0006-development-images.json', 'protocols/0006-conditional-residual-stage0.md', 'scripts/conditional_residual.py', 'scripts/conditional_residual_training.py', 'scripts/conditional_residual_metrics.py', 'scripts/run_conditional_residual_stage0.py', 'scripts/score_conditional_residual_stage0.py', 'tests/test_conditional_residual.py', 'tests/test_conditional_residual_training.py', 'tests/test_conditional_residual_metrics.py', 'tests/test_conditional_residual_supervision.py', 'tests/test_conditional_residual_acquisition.py')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze-commit',required=True)
    args=parser.parse_args();started=time.monotonic()
    if re.fullmatch(r'[0-9a-f]{40}',args.freeze_commit) is None:raise RuntimeError('Exact freeze commit required')
    if subprocess.check_output(['git','show',args.freeze_commit+':protocols/0006-stage0-freeze.json'],cwd=ROOT)!=(ROOT/'protocols/0006-stage0-freeze.json').read_bytes():raise RuntimeError('Canonical receipt changed')
    freeze=read(ROOT/'protocols/0006-stage0-freeze.json')
    if set(freeze['files'])!=set(FROZEN_PATHS) or re.fullmatch(r'[0-9a-f]{40}',freeze['source_commit']) is None:raise RuntimeError('Incomplete freeze')
    for name,h in freeze['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Frozen file changed: '+name)
        if hashlib.sha256(subprocess.check_output(['git','show',freeze['source_commit']+':'+name],cwd=ROOT)).hexdigest()!=h:
            raise RuntimeError('Source commit mismatch')
    spec=importlib.util.spec_from_file_location('_evr0006_metrics',ROOT/'scripts/conditional_residual_metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    score_pair,aggregate_cases=metrics.score_pair,metrics.aggregate_cases
    if args.output.exists():raise RuntimeError('Refusing to overwrite a score result')
    cfg=read(ROOT/'configs/0006-stage0.json');data=read(ROOT/'configs/0006-development-images.json');prepared=read(args.cache/'prepared.json')['records']
    cases=[];secondary=[];measurements=[];bindings=[]
    for item in data['images']:
        ref=args.cache/'prepared'/(item['id']+'-reference.npz')
        reference_hashes={r['reference_sha256'] for r in prepared if r['image_id']==item['id']}
        if reference_hashes!={sha(ref)}:raise RuntimeError('Reference file changed')
        with np.load(ref,allow_pickle=False) as p:
            if set(p.files)!={'clean'}:raise RuntimeError('Unexpected reference schema')
            clean=p['clean']
        if clean.shape!=(128,128) or clean.dtype!=np.float32 or not np.isfinite(clean).all():raise RuntimeError('Bad reference')
        for condition in cfg['conditions']:
            for seed in cfg['seeds']:
                matched=[]
                for method in cfg['methods']:
                    name=f'{item["id"]}-{condition}-{seed}-{method}';folder=args.cache/'cases'/name
                    execution=read(args.cache/'cases'/(name+'.execution.json'))
                    if execution['exit_code']!=0 or execution['timed_out']:raise RuntimeError('Unsuccessful case: '+name)
                    receipt_path=folder/'receipt.json';receipt=read(receipt_path)
                    if sha(receipt_path)!=execution['receipt_sha256']:raise RuntimeError('Receipt changed')
                    if (receipt['image_id'],receipt['condition'],receipt['seed'],receipt['method'])!=(item['id'],condition,seed,method):raise RuntimeError('Case identity mismatch')
                    for file,h in receipt['files'].items():
                        if sha(folder/file)!=h:raise RuntimeError('Case output changed')
                    pred=np.load(folder/'prediction.npy',allow_pickle=False)
                    if pred.shape!=(122,122) or pred.dtype!=np.float32:raise RuntimeError('Bad prediction schema')
                    score=score_pair(clean[3:-3,3:-3],pred)
                    cases.append({'group':item['id'],'condition':condition,'seed':seed,'method':method,'score':score})
                    if (folder/'direct_prediction.npy').exists():
                        secondary.append({'group':item['id'],'condition':condition,'seed':seed,'method':method,'secondary_direct_only':True,'score':score_pair(clean[3:-3,3:-3],np.load(folder/'direct_prediction.npy',allow_pickle=False))})
                    bindings.append({'case':name,'execution_sha256':sha(args.cache/'cases'/(name+'.execution.json')),'receipt_sha256':sha(receipt_path),'reference_sha256':sha(ref),'prediction_file_sha256':sha(folder/'prediction.npy')})
                    measurements.append({'case':name,'outer_wall_seconds':execution['wall_seconds'],'peak_rss_kib':execution['peak_rss_kib'],'receipt':receipt})
                    matched.append(receipt)
                # Trainer receipts are checked separately for complete updates and paired initialization.
                # Keys are fixed by runner_core; no parameter/epoch choice uses the reference.
                initial={r['fit']['initial_weight_sha256'] for r in matched}
                batches={r['fit']['batch_indices_sha256'] for r in matched}
                if len(initial)!=1 or len(batches)!=1 or any(r['fit']['completed_steps']!=500 for r in matched):
                    raise RuntimeError('Initialization/index stream/update count not matched')
    result=aggregate_cases(cases,groups=cfg['image_ids'],seeds=cfg['seeds'])
    result.update({'source_commit':freeze['source_commit'],'freeze_sha256':sha(ROOT/'protocols/0006-stage0-freeze.json'),'secondary_direct_r2r':secondary,'case_bindings':bindings,'runtime_measurements':measurements,'scorer_wall_seconds':time.monotonic()-started,'numpy_version':np.__version__})
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','case_count','mean_colored_gain_db','group_mean_colored_gain_db','gates']},indent=2))


if __name__=='__main__':main()
