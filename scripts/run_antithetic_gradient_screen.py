#!/usr/bin/env python3
"""One-shot, cache-only 0007 supervisor. No experiment at import."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FREEZE = 'protocols/0007-gradient-freeze.json'
REQUIRED = (
 'configs/0007-gradient-screen.json', 'protocols/0007-antithetic-gradient-screen.md',
 'scripts/antithetic_gradient_core.py', 'scripts/antithetic_gradient_metrics.py',
 'scripts/run_antithetic_gradient_screen.py',
 'tests/test_antithetic_gradient_core.py', 'tests/test_antithetic_gradient_metrics.py',
 'tests/test_antithetic_gradient_supervision.py',
)
THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
CHILD = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_freeze(commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise RuntimeError('full canonical freeze commit required')
    path = ROOT / FREEZE
    if subprocess.check_output(['git','show',commit+':'+FREEZE], cwd=ROOT) != path.read_bytes():
        raise RuntimeError('canonical freeze differs')
    frozen = read(path)
    if set(frozen['files']) != set(REQUIRED) or not re.fullmatch('[0-9a-f]{40}', frozen['source_commit']):
        raise RuntimeError('unrecognized freeze input set')
    for name, sha in frozen['files'].items():
        blob = subprocess.check_output(['git','show',frozen['source_commit']+':'+name], cwd=ROOT)
        if digest(ROOT/name) != sha or hashlib.sha256(blob).hexdigest() != sha:
            raise RuntimeError('source differs: '+name)
    return frozen


def module(name):
    relative = 'scripts/'+name+'.py'
    frozen = read(ROOT/FREEZE)
    if relative not in REQUIRED or digest(ROOT/relative) != frozen['files'][relative]:
        raise RuntimeError('module is not bound')
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def check_inputs(cache, cfg):
    """Only the 24 exact noisy/weight/identity files; no predictions or references."""
    checked = []
    for item in cfg['cases']:
        for kind in ('input','weights','receipt'):
            rel, expected = item[kind+'_relative'], item[kind+'_sha256']
            path = cache/rel
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
                raise RuntimeError('missing or indirect cache file: '+rel)
            actual = digest(path)
            if actual != expected:
                raise RuntimeError('cache identity differs: '+rel)
            checked.append({'relative':rel,'sha256':actual,'bytes':path.stat().st_size})
    return checked


def set_limits():
    if sys.platform != 'linux':
        raise RuntimeError('production execution requires existing Linux CPU runtime')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError('four CPU affinity slots required')
    os.sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3,8*1024**3))
    for key in THREAD_ENV:
        os.environ[key] = '4'
    return cpus


def runtime(root):
    sys.path.insert(0, str(root.resolve()))
    import numpy as np
    import torch
    for mod, version in ((np,'2.0.2'),(torch,'2.10.0+cpu')):
        if mod.__version__ != version or not Path(mod.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError('cached runtime mismatch: '+mod.__name__)
    if torch.version.cuda is not None:
        raise RuntimeError('CPU-only runtime required')
    return np, {'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
                'module_entry_sha256':{m.__name__:digest(m.__file__) for m in (np,torch)}}


def case(args, cfg, frozen):
    cpus = set_limits()
    start_record=read(args.output/'study.started.json')
    if (start_record['freeze_commit'] != args.freeze_commit
            or start_record['source_commit'] != frozen['source_commit']
            or start_record['freeze_sha256'] != digest(ROOT/FREEZE)):
        raise RuntimeError('case requires matching started study')
    np, env = runtime(args.runtime_root)
    core = module('antithetic_gradient_core')
    item = cfg['cases'][args.case_index]
    check_inputs(args.cache, {'cases':[item]})
    out = args.output/'cases'/item['name']; out.mkdir()
    write(out/'started.json', {'utc':utc(),'case_index':item['index'],'source_commit':frozen['source_commit']})
    with np.load(args.cache/item['input_relative'], allow_pickle=False) as f:
        if set(f.files) != {'noisy','covariance'}:
            raise RuntimeError('unexpected input members')
        noisy, covariance = f['noisy'], f['covariance']
    with np.load(args.cache/item['weights_relative'], allow_pickle=False) as f:
        weights = {k:f[k] for k in f.files}
    if core.state_dict_sha256(weights) != item['weight_content_sha256']:
        raise RuntimeError('weight content differs from original fit')
    result = core.run_case(noisy, covariance, weights, case_index=item['index'])
    for key in ('gradients','walls','losses','completed'):
        np.save(out/(key+'.npy'), result[key], allow_pickle=False)
    receipt = result['receipt']
    old_receipt = read(args.cache/item['receipt_relative'])
    if any(receipt[key] != old_receipt[key] for key in ('covariance_sha256','factor_sha256')):
        receipt['status'] = 'failed'
        receipt['failure'] = {'type':'InputIdentity','message':'covariance/factor differ from original fit'}
    receipt.update({'case_identity':item,'freeze_commit':args.freeze_commit,'freeze_sha256':digest(ROOT/FREEZE),
                    'source_commit':frozen['source_commit'],'environment':env,'cpu_affinity':cpus,
                    'input_files_unchanged':check_inputs(args.cache, {'cases':[item]}),
                    'files':{k+'.npy':digest(out/(k+'.npy')) for k in ('gradients','walls','losses','completed')}})
    write(out/'receipt.json', receipt)
    print(json.dumps({'case':item['name'],'receipt_sha256':digest(out/'receipt.json'),'completed':bool(result['completed'].all())}),flush=True)
    if (not result['completed'].all() or receipt['status'] != 'completed'
            or not receipt['weight_unchanged']
            or receipt['weight_before_sha256'] != item['weight_content_sha256']):
        raise RuntimeError('failed gradient case; preserve, do not rerun')


def supervise(command, out, hard_seconds, env=None, receipt_path=None):
    global CHILD
    started = time.monotonic(); start_utc = utc(); timed_out = False
    write(out.with_suffix('.command.private.json'), {'argv':command,'cwd':str(Path.cwd())})
    with out.with_suffix('.stdout.log').open('xb') as stdout, out.with_suffix('.stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        CHILD = child.pid
        try:
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    break
                if time.monotonic()-started >= hard_seconds:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    _, status, usage = os.wait4(child.pid, 0)
                    break
                time.sleep(.05)
            child.returncode = os.waitstatus_to_exitcode(status)
        finally:
            CHILD = None
        stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    record = {'utc_started':start_utc,'utc_completed':utc(),'wall_seconds':time.monotonic()-started,
              'actual_exit':child.returncode,'raw_wait_status':status,'hard_timeout':timed_out,
              'hard_seconds':hard_seconds,'peak_rss_kib':usage.ru_maxrss,
              'stdout_sha256':digest(out.with_suffix('.stdout.log')),'stderr_sha256':digest(out.with_suffix('.stderr.log')),
              'command_file_sha256':digest(out.with_suffix('.command.private.json')),
              'receipt_sha256':digest(receipt_path) if receipt_path is not None and receipt_path.is_file() else None}
    write(out.with_suffix('.execution.json'), record)
    return record


def interrupted(sig, frame):
    if CHILD is not None:
        try:
            os.killpg(CHILD, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise SystemExit(128+sig)


def study(args, cfg, frozen):
    set_limits()
    started = time.monotonic()
    checked = check_inputs(args.cache, cfg)
    # Directory existence is the persistent one-shot marker; never replace it.
    args.output.mkdir()
    write(args.output/'study.started.json',{'utc':utc(),'freeze_commit':args.freeze_commit,
        'freeze_sha256':digest(ROOT/FREEZE),'source_commit':frozen['source_commit'],'inputs':checked})
    (args.output/'cases').mkdir(); (args.output/'control').mkdir()
    rows = []
    for item in cfg['cases']:
        remaining = 900-(time.monotonic()-started)
        if remaining <= 0:
            raise RuntimeError('study hard budget exhausted')
        command = [sys.executable,str(Path(__file__).resolve()),'case','--cache',str(args.cache),
                   '--output',str(args.output),'--runtime-root',str(args.runtime_root),
                   '--freeze-commit',args.freeze_commit,'--case-index',str(item['index'])]
        row = supervise(command,args.output/'control'/item['name'],min(120,remaining),
                        receipt_path=args.output/'cases'/item['name']/'receipt.json')
        rows.append({'case':item['name'],**row})
        if row['actual_exit'] != 0 or row['receipt_sha256'] is None:
            write(args.output/'study.failed.json',{'utc':utc(),'cases':rows,'reason':'case failed or timed out'})
            raise RuntimeError('case failed; no retry or selected-case analysis')
    write(args.output/'study.completed.json',{'utc':utc(),'wall_seconds':time.monotonic()-started,
        'cases':rows,'inputs_after':check_inputs(args.cache,cfg),'source_commit':frozen['source_commit']})
    print(json.dumps({'status':'all eight cases complete','elapsed_seconds':time.monotonic()-started}),flush=True)


def validate_case_evidence(output,item,frozen,freeze_commit,study_row):
    out=output/'cases'/item['name']; receipt=read(out/'receipt.json')
    control=output/'control'/item['name']
    execution=read(control.with_suffix('.execution.json'))
    if ({'case':item['name'],**execution} != study_row
            or execution['actual_exit'] != 0 or execution['raw_wait_status'] != 0
            or execution['hard_timeout'] or execution['receipt_sha256'] != digest(out/'receipt.json')
            or execution['stdout_sha256'] != digest(control.with_suffix('.stdout.log'))
            or execution['stderr_sha256'] != digest(control.with_suffix('.stderr.log'))
            or execution['command_file_sha256'] != digest(control.with_suffix('.command.private.json'))):
        raise RuntimeError('execution, logs or receipt binding differs')
    if (receipt['completed_measurements'] != 512 or receipt['expected_measurements'] != 512
            or receipt['optimizer_steps'] != 0 or receipt['parameter_count'] != 5313
            or receipt['source_sha256'] != frozen['files']['scripts/antithetic_gradient_core.py']
            or receipt['freeze_sha256'] != digest(ROOT/FREEZE)
            or receipt['status'] != 'completed' or not receipt['weight_unchanged']
            or receipt['weight_before_sha256'] != item['weight_content_sha256']
            or receipt['weight_after_sha256'] != item['weight_content_sha256']):
        raise RuntimeError('case integrity failed')
    if set(receipt['files']) != {'gradients.npy','walls.npy','losses.npy','completed.npy'}:
        raise RuntimeError('unexpected case file manifest')
    if receipt['case_identity'] != item or receipt['source_commit'] != frozen['source_commit'] or receipt['freeze_commit'] != freeze_commit:
        raise RuntimeError('case binding mismatch')
    for name, sha in receipt['files'].items():
        if digest(out/name) != sha:
            raise RuntimeError('saved gradient artifact differs')
    return receipt


def score(args,cfg,frozen):
    set_limits(); np, env = runtime(args.runtime_root)
    if not (args.output/'study.completed.json').is_file():
        raise RuntimeError('complete study required')
    write(args.output/'score.started.json',{'utc':utc(),'freeze_commit':args.freeze_commit})
    study_record=read(args.output/'study.completed.json')
    start_record=read(args.output/'study.started.json')
    if (study_record['source_commit'] != frozen['source_commit']
            or start_record['freeze_commit'] != args.freeze_commit
            or start_record['freeze_sha256'] != digest(ROOT/FREEZE)
            or [r['case'] for r in study_record['cases']] != [i['name'] for i in cfg['cases']]
            or start_record['inputs'] != study_record['inputs_after']):
        raise RuntimeError('study identity or input preservation mismatch')
    gradients=[]; walls=[]; bindings=[]
    for item in cfg['cases']:
        out=args.output/'cases'/item['name']
        receipt=validate_case_evidence(args.output,item,frozen,args.freeze_commit,study_record['cases'][item['index']])
        completed=np.load(out/'completed.npy',allow_pickle=False)
        if completed.shape != (2,4,64) or completed.dtype != np.dtype('bool') or not completed.all():
            raise RuntimeError('incomplete case')
        gradients.append(np.load(out/'gradients.npy',allow_pickle=False)); walls.append(np.load(out/'walls.npy',allow_pickle=False))
        bindings.append({'case':item['name'],'receipt_sha256':digest(out/'receipt.json')})
    metrics=module('antithetic_gradient_metrics')
    result=metrics.evaluate_gradients(np.stack(gradients),np.stack(walls))
    result['bindings']={'cases':bindings,'freeze_commit':args.freeze_commit,'source_commit':frozen['source_commit'],'environment':env}
    write(args.output/'scores.json',result)
    print(json.dumps({'status':'scored saved gradients','scores_sha256':digest(args.output/'scores.json')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('verify','run','case','score'))
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True);parser.add_argument('--freeze-commit',required=True)
    parser.add_argument('--case-index',type=int,choices=range(8))
    args=parser.parse_args()
    args.cache=args.cache.resolve();args.output=args.output.resolve();args.runtime_root=args.runtime_root.resolve()
    if args.output != args.cache.parent/'antithetic-gradient-0007':
        parser.error('require unique frozen sibling output: antithetic-gradient-0007')
    frozen=verify_freeze(args.freeze_commit);cfg=read(ROOT/'configs/0007-gradient-screen.json')
    if args.mode=='verify':
        print(json.dumps({'status':'verified','files':check_inputs(args.cache,cfg),'source_commit':frozen['source_commit']}));return
    if args.mode=='case' and args.case_index is None:
        parser.error('--case-index required')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    # The supervisor kills its active child before exiting at the overall cap.
    signal.setitimer(signal.ITIMER_REAL, {'run':900,'case':120,'score':120}[args.mode])
    try:
        {'run':study,'case':case,'score':score}[args.mode](args,cfg,frozen)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__=='__main__':
    main()
