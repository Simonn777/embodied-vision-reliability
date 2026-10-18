#!/usr/bin/env python3
"""Frozen 0006 CPU development stages. No data or science is performed on import."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import re
import fcntl
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATHS = ('configs/0006-stage0.json', 'configs/0006-development-images.json', 'protocols/0006-conditional-residual-stage0.md', 'scripts/conditional_residual.py', 'scripts/conditional_residual_training.py', 'scripts/conditional_residual_metrics.py', 'scripts/run_conditional_residual_stage0.py', 'scripts/score_conditional_residual_stage0.py', 'tests/test_conditional_residual.py', 'tests/test_conditional_residual_training.py', 'tests/test_conditional_residual_metrics.py', 'tests/test_conditional_residual_supervision.py', 'tests/test_conditional_residual_acquisition.py')
CFG_PATH = ROOT / 'configs/0006-stage0.json'
DATA_PATH = ROOT / 'configs/0006-development-images.json'
CHILD_PID = None

def terminate_family(signum, frame):
    if CHILD_PID is not None:
        try:
            os.killpg(CHILD_PID, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)



def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_freeze(freeze_commit):
    if re.fullmatch(r'[0-9a-f]{40}', freeze_commit) is None:
        raise RuntimeError('Require explicit reviewed freeze commit SHA')
    freeze_path = ROOT / 'protocols/0006-stage0-freeze.json'
    committed = subprocess.check_output(['git','show',freeze_commit+':protocols/0006-stage0-freeze.json'],cwd=ROOT)
    if committed != freeze_path.read_bytes():
        raise RuntimeError('Canonical committed freeze receipt differs')
    frozen = read(freeze_path)
    if set(frozen['files']) != set(FROZEN_PATHS) or re.fullmatch(r'[0-9a-f]{40}',frozen['source_commit']) is None:
        raise RuntimeError('Incomplete or unrecognized freeze identity')
    for name, expected in frozen['files'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Frozen file changed: ' + name)
        blob = subprocess.check_output(['git', 'show', frozen['source_commit']+':'+name], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError('Frozen source Git bytes differ: ' + name)
    return frozen


def load_frozen_module(name, relative):
    frozen = read(ROOT / 'protocols/0006-stage0-freeze.json')
    path = ROOT / relative
    if relative not in FROZEN_PATHS or digest(path) != frozen['files'][relative]:
        raise RuntimeError('Unbound local scientific module')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_runtime(args):
    # Explicit cached origins; never pip/install/download as a fallback.
    runtime = args.runtime_root.resolve()
    overlay = args.overlay.resolve()
    sys.path[:0] = [str(runtime), str(overlay)]
    import numpy as np
    import PIL
    import torch
    for module, version, parent in ((np, '2.0.2', runtime), (torch, '2.10.0+cpu', runtime), (PIL, '13.0.0', overlay)):
        path = Path(module.__file__).resolve()
        if module.__version__ != version or not path.is_relative_to(parent):
            raise RuntimeError('Cached runtime identity mismatch: '+module.__name__)
    if torch.version.cuda is not None:
        raise RuntimeError('CPU-only PyTorch required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return {'python': sys.version, 'versions': {m.__name__: m.__version__ for m in (np, torch, PIL)},
            'module_files': {m.__name__: {'path': m.__file__, 'sha256': digest(m.__file__)} for m in (np, torch, PIL)}}


def limits():
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if not cpus:
        raise RuntimeError('No CPU affinity available')
    os.sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    return cpus


def acquire(args, cfg, data):
    folder = args.cache / 'assets'
    folder.mkdir(parents=True, exist_ok=True)
    planned = sum(item['bytes'] for item in data['images'])
    if any(isinstance(x['bytes'],bool) or not isinstance(x['bytes'],int) or x['bytes']<=0 or x['bytes']>cfg['download_cap_per_image_bytes'] for x in data['images']):
        raise RuntimeError('Frozen asset exceeds per-image cap before acquisition')
    # One possible overlength sentinel byte per attempt is reserved in advance.
    if cfg['acquisition_old_ceiling_bytes']+cfg['new_metadata_allowance_bytes']+planned+len(data['images']) > 10*1024**3:
        raise RuntimeError('Acquisition ceiling exceeded before GET')
    records = []
    for item in data['images']:
        dest = folder / Path(item['path']).name
        partial = dest.with_suffix('.partial')
        marker = folder / (item['id']+'.started.json')
        receipt_path = folder / (item['id']+'.json')
        url = 'https://raw.githubusercontent.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/'+data['source_commit']+'/'+item['path']
        def validate():
            body = dest.read_bytes()
            blob = hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
            if len(body) != item['bytes'] or blob != item['git_blob_sha1']:
                raise RuntimeError('Acquisition identity differs: '+item['id'])
            return {'id': item['id'], 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(), 'git_blob_sha1': blob, 'source_url': url, 'attempts': 1}
        if dest.exists():
            expected = validate()
            if not receipt_path.exists():
                raise RuntimeError('Existing asset has no matching completed receipt')
            result = read(receipt_path)
            if any(result.get(k)!=v for k,v in expected.items()) or partial.exists():
                raise RuntimeError('Cached acquisition receipt identity differs')
            records.append(result)
            continue
        if marker.exists() or receipt_path.exists() or partial.exists():
            raise RuntimeError('Preserved incomplete acquisition; no automatic retry')
        write(marker, {'started_utc': utc(), 'url': url, 'expected_bytes': item['bytes']})
        count = 0
        try:
            with urllib.request.urlopen(url, timeout=30) as response, partial.open('xb') as f:
                if response.status != 200:
                    raise RuntimeError('Expected HTTP 200')
                while True:
                    block = response.read(min(16384,item['bytes']-count+1))
                    if not block:
                        break
                    count += len(block)
                    f.write(block)
                    if count > item['bytes']:
                        raise RuntimeError('Download exceeded frozen bound (one sentinel byte)')
            if count != item['bytes']:
                raise RuntimeError('Download length mismatch')
            body = partial.read_bytes()
            if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest() != item['git_blob_sha1']:
                raise RuntimeError('Git blob identity mismatch')
            partial.rename(dest)
            result = validate()
            result['finished_utc'] = utc()
            write(receipt_path, result)
            records.append(result)
        except BaseException as exc:
            failure = folder/(item['id']+'.failure.json')
            if not failure.exists():
                write(failure, {'failed_utc':utc(),'bytes_read':count,'source_url':url,'error_type':type(exc).__name__,'error':str(exc),'partial_sha256':digest(partial) if partial.exists() else None})
            raise
    total = sum(x['bytes'] for x in records)
    ledger = {'historical_ceiling_bytes': cfg['acquisition_old_ceiling_bytes'], 'new_primary_document_metadata_allowance_bytes': cfg['new_metadata_allowance_bytes'], 'data_payload_bytes': total,
              'conservative_ceiling_bytes': cfg['acquisition_old_ceiling_bytes']+cfg['new_metadata_allowance_bytes']+total,
              'limit_bytes': 10*1024**3, 'assets': records}
    path = args.cache / 'acquisition.json'
    if not path.exists():
        write(path, ledger)
    elif read(path) != ledger:
        raise RuntimeError('Existing acquisition ledger differs')
    return ledger


def array_sha(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def prepare(args, cfg, data):
    import numpy as np
    from PIL import Image
    component = load_frozen_module("_evr0006_transform", "scripts/conditional_residual.py")
    covariance_from_kernel, default_kernel = component.covariance_from_kernel, component.default_kernel
    marker = args.cache / 'prepare.started.json'
    out = args.cache / 'prepared'
    if marker.exists() or out.exists():
        raise RuntimeError('Preparation already started; verify existing artifacts, do not repeat')
    write(marker, {'utc': utc()})
    out.mkdir()
    records = []
    for index, item in enumerate(data['images']):
        image_path = args.cache / 'assets' / Path(item['path']).name
        record = read(args.cache / 'assets' / (item['id']+'.json'))
        if digest(image_path) != record['sha256']:
            raise RuntimeError('Asset hash changed')
        with Image.open(image_path) as image:
            if image.format != 'JPEG' or image.size != (512,512) or image.mode not in ('RGB','L'):
                raise RuntimeError('Unexpected reference image schema')
            rgb = np.asarray(image.convert('RGB'), dtype=np.float64)
        gray = np.sum(rgb * np.array(cfg['grayscale_weights']), axis=2) / 255.0
        clean = gray[192:320, 192:320].astype(np.float32)
        ref_path = out / (item['id']+'-reference.npz')
        np.savez(ref_path, clean=clean)
        for condition in cfg['conditions']:
            h = np.ones((1,1), dtype=np.float64) if condition == 'white' else default_kernel(np.float64)
            covariance = covariance_from_kernel(h, cfg['sigma'])
            for seed in cfg['seeds']:
                noise_seed = 610000 + 100*index + seed
                eps = np.random.default_rng(noise_seed).standard_normal((128+h.shape[0]-1,128+h.shape[1]-1))
                noise = np.zeros((128,128), dtype=np.float64)
                for row in range(h.shape[0]):
                    for col in range(h.shape[1]):
                        noise += h[row,col] * eps[row:row+128,col:col+128]
                noisy = (clean.astype(np.float64) + cfg['sigma']*noise).astype(np.float32)
                name = f'{item["id"]}-{condition}-{seed}'
                inp = out / (name+'-input.npz')
                np.savez(inp, noisy=noisy, covariance=covariance)
                records.append({'name':name, 'image_id':item['id'], 'group':item['group'], 'condition':condition, 'seed':seed, 'noise_seed':noise_seed,
                                'input_sha256':digest(inp), 'reference_sha256':digest(ref_path), 'reference_array_sha256':array_sha(clean), 'noisy_array_sha256':array_sha(noisy), 'covariance_sha256':array_sha(covariance), 'kernel':h.tolist()})
    write(args.cache/'prepared.json', {'records':records, 'count':len(records)})
    return {'prepared_inputs':len(records)}


def case(args, cfg, data):
    # Exclusive directory is the irreversible per-case start marker, before data/fit.
    args.output.mkdir()
    import numpy as np
    component = load_frozen_module("_evr0006_trainer", "scripts/conditional_residual_training.py")
    fit_one = component.fit_one
    image_index = cfg['image_ids'].index(args.image_id)
    condition_index = cfg['conditions'].index(args.condition)
    if args.seed not in cfg['seeds'] or args.method not in cfg['methods']:
        raise RuntimeError('Unfrozen case')
    name = f'{args.image_id}-{args.condition}-{args.seed}'
    recs = read(args.cache/'prepared.json')['records']
    rec = next(x for x in recs if x['name']==name)
    inp = args.cache/'prepared'/(name+'-input.npz')
    if digest(inp) != rec['input_sha256']:
        raise RuntimeError('Input file changed')
    # No reference file is opened by the fitting process.
    with np.load(inp,allow_pickle=False) as payload:
        if set(payload.files) != {'noisy','covariance'}:
            raise RuntimeError('Unexpected input archive member')
        noisy=payload['noisy']; covariance=payload['covariance']
    base = 620000+100*image_index+10*args.seed+condition_index
    result = fit_one(noisy,covariance,args.method,init_seed=base,batch_seed=base+1,augment_seed=base+2,infer_seed=base+3)
    receipt = result['receipt']
    receipt.update({'image_id':args.image_id,'condition':args.condition,'seed':args.seed,'method':args.method,'input_file_sha256':digest(inp)})
    if result.get('state_dict_np') is not None:
        np.savez(args.output/'weights.npz',**result['state_dict_np'])
    if result['status'] != 'completed':
        receipt['files']={p.name:digest(p) for p in args.output.iterdir() if p.is_file()}
        write(args.output/'receipt.json',receipt)
        raise RuntimeError('Incomplete scientific case: '+result['status'])
    np.save(args.output/'prediction.npy',result['prediction'],allow_pickle=False)
    if 'direct_prediction' in result.get('extras',{}):
        np.save(args.output/'direct_prediction.npy',result['extras']['direct_prediction'],allow_pickle=False)
    receipt['files']={p.name:digest(p) for p in args.output.iterdir() if p.is_file()}
    write(args.output/'receipt.json',receipt)
    return {'status':result['status'],'case':name+'-'+args.method,'receipt_sha256':digest(args.output/'receipt.json')}


def group(args, cfg, data):
    global CHILD_PID
    if args.image_id not in cfg['image_ids']:
        raise RuntimeError('Unfrozen image group')
    start=time.monotonic(); results=[]
    folder=args.cache/'cases';folder.mkdir(exist_ok=True)
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            for method in cfg['methods']:
                name=f'{args.image_id}-{condition}-{seed}-{method}'
                output=folder/name; execution=folder/(name+'.execution.json'); marker=folder/(name+'.started.json')
                if execution.exists():
                    old=read(execution)
                    if old['exit_code'] != 0 or not (output/'receipt.json').exists():
                        raise RuntimeError('Prior failed case preserved; no automatic retry')
                    if digest(output/'receipt.json') != old['receipt_sha256']:
                        raise RuntimeError('Completed receipt changed')
                    for file,sha in read(output/'receipt.json')['files'].items():
                        if digest(output/file)!=sha:raise RuntimeError('Completed case output changed')
                    results.append(old);continue
                if marker.exists() or output.exists():
                    raise RuntimeError('Incomplete case preserved; no automatic retry')
                if time.monotonic()-start > cfg['group_timeout_seconds']-cfg['case_timeout_seconds']-10:
                    return {'status':'partial_time_budget','completed_this_group':len(results)}
                command=[sys.executable,str(Path(__file__).resolve()),'case','--cache',str(args.cache),'--runtime-root',str(args.runtime_root),'--overlay',str(args.overlay),'--freeze-commit',args.freeze_commit,'--image-id',args.image_id,'--condition',condition,'--seed',str(seed),'--method',method,'--output',str(output)]
                write(marker,{'started_utc':utc(),'command':command})
                before=time.monotonic();started=utc()
                with (folder/(name+'.stdout.log')).open('xb') as so, (folder/(name+'.stderr.log')).open('xb') as se:
                    p=subprocess.Popen(command,cwd=ROOT,stdout=so,stderr=se,start_new_session=True)
                    CHILD_PID=p.pid
                    timed_out=False
                    try:
                        pid,status,usage=os.wait4(p.pid,os.WNOHANG)
                        while pid==0:
                            if time.monotonic()-before > cfg['case_timeout_seconds']:
                                os.killpg(p.pid,signal.SIGKILL);timed_out=True
                                pid,status,usage=os.wait4(p.pid,0);break
                            time.sleep(.05);pid,status,usage=os.wait4(p.pid,os.WNOHANG)
                    finally:
                        if p.returncode is None and 'status' in locals():p.returncode=os.waitstatus_to_exitcode(status)
                CHILD_PID=None
                code=os.waitstatus_to_exitcode(status)
                record={'case':name,'command':command,'started_utc':started,'ended_utc':utc(),'exit_code':code,'raw_wait_status':status,'wall_seconds':time.monotonic()-before,'peak_rss_kib':usage.ru_maxrss,'timed_out':timed_out,'stdout_sha256':digest(folder/(name+'.stdout.log')),'stderr_sha256':digest(folder/(name+'.stderr.log')),'receipt_sha256':digest(output/'receipt.json') if (output/'receipt.json').exists() else None}
                write(execution,record);results.append(record)
                print(json.dumps({'case':name,'exit':code,'wall_seconds':record['wall_seconds']}),flush=True)
                if code!=0:raise RuntimeError('Scientific case failed; retain artifacts and stop group')
                if not (output/'receipt.json').exists() or read(output/'receipt.json').get('status')!='completed':
                    raise RuntimeError('Successful exit without a completed scientific receipt')
    return {'status':'completed','image_id':args.image_id,'case_count':len(results)}


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['acquire','prepare','case','group'])
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--freeze-commit',required=True);p.add_argument('--image-id');p.add_argument('--condition');p.add_argument('--seed',type=int);p.add_argument('--method');p.add_argument('--output',type=Path)
    args=p.parse_args();args.cache=args.cache.resolve();args.cache.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,terminate_family)
    signal.signal(signal.SIGALRM,terminate_family)
    if args.stage=='group':signal.alarm(1800)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='4'
    cpus=limits();frozen=verify_freeze(args.freeze_commit)
    lock=None
    if args.stage != 'case':
        lock=(args.cache/'execution.lock').open('a')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=configure_runtime(args)
    cfg=read(CFG_PATH);data=read(DATA_PATH)
    print(json.dumps({'stage':args.stage,'started_utc':utc(),'affinity':cpus,'source_commit':frozen['source_commit'],'environment':env}),flush=True)
    result=globals()[args.stage](args,cfg,data)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
