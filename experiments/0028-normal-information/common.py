"""Small immutable-file and source barriers for the prospective 0028 study.

No model, numeric input or truth access at import. Prediction and evaluation
are separate programs; a valid file seal never substitutes for outer exit.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN_SHA = 'cc9b02cac8946107f1c945643b316695e5ebbf6339b34baf737541f8cec886d5'
INPUT_FREEZE = 'acf502b831ad9e3485ab56549217c8a603d56bf7'
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
SCENES = ('s00513','s00475','s00478')
ARMS = ('E','A','B','C','LS','IRLS','ANG','FULL','NORMAL','NORMAL_REFINE','AFFINE')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for k,v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def reject(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)


def serial(value):
    if hasattr(value, 'tolist'):
        return serial(value.tolist())
    if isinstance(value, dict):
        return {str(k): serial(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [serial(v) for v in value]
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite scientific record')
    require(value is None or isinstance(value,(dict,list,str,int,float,bool)),
            'unsupported record type')
    return value


def write(path, value, *, replace=False):
    path = Path(path)
    encoded = json.dumps(serial(value),sort_keys=True,indent=2,allow_nan=False)+'\n'
    if replace:
        temp = path.with_suffix(path.suffix+'.pending')
        with temp.open('w') as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        temp.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def source_hashes():
    paths = sorted(HERE.glob('*.py'))+[HERE/'protocol.md']
    paths += [ROOT/'experiments/0027-affine-tangent/supervise.py',
              ROOT/'experiments/0025c-rs-noise/supervise.py',
              ROOT/'experiments/0023-mapfree/predict.py',
              ROOT/'experiments/0023-mapfree/cache_inputs.py',
              ROOT/'experiments/0026-calibration-moments/native_bridge/bridge.py',
              ROOT/'experiments/0026-calibration-moments/native_bridge/input-lock.json',
              ROOT/'configs/0017-libraries.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def verify_source(commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None, 'full commit required')
    head = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    require(head == commit, 'HEAD differs from scientific freeze')
    hashes = source_hashes()
    for name,digest in hashes.items():
        blob = subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name])
        require(hashlib.sha256(blob).hexdigest() == digest, 'unfrozen source: '+name)
    return hashes


def display(value):
    return str(value).replace(sys.executable,'$PYTHON').replace(str(ROOT),'$REPOSITORY').replace(str(ROOT.parent),'$PRIVATE')


def verify_outer(receipt_path, result_path, commit, command_script, job_path):
    """Reject a sealed result when its process, logs or supervision failed."""
    receipt_path, result_path = Path(receipt_path), Path(result_path)
    r = read(receipt_path)
    require(r['schema'] == 'evr0026.operation.v1' and r['status'] == 'completed'
            and r['exit_code'] == 0 and r['termination'] is None
            and r['remaining_descendant_pids'] == [] and r['cleanup_errors'] == [],
            'unclean prior outer operation')
    require(r['expected_commit'] == commit and 0 < r['hard_timeout_seconds'] <= 1800
            and 0 <= r['wall_seconds'] <= r['hard_timeout_seconds']
            and r['group_rss_limit_kib'] == 8388608
            and 0 <= r['peak_group_rss_kib'] <= 8388608
            and r['thread_environment'] == {k:'4' for k in THREAD_KEYS}, 'outer limits')
    require(r['supervisor_sha256'] == sha(ROOT/'experiments/0027-affine-tangent/supervise.py')
            and r['monitor_sha256'] == sha(ROOT/'experiments/0025c-rs-noise/supervise.py'),
            'outer supervisor changed')
    require(set(r['log_sha256']) == {'stdout.log','stderr.log'}, 'outer log schema')
    for name,digest in r['log_sha256'].items():
        require(sha(receipt_path.parent/name) == digest, 'outer log changed')
    require(r['output_sha256'] == sha(result_path), 'outer result binding')
    expected = '$REPOSITORY/experiments/0028-normal-information/'+command_script
    command = r['command']
    require(command[:4] == ['$PYTHON','-W','error',expected]
            and len(command) == 10, 'outer executed different program')
    flags = {}
    for key,value in zip(command[4::2],command[5::2]):
        require(key not in flags, 'duplicate outer command flag')
        flags[key] = value
    require(flags == {'--expected-commit':commit,'--job':display(Path(job_path).resolve()),
                      '--output':display(result_path.parent.resolve())},
            'outer command job/commit/output linkage')
    return dict(receipt_sha256=sha(receipt_path),result_sha256=sha(result_path),
                spec_sha256=r['spec_sha256'],wall_seconds=r['wall_seconds'])


def original_frontend(job):
    """Reuse the existing qualified provenance loader before any new import guard."""
    import argparse
    old = module('_evr0028_original_predict', ROOT/'experiments/0023-mapfree/predict.py')
    args = argparse.Namespace(frontend_cache=Path(job['cache']),
        frontend_supervision=Path(job['frontend_outer']),plan=Path(job['plan']),
        plan_sha256=PLAN_SHA,protocol=ROOT/'protocols/0023-mapfree-directional-instability.md',
        expected_commit=INPUT_FREEZE,input_freeze=INPUT_FREEZE)
    frontend, fa, receipt, outer, frozen = old.frontend_admission(args)
    cache = module('_evr0028_original_cache', ROOT/'experiments/0023-mapfree/cache_inputs.py')
    bindings = {k:receipt[k] for k in cache.BINDING_FIELDS}
    bindings['frontend_receipt_sha256'] = sha(args.frontend_cache/'receipt.json')
    verified = cache.verify_cache(args.plan,args.frontend_cache,bindings)
    return old,frontend,fa,receipt,outer,frozen,cache,verified


def selected_layout(verified):
    selected = []
    for layout in verified['layouts']:
        for edge in layout['query_edges']:
            selected.append(edge)
    require(len(selected) == 24 and tuple(dict.fromkeys(e['scene'] for e in selected)) == SCENES,
            'fixed query population')
    names = {e[k] for e in selected for k in ('source','target')}
    images = [r for r in verified['receipt']['images'] if r['filename'] in names]
    edges = [r for r in verified['receipt']['edges'] if r['role'] == 'A_to_B']
    require(len(images) == 27 and len(edges) == 24, 'fixed 27-image24-edge coverage')
    for actual,expected in zip(edges,selected):
        require(all(actual[k] == v for k,v in expected.items()), 'query order differs')
    return images,edges


def save_arrays(path, arrays):
    import numpy as np
    path = Path(path)
    require(not path.exists(), 'immutable arrays already exist')
    require(all(isinstance(k,str) and isinstance(v,np.ndarray) and v.dtype.kind != 'O'
                for k,v in arrays.items()), 'numeric array contract')
    np.savez_compressed(path, **arrays)
    return dict(filename=path.name,sha256=sha(path),bytes=path.stat().st_size,
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),
            raw_c_order_sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest())
            for k,v in arrays.items()})


def load_arrays(root, meta, fields=None):
    import numpy as np
    require(Path(meta['filename']).name == meta['filename'], 'unsafe array path')
    path = Path(root)/meta['filename']
    require(not path.is_symlink() and path.stat().st_size == meta['bytes']
            and sha(path) == meta['sha256'], 'sealed array changed')
    result = {}
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files) == set(meta['arrays']), 'array inventory changed')
        for name in sorted(meta['arrays'] if fields is None else fields):
            a = archive[name]; info = meta['arrays'][name]
            require(a.dtype.kind != 'O' and list(a.shape) == info['shape'] and str(a.dtype) == info['dtype']
                    and hashlib.sha256(a.tobytes(order='C')).hexdigest() == info['raw_c_order_sha256'],
                    'array content changed')
            result[name] = a.copy()
    return result
