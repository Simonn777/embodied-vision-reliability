"""Prospective artificial solver admission. Production entry is externally supervised."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parent
PATH_LINE = "UPSTREAM_PATH = Path('/tmp/evr-stata-source-admission/source/solvers/StatA.py')"
PORTABLE_LINE = "UPSTREAM_PATH = Path(__file__).resolve().parent / 'upstream' / 'StatA.py'"


def sha(data): return hashlib.sha256(data).hexdigest()

def array_hash(a):
    return sha(str((a.shape, a.dtype.str)).encode() + b'\0' + a.tobytes(order='C'))


def verify_bundle(expected):
    raw = (ROOT / 'manifest.json').read_bytes()
    if sha(raw) != expected: raise ValueError('manifest_sha256')
    manifest = json.loads(raw)
    for name, digest in manifest['files'].items():
        p = ROOT / name
        if p.is_symlink() or not p.is_file() or p.resolve().relative_to(ROOT).as_posix() != name or sha(p.read_bytes()) != digest:
            raise ValueError('bundle_file:' + name)
    return manifest


def load_exact(name, path, *, portable=False, test_threads=False):
    raw = path.read_bytes(); text = raw.decode()
    if portable:
        if text.count(PATH_LINE) != 1: raise ValueError('portable_path_count')
        text = text.replace(PATH_LINE, PORTABLE_LINE, 1)
    if test_threads:
        old = 'torch.set_num_interop_threads(1)'
        if text.count(old) != 1: raise ValueError('test_thread_substitution_count')
        text = text.replace(old, 'torch.set_num_interop_threads(4)', 1)
    m = ModuleType(name); m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(text, str(path), 'exec'), m.__dict__)
    return m, dict(original_sha256=sha(raw), executed_sha256=sha(text.encode()),
                   exact_path_substitutions=1 if portable else 0,
                   exact_test_interop_substitutions=1 if test_threads else 0)


def checkpoint(path, record):
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as f:
        json.dump(record, f, indent=2, allow_nan=False); f.write('\n')
    temporary.replace(path)


def resource_gate(cfg):
    if not sys.platform.startswith('linux'): raise ValueError('linux_required')
    actual = dict(affinity=sorted(os.sched_getaffinity(0)), rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),
                  cgroup_memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(), thread_environment={})
    if actual['affinity'] != cfg['cpu_affinity']: raise ValueError('affinity')
    if actual['rlimit_as'] != [cfg['address_space_bytes']]*2: raise ValueError('address_space')
    if actual['cgroup_memory_max'] != str(cfg['cgroup_memory_max_bytes']): raise ValueError('cgroup_memory')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        actual['thread_environment'][name] = os.environ.get(name)
        if os.environ.get(name) != str(cfg['torch_threads']): raise ValueError('thread_environment:' + name)
    return actual


def make_arrays(np, seed, n=64, d=512, k=10):
    rng = np.random.Generator(np.random.PCG64(seed))
    x = rng.standard_normal((n,d)); anchors = rng.standard_normal((k,d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    return x.astype('float32'), anchors.T[None].astype('float32')


def measure(core, x, p, ids, source_path, record, save, clock=time.perf_counter):
    """The only fit caller; tiny fake backends may exercise it locally."""
    phases = [('T63',1),('T641',1),('T644',4)]
    for tag, alpha in phases:
        event = dict(phase=tag,alpha=alpha,status='attempted',native_solver_execution_count_unknown=True)
        record['attempts'].append(event); save()
        began = clock()
        try:
            state = core.fit(x[1:] if tag=='T63' else x, p,
                             ids=ids[1:] if tag=='T63' else ids, alpha=alpha, source_path=source_path)
            if tag=='T63': core.heldout_readout(state,x[0],query_id=ids[0],source_path=source_path)
            if state.get('status')!='ok' or state.get('reference_bitwise_equal') is not True or state.get('native_solver_executions')!=2:
                raise ValueError('fit_parity')
            event.update(status='completed',native_solver_executions=2,native_solver_execution_count_unknown=False)
        except Exception as e:
            event.update(status='failed',error_code=getattr(e,'code',type(e).__name__))
            raise
        finally:
            event['seconds'] = clock()-began
            if not math.isfinite(event['seconds']) or event['seconds']<0: raise ValueError('nonfinite_timing')
            save()
    times = [event['seconds'] for event in record['attempts']]
    record['projected_seconds'] = 1.5*(128*times[0]+2*times[1]+2*times[2])
    record['resource_admitted'] = record['projected_seconds']<=360
    record['status'] = 'completed' if record['resource_admitted'] else 'resource_not_admitted'
    save()


def child(mode, out, expected):
    manifest = verify_bundle(expected); cfg = json.loads((ROOT/'config.json').read_bytes())
    out.mkdir(parents=True,exist_ok=False)
    rec = dict(schema=1,scope=cfg['scope'],mode=mode,status='started',source_files=manifest['files'],
               manifest_sha256=expected,attempts=[],argv=sys.argv,started_unix=time.time())
    path=out/'receipt.json'; save=lambda:checkpoint(path,rec)
    save(); start=time.perf_counter()
    try:
        rec['resource']=resource_gate(cfg); save()
        import numpy as np
        import torch
        rec['runtime']=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                            numpy_path=np.__file__,torch_path=torch.__file__)
        save()
        if '.'.join(platform.python_version_tuple()[:2])!=cfg['runtime_versions']['python_major_minor'] or np.__version__!=cfg['runtime_versions']['numpy'] or torch.__version__!=cfg['runtime_versions']['torch']:
            raise ValueError('runtime_version')
        core,binding=load_exact('stata_cpu',ROOT/'stata_cpu.py',portable=True)
        rec['adapter_execution']=binding; save()
        if mode=='functional':
            # Only the test thread setter is transparently changed from interop=1 to 4.
            test,binding=load_exact('_stata_functional_tests',ROOT/'test_stata_cpu.py',test_threads=True)
            rec['test_execution']=binding
            if (torch.get_num_threads(),torch.get_num_interop_threads())!=(4,4): raise ValueError('torch_threads')
            rec['actual_torch_threads']=[torch.get_num_threads(),torch.get_num_interop_threads()]
            rec['functional_attempt']='started'; save()
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test))
            rec['tests']=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
            rec['status']='completed' if result.wasSuccessful() and result.testsRun==20 and not result.skipped else 'failed'
        else:
            torch.set_num_threads(4); torch.set_num_interop_threads(4)
            rec['actual_torch_threads']=[torch.get_num_threads(),torch.get_num_interop_threads()]
            if rec['actual_torch_threads'] != [4,4]: raise ValueError('torch_threads')
            began=time.perf_counter()
            x,p=make_arrays(np,cfg['seed']); ids=tuple('artificial-%03d'%i for i in range(64))
            np.savez(out/'artificial-inputs.npz',features=x,prototypes=p)
            rec['inputs']=dict(seed=cfg['seed'],rng=cfg['rng'],features_sha256=array_hash(x),prototypes_sha256=array_hash(p),
                               archive_sha256=sha((out/'artificial-inputs.npz').read_bytes()),ids=list(ids),shape_features=list(x.shape),shape_prototypes=list(p.shape))
            tx,tp=torch.from_numpy(x),torch.from_numpy(p)
            rec['input_setup_seconds']=time.perf_counter()-began; save()
            measure(core,tx,tp,ids,ROOT/'upstream/StatA.py',rec,save)
            rec['inputs_unchanged'] = (array_hash(x)==rec['inputs']['features_sha256'] and array_hash(p)==rec['inputs']['prototypes_sha256'])
            if not rec['inputs_unchanged']: raise ValueError('input_mutation')
        return 0 if rec['status']=='completed' else 1
    except Exception as e:
        rec.update(status='failed',error_code=getattr(e,'code',type(e).__name__))
        traceback.print_exc(file=sys.stderr)
        print('admission failed: '+str(rec['error_code']),file=sys.stderr,flush=True)
        return 1
    finally:
        rec['wall_before_final_save_seconds']=time.perf_counter()-start; save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('functional','timing'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args(); manifest=verify_bundle(args.manifest_sha256)
    if args.child: return child(args.mode,Path(args.output),args.manifest_sha256)
    # This contains no solver calls. Actual Linux child isolation comes from the reused supervisor.
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=False)
    supervisor,_=load_exact('_existing_supervisor',ROOT/'supervise_0009.py')
    command=[sys.executable,str(ROOT/'run_admission.py'),args.mode,'--child','--output',str(root/'child'),
             '--manifest-sha256',args.manifest_sha256]
    receipt=supervisor.supervise(command,root/'supervision',seconds=120,cpu_ids=[0,1,2,3])
    records={}
    for p in sorted(root.rglob('*')):
        if p.is_file(): records[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
    (root/'artifact-manifest.json').write_text(json.dumps(dict(files=records,self_excluded='artifact-manifest.json',source_manifest_sha256=args.manifest_sha256),indent=2)+'\n')
    return receipt['supervisor_exit_code']


if __name__=='__main__': raise SystemExit(main())
