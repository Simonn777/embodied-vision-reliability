"""Run only the authorized frozen-v2 g-040 inference with an outer audit trail.

This original orchestration wrapper never edits frozen inputs or imports the
model. The unchanged, main-published runner performs inference in a separate
credential-free process. A cache-scoped marker/lock prevents duplicate launches.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path('results/0002-pilot-execution/g-040-20261017')
OUTPUT = Path('results/0002-pilot-predictions/v2-reconstructed/g-040')
CACHE = Path('scratch/boot-cache')
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '2')
PYTHONPATH = 'scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src'

# No archive decoding, model import, downloads, score calculation or GT errors.
PREFLIGHT = r'''
import hashlib, importlib.metadata, json, os, platform, resource, shutil, sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
available=sorted(os.sched_getaffinity(0))
if len(available)<2: raise ValueError('two CPU affinity slots required')
os.sched_setaffinity(0,available[:2])
from evr.pilot_frozen import verify_freeze, CANONICAL_RECEIPT
from evr.bootstapir_dev import sha256, verify_upstream, OBJECTS
root=Path.cwd();r=verify_freeze(root,CANONICAL_RECEIPT)
assert sha256(root/CANONICAL_RECEIPT)=='bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b'
assert r['input_commit']=='bbe7428e126a566a62d3c123d7158f120aa66e2b'
manifest=json.loads((root/r['manifest_path']).read_text())
groups=[g for g in manifest['groups'] if g['group_id']=='g-040']
assert len(groups)==1 and groups[0]['members']==[40] and groups[0]['partition']=='calibration'
cache=Path('scratch/boot-cache');objects={}
for name,key in [('weights.pt','checkpoint_sha256'),('rgb.zip','archive_sha256')]:
 p=cache/name
 if not p.is_file() or p.stat().st_size!=OBJECTS[name][1]:raise ValueError('required cache missing/size: '+name)
 digest=sha256(p)
 if digest!=r[key]:raise ValueError('required cached SHA256 mismatch: '+name)
 objects[name]={'bytes':p.stat().st_size,'sha256':digest,'cache_hit':True}
verify_upstream(cache/'source')
versions={p:importlib.metadata.version(p) for p in r['dependency_versions']}
import importlib.util,numpy,torch
assert numpy.__version__==versions['numpy']=='2.0.2'
assert torch.__version__==versions['torch']=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(2);torch.set_num_interop_threads(2)
modules={'torch':'torch','numpy':'numpy','dm-tree':'tree','einshape':'einshape','absl-py':'absl','attrs':'attrs','wrapt':'wrapt','sympy':'sympy','filelock':'filelock','networkx':'networkx','jinja2':'jinja2','fsspec':'fsspec','mpmath':'mpmath','MarkupSafe':'markupsafe','typing-extensions':'typing_extensions','setuptools':'setuptools'}
historical=json.loads((root/r['dependency_identity_receipt']).read_text())
origins={}
def cache_label(p):return '<cache>/'+str(Path(p).resolve().relative_to(cache.resolve()))
for package,module in modules.items():
 d=importlib.metadata.distribution(package);spec=importlib.util.find_spec(module)
 origin=Path(spec.origin).resolve();dist=Path(d._path).resolve()
 assert origin.is_relative_to(dist.parent) and origin.is_relative_to(cache.resolve()),package
 record=(d.read_text('RECORD') or '').encode()
 loaded=sys.modules.get(module);version=getattr(loaded,'__version__',None) if loaded else None
 origins[package]={'module':module,'resolved_module_path':cache_label(origin),'module_entry_sha256':sha256(origin),'dist_info_path':cache_label(dist),'metadata_version':d.version,'loaded_module_version':str(version) if version is not None else None,'module_origin_matches_distribution':True,'installed_RECORD_sha256':hashlib.sha256(record).hexdigest(),'historical_RECORD_sha256':historical['packages'][package]['installed_RECORD_sha256'],'RECORD_matches_historical':hashlib.sha256(record).hexdigest()==historical['packages'][package]['installed_RECORD_sha256']}
assert Path(numpy.__file__).resolve()==Path(importlib.util.find_spec('numpy').origin).resolve()
assert Path(torch.__file__).resolve()==Path(importlib.util.find_spec('torch').origin).resolve()
assert Path(torch._C.__file__).resolve().is_relative_to(cache.resolve())
assert Path(numpy._core._multiarray_umath.__file__).resolve().is_relative_to(cache.resolve())

mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
 k,v=line.split(':',1)
 if k in ('MemTotal','MemAvailable'):mem[k]=int(v.split()[0])
cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
cgroup={}
for name in ['memory.max','memory.current','cpu.max']:
 p=Path('/sys/fs/cgroup')/name
 cgroup[name]=p.read_text().strip() if p.exists() else None
disk=shutil.disk_usage('.')
report={'status':'passed','gate':'unchanged real verify_freeze; no mocks or bypass',
 'head':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'origin_main':__import__('subprocess').check_output(['git','rev-parse','origin/main'],text=True).strip(),
 'freeze_input_commit':r['input_commit'],'receipt_sha256':sha256(root/CANONICAL_RECEIPT),
 'input_sha256':r['input_sha256'],'actual_dependency_versions':versions,'actual_module_origins':origins,'numpy_actual_version':numpy.__version__,'torch_actual_version':torch.__version__,'torch_cuda':torch.version.cuda,'torch_threads':[torch.get_num_threads(),torch.get_num_interop_threads()],'numpy_native_module':cache_label(numpy._core._multiarray_umath.__file__),'torch_native_module':cache_label(torch._C.__file__),
 'cached_objects':objects,'upstream_module_sha256':r['upstream_module_sha256'],
 'group':groups[0],'inference_scope':'g-040 only; video40 first64 frames, up to first2 frame0-visible points',
 'python':platform.python_version(),'platform':platform.platform(),'cpu_model':cpu,
 'numeric_thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},
 'affinity':sorted(os.sched_getaffinity(0)),'address_space_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],
 'meminfo_start_kib':mem,'cgroup':cgroup,'disk_start_bytes':dict(zip(['total','used','free'],disk)),
 'new_download_bytes_this_step':0,'prior_reconstruction_wheel_bytes':19236173,'prior_reconstruction_pypi_metadata_bytes':90923,'environment_identity_scope':'version-matched environment reconstruction; new wheel/RECORD identities are recorded, not retroactively frozen'}
Path('results/0002-pilot-execution/g-040-20261017/preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('Real frozen-v2 gate passed: 29 inputs, 16 installed dependencies, verified cached data/weights/source; g-040 only.')
'''


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    # Refuse reuse before touching an existing experiment or its receipts.
    if OUTPUT.exists():
        raise FileExistsError('g-040 output already exists: verify/publish it, never rerun')
    AUDIT.mkdir(parents=True, exist_ok=False)
    rec={'status':'starting','started_at_utc':utc(),'group_id':'g-040',
         'output_path':str(OUTPUT),'wrapper_sha256':digest(__file__),
         'phases':{},'environment_reconstruction':'official NumPy2.0.2 version-matched reconstruction; not byte-identical environment reproduction'}
    lock=None;global_lock=None;owned_marker=False;marker=CACHE/'frozen-v2-g040-launch.json'
    def save():
        (AUDIT/'outer.json').write_text(json.dumps(rec,indent=2)+'\n')
    save()
    (AUDIT/'launcher-source.py').write_bytes(Path(__file__).read_bytes())
    (AUDIT/'prelaunch.json').write_bytes(Path('scratch/prelaunch.json').read_bytes())
    start=time.perf_counter()
    try:
        global_lock=(CACHE/'frozen-v2-research.lock').open('a')
        fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=(CACHE/'frozen-v2-g040.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with marker.open('x') as f:
            json.dump({'status':'reserved','started_at_utc':rec['started_at_utc'],
                       'audit_path':str(AUDIT),'output_path':str(OUTPUT)},f)
        owned_marker=True
        rec['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        rec['origin_main']=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
        if rec['source_commit']!='3c6b26ac1997216eb60b8509df9abea71ed093d3' or rec['origin_main']!=rec['source_commit']:
            raise ValueError('expected freshly fetched main mismatch')
        prefix=['env','-i','PATH='+os.environ['PATH'],'PYTHONPATH='+PYTHONPATH,
                *[k+'='+v for k,v in THREADS.items()],'timeout','900','python']
        def run_phase(name,args,display):
            phase={'started_at_utc':utc(),'command':display,'status':'running'}
            rec['phases'][name]=phase;save()
            before=time.perf_counter()
            with (AUDIT/(name+'.stdout.log')).open('x') as out, (AUDIT/(name+'.stderr.log')).open('x') as err:
                child=subprocess.Popen(prefix+args,stdout=out,stderr=err)
                pid,status,usage=os.wait4(child.pid,0)
                child.returncode=os.waitstatus_to_exitcode(status)
            phase.update(ended_at_utc=utc(),wall_seconds=time.perf_counter()-before,
                         exit_code=child.returncode,raw_wait_status=status,
                         peak_rss_kib=usage.ru_maxrss,cpu_user_seconds=usage.ru_utime,
                         cpu_system_seconds=usage.ru_stime,
                         status='completed' if child.returncode==0 else 'failed')
            phase['stdout_sha256']=digest(AUDIT/(name+'.stdout.log'))
            phase['stderr_sha256']=digest(AUDIT/(name+'.stderr.log'));save()
            if child.returncode!=0:
                raise RuntimeError(name+' failed with exit '+str(child.returncode)+'; no retry')
        display='env -i PATH="$PATH" PYTHONPATH='+PYTHONPATH+' '+ ' '.join(k+'=2' for k in THREADS)+' timeout 900 python '
        (AUDIT/'preflight-code.py').write_text(PREFLIGHT)
        run_phase('preflight',['-c',PREFLIGHT],display+'-c <preflight-code.py contents>')
        command=display+'scripts/run_frozen_pilot.py infer --cache '+str(CACHE)+' --group-id g-040 --output '+str(OUTPUT)
        (AUDIT/'command.txt').write_text(command+'\n')
        run_phase('infer',['scripts/run_frozen_pilot.py','infer','--cache',str(CACHE),
                          '--group-id','g-040','--output',str(OUTPUT)],command)
        receipt=json.loads((OUTPUT/'receipt.json').read_text())
        if receipt['status']!='completed' or receipt['group_id']!='g-040' or receipt['partition']!='calibration' or set(receipt['videos'])!={'40'}:
            raise ValueError('completed g-040 receipt required; no retry')
        rec['status']='completed'
    except Exception as exc:
        rec.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                   traceback=traceback.format_exc().replace(str(ROOT),'<repo>'))
    finally:
        rec.update(ended_at_utc=utc(),wall_seconds=time.perf_counter()-start)
        save()
        # Preserve the cache-scoped reservation after success OR failure.
        if lock is not None:
            if owned_marker:
                marker.write_text(json.dumps({'status':rec['status'],'started_at_utc':rec['started_at_utc'],
                    'ended_at_utc':rec['ended_at_utc'],'audit_path':str(AUDIT),'output_path':str(OUTPUT)},indent=2)+'\n')
            lock.close()
        if global_lock is not None:global_lock.close()
    print(json.dumps({'status':rec['status'],'phases':{k:v.get('exit_code') for k,v in rec['phases'].items()},
                      'audit_path':str(AUDIT),'output_path':str(OUTPUT)}))
    return 0 if rec['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
