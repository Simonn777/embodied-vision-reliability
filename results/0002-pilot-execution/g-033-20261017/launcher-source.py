"""Run only the authorized frozen-v2 g-033 inference with an outer audit trail.

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
AUDIT = Path('results/0002-pilot-execution/g-033-20261017')
OUTPUT = Path('results/0002-pilot-predictions/g-033')
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
assert r['input_commit']=='bbe7428e126a566a62d3c123d7158f120aa66e2b'
manifest=json.loads((root/r['manifest_path']).read_text())
groups=[g for g in manifest['groups'] if g['group_id']=='g-033']
assert len(groups)==1 and groups[0]['members']==[33] and groups[0]['partition']=='calibration'
cache=Path('scratch/boot-cache');objects={}
for name,key in [('weights.pt','checkpoint_sha256'),('rgb.zip','archive_sha256')]:
 p=cache/name
 if not p.is_file() or p.stat().st_size!=OBJECTS[name][1]:raise ValueError('required cache missing/size: '+name)
 digest=sha256(p)
 if digest!=r[key]:raise ValueError('required cached SHA256 mismatch: '+name)
 objects[name]={'bytes':p.stat().st_size,'sha256':digest,'cache_hit':True}
verify_upstream(cache/'source')
versions={p:importlib.metadata.version(p) for p in r['dependency_versions']}
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
 'input_sha256':r['input_sha256'],'actual_dependency_versions':versions,
 'cached_objects':objects,'upstream_module_sha256':r['upstream_module_sha256'],
 'group':groups[0],'inference_scope':'g-033 only; video33 first64 frames, up to first2 frame0-visible points',
 'python':platform.python_version(),'platform':platform.platform(),'cpu_model':cpu,
 'numeric_thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},
 'affinity':sorted(os.sched_getaffinity(0)),'address_space_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],
 'meminfo_start_kib':mem,'cgroup':cgroup,'disk_start_bytes':dict(zip(['total','used','free'],disk)),
 'new_data_model_or_dependency_download_bytes':0}
Path('results/0002-pilot-execution/g-033-20261017/preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('Real frozen-v2 gate passed: 29 inputs, 16 installed dependencies, verified cached data/weights/source; g-033 only.')
'''


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    # Refuse reuse before touching an existing experiment or its receipts.
    if OUTPUT.exists():
        raise FileExistsError('g-033 output already exists: verify/publish it, never rerun')
    AUDIT.mkdir(parents=True, exist_ok=False)
    rec={'status':'starting','started_at_utc':utc(),'group_id':'g-033',
         'output_path':str(OUTPUT),'wrapper_sha256':digest(__file__),
         'phases':{},'new_binary_download_bytes':0}
    lock=None;marker=CACHE/'frozen-v2-g033-launch.json'
    def save():
        (AUDIT/'outer.json').write_text(json.dumps(rec,indent=2)+'\n')
    save()
    start=time.perf_counter()
    try:
        lock=(CACHE/'frozen-v2-g033.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with marker.open('x') as f:
            json.dump({'status':'reserved','started_at_utc':rec['started_at_utc'],
                       'audit_path':str(AUDIT),'output_path':str(OUTPUT)},f)
        rec['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        rec['origin_main']=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
        if rec['source_commit']!='05c2d7cb4ec1f81b6bf080252deb69bd9a72a319' or rec['origin_main']!=rec['source_commit']:
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
        command=display+'scripts/run_frozen_pilot.py infer --cache '+str(CACHE)+' --group-id g-033 --output '+str(OUTPUT)
        (AUDIT/'command.txt').write_text(command+'\n')
        run_phase('infer',['scripts/run_frozen_pilot.py','infer','--cache',str(CACHE),
                          '--group-id','g-033','--output',str(OUTPUT)],command)
        receipt=json.loads((OUTPUT/'receipt.json').read_text())
        if receipt['status']!='completed' or receipt['group_id']!='g-033' or receipt['partition']!='calibration' or set(receipt['videos'])!={'33'}:
            raise ValueError('completed g-033 receipt required; no retry')
        rec['status']='completed'
    except Exception as exc:
        rec.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                   traceback=traceback.format_exc().replace(str(ROOT),'<repo>'))
    finally:
        rec.update(ended_at_utc=utc(),wall_seconds=time.perf_counter()-start)
        save()
        # Preserve the cache-scoped reservation after success OR failure.
        if lock is not None:
            if marker.exists():
                marker.write_text(json.dumps({'status':rec['status'],'started_at_utc':rec['started_at_utc'],
                    'ended_at_utc':rec['ended_at_utc'],'audit_path':str(AUDIT),'output_path':str(OUTPUT)},indent=2)+'\n')
            lock.close()
    print(json.dumps({'status':rec['status'],'phases':{k:v.get('exit_code') for k,v in rec['phases'].items()},
                      'audit_path':str(AUDIT),'output_path':str(OUTPUT)}))
    return 0 if rec['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
