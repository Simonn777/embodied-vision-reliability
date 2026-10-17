
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
