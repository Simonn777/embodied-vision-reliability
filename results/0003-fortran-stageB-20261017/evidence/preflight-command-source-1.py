"""Real frozen gate + duplicate/cache existence checks, no numeric GT decode."""
import fcntl,hashlib,importlib.util,json,os,platform,subprocess
from pathlib import Path
spec=importlib.util.spec_from_file_location('approved','scripts/audit_0003_fortran_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.verify_published();cache=Path('../bootstapir-cache/0003-pointodyssey');output=Path('results/0003-fortran-stageB-20261017/audit')
assert r['input_commit']=='93be3c95dee7cb5d23a5eae9b110a08cfce2a78d'
assert hashlib.sha256(Path(m.RECEIPT).read_bytes()).hexdigest()=='ad6250bb93453b0ff8f0e9ef4e25b853cac99eb37f910ea007fa389f285f2da1'
assert not (cache/'fortran-v2-started.json').exists() and not output.exists()
locks=[]
for name in ['0003-design.lock','0003-sample.lock','0003-fortran-v2.lock']:
 f=(cache.parent/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
files=[(cache/'sample-v1.2.tar.gz',3324284510),(cache/'resource-audit-v1/selected-annotation.npz',98527880)]
for p,n in files:assert p.is_file() and not p.is_symlink() and p.stat().st_size==n
cpu=next((x.split(':',1)[1].strip()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('model name')),'unknown')
mem={x.split(':',1)[0]:x.split(':',1)[1].strip()for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith(('MemTotal:','MemAvailable:'))}
prior_unique=set()
for p in m.PRIOR:prior_unique.update(json.loads(Path(p).read_text())['inputs_sha256'])
print(json.dumps({'status':'passed_real_verify_published','input_commit':r['input_commit'],'receipt_commit':m.git('rev-parse','HEAD').decode().strip(),'remote_ref':m.git('rev-parse',m.BRANCH).decode().strip(),'mandatory':len(m.INPUTS),'prior_receipt_maps':3,'unique_prior_inputs':len(prior_unique),'bindings':'current/index/HEAD/input/remote receipt and input blobs, ancestry, unchanged prior hashes; exact frozen runner code','marker_and_output':'absent','locks':'all obtainable and released when this preflight exits; real runner reacquires','cache_stat':'both exact sizes; whole hashes enforced by real audit before data access','numeric_GT_access':0,'python':platform.python_version(),'platform':platform.platform(),'CPU':cpu,'affinity':sorted(os.sched_getaffinity(0)),'memory':mem,'free_disk_bytes':os.statvfs(cache).f_bavail*os.statvfs(cache).f_frsize},indent=2))
