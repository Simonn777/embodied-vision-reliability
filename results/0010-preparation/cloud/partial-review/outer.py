"""Actual parent wait4 for the one read-only partial diagnosis."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert {0,1,2,3}<=os.sched_getaffinity(0);os.sched_setaffinity(0,{0,1,2,3});resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
cmd=['timeout','--kill-after=5s','120',sys.executable,str(HERE/'diagnose.py')]
with (HERE/'command.json').open('x') as f:json.dump({'argv':cmd,'cwd':str(ROOT),'numeric_threads':4,'affinity':[0,1,2,3],'AS_bytes':8*1024**3},f,indent=2);f.write('\n')
start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
with (HERE/'stdout.log').open('xb') as out,(HERE/'stderr.log').open('xb') as err:
 p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True);_,status,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status);out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
d={'actual_exit':p.returncode,'raw_wait_status':status,'utc_started':start,'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-clock,'user_cpu_seconds':u.ru_utime,'system_cpu_seconds':u.ru_stime,'peak_rss_kib':u.ru_maxrss,'command_sha256':sha(HERE/'command.json'),'stdout_sha256':sha(HERE/'stdout.log'),'stderr_sha256':sha(HERE/'stderr.log'),'observer_source_sha256':sha(__file__),'parent_boundary':'wait4 observes timeout/diagnosis child; observer itself has no recursively added supervisor.'}
with (HERE/'execution.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(d,sort_keys=True));raise SystemExit(p.returncode)
