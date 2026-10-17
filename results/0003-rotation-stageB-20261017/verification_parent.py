"""One actual wait4 for saved-artifact verification, no further supervisor."""
import datetime,hashlib,json,os,resource,subprocess,time
from pathlib import Path
B=Path('results/0003-rotation-stageB-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();source=B/'verify_saved.py'
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src'}
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limit():resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat();command=['timeout','900','python',str(source)]
with(B/'verification.stdout.log').open('xb')as out,(B/'verification.stderr.log').open('xb')as err:
 p=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=limit);_,status,u=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status);p.returncode=code
r={'command':command,'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'peak_rss_kib':u.ru_maxrss,'CPU_user_seconds':u.ru_utime,'CPU_system_seconds':u.ru_stime,'observed_exit_code':code,'raw_wait_status':status,'source_sha256':{str(source):h(source),str(Path(__file__).resolve().relative_to(Path.cwd())):h(__file__)},'input_sha256':{str(p):h(p)for p in sorted((B/'audit').glob('*.json'))},'logs_sha256':{st:h(B/('verification.'+st+'.log'))for st in ['stdout','stderr']},'AS_bytes':6*1024**3,'affinity':affinity,'timeout_seconds':900,'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'boundary':'actual parent wait4 of verification child; parent itself has no further observed exit receipt'}
with(B/'verification-parent.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));raise SystemExit(code)
