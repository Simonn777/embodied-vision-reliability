"""One actual wait4 layer for preflight then one immutable diagnostic invocation."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path('results/0003-rotation-stageB-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src'}
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limit():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
phases=[('preflight',['python',str(B/'preflight.py')]),('diagnostic',['python','scripts/run_rotation_diagnostic_v1.py','--cache','../bootstapir-cache/0003-pointodyssey','--output',str(B/'audit')])]
for phase,command in phases:
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
 with(B/(phase+'.stdout.log')).open('xb')as out,(B/(phase+'.stderr.log')).open('xb')as err:
  child=subprocess.Popen(['timeout','900',*command],env=env,stdout=out,stderr=err,preexec_fn=limit)
  _,status,u=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
 r={'phase':phase,'command':['timeout','900',*command],'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'peak_rss_kib':u.ru_maxrss,'cpu_user_seconds':u.ru_utime,'cpu_system_seconds':u.ru_stime,'observed_exit_code':code,'raw_wait_status':status,'affinity':affinity,'AS_bytes':6*1024**3,'timeout_seconds':900,'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'source_sha256':{p:h(p)for p in [__file__,command[1],'src/evr/rotation_diagnostic_v1.py','configs/0003-rotation-diagnostic-v1.json','protocols/0003-rotation-diagnostic-v1.md','protocols/0003-rotation-diagnostic-freeze-v1.json']},'logs_sha256':{st:h(B/(phase+'.'+st+'.log'))for st in ['stdout','stderr']},'boundary':'actual parent wait4 of timeout/child; parent has no further observed supervisor exit receipt'}
 with(B/(phase+'-parent.json')).open('x')as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(r),flush=True)
 if code:raise SystemExit(code)
