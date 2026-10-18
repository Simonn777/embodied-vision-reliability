"""One actual parent observation of the independent saved-prediction checker."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
OUT=Path('results/0004-stage0');target=OUT/'independent-check-execution.json'
if target.exists():raise ValueError('exclusive checker receipt')
env={'PATH':os.environ['PATH'],'PYTHONPATH':'../bootstapir-cache/runtime-fixed:../bootstapir-cache/runtime:../bootstapir-cache/source:src','PYTHONDONTWRITEBYTECODE':'1'}
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limits():os.sched_setaffinity(0,affinity);resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
command=['timeout','--kill-after=5s','900',sys.executable,str(OUT/'check_saved.py')]
record={'state':'STARTING','command':command,'environment':{**env,'PATH':'inherited runtime PATH omitted'},'affinity':affinity,'AS_bytes':6*1024**3,'checker_sha256':hashlib.sha256((OUT/'check_saved.py').read_bytes()).hexdigest(),'parent_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observed_exit_code':None,'raw_wait_status':None,'boundary':'parent observes checker child once; no additional parent exit receipt'}
target.write_text(json.dumps(record,indent=2)+'\n');t=time.perf_counter()
with(OUT/'independent-check.stdout.log').open('xb')as out,(OUT/'independent-check.stderr.log').open('xb')as err:
 child=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=limits)
 _,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
record.update(state='FINISHED',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-t,observed_exit_code=child.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,logs_sha256={s:hashlib.sha256((OUT/('independent-check.'+s+'.log')).read_bytes()).hexdigest()for s in ['stdout','stderr']})
temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(target)
print(json.dumps(record,indent=2));raise SystemExit(child.returncode)
