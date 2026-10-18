"""Single-layer child observation for synthetic software validation only."""
import datetime, hashlib, importlib.util, json, os, resource, subprocess, time
from pathlib import Path

OUT=Path(__file__).parent
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src','MPLBACKEND':'Agg'}
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limits():
 os.sched_setaffinity(0,affinity);resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
def run(name,command):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
 with(OUT/(name+'.stdout.log')).open('xb')as out,(OUT/(name+'.stderr.log')).open('xb')as err:
  child=subprocess.Popen(['timeout','900',*command],env=env,stdout=out,stderr=err,preexec_fn=limits)
  _,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 r={'command':['timeout','900',*command],'environment':{**env,'PATH':'inherited runtime PATH value omitted'},'affinity':affinity,'AS_bytes':6*1024**3,'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'observed_exit_code':child.returncode,'raw_wait_status':status,'peak_rss_kib':u.ru_maxrss,'CPU_user_seconds':u.ru_utime,'CPU_system_seconds':u.ru_stime,'parent_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'logs_sha256':{s:hashlib.sha256((OUT/(name+'.'+s+'.log')).read_bytes()).hexdigest()for s in ['stdout','stderr']},'boundary':'parent observes child once; no further supervisor exit receipt'}
 (OUT/(name+'-execution.json')).write_text(json.dumps(r,indent=2)+'\n');print(name,child.returncode,r['wall_seconds']);return child.returncode
probe={key:importlib.util.find_spec(key)is not None for key in ['pytest','numpy','PIL','matplotlib']}
(OUT/'software-availability.json').write_text(json.dumps({'available':probe,'local_pytest_tests_run':0,'no_installation':True},indent=2)+'\n')
code=run('synthetic', ['python','-m','unittest','discover','-s','tests','-p','test_annotation_consistency_v1.py','-v'])
if all(probe[k]for k in ['numpy','PIL','matplotlib']):
 code=max(code,run('smoke',['python','scripts/run_experiment.py','--smoke','--output','scratch/cache-readiness-smoke','--protocol','protocols/0001-synthetic-localization.md']))
else:
 (OUT/'smoke-not-run.json').write_text(json.dumps({'reason':'required installed toy dependencies unavailable; no installation attempted','available':probe},indent=2)+'\n')
raise SystemExit(code)
