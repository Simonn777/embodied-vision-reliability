"""Credential-free software checks; synthetic smoke only, no model replay."""
import datetime,hashlib,json,os,resource,subprocess,time
from pathlib import Path
A=Path('results/0002-pilot-execution/g-031-20261017')
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
env={'PATH':os.environ['PATH'],'PYTHONPATH':'scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src','MPLBACKEND':'Agg',**dict.fromkeys(['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'],'2')}
records={}
commands={'pytest_availability':['python','-c','import importlib.util,json; print(json.dumps({"pytest_available": importlib.util.find_spec("pytest") is not None, "tests_executed":0}))'], 'toy_smoke':['timeout','900','python','scripts/run_experiment.py','--smoke','--output','scratch/g031-software-smoke','--protocol','protocols/0001-synthetic-localization.md']}
for name,command in commands.items():
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();before=time.perf_counter()
 with (A/(name+'.stdout.log')).open('x')as out,(A/(name+'.stderr.log')).open('x')as err:
  p=subprocess.Popen(command,env=env,preexec_fn=limits,stdout=out,stderr=err);_,status,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status)
 records[name]={'command':command,'environment':{k:v for k,v in env.items()if k!='PATH'},'path_policy':'inherited PATH only; no other inherited variables','started_at_utc':start,'ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-before,'exit_code':p.returncode,'raw_wait_status':status,'peak_rss_kib':u.ru_maxrss,'log_sha256':{stream:hashlib.sha256((A/(name+'.'+stream+'.log')).read_bytes()).hexdigest()for stream in ['stdout','stderr']}}
records['pytest_scope']='Availability probe only; if unavailable, no Work tests executed/pytest installed. Full tests run in exact-head public Research checks CI.'
records['no_model_replay']=True
(A/'software-checks.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))

raise SystemExit(0 if all(v["exit_code"]==0 for v in records.values() if isinstance(v,dict)) else 1)
