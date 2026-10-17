"""Actual one-layer wait4 for synthetic-only Stage A commands, no auto retry."""
import datetime,hashlib,importlib.util,json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path('results/0003-rotation-stageA-20261017');phase=sys.argv[1];command=sys.argv[2:];h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('candidate','scripts/run_rotation_diagnostic_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
inputs={p:h(p)for p in sorted(m.MANDATORY)}
for p in m.MANDATORY:
 if p.startswith(('protocols/0003-rotation','configs/0003-rotation','src/evr/rotation','scripts/run_rotation','tests/test_rotation','results/0003-rotation')):
  dest=B/(phase+'-source')/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes())
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src'}
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limit():resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with(B/(phase+'.stdout.log')).open('xb')as out,(B/(phase+'.stderr.log')).open('xb')as err:
 p=subprocess.Popen(['timeout','900',*command],env=env,stdout=out,stderr=err,preexec_fn=limit)
 _,status,u=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status);p.returncode=code
r={'phase':phase,'command':['timeout','900',*command],'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':inputs,'command_source_sha256':{p:h(p)for p in command if p.endswith('.py')and Path(p).is_file()},'parent_source_sha256':h(__file__),'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'affinity':affinity,'AS_bytes':6*1024**3,'timeout_seconds':900,'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'peak_rss_kib':u.ru_maxrss,'cpu_user_seconds':u.ru_utime,'cpu_system_seconds':u.ru_stime,'observed_exit_code':code,'raw_wait_status':status,'logs_sha256':{s:h(B/(phase+'.'+s+'.log'))for s in ['stdout','stderr']},'boundary':'one actual parent wait4; parent itself has no additional supervisor exit receipt; no real camera diagnostic command launched'}
(B/(phase+'-parent.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(code)
