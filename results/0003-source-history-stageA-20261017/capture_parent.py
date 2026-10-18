"""One actual wait4 per bounded source audit or synthetic test child; no recursive supervisor."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path('results/0003-source-history-stageA-20261017');phase=sys.argv[1];command=sys.argv[2:]
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=['src/evr/point_source_history_v1.py','scripts/audit_0003_source_history_v1.py','tests/test_point_source_history_v1.py','protocols/0003-annotation-consistency-candidate-v1.md','configs/0003-annotation-consistency-candidate-v1.json']
source={p:h(p)for p in inputs};source[str(Path(__file__).resolve().relative_to(Path.cwd()))]=h(__file__)
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src','GIT_NO_LAZY_FETCH':'1'}
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limit():resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with(B/(phase+'.stdout.log')).open('xb')as out,(B/(phase+'.stderr.log')).open('xb')as err:
 p=subprocess.Popen(['timeout','900',*command],env=env,stdout=out,stderr=err,preexec_fn=limit)
 _,status,u=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status);p.returncode=code
r={'phase':phase,'execution_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'command':['timeout','900',*command],'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'peak_rss_kib':u.ru_maxrss,'CPU_user_seconds':u.ru_utime,'CPU_system_seconds':u.ru_stime,'observed_exit_code':code,'raw_wait_status':status,'source_sha256':source,'logs_sha256':{st:h(B/(phase+'.'+st+'.log'))for st in ['stdout','stderr']},'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'affinity':affinity,'AS_bytes':6*1024**3,'timeout_seconds':900,'scope':'source text/published JSON or synthetic fixtures only; no numeric archive reads or projection','boundary':'actual parent wait4; parent itself has no further observed exit receipt'}
with(B/(phase+'-parent.json')).open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));raise SystemExit(code)
