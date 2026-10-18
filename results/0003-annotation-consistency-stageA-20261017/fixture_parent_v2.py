"""Source-bound synthetic suite only; never invoke real geometry main or cache."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path('results/0003-annotation-consistency-stageA-20261017');phase=sys.argv[1]
paths=['src/evr/__init__.py','src/evr/annotation_consistency_v1.py','scripts/audit_0003_annotation_consistency_v1.py','scripts/run_0003_annotation_consistency_bounded_v1.py','tests/test_annotation_consistency_v1.py','protocols/0003-annotation-consistency-v1.md','configs/0003-annotation-consistency-v1.json']
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sources={p:h(p)for p in paths}
for p in paths:
 q=B/(phase+'-source')/p;q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('xb')as f:f.write(Path(p).read_bytes())
command=['timeout','900','python','-m','unittest','discover','-s','tests','-p','test_annotation_consistency_v1.py','-v']
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src','GIT_NO_LAZY_FETCH':'1'}
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limit():resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with(B/(phase+'.stdout.log')).open('xb')as out,(B/(phase+'.stderr.log')).open('xb')as err:
 p=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=limit);_,status,u=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status);p.returncode=code
r={'phase':phase,'command':command,'execution_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'peak_rss_kib':u.ru_maxrss,'CPU_user_seconds':u.ru_utime,'CPU_system_seconds':u.ru_stime,'observed_exit_code':code,'raw_wait_status':status,'source_sha256':{**sources,str(Path(__file__).resolve().relative_to(Path.cwd())):h(__file__)},'logs_sha256':{st:h(B/(phase+'.'+st+'.log'))for st in ['stdout','stderr']},'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'AS_bytes':6*1024**3,'affinity':affinity,'timeout_seconds':900,'scope':'synthetic-only; zero data archive/NPZ/GT/model/projection on real data','boundary':'actual parent wait4 of synthetic suite; parent has no further observed exit receipt'}
with(B/(phase+'-parent.json')).open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));raise SystemExit(code)
