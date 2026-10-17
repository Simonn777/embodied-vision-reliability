"""One actual wait4 per synthetic-only phase; no recursive supervisor."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path('results/0003-fortran-stageA-20261017')
phase=sys.argv[1];command=sys.argv[2:]
paths=['protocols/0003-fortran-audit-v2.md','configs/0003-fortran-audit-v2.json','src/evr/point_geometry_v2.py','scripts/audit_0003_fortran_v2.py','tests/test_point_geometry_v2.py','results/0003-fortran-stageA-20261017/source-convention.json','protocols/0003-design-freeze-v1.json','protocols/0003-timing-addendum-freeze-v1.json','protocols/0003-resource-addendum-freeze-v1.json']
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in paths:
 dest=B/(phase+'-source')/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes())
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src'}
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='2'
def limit():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.perf_counter()
with(B/(phase+'.stdout.log')).open('xb')as out,(B/(phase+'.stderr.log')).open('xb')as err:
 p=subprocess.Popen(['timeout','900',*command],env=env,stdout=out,stderr=err,preexec_fn=limit)
 _,status,usage=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status);p.returncode=code
r={'phase':phase,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':{p:h(p)for p in paths},'parent_source_sha256':h(__file__),'command':['timeout','900',*command], 'env':env,'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-clock,'observed_exit_code':code,'raw_wait_status':status,'peak_rss_kib':usage.ru_maxrss,'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,'limits':{'AS_bytes':6*1024**3,'cpu_slots':2,'timeout':900},'logs_sha256':{st:h(B/(phase+'.'+st+'.log'))for st in ['stdout','stderr']},'boundary':'parent has no further supervisor receipt; synthetic software only'}
(B/(phase+'-parent.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));sys.exit(code)
