"""One real parent wait for bounded follow-up children; configured route retained."""
import datetime,hashlib,json,os,platform,resource,subprocess,sys,time
from pathlib import Path
B=Path('results/0003-network-followup-20261017');phase=sys.argv[1];source=sys.argv[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(B/(phase+'-parent.json')).exists();(B/(phase+'-source.py')).write_bytes(Path(source).read_bytes())
archive=B/'followup-parent-source.py'
if archive.exists():assert archive.read_bytes()==Path(__file__).read_bytes()
else:archive.write_bytes(Path(__file__).read_bytes())
command=['timeout','900','python',source];env=os.environ.copy();env.update(dict.fromkeys(['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'],'2'));env['PYTHONPATH']='src'
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
r={'phase':phase,'command':command,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':sha(source),'parent_source_sha256':sha(__file__),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':'default runtime route/CA unchanged, only numeric vars=2/PYTHONPATH=src; no values/credentials logged','numeric_variable_names':['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'],'numeric_variable_value':'2','AS_bytes':6*1024**3,'affinity':sorted(os.sched_getaffinity(0))[:2],'python':platform.python_version(),'platform':platform.platform(),'timeout_seconds':900,'boundary':'one parent wait; parent own exit has no further supervisor receipt'};start=time.perf_counter()
with(B/(phase+'.stdout.log')).open('x')as out,(B/(phase+'.stderr.log')).open('x')as err:
 p=subprocess.Popen(command,env=env,preexec_fn=limits,stdout=out,stderr=err);_,status,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status)
r.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,child_exit_code=p.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,cpu_user_seconds=u.ru_utime,cpu_system_seconds=u.ru_stime,log_sha256={s:sha(B/(phase+'.'+s+'.log'))for s in ['stdout','stderr']});(B/(phase+'-parent.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if p.returncode==0 else 1)
