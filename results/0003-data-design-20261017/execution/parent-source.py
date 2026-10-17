import datetime,fcntl,hashlib,json,os,platform,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);B=Path('results/0003-data-design-20261017');E=B/'execution';E.mkdir(exist_ok=True)
phase=sys.argv[1];command=sys.argv[2:];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
locks=[]
for name in ['frozen-v2-research.lock','0003-design.lock']:
 p=(Path('../bootstapir-cache')/name).open('a');fcntl.flock(p,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(p)
archive=E/'parent-source.py'
if archive.exists():assert archive.read_bytes()==Path(__file__).read_bytes()
else:archive.write_bytes(Path(__file__).read_bytes())
receipt=E/(phase+'.json')
with receipt.open('x')as f:f.write('{}\n')
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src',**dict.fromkeys(['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'],'2')}
r={'phase':phase,'command':['timeout','900']+command,'environment':{k:v for k,v in env.items()if k!='PATH'},'path':'inherited PATH only','started_utc':utc(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'child_source_sha256':sha(command[1])if len(command)>1 and Path(command[1]).is_file()else None,'parent_source_sha256':sha(__file__),'python':platform.python_version(),'platform':platform.platform(),'cpu_model':next(x.split(':',1)[1].strip()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('model name')),'affinity':sorted(os.sched_getaffinity(0))[:2],'AS_bytes':6*1024**3,'timeout_seconds':900,'boundary':'actual parent wait4 observes child; parent own exit has no further supervisor receipt'}
start=time.perf_counter()
try:
 with(E/(phase+'.stdout.log')).open('x')as out,(E/(phase+'.stderr.log')).open('x')as err:
  p=subprocess.Popen(r['command'],env=env,preexec_fn=limits,stdout=out,stderr=err);_,raw,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(raw)
 r.update(exit_code=p.returncode,raw_wait_status=raw,peak_rss_kib=u.ru_maxrss,cpu_user_seconds=u.ru_utime,cpu_system_seconds=u.ru_stime,status='completed'if p.returncode==0 else'failed')
except Exception as exc:r.update(status='failed',error_type=type(exc).__name__,error=str(exc).replace(str(ROOT),'<repo>'))
finally:
 r.update(ended_utc=utc(),elapsed_seconds=time.perf_counter()-start,log_sha256={s:sha(E/(phase+'.'+s+'.log'))for s in ['stdout','stderr']if(E/(phase+'.'+s+'.log')).exists()});receipt.write_text(json.dumps(r,indent=2)+'\n')
 for f in locks:f.close()
print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='completed'else 1)
