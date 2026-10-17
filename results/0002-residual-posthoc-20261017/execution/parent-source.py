"""One parent wait4 layer for bounded diagnostic/test/audit children."""
import datetime,fcntl,hashlib,json,os,platform,resource,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
B=Path('results/0002-residual-posthoc-20261017');E=B/'execution';E.mkdir(exist_ok=True)
COMMANDS={'tests':['python','-m','unittest','discover','-s','tests','-p','test_residual_posthoc.py','-v'],'analysis':['python','scripts/analyze_0002_residuals.py','--output',str(B/'analysis')],'verification':['python','scratch/verify_posthoc.py']}
SOURCES={'tests':Path('tests/test_residual_posthoc.py'),'analysis':Path('scripts/analyze_0002_residuals.py'),'verification':Path('scratch/verify_posthoc.py')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
phase=sys.argv[1];command=['timeout','900']+COMMANDS[phase];receipt=E/(phase+'.json');archive=E/'parent-source.py'
if archive.exists():assert archive.read_bytes()==Path(__file__).read_bytes()
else:archive.write_bytes(Path(__file__).read_bytes())
with receipt.open('x')as z:z.write('{}\n')
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src',**dict.fromkeys(['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'],'2')}
cpu=next((x.split(':',1)[1].strip()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('model name')),'unknown')
r={'phase':phase,'status':'starting','started_at_utc':utc(),'command':command,'environment':{k:v for k,v in env.items()if k!='PATH'},'path_policy':'inherited PATH only; no other inherited variables','python':platform.python_version(),'platform':platform.platform(),'cpu_model':cpu,'affinity':sorted(os.sched_getaffinity(0))[:2],'address_space_limit_bytes':6*1024**3,'timeout_seconds':900,'execution_main':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'parent_source_sha256':sha(__file__),'child_source_sha256':sha(SOURCES[phase]),'plan_sha256':sha('docs/analysis/0002-motion-residual-posthoc-plan.md'),'config_sha256':sha('configs/0002-exploratory-v2.json'),'protocol_sha256':sha('protocols/0002-exploratory-frozen-v2.md'),'observation_scope':'one actual parent wait4; parent own exit has no further supervisor receipt'}
def save():receipt.write_text(json.dumps(r,indent=2)+'\n')
save();started=time.perf_counter();locks=[];owned=False;marker=Path('../bootstapir-cache/0002-residual-posthoc-launch.json')
try:
 for name in ['frozen-v2-research.lock','0002-residual-posthoc.lock']:
  z=(Path('../bootstapir-cache')/name).open('a');fcntl.flock(z,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(z)
 if phase=='analysis':
  with marker.open('x')as z:json.dump({'status':'reserved','started_at_utc':r['started_at_utc'],'output':str(B/'analysis')},z)
  owned=True
 with(E/(phase+'.stdout.log')).open('x')as out,(E/(phase+'.stderr.log')).open('x')as err:
  child=subprocess.Popen(command,cwd=ROOT,env=env,preexec_fn=limits,stdout=out,stderr=err);_,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 r.update(status='completed'if child.returncode==0 else'failed',child_exit_code_observed_by_parent=child.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,cpu_user_seconds=u.ru_utime,cpu_system_seconds=u.ru_stime)
except Exception as exc:r.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc().replace(str(ROOT),'<repo>'))
finally:
 r.update(ended_at_utc=utc(),elapsed_seconds=time.perf_counter()-started,log_sha256={k:sha(E/(phase+'.'+k+'.log'))for k in ['stdout','stderr']if(E/(phase+'.'+k+'.log')).exists()});save()
 if owned:marker.write_text(json.dumps({'status':r['status'],'output':str(B/'analysis'),'started_at_utc':r['started_at_utc'],'ended_at_utc':r['ended_at_utc']},indent=2)+'\n')
 for z in locks:z.close()
print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='completed'else 1)
