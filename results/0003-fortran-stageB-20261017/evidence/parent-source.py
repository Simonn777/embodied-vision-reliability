"""One wait4 layer for exact approved commands. No retries or scientific changes."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path('results/0003-fortran-stageB-20261017/evidence');phase=sys.argv[1];command=sys.argv[2:]
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt='protocols/0003-fortran-audit-freeze-v2.json';r=json.loads(Path(receipt).read_text())
paths=list(r['inputs_sha256'])+[receipt]
source={p:h(p)for p in paths}
for i,p in enumerate(command):
 if p.endswith('.py')and Path(p).is_file():
  dest=B/(phase+'-command-source-'+str(i)+'.py');dest.write_bytes(Path(p).read_bytes());source[str(dest)]=h(p)
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src'}
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limit():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with(B/(phase+'.stdout.log')).open('xb')as out,(B/(phase+'.stderr.log')).open('xb')as err:
 p=subprocess.Popen(['timeout','900',*command],env=env,stdout=out,stderr=err,preexec_fn=limit)
 _,status,u=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(status);p.returncode=code
v={'phase':phase,'command':['timeout','900',*command],'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'input_commit':r['input_commit'],'source_sha256':source,'parent_sha256':h(__file__),'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'peak_rss_kib':u.ru_maxrss,'cpu_user_seconds':u.ru_utime,'cpu_system_seconds':u.ru_stime,'raw_wait_status':status,'observed_exit_code':code,'affinity':affinity,'AS_bytes':6*1024**3,'timeout_seconds':900,'logs_sha256':{st:h(B/(phase+'.'+st+'.log'))for st in ['stdout','stderr']},'boundary':'one parent actual wait; parent itself has no additional supervisor exit receipt; source command is archived before launch'}
(B/(phase+'-parent.json')).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2));sys.exit(code)
