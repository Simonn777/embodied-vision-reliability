"""Measurement only: invoke frozen tests/run once; never modifies scientific inputs."""
import datetime,fcntl,hashlib,json,os,platform,resource,subprocess,sys,time
from pathlib import Path

OUT=Path('results/0004-stage0');CACHE=Path('../bootstapir-cache')
PYTHON='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python'
FREEZE='e14bb4f3f4b4dd91c96fe63b1d28bc9a95636d560854de190a701c25dc34667d'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,x):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(path)
def main():
 mode=sys.argv[1];assert mode in ('software','scientific')
 record_path=OUT/(mode+'-child.json')
 if record_path.exists():raise ValueError('exclusive measurement receipt')
 freeze=json.loads(Path('protocols/0004-stage0-freeze.json').read_text())
 assert sha('protocols/0004-stage0-freeze.json')==FREEZE
 assert git('rev-parse','HEAD')=='ddadc5e17a6ceff3c358d3e3f440d80bf3fd0306'
 assert git('rev-parse','origin/main')=='ddadc5e17a6ceff3c358d3e3f440d80bf3fd0306'
 for path,want in freeze['files'].items():
  assert sha(path)==want
  for spec in [freeze['source_commit']+':'+path,'HEAD:'+path,':'+path,'origin/main:'+path]:
   assert hashlib.sha256(subprocess.check_output(['git','show',spec])).hexdigest()==want
 assert len(freeze['files'])==8
 source={p:sha(p)for p in ['results/0004-stage0/launch_wrapper.py','results/0004-stage0/launch_supervisor.py']}
 env={'PATH':os.environ['PATH'],'PYTHONPATH':'../bootstapir-cache/runtime-fixed:../bootstapir-cache/runtime:../bootstapir-cache/source:src','PYTHONDONTWRITEBYTECODE':'1','GIT_NO_LAZY_FETCH':'1'}
 for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
 affinity=sorted(os.sched_getaffinity(0))[:2];assert len(affinity)==2
 def limits():
  os.sched_setaffinity(0,affinity);resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
 if mode=='software':
  baseline={p:sha(p)for p in git('ls-files').splitlines()}
  save(OUT/'tracked-baseline.json',baseline)
  command=['timeout','--kill-after=5s','900',PYTHON,'-m','unittest','discover','-s','tests','-p','test_state_intervention*.py','-v']
 else:
  test=json.loads((OUT/'software-child.json').read_text());assert test['exit_code']==0
  lock=(CACHE/'0004-stage0.launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not(CACHE/'0004-stage0.started.json').exists()
  assert not(OUT/'run').exists()
  config=json.loads(Path('configs/0004-stage0.json').read_text())
  assert sha(CACHE/'weights.pt')==config['checkpoint_sha256']
  assert sha(CACHE/'development-000.npz')==config['development_npz_sha256']
  command=['timeout','--kill-after=5s','900',PYTHON,'scripts/run_state_intervention.py','--cache',str(CACHE),'--output',str(OUT/'run'),'--freeze-sha256',FREEZE]
 record={'mode':mode,'state':'STARTING','command':command,'environment':{**env,'PATH':'inherited runtime PATH; value omitted'},'execution_commit':git('rev-parse','HEAD'),'source_commit':freeze['source_commit'],'freeze_sha256':FREEZE,'frozen_inputs_sha256':freeze['files'],'outer_source_sha256':source,'limits':{'affinity':affinity,'AS_bytes':6*1024**3,'timeout_seconds':900,'kill_after_seconds':5},'python':platform.python_version(),'platform':platform.platform(),'started_utc':now(),'exit_code':None,'raw_wait_status':None}
 save(record_path,record);start=time.perf_counter()
 with(OUT/(mode+'.stdout.log')).open('xb')as out,(OUT/(mode+'.stderr.log')).open('xb')as err:
  child=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=limits,start_new_session=True)
  _,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 record.update(state='FINISHED',ended_utc=now(),wall_seconds=time.perf_counter()-start,exit_code=child.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,logs_sha256={s:sha(OUT/(mode+'.'+s+'.log'))for s in ['stdout','stderr']})
 if mode=='scientific'and(CACHE/'0004-stage0.started.json').exists():record['persistent_marker_sha256']=sha(CACHE/'0004-stage0.started.json')
 save(record_path,record);print(json.dumps(record,indent=2));return child.returncode
if __name__=='__main__':raise SystemExit(main())
