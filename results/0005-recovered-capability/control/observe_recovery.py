"""One serial external observer. No scientific logic or runtime repair."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0005-recovery';C=Path(__file__).parent
PY='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python'
FREEZE='916cdabbdc8d5be7dcb0060ca54216457d982a3bdc564c6509c79195e51966ef'
PREPARER='63d5cccc68f380ceeaaea67c673cf43bffecb0f385d2879c398eaf9fa4f044e6'
CONFIG='97bb7e18bddf6c979d97ca825769fb37ab2175aa60608e84c3783f9ab65a257c'
SUPERVISOR='37356e4b87140e16d5c6431a10de617cd303963e25b70f3c99398aa0d800d7ec'
MAIN='f0787ab50f5dfc31f8b8a58bdcb165a12d2d83a6'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def closed(name):
 x=json.loads((C/(name+'-execution.json')).read_bytes());assert x['exit_code']==0 and x['raw_wait_status']==0
 for s in ['stdout','stderr']:assert sha(C/(name+'.'+s+'.log'))==x[s+'_sha256']
 return x
def execute(name,command):
 target=C/(name+'-execution.json');assert not target.exists(),'existing attempt: never retry'
 env={'PATH':os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','GIT_NO_LAZY_FETCH':'1'}
 nv=['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy','SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE']
 for n in nv:
  if n in os.environ:env[n]=os.environ[n]
 for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='4'
 if name=='software':env['PYTHONPATH']=str(B/'bootstapir-cache/runtime-fixed')+':'+str(B/'bootstapir-cache/runtime')+':'+str(R/'src')
 rec={'stage':name,'state':'STARTING','execution_main':MAIN,'command':['timeout','--kill-after=5s','1800',PY,*command], 'cwd':str(R),'environment':{k:v for k,v in env.items()if k not in nv and k!='PATH'},'inherited_network_setting_names':[n for n in nv if n in env],'network_setting_values':'not recorded','PATH':'existing runtime PATH omitted','observer_source_sha256':sha(__file__),'freeze_sha256':FREEZE,'source_sha256':{'supervisor':sha(R/'scripts/run_tapnext_recovered_capability.py'),'preparer':sha(R/'scripts/prepare_tapnext_recovery.py'),'config':sha(R/'configs/0005-native-recovery.json')},'started_utc':utc(),'exit_code':None,'raw_wait_status':None,'limits':{'affinity':sorted(os.sched_getaffinity(0)),'AS_bytes':resource.getrlimit(resource.RLIMIT_AS)[0]},'boundary':'This parent wait4 observes the external timeout command once. The parent itself has no additional durable outer wait4 receipt.'}
 save(target,rec);t=time.perf_counter()
 with(C/(name+'.stdout.log')).open('xb')as out,(C/(name+'.stderr.log')).open('xb')as err:
  p=subprocess.Popen(rec['command'],cwd=R,env=env,stdout=out,stderr=err,start_new_session=True)
  _,status,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status)
 rec.update(state='FINISHED',ended_utc=utc(),exit_code=p.returncode,raw_wait_status=status,wall_seconds=time.perf_counter()-t,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,peak_rss_kib=u.ru_maxrss,stdout_sha256=sha(C/(name+'.stdout.log')),stderr_sha256=sha(C/(name+'.stderr.log')))
 save(target,rec);print(json.dumps(rec,indent=2),flush=True);return p.returncode
def main():
 mode=sys.argv[1];assert mode in ['dependencies','checkpoint','native','software']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==MAIN
 assert sha(R/'scripts/run_tapnext_recovered_capability.py')==SUPERVISOR
 os.sched_setaffinity(0,[0,1,2,3]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 common=['scripts/run_tapnext_recovered_capability.py','--failed-cache','../tapnextpp-cache','--recovery-root','../tapnextpp-recovery','--bootstapir-cache','../bootstapir-cache','--output',str(B/'scratch/0005-recovered-native-capability'),'--freeze-sha256',FREEZE]
 if mode in ['dependencies','checkpoint']:
  if mode=='checkpoint':
   closed('dependencies');assert json.loads((C/'dependencies-audit.json').read_bytes())['status']=='passed'
  code=execute('verify-before-'+mode,[*common,'--verify-only'])
  if code:return code
  assert sha(R/'scripts/prepare_tapnext_recovery.py')==PREPARER
  return execute(mode,['scripts/prepare_tapnext_recovery.py','--stage',mode,'--config','configs/0005-native-recovery.json','--config-sha256',CONFIG,'--old-cache','../bootstapir-cache','--failed-cache','../tapnextpp-cache','--recovery-root','../tapnextpp-recovery','--repository',str(R)])
 if mode=='native':
  closed('dependencies');closed('checkpoint')
  assert json.loads((C/'checkpoint-audit.json').read_bytes())['status']=='passed'
  return execute(mode,common)
 return execute(mode,['-m','unittest','discover','-s','tests','-p','test_tapnext_recovery.py','-v'])
if __name__=='__main__':raise SystemExit(main())
