"""Serial command observation; frozen runner owns all scientific operations."""
import datetime,hashlib,json,os,re,resource,subprocess,sys,time
from pathlib import Path
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0006-screen';C=Path(__file__).parent;CACHE=B/'conditional-residual-0006'
PY='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python'
FC='7763fee2351bd4d58b30d1b5594c80b03ac7a0b5';FS='1c71bbb85cbce43b054ace50502d949df2a599c2917e28053351e8896c3d9c96'
MAIN='1edf58df657a3694a027f4b08b0cce76a9bfabfb'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def closed(name):
 x=read(C/(name+'-execution.json'));assert x['exit_code']==0 and x.get('validation_passed',True)
 for stream in ['stdout','stderr']:assert sha(C/(name+'.'+stream+'.log'))==x[stream+'_sha256']
 return x
def main():
 stage=sys.argv[1];assert stage in ['tests','acquire','prepare','group-d1','group-d2','group-d3','group-d4','score']
 target=C/(stage+'-execution.json');assert not target.exists(),'Existing operation; do not repeat'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==MAIN
 freeze=read(R/'protocols/0006-stage0-freeze.json');assert sha(R/'protocols/0006-stage0-freeze.json')==FS and len(freeze['files'])==13
 assert subprocess.check_output(['git','show',FC+':protocols/0006-stage0-freeze.json'],cwd=R)==(R/'protocols/0006-stage0-freeze.json').read_bytes()
 for p,h in freeze['files'].items():
  assert sha(R/p)==h
  for ref in ['HEAD:',':',freeze['source_commit']+':']:assert hashlib.sha256(subprocess.check_output(['git','show',ref+p],cwd=R)).hexdigest()==h
 if stage!='tests':closed('tests')
 if stage=='prepare':closed('acquire')
 if stage.startswith('group-'):closed('prepare')
 if stage=='score':
  for n in ['group-d1','group-d2','group-d3','group-d4']:closed(n)
 common=['--cache',str(CACHE),'--runtime-root',str(B/'bootstapir-cache/runtime-fixed'),'--overlay',str(B/'tapnextpp-cache/overlay'),'--freeze-commit',FC]
 if stage=='tests':cmd=[PY,'-W','error','-m','unittest','discover','-s','tests','-p','test_conditional_residual*.py','-v']
 elif stage=='score':cmd=[PY,'scripts/score_conditional_residual_stage0.py','--cache',str(CACHE),'--output',str(C/'scores.json'),'--freeze-commit',FC]
 elif stage.startswith('group-'):cmd=[PY,'scripts/run_conditional_residual_stage0.py','group',*common,'--image-id',stage[-2:]]
 else:cmd=[PY,'scripts/run_conditional_residual_stage0.py',stage,*common]
 env={'PATH':os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','GIT_NO_LAZY_FETCH':'1','PYTHONPATH':str(B/'bootstapir-cache/runtime-fixed')+':'+str(B/'tapnextpp-cache/overlay')}
 network=['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy','SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE']
 for name in network:
  if name in os.environ:env[name]=os.environ[name]
 for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='4'
 os.sched_setaffinity(0,[0,1,2,3]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 x={'stage':stage,'state':'STARTING','execution_main':MAIN,'source_commit':freeze['source_commit'],'freeze_commit':FC,'freeze_sha256':FS,'frozen_inputs':freeze['files'],'observer_sha256':sha(__file__),'command':['timeout','--kill-after=5s','1800',*cmd],'cwd':str(R),'environment':{k:v for k,v in env.items()if k not in network and k!='PATH'},'PATH':'existing runtime PATH omitted','inherited_network_setting_names':[n for n in network if n in env],'network_setting_values':'not recorded','started_utc':utc(),'exit_code':None,'raw_wait_status':None,'limits':{'affinity':sorted(os.sched_getaffinity(0)),'AS_bytes':resource.getrlimit(resource.RLIMIT_AS)[0]},'boundary':'This parent wait4 observes the external timeout/stage process; parent itself has no further durable outer wait receipt.'}
 save(target,x);t=time.perf_counter()
 with(C/(stage+'.stdout.log')).open('xb')as out,(C/(stage+'.stderr.log')).open('xb')as err:
  p=subprocess.Popen(x['command'],cwd=R,env=env,stdout=out,stderr=err,start_new_session=True);_,s,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(s)
 x.update(state='FINISHED',ended_utc=utc(),exit_code=p.returncode,raw_wait_status=s,wall_seconds=time.perf_counter()-t,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,peak_rss_kib=u.ru_maxrss,stdout_sha256=sha(C/(stage+'.stdout.log')),stderr_sha256=sha(C/(stage+'.stderr.log')))
 if stage=='tests':
  log=(C/'tests.stderr.log').read_text();m=re.search(r'Ran (\d+) tests? in ',log);x['tests_observed']=int(m.group(1))if m else None;x['skipped_lines']=sum(' ... skipped 'in line for line in log.splitlines());x['validation_passed']=p.returncode==0 and x['tests_observed']==66 and x['skipped_lines']==0 and not re.search(r'skipped=\d+',log)
 save(target,x);print(json.dumps(x,indent=2),flush=True)
 return p.returncode if p.returncode else (0 if x.get('validation_passed',True)else 1)
if __name__=='__main__':raise SystemExit(main())
