"""Observe each frozen stage exactly once; no scientific implementation changes."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path('/workspace/scratch/a349d03a0e0a/research-0005')
OUT=Path(__file__).parent;ASSET=ROOT.parent/'tapnextpp-cache';OLD=ROOT.parent/'bootstapir-cache'
OUTPUT=ROOT.parent/'scratch/0005-native-capability'
PYTHON='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python'
FREEZE='3e5aad0b0123affb3991997607c04310bd7d08a38172b8ffbc2c0f29b01f36ef'
CONFIG='37faa3b23182c5fe2a0854bb377b5ca29bec2bffb0031e00f8fd3b090501fc1c'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
 mode=sys.argv[1];assert mode in ['verify','dependencies','checkpoint','native','software']
 target=OUT/(mode+'-execution.json');assert not target.exists(),'stage receipt exists; never retry'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='e168ba830f9f0eff37192ee8745207ff7d3484ef'
 freeze=json.loads((ROOT/'protocols/0005-native-capability-freeze.json').read_text())
 assert sha(ROOT/'protocols/0005-native-capability-freeze.json')==FREEZE and len(freeze['files'])==7
 for p,h in freeze['files'].items():assert sha(ROOT/p)==h
 if mode=='verify':
  paths=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
  save(OUT/'tracked-baseline.json',{p:sha(ROOT/p)for p in paths})
  save(OUT/'old-cache-markers.json',{p.name:sha(p)for p in OLD.iterdir()if p.is_file()and('started'in p.name or'launch'in p.name)and p.suffix=='.json'})
  assert not ASSET.exists()and not OUTPUT.exists()
 if mode in ['dependencies','checkpoint','native']:
  assert json.loads((OUT/'verify-execution.json').read_text())['exit_code']==0
 if mode in ['checkpoint','native']:assert json.loads((OUT/'dependencies-execution.json').read_text())['exit_code']==0
 if mode=='native':assert json.loads((OUT/'checkpoint-execution.json').read_text())['exit_code']==0
 if mode in ['verify','native']:
  command=['timeout','--kill-after=5s','1800',PYTHON,'scripts/run_tapnext_capability.py','--asset-cache',str(ASSET),'--bootstapir-cache',str(OLD),'--output',str(OUTPUT),'--freeze-sha256',FREEZE]
  if mode=='verify':command.append('--verify-only')
 elif mode=='software':command=['timeout','--kill-after=5s','1800',PYTHON,'-m','unittest','discover','-s','tests','-p','test_tapnext*.py','-v']
 else:command=['timeout','--kill-after=5s','1800',PYTHON,'scripts/prepare_tapnext_assets.py','--config','configs/0005-native-capability.json','--config-sha256',CONFIG,'--old-cache',str(OLD),'--new-cache',str(ASSET),'--stage',mode,'--repository',str(ROOT)]
 # Preserve only the already configured public-HTTP route and CA settings; values
 # never enter receipts. No proxy/CA address is invented or security setting changed.
 env={'PATH':os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','GIT_NO_LAZY_FETCH':'1'}
 network_names=['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy','SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE']
 for n in network_names:
  if n in os.environ:env[n]=os.environ[n]
 for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='4'
 # The fixed scientific supervisor/preparer independently establishes its own
 # import path. Synthetic tests need only the old exact NumPy and project code.
 if mode=='software':env['PYTHONPATH']=str(OLD/'runtime-fixed')+':'+str(OLD/'runtime')+':'+str(ROOT/'src')
 recorded_env={k:v for k,v in env.items()if k not in network_names and k!='PATH'};recorded_env['PATH']='existing runtime PATH omitted'
 r={'stage':mode,'state':'STARTING','command':command,'cwd':'research-0005','environment':recorded_env,'inherited_network_setting_names':[n for n in network_names if n in env],'network_setting_values':'not recorded','source_sha256':sha(__file__),'frozen_input_sha256':freeze['files'],'freeze_sha256':FREEZE,'execution_main':'e168ba830f9f0eff37192ee8745207ff7d3484ef','started_utc':now(),'exit_code':None,'raw_wait_status':None,'boundary':'parent observes external timeout/stage process once; no further durable parent wait4 receipt'}
 save(target,r);t=time.perf_counter()
 with(OUT/(mode+'.stdout.log')).open('xb')as stdout,(OUT/(mode+'.stderr.log')).open('xb')as stderr:
  child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  _,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 r.update(state='FINISHED',ended_utc=now(),wall_seconds=time.perf_counter()-t,exit_code=child.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,stdout_sha256=sha(OUT/(mode+'.stdout.log')),stderr_sha256=sha(OUT/(mode+'.stderr.log')))
 save(target,r);print(json.dumps(r,indent=2));return child.returncode if child.returncode>=0 else 1
if __name__=='__main__':raise SystemExit(main())
