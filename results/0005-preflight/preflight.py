"""Metadata and import-only preparation; no model construction, data decode or acquisition."""
import datetime,hashlib,json,os,platform,resource,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path('/workspace/scratch/a349d03a0e0a');CACHE=ROOT/'bootstapir-cache';OUT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 start=time.perf_counter();record={'started_utc':now(),'scope':'0005 import/resource metadata only','model_constructed':False,'forward_calls':0,'data_or_GT_decoded':False,'new_download_bytes':0,'new_installs':0,'python':{'path':sys.executable,'version':platform.python_version()},'platform':platform.platform()}
 active=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit()or int(d.name)==os.getpid():continue
  try:
   names=[Path(v.decode(errors='replace')).name for v in(d/'cmdline').read_bytes().split(b'\0')if v]
   hits=[n for n in names if n.endswith('.py')and(n.startswith(('run_','audit_','measure_','infer_','experiment'))or'state_intervention'in n)]
   if hits:active.append({'pid':int(d.name),'script_basenames':hits})
  except OSError:pass
 record['visible_scientific_processes']=active
 if active:raise RuntimeError('Active scientific process; no import probe started')
 record['source_files']=[]
 for rel in ['tapnet/tapnext/tapnext_torch.py','tapnet/tapnext/tapnext_lru_modules.py','tapnet/tapnext/pscan.py','tapnextpp/votsp2026/model.py','tapnet/tapnextpp/votsp2026/model.py']:
  p=CACHE/'source'/rel;entry={'relative_source_path':rel,'exists':p.is_file()}
  if p.is_file():entry.update(bytes=p.stat().st_size,sha256=sha(p))
  record['source_files'].append(entry)
 record['cached_objects']=[]
 with os.scandir(CACHE)as it:
  for i,e in enumerate(it):
   if i>=128:record['cache_direct_listing_truncated']=True;break
   if e.is_file(follow_symlinks=False)and any(k in e.name.lower()for k in ['weights','rgb.zip','development-000','tapnext','sample.tar','pointodyssey']):
    record['cached_objects'].append({'path':str(Path(e.path)),'bytes':e.stat(follow_symlinks=False).st_size,'kind':'existing complete or partial cache object; byte length is not cumulative acquisition'})
 record['tapnextpp_checkpoint_exists']=(CACHE/'tapnextpp_ckpt.pt').is_file()
 record['download_ledgers']=[]
 for p in [CACHE/'transfer-ledger.json',CACHE/'acquisition-ledger.json',ROOT/'research-0004/results/0002-development/transfer-ledger.json',ROOT/'research-0004/results/0002-pilot-execution/g-033-20261017-recovery/acquisition-summary.json']:
  item={'path':str(p),'exists':p.is_file()}
  if p.is_file():
   assert p.stat().st_size<65536
   item.update(bytes=p.stat().st_size,sha256=sha(p),record=json.loads(p.read_text()))
  record['download_ledgers'].append(item)
 record['resources']={'available_affinity':sorted(os.sched_getaffinity(0)),'memory':{s.split(':',1)[0]:s.split(':',1)[1].strip()for s in Path('/proc/meminfo').read_text().splitlines()if s.split(':',1)[0]in ['MemTotal','MemAvailable']},'disk':dict(zip(['total_bytes','used_bytes','free_bytes'],shutil.disk_usage(CACHE)))}
 for name in ['memory.max','memory.current','cpu.max']:
  p=Path('/sys/fs/cgroup')/name
  if p.is_file():record['resources'][name]=p.read_text().strip()
 probe="""import hashlib,importlib,importlib.metadata as m,json,platform,sys
from pathlib import Path
r={'python_path':sys.executable,'python_version':platform.python_version(),'packages':{}}
for name in ['torch','torchvision','einops','numpy']:
 item={};r['packages'][name]=item
 try:
  d=m.distribution(name);p=Path(d._path);item.update(metadata_version=d.version,dist_info=str(p),metadata_sha256=hashlib.sha256((p/'METADATA').read_bytes()).hexdigest())
 except m.PackageNotFoundError:item.update(metadata_status='MISSING');continue
 try:
  x=importlib.import_module(name)
  if name=='torch':x.set_num_threads(2);x.set_num_interop_threads(2);item.update(threads=x.get_num_threads(),interop_threads=x.get_num_interop_threads(),build_cuda=x.version.cuda)
  p=Path(x.__file__);item.update(import_status='OK',module_version=getattr(x,'__version__',None),module_path=str(p),module_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 except Exception as e:item.update(import_status='FAILED',error_type=type(e).__name__,error_message=str(e))
print(json.dumps(r,indent=2))
"""
 (OUT/'import_probe.py').write_text(probe)
 env={'PATH':os.environ['PATH'],'PYTHONPATH':'../bootstapir-cache/runtime-fixed:../bootstapir-cache/runtime:../bootstapir-cache/source:src','PYTHONDONTWRITEBYTECODE':'1'}
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='2'
 affinity=sorted(os.sched_getaffinity(0))[:2]
 def limits():os.sched_setaffinity(0,affinity);resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
 command=['timeout','--kill-after=5s','30',sys.executable,str(OUT/'import_probe.py')]
 execution={'command':command,'cwd':str(ROOT/'research-0004'),'environment':{**env,'PATH':'existing runtime PATH value omitted'},'affinity':affinity,'AS_bytes':6*1024**3,'timeout_seconds':30,'source_sha256':sha(Path(__file__)),'import_source_sha256':sha(OUT/'import_probe.py'),'started_utc':now(),'exit_code':None,'raw_wait_status':None,'boundary':'one real parent wait4 for import child; parent has no further durable exit receipt'}
 (OUT/'pre-execution.json').write_text(json.dumps(execution,indent=2)+'\n');t=time.perf_counter()
 with(OUT/'stdout.log').open('xb')as stdout,(OUT/'stderr.log').open('xb')as stderr:
  child=subprocess.Popen(command,cwd=ROOT/'research-0004',env=env,stdout=stdout,stderr=stderr,preexec_fn=limits)
  _,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 execution.update(ended_utc=now(),wall_seconds=time.perf_counter()-t,exit_code=child.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,stdout_sha256=sha(OUT/'stdout.log'),stderr_sha256=sha(OUT/'stderr.log'))
 if child.returncode==0:record['import_report']=json.loads((OUT/'stdout.log').read_text())
 record.update(ended_utc=now(),elapsed_seconds=time.perf_counter()-start,import_exit_code=child.returncode,interpretation='Environment preparation only; future TAPNext++ model/forward feasibility remains untested')
 (OUT/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');(OUT/'record.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2));print('EXECUTION:');print(json.dumps(execution,indent=2));print('STDERR:');print((OUT/'stderr.log').read_text());return child.returncode
if __name__=='__main__':raise SystemExit(main())
