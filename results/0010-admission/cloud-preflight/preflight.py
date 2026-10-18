"""Bounded names/sizes cache lookup and actual imports only. No network."""
import datetime,hashlib,json,os,platform,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PACKAGES=('torch','numpy','torchvision','PIL','ftfy','regex','packaging','tqdm','wcwidth','scipy')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,d):
 with (HERE/name).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def scan():
 roots=[ROOT/n for n in ('bootstapir-cache','tapnextpp-cache','tapnextpp-recovery','conditional-residual-0006')]
 discovered=[]
 for p in ROOT.iterdir():
  if p.is_dir() and not p.is_symlink() and any(n in p.name.lower() for n in ('clip','eurosat','0010')):discovered.append(p)
 roots+=discovered;roots=list(dict.fromkeys(roots));stack=[(p,0) for p in reversed(roots) if p.is_dir()]
 skipped=[];matches=[];assets=[];visited=0;clock=time.monotonic();bounded=False
 skip={'runtime','runtime-fixed','overlay','wheels','__pycache__','.git','site-packages','cases','predictions','references'}
 while stack:
  directory,depth=stack.pop()
  with os.scandir(directory) as entries:
   for e in entries:
    visited+=1
    if visited>5000 or time.monotonic()-clock>15:bounded=True;break
    p=Path(e.path);low=e.name.lower()
    if e.is_symlink():continue
    if e.is_dir(follow_symlinks=False):
     if e.name in skip or depth>=4:skipped.append(str(p.relative_to(ROOT)));continue
     stack.append((p,depth+1));continue
    if not e.is_file(follow_symlinks=False):continue
    size=e.stat(follow_symlinks=False).st_size
    if any(s in low for s in ('clip','eurosat','bpe','0010')) or size in (353976522,94280567):matches.append({'path':str(p.relative_to(ROOT)),'bytes':size})
    if len(assets)<40 and (directory in roots or directory.parent in roots) and p.suffix.lower() in ('.pt','.zip','.gz','.npz'):assets.append({'path':str(p.relative_to(ROOT)),'bytes':size})
   if bounded:break
 for m in matches:
  p=ROOT/m['path'];low=p.name.lower()
  if m['bytes'] in (353976522,94280567) or ('bpe' in low and m['bytes']<10*1024**2) or ('clip' in low and m['bytes']<2*1024**2):m['sha256']=sha(p)
 return {'roots':[str(p.relative_to(ROOT)) for p in roots],'max_depth':4,'max_entries':5000,'max_metadata_seconds':15,'visited_entries':visited,'limit_hit':bounded,'skipped_directories':skipped,'plausible_named_or_sized_matches':matches,'top_level_existing_asset_names_sizes_first40':assets,'coverage':'Bounded listed project caches only; package trees and old case/reference/weight folders pruned. Not a search of all workspace or user/home caches. Exact hashes only plausible names/sizes; no payload decode.','expected_assets':{'CLIP_ViT_B32':{'bytes':353976522,'sha256':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'},'EuroSAT.zip':{'bytes':94280567,'sha256':'8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd'}}}
def main():
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal():continue
  try:a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except OSError:continue
  scripts=[Path(x).name for x in a if '\n' not in x and x.endswith('.py')]
  names=[x for x in scripts if x in {'supervise_0009.py','run_geometric_stage0.py','run_antithetic_training.py','run_antithetic_gradient_screen.py','run_conditional_residual_stage0.py'} or ('0010' in x and x!='preflight.py')]
  if names:active.append({'pid':int(p.name),'scripts':names})
 assert not active,active
 metadata=scan();save('cache-search.json',metadata)
 imports={}
 for name in PACKAGES:
  cmd=['timeout','--kill-after=1s','12',sys.executable,str(HERE/'import_probe.py'),name]
  started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
  with (HERE/(name+'.stdout.log')).open('xb') as out,(HERE/(name+'.stderr.log')).open('xb') as err:
   child=subprocess.Popen(cmd,stdout=out,stderr=err);_,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status);out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
  result={'command':cmd,'actual_exit':child.returncode,'raw_wait_status':status,'wall_seconds':time.monotonic()-clock,'peak_rss_kib':usage.ru_maxrss,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'utc_started':started,'stdout_sha256':sha(HERE/(name+'.stdout.log')),'stderr_sha256':sha(HERE/(name+'.stderr.log'))}
  try:result['import']=json.loads((HERE/(name+'.stdout.log')).read_text())
  except Exception:result['import']={'status':'no_complete_JSON'}
  imports[name]=result
 env_names=('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy','SSL_CERT_FILE','REQUESTS_CA_BUNDLE')
 d={'scope':'READ_ONLY_IMPORT_AND_CACHE_QUALIFICATION; no model/tensor/data decode/science/network/install','status':'completed','existing_preflight_before_claim':False,'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),'address_space_limit':list(resource.getrlimit(resource.RLIMIT_AS)),'cgroup_memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'cgroup_memory_current':Path('/sys/fs/cgroup/memory.current').read_text().strip(),'cpu_model':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'disk_free_bytes':__import__('shutil').disk_usage(ROOT).free,'PYTHONPATH':os.environ['PYTHONPATH'],'path_order_evidence':{'path':'research-0005-recovery/scripts/prepare_tapnext_recovery.py','line':298,'sha256':sha(ROOT/'research-0005-recovery/scripts/prepare_tapnext_recovery.py'),'interpretation':'Existing approved overlay, old runtime-fixed, old runtime, recovery source in that order.'},'imports':imports,'cache_search_sha256':sha(HERE/'cache-search.json'),'environment_status_tool':'unavailable in active tool inventory','network_setting_presence_only':{k:bool(os.environ.get(k)) for k in env_names},'network_policy':'No network request attempted. Current context lists allowed hosts; connectivity not probed. No credential/environment values read or printed.','new_asset_bytes':0,'script_sha256':sha(__file__),'import_probe_sha256':sha(HERE/'import_probe.py')}
 save('record.json',d)
 print(json.dumps({'status':'completed','imports':{n:{'exit':v['actual_exit'],**v['import']} for n,v in imports.items()},'metadata_entries':metadata['visited_entries'],'cache_match_count':len(metadata['plausible_named_or_sized_matches']),'search_limit_hit':metadata['limit_hit'],'record_sha256':sha(HERE/'record.json')},sort_keys=True))
if __name__=='__main__':main()
