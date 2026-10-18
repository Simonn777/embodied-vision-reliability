"""Compact existing observation plus bounded checkpoint filename metadata search."""
import json,os,time
from pathlib import Path
OUT=Path(__file__).parent;root=Path('/workspace/scratch/a349d03a0e0a/bootstapir-cache')
start=time.perf_counter();stack=[(root,0)];count=0;matches=[];truncated=False
while stack:
 directory,depth=stack.pop()
 with os.scandir(directory)as entries:
  for e in entries:
   count+=1
   if count>2000 or time.perf_counter()-start>3:truncated=True;break
   if e.name=='tapnextpp_ckpt.pt':matches.append({'path':e.path,'bytes':e.stat(follow_symlinks=False).st_size})
   if e.is_dir(follow_symlinks=False)and depth<4 and e.name not in ['runtime','runtime-fixed','.git','__pycache__']:stack.append((Path(e.path),depth+1))
 if truncated:break
r=json.loads((OUT/'record.json').read_text());execution=json.loads((OUT/'execution.json').read_text())
summary={'python':r['python'],'packages':r['import_report']['packages'],'resources':r['resources'],'cached_objects':r['cached_objects'],'source_files':r['source_files'],'tapnextpp_checkpoint_filename_search':{'scope':'existing project cache only; no symlink traversal; runtime/runtime-fixed/Git caches excluded','max_depth':4,'max_entries':2000,'timeout_seconds':3,'entries_seen':count,'truncated':truncated,'elapsed_seconds':time.perf_counter()-start,'matches':matches},'download_ledger_references':[dict({k:v for k,v in item.items()if k!='record'},**({'recorded_0002_acquisition_ceiling_bytes':item['record']['updated_research_acquisition_ceiling_bytes']}if 'record'in item and 'updated_research_acquisition_ceiling_bytes'in item['record']else{}))for item in r['download_ledgers']],'ledger_boundary':'0002 setup/recovery ledgers were read. Global project acquisition total, including historical 0003, was not recomputed; cache byte size is not a transfer total.','command':execution['command'],'cwd':execution['cwd'],'import_exit_code':execution['exit_code'],'raw_wait_status':execution['raw_wait_status'],'import_wall_seconds':execution['wall_seconds'],'total_preflight_seconds':r['elapsed_seconds'],'peak_rss_kib':execution['peak_rss_kib'],'stderr_empty':(OUT/'stderr.log').stat().st_size==0,'scientific_processes_seen':r['visible_scientific_processes'],'models_constructed':0,'forward_calls':0,'downloads':0,'installs':0,'future_inference_feasibility':'UNTESTED'}
(OUT/'compact.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'python':r['python']['version'],'packages':{n:{k:v for k,v in p.items()if k in ['metadata_version','metadata_status','import_status','module_version','threads','interop_threads','build_cuda']}for n,p in r['import_report']['packages'].items()},'memory_limit_bytes':int(r['resources']['memory.max']),'free_disk_bytes':r['resources']['disk']['free_bytes'],'checkpoint_search':summary['tapnextpp_checkpoint_filename_search'],'requested_sources_present':sum(p['exists']for p in r['source_files']),'import_exit_code':execution['exit_code'],'raw_wait_status':execution['raw_wait_status'],'import_wall_seconds':execution['wall_seconds'],'total_preflight_seconds':r['elapsed_seconds'],'metadata_followup_seconds':time.perf_counter()-start},indent=2))
