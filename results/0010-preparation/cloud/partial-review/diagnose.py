"""Read-only cached-byte and provenance diagnosis; no deserialization or repair."""
import hashlib,json,os,stat,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0010-capability';CONTROL=ROOT/'scratch/0010-capability-control'
def snapshot(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
 return {'dev':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def digest(p):
 h=hashlib.sha256()
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
paths={'final':ROOT/'clip-0010-downloads/ViT-B-32.pt','asset':ROOT/'clip-0010-assets/weights/ViT-B-32.pt','partial':ROOT/'clip-0010-downloads/ViT-B-32.pt.partial'}
before={k:snapshot(p) for k,p in paths.items()};hashes={k:digest(p) for k,p in paths.items()}
assert before['final']['size']==before['asset']['size']==353976522
assert hashes['final']==hashes['asset']=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
assert before['partial']['size']==343605248
compared=0;first=None
with os.fdopen(os.open(paths['partial'],os.O_RDONLY|os.O_NOFOLLOW),'rb') as p,os.fdopen(os.open(paths['final'],os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
 while True:
  b=p.read(1048576)
  if not b:break
  a=f.read(len(b))
  if first is None and a!=b:
   first=compared+next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
  compared+=len(b)
after_reads={k:snapshot(p) for k,p in paths.items()};assert before==after_reads
related=[];holders=[];unreadable_proc_entries=0;fd_entries_checked=0
for d in Path('/proc').iterdir():
 if not d.name.isdecimal():continue
 try:a=[x.decode(errors='replace') for x in (d/'cmdline').read_bytes().split(bytes([0])) if x]
 except OSError:unreadable_proc_entries+=1;continue
 scripts=[x for x in a if '\n' not in x and x.endswith('.py') and ('0010' in x or Path(x).name in ('run_geometric_stage0.py','run_antithetic_training.py','run_antithetic_gradient_screen.py'))]
 if scripts:related.append({'pid':int(d.name),'script_basenames':[Path(x).name for x in scripts]})
 try:
  for fd in (d/'fd').iterdir():
   fd_entries_checked+=1
   try:s=fd.stat()
   except OSError:continue
   kinds=[k for k,v in before.items() if (s.st_dev,s.st_ino)==(v['dev'],v['inode'])]
   if kinds:holders.append({'pid':int(d.name),'fd':fd.name,'matches':kinds})
 except OSError:unreadable_proc_entries+=1
source=REPO/'experiments/0010/acquire.py';lines=source.read_text().splitlines();source_statements=[{'line':i+1,'statement':line.strip()} for i,line in enumerate(lines) if 'os.link(partial, target)' in line or 'partial.unlink()' in line or "event('asset_published'" in line]
assert digest(source)=='412c6cddba8fcd32241026611edb627c76690bf8dcc93195d6360a4ef24b2d75'
ledger_path=ROOT/'clip-0010-downloads/ledger.json';events_path=ROOT/'clip-0010-downloads/events.jsonl';ledger=json.loads(ledger_path.read_text());events=[json.loads(x) for x in events_path.read_text().splitlines()]
rec=next(x for x in ledger['assets'] if x['id']=='ViT-B-32-pt');published=[x for x in events if x.get('kind')=='asset_published' and x.get('id')=='ViT-B-32-pt']
receipt=json.loads((REPO/'protocols/0010-clip-capability-freeze.json').read_text());frozen={}
for n,v in receipt['inputs'].items():
 p=REPO/n;h=digest(p);assert p.stat().st_size==v['bytes'] and h==v['sha256'];frozen[n]=h
assert len(frozen)==25
cfg=json.loads((REPO/'experiments/0010/config.json').read_text());assets={}
for n,s in cfg['assets'].items():
 p=ROOT/'clip-0010-assets'/n;st=snapshot(p);h=digest(p);g=hashlib.sha1(b'blob '+str(st['size']).encode()+b'\0')
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):g.update(b)
 assert st['size']==s['bytes'] and (s['sha256'] is None or h==s['sha256']) and (s['git_blob_sha1'] is None or g.hexdigest()==s['git_blob_sha1']);assets[n]={'bytes':st['size'],'sha256':h,'git_blob_sha1':g.hexdigest()};assert snapshot(p)==st
assert not (ROOT/'scratch/0010-capability-output').exists() and not (ROOT/'clip-0010-assets/0010-probe-started.json').exists()
final_stats={k:snapshot(p) for k,p in paths.items()};assert before==final_stats
result={'scope':'Cached file bytes and saved provenance only; no probe/download/install/model, repair, deletion or retry.','status':'diagnosis_completed','files':{k:{'path':str(p.relative_to(ROOT)),'before_stat':before[k],'after_read_stat':after_reads[k],'final_stat':final_stats[k],'sha256':hashes[k]} for k,p in paths.items()},'partial_prefix_comparison':{'bytes_compared':compared,'partial_equals_final_prefix':first is None,'first_mismatch_byte_offset':first,'algorithm':'Sequential 1MiB block exact-byte comparison; byte-level first mismatch within first differing block.'},'process_review':{'cmdline_separator':'bytes([0])','related_scripts':related,'matching_file_descriptors':holders,'fd_entries_checked':fd_entries_checked,'unreadable_or_raced_proc_entries':unreadable_proc_entries,'boundary':'Read-only Linux /proc snapshot; inability to inspect hidden/platform processes or prior exited writers remains.'},'downloader_source':{'path':'experiments/0010/acquire.py','sha256':digest(source),'statements':source_statements},'saved_acquisition':{'ledger_sha256':digest(ledger_path),'events_sha256':digest(events_path),'status':ledger['status'],'checkpoint_record':rec,'checkpoint_asset_published_events':published,'interpretation':'Source sequences link, unlink, then publication event. Saved completed ledger/event is consistent with that code path having reported completion; observed residual partial does not establish unlink persistence, creator, filesystem behavior or a root cause. No syscall/unlink audit log was recorded.'},'cause':'UNKNOWN','probe_output_exists':False,'probe_marker_exists':False,'frozen_input_count':25,'frozen_sha256':frozen,'asset_identities':assets,'source_sha256':digest(__file__)}
with (HERE/'review.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':result['status'],'checkpoint_sha256':hashes['final'],'asset_sha256':hashes['asset'],'partial_sha256':hashes['partial'],'partial_equals_final_prefix':first is None,'compared_bytes':compared,'first_mismatch_offset':first,'stat_unchanged':True,'matching_fds':len(holders),'frozen_inputs':25,'probe_started':False,'cause':'UNKNOWN','review_sha256':digest(HERE/'review.json')}))
