"""Safe metadata-only inventory of the one checksum-verified cached sample."""
import collections,datetime,hashlib,json,tarfile,time
from pathlib import Path
from evr.motion_design import safe_member,verify_design
B=Path('results/0003-network-followup-20261017');C=Path('../bootstapir-cache/0003-pointodyssey');archive=C/'sample-v1.2.tar.gz'
a=json.loads((B/'sample-acquisition.json').read_text());assert a['whole_archive_verified']and a['received_bytes']==3324284510 and archive.stat().st_size==3324284510
verify_design(Path.cwd());r={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'in_progress','archive_sha256':a['sha256_received'],'members_seen':0,'unpacked_member_bytes':0,'metadata_only':True,'RGB_decodes':0,'annotation_decodes':0,'max_members':200000,'max_unpacked_bytes':20*1024**3,'max_single_annotation_bytes':512*1024**2};t=time.perf_counter();seen=set();scenes={};annos=[];h=hashlib.sha256()
try:
 with tarfile.open(archive,mode='r|gz')as tar:
  for m in tar:
   kind='file'if m.isfile()else'directory'if m.isdir()else'other';name=safe_member(m.name,kind)
   if name in seen:raise ValueError('duplicate canonical member path')
   seen.add(name);r['members_seen']+=1;r['unpacked_member_bytes']+=m.size
   if r['members_seen']>200000 or r['unpacked_member_bytes']>20*1024**3:raise ValueError('inventory count/unpacked-size cap')
   entry=[name,kind,m.size];h.update((json.dumps(entry,separators=(',',':'))+'\n').encode())
   parts=name.split('/');scene=parts[1]if len(parts)>=2 else'<archive-root>'
   s=scenes.setdefault(scene,{'files':0,'bytes':0,'modalities':{},'annotation_members':[]})
   if kind=='file':
    s['files']+=1;s['bytes']+=m.size;modality=parts[-2]if len(parts)>2 else'<root>';s['modalities'][modality]=s['modalities'].get(modality,0)+1
    if Path(name).name in ['anno.npz','annot.npz']:
     if m.size>512*1024**2:raise ValueError('single annotation exceeds frozen 512 MiB cap')
     info={'path':name,'size':m.size};annos.append(info);s['annotation_members'].append(info)
 r['status']='passed';r['inventory_complete']=True
except Exception as exc:r.update(status='blocked',inventory_complete=False,error_type=type(exc).__name__,reason=str(exc).replace(str(Path.cwd()),'<repo>'))
r.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-t,canonical_inventory_sha256=h.hexdigest(),scenes=scenes,annotation_members=sorted(annos,key=lambda x:x['path']),quarantine='entire sample conservatively AUDIT_UNKNOWN_FAMILY; names/different RGB hashes do not establish independent families')
(B/'inventory.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='scenes'},indent=2));raise SystemExit(0 if r['status']=='passed'else 1)
