"""Transport only. Read saved files; never invoke any scientific stage."""
import base64,gzip,hashlib,json,re
from pathlib import Path
b=Path('/workspace/scratch/a349d03a0e0a');c=b/'scratch/0010-cached-admission-control'
def sha(x):return hashlib.sha256(x).hexdigest()
files={};omitted={}
def add(p):
 assert p.is_file() and not p.is_symlink();n=p.relative_to(b).as_posix();raw=p.read_bytes();assert len(raw)<=1<<20
 assert not re.search(rb'https?://[^\s\"<>]+[?][^\s\"<>]*(?:signature|token|credential|X-Amz|X-Goog)',raw,re.I)
 files[n]={'bytes':len(raw),'sha256':sha(raw),'base64':base64.b64encode(raw).decode()}
for root in [c,b/'scratch/0010-capability-control']:
 for p in sorted(root.iterdir()):
  if not p.is_file():continue
  if p.name=='pre-execution.json':omitted[p.relative_to(b).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'reason':'Large old tracked inventory omitted; preserved original, count and checks reported separately.'};continue
  if p.suffix in ('.json','.log','.py') and not p.name.startswith(('public-','private-','transfer')):add(p)
for rel in ['scratch/0010-cached-guard','scratch/0010-cached-guard-supervision','scratch/0010-capability-control/probe','scratch/0010-capability-output','scratch/0010-solver-functional','scratch/0010-solver-timing']:
 for p in sorted((b/rel).rglob('*')):
  if p.is_file() and p.suffix in ('.json','.log'):add(p)
for rel in ['protocols/0010-cached-probe-solver-freeze.json','protocols/0010-clip-capability-freeze.json','experiments/0010-solver/manifest.json']:
 add(b/'research-0010-cached-admission'/rel)
summary=json.loads((c/'postcheck.json').read_text())
(c/'public-README.txt').write_text('Cached guard PASS; original first capability completed 8 image forwards and 1 text forward. Saved features checked, no accuracy. Artificial solver 20 tests passed, three timing fits/six authored executions completed. Resource projections only; no 128-image study, quality or manuscript claim. Old auxiliary exit1 and partial/temp copies preserved; cause UNKNOWN. Capture source_main/cwd identify original capability checkout; solver source is independently pinned by new receipt. No new HTTP/install; direct body charge remains456087034 and allHTTP UNKNOWN. First evidence preflight exited1 on export alias control/assets.command.json; corrected alias mapping and 65 saved files verified before guard. This was not a scientific-stage failure. Public package omits all NPZ; separate audit-private package must not be published. All copies below are byte-exact; export wrapper files excluded from self reference.\n')
add(c/'public-README.txt')
(c/'transport-mapping-correction.json').write_text(json.dumps({'scope':'Non-scientific preservation inventory alias correction','actual_tool_chunk_id':'3d9f4e','actual_exit':1,'stdout':'UNRESOLVED_MEMBER control/assets.command.json\n','stderr':'','correction':'control/ maps to scratch/0010-capability-control/; partial-review/ maps to scratch/0010-partial-review/. Corrected read-only preflight exit0 verified65 files. No marker or scientific operation at the failed check.','no_fabricated_pid_or_wait':True},indent=2)+'\n');add(c/'transport-mapping-correction.json')
# Include transport README and exception only after creation; all source bytes retained.
def build(label,mapping,scope):
 d={'schema':'0010-cached-admission-export-v1','scope':scope,'files':mapping,'file_count':len(mapping),'total_original_bytes':sum(x['bytes'] for x in mapping.values()),'omitted_boundaries':omitted if label=='public' else {},'self_excluded':'Transport gzip/base64/metadata files and no recursive hash of this package.'}
 raw=json.dumps(d,sort_keys=True,separators=(',',':')).encode();z=gzip.compress(raw,mtime=0);encoded=base64.b64encode(z).decode();assert json.loads(gzip.decompress(z))==d
 for n,s in mapping.items():assert sha((b/n).read_bytes())==s['sha256'];assert base64.b64decode(s['base64'])==(b/n).read_bytes()
 for suffix,value in [('json.gz',z),('base64.txt',encoded.encode())]:
  with (c/(label+'-transfer.'+suffix)).open('xb') as f:f.write(value)
 meta={'schema':'0010-cached-admission-transfer-v1','label':label,'file_count':len(mapping),'total_original_bytes':d['total_original_bytes'],'json_bytes':len(raw),'json_sha256':sha(raw),'gzip_bytes':len(z),'gzip_sha256':sha(z),'base64_chars':len(encoded),'chunk_size':12000,'chunks':(len(encoded)+11999)//12000}
 with (c/(label+'-transfer.metadata.json')).open('x') as f:json.dump(meta,f,indent=2);f.write('\n')
 return meta
pub=build('public',files,'PUBLIC_TEXT_ONLY; original bytes, cloud paths retained, no NPZ/model/image/weights.')
private={}
for rel,s in summary['binary_not_public'].items():
 p=b/rel;raw=p.read_bytes();assert sha(raw)==s['sha256'] and len(raw)==s['bytes'];private[rel]={**s,'base64':base64.b64encode(raw).decode()}
priv=build('private',private,'AUDIT_PRIVATE_ONLY_DO_NOT_COMMIT; saved CLIP features/partial features and artificial solver inputs. No original JPEG, model, weights or other data.')
print(json.dumps({'public':pub,'audit_private':priv,'postcheck_sha256':sha((c/'postcheck.json').read_bytes()),'saved_probe_review_sha256':sha((c/'saved-probe-review.json').read_bytes())},sort_keys=True))
