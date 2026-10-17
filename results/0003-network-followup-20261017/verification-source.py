"""Read-only byte/Git/receipt audit; never decodes array/RGB data or runs model."""
import datetime,hashlib,json,subprocess
from pathlib import Path
from evr.motion_design import verify_design
from evr.motion_phases import verify_timing
B=Path('results/0003-network-followup-20261017');C=Path('../bootstapir-cache/0003-pointodyssey');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v1=verify_design(Path.cwd());timing=verify_timing(Path.cwd());baseline=json.loads((B/'baseline.json').read_text())
for p,h in baseline['sha256'].items():assert sha(p)==h,p
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show','b14e759b03ae992682c95b53466a9d220c4e86a1:notes/2026-10-17.md']))
for p,h in json.loads((B/'timing-reviewed-tests.json').read_text())['source_sha256'].items():assert sha(p)==h
acquisition=json.loads((B/'sample-acquisition.json').read_text());assert acquisition['received_bytes']==3324284510 and acquisition['whole_archive_verified']
h=hashlib.sha256();n=0
with(C/'sample-v1.2.tar.gz').open('rb')as f:
 while chunk:=f.read(2**20):h.update(chunk);n+=len(chunk)
assert n==3324284510 and h.hexdigest()==acquisition['sha256_received']=='436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f'
diag=json.loads((B/'diagnosis.json').read_text());assert diag['http_status']==200 and diag['received_bytes']==2328
assert sha('scratch/0003-hf-metadata.body')==diag['received_body_sha256']
for meta in diag['metadata_summary']:
 if meta['path']=='sample.tar.gz':assert meta['size']==n and meta['lfs']['oid']==h.hexdigest()
for file in ['pinned-source-http.json','geometry-source-http.json']:
 for e in json.loads((B/file).read_text())['events']:
  assert e['http_status']==200 and sha(C/('official-'+e['name']+'.body'))==e['received_sha256']
e=json.loads((B/'export-source-http.json').read_text());assert e['http_status']==200 and sha(C/'official-export.body')==e['received_sha256']
for receipt in B.glob('*-parent.json'):
 r=json.loads(receipt.read_text());phase=r['phase'];expected=1 if phase=='inventory'else 0
 assert r['child_exit_code']==expected and r['raw_wait_status']==(256 if expected else 0)
 assert r['source_sha256']==sha(B/(phase+'-source.py'))and r['parent_source_sha256']==sha(B/'followup-parent-source.py')
 for stream,d in r['log_sha256'].items():assert sha(B/(phase+'.'+stream+'.log'))==d
r=json.loads((B/'parent.json').read_text());assert r['child_exit_code']==r['raw_wait_status']==0
for stream,d in r['log_sha256'].items():assert sha(B/(stream+'.log'))==d
assert r['child_source_sha256']==sha(B/'diagnosis-source.py')and r['parent_source_sha256']==sha(B/'parent-source.py')
inventory=json.loads((B/'inventory.json').read_text());assert inventory['status']=='blocked'and inventory['members_seen']==2756 and not inventory['inventory_complete']and inventory['reason']=='single annotation exceeds frozen 512 MiB cap'
access=json.loads((B/'access-and-readiness.json').read_text());assert all(access[k]==0 for k in ['RGB_frames_decoded','GT_arrays_decoded','frame0_query_GT_fields_read','predictions_generated_or_read'])
ledger=json.loads((B/'acquisition-ledger.json').read_text());assert ledger['conservative_cumulative_ceiling_bytes']==4823387101<ledger['budget_bytes']
exclude={B/'final-verification.json',B/'verification-parent.json',B/'verification.stdout.log',B/'verification.stderr.log'}
files={str(p):sha(p)for p in B.rglob('*')if p.is_file()and p not in exclude}
for p in files:
 data=Path(p).read_bytes();assert len(data)<1048576 and b'\0'not in data and str(Path.cwd()).encode()not in data;data.decode()
out={'status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'byte integrity, source/log/wait and additive freeze verification only; no data decode/inference/old result analysis','execution_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'v1':v1,'timing':timing,'v1_canonical_receipt_sha256':sha('protocols/0003-design-freeze-v1.json'),'timing_canonical_receipt_sha256':sha('protocols/0003-timing-addendum-freeze-v1.json'),'old_candidate_immutable_files_preserved':len(baseline['sha256']),'archive_independent_rehash_bytes':n,'archive_independent_sha256':h.hexdigest(),'model_inferences':0,'array_or_RGB_decodes':0,'phase_fixtures':11,'archive_inventory':'blocked unchanged512MiB cap; retained real failed receipt','closed_sha256':files,'exclusions':[str(p)for p in sorted(exclude)],'boundary':'actual outer parent closes current phase logs/exit afterwards; final Git tree binds exclusions and self, no recursive supervisor'}
with(B/'final-verification.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k!='closed_sha256'},indent=2))
