"""Independent metadata/byte closure for stopped resource audit; no array reads."""
import csv,hashlib,importlib.util,json,subprocess,datetime
from pathlib import Path
from evr.motion_design import safe_member
B=Path('results/0003-resource-audit-20261017');F=Path('results/0003-network-followup-20261017');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('cached_audit','scripts/audit_point_sample.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);receipt=m.frozen()
a=json.loads((B/'audit.json').read_text());assert a['status']=='stopped'and a['reason']=='C order only'and a['inventory_complete']and not a['numeric_GT_values_accessed']and a['numeric_data_bytes_returned']==0 and a['point_checks_attempted']==0
rows=[]
for p in sorted(B.glob('inventory-*.csv')):
 with p.open(newline='')as f:rows.extend(csv.DictReader(f))
assert len(rows)==14415;assert[int(r['ordinal'])for r in rows]==list(range(1,14416));assert len({r['canonical_path']for r in rows})==14415
h=hashlib.sha256();total=0;annotations=[]
for r in rows:
 i=int(r['ordinal']);size=int(r['declared_bytes']);safe_member(r['canonical_path'],r['type']);total+=size;h.update((json.dumps([i,r['canonical_path'],r['type'],size],separators=(',',':'))+'\n').encode())
 if Path(r['canonical_path']).name in['anno.npz','annot.npz','annotations.npz']and r['type']=='file':annotations.append(r)
assert total==3334075426==a['inventory_declared_bytes']and h.hexdigest()==a['inventory_sha256']
old=rows[2755];assert old['canonical_path']=='sample/scene_d78_0318_3rd/anno.npz'and int(old['declared_bytes'])==705904917>512*1024**2
assert len(annotations)==2 and sorted(annotations,key=lambda r:r['canonical_path'])[0]['canonical_path']==a['selected_annotation']['path']=='sample/r4_new_f/anno.npz'
annotation=Path('../bootstapir-cache/0003-pointodyssey/resource-audit-v1/selected-annotation.npz');assert annotation.stat().st_size==98527880==a['annotation_bytes'];h=hashlib.sha256()
with annotation.open('rb')as f:
 while part:=f.read(2**20):h.update(part)
assert h.hexdigest()==a['annotation_sha256']=='382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77'
metadata=json.loads((B/'unsupported-layout-metadata.json').read_text());assert metadata['cumulative_all_audit_and_diagnosis_charged_bytes']==590976<67108864 and metadata['numeric_GT_values_read']==0
assert [r['member']for r in metadata['schema']if r['Fortran_rejected_by_frozen_reader']]==['valids.npy','visibs.npy']
for r in metadata['schema']:assert r['numeric_bytes_read']==0
baseline=json.loads((B/'baseline.json').read_text())
for p,d in baseline['sha256'].items():assert sha(p)==d,p
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show','661a75d3af884186a8cacdcfbc65e1ae17cb1f0a:notes/2026-10-17.md']))
for p,d in json.loads((B/'final-fixture-validation.json').read_text())['source_sha256'].items():assert sha(p)==d
for phase,code in [('resource-fixtures',1),('resource-fixtures-fixed',0),('resource-final-fixtures',0),('resource-audit',1),('resource-header-diagnosis',0)]:
 r=json.loads((F/(phase+'-parent.json')).read_text());assert r['child_exit_code']==code and r['raw_wait_status']==(256 if code else 0);assert sha(F/(phase+'-source.py'))==r['source_sha256']
 for stream,d in r['log_sha256'].items():
  if phase=='resource-fixtures'and stream=='stderr':
   disclosure=json.loads((B/'fixture-attempt1/failure-and-repair.json').read_text());assert d==disclosure['raw_stderr_sha256']and sha(F/'resource-fixtures.stderr.public.log')==disclosure['public_redacted_stderr_sha256']
  else:assert sha(F/(phase+'.'+stream+'.log'))==d
exclude={B/'verification.json',F/'resource-verification-parent.json',F/'resource-verification.stdout.log',F/'resource-verification.stderr.log'}
files={str(p):sha(p)for root in[B,F]for p in root.rglob('*')if p.is_file()and p not in exclude}
for p in files:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw and str(Path.cwd()).encode()not in raw;raw.decode()
out={'status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'resource_input_commit':receipt['input_commit'],'resource_receipt_sha256':sha(m.RECEIPT),'mandatory_resource_files_verified':len(m.INPUTS),'preserved_pre_resource_files':len(baseline['sha256']),'complete_inventory_rows':len(rows),'declared_bytes':total,'annotation_raw_byte_rehash':h.hexdigest(),'numeric_GT_values_interpreted':0,'projection_results':'not attempted; max/RMS undefined','readiness':'BLOCKED_UNSUPPORTED_FORTRAN_AND_NO_FINAL_COHORT','scope':'independent CSV/hash/header-metadata/source/log/wait audit only; no archive re-inventory, array decode/model or old-test analysis','closed_sha256':files,'exclusions':[str(p)for p in sorted(exclude)],'boundary':'current parent closes source/log/exit after this child; final Git tree binds exclusions/self; parent own exit has no more supervisor receipt'}
with(B/'verification.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k!='closed_sha256'},indent=2))
