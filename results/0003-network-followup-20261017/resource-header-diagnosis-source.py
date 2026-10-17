"""Metadata-only unsupported-layout diagnosis; no NPY numeric bytes are read."""
import ast,datetime,hashlib,json,struct,zipfile
from pathlib import Path
from evr.bounded_point_audit import ReadBudget
B=Path('results/0003-resource-audit-20261017');audit=json.loads((B/'audit.json').read_text());assert audit['status']=='stopped'and audit['reason']=='C order only'and not audit['numeric_GT_values_accessed']
file=Path('../bootstapir-cache/0003-pointodyssey/resource-audit-v1/selected-annotation.npz');assert file.stat().st_size==audit['annotation_bytes']
budget=ReadBudget(64*1024**2-audit['total_read_budget_charged']);records=[]
with zipfile.ZipFile(file)as z:
 infos=z.infolist();assert len(infos)<=64
 for i,info in enumerate(infos):
  budget.open_stream()
  with z.open(info)as f:
   if budget.read(f,6)!=b'\x93NUMPY':raise ValueError('NPY magic')
   version=tuple(budget.read(f,2));assert version in[(1,0),(2,0),(3,0)]
   n=struct.unpack('<H'if version==(1,0)else'<I',budget.read(f,2 if version==(1,0)else 4))[0];assert n<=65536
   header=budget.read(f,n);metadata=ast.literal_eval(header.decode('utf-8'if version==(3,0)else'latin1').strip())
   records.append({'zip_ordinal':i+1,'member':info.filename,'uncompressed_member_bytes':info.file_size,'compressed_member_bytes':info.compress_size,'NPY_version':list(version),'header':metadata,'header_text_sha256':hashlib.sha256(header).hexdigest(),'numeric_bytes_read':0,'Fortran_rejected_by_frozen_reader':metadata.get('fortran_order')is True})
r={'status':'metadata_diagnosis_complete_no_numeric_decode','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_failure_retained':True,'schema':records,'prior_audit_charged_bytes':audit['total_read_budget_charged'],'diagnosis_uncompressed_header_bytes_returned':budget.returned_bytes,'diagnosis_internal_buffer_reserve_bytes':budget.stream_reserve_bytes,'cumulative_all_audit_and_diagnosis_charged_bytes':audit['total_read_budget_charged']+budget.charged_bytes,'hard_limit_bytes':64*1024**2,'numeric_GT_values_read':0,'interpretation':'header metadata only, no change to Fortran rejection, no numeric rerun or further resource expansion; actual geometry remains blocked'}
assert r['cumulative_all_audit_and_diagnosis_charged_bytes']<=r['hard_limit_bytes']
(B/'unsupported-layout-metadata.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
