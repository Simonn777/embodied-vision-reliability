"""Independent saved-record arithmetic/byte audit; no array/archive/model reads."""
import hashlib,importlib.util,json,math,subprocess
from pathlib import Path
B=Path('results/0003-fortran-stageB-20261017');E=B/'evidence'
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((B/'audit/audit.json').read_text());outer=json.loads((E/'audit-parent.json').read_text())
assert a['status']=='stopped'and a['reason']=='storage-aware proper rotation gate'
assert outer['observed_exit_code']==1 and outer['raw_wait_status']==256
assert a['audit_complete']and a['metadata_access_state']=='COMPLETED'
assert (E/'audit.stdout.log').read_bytes()==(B/'audit/audit.json').read_bytes()
assert a['selected_original_point_indices']==[0,1]and a['planned_point_checks']==6
assert a['attempted_point_checks']==1 and a['unattempted_point_checks']==5
assert a['point_checks']==[{'frame':0,'point_index':0,'status':'failed','projection_consistent':None,'valid_visible_GT_support':False,'validity':1,'visibility':1,'reason':'storage-aware proper rotation gate'}]
assert a['finite_error_count']==0 and a['max_error_original_px']is None and a['rms_error_original_px']is None
assert a['projection_consistent_count']==a['valid_visible_GT_support_count']==0
spans=retained=discard=scalar_count=0
for key,entry in a['numeric_access'].items():
 s=a['schema'][key];shape=s['shape'];item=s['itemsize'];T=shape[0]
 saved=[3]+shape[1:];n=math.prod(saved);scalar_count+=n
 if s['order']=='C':span=n*item
 else:
  # Independently sum maximum first-three-time physical stride; no reader import.
  stride=item;last=0
  for extent,full in zip(saved,shape):last+=(extent-1)*stride;stride*=full
  span=last+item
 assert entry['counts_complete']and entry['status']=='completed'
 assert entry['numeric_bytes_returned']==entry['numeric_stream_span_bytes']==span
 assert entry['retained_scalars']==n and entry['discarded_bytes']==span-n*item
 spans+=span;retained+=n*item;discard+=span-n*item
assert spans==20604126 and retained==773886 and discard==19830240 and scalar_count==246216
metadata=a['metadata_access'];assert len(metadata)==4
raw_metadata=sum(v['bytes']for v in metadata.values());assert raw_metadata==6122
info_headers=sum(vv['header']['header_bytes']for v in metadata.values()for vv in v.get('headers',[]));assert info_headers==1536
for v in metadata.values():
 assert v['status']=='completed'and v['counts_complete']
 for hh in v.get('headers',[]):assert not hh['numeric_payload_interpreted']and not hh['object_payload_decoded']
annotation_headers=2*sum(x['header_bytes']for x in a['schema'].values());assert annotation_headers==1536
new_returned=spans+raw_metadata+info_headers+annotation_headers+3*22
opens=12+6+6;directories=a['implicit_directory_reserved_bytes'];assert directories==1110
reserve=589824+opens*65536+directories
assert new_returned+1152==a['returned_bytes_including_carried']==20614538
assert reserve==a['stream_and_directory_reserves_including_carried']==2163798
assert new_returned+1152+reserve==a['charged_bytes']==22778336<64*1024**2
assert a['family_graph']['independent_calibration_or_evaluation_families_established']==0
assert all(a[x]==0 for x in ['RGB','model','new_GET'])
baseline=json.loads((E/'protected-baseline.json').read_text())
for p,d in baseline['sha256'].items():assert h(p)==d,p
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show',baseline['source_commit']+':notes/2026-10-17.md']))
spec=importlib.util.spec_from_file_location('frozen','scripts/audit_0003_fortran_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);r=m.verify_published()
for phase in ['preflight','audit']:
 p=json.loads((E/(phase+'-parent.json')).read_text())
 assert h(E/'parent-source.py')==p['parent_sha256']
 for name,d in p['source_sha256'].items():assert h(name)==d,name
 for st,d in p['logs_sha256'].items():assert h(E/(phase+'.'+st+'.log'))==d
marker=Path('../bootstapir-cache/0003-pointodyssey/fortran-v2-started.json');mr=json.loads(marker.read_text());assert mr['input_commit']==r['input_commit']and mr['no_automatic_retry']
(E/'one-shot-marker.json').write_bytes(marker.read_bytes())
summary={'status':'passed_independent_saved_evidence_audit','audit_status':'stopped at strict rotation gate, not a geometry pass','reprojection_independent_math':'not possible: failed camera gate precedes pixel/error calculation; no offending matrix saved; no additional GT read authorized or performed','audited_numeric_spans':spans,'retained_numeric_bytes':retained,'retained_scalars':scalar_count,'discarded_physical_gap_bytes':discard,'charged_bytes':a['charged_bytes'],'remaining_budget_bytes':64*1024**2-a['charged_bytes'],'metadata_container_bytes':raw_metadata,'metadata_member_hashes':{k:v['source_member_sha256']for k,v in metadata.items()},'input_commit':r['input_commit'],'receipt_sha256':h(m.RECEIPT),'unchanged_frozen_inputs':len(r['inputs_sha256']),'protected_files_verified':len(baseline['sha256']),'audit_json_sha256':h(B/'audit/audit.json'),'scope':'stdlib JSON/hash/counter arithmetic only; no archive/NPZ numeric read or new geometry/model call; one shot marker retained'}
(E/'independent-verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
