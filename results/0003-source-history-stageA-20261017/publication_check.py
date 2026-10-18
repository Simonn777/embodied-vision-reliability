"""Byte/source/log closure and prospective budget algebra only; no archive or projection."""
import hashlib,json,subprocess
from pathlib import Path
B=Path('results/0003-source-history-stageA-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=json.loads((B/'baseline.json').read_text());base=b['base'];paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base],text=True).splitlines()
mutable={'CURRENT_STATE.md','NEXT_ACTIONS.md','README.md','notes/2026-10-17.md'};digests={}
for p in paths:
 old=subprocess.check_output(['git','show',base+':'+p])
 if p not in mutable:assert Path(p).read_bytes()==old,p;digests[p]=h(p)
 elif p.startswith('notes/'):assert Path(p).read_bytes().startswith(old)
assert len(paths)==1548 and len(digests)==1544 and sum(p.startswith('results/')for p in digests)==1420
assert hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()==b['protected_path_sha256_map_digest']
cache=Path('../bootstapir-cache/0003-pointodyssey')
for name,d in b['old_marker_sha256'].items():assert h(cache/name)==d
for phase in ['source-audit','fixtures']:
 r=json.loads((B/(phase+'-parent.json')).read_text());assert r['observed_exit_code']==r['raw_wait_status']==0
 for p,d in r['source_sha256'].items():assert h(p)==d,p
 for st,d in r['logs_sha256'].items():assert h(B/(phase+'.'+st+'.log'))==d
 assert r['affinity']==[0,1] and r['AS_bytes']==6*1024**3 and r['timeout_seconds']==900
assert 'Ran 3 tests' in (B/'fixtures.stderr.log').read_text()
a=json.loads((B/'audit/audit.json').read_text());assert a['verified_distinct_frozen_inputs']==38 and a['total_source_and_tree_bytes_read']==39425
assert a['local_upstream_objects']['missing_count']==39 and a['local_upstream_objects']['available_records']==[]
c=json.loads(Path('configs/0003-annotation-consistency-candidate-v1.json').read_text())
assert c['status']=='UNFROZEN_STAGE_A_CANDIDATE_NO_EXECUTION_AUTHORITY'
assert c['frames']==[0,1,2] and c['fixed_original_point_ids']==[0,1] and c['max_descriptive_checks']==6
assert c['carried_returned_bytes']+c['carried_reserved_bytes']==c['carried_charge_bytes']==22910156
assert c['carried_directory_reserve_subset_bytes']==1480
# Independent physical-offset formula, not a real sparse reader or geometry check.
N=11721;T=849
spans={'trajs_2d':(2*N*2+2*2)*4,'trajs_3d':(2*N*3+2*3)*4,'visibs':T+3,'valids':T+3,'intrinsics':3*3*3*4,'extrinsics':3*4*4*4}
assert spans==c['prospective_sparse_spans_bytes']
returned=22+6*128+sum(spans.values());reserved=6*65536+348+22
assert returned==c['prospective_new_returned_bytes']==471674 and reserved==c['prospective_new_reserves_bytes']==393586
assert 22910156+returned+reserved==c['prospective_total_charged_bytes']==23775416<67108864
new_paths=[str(p)for p in B.rglob('*')if p.is_file()and p.name!='publication-check.json']
extra=['src/evr/point_source_history_v1.py','scripts/audit_0003_source_history_v1.py','tests/test_point_source_history_v1.py','configs/0003-annotation-consistency-candidate-v1.json','protocols/0003-annotation-consistency-candidate-v1.md','docs/0003-source-history-stageA-2026-10-17.md']
for p in new_paths+extra:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw,p
 assert (bytes([47])+b'workspace'+bytes([47]))not in raw and (bytes([47])+b'root'+bytes([47,46]))not in raw,p
 raw.decode('utf8')
r={'status':'passed','base':base,'protected_old_paths':1544,'prior_result_files_byte_preserved':1420,'old_markers':'both unchanged','frozen_chains':5,'distinct_frozen_inputs':38,'new_numeric_data_access':0,'new_geometry_or_compatibility_analysis':False,'candidate':'UNFROZEN; no new canonical receipt or execution authority','prospective_budget_algebra':'passed; not a data measurement','new_artifact_sha256':{p:h(p)for p in sorted(new_paths+extra)},'baseline_map_digest':b['protected_path_sha256_map_digest'],'boundary':'final closure source/output are bound by Git tree; output not self-hashed and no extra wait4 claimed; source audit and synthetic fixtures have actual parent receipts'}
with(B/'publication-check.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items()if k!='new_artifact_sha256'},indent=2))
