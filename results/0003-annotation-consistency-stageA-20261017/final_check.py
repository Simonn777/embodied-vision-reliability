"""Review map and byte closure only; no real geometry, payload read or proposal rerun."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
B=Path('results/0003-annotation-consistency-stageA-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base='2826fac713fd998e4edf1666017c4944f98002d3';initial='a3f28d7f8d4611d0b379f6acefa65514ac55df71';method='c1bcc597db980cd3aa14c0747af0183da266e6af'
mutable={'CURRENT_STATE.md','NEXT_ACTIONS.md','README.md','notes/2026-10-17.md'};paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base],text=True).splitlines();old={}
for p in paths:
 raw=subprocess.check_output(['git','show',base+':'+p])
 if p not in mutable:assert Path(p).read_bytes()==raw,p;old[p]=h(p)
 elif p.startswith('notes/'):assert Path(p).read_bytes().startswith(raw)
assert len(old)==1544 and sum(p.startswith('results/')for p in old)==1420
old_source=subprocess.check_output(['git','ls-tree','-r','--name-only',initial,'--','results/0003-source-history-stageA-20261017'],text=True).splitlines()
for p in old_source+['docs/0003-source-history-stageA-2026-10-17.md','protocols/0003-annotation-consistency-candidate-v1.md','configs/0003-annotation-consistency-candidate-v1.json']:assert Path(p).read_bytes()==subprocess.check_output(['git','show',initial+':'+p]),p
spec=importlib.util.spec_from_file_location('r','scripts/audit_0003_annotation_consistency_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not Path(m.RECEIPT).exists();cache=Path('../bootstapir-cache/0003-pointodyssey');assert not(cache/'annotation-consistency-v1-started.json').exists()
assert not Path('results/0003-annotation-consistency-v1-execution').exists()
prior={}
for p,want in m.PRIOR.items():
 assert h(p)==want
 r=json.loads(Path(p).read_text())
 for name,d in r['inputs_sha256'].items():
  assert h(name)==d and hashlib.sha256(subprocess.check_output(['git','show',r['input_commit']+':'+name])).hexdigest()==d
  prior[name]=d
assert len(prior)==38
c=json.loads(Path('configs/0003-annotation-consistency-v1.json').read_text())
for name,d in c['prior_marker_sha256'].items():assert h(cache/name)==d
f=json.loads((B/'attempt1-parent.json').read_text());assert f['observed_exit_code']==f['raw_wait_status']==0
for p,d in f['source_sha256'].items():assert h(p)==d,p
for suffix,d in f['logs_sha256'].items():assert h(B/('attempt1.'+suffix+'.log'))==d
assert 'Ran 14 tests'in(B/'attempt1.stderr.log').read_text()
for name in ['protocols/0003-half-cell-analysis-v1.md','configs/0003-half-cell-analysis-v1.json','scripts/half_cell_certificate_v1.py','tests/test_half_cell_certificate_v1.py']:assert Path(name).read_bytes()==subprocess.check_output(['git','show',method+':'+name])
H=Path('results/0003-half-cell-analysis-20261017');v=json.loads((H/'verification-execution.json').read_text());assert v['observed_exit_code']==v['raw_wait_status']==0
for p,d in v['source_sha256'].items():assert h(p)==d
for p,d in v['input_sha256'].items():assert h(p)==d
for suffix,d in v['logs_sha256'].items():assert h(H/('verification-output.json'if suffix=='stdout'else'verification.stderr.log'))==d
assert json.loads((H/'output.json').read_text())['status']=='INCONCLUSIVE'
assert c['carried_returned_bytes']+c['carried_reserved_bytes']==22910156 and c['carried_directory_reserve_subset_bytes']==1480
assert sum(c['prospective_sparse_spans_bytes'].values())==470884
assert 22910156+c['prospective_new_returned_bytes']+c['prospective_new_reserves_bytes']==c['prospective_total_charged_bytes']==23775416
mandatory={p:h(p)for p in sorted(m.MANDATORY)}
with(B/'review-input-map.json').open('x')as out:out.write(json.dumps({'status':'REVIEW_MAP_NOT_A_FREEZE_RECEIPT','canonical_receipt':'absent','mandatory_count':len(mandatory),'mandatory_sha256':mandatory,'prior_transitive_count':38,'prior_transitive_sha256':prior,'scope':'complete prospective executable input; no new data/geometry authority'},indent=2)+'\n')
files=[p for p in B.rglob('*')if p.is_file()and p.name!='final-check.json']
new_inputs=['docs/0003-source-history-and-geometry-plan-2026-10-17.md','protocols/0003-annotation-consistency-v1.md','configs/0003-annotation-consistency-v1.json','src/evr/annotation_consistency_v1.py','scripts/audit_0003_annotation_consistency_v1.py','scripts/run_0003_annotation_consistency_bounded_v1.py','tests/test_annotation_consistency_v1.py']
for p in [*files,*map(Path,new_inputs)]:
 raw=p.read_bytes();assert len(raw)<1048576 and b'\0'not in raw,p
 assert (bytes([47])+b'workspace'+bytes([47]))not in raw and (bytes([47])+b'root'+bytes([47,46]))not in raw,p
 raw.decode('utf8')
r={'status':'passed','base':base,'prior_tracked':1548,'protected_prior_paths':1544,'prior_results_preserved':1420,'initial_source_artifacts_preserved':len(old_source),'initial_source_report_and_candidate':'byte-identical','original_half_method_commit':method,'method_four_files':'byte-identical','mandatory_input_count':len(mandatory),'old_chains':5,'old_transitive_inputs':38,'new_geometry_receipt_marker_output':'all absent','old_markers':'unchanged','new_dataset_payload_reads':0,'new_real_geometry_execution':False,'one_saved_R_analysis':'INCONCLUSIVE, no rerun','new_artifact_sha256':{str(p):h(p)for p in sorted(files)},'current_complete_input_sha256':{p:h(p)for p in new_inputs},'boundary':'Git publication binds this final source/output, no selfhash or extra wait claimed; synthetic suite and saved analysis/verification have actual wait4'}
with(B/'final-check.json').open('x')as out:out.write(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items()if k not in ('new_artifact_sha256','current_complete_input_sha256')},indent=2))
