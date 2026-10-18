"""Pre-freeze package-binding review closure; no dataset read or numerical rerun."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
B=Path('results/0003-annotation-consistency-stageA-20261017');base='ec26b8fc1a4901d9e249480da6cc9f772885baca';h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
runner='scripts/audit_0003_annotation_consistency_v1.py';test='tests/test_annotation_consistency_v1.py';initializer='src/evr/__init__.py'
allowed={runner,test,'CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-17.md'}
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base],text=True).splitlines();preserved=[]
for p in paths:
 raw=subprocess.check_output(['git','show',base+':'+p])
 if p not in allowed:assert Path(p).read_bytes()==raw,p;preserved.append(p)
 elif p.startswith('notes/'):assert Path(p).read_bytes().startswith(raw)
assert h(initializer)=='631cc3c910f6e790b2b5661bf7d28e99a612c534e82619c54084d9b64f9c5abe'
spec=importlib.util.spec_from_file_location('r',runner);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=json.loads((B/'review-input-map-v2.json').read_text())['mandatory_sha256'];current={p:h(p)for p in sorted(m.MANDATORY)}
assert set(current)==set(old)|{initializer} and len(current)==45
assert {p for p in old if old[p]!=current[p]}=={runner,test}
m.verify_module(Path.cwd(),m.evr,initializer,current[initializer])
r=json.loads((B/'attempt3-parent.json').read_text());assert r['observed_exit_code']==r['raw_wait_status']==0
for p,d in r['source_sha256'].items():assert h(p)==d,p
for suffix,d in r['logs_sha256'].items():assert h(B/('attempt3.'+suffix+'.log'))==d
assert 'Ran 17 tests'in(B/'attempt3.stderr.log').read_text()
assert not Path(m.RECEIPT).exists();cache=Path('../bootstapir-cache/0003-pointodyssey');assert not(cache/'annotation-consistency-v1-started.json').exists()
for name,d in json.loads(Path('configs/0003-annotation-consistency-v1.json').read_text())['prior_marker_sha256'].items():assert h(cache/name)==d
prior=json.loads((B/'review-input-map-v2.json').read_text())['prior_transitive_sha256'];assert len(prior)==38
for p,d in prior.items():assert h(p)==d
with(B/'review-input-map-v3.json').open('x')as f:f.write(json.dumps({'status':'REVIEW_MAP_NOT_A_FREEZE_RECEIPT','canonical_receipt':'absent','mandatory_count':45,'mandatory_sha256':current,'prior_transitive_count':38,'prior_transitive_sha256':prior,'change':'evr package initializer binding added; only unfrozen runner/tests changed'},indent=2)+'\n')
files=[p for p in B.rglob('*')if p.is_file()and('attempt3'in str(p)or p.name in ['fixture_parent_v2.py','initializer_fix_check.py','review-input-map-v3.json'])]
for p in files+[Path('docs/0003-initializer-binding-review-fix-2026-10-17.md')]:
 raw=p.read_bytes();assert len(raw)<1048576 and b'\0'not in raw
 assert (bytes([47])+b'workspace'+bytes([47]))not in raw
 raw.decode('utf8')
out={'status':'passed','previous_candidate':base,'prior_tracked_paths':len(paths),'previous_paths_byte_preserved_except_authorized':len(preserved),'previous_results_preserved':sum(p.startswith('results/')for p in preserved),'initializer_unchanged_sha256':h(initializer),'old_mandatory_count':44,'current_mandatory_count':45,'old_transitive_count':38,'only_existing_code_changes':[runner,test],'actual_package_path_and_hash':'passed','all_old_attempts_and_half_evidence':'byte-identical','new_geometry_receipt_and_marker':'absent','new_dataset_access_or_saved_R_rerun':False,'new_artifact_sha256':{str(p):h(p)for p in sorted(files)},'boundary':'Git tree binds this source/output without selfhash; no extra wait4 claimed for byte closure, attempt3 has actual source/log/wait receipt'}
with(B/'initializer-fix-check.json').open('x')as f:f.write(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k!='new_artifact_sha256'},indent=2))
