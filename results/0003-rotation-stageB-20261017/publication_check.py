"""Historical bytes and published source/artifact closure; no numeric cache reread."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
B=Path('results/0003-rotation-stageB-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base='8ffb8bcfbc5c7a9fe87e5a624d3f3dc1e0d97072';mutable={'README.md','CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-17.md'}
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base],text=True).splitlines();digests={}
for p in paths:
 old=subprocess.check_output(['git','show',base+':'+p])
 if p in mutable:
  if p.startswith('notes/'):assert Path(p).read_bytes().startswith(old)
 else:
  assert Path(p).read_bytes()==old,p;digests[p]=h(p)
base_digest=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()
spec=importlib.util.spec_from_file_location('r','scripts/run_rotation_diagnostic_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.verify_published();assert len(m.verify_prior(Path.cwd()))==30
cache=Path('../bootstapir-cache/0003-pointodyssey')
assert h(cache/'fortran-v2-started.json')==json.loads(Path('configs/0003-rotation-diagnostic-v1.json').read_text())['old_marker_sha256']
marker=(cache/'rotation-diagnostic-v1-started.json').read_bytes();assert json.loads(marker)=={'input_commit':r['input_commit'],'no_retry':True}
with(B/'reservation.json').open('x')as f:f.write(marker.decode()+'\n')
v=json.loads((B/'verification-parent.json').read_text());assert v['observed_exit_code']==v['raw_wait_status']==0
for p,d in v['source_sha256'].items():assert h(p)==d
for p,d in v['input_sha256'].items():assert h(p)==d
for st,d in v['logs_sha256'].items():assert h(B/('verification.'+st+'.log'))==d
files={str(p):h(p)for p in B.rglob('*')if p.is_file()and p.name!='publication-check.json'}
for p in [*files,'docs/0003-rotation-diagnostic-stageB-2026-10-17.md']:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw,p
 assert (bytes([47])+b'workspace'+bytes([47]))not in raw and (bytes([47])+b'root'+bytes([47,46]))not in raw,p
 raw.decode('utf8')
summary={'status':'passed','baseline_receipt_commit':base,'prior_tracked_paths':len(paths),'protected_prior_paths_byte_preserved':len(digests),'prior_results_byte_preserved':sum(p.startswith('results/')for p in digests),'prior_rotation_StageA_artifacts_byte_preserved':sum(p.startswith('results/0003-rotation-stageA-20261017/')for p in digests),'protected_path_SHA256_map_digest':base_digest,'old_marker':'unchanged','new_marker':'retained; exclusive no-retry reservation archived','new_marker_sha256':hashlib.sha256(marker).hexdigest(),'canonical_receipt_sha256':h(m.RECEIPT),'frozen_input_commit':r['input_commit'],'frozen_inputs_sha256':r['inputs_sha256'],'prior_frozen_distinct_inputs':30,'new_artifact_sha256':files,'report_sha256':h('docs/0003-rotation-diagnostic-stageB-2026-10-17.md'),'boundary':'final closure source/output are bound by Git tree, not recursive selfhash; no separate wait4 claimed for this byte audit; actual diagnostic and saved verification have parent wait4 records'}
with(B/'publication-check.json').open('x')as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items()if k not in ('frozen_inputs_sha256','new_artifact_sha256')},indent=2))
