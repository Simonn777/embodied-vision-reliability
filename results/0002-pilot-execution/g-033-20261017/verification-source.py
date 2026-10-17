"""Independent stdlib checks of the one blocked attempt; no candidate imports."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

root=Path.cwd();folder=Path('results/0002-pilot-execution/g-033-20261017')
h=lambda b:hashlib.sha256(b).hexdigest()
outer=json.loads((folder/'outer.json').read_text())
public=json.loads((folder/'publication.json').read_text())
diagnosis=json.loads((folder/'dependency-diagnosis.json').read_text())
assert outer['status']=='failed' and list(outer['phases'])==['preflight']
phase=outer['phases']['preflight'];assert phase['exit_code']==1 and phase['raw_wait_status']==256
assert datetime.fromisoformat(outer['started_at_utc'])<datetime.fromisoformat(outer['ended_at_utc'])
assert phase['wall_seconds']>0 and outer['wall_seconds']>=phase['wall_seconds']
assert h((folder/'launcher-source.py').read_bytes())==outer['wrapper_sha256']==public['launcher_source_sha256']
assert h((folder/'preflight.stdout.log').read_bytes())==phase['stdout_sha256']
raw=Path('scratch/g033-original-preflight.stderr.log').read_bytes()
assert h(raw)==phase['stderr_sha256']==public['stderr_original_sha256']
assert raw.replace(str(root).encode(),b'<repo>')==(folder/'preflight.stderr.log').read_bytes()
assert h((folder/'preflight.stderr.log').read_bytes())==public['stderr_published_sha256']
assert b'ValueError: frozen dependency version changed: numpy' in raw
module=ast.parse((folder/'launcher-source.py').read_text())
preflight=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREFLIGHT' for t in n.targets))
assert preflight==(folder/'preflight-code.py').read_text()
receipt=json.loads(Path('protocols/0002-freeze-v2.json').read_text())
assert receipt['input_commit']=='bbe7428e126a566a62d3c123d7158f120aa66e2b'
for ref in ['HEAD','origin/main']:
 subprocess.run(['git','merge-base','--is-ancestor',receipt['input_commit'],ref],check=True)
assert subprocess.check_output(['git','show','HEAD:protocols/0002-freeze-v2.json'])==subprocess.check_output(['git','show','origin/main:protocols/0002-freeze-v2.json'])==Path('protocols/0002-freeze-v2.json').read_bytes()
for path,expected in receipt['input_sha256'].items():
 assert h(Path(path).read_bytes())==expected
 assert h(subprocess.check_output(['git','show',receipt['input_commit']+':'+path]))==expected
 assert h(subprocess.check_output(['git','show','HEAD:'+path]))==expected
assert len(receipt['input_sha256'])==29
assert len(diagnosis['actual_dependency_versions'])==16
mismatch={p:v for p,v in diagnosis['actual_dependency_versions'].items() if not v['matches']}
assert list(mismatch)==['numpy'] and mismatch['numpy']['actual']=='2.3.5' and mismatch['numpy']['expected']=='2.0.2'
assert all(x['matches'] for x in diagnosis['verified_cache_objects'].values())
assert all(x['matches'] for x in diagnosis['verified_upstream_sources'].values())
assert not Path('results/0002-pilot-predictions/g-033').exists()
assert not list(folder.glob('*.csv'))
result={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'independent standard-library artifact/Git checks; no second gate call, model, archive decoding or metrics',
 'freeze_input_hashes_verified_against_commit_head_worktree':29,'dependency_records_checked':16,
 'dependency_mismatch':{'numpy':{'expected':'2.0.2','actual':'2.3.5'}},
 'outer_exit_code':phase['exit_code'],'outer_failed_status_verified':True,
 'original_log_digest_and_published_redaction_verified':True,'launcher_and_preflight_source_hashes_verified':True,
 'infer_subprocess_started':False,'raw_prediction_csv_produced':False,
 'calibration_groups_completed':0,'evaluation_groups_completed':0,
 'artifact_sha256':{p.name:h(p.read_bytes()) for p in folder.iterdir() if p.is_file() and p.name!='independent-verification.json'}}
(folder/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'}))
