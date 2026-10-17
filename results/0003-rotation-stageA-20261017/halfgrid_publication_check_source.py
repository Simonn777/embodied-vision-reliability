"""Append-only closure for the half-grid revision; no data payload access."""
import hashlib,json,subprocess
from pathlib import Path
B=Path('results/0003-rotation-stageA-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
previous='fbb25bf5791dd0dfd30d32c3eaa0a1a90672bdd7'
b=json.loads((B/'protected-baseline.json').read_text())
for p,d in b['sha256'].items():assert h(p)==d,p
oldpaths=subprocess.check_output(['git','ls-tree','-r','--name-only',previous,'--',str(B)],text=True).splitlines()
for p in oldpaths:
 assert Path(p).read_bytes()==subprocess.check_output(['git','show',previous+':'+p]),p
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show',previous+':notes/2026-10-17.md']))
assert h('../bootstapir-cache/0003-pointodyssey/fortran-v2-started.json')==b['old_marker_sha256']
assert not Path('../bootstapir-cache/0003-pointodyssey/rotation-diagnostic-v1-started.json').exists()
assert not Path('protocols/0003-rotation-diagnostic-freeze-v1.json').exists()
r=json.loads((B/'halfgrid-fixtures-parent.json').read_text());assert r['observed_exit_code']==r['raw_wait_status']==0
for p,d in r['source_sha256'].items():assert h(p)==d,p
for p,d in r['command_source_sha256'].items():assert h(p)==h(B/Path(p).name)==d
assert h(B/'rotation_stageA_parent.py')==r['parent_source_sha256']
for st,d in r['logs_sha256'].items():assert h(B/('halfgrid-fixtures.'+st+'.log'))==d
assert 'Ran 19 tests' in (B/'halfgrid-fixtures.stderr.log').read_text()
newpaths=[p for p in B.rglob('*')if p.is_file()and str(p)not in oldpaths and p.name!='halfgrid-publication-check.json']
files={str(p):h(p)for p in newpaths}
for p in files:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw,p
 assert (bytes([47])+b'workspace'+bytes([47]))not in raw and (bytes([47])+b'root'+bytes([47,46]))not in raw,p
 raw.decode()
out={'status':'passed','previous_candidate':previous,'protected_old_files':len(b['sha256']),'prior_result_files_preserved':1353,'previous_stageA_artifacts_byte_preserved':len(oldpaths),'old_marker':'unchanged','new_marker_and_receipt':'absent','additional_real_camera_values_decoded_in_this_StageA':0,'prior_frozen_inputs':30,'current_mandatory_sha256':r['source_sha256'],'new_artifact_sha256':files,'boundary':'closure source/output bound by final Git tree, not self-hashed; no separate wait4 for this byte audit; targeted fixtures have actual parent wait4 and archived source/logs'}
(B/'halfgrid-publication-check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ('new_artifact_sha256','current_mandatory_sha256')},indent=2))
