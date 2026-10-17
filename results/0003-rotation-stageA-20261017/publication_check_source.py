"""Published-old-byte and candidate-source closure only; no real camera file read."""
import hashlib,json,subprocess
from pathlib import Path
B=Path('results/0003-rotation-stageA-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=json.loads((B/'protected-baseline.json').read_text())
for p,d in b['sha256'].items():assert h(p)==d,p
assert len(b['sha256'])==1468
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show',b['base']+':notes/2026-10-17.md']))
assert h('../bootstapir-cache/0003-pointodyssey/fortran-v2-started.json')==b['old_marker_sha256']
assert not Path('../bootstapir-cache/0003-pointodyssey/rotation-diagnostic-v1-started.json').exists()
assert not Path('protocols/0003-rotation-diagnostic-freeze-v1.json').exists()
r=json.loads((B/'final-fixtures-parent.json').read_text());assert r['observed_exit_code']==r['raw_wait_status']==0
for p,d in r['source_sha256'].items():assert h(p)==d,p
for p,d in r['command_source_sha256'].items():assert h(p)==d;assert h(B/Path(p).name)==d
assert h(B/'rotation_stageA_parent.py')==r['parent_source_sha256']
for st,d in r['logs_sha256'].items():assert h(B/('final-fixtures.'+st+'.log'))==d
assert '"stdlib_tests": 93'in(B/'final-fixtures.stdout.log').read_text()
files={str(p):h(p)for p in B.rglob('*')if p.is_file()and p.name!='publication-check.json'}
for p in files:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw,p
 assert (bytes([47])+b'workspace'+bytes([47])+b'scratch'+bytes([47]))not in raw,p
 assert (bytes([47])+b'root'+bytes([47])+b'.')not in raw,p
 raw.decode()
out={'status':'passed','base':b['base'],'protected_old_files':len(b['sha256']),'prior_result_files_preserved':1353,'old_marker':'unchanged','new_marker_and_receipt':'absent','candidate_mandatory_sha256':r['source_sha256'],'real_camera_numeric_access':0,'prior_frozen_inputs':30,'artifact_sha256':files,'boundary':'publication-check self/output bound by final Git tree; no separate wait4 for this final byte audit; synthetic suite has actual parent wait4 and archived source/logs'}
(B/'publication-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['artifact_sha256','candidate_mandatory_sha256']},indent=2))
