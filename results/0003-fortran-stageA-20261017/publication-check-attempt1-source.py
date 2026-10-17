"""Byte/source/log closure only; no dataset/archive or old-result numeric parser."""
import hashlib,json,subprocess
from pathlib import Path
B=Path('results/0003-fortran-stageA-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=json.loads((B/'protected-baseline.json').read_text())
for p,d in b['sha256'].items():assert h(p)==d,p
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show',b['main']+':notes/2026-10-17.md']))
assert not Path('protocols/0003-fortran-audit-freeze-v2.json').exists()
last=json.loads((B/'fixtures-final-parent.json').read_text())
for p,d in last['source_sha256'].items():assert h(p)==d,p
for phase in ['fixtures-v1','fixtures-v2','fixtures-final','toy-smoke']:
 r=json.loads((B/(phase+'-parent.json')).read_text());assert r['observed_exit_code']==r['raw_wait_status']==0
 for p,d in r['source_sha256'].items():assert h(B/(phase+'-source')/p)==d,p
 for st,d in r['logs_sha256'].items():assert h(B/(phase+'.'+st+'.log'))==d
 assert h(B/('fixture-parent-v1-source.py'if phase=='fixtures-v1'else'fortran_stageA_parent.py'))==r['parent_source_sha256']
assert '"stdlib_tests": 75' in (B/'fixtures-final.stdout.log').read_text()
assert '1240 scored frames' in (B/'toy-smoke.stdout.log').read_text()
files={str(p):h(p)for p in B.rglob('*')if p.is_file()and p.name!='publication-check.json'}
for p in files:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw and b'/workspace/'not in raw and b'/root/'not in raw,p;raw.decode()
source_map=json.loads((B/'execution-source-map.json').read_text())
source_map['synthetic_source_sha256']={n:h(B/n)for n in ['stageA_software.py','stageA_smoke.py']}
(B/'execution-source-map.json').write_text(json.dumps(source_map,indent=2)+'\n');files[str(B/'execution-source-map.json')]=h(B/'execution-source-map.json')
r={'status':'passed','execution_base':b['main'],'protected_files':len(b['sha256']),'notes':'append-only prefix matches exact main','candidate_mandatory_sha256':last['source_sha256'],'canonical_v2_receipt':'ABSENT; review only','numeric_GT_decode':0,'new_GET':0,'new_model_run':0,'stageA_artifact_sha256':files,'self_exclusion':'publication-check.json and its enclosing command output are bound only by final Git tree; no own wait receipt claimed'}
(B/'publication-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k not in ['stageA_artifact_sha256','candidate_mandatory_sha256']},indent=2))
