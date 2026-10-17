"""Final closure for publication; saved text/hash arithmetic only, no GT reads."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
B=Path('results/0003-fortran-stageB-20261017');E=B/'evidence';h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline=json.loads((E/'protected-baseline.json').read_text())
for p,d in baseline['sha256'].items():assert h(p)==d,p
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show',baseline['source_commit']+':notes/2026-10-17.md']))
spec=importlib.util.spec_from_file_location('v2','scripts/audit_0003_fortran_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);receipt=m.verify_published()
for phase,code in [('preflight',0),('audit',1),('independent-verification',0),('software',0)]:
 r=json.loads((E/(phase+'-parent.json')).read_text());assert r['observed_exit_code']==code and r['raw_wait_status']==(256 if code else 0)
 assert r['parent_sha256']==h(E/'parent-source.py')
 for p,d in r['source_sha256'].items():assert h(p)==d,p
 for st,d in r['logs_sha256'].items():assert h(E/(phase+'.'+st+'.log'))==d
v=json.loads((E/'independent-verification.json').read_text());assert v['audit_json_sha256']==h(B/'audit/audit.json')
a=json.loads((B/'audit/audit.json').read_text());assert a['status']=='stopped'and a['reason']=='storage-aware proper rotation gate'and len(a['point_checks'])==1
assert '"stdlib_tests": 76'in(E/'software.stdout.log').read_text()
excluded={E/'publication-verification.json',E/'publication-verification-parent.json',E/'publication-verification.stdout.log',E/'publication-verification.stderr.log'}
files={str(p):h(p)for p in B.rglob('*')if p.is_file()and p not in excluded}
for p in files:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw,p
 assert(bytes([47])+b'workspace'+bytes([47])+b'scratch'+bytes([47]))not in raw,p
 assert(bytes([47])+b'root'+bytes([47])+b'.')not in raw,p
 raw.decode()
r={'status':'passed','execution_source_commit':baseline['source_commit'],'input_commit':receipt['input_commit'],'receipt_sha256':h(m.RECEIPT),'mandatory_frozen_inputs':len(receipt['inputs_sha256']),'protected_files':len(baseline['sha256']),'audit_status':a['status'],'geometry_pass':False,'actual_audit_invocations':1,'artifact_sha256':files,'exclusions':[str(p)for p in sorted(excluded)],'boundary':'current parent closes its source/log/exit after child; final Git tree binds exclusions/self; no more supervisor exit for parent','no_new_archive_or_NPZ_read':True}
(E/'publication-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='artifact_sha256'},indent=2))
