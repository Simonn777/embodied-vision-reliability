"""Post-receipt byte/publication check; does not run tests or scientific data."""
import datetime,hashlib,json,subprocess
from pathlib import Path
from evr.motion_design import MANDATORY,RECEIPT,verify_design
B=Path('results/0003-data-design-20261017');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();BASE='a2d4827c8d83d3745989d57c901dcddf16642e06'
verified=verify_design(Path.cwd());receipt=json.loads(Path(RECEIPT).read_text());assert verified['input_commit']=='66e5330ad7404b4e7678d006e2537e4703f4683d'
for p,h in receipt['inputs_sha256'].items():assert sha(p)==h
for p,h in json.loads((B/'review-validation.json').read_text())['mandatory_input_sha256_at_test_time'].items():assert sha(p)==h
baseline=json.loads((B/'baseline.json').read_text())
for p,h in baseline['protected_sha256'].items():
 assert sha(p)==h
 assert hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+p])).hexdigest()==h
for p in ['notes/2026-10-17.md']:
 assert Path(p).read_bytes().startswith(subprocess.check_output(['git','show',BASE+':'+p]))
assert Path('README.md').read_bytes()==subprocess.check_output(['git','show',BASE+':README.md'])
for p,h in json.loads((B/'review-validation.json').read_text())['closed_evidence_sha256'].items():assert sha(p)==h
phase_records={}
for phase in ['acquisition','fixtures','validation','validation-final','closure','review','design-gate','design-gate-refreshed']:
 r=json.loads((B/'execution'/(phase+'.json')).read_text());assert r['exit_code']==(1 if phase=='design-gate'else 0);assert r['raw_wait_status']==(256 if phase=='design-gate'else 0)
 assert r['AS_bytes']==6*1024**3 and r['timeout_seconds']==900 and len(r['affinity'])==2
 for stream,h in r['log_sha256'].items():
  if phase=='design-gate'and stream=='stderr':
   recovery=json.loads((B/'gate-environment-recovery.json').read_text());assert h==recovery['raw_stderr_sha256'];assert sha(B/'execution/design-gate.stderr.public.log')==recovery['public_redacted_stderr_sha256']
  else:assert sha(B/'execution'/(phase+'.'+stream+'.log'))==h
 phase_records[phase]={'exit':r['exit_code'],'raw_wait':r['raw_wait_status'],'wall_seconds':r['elapsed_seconds'],'peak_rss_kib':r['peak_rss_kib']}
exclude={B/'publication-check.json',*[B/'execution'/('publication-audit.'+s)for s in ['json','stdout.log','stderr.log']]}
closed={str(p):sha(p)for p in B.rglob('*')if p.is_file()and p not in exclude}
for p in closed:
 raw=Path(p).read_bytes();assert len(raw)<1048576 and b'\0'not in raw and str(Path.cwd()).encode()not in raw;raw.decode()
out={'status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'published_input_commit':verified['input_commit'],'published_receipt_commit':'adb3cf20537c108ecc913022ec1bb3b7b2c0f5a4','canonical_receipt_sha256':sha(RECEIPT),'freeze_verification':verified,'input_files_verified':len(MANDATORY),'protected_old_paths_verified':len(baseline['protected_sha256']),'final_new_unittest_fixtures':35,'data_readiness':'BLOCKED_NO_FINAL_COHORT','new_model_inferences':0,'old_result_reanalyses':0,'new_data_weights_dependencies_bytes':0,'phase_observations':phase_records,'closed_artifact_sha256':closed,'exclusions':[str(p)for p in sorted(exclude)],'boundary':'current audit parent closes actual child exit/logs afterwards; final Git tree binds self and those files; parent own exit has no further supervisor receipt'}
with(B/'publication-check.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k!='closed_artifact_sha256'},indent=2))
