"""Bounded software + preservation audit, not an experiment."""
import datetime,hashlib,importlib.util,json,subprocess,sys,unittest
from pathlib import Path
B=Path('results/0003-data-design-20261017');BASE='a2d4827c8d83d3745989d57c901dcddf16642e06';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
from evr.motion_design import MANDATORY,readiness,decompose
source={p:sha(p)for p in MANDATORY}
suite=unittest.defaultTestLoader.discover('tests',pattern='test_motion_design.py');result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
b=json.loads((B/'baseline.json').read_text());assert len(b['protected_sha256'])==1149
for p,h in b['protected_sha256'].items():
 assert sha(p)==h
 assert hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+p])).hexdigest()==h
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show',BASE+':notes/2026-10-17.md']))
r=json.loads((B/'readiness.json').read_text());assert readiness(r['evidence'])['status']=='BLOCKED_NO_FINAL_COHORT';assert r['final_cohort_manifest']is None
http=json.loads((B/'http-audit.json').read_text());assert len(http['events'])==11 and all(e['bytes_received']==0 and e.get('error_type')=='URLError'for e in http['events'])
ledger=json.loads((B/'acquisition-ledger.json').read_text());assert ledger['updated_conservative_ceiling_bytes']<ledger['budget_bytes']and ledger['new_data_weights_dependency_bytes']==0
for phase in ['acquisition','fixtures']:
 receipt=json.loads((B/'execution'/(phase+'.json')).read_text());assert receipt['exit_code']==receipt['raw_wait_status']==0
 for stream,h in receipt['log_sha256'].items():assert sha(B/'execution'/(phase+'.'+stream+'.log'))==h
assert sha(B/'acquisition-source.py')==json.loads((B/'execution/acquisition.json').read_text())['child_source_sha256']
assert json.loads((B/'software-probe.json').read_text())['pytest_available']is False
# Separate, hand arithmetic checks rather than recomputing old observations.
d=decompose([0,1,2],[(0,0),(0,0),(3,4)],[(0,0),(0,0),(3,0)])
assert d['d_pred']==5 and d['d_gt_image']==3 and d['d_error']==4
assert d['r_gt_image'][0]+d['r_error'][0]==d['r_pred'][0]and d['r_gt_image'][1]+d['r_error'][1]==d['r_pred'][1]
for p,h in source.items():assert sha(p)==h
out={'status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'new software hand fixtures + immutable file byte audit only; no model/archive/old result parsing','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'protected_old_files':1149,'old_test_reanalyses':0,'model_inferences':0,'archive_decodes':0,'mandatory_input_sha256_at_test_time':source,'parent_source_sha256':sha(B/'execution/parent-source.py'),'audit_source_sha256':sha(__file__),'pytest_available':False,'pytest_tests_in_Work':0,'data_readiness':'BLOCKED_NO_FINAL_COHORT','fixture_units':'world coordinates per timestamp-unit squared; no verified dataset units/FPS','closed_evidence_sha256':{str(p):sha(p)for p in B.rglob('*')if p.is_file()and not p.name.startswith('validation.')}}
with(B/'software-validation.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['closed_evidence_sha256','mandatory_input_sha256_at_test_time']},indent=2))
