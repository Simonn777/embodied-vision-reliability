"""No dataset access: byte closure, logs, wait evidence and fixed source bindings."""
import hashlib,json,subprocess
from pathlib import Path

OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline=json.loads((OUT/'tracked-baseline.json').read_text())
mutable={'CURRENT_STATE.md','NEXT_ACTIONS.md','README.md'}
for name,want in baseline.items():
 assert hashlib.sha256(subprocess.check_output(['git','show','eb4769c23f5947e5789dbea82dfb8072ddf23726:'+name])).hexdigest()==want
 if name not in mutable:assert sha(name)==want,name
assert len(baseline)==1652
receipt=json.loads(Path('protocols/0003-annotation-consistency-freeze-v1.json').read_text())
assert len(receipt['inputs_sha256'])==45
for name,want in receipt['inputs_sha256'].items():assert sha(name)==want,name
gate=json.loads((OUT/'execution.json').read_text());out=json.loads((OUT/'gate.stdout.log').read_text())
assert gate['status']=='BLOCKED_CACHE_NOT_VISIBLE_NO_SCIENTIFIC_ATTEMPT'
assert gate['observed_exit_code']==gate['raw_wait_status']==0
assert gate['geometry_checks']==gate['model_calls']==gate['new_downloads_bytes']==0
assert out['published_gate']=='PASS'and out['mandatory_count']==45 and out['prior_distinct_inputs']==38
assert all(not value for value in gate['cache_stat_checks'].values())
for name,want in gate['log_sha256'].items():assert sha(OUT/name)==want
for label in ['synthetic','smoke']:
 record=json.loads((OUT/(label+'-execution.json')).read_text())
 assert record['observed_exit_code']==record['raw_wait_status']==0
 assert record['parent_sha256']==sha(OUT/'software_check_parent.py')
 for stream,want in record['logs_sha256'].items():assert sha(OUT/(label+'.'+stream+'.log'))==want
assert 'Ran 17 tests'in(OUT/'synthetic.stderr.log').read_text()
assert 'Completed 1240 scored frames'in(OUT/'smoke.stdout.log').read_text()
redaction=json.loads((OUT/'source-publication-note.json').read_text())
assert redaction['executed_script_sha256']==gate['script_sha256']
assert redaction['published_redacted_script_sha256']==sha('scripts/check_0003_cached_readiness_20261018.py')
assert redaction['executed_script_sha256']!=redaction['published_redacted_script_sha256']
files=sorted(p for p in OUT.rglob('*')if p.is_file()and p.name!='verification.json'and '__pycache__'not in p.parts)
for path in files:
 assert path.stat().st_size<1024**2
 text=path.read_text();assert '\0'not in text and ('/workspace/'+'scratch/')not in text
result={'status':'PASS','execution_baseline':'eb4769c23f5947e5789dbea82dfb8072ddf23726','baseline_files':len(baseline),'unchanged_except_three_mutable_status_files':len(baseline)-len(mutable),'historical_result_files_byte_preserved':sum(p.startswith('results/')for p in baseline),'frozen_inputs_unchanged':45,'gate_log_and_child_wait_verified':True,'software_logs_and_child_wait_verified':True,'scientific_attempts':0,'source_redaction_disclosed':True,'source_sha256':sha(__file__),'artifacts_sha256':{str(p):sha(p)for p in files},'boundary':'in-process assertions and file hashes; no claim of separately supervised verifier exit'}
(OUT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='artifacts_sha256'},indent=2))
