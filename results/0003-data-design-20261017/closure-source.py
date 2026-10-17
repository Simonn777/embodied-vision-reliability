"""Close saved evidence with explicit current-phase exclusions; no science runs."""
import datetime,hashlib,json,subprocess
from pathlib import Path
B=Path('results/0003-data-design-20261017');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
repairs=[]
for name in ['software-validation.json','software-validation-v2.json']:
 r=json.loads((B/name).read_text())
 for path,digest in r['closed_evidence_sha256'].items():
  actual=sha(path)
  if actual!=digest:
   assert name=='software-validation-v2.json'and path in [str(B/'execution/validation-final.json'),str(B/'execution/validation-final.stdout.log')]
   repairs.append({'map':str(B/name),'path':path,'recorded_in_progress_sha256':digest,'closed_sha256':actual,'reason':'original map accidentally included current parent placeholder receipt/current stdout before child exit; this closure uses stable completed bytes, original map retained'})
assert len(repairs)==2
baseline=json.loads((B/'baseline.json').read_text())
for path,digest in baseline['protected_sha256'].items():assert sha(path)==digest
for phase in ['acquisition','fixtures','validation','validation-final']:
 r=json.loads((B/'execution'/(phase+'.json')).read_text());assert r['exit_code']==r['raw_wait_status']==0
 assert r['parent_source_sha256']==sha(B/'execution/parent-source.py')
 for stream,digest in r['log_sha256'].items():assert sha(B/'execution'/(phase+'.'+stream+'.log'))==digest
 if r['child_source_sha256']:
  source={'acquisition':B/'acquisition-source.py','validation':B/'software-validation-source.py','validation-final':B/'software-validation-v2-source.py'}[phase];assert sha(source)==r['child_source_sha256']
# Scientific module/tests/config/protocol unchanged after final software run.
r=json.loads((B/'software-validation-v2.json').read_text())
for path,digest in r['mandatory_input_sha256_at_test_time'].items():assert sha(path)==digest
exclude={B/'evidence-closure.json',*[B/'execution'/('closure.'+suffix)for suffix in ['json','stdout.log','stderr.log']]}
files={str(p):sha(p)for p in B.rglob('*')if p.is_file()and p not in exclude}
for path in files:
 data=Path(path).read_bytes();assert len(data)<1024**2 and b'\0'not in data and str(Path.cwd()).encode()not in data;data.decode()
out={'status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'artifact closure repair only; no tests/model/inference/evaluation/old result parsing repeated','repairs':repairs,'preserved_old_paths':len(baseline['protected_sha256']),'final_software_tests':r['tests_run'],'closed_sha256':files,'exclusions':[str(p)for p in sorted(exclude)],'boundary':'current closure parent wait/log record is closed after this child; final Git tree binds all four excluded files, without self-reference'}
with(B/'evidence-closure.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k!='closed_sha256'},indent=2))
