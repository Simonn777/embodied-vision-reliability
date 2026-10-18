import hashlib,importlib.util,json,sys,time,platform
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/Users/simonxie/Documents/Codex/2026-10-12/ban/work/embodied-vision-reliability')
def readmodule(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
t=readmodule('independent_recovery_stage_fixtures',ROOT/'tests/test_tapnext_recovery.py')
s=readmodule('independent_recovery_supervisor',ROOT/'scripts/run_tapnext_recovered_capability.py')
orig_save=t.h.save
def saving(path,obj,*a,**kw):
 if Path(path).name=='original.json':obj={**obj,'cpu_affinity':[0,1,2,3],'address_space_bytes':8*1024**3}
 return orig_save(path,obj,*a,**kw)
results=[];start=time.monotonic()
with patch.object(t.h,'save',side_effect=saving):
 case=t.RecoveryStageTests()
 case.setUp()
try:
 # Fixture bytes only. Copy the real preparer's code so the supervisor binds the producer hash.
 p=case.repo/'scripts/prepare_tapnext_recovery.py';p.parent.mkdir();p.write_bytes((ROOT/'scripts/prepare_tapnext_recovery.py').read_bytes())
 p=case.repo/s.CONFIG_PATH;p.parent.mkdir();p.write_bytes(case.cp.read_bytes())
 before=t.m.inventory(case.failed)
 assert case.run_stage('dependencies')==0 and case.calls==0 and case.import_calls==1
 assert case.run_stage('checkpoint')==0 and case.calls==1 and case.import_calls==1
 receipt_hashes=s.verify_recovery_preparation(case.repo,case.old,case.failed,case.root)
 assert set(receipt_hashes)=={'dependencies','checkpoint'}
 assert before==t.m.inventory(case.failed)
 results.append({'case':'real_producer_to_supervisor_with_tiny_assets_and_mocked_import_HTTP','status':'passed','dependency_import_stubs':case.import_calls,'HTTP_stubs':case.calls,'failed_cache_unchanged':True})
 p=case.root/'weights/tapnextpp_ckpt.pt';p.write_bytes(b'changed')
 try:s.verify_recovery_preparation(case.repo,case.old,case.failed,case.root)
 except ValueError as error:
  assert 'checkpoint bytes differ' in str(error)
  results.append({'case':'changed_actual_checkpoint_rejected','status':'passed','error':str(error)})
 else:raise AssertionError('changed checkpoint accepted')
finally:case.tearDown()
case=t.RecoveryStageTests();case.setUp()
try:
 (case.failed/'unbound').write_bytes(b'')
 try:case.run_stage('dependencies')
 except ValueError:
  assert not case.root.exists() and case.calls==case.import_calls==0
  results.append({'case':'failed_cache_mutation_stops_before_new_directory_import_HTTP','status':'passed'})
 else:raise AssertionError('mutated failure cache accepted')
finally:case.tearDown()
assert not any(x in sys.modules for x in ['torch','torchvision','numpy','tapnet'])
files=['configs/0005-native-recovery.json','protocols/0005-native-recovery.md','scripts/prepare_tapnext_recovery.py','scripts/run_tapnext_recovered_capability.py','tests/test_tapnext_recovery.py','tests/test_tapnext_recovered_supervisor.py']
result={'status':'passed','checks':results,'elapsed_seconds':time.monotonic()-start,'python':platform.python_version(),'platform':platform.system(),'reviewed_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in files},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'boundary':'Local synthetic files, stub HTTP, stub import child, runtime affinity/platform/limits mocked. Tests real producer plus real verifier; no network, real dependencies, checkpoints, data, model or science.'}
Path('/tmp/evr-0005-recovery-review/integration.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
