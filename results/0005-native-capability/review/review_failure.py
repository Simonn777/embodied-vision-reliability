"""Read-only saved-evidence audit. Does not import ML packages or execute science."""
import hashlib,json,os,platform,subprocess
from pathlib import Path
BASE=Path('/workspace/scratch/a349d03a0e0a')
REPO=BASE/'research-0005'; CACHE=BASE/'tapnextpp-cache'; OUT=Path(__file__).parent
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def save(name,x):
 p=OUT/name
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 receipt=load(CACHE/'preparation-dependencies.json');outer=load(OUT/'dependencies-execution.json')
 assert receipt['status']=='failed' and outer['exit_code']==1 and outer['raw_wait_status']==256
 assert receipt['error']=={'type':'PrepFailure','reason':'Unexpected or changed source entry: tapnet/tapnext/tapnext_lru_modules.py.partial'}
 for mode in ['verify','dependencies','software']:
  r=load(OUT/(mode+'-execution.json'))
  for stream in ['stdout','stderr']:assert sha(OUT/(mode+'.'+stream+'.log'))==r[stream+'_sha256']
  assert sha(OUT/'stage_parent.py')==r['source_sha256']
 install=load(CACHE/'install.json');assert install==receipt['installation'] and install['exit_code']==0 and install['raw_wait_status']==0
 for stream in ['stdout','stderr']:assert sha(CACHE/('install.'+stream+'.log'))==install[stream+'_sha256']
 cfg=load(REPO/'configs/0005-native-capability.json')['preparation']; assets={a['id']:a for a in cfg['assets']}
 payload=0
 for a in receipt['assets']:
  spec=assets[a['asset']['id']];assert a['asset']==spec
  assert a==load(CACHE/'asset-records'/(spec['id']+'.json'))
  p=CACHE/spec['relative_path'];assert p.stat().st_size==spec['bytes'] and sha(p)==spec['sha256']
  assert a['status']=='downloaded_verified' and a['http_status']==200 and a['http_payload_attempts']==1
  assert a['payload_bytes_observed']==spec['bytes'];payload+=a['payload_bytes_observed']
 assert len(receipt['assets'])==8 and payload==9179574
 ledger=load(CACHE/'acquisition-ledger.json')
 assert ledger==load(CACHE/'acquisition-ledger-dependencies.json')
 assert sha(CACHE/'acquisition-ledger-dependencies.json')==receipt['ledger_snapshot']['sha256']
 assert ledger['payload_bytes_observed']==payload and ledger['conservative_cumulative_ceiling_bytes']==4966784411
 overlay={};inventory={}
 for p in sorted(CACHE.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():
   s=p.stat();entry={'bytes':s.st_size,'sha256':sha(p)};inventory[p.relative_to(CACHE).as_posix()]=entry
   if p.is_relative_to(CACHE/'overlay'):overlay[p.relative_to(CACHE/'overlay').as_posix()]=entry
 assert overlay and not(CACHE/'overlay-manifest.json').exists()
 save('postfailure-overlay-manifest.json',{'status':'postfailure_inventory_only_not_preparation_gate_approval','files':overlay})
 save('postfailure-cache-inventory.json',inventory)
 baseline=load(OUT/'tracked-baseline.json');assert all(sha(REPO/p)==h for p,h in baseline.items())
 frozen=load(REPO/'protocols/0005-native-capability-freeze.json')['files']
 for p,h in frozen.items():
  assert sha(REPO/p)==h
  for ref in ['HEAD',':']:
   spec=':'+p if ref==':' else ref+':'+p
   assert hashlib.sha256(subprocess.check_output(['git','show',spec],cwd=REPO)).hexdigest()==h
 oldmarkers=load(OUT/'old-cache-markers.json')
 assert all(sha(BASE/'bootstapir-cache'/p)==h for p,h in oldmarkers.items())
 forbidden=['preparation-checkpoint.json','markers/checkpoint.started.json','0005-native-started.json','import-check.json']
 assert not any((CACHE/p).exists()for p in forbidden)
 assert not(BASE/'scratch/0005-native-capability').exists()
 save('failure-review.json',{'status':'passed_saved_failure_evidence_checks','dependency_status':'failed','dependency_process_exit':1,'dependency_raw_wait':256,'installation_exit':0,'import_check':'not_started','checkpoint':'not_started','native':'not_started','actual_new_asset_payload_bytes':payload,'historical_acquisition_ceiling_bytes':4823387101,'historical_ceiling_plus_new_payload_bytes':4823387101+payload,'conservative_cumulative_ceiling_bytes':ledger['conservative_cumulative_ceiling_bytes'],'full_plan_ceiling_bytes':ledger['planned_ceiling_bytes'],'asset_count':8,'overlay_files':len(overlay),'overlay_inventory_sha256':sha(OUT/'postfailure-overlay-manifest.json'),'cache_inventory_sha256':sha(OUT/'postfailure-cache-inventory.json'),'unexpected_source_partials':{p:v for p,v in inventory.items()if p.startswith('source/')and p.endswith('.partial')},'baseline_files_preserved':len(baseline),'prior_result_files_preserved':sum(p.startswith('results/')for p in baseline),'old_markers_preserved':oldmarkers,'frozen_files_unchanged':frozen,'platform_observation':{'python':platform.python_version(),'platform':platform.platform(),'current_affinity':sorted(os.sched_getaffinity(0))},'interpretation':'The frozen source allowlist rejected zero-byte partial entries after successful asset verification and installation. The mechanism creating/persisting those entries was not investigated or repaired. No dependency import check, checkpoint GET, model construction, forward or GT access occurred. Postfailure inventory is not a completed preparation manifest. Source/asset bytes are not published.'})
 print(json.dumps(load(OUT/'failure-review.json'),indent=2))
if __name__=='__main__':main()
