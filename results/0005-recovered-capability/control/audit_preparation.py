"""Independent stdlib byte/receipt checks; never imports a tracker or decoder."""
import hashlib,json,sys
from pathlib import Path
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0005-recovery';F=B/'tapnextpp-cache';N=B/'tapnextpp-recovery';C=Path(__file__).parent
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def inventory(root):
 result={}
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():result[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
 return result
def closed(prefix,base):
 rec=load(base/(prefix+'.json'));assert rec['exit_code']==0 and rec['raw_wait_status']==0
 for s in ['stdout','stderr']:assert sha(base/(prefix+'.'+s+'.log'))==rec[s+'_sha256']
 return rec
def main():
 stage=sys.argv[1];assert stage in ['dependencies','checkpoint'];assert not(C/(stage+'-audit.json')).exists()
 outer=closed(stage+'-execution',C)
 # External logs use stage.stdout/stderr rather than stage-execution.stdout/stderr.
 return stage,outer
def audit(stage):
 outer=load(C/(stage+'-execution.json'));assert outer['exit_code']==0 and outer['raw_wait_status']==0
 for s in ['stdout','stderr']:assert sha(C/(stage+'.'+s+'.log'))==outer[s+'_sha256']
 assert sha(C/'observe_recovery.py')==outer['observer_source_sha256']
 dep=load(N/'recovery-dependencies.json');assert dep['status']=='completed' and dep['preserved_failed_cache']is True
 assert dep['new_network_payload_bytes']==0 and dep['model_constructed']is False and dep['checkpoint_started']is False
 assert dep['config_sha256']==sha(R/'configs/0005-native-recovery.json')
 assert dep['source_sha256']==sha(R/'scripts/prepare_tapnext_recovery.py')
 assert dep['marker_sha256']==sha(N/'markers/dependencies.started.json')
 imp=closed('import-check',N);assert imp==dep['import_check']
 probe=load(N/'import-check.stdout.log');assert probe['status']=='passed' and probe['no_model_constructed_or_forward_called']is True
 cfg=load(R/'configs/0005-native-capability.json');prep=cfg['preparation'];pkgs={**prep['runtime']['old_packages'],**prep['runtime']['overlay_packages']}
 for name,version in pkgs.items():
  row=probe['imports'][name];assert row['version']==row['module_version']==version
  expected=B/'bootstapir-cache/runtime-fixed'if name in ['torch','numpy']else F/'overlay'
  assert Path(row['path']).is_relative_to(expected)
 assert inventory(F)==load(R/'results/0005-native-capability/review/postfailure-cache-inventory.json')
 for p,h in dep['old_preservation'].items():assert sha(B/'bootstapir-cache'/p)==h
 for kind,root in [('source',N/'source'),('overlay',F/'overlay')]:
  ref=dep[kind+'_manifest'];assert sha(N/ref['relative_path'])==ref['sha256']
  assert inventory(root)==load(N/ref['relative_path'])['files']
  if kind=='source':assert len(inventory(root))==5
 oldledger=load(F/'acquisition-ledger.json');dledger=load(N/'acquisition-ledger-dependencies.json')
 assert sha(N/'acquisition-ledger-dependencies.json')==dep['ledger_snapshot']['sha256']
 assert dledger['payload_bytes_observed']==oldledger['payload_bytes_observed']==9179574
 assert dledger['new_payload_bytes_observed']==0
 item={'status':'passed','scope':'saved_receipt_and_byte_checks_only','stage':stage,'actual_outer_exit':outer['exit_code'],'actual_outer_raw_wait':outer['raw_wait_status'],'import_exit':imp['exit_code'],'failed_cache_files_preserved':len(inventory(F)),'source_files':5,'overlay_files':len(load(N/'overlay-manifest.json')['files']),'new_dependency_payload_bytes':0,'import_versions':pkgs,'dependency_receipt_sha256':sha(N/'recovery-dependencies.json')}
 if stage=='checkpoint':
  ck=load(N/'recovery-checkpoint.json');assert ck['status']=='completed' and ck['preserved_failed_cache']is True
  assert ck['dependency_receipt_sha256']==sha(N/'recovery-dependencies.json')
  a=ck['checkpoint_asset'];assert a==load(N/'asset-records'/('checkpoint.json'))
  spec=next(x for x in prep['assets']if x['kind']=='checkpoint');assert a['asset']==spec
  assert a['status']=='downloaded_verified' and a['http_payload_attempts']==1 and a['http_status']==200
  assert a['payload_bytes_observed']==spec['bytes']==2532282370
  assert a['observed_payload_sha256']==a['sha256']==spec['sha256']
  p=N/spec['relative_path'];assert p.stat().st_size==spec['bytes']and sha(p)==spec['sha256']
  assert a['partial_exists']is False and a['partial_bytes_stored']==0 and not p.with_name(p.name+'.partial').exists()
  assert sha(N/'asset-records/checkpoint.json')==ck['checkpoint_asset_record_sha256']
  assert sha(N/'acquisition-ledger-checkpoint.json')==ck['ledger_snapshot']['sha256']
  led=load(N/'acquisition-ledger.json');assert led==load(N/'acquisition-ledger-checkpoint.json')
  assert led['new_payload_bytes_observed']==2532282370 and led['payload_bytes_observed']==2541461944
  assert led['conservative_cumulative_ceiling_bytes']==7499066782
  item.update(checkpoint_bytes=spec['bytes'],checkpoint_sha256=spec['sha256'],checkpoint_receipt_sha256=sha(N/'recovery-checkpoint.json'),conservative_cumulative_ceiling_bytes=led['conservative_cumulative_ceiling_bytes'])
 else:
  assert not(N/'recovery-checkpoint.json').exists()
  item['conservative_cumulative_ceiling_bytes']=dledger['conservative_cumulative_ceiling_bytes']
 with(C/(stage+'-audit.json')).open('x')as f:json.dump(item,f,indent=2);f.write('\n')
 print(json.dumps(item,indent=2))
if __name__=='__main__':audit(sys.argv[1])
