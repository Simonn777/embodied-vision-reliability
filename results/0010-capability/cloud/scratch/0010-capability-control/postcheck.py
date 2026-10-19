"""Read-only preservation/exit review after terminal pre-probe inspection failure."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0010-capability'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
base=load(HERE/'pre-execution.json')
for n,h in base['tracked'].items():assert sha(REPO/n)==h,n
for n,d in base['preserved_original_markers_and_preflight'].items():assert sha(ROOT/n)==d['sha256'] and (ROOT/n).stat().st_size==d['bytes'],n
m={'__name__':'_guard','__file__':str(HERE/'preflight.py')};exec(compile((HERE/'preflight.py').read_bytes(),str(HERE/'preflight.py'),'exec'),m);m['gate']()
phases={}
for label in ('preflight','software-tests','wheels','offline-install','imports','assets','before-probe'):
 d=load(HERE/(label+'.execution.json'))
 for suffix,key in (('.command.json','command_sha256'),('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256')):assert sha(HERE/(label+suffix))==d[key]
 assert d['capture_source_sha256']==sha(HERE/'capture.py')
 phases[label]={k:d[k] for k in ('actual_exit','raw_wait_status','wall_seconds','peak_rss_kib','user_cpu_seconds','system_cpu_seconds')}
assert phases['before-probe']['actual_exit']==1 and all(d['actual_exit']==0 for n,d in phases.items() if n!='before-probe')
assert not (ROOT/'scratch/0010-capability-output').exists() and not (ROOT/'clip-0010-assets/0010-probe-started.json').exists()
wheels=load(ROOT/'clip-0010-wheels/ledger.json');assets=load(ROOT/'clip-0010-downloads/ledger.json');assert wheels['status']==assets['status']=='completed'
for label in ('wheels','assets'):
 d=load(HERE/label/'execution.json');assert d['child_exit_code']==d['supervisor_exit_code']==0
 for n,v in d['logs'].items():assert sha(HERE/label/n)==v['sha256']
inventory={}
for kind in ('clip-0010-wheels','clip-0010-downloads','clip-0010-assets'):
 inventory[kind]=[{'path':str(p.relative_to(ROOT/kind)),'bytes':p.stat().st_size,'symlink':p.is_symlink()} for p in sorted((ROOT/kind).rglob('*')) if p.is_file()]
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO)
result={'status':'PRESERVATION_PASS_OPERATION_BLOCKED_BEFORE_PROBE','reason':'Non-frozen before_probe.py line24 asserts no residual .partial. It failed after completed acquisition and exact asset checks. Frozen scientific CLI probe was never invoked. Do not classify this as model success or model failure.','exception_sha256':sha(HERE/'before-probe.stderr.log'),'originals_preserved':{'tracked_files':len(base['tracked']),'frozen_inputs':25,'original_marker_and_preflight_files':len(base['preserved_original_markers_and_preflight'])},'phases':phases,'probe':{'invoked':False,'image_forward_attempts':0,'text_forward_attempts':0,'output_exists':False,'marker_exists':False},'partial_inspection_sha256':sha(HERE/'partial-inspection.json'),'assets_layout_sha256':sha(ROOT/'clip-0010-assets/layout-receipt.json'),'imports_sha256':sha(HERE/'imports.json'),'inventory':inventory,'budget':{'new_wheel_response_body_bytes':wheels['actual_body_bytes'],'new_asset_response_body_bytes':assets['actual_body_bytes'],'new_total_response_body_bytes':wheels['actual_body_bytes']+assets['actual_body_bytes'],'prior_direct_charges':5235758,'total_direct_charges':5235758+wheels['actual_body_bytes']+assets['actual_body_bytes'],'direct_cap':536870912,'wheel_cap':4194304,'asset_cap':527440850,'new_checkpoint_and_dataset_payload_bytes':353976522+94280567,'old_scientific_asset_ceiling':7532690072,'old_known_directory_increment':163286,'historical_all_HTTP_total':'UNKNOWN; browser/old redirect gap not repaired.'},'memory_after_failure':{n:Path('/sys/fs/cgroup/'+n).read_text() for n in ('memory.max','memory.current','memory.stat','memory.events')},'no_retries_no_deletions_no_model_calls':True,'published':False,'source_sha256':sha(__file__)}
with (HERE/'postcheck.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'preserved':result['originals_preserved'],'new_total_body_bytes':result['budget']['new_total_response_body_bytes'],'total_direct_charges':result['budget']['total_direct_charges'],'probe_invoked':False,'postcheck_sha256':sha(HERE/'postcheck.json')}))
