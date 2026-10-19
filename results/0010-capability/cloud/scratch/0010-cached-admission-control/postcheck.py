"""Read-only preservation and closed-receipt check, no model/accuracy."""
import hashlib,json,subprocess
from pathlib import Path
b=Path('/workspace/scratch/a349d03a0e0a');c=b/'scratch/0010-cached-admission-control';old=b/'research-0010-capability';new=b/'research-0010-cached-admission';ctl=b/'scratch/0010-capability-control'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def spec(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def stat(p):
 s=p.lstat();assert not p.is_symlink();return dict(dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def verify(p,s):assert spec(p)=={k:s[k] for k in ('bytes','sha256')}
f=json.loads((c/'preflight.json').read_text());base=json.loads((ctl/'pre-execution.json').read_text());guard=json.loads((b/'scratch/0010-cached-guard/guard.json').read_text())
for p,d in f['old_export_files'].items():verify(b/p,d)
for p,d in base['preserved_original_markers_and_preflight'].items():verify(b/p,d)
for p,h in base['tracked'].items():assert sha(old/p)==h
for repo,records in [(old,guard['source_checks']),(new,f['inputs'])]:
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
 for p,d in records.items():
  verify(repo/p,d)
  if 'before' in d:assert stat(repo/p)==d['before']
for p,d in guard['preserved'].items():verify(b/p,d);assert stat(b/p)==d['before']
for p,d in guard['assets'].items():verify(b/'clip-0010-assets'/p,d);assert stat(b/'clip-0010-assets'/p)==d['before']
for p,d in guard['runtime_recheck']['files'].items():verify(Path(p),d);assert stat(Path(p))==d['before']
phases={}
for label in ('cached-guard','probe','saved-probe-review','solver-functional','solver-timing'):
 e=json.loads((ctl/(label+'.execution.json')).read_text());assert e['actual_exit']==e['raw_wait_status']==0
 for suffix,key in [('.command.json','command_sha256'),('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256')]:assert sha(ctl/(label+suffix))==e[key]
 phases[label]=e
assert json.loads((ctl/'before-probe.execution.json').read_text())['actual_exit']==1
for root in ['scratch/0010-cached-guard-supervision','scratch/0010-capability-control/probe','scratch/0010-solver-functional/supervision','scratch/0010-solver-timing/supervision']:
 p=b/root;e=json.loads((p/'execution.json').read_text());assert e['supervisor_exit_code']==e['child_exit_code']==0
 for name,s in e['logs'].items():verify(p/name,s)
pr=json.loads((c/'saved-probe-review.json').read_text());assert pr['resource_admitted']
fr=json.loads((b/'scratch/0010-solver-functional/child/receipt.json').read_text());tr=json.loads((b/'scratch/0010-solver-timing/child/receipt.json').read_text())
assert fr['status']=='completed' and fr['tests']==dict(run=20,failures=0,errors=0,skipped=0)
assert tr['status']=='completed' and tr['resource_admitted'] and tr['inputs_unchanged'];assert [x['phase'] for x in tr['attempts']]==['T63','T641','T644'];assert all(x['status']=='completed' and x['native_solver_executions']==2 for x in tr['attempts'])
assert abs(tr['projected_seconds']-1.5*(128*tr['attempts'][0]['seconds']+2*sum(x['seconds'] for x in tr['attempts'][1:])))<1e-12
binary={}
for rel in ['scratch/0010-capability-output/features.npz','scratch/0010-capability-output/partial-features.npz','scratch/0010-solver-timing/child/artificial-inputs.npz']:binary[rel]=spec(b/rel)
markers={'clip-0010-assets/0010-probe-started.json':spec(b/'clip-0010-assets/0010-probe-started.json')}
d=dict(status='PASS_PRESERVATION_AND_SAVED_RECEIPTS',old_tracked_count=len(base['tracked']),old_markers_preflight_count=len(base['preserved_original_markers_and_preflight']),old_export_count=len(f['old_export_files']),new_frozen_inputs=18,old_frozen_inputs=25,runtime_files=142,phases=phases,probe_projection_seconds=pr['projection_seconds'],solver_tests=fr['tests'],timing_attempts=tr['attempts'],solver_projection_seconds=tr['projected_seconds'],solver_resource_admitted=True,binary_not_public=binary,new_markers=markers,old_failure_exit=1,new_asset_bytes=0,direct_body_charge_unchanged=456087034,historical_all_HTTP='UNKNOWN',no_accuracy_no_quality_study=True,source_main_field_boundary='capture.py source_main and cwd refer to old capability checkout; solver source is separately bound by new 18-input freeze and manifest.',memory_after={n:(Path('/sys/fs/cgroup')/n).read_text() for n in ['memory.current','memory.stat','memory.events']})
with (c/'postcheck.json').open('x') as out:json.dump(d,out,indent=2);out.write('\n')
print(json.dumps({k:d[k] for k in ['status','old_tracked_count','old_export_count','old_markers_preflight_count','probe_projection_seconds','solver_projection_seconds','old_failure_exit','new_asset_bytes']}))
