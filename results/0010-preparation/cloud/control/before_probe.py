"""Saved stage and byte checks only, before the unique frozen probe."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0010-capability'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
m={'__name__':'_guard','__file__':str(HERE/'preflight.py')};exec(compile((HERE/'preflight.py').read_bytes(),str(HERE/'preflight.py'),'exec'),m);m['gate']()
for label in ('wheels','offline-install','imports','assets'):
 e=json.loads((HERE/(label+'.execution.json')).read_text());assert e['actual_exit']==e['raw_wait_status']==0
 for suffix,key in (('.command.json','command_sha256'),('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256')):assert sha(HERE/(label+suffix))==e[key]
ledger=json.loads((ROOT/'clip-0010-downloads/ledger.json').read_text());assert ledger['status']=='completed' and ledger['actual_body_bytes']<=527440850
sup=json.loads((HERE/'assets/execution.json').read_text());assert sup['child_exit_code']==sup['supervisor_exit_code']==0 and not sup['timed_out']
for n,d in sup['logs'].items():assert sha(HERE/'assets'/n)==d['sha256']
cfg=json.loads((REPO/'experiments/0010/config.json').read_text());identities={}
for n,s in cfg['assets'].items():
 p=ROOT/'clip-0010-assets'/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==s['bytes']
 h=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0');hh=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b);hh.update(b)
 identities[n]={'bytes':p.stat().st_size,'sha256':hh.hexdigest(),'git_blob_sha1':h.hexdigest()};assert all(v is None or identities[n][k]==v for k,v in s.items())
assert not list((ROOT/'clip-0010-downloads').glob('*.partial'))
assert not (ROOT/'scratch/0010-capability-output').exists() and not (ROOT/'clip-0010-assets/0010-probe-started.json').exists()
memory={n:Path('/sys/fs/cgroup/'+n).read_text() for n in ('memory.max','memory.current','memory.stat','memory.events')};assert memory['memory.max'].strip()=='8589934592';assert {0,1,2,3}<=os.sched_getaffinity(0)
d={'status':'PASS','frozen_inputs':25,'assets':identities,'asset_ledger_sha256':sha(ROOT/'clip-0010-downloads/ledger.json'),'asset_body_bytes':ledger['actual_body_bytes'],'physical_asset_requests':len(ledger['requests']),'cgroup_before_probe':memory,'available_affinity':sorted(os.sched_getaffinity(0)),'note':'8GiB is a limit, not free RAM; caches were not purged and limits not raised.','source_sha256':sha(__file__)}
with (HERE/'before-probe.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','assets':len(identities),'asset_body_bytes':ledger['actual_body_bytes'],'physical_requests':len(ledger['requests']),'cgroup_memory_current':memory['memory.current'].strip(),'before_probe_sha256':sha(HERE/'before-probe.json')}))
