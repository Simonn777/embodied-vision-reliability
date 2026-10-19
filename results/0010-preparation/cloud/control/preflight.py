"""Read-only frozen Git binding and bounded exact-name reuse check."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0010-capability'
MAIN='e9e7c7609e53e1bc1b3228096f9b36008cb78d6b';SOURCE='ececb253bfa00acffdb8d84c91d848c2802444a0';FREEZE='93629d385b286b9471167e0c15c49bf29120096e';RECEIPT='protocols/0010-clip-capability-freeze.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO)
def save(name,d):
 with (HERE/name).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def gate():
 assert git('rev-parse','HEAD').decode().strip()==git('rev-parse','origin/main').decode().strip()==MAIN
 assert not git('status','--porcelain')
 assert git('rev-list','--parents','-n','1',FREEZE).decode().split()==[FREEZE,SOURCE]
 assert git('diff-tree','--no-commit-id','--name-only','-r',SOURCE,FREEZE).decode().splitlines()==[RECEIPT]
 subprocess.run(['git','merge-base','--is-ancestor',FREEZE,'HEAD'],cwd=REPO,check=True)
 raw=(REPO/RECEIPT).read_bytes();assert hashlib.sha256(raw).hexdigest()=='8dfec7362d01fac80c46f9871fe4e8696c90d12ba07038c2635a6f8dd947b29f'
 assert raw==git('show',FREEZE+':'+RECEIPT)==git('show','HEAD:'+RECEIPT)==git('show',':'+RECEIPT)
 f=json.loads(raw);assert f['source_commit']==SOURCE and len(f['inputs'])==25
 for n,s in f['inputs'].items():
  b=(REPO/n).read_bytes();assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==s['git_blob_sha1']
  for ref in (SOURCE+':'+n,'HEAD:'+n,':'+n):assert git('show',ref)==b,n
 return f
def main():
 f=gate();assert Path('/sys/fs/cgroup/memory.max').read_text().strip()=='8589934592'
 assert {0,1,2,3}<=os.sched_getaffinity(0)
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal():continue
  try:a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except OSError:continue
  scripts=[x for x in a if '\n' not in x and x.endswith('.py')]
  relevant=[x for x in scripts if ('experiments/0010/' in x and Path(x).name in ('cli.py','run_capability.py')) or Path(x).name in ('run_antithetic_training.py','run_antithetic_gradient_screen.py','run_geometric_stage0.py')]
  if relevant:active.append({'pid':int(p.name),'scripts':relevant})
 assert not active,active
 tracked={n:sha(REPO/n) for n in git('ls-files','-z').decode().split('\0') if n}
 preserved={}
 for base in ('bootstapir-cache','tapnextpp-cache','tapnextpp-recovery','conditional-residual-0006','antithetic-gradient-0007','antithetic-training-0008','scratch/0010-preflight'):
  d=ROOT/base
  for p in d.iterdir():
   if p.is_file() and not p.is_symlink() and (base.endswith('0010-preflight') or any(s in p.name for s in ('started','completed','marker','lock'))):preserved[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
 wheels=json.loads((REPO/'experiments/0010/dependencies.json').read_text())['assets'];assets=json.loads((REPO/'experiments/0010/downloads.json').read_text())['assets'];checks=[];reuse=[]
 for a in wheels+assets:
  roots=[ROOT/n for n in ('bootstapir-cache/wheels','tapnextpp-cache/wheels','tapnextpp-recovery/wheels','clip-0010-wheels')] if a in wheels else [ROOT/n for n in ('bootstapir-cache','bootstapir-cache/weights','bootstapir-cache/data','bootstapir-cache/source/clip','tapnextpp-cache','tapnextpp-cache/weights','tapnextpp-cache/data','tapnextpp-cache/source/clip','tapnextpp-recovery','tapnextpp-recovery/weights','tapnextpp-recovery/data','tapnextpp-recovery/source/clip','clip-0010-downloads')]
  for d in roots:
   p=d/a['filename'];partial=p.with_name(p.name+'.partial');entry={'path':str(p.relative_to(ROOT)),'exists':p.exists(),'partial_exists':partial.exists()}
   if p.exists():
    assert p.is_file() and not p.is_symlink();entry['bytes']=p.stat().st_size
    if entry['bytes']==a['bytes']:
     entry['sha256']=sha(p);rawgit=hashlib.sha1(b'blob '+str(entry['bytes']).encode()+b'\0')
     with p.open('rb') as fp:
      for b in iter(lambda:fp.read(1048576),b''):rawgit.update(b)
     entry['git_blob_sha1']=rawgit.hexdigest();entry['matches']=all(entry[k]==a[k] for k in ('sha256','git_blob_sha1') if k in a)
     if entry['matches']:reuse.append({'source':str(p),'target_kind':'wheels' if a in wheels else 'downloads','filename':a['filename'],'bytes':a['bytes'],'sha256':entry['sha256']})
   checks.append(entry)
 for n in ('clip-0010-wheels','clip-0010-downloads','clip-0010-assets','clip-0010-overlay','scratch/0010-capability-output'):assert not (ROOT/n).exists(),n
 save('pre-execution.json',{'status':'PASS','main':MAIN,'source':SOURCE,'freeze':FREEZE,'canonical_sha256':sha(REPO/RECEIPT),'inputs':f['inputs'],'tracked':tracked,'preserved_original_markers_and_preflight':preserved,'named_cache_checks':checks,'reusable_exact_files':reuse,'bounds':'Exact names at listed known paths only; no recursive payload sweep or global absence claim.','active_related':active,'python':sys.version,'cgroup_memory_max':'8589934592','new_asset_bytes_before':0,'historical_all_HTTP_total':'UNKNOWN','direct_known_charges':5235758,'wheel_cap':4194304,'asset_cap':527440850})
 print(json.dumps({'status':'PASS','inputs':25,'tracked':len(tracked),'old_markers_preflight_preserved':len(preserved),'exact_named_checks':len(checks),'verified_reuse_files':len(reuse),'canonical_sha256':sha(REPO/RECEIPT)}))
if __name__=='__main__':main()
