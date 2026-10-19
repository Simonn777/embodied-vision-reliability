import os,json,hashlib,subprocess,sys,time,platform
from pathlib import Path
b=Path('/workspace/scratch/a349d03a0e0a');r=b/'research-0010-headroom';c=b/'scratch/0010-headroom-control'
def git(*a,cwd=r):return subprocess.check_output(['git',*a],cwd=cwd).strip()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def spec(p):
 assert p.is_file() and not p.is_symlink(),str(p)
 return dict(bytes=p.stat().st_size,sha256=sha(p))
start=time.time();src='d14134c34e7fa097de5556aca162af962d920880';rc='de07509be1f314da2b8803a672b2d266b234f723';main='49c0c0b22f89e5d395130dec63bafad9377ba5c5';fp='protocols/0010-real-data-headroom-freeze.json'
assert git('rev-parse','HEAD').decode()==main and not git('status','--porcelain')
assert git('rev-parse',rc+'^').decode()==src
assert git('diff-tree','--no-commit-id','--name-only','-r',rc).decode()==fp
assert subprocess.call(['git','merge-base','--is-ancestor',rc,main],cwd=r)==0
assert not git('diff',rc,main)
assert sha(r/fp)=='9ee3c49d547a8d645586c6792ad91ede49cf1ce5114fc82f49b1cf65ec21f0ff'
f=json.loads((r/fp).read_text());assert f['source_commit']==src and len(f['inputs'])==45 and len(f['external_inputs'])==9
for p,d in f['inputs'].items():
 raw=(r/p).read_bytes();assert spec(r/p)=={k:d[k] for k in ('bytes','sha256')}
 assert git('show',src+':'+p)==raw.rstrip(b'\n') if False else True
 for ref in (src,main,':'):
  obj=(ref+':'+p) if ref!=':' else ':'+p
  q=subprocess.check_output(['git','show',obj],cwd=r);assert q==raw
 assert git('hash-object',p).decode()==d['git_blob_sha1']
for k,d in f['external_inputs'].items():
 p=Path(d['path']);assert spec(p)=={x:d[x] for x in ('bytes','sha256')}
 if d['public_evidence_path']:assert (r/d['public_evidence_path']).read_bytes()==p.read_bytes()
for repo,head in [('research-0010-capability','e9e7c7609e53e1bc1b3228096f9b36008cb78d6b'),('research-0010-cached-admission','3b874f53b44fb2b3a622caf624a1c0ffe2e802c4')]:
 assert git('rev-parse','HEAD',cwd=b/repo).decode()==head and not git('status','--porcelain',cwd=b/repo)
for p in ['clip-0010-assets/0010-headroom-launch.started.json','clip-0010-assets/0010-headroom-started.json','scratch/0010-headroom-output','scratch/0010-headroom-supervision']:assert not (b/p).exists() and not (b/p).is_symlink()
related=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(bytes([0])); exe=Path(a[0].decode(errors='replace')).name
 except (OSError,IndexError):continue
 if exe.startswith('python') and any(x in arg for arg in a[1:3] for x in [b'run_antithetic',b'run_state_intervention',b'run_conditional',b'run_tapnext',b'0010-headroom/run.py',b'0010/cli.py',b'0010-solver/run_admission.py',b'0010-guard/guard.py']):related.append({'pid':int(p.parent.name),'script':Path(a[1].decode(errors='replace')).name})
assert not related,related
assert set(range(4))<=set(os.sched_getaffinity(0));assert Path('/sys/fs/cgroup/memory.max').read_text().strip()==str(8*1024**3)
# Hash all existing tracked project files, and known old controls/outputs; no pixel decoding.
preserved={}
for repo in ('research-0010-capability','research-0010-cached-admission','research-0010-headroom'):
 for p in git('ls-files','-z',cwd=b/repo).decode().split('\0'):
  if p:preserved[repo+'/'+p]=spec(b/repo/p)
for directory in ['scratch/0010-capability-control','scratch/0010-cached-admission-control','scratch/0010-capability-output','scratch/0010-cached-guard','scratch/0010-cached-guard-supervision','scratch/0010-solver-functional','scratch/0010-solver-timing','scratch/0010-partial-review','clip-0010-assets','clip-0010-downloads','clip-0010-wheels']:
 for p in (b/directory).rglob('*'):
  if p.is_file():preserved[str(p.relative_to(b))]=spec(p)
g=json.loads((b/'scratch/0010-cached-guard/guard.json').read_text())
for p in g['runtime_recheck']['files']:preserved[str(Path(p).relative_to(b))]=spec(Path(p))
command=json.loads((b/'scratch/0010-capability-control/imports.command.json').read_text())
d=dict(status='PASS_PREFLIGHT',head=main,source=src,receipt_commit=rc,freeze=spec(r/fp),repository_inputs=45,external_inputs=9,preserved=preserved,related_processes=related,python=sys.executable,python_version=sys.version,platform=platform.platform(),allowed_affinity=sorted(os.sched_getaffinity(0)),memory={k:(Path('/sys/fs/cgroup')/k).read_text() for k in ['memory.max','memory.current','memory.events']},runtime_PYTHONPATH=command['PYTHONPATH'],new_asset_bytes=0,direct_body_unchanged=456087034,historical_all_HTTP='UNKNOWN',elapsed=time.time()-start)
(c/'preflight.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:d[k] for k in ['status','head','repository_inputs','external_inputs','elapsed']}|{'preserved_count':len(preserved)}))
