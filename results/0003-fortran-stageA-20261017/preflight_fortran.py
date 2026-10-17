import datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
B=Path('results/0003-fortran-stageA-20261017');C=Path('../bootstapir-cache')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
locks=[]
for name in ['0003-design.lock','0003-sample.lock']:
 f=(C/name).open('r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
paths=subprocess.check_output(['git','ls-files'],text=True).splitlines()
(B/'protected-baseline.json').write_text(json.dumps({'main':'863c7f43ee1ec2567126a79f15c6b48b0dac2d0e','sha256':{p:sha(p)for p in paths if p not in ['CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-17.md']},'note_prefix_sha256':sha('notes/2026-10-17.md')},indent=2)+'\n')
identities=[]
for p,n,d in [(C/'0003-pointodyssey/sample-v1.2.tar.gz',3324284510,'436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f'),(C/'0003-pointodyssey/resource-audit-v1/selected-annotation.npz',98527880,'382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77')]:
 h=hashlib.sha256()
 with p.open('rb')as f:
  while q:=f.read(2**20):h.update(q)
 assert p.stat().st_size==n and h.hexdigest()==d
 identities.append({'name':p.name,'bytes':n,'sha256':d,'access':'raw-byte hash only; no GT interpretation'})
sources=[]
for name,d in [('official-reprojection.body','c226ff24be5037fbce6ed747bf31feb1fea1de17b7a8260317fd84737b5f1c00'),('official-tracking.body','413f680d2dd8b7bf65179bdf305baf94207441dd186c4fbba0d7d9bf66787f2a')]:
 p=C/'0003-pointodyssey'/name;assert sha(p)==d;sources.append({'name':name,'sha256':d,'bytes':p.stat().st_size})
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'cached identities verified; locks obtainable','identities':identities,'source_text':sources,'protected_files':len(paths)-3,'new_GETs':0,'numeric_GT_access':0,'historical_partial_files':'present and preserved; not used, deleted, redownloaded or added to Git; no new acquisition claim'},indent=2))
