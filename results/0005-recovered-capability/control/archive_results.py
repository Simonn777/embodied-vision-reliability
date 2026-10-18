"""Byte-exact small evidence copies. Acquired payloads are never published."""
import hashlib,json,shutil
from pathlib import Path
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0005-recovery';C=Path(__file__).parent;N=B/'tapnextpp-recovery';O=B/'scratch/0005-recovered-native-capability';D=R/'results/0005-recovered-capability'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert not D.exists();D.mkdir();pairs=[]
 for p in sorted(C.iterdir()):
  assert p.is_file()and p.suffix in ['.py','.json','.log','.csv'],p
  pairs.append((p,'control/'+p.name))
 for p in sorted(O.rglob('*')):
  if p.is_file():pairs.append((p,'native/'+p.relative_to(O).as_posix()))
 names=['recovery-dependencies.json','recovery-checkpoint.json','import-check.json','import-check.stdout.log','import-check.stderr.log','source-manifest.json','overlay-manifest.json','acquisition-ledger.json','acquisition-ledger-dependencies.json','acquisition-ledger-checkpoint.json','asset-records/checkpoint-256.json','markers/dependencies.started.json','markers/checkpoint.started.json','0005-recovered-native-started.json','0005-recovered-supervisor-started.json']
 for name in names:pairs.append((N/name,'preparation/'+name))
 result=[]
 for src,rel in pairs:
  assert src.is_file()and not src.is_symlink()and src.stat().st_size<1048576
  dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
  assert sha(dest)==sha(src)
  result.append({'source':src.relative_to(B).as_posix(),'copied':dest.relative_to(R).as_posix(),'bytes':dest.stat().st_size,'sha256':sha(dest)})
 with(D/'copy-manifest.json').open('x')as f:json.dump({'schema_version':1,'self_excluded':True,'files':result},f,indent=2);f.write('\n')
 print(json.dumps({'copied_files':len(result),'copied_bytes':sum(x['bytes']for x in result),'destination':str(D)}))
if __name__=='__main__':main()
