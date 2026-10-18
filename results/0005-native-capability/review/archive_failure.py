"""Copy only original small metadata/code/logs; exclude acquired payloads."""
import hashlib,json,shutil
from pathlib import Path
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0005';C=B/'tapnextpp-cache';S=Path(__file__).parent
D=R/'results/0005-native-capability'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False);mapping=[]
 pairs=[]
 for mode in ['verify','dependencies','software','review']:
  dest={'verify':'verification','dependencies':'preparation-dependencies','software':'software','review':'review'}[mode]
  for name in [mode+'-execution.json',mode+'.stdout.log',mode+'.stderr.log']:
   pairs.append((S/name,dest+'/'+name))
 for name in ['stage_parent.py','review_failure.py','tracked-baseline.json','old-cache-markers.json','failure-review.json','postfailure-cache-inventory.json','postfailure-overlay-manifest.json','archive_failure.py']:
  pairs.append((S/name,'review/'+name))
 for name in ['preparation-dependencies.json','acquisition-ledger.json','acquisition-ledger-dependencies.json','install.json','install.stdout.log','install.stderr.log','markers/dependencies.started.json']:
  pairs.append((C/name,'preparation-dependencies/'+name))
 for p in sorted((C/'asset-records').glob('*.json')):pairs.append((p,'preparation-dependencies/asset-records/'+p.name))
 for src,rel in pairs:
  dst=D/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
  assert sha(src)==sha(dst)
  mapping.append({'source':src.relative_to(B).as_posix(),'copied':dst.relative_to(R).as_posix(),'bytes':dst.stat().st_size,'sha256':sha(dst)})
 (D/'copy-manifest.json').write_text(json.dumps({'status':'byte_exact_copies','self_excluded':True,'files':mapping},indent=2)+'\n')
 print(json.dumps({'copied_files':len(mapping),'copied_bytes':sum(x['bytes']for x in mapping),'destination':str(D)}))
if __name__=='__main__':main()
