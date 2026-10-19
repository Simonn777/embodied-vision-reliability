"""Actual dependency imports/origins only; no CLIP import/model/tensor operation."""
import hashlib,importlib,importlib.metadata,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0010-capability'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((REPO/'experiments/0010/config.json').read_text());expected=dict(cfg['runtime_versions'],wcwidth='0.9.2');observed={}
failure=None
try:
 for n,version in expected.items():
  mod=importlib.import_module(n);dist='Pillow' if n=='PIL' else n;p=Path(mod.__file__).resolve();v=importlib.metadata.version(dist)
  observed[n]={'distribution':dist,'distribution_version':v,'module_version':str(getattr(mod,'__version__','')),'file':str(p),'module_entry_sha256':sha(p)}
  assert v==version,(n,v,version)
  origin=ROOT/('clip-0010-overlay' if n in ('ftfy','regex','tqdm','wcwidth') else 'bootstapir-cache/runtime-fixed' if n in ('torch','numpy') else 'tapnextpp-cache/overlay')
  if n!='packaging':assert p.is_relative_to(origin),(n,p,origin)
 assert sys.version.split()[0]=='3.12.14';assert importlib.import_module('torch').version.cuda is None
 assert 'clip' not in sys.modules
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)}
manifest={str(p.relative_to(ROOT/'clip-0010-overlay')):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (ROOT/'clip-0010-overlay').rglob('*') if p.is_file()}
result={'status':'PASS' if failure is None else 'FAIL','runtime':observed,'failure':failure,'PYTHONPATH':os.environ['PYTHONPATH'],'python':sys.version,'overlay_files':manifest,'no_model_constructed_or_forward':True,'source_sha256':sha(__file__)}
with (HERE/'imports.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'versions':{n:v['distribution_version'] for n,v in observed.items()},'failure':failure,'overlay_files':len(manifest),'imports_sha256':sha(HERE/'imports.json')}));raise SystemExit(0 if failure is None else 1)
