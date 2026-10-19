"""Saved-output identity/normalization check; no torch/model/forward/accuracy."""
import hashlib,json,types
from pathlib import Path
import numpy as np
b=Path('/workspace/scratch/a349d03a0e0a');r=b/'research-0010-capability';o=b/'scratch/0010-capability-output';c=b/'scratch/0010-cached-admission-control'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ident(p):
 d=p.read_bytes();return dict(bytes=len(d),sha256=hashlib.sha256(d).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest())
a=json.loads((o/'receipt.json').read_text());assert a['status']=='completed' and a['failure'] is None
assert [a[n] for n in ['image_attempted','image_completed','text_attempted','text_completed']]==[8,8,1,1]
assert json.loads((b/'scratch/0010-capability-control/probe.execution.json').read_text())['actual_exit']==0
s=json.loads((b/'scratch/0010-capability-control/probe/execution.json').read_text());assert s['child_exit_code']==s['supervisor_exit_code']==0
cap=types.ModuleType('saved_identity_helpers');cap.__file__=str(r/'experiments/0010/run_capability.py');exec(compile((r/'experiments/0010/run_capability.py').read_bytes(),cap.__file__,'exec'),cap.__dict__)
cfg=json.loads((r/'experiments/0010/config.json').read_text());cap.validate_config(cfg)
assert a['config']==ident(r/'experiments/0010/config.json');assert a['source']==ident(r/'experiments/0010/run_capability.py');assert a['selection']==ident(r/'experiments/0010/selection.json')
assert a['assets']==cap.verify_assets(b/'clip-0010-assets',cfg)
assert a['names']==cap.selected_names(b/'clip-0010-assets',json.loads((r/'experiments/0010/selection.json').read_text()))
assert a['prompts']==cap.prompts(b/'clip-0010-assets')
# Local frozen tokenizer only: no clip package, Torch, model or encoder import.
tp=b/'clip-0010-assets/source/clip/simple_tokenizer.py';tm=types.ModuleType('saved_tokenizer');tm.__file__=str(tp);exec(compile(tp.read_bytes(),str(tp),'exec'),tm.__dict__)
tok=tm.SimpleTokenizer();expected=[]
for text in a['prompts']:
 ids=[tok.encoder['<|startoftext|>']]+tok.encode(text)+[tok.encoder['<|endoftext|>']];assert len(ids)<=77;expected.append(ids+[0]*(77-len(ids)))
assert a['token_ids']==expected
assert a['features']==ident(o/'features.npz') and a['partial_features']==ident(o/'partial-features.npz')
checks={};errors={};unit={}
with np.load(o/'features.npz',allow_pickle=False) as z:
 assert set(z.files)=={'image_raw','image_normalized','text_raw','text_normalized'}
 for k in z.files:
  v=z[k];n=8 if k.startswith('image') else 10;assert v.shape==(n,512) and v.dtype==np.float32 and np.isfinite(v).all()
  spec=dict(shape=list(v.shape),dtype=str(v.dtype),sha256_f32le=hashlib.sha256(v.astype('<f4',copy=False).tobytes(order='C')).hexdigest());assert spec==a['array_hashes'][k];checks[k]=spec
 for k in ('image','text'):
  raw=z[k+'_raw'];normal=z[k+'_normalized'];norm=np.linalg.norm(raw,axis=1,keepdims=True);assert np.isfinite(norm).all() and (norm>0).all()
  errors[k]=float(np.max(np.abs(normal-raw/norm)));unit[k]=float(np.max(np.abs(np.linalg.norm(normal,axis=1)-1)))
  assert errors[k]<=1e-6 and unit[k]<=1e-6
 with np.load(o/'partial-features.npz',allow_pickle=False) as p:
  assert set(p.files)=={'image_raw','text_raw'} and all(np.array_equal(p[k],z[k]) for k in p.files)
t=a['timings'];names=['identities','runtime_import','load']+[f'image_{i}_{phase}' for i in range(8) for phase in ('preprocess','encode')];assert all(k in t and np.isfinite(t[k]) and t[k]>=0 for k in names)
image_sum=sum(t[f'image_{i}_{phase}'] for i in range(8) for phase in ('preprocess','encode'))
projection=1.5*(t['identities']+t['runtime_import']+t['load']+16*image_sum)
d=dict(status='PASS_SAVED_OUTPUT_ONLY',counts=[8,8,1,1],features=ident(o/'features.npz'),arrays=checks,normalization_max_abs=errors,unit_norm_max_abs=unit,atol=1e-6,timing_fields=names,timings=t,projection_seconds=projection,projection_limit_seconds=450,resource_admitted=projection<=450,token_ids_sha256=hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest(),no_model_calls=True,no_accuracy=True)
with (c/'saved-probe-review.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps({k:d[k] for k in ('status','counts','normalization_max_abs','unit_norm_max_abs','projection_seconds','resource_admitted')}));assert d['resource_admitted']
