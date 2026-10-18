"""Saved-artifact integrity and independent fsum scoring; no trainer/model import."""
import hashlib,itertools,json,math,statistics,sys
from pathlib import Path
import numpy as np
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0006-screen';C=Path(__file__).parent;N=B/'conditional-residual-0006'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def save(name,x):
 with(C/name).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def ah(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':a.dtype.str},sort_keys=True,separators=(',',':')).encode('ascii'));h.update(b'\0');h.update(a.tobytes());return h.hexdigest()
def wh(arrays):
 h=hashlib.sha256(b'evr-mlp-state-v1\0')
 for key in sorted(arrays):
  a=np.ascontiguousarray(arrays[key]).astype('<f4',copy=False);header=json.dumps({'name':key,'shape':list(a.shape),'dtype':'<f4'},sort_keys=True,separators=(',',':')).encode('ascii');body=a.tobytes();h.update(len(header).to_bytes(8,'little'));h.update(header);h.update(len(body).to_bytes(8,'little'));h.update(body)
 return h.hexdigest()
def grid():
 cfg=read(R/'configs/0006-stage0.json');prepared=read(N/'prepared.json');assert prepared['count']==len(prepared['records'])==16
 expected={f'{g}-{c}-{s}-{m}'for g,c,s,m in itertools.product(cfg['image_ids'],cfg['conditions'],cfg['seeds'],cfg['methods'])}
 assert {p.name for p in(N/'cases').iterdir()if p.is_dir()}==expected
 assert {p.name.removesuffix('.execution.json')for p in(N/'cases').glob('*.execution.json')}==expected
 assert {p.name.removesuffix('.started.json')for p in(N/'cases').glob('*.started.json')}==expected
 records={x['name']:x for x in prepared['records']};paired={};predictions={};rows=[];log_mismatches=[]
 for name in sorted(expected):
  folder=N/'cases'/name;rec=read(folder/'receipt.json');ex=read(N/'cases'/(name+'.execution.json'));fit=rec['fit'];g,co,s,m=rec['image_id'],rec['condition'],rec['seed'],rec['method']
  assert name==f'{g}-{co}-{s}-{m}'and ex['case']==name and ex['exit_code']==ex['raw_wait_status']==0 and ex['timed_out']is False
  assert sha(folder/'receipt.json')==ex['receipt_sha256']
  for st in ['stdout','stderr']:
   logpath=N/'cases'/(name+'.'+st+'.log');observed=sha(logpath)
   if observed!=ex[st+'_sha256']:log_mismatches.append({'case':name,'stream':st,'recorded_sha256':ex[st+'_sha256'],'observed_sha256':observed,'observed_bytes':logpath.stat().st_size})
  assert rec['status']==fit['status']=='completed'and fit['completed_steps']==fit['requested_steps']==500 and fit['failure_reason']is None
  assert fit['fit_seconds']<90 and ex['wall_seconds']<120
  assert rec['device']=='cpu'and rec['dtype']=='float32'and rec['torch_threads']==4 and rec['torch_interop_threads']==1 and rec['clean_input_accessed']is False
  assert rec['torch_version']=='2.10.0+cpu'and rec['numpy_version']=='2.0.2'and rec['deterministic_algorithms']is True
  assert rec['batch_size']==512 and rec['optimizer']['learning_rate']==.001
  assert [x['step']for x in fit['loss_every_50']]==list(range(50,501,50))and all(math.isfinite(x['mean_batch_loss'])for x in fit['loss_every_50'])
  for f,h in rec['files'].items():assert sha(folder/f)==h
  pr=records[f'{g}-{co}-{s}'];inp=N/'prepared'/(pr['name']+'-input.npz');assert sha(inp)==pr['input_sha256']==rec['input_file_sha256']
  with np.load(inp,allow_pickle=False)as z:
   assert set(z.files)=={'noisy','covariance'};noisy=z['noisy'];cov=z['covariance']
  assert noisy.shape==(128,128)and noisy.dtype==np.float32 and cov.shape==(49,49)and cov.dtype==np.float64 and np.isfinite(noisy).all()and np.isfinite(cov).all()
  assert ah(noisy)==rec['noisy_sha256']and ah(cov)==rec['covariance_sha256']
  assert hashlib.sha256(noisy.tobytes()).hexdigest()==pr['noisy_array_sha256']and hashlib.sha256(cov.tobytes()).hexdigest()==pr['covariance_sha256']
  assert sha(N/'prepared'/(g+'-reference.npz'))==pr['reference_sha256'] # bytes only, no reference decode in grid phase
  pred=np.load(folder/'prediction.npy',allow_pickle=False);assert pred.shape==(122,122)and pred.dtype==np.float32 and np.isfinite(pred).all()and rec['finite_output']is True and ah(pred)==rec['prediction_sha256'];predictions[name]=pred
  with np.load(folder/'weights.npz',allow_pickle=False)as z:weights={k:z[k]for k in z.files}
  shapes={'0.weight':(64,49),'0.bias':(64,),'2.weight':(32,64),'2.bias':(32,),'4.weight':(1,32),'4.bias':(1,)}
  assert set(weights)==set(shapes)
  for k,a in weights.items():assert a.shape==shapes[k]and a.dtype==np.float32 and np.isfinite(a).all()
  assert wh(weights)==fit['weight_sha256']
  key=(g,co,s);paired.setdefault(key,[]).append((fit['initial_weight_sha256'],fit['batch_indices_sha256']))
  base=620000+100*cfg['image_ids'].index(g)+10*s+cfg['conditions'].index(co);assert rec['seeds']=={'init_seed':base,'batch_seed':base+1,'augment_seed':base+2,'infer_seed':base+3}
  if m=='safe_blindspot':assert rec['safe_blindspot_zero_feature_indices']==np.flatnonzero(cov[:,24]!=0).tolist()
  if m=='r2r_alpha05':
   assert rec['alpha']==.5 and rec['mc_samples']==50 and fit['augmentation_standard_normal_draws']==12544000
   assert rec['inference']['primary_forward_calls']==750 and rec['inference']['standard_normal_draws']==36465800 and rec['inference']['direct_forward_calls']==15
   direct=np.load(folder/'direct_prediction.npy',allow_pickle=False);assert direct.shape==(122,122)and direct.dtype==np.float32 and np.isfinite(direct).all()and ah(direct)==rec['direct_prediction_sha256']
  else:assert rec['inference']['primary_forward_calls']==15 and rec['mc_samples']==1
  rows.append({'case':name,'receipt_sha256':sha(folder/'receipt.json'),'execution_sha256':sha(N/'cases'/(name+'.execution.json')),'prediction_file_sha256':sha(folder/'prediction.npy'),'weight_file_sha256':sha(folder/'weights.npz'),'input_sha256':sha(inp),'completed_steps':500,'fit_seconds':fit['fit_seconds'],'case_wall_seconds':ex['wall_seconds'],'peak_rss_kib':ex['peak_rss_kib']})
 assert len(paired)==16 and all(len(v)==4 and len(set(v))==1 for v in paired.values())
 whites=[]
 for g,s in itertools.product(cfg['image_ids'],cfg['seeds']):
  blind=predictions[f'{g}-white-{s}-blindspot'];assert np.array_equal(blind,predictions[f'{g}-white-{s}-residual'])and np.array_equal(blind,predictions[f'{g}-white-{s}-safe_blindspot']);whites.append({'group':g,'seed':s,'three_predictions_exact_equal':True})
 assert 'torch'not in sys.modules
 result={'status':'BLOCKED_LOG_PROVENANCE'if log_mismatches else'INVENTORY_COMPLETE_NOT_SCORE_AUTHORITY','log_mismatches':log_mismatches,'scientific_scoring':'not_started','continuation_gate':'NOT_EVALUATED','interpretation':'Exit 0 means this postfailure inventory was recorded, never that the failed strict review passed. Frozen scorer is not invoked.','case_count':64,'all_updates':500,'matched_initial_and_batch_stream_sets':16,'white_controls':whites,'finite_prediction_scalars':64*122*122,'case_records':rows,'fit_seconds_sum':sum(x['fit_seconds']for x in rows),'case_wall_seconds_sum':sum(x['case_wall_seconds']for x in rows),'peak_case_rss_kib':max(x['peak_rss_kib']for x in rows),'reference_numeric_access':'none; reference file hashes only in this pre-score grid check','no_trainer_or_torch_import':True}
 save('postfailure-integrity.json',result);print(json.dumps({k:v for k,v in result.items()if k not in ['case_records','white_controls']},indent=2))

if __name__=='__main__':grid()
