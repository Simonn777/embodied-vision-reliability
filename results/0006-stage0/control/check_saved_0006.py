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
 records={x['name']:x for x in prepared['records']};paired={};predictions={};rows=[]
 for name in sorted(expected):
  folder=N/'cases'/name;rec=read(folder/'receipt.json');ex=read(N/'cases'/(name+'.execution.json'));fit=rec['fit'];g,co,s,m=rec['image_id'],rec['condition'],rec['seed'],rec['method']
  assert name==f'{g}-{co}-{s}-{m}'and ex['case']==name and ex['exit_code']==ex['raw_wait_status']==0 and ex['timed_out']is False
  assert sha(folder/'receipt.json')==ex['receipt_sha256']
  for st in ['stdout','stderr']:assert sha(N/'cases'/(name+'.'+st+'.log'))==ex[st+'_sha256']
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
 result={'status':'passed','case_count':64,'all_updates':500,'matched_initial_and_batch_stream_sets':16,'white_controls':whites,'finite_prediction_scalars':64*122*122,'case_records':rows,'fit_seconds_sum':sum(x['fit_seconds']for x in rows),'case_wall_seconds_sum':sum(x['case_wall_seconds']for x in rows),'peak_case_rss_kib':max(x['peak_rss_kib']for x in rows),'reference_numeric_access':'none; reference file hashes only in this pre-score grid check','no_trainer_or_torch_import':True}
 save('grid-integrity.json',result);print(json.dumps({k:v for k,v in result.items()if k not in ['case_records','white_controls']},indent=2))
def independent_score(clean,pred):
 assert clean.shape==pred.shape==(122,122)and pred.dtype==np.float32 and np.isfinite(clean).all()and np.isfinite(pred).all()
 e=[[float(pred[i,j])-float(clean[i,j])for j in range(122)]for i in range(122)]
 sse=math.fsum(x*x for row in e for x in row);hs=math.fsum((row[j+1]-row[j])**2 for row in e for j in range(121));vs=math.fsum((e[i+1][j]-e[i][j])**2 for i in range(121)for j in range(122))
 starts=[0]
 for i in range(8):starts.append(starts[-1]+15+(i<2))
 blocks=[{'row':i,'col':j,'count':(starts[i+1]-starts[i])*(starts[j+1]-starts[j]),'sse':math.fsum(e[y][x]**2 for y in range(starts[i],starts[i+1])for x in range(starts[j],starts[j+1]))}for i in range(8)for j in range(8)]
 return {'sse':sse,'count':14884,'mse':sse/14884,'psnr_db':-10*math.log10(sse/14884)if sse else None,'horizontal_sse':hs,'vertical_sse':vs,'gradient_mse':(hs+vs)/(2*122*121),'blocks':blocks}
def metrics():
 official=read(C/'scores.json');assert official['case_count']==64 and read(C/'grid-integrity.json')['status']=='passed'
 values={};comparisons=0;maxdiff=0.
 def close(a,b):
  nonlocal comparisons,maxdiff
  if a is None or b is None:assert a is b;return
  maxdiff=max(maxdiff,abs(a-b));comparisons+=1;assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
 for row in official['cases']+official['secondary_direct_r2r']:
  g,co,s,m=row['group'],row['condition'],row['seed'],row['method'];name=f'{g}-{co}-{s}-{m}';direct=row.get('secondary_direct_only',False)
  with np.load(N/'prepared'/(g+'-reference.npz'),allow_pickle=False)as z:clean=z['clean'][3:-3,3:-3]
  pred=np.load(N/'cases'/name/('direct_prediction.npy'if direct else'prediction.npy'),allow_pickle=False);v=independent_score(clean,pred);sc=row['score']
  for k in ['sse','mse','psnr_db']:close(v[k],sc[k])
  assert sc['count']==14884
  close(v['horizontal_sse'],sc['first_difference']['horizontal']['sse']);close(v['vertical_sse'],sc['first_difference']['vertical']['sse']);close(v['gradient_mse'],sc['first_difference']['combined']['mse'])
  assert len(sc['blocks'])==64
  for a,b in zip(v['blocks'],sc['blocks']):assert a['row']==b['row']and a['col']==b['col']and a['count']==b['count'];close(a['sse'],b['sse'])
  if not direct:assert(g,co,s,m)not in values;values[g,co,s,m]=v
 assert len(values)==64
 order=['blindspot','safe_blindspot','r2r_alpha05'];pairs=[]
 for g,s in itertools.product(['d1','d2','d3','d4'],[0,1]):
  best=min(order,key=lambda m:values[g,'colored',s,m]['mse']);cand=values[g,'colored',s,'residual'];base=values[g,'colored',s,best];gain=cand['psnr_db']-base['psnr_db'];pairs.append({'group':g,'seed':s,'oracle_best_baseline':best,'gain_db':gain,'candidate_gradient_mse':cand['gradient_mse'],'baseline_gradient_mse':base['gradient_mse']})
 group={g:statistics.mean(p['gain_db']for p in pairs if p['group']==g)for g in ['d1','d2','d3','d4']};mean=statistics.mean(p['gain_db']for p in pairs);cg=statistics.mean(p['candidate_gradient_mse']for p in pairs);bg=statistics.mean(p['baseline_gradient_mse']for p in pairs)
 checks={'colored_pair_mean_ge_0_3_db':mean>=.3,'colored_at_least3_group_means_gt0':sum(x>0 for x in group.values())>=3,'colored_each_group_mean_ge_minus0_2_db':all(x>=-.2 for x in group.values()),'colored_first_difference_le_1_05':cg<=1.05*bg,'white_predictions_bitwise_equal':True,'white_safe_predictions_bitwise_equal':True,'white_psnr_harm_zero':True}
 assert checks==official['gates'];close(mean,official['mean_colored_gain_db']);close(cg,official['mean_candidate_first_difference_mse']);close(bg,official['mean_corresponding_baseline_first_difference_mse'])
 for g,x in group.items():close(x,official['group_mean_colored_gain_db'][g])
 for a,b in zip(pairs,official['colored_pairs']):assert a['group']==b['group']and a['seed']==b['seed']and a['oracle_best_baseline']==b['oracle_best_baseline'];close(a['gain_db'],b['gain_db'])
 status='continue_mechanism_exploration'if all(checks.values())else'stop_this_candidate';assert status==official['status']
 result={'status':'passed_independent_recomputation','scientific_status':status,'primary_cases':64,'secondary_direct_cases':len(official['secondary_direct_r2r']),'arithmetic':'Python float64 subtraction and math.fsum squares, no frozen metrics/trainer imported','tolerance':{'relative':1e-12,'absolute':1e-12},'numeric_comparisons':comparisons,'max_absolute_numeric_difference':maxdiff,'mean_colored_gain_db':mean,'group_mean_colored_gain_db':group,'gradient_ratio':cg/bg,'gates':checks,'colored_pairs':pairs,'scores_file_sha256':sha(C/'scores.json'),'no_torch_or_trainer_import':True}
 assert'torch'not in sys.modules
 save('independent-scores.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ['grid','metrics'];globals()[mode]()
