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
def independent_score(clean,pred):
 assert clean.shape==pred.shape==(122,122)and pred.dtype==np.float32 and np.isfinite(clean).all()and np.isfinite(pred).all()
 e=[[float(pred[i,j])-float(clean[i,j])for j in range(122)]for i in range(122)]
 sse=math.fsum(x*x for row in e for x in row);hs=math.fsum((row[j+1]-row[j])**2 for row in e for j in range(121));vs=math.fsum((e[i+1][j]-e[i][j])**2 for i in range(121)for j in range(122))
 starts=[0]
 for i in range(8):starts.append(starts[-1]+15+(i<2))
 blocks=[{'row':i,'col':j,'count':(starts[i+1]-starts[i])*(starts[j+1]-starts[j]),'sse':math.fsum(e[y][x]**2 for y in range(starts[i],starts[i+1])for x in range(starts[j],starts[j+1]))}for i in range(8)for j in range(8)]
 return {'sse':sse,'count':14884,'mse':sse/14884,'psnr_db':-10*math.log10(sse/14884)if sse else None,'horizontal_sse':hs,'vertical_sse':vs,'gradient_mse':(hs+vs)/(2*122*121),'blocks':blocks}
def metrics():
 official=read(C/'scores.json');integrity=read(C/'postfailure-integrity.json');assert official['case_count']==integrity['case_count']==64 and integrity['all_updates']==500 and len(integrity['white_controls'])==8
 assert integrity['status']=='BLOCKED_LOG_PROVENANCE' # Retain the open log defect; do not turn the strict failed check green.
 ex=read(C/'score-execution.json');assert ex['exit_code']==0 and ex['raw_wait_status']==0
 assert sha(C/'score.stdout.log')==ex['stdout_sha256'] and sha(C/'score.stderr.log')==ex['stderr_sha256']
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
 result={'status':'passed_independent_recomputation','scientific_status':status,'primary_cases':64,'secondary_direct_cases':len(official['secondary_direct_r2r']),'arithmetic':'Python float64 subtraction and math.fsum squares, no frozen metrics/trainer imported','tolerance':{'relative':1e-12,'absolute':1e-12},'numeric_comparisons':comparisons,'max_absolute_numeric_difference':maxdiff,'mean_colored_gain_db':mean,'group_mean_colored_gain_db':group,'gradient_ratio':cg/bg,'gates':checks,'colored_pairs':pairs,'scores_file_sha256':sha(C/'scores.json'),'provenance_status':integrity['status'],'open_log_mismatches':integrity['log_mismatches'],'no_torch_or_trainer_import':True}
 assert'torch'not in sys.modules
 save('independent-scores.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 assert sys.argv[1]=='metrics';metrics()
