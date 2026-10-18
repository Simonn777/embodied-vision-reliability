"""Independent scalar-loop recomputation; no model/controller/official scorer import."""
import csv,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

OUT=Path('results/0004-stage0');RUN=OUT/'run';CACHE=Path('../bootstapir-cache')
ATOL=1e-10
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
def mean(x):return statistics.fmean(x)if x else None
def ratio(a,b):return a/b if b else None
def stable_negative_sigmoid(z):
 if z>=0:
  e=math.exp(-z);return e/(1+e)
 return 1/(1+math.exp(z))
def main():
 receipt=json.loads((RUN/'receipt.json').read_text());official=json.loads((RUN/'summary.json').read_text())
 config=json.loads(Path('configs/0004-stage0.json').read_text());freeze=json.loads(Path('protocols/0004-stage0-freeze.json').read_text())
 assert receipt['status']=='completed'
 assert sha(CACHE/'development-000.npz')==config['development_npz_sha256']
 for p,h in freeze['files'].items():assert sha(p)==h
 assert receipt['predictions_sha256_before_scoring']==sha(RUN/'predictions.csv')
 # Numeric-only NPZ decoder, access only development points/occlusion labels.
 with np.load(CACHE/'development-000.npz',allow_pickle=False)as archive:
  points=archive['points'];occluded=archive['occluded']
 manifest=json.loads(Path('results/0002-development/decode/development-manifest.json').read_text())
 for key,array in [('points',points),('occluded',occluded)]:assert hashlib.sha256(array.tobytes()).hexdigest()==manifest[key+'_sha256']
 ids=[i for i in range(len(points))if not bool(occluded[i,0])][:8]
 assert receipt['point_indices']==ids and len(ids)==8
 starts=config['shadow_starts'];horizon=config['shadow_horizon'];frames=config['frames'];cuts=config['pixel_thresholds'];controls=config['control_frames']
 expected={(b,t,-1,p)for b in ['keep','zero']for t in range(frames)for p in ids}
 expected|={('clone_control',t,-1,p)for t in controls for p in ids}
 expected|={('shadow',t,s,p)for s in starts for t in range(s,s+horizon)for p in ids}
 rows={};groups=defaultdict(list)
 with(RUN/'predictions.csv').open()as f:
  reader=csv.DictReader(f);assert set(reader.fieldnames)=={'branch','frame','point_index','shadow_start','shadow_offset','x','y','occlusion_logit','expected_distance_logit'}
  for r in reader:
   b=r['branch'];t,p,s,o=[int(r[k])for k in ['frame','point_index','shadow_start','shadow_offset']]
   values=tuple(float(r[k])for k in ['x','y','occlusion_logit','expected_distance_logit']);assert all(math.isfinite(v)for v in values)
   key=(b,t,s,p);assert key in expected and key not in rows
   assert o==t-s if b=='shadow'else s==o==-1
   rows[key]=values;groups[(b,t,s)].append(p)
 assert set(rows)==expected and len(rows)==4488 and all(ps==ids for ps in groups.values())
 metrics={};audit=[];margins=[]
 for key,raw in rows.items():
  b,t,s,p=key;x,y,occ,dist=raw
  e=math.hypot(x-float(points[p,t,0])*256,y-float(points[p,t,1])*256)
  confidence=stable_negative_sigmoid(occ)*stable_negative_sigmoid(dist)
  v=not bool(occluded[p,t]);pred=confidence>.5;margins.append(abs(confidence-.5))
  metrics[key]=(e,v,pred)
  audit.append({'branch':b,'frame':t,'point_index':p,'shadow_start':s,'epe_px':e,'gt_visible':int(v),'predicted_visible':int(pred)})
 with(OUT/'independent-errors.csv').open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
 branch_summaries={}
 for b in ['keep','zero']:
  data=[metrics[(b,t,-1,p)]for p in ids for t in range(1,frames)];n=sum(v for e,v,pred in data)
  result={'post_query_rows':len(data),'gt_visible_count':n,'predicted_visible_count':sum(pred for e,v,pred in data),'predicted_visible_fraction':sum(pred for e,v,pred in data)/len(data),'visibility_accuracy':sum(pred==v for e,v,pred in data)/len(data),'mean_visible_epe_px':mean([e for e,v,pred in data if v]),'thresholds':[]}
  for cut in cuts:
   correct=sum(v and e<=cut for e,v,pred in data)
   tp=sum(v and pred and e<=cut for e,v,pred in data)
   fp=sum(pred and(not v or e>cut)for e,v,pred in data)
   result['thresholds'].append({'threshold_px':cut,'coordinate_correct_count':correct,'visible_coordinate_error_count':n-correct,'visible_coordinate_error_rate':ratio(n-correct,n),'coordinate_correct_fraction_among_gt_visible':ratio(correct,n),'visible_correct_fraction_among_gt_visible':ratio(tp,n),'true_positive_count':tp,'false_positive_count':fp,'jaccard':ratio(tp,n+fp)})
  js=[x['jaccard']for x in result['thresholds']];result['mean_threshold_jaccard']=mean(js)if all(x is not None for x in js)else None
  branch_summaries[b]=result
 per_query=[];recovered=harmed=baseline_errors=0
 for p in ids:
  ke=[];ze=[];diff=[];rec=harm=errors=0
  for t in range(1,frames):
   a,v,_=metrics[('keep',t,-1,p)];b,_,_=metrics[('zero',t,-1,p)]
   if not v:continue
   ke.append(a);ze.append(b);diff.append(b-a);errors+=a>4;rec+=a>4 and b<=4;harm+=a<=4 and b>4
  baseline_errors+=errors;recovered+=rec;harmed+=harm
  per_query.append({'point_index':p,'gt_visible_count':len(ke),'mean_keep_epe_px':mean(ke),'mean_zero_epe_px':mean(ze),'mean_epe_zero_minus_keep_px':mean(diff),'coordinate_recoveries_at_4px':rec,'coordinate_harms_at_4px':harm})
 def loss(key):
  e,v,pred=metrics[key]
  return int(not(pred and e<=4))if v else int(pred)
 episodes=[];beneficial=set();total_gain=0
 for s in starts:
  for p in ids:
   ts=list(range(s,s+horizon));a=[loss(('keep',t,-1,p))for t in ts];b=[loss(('shadow',t,s,p))for t in ts];d=[x-y for x,y in zip(a,b)];gain=sum(d);total_gain+=gain
   if gain>0:beneficial.add(p)
   episodes.append({'shadow_start':s,'point_index':p,'frames':horizon,'keep_loss_count':sum(a),'shadow_loss_count':sum(b),'frame_indices':ts,'keep_losses':a,'shadow_losses':b,'frame_loss_deltas':d,'loss_delta_count':gain,'return_keep_minus_shadow':gain/horizon,'outcome':'beneficial'if gain>0 else'harmful'if gain<0 else'neutral'})
 maximum={'clone_control':0.,'frame_zero':0.,'shadow_start':0.}
 for name,left,right in [('frame_zero',('keep',0,-1,p),('zero',0,-1,p))for p in ids]+[('clone_control',('keep',t,-1,p),('clone_control',t,-1,p))for t in controls for p in ids]+[('shadow_start',('zero',s,-1,p),('shadow',s,s,p))for s in starts for p in ids]:
  d=max(abs(a-b)for a,b in zip(rows[left],rows[right]));maximum[name]=max(maximum[name],d);assert d<=config['clone_atol']
 outcomes=Counter(e['outcome']for e in episodes)
 criteria={'baseline_errors_at_least_20':baseline_errors>=20,'coordinate_recoveries_at_least_5':recovered>=5,'aggregate_shadow_return_positive':total_gain>0,'at_least_2_queries_with_beneficial_shadow':len(beneficial)>=2}
 decision='undersensitive'if baseline_errors<20 else'continue_independent_development'if all(criteria.values())else'stop_this_candidate'
 result={'scope':'single-video development mechanism diagnostic; no CI or deployed selector','query_frame_excluded':0,'point_ids':ids,'frames':frames,'validated_rows':len(rows),'control_max_absolute_differences':maximum,'branch_summaries':branch_summaries,'per_query':per_query,'paired_coordinate_changes':{'threshold_px':4,'baseline_visible_errors':baseline_errors,'recoveries':recovered,'harms':harmed,'net_recoveries_minus_harms':recovered-harmed},'shadow_episodes':episodes,'shadow_aggregate':{'episode_count':len(episodes),'loss_delta_count':total_gain,'mean_return_keep_minus_shadow':total_gain/(len(episodes)*horizon),'beneficial_episode_count':outcomes['beneficial'],'harmful_episode_count':outcomes['harmful'],'neutral_episode_count':outcomes['neutral'],'queries_with_beneficial_shadow':sorted(beneficial)},'continuation':{'status':decision,'criteria':criteria,'permits_independent_development_only':decision=='continue_independent_development'}}
 comparisons=[];maxdiff=0.
 def compare(a,b,path='summary'):
  nonlocal maxdiff
  assert type(a)is type(b)or isinstance(a,(int,float))and not isinstance(a,bool)and isinstance(b,(int,float))and not isinstance(b,bool),(path,'type')
  if isinstance(a,dict):
   assert set(a)==set(b),(path,'keys')
   for k in a:compare(a[k],b[k],path+'.'+k)
  elif isinstance(a,list):
   assert len(a)==len(b),(path,'length')
   for i,(x,y)in enumerate(zip(a,b)):compare(x,y,path+'['+str(i)+']')
  elif isinstance(a,(int,float))and not isinstance(a,bool):
   d=abs(a-b);maxdiff=max(maxdiff,d);assert d<=ATOL,(path,a,b,d);comparisons.append(path)
  else:assert a==b,(path,a,b)
 compare(result,official)
 timings=json.loads((RUN/'timings.json').read_text());assert [x['frame']for x in timings]==list(range(250))
 refiner_count=sum(len(x['refiner_seconds'])for x in timings)
 assert refiner_count==561 and receipt['counts']['total_calls']==561 and receipt['counts']['completed_frames']==250 and receipt['counts']['failed']is False
 assert len(receipt['controls'])==13 and all(c['passed']and c['same_next_context']for c in receipt['controls'])
 for name,want in receipt['artifacts'].items():assert sha(RUN/name)==want
 (OUT/'independent-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 validation={'status':'PASS','independent_implementation':'scalar Python loops, math.hypot, stable scalar sigmoid; official controller/scorer/model not imported','comparison_atol':ATOL,'comparison_rtol':0,'numeric_leaves_compared':len(comparisons),'maximum_absolute_difference':maxdiff,'all_shadow_episodes_compared':len(episodes),'raw_grid_rows':len(rows),'all_raw_numeric_values_finite':True,'encoder_calls':len(timings),'refiner_calls':refiner_count,'equality_controls':len(receipt['controls']),'minimum_visibility_threshold_margin':min(margins),'predictions_sha256':sha(RUN/'predictions.csv'),'NPZ_sha256':sha(CACHE/'development-000.npz'),'checker_source_sha256':sha(__file__),'labels_accessed':'only existing development video0 points/occluded arrays; no RGB array decode or new inference','next_context_boundary':'digest equality asserted from recorded controls; context tensors were not reloaded or recomputed','timing_totals_seconds':{'encoder':sum(x['encoder_seconds']for x in timings),'refiner':sum(sum(x['refiner_seconds'])for x in timings),'whole_frame':sum(x['whole_frame_seconds']for x in timings)}}
 (OUT/'independent-verification.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
 print(json.dumps(validation,indent=2));print(json.dumps({'paired':result['paired_coordinate_changes'],'shadows':result['shadow_aggregate'],'continuation':result['continuation']},indent=2))
if __name__=='__main__':main()
