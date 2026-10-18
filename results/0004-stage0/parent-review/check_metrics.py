"""Read immutable Git blobs; recompute metrics from published derived errors only."""
import collections,csv,hashlib,io,json,math,statistics,subprocess
from pathlib import Path
ROOT=str(Path.cwd())
HEAD='ca68cfd56984dedf1cf875d9b4a6c68440565ee1'
PREFIX='results/0004-stage0/'
hashes={}
def blob(path):
 b=subprocess.check_output(['git','-C',ROOT,'show',HEAD+':'+path]);hashes[path]=hashlib.sha256(b).hexdigest();return b
config=json.loads(blob('configs/0004-stage0.json'))
official=json.loads(blob(PREFIX+'run/summary.json'))
cloud=json.loads(blob(PREFIX+'independent-summary.json'))
raw_list=list(csv.DictReader(io.StringIO(blob(PREFIX+'run/predictions.csv').decode())))
errors_list=list(csv.DictReader(io.StringIO(blob(PREFIX+'independent-errors.csv').decode())))
ids=official['point_ids'];nframe=config['frames'];starts=config['shadow_starts'];h=config['shadow_horizon'];controls=config['control_frames']
assert ids==list(range(8)) and nframe==250 and h==8
expected={(b,t,p,-1) for b in ('keep','zero') for t in range(nframe) for p in ids}|{('clone_control',t,p,-1) for t in controls for p in ids}|{('shadow',t,p,s) for s in starts for t in range(s,s+h) for p in ids}
def key(r):return r['branch'],int(r['frame']),int(r['point_index']),int(r['shadow_start'])
raw={};errors={};batches=collections.defaultdict(list)
for r in raw_list:
 k=key(r);assert k in expected and k not in raw
 vals=tuple(float(r[c]) for c in ('x','y','occlusion_logit','expected_distance_logit'))
 assert all(math.isfinite(x) for x in vals)
 assert int(r['shadow_offset'])==(k[1]-k[3] if k[0]=='shadow' else -1)
 raw[k]=vals;batches[(k[0],k[1],k[3])].append(k[2])
for r in errors_list:
 k=key(r);assert k in expected and k not in errors
 e=float(r['epe_px']);assert math.isfinite(e) and e>=0
 assert r['gt_visible'] in ('0','1') and r['predicted_visible'] in ('0','1')
 errors[k]=(e,r['gt_visible']=='1',r['predicted_visible']=='1')
assert set(raw)==set(errors)==expected and len(raw)==4488 and all(p==ids for p in batches.values())
def signeg(x):
 return math.exp(-x)/(1+math.exp(-x)) if x>=0 else 1/(1+math.exp(x))
for k,vals in raw.items():
 assert (signeg(vals[2])*signeg(vals[3])>.5)==errors[k][2]
 assert errors[k][1]==errors['keep',k[1],k[2],-1][1]
def mean(v):return statistics.fmean(v) if v else None
def rat(a,b):return a/b if b else None
branches={}
for b in ('keep','zero'):
 data=[errors[b,t,p,-1] for p in ids for t in range(1,nframe)]
 vis=sum(v for e,v,pv in data)
 m={'post_query_rows':len(data),'gt_visible_count':vis,'predicted_visible_count':sum(pv for e,v,pv in data),'predicted_visible_fraction':sum(pv for e,v,pv in data)/len(data),'visibility_accuracy':sum(v==pv for e,v,pv in data)/len(data),'mean_visible_epe_px':mean([e for e,v,pv in data if v]),'thresholds':[]}
 for cut in config['pixel_thresholds']:
  correct=sum(v and e<=cut for e,v,pv in data);tp=sum(v and pv and e<=cut for e,v,pv in data);fp=sum(pv and (not v or e>cut) for e,v,pv in data)
  m['thresholds'].append(dict(threshold_px=cut,coordinate_correct_count=correct,visible_coordinate_error_count=vis-correct,visible_coordinate_error_rate=rat(vis-correct,vis),coordinate_correct_fraction_among_gt_visible=rat(correct,vis),visible_correct_fraction_among_gt_visible=rat(tp,vis),true_positive_count=tp,false_positive_count=fp,jaccard=rat(tp,vis+fp)))
 m['mean_threshold_jaccard']=mean([r['jaccard'] for r in m['thresholds']]);branches[b]=m
per=[];rec=harm=baseerr=0
for p in ids:
 pairs=[(errors['keep',t,p,-1][0],errors['zero',t,p,-1][0]) for t in range(1,nframe) if errors['keep',t,p,-1][1]]
 r=sum(a>4>=b for a,b in pairs);d=sum(b>4>=a for a,b in pairs);rec+=r;harm+=d;baseerr+=sum(a>4 for a,b in pairs)
 per.append(dict(point_index=p,gt_visible_count=len(pairs),mean_keep_epe_px=mean([a for a,b in pairs]),mean_zero_epe_px=mean([b for a,b in pairs]),mean_epe_zero_minus_keep_px=mean([b-a for a,b in pairs]),coordinate_recoveries_at_4px=r,coordinate_harms_at_4px=d))
def loss(k):
 e,v,pv=errors[k];return int(not(pv and e<=4)) if v else int(pv)
episodes=[];details=[];offsets=[0]*h;perstart={};pos=neg=0;by_vis={True:collections.Counter(),False:collections.Counter()}
for s in starts:
 perstart[s]=0
 for p in ids:
  ts=list(range(s,s+h));kl=[loss(('keep',t,p,-1)) for t in ts];sl=[loss(('shadow',t,p,s)) for t in ts];ds=[a-b for a,b in zip(kl,sl)];gain=sum(ds)
  outcome='beneficial' if gain>0 else 'harmful' if gain<0 else 'neutral'
  episodes.append(dict(shadow_start=s,point_index=p,frames=h,keep_loss_count=sum(kl),shadow_loss_count=sum(sl),frame_indices=ts,keep_losses=kl,shadow_losses=sl,frame_loss_deltas=ds,loss_delta_count=gain,return_keep_minus_shadow=gain/h,outcome=outcome))
  for j,(t,d) in enumerate(zip(ts,ds)):
   offsets[j]+=d;perstart[s]+=d;pos+=d>0;neg+=d<0
   by_vis[errors['keep',t,p,-1][1]][d]+=1
  if gain>0:
   details.append(dict(start=s,point=p,gain=gain,visible_frames=sum(errors['keep',t,p,-1][1] for t in ts),deltas=ds))
counts=collections.Counter(r['outcome'] for r in episodes);benef=sorted({r['point_index'] for r in episodes if r['loss_delta_count']>0});gain=sum(offsets)
maximum={}
sets={'frame_zero':[(('keep',0,p,-1),('zero',0,p,-1)) for p in ids], 'clone_control':[(('keep',t,p,-1),('clone_control',t,p,-1)) for t in controls for p in ids], 'shadow_start':[(('zero',s,p,-1),('shadow',s,p,s)) for s in starts for p in ids]}
for name,pairs in sets.items():
 maximum[name]=max(abs(x-y) for a,b in pairs for x,y in zip(raw[a],raw[b]));assert maximum[name]<=config['clone_atol']
crit=dict(baseline_errors_at_least_20=baseerr>=20,coordinate_recoveries_at_least_5=rec>=5,aggregate_shadow_return_positive=gain>0,at_least_2_queries_with_beneficial_shadow=len(benef)>=2)
decision='undersensitive' if baseerr<20 else 'continue_independent_development' if all(crit.values()) else 'stop_this_candidate'
result=dict(scope=official['scope'],query_frame_excluded=0,point_ids=ids,frames=nframe,validated_rows=len(raw),control_max_absolute_differences=maximum,branch_summaries=branches,per_query=per,paired_coordinate_changes=dict(threshold_px=4,baseline_visible_errors=baseerr,recoveries=rec,harms=harm,net_recoveries_minus_harms=rec-harm),shadow_episodes=episodes,shadow_aggregate=dict(episode_count=len(episodes),loss_delta_count=gain,mean_return_keep_minus_shadow=gain/(h*len(episodes)),beneficial_episode_count=counts['beneficial'],harmful_episode_count=counts['harmful'],neutral_episode_count=counts['neutral'],queries_with_beneficial_shadow=benef),continuation=dict(status=decision,criteria=crit,permits_independent_development_only=decision=='continue_independent_development'))
leaves=0;maxdiff=0
def compare(a,b,path=''):
 global leaves,maxdiff
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),path
  for k in a:compare(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  assert isinstance(b,(int,float)) and not isinstance(b,bool) and math.isfinite(b),path
  d=abs(a-b);assert d<=1e-10,(path,a,b,d);leaves+=1;maxdiff=max(maxdiff,d)
 else:assert a==b,(path,a,b)
compare(result,official);official_count=leaves;compare(result,cloud)
output={'candidate':HEAD,'status':'PASS','scope':'stdlib arithmetic from saved derived errors; raw grid/logit/controls cross-check; no GT/cache decode or candidate imports','GT_boundary':'EPE and GT-visible flags accepted from published independent-errors.csv; GT-coordinate correctness not re-established here','input_sha256':hashes,'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numeric_leaves_per_summary':official_count,'summaries_compared':2,'maximum_absolute_difference':maxdiff,'paired':result['paired_coordinate_changes'],'shadow':result['shadow_aggregate'],'decision':decision,'posthoc_descriptive_only':{'offset_keep_minus_shadow_counts':offsets,'per_start_keep_minus_shadow_counts':perstart,'beneficial_episodes':details,'improved_loss_frames':pos,'harmed_loss_frames':neg,'loss_delta_counts_by_GT_visibility':{str(k):dict(v) for k,v in by_vis.items()},'sum_positive_episode_gains':sum(max(0,r['loss_delta_count']) for r in episodes),'sum_negative_episode_gains':sum(min(0,r['loss_delta_count']) for r in episodes)}}
print(json.dumps(output,indent=2))
