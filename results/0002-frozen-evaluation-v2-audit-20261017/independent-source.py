"""Independent CSV/math statistic implementation; NumPy is used only for PCG64 draws.

No evr, candidate model, dataset decoder, evaluate, threshold refit in bootstrap,
or inference call. EPE/visibility labels are frozen row-builder audit exports.
"""
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
B=Path('results/0002-frozen-evaluation-v2-audit-20261017');O=Path('results/0002-frozen-evaluation-v2-20261017')
METHODS=['native','augmented','visibility','local','random','oracle'];ATOL=1e-12
def load(p):return json.loads(Path(p).read_text())
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=load('protocols/0002-freeze-v2.json');manifest=load(f['manifest_path']);export=load(B/'row-export.json');baseline=load(B/'baseline.json')
groups=[g['group_id']for g in manifest['groups']if g['partition']=='evaluation'];allrows=[];score_checks=0;score_maxdiff=0.;tie_checks=0
bools=['post_query','visible','opportunity','coordinate_valid','logit_valid','available','failure']
for path,digest in export['sha256'].items():
 assert h(path)==digest
 with open(path,newline='')as z:
  for r in csv.DictReader(z):
   for k in ['archive_index','point','frame']:r[k]=int(r[k])
   for k in bools:r[k]=None if r[k]==''else bool(int(r[k]))
   for k in ['error_px','occlusion_logit','expected_dist_logit','x','y']+['score_'+m for m in METHODS]:r[k]=None if r[k]==''else float(r[k])
   assert r['post_query']==(r['frame']>0)and r['opportunity']==(r['visible']and r['post_query'])
   assert r['failure']==(r['error_px']>4 if r['error_px']is not None else None)
   vid=manifest['identities'][r['archive_index']]['video_sha256'];tie=hashlib.sha256(f"0002|{r['archive_index']}|{vid}|{r['point']}|{r['frame']}".encode()).hexdigest();assert r['tie']==tie;tie_checks+=1
   allrows.append(r)
# Independent causal-score check against raw saved xy/logits, including all frames.
cap=-math.log(1e-8)
def softplus(v):return max(v,0)+math.log1p(math.exp(-abs(v)))
for video,point in sorted({(r['archive_index'],r['point'])for r in allrows}):
 history=[];source=list(csv.DictReader(open(Path('results/0002-pilot-predictions/v2-reconstructed')/f'g-{video:03}'/f'video-{video:03}.csv')));raw={(int(r['point']),int(r['frame'])):r for r in source}
 for r in sorted((r for r in allrows if r['archive_index']==video and r['point']==point),key=lambda r:r['frame']):
  t=r['frame'];x,y=r['x'],r['y'];o,u=r['occlusion_logit'],r['expected_dist_logit'];original=raw[point,t]
  assert x==float(original['x'])and y==float(original['y'])
  for k,v in [('occlusion_logit',o),('expected_dist_logit',u)]:
   q=float(original[k]);assert (v==q if math.isfinite(q)else v is None or not math.isfinite(v))
  coord=all(v is not None and math.isfinite(v)for v in [x,y]);logits=all(v is not None and math.isfinite(v)for v in [o,u]);available=coord and logits
  assert r['coordinate_valid']==coord and r['logit_valid']==logits and r['available']==available
  expected={m:None for m in METHODS}
  if coord:
   residual=0.
   if len(history)>1:
    t0,x0,y0=history[-2];t1,x1,y1=history[-1];ratio=(t-t1)/(t1-t0);residual=math.hypot(x-(x1+(x1-x0)*ratio),y-(y1+(y1-y0)*ratio))
   if available:
    v,l=min(softplus(o),cap),min(softplus(u),cap);native=min(v+l,cap)
    expected.update(native=native,augmented=native+min(residual/8,10),visibility=v,local=l,random=0.,oracle=r['error_px']if r['visible']else None)
   history.append((t,x,y))
  for method,value in expected.items():
   actual=r['score_'+method]
   if value is None:assert actual is None
   else:
    diff=abs(value-actual);score_maxdiff=max(score_maxdiff,diff);assert diff<=ATOL,(method,diff)
   score_checks+=1
cal=[r for r in allrows if r['partition']=='calibration'];ev=[r for r in allrows if r['partition']=='evaluation']
def eligible(rows):return [r for r in rows if r['opportunity']]
def ordered(rows,method):return sorted((r for r in eligible(rows)if r['available']and r['score_'+method]is not None and math.isfinite(r['score_'+method])),key=lambda r:(r['score_'+method],r['tie']))
def risk(rows):return sum(bool(r['failure'])for r in rows)/len(rows)if rows else None
def matched(rows,method):
 n=len(eligible(rows));m=math.ceil(.8*n);order=ordered(rows,method);ok=n>0 and len(order)>=m
 return {'opportunities':n,'target_accepts':m,'available':len(order),'reachable':ok,'attainable_coverage':len(order)/n if n else None,'coverage':m/n if ok else None,'risk':risk(order[:m])if ok else None}
def fit(rows,method):
 n=len(eligible(rows));m=math.ceil(.8*n);order=ordered(rows,method);k=min(m,len(order));boundary=order[k-1]if order else None
 return {'method':method,'threshold':[boundary['score_'+method],boundary['tie']]if boundary else None,'opportunities':n,'target_accepts':m,'unreachable':n==0 or len(order)<m,'accepted_calibration':k,'calibration_coverage':k/n if n else None,'calibration_risk':risk(order[:m])if order and m else None}
thresholds={method:fit(cal,method)for method in METHODS[:-1]}
def operation(rows,method):
 n=len(eligible(rows));threshold=thresholds[method]['threshold'];selected=[]if threshold is None else[r for r in ordered(rows,method)if(r['score_'+method],r['tie'])<=tuple(threshold)]
 return {'opportunities':n,'accepted':len(selected),'coverage':len(selected)/n if n else None,'risk':risk(selected)}
def curve(rows,method):
 n=len(eligible(rows));out=[];fail=0
 for k,r in enumerate(ordered(rows,method),1):fail+=bool(r['failure']);out.append([k/n,fail/k])
 return out
def diff(a,b):return None if a is None or b is None else a-b
def effects(rows):
 ma,mn=matched(rows,'augmented'),matched(rows,'native');oa,on=operation(rows,'augmented'),operation(rows,'native')
 return {'matched_risk_difference':diff(ma['risk'],mn['risk']),'operational_risk_difference':diff(oa['risk'],on['risk']),'operational_coverage_difference':diff(oa['coverage'],on['coverage'])}
keys=['matched_risk_difference','operational_risk_difference','operational_coverage_difference'];samples={k:[]for k in keys};bygroup={g:[r for r in ev if r['group_id']==g]for g in groups};rng=np.random.default_rng(20261018)
with(B/'bootstrap-draws.csv').open('x',newline='')as d,(B/'bootstrap-replicates.csv').open('x',newline='')as z:
 dw=csv.writer(d,lineterminator='\n');dw.writerow(['replicate']+['draw_'+str(i)for i in range(len(groups))]);w=csv.writer(z,lineterminator='\n');w.writerow(['replicate','opportunities']+keys)
 for replicate in range(2000):
  draw=[groups[i]for i in rng.integers(0,len(groups),len(groups))];dw.writerow([replicate]+draw)
  # Explicit expansion retains all multiplicities and original ties, even when
  # the same group is sampled repeatedly. No group/video/row-ID deduplication.
  expanded=[r for g in draw for r in bygroup[g]];value=effects(expanded);w.writerow([replicate,len(eligible(expanded))]+[value[k]for k in keys])
  for k in keys:
   if value[k]is not None and math.isfinite(value[k]):samples[k].append(value[k])
def percentile(values,p):
 a=sorted(values);q=(len(a)-1)*p;i=math.floor(q);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(q-i)
boot={'unit':'allocated exact-duplicate group; empty groups retained','allocated_groups':groups,'repeats':2000,'seed':20261018,'intervals':{k:{'defined_replicates':len(v),'ci95':[percentile(v,.025),percentile(v,.975)]if len(v)>=1900 else None}for k,v in samples.items()}}
opps=eligible(ev);n=len(opps);coord=[r for r in opps if r['coordinate_valid']];post=[r for r in ev if r['post_query']];vis_correct=sum(r['occlusion_logit']is not None and ((r['occlusion_logit']<0)==r['visible'])for r in post);effect=effects(ev);point=effect['matched_risk_difference'];interval=boot['intervals']['matched_risk_difference']['ci95']
result={'thresholds':thresholds,'matched':{m:matched(ev,m)for m in METHODS},'operational':{m:operation(ev,m)for m in METHODS[:-1]},'curves':{m:curve(ev,m)for m in METHODS},'effect':effect,'bootstrap':boot,'primary_opportunities':n,'coordinate_missing_rate':1-len(coord)/n if n else None,'joint_score_unavailable_rate':sum(not r['available']for r in opps)/n if n else None,'estimator_failure_including_missing':sum(not r['coordinate_valid']or bool(r['failure'])for r in opps)/n if n else None,'accept_all_coordinates':{'coverage':len(coord)/n if n else None,'risk':risk(coord)},'visibility_accuracy':vis_correct/len(post)if post else None,'practical_signal':bool(point is not None and point<=-.02 and interval is not None and interval[1]<0),'interpretation':'exploratory; unknown scene dependence and upstream exposure; no conformal/general efficacy guarantee','calibration_groups':[g['group_id']for g in manifest['groups']if g['partition']=='calibration'],'evaluation_groups':groups}
expected=load(O/'evaluation.json');numeric=0;exact=0;maxdiff=0.
def compare(a,b,path='root'):
 global numeric,exact,maxdiff
 if isinstance(a,dict):
  assert set(a)==set(b),(path,set(a)^set(b))
  for k in a:compare(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),(path,len(a),len(b))
  for i,(x,y)in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
 elif type(a)is float:
  assert type(b)in(int,float)and math.isfinite(a)and math.isfinite(b),path
  d=abs(a-b);numeric+=1;maxdiff=max(maxdiff,d);assert d<=ATOL,(path,a,b,d)
 else:exact+=1;assert a==b,(path,a,b)
compare(result,expected)
for p,digest in baseline['historical_sha256'].items():assert h(p)==digest
assert 'evr.pilot_metrics'not in sys.modules and 'torch'not in sys.modules and not any(k.startswith('tapnet')for k in sys.modules)
(B/'independent-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
report={'status':'passed','absolute_tolerance':ATOL,'relative_tolerance':0,'numeric_leaves_compared':numeric,'exact_leaves_compared':exact,'max_metric_absolute_difference':maxdiff,'causal_score_values_checked':score_checks,'max_score_absolute_difference':score_maxdiff,'original_ties_checked':tie_checks,'calibration_rows':len(cal),'evaluation_rows':len(ev),'calibration_opportunities':len(eligible(cal)),'evaluation_opportunities':n,'bootstrap_replicates_recomputed':2000,'defined_counts':{k:len(v)for k,v in samples.items()},'sampling_frame_complete_groups':groups,'thresholds_refit_in_bootstrap':False,'historical_files_preserved':len(baseline['historical_sha256']),'scope':'independent CSV/math scoring/ranking/threshold/operational/curve/diagnostic/paired-bootstrap implementation; NumPy only for identical PCG64 seeded draws; GT EPE/visibility from frozen row builder, not independently relabeled; no frozen evaluate call or inference','sha256':{str(p):h(p)for p in [B/'independent-result.json',B/'bootstrap-draws.csv',B/'bootstrap-replicates.csv']}}
(B/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
