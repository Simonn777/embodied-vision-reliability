"""Post-hoc descriptive diagnosis from published CSVs; standard library only.

Never imports the frozen evaluator/model/decoder or fits/calibrates any threshold.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

BASE='83a8971ef0abbbfe1d929f4f77bf3a65c4f5bd6c'
CATEGORIES=('both','native_only','augmented_only','neither')
ATOL=1e-12

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def identity(r):return (r['archive_index'],r['point'],r['frame'])
def opportunity(r):return r['frame']>0 and r['visible']
def failed(r):return not r['coordinate_valid'] or r['error_px'] is not None and r['error_px']>4.
def category(n,a):return 'both'if n and a else'native_only'if n else'augmented_only'if a else'neither'

def order(rows,method):
 return sorted((r for r in rows if opportunity(r)and r['available']and r[method]is not None and math.isfinite(r[method])),key=lambda r:(r[method],r['tie']))

def matched_set(rows,method):
 n=sum(opportunity(r)for r in rows);m=math.ceil(.8*n);ranked=order(rows,method)
 if n==0 or len(ranked)<m:raise ValueError('fixed matched target unreachable; do not lower it')
 return {identity(r)for r in ranked[:m]},ranked

def operational_set(rows,method,boundary):
 return set()if boundary is None else{identity(r)for r in order(rows,method)if(r[method],r['tie'])<=tuple(boundary)}

def require_published(rows,selected,n,accepts,failures):
 if sum(opportunity(r)for r in rows)!=n or len(selected)!=accepts or sum(failed(r)for r in rows if identity(r)in selected)!=failures:
  raise ValueError('published opportunity/selection/failure gate mismatch')

def residuals(points):
 """Only frame-index/raw-xy inputs: no acceptance, GT, logits or future history."""
 history=[];out={};previous=None
 for t,x,y in points:
  if type(t)is not int or previous is not None and t<=previous:raise ValueError('frame indices must strictly increase')
  previous=t;finite=math.isfinite(x)and math.isfinite(y);past=history[-2:]
  d=None
  if finite:
   d=0.
   if len(past)==2:
    t0,x0,y0=past[0];t1,x1,y1=past[1];gap=(t-t1)/(t1-t0)
    d=math.hypot(x-(x1+(x1-x0)*gap),y-(y1+(y1-y0)*gap))
  out[t]={'residual_px':d,'penalty':min(d/8.,10.)if d is not None else None,'history_frames':[p[0]for p in past],'history_insufficient':finite and len(past)<2}
  if finite:history.append((t,x,y))
 return out

def quantile(a,q):
 a=sorted(a);p=(len(a)-1)*q;i=math.floor(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)

def distribution(values):
 a=[v for v in values if v is not None and math.isfinite(v)]
 return {'n':len(a),**({k:None for k in ['min','q25','median','q75','p90','p95','max','mean']}if not a else{'min':min(a),'q25':quantile(a,.25),'median':quantile(a,.5),'q75':quantile(a,.75),'p90':quantile(a,.9),'p95':quantile(a,.95),'max':max(a),'mean':statistics.fmean(a)})}

FIELDS=['archive_index','group_id','partition','point','frame','tie','visible','opportunity','coordinate_valid','logit_valid','available','error_px','failure','native','augmented','visibility_score','local_score','residual_px','penalty','observed_increment','increment_absdiff','history_insufficient','history_frames','x','y','matched_category','operational_category','native_matched','augmented_matched','native_operational','augmented_operational']
def write_rows(path,rows,extra=()):
 with Path(path).open('x',newline='')as out:
  w=csv.DictWriter(out,FIELDS+list(extra),extrasaction='ignore',lineterminator='\n');w.writeheader()
  for row in rows:
   record={k:row.get(k)for k in FIELDS+list(extra)}
   record={k:(int(v)if type(v)is bool else '|'.join(map(str,v))if isinstance(v,list)else v)for k,v in record.items()};w.writerow(record)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 if a.output.exists():raise FileExistsError('output exists; verify it instead of overwriting')
 a.output.mkdir(parents=True,exist_ok=False);inputs={}
 def read_json(path):
  path=Path(path);inputs[str(path)]=sha(root/path);return load(root/path)
 frozen=read_json('protocols/0002-freeze-v2.json');manifest=read_json(frozen['manifest_path']);config=read_json(frozen['config_path']);result=read_json('results/0002-frozen-evaluation-v2-20261017/evaluation.json');export=read_json('results/0002-frozen-evaluation-v2-audit-20261017/row-export.json');index=read_json('results/0002-evaluation-completion/index.json')
 assert config['motion_residual_scale_px']==8 and config['motion_residual_cap']==10 and config['failure_threshold_px_strict_gt']==4
 assert config['matched_coverage_target']==.8 and config['practical_risk_difference_target']==-.02
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==BASE
 for path,digest in frozen['input_sha256'].items():assert sha(root/path)==digest
 raw={};rows=[];bytrajectory={}
 for e in index['groups']:
  for path,digest in [(e['csv_path'],e['csv_sha256']),(e['runner_receipt_path'],e['runner_receipt_sha256'])]:assert sha(root/path)==digest;inputs[path]=digest
  with(root/e['csv_path']).open(newline='')as z:
   for r in csv.DictReader(z):
    key=(e['archive_index'],int(r['point']),int(r['frame']));assert key not in raw;raw[key]=r
 for path,digest in export['sha256'].items():
  assert sha(root/path)==digest;inputs[path]=digest
  with(root/path).open(newline='')as z:
   for source in csv.DictReader(z):
    r=dict(source)
    for k in ['archive_index','point','frame']:r[k]=int(r[k])
    for k in ['visible','opportunity','coordinate_valid','logit_valid','available']:r[k]=bool(int(r[k]))
    for k in ['error_px','x','y']:r[k]=None if r[k]==''else float(r[k])
    r['failure']=None if r['failure']==''else bool(int(r['failure']))
    for name,column in [('native','score_native'),('augmented','score_augmented'),('visibility_score','score_visibility'),('local_score','score_local')]:r[name]=None if r[column]==''else float(r[column])
    assert opportunity(r)==r['opportunity']and r['failure']==(r['error_px']>4 if r['error_px']is not None else None)
    record=raw[identity(r)];r['x'],r['y']=float(record['x']),float(record['y']);occ,local=float(record['occlusion_logit']),float(record['expected_dist_logit']);coord=math.isfinite(r['x'])and math.isfinite(r['y']);logits=math.isfinite(occ)and math.isfinite(local)
    assert coord==r['coordinate_valid']and logits==r['logit_valid']and(coord and logits)==r['available']
    expected_tie=hashlib.sha256(f"0002|{r['archive_index']}|{manifest['identities'][r['archive_index']]['video_sha256']}|{r['point']}|{r['frame']}".encode()).hexdigest();assert r['tie']==expected_tie
    rows.append(r);bytrajectory.setdefault((r['archive_index'],r['point']),[]).append(r)
 assert len(rows)==len(raw)==3840 and len({identity(r)for r in rows})==3840
 differences=[]
 for trajectory in bytrajectory.values():
  trajectory.sort(key=lambda r:r['frame']);d=residuals([(r['frame'],r['x'],r['y'])for r in trajectory])
  for r in trajectory:
   r.update(d[r['frame']]);r['observed_increment']=r['augmented']-r['native']if r['available']else None;r['increment_absdiff']=None if not r['available']else abs(r['observed_increment']-r['penalty'])
   if r['available']:assert r['increment_absdiff']<=ATOL;differences.append(r['increment_absdiff']);assert abs(r['native']-min(r['visibility_score']+r['local_score'],-math.log(1e-8)))<=ATOL
 ev=[r for r in rows if r['partition']=='evaluation'];eligible=[r for r in ev if opportunity(r)];assert len(eligible)==2209
 nm,norder=matched_set(ev,'native');am,aorder=matched_set(ev,'augmented');no=operational_set(ev,'native',result['thresholds']['native']['threshold']);ao=operational_set(ev,'augmented',result['thresholds']['augmented']['threshold'])
 for selection,accepts,errors in [(nm,1768,4),(am,1768,7),(no,1855,8),(ao,1826,9)]:require_published(ev,selection,2209,accepts,errors)
 for r in rows:
  key=identity(r)
  for field,selection in [('native_matched',nm),('augmented_matched',am),('native_operational',no),('augmented_operational',ao)]:r[field]=key in selection
  for policy,native,aug in [('matched',nm,am),('operational',no,ao)]:r[policy+'_category']=category(key in native,key in aug)if r['partition']=='evaluation'and opportunity(r)else'not_opportunity'
 summaries={};cellrows=[]
 for policy,native,aug in [('matched',nm,am),('operational',no,ao)]:
  cells={c:{'rows':len(part:=[r for r in eligible if r[policy+'_category']==c]),'failures':sum(failed(r)for r in part),'correct':sum(not failed(r)for r in part)}for c in CATEGORIES}
  summaries[policy]={'cells':cells,'native_accepts':len(native),'augmented_accepts':len(aug),'accepts_difference':len(aug)-len(native),'failure_difference':cells['augmented_only']['failures']-cells['native_only']['failures'],'original_native_threshold':result['thresholds']['native']['threshold']if policy=='operational'else None,'original_augmented_threshold':result['thresholds']['augmented']['threshold']if policy=='operational'else None}
  write_rows(a.output/('decision-changes-'+policy+'.csv'),sorted((r for r in eligible if r[policy+'_category']in('native_only','augmented_only')),key=lambda r:identity(r)))
  for group in result['evaluation_groups']:
   for scope,point,part in [('group','all',[r for r in eligible if r['group_id']==group])]+[('trajectory',str(p),[r for r in eligible if r['group_id']==group and r['point']==p])for p in sorted({r['point']for r in ev if r['group_id']==group})]:
    for c in CATEGORIES:
     sub=[r for r in part if r[policy+'_category']==c];cellrows.append({'policy':policy,'scope':scope,'group':group,'point':point,'category':c,'rows':len(sub),'failures':sum(failed(r)for r in sub),'correct':sum(not failed(r)for r in sub)})
 with(a.output/'group-point-cells.csv').open('x',newline='')as z:
  w=csv.DictWriter(z,list(cellrows[0]),lineterminator='\n');w.writeheader();w.writerows(cellrows)
 # Complete descriptive component strata, not independent-frame inference.
 strata={}
 for r in rows:
  partition=r['partition'];keys=['raw:'+partition]
  if opportunity(r):
   outcome='failure'if failed(r)else'correct';keys+=['opportunity:'+partition,'outcome:'+partition+':'+outcome,'group:'+r['group_id']+':'+outcome,'trajectory:'+r['group_id']+':'+str(r['point'])+':'+outcome]
   if partition=='evaluation':
    for policy in ['matched','operational']:keys+=['exchange:'+policy+':'+r[policy+'_category']+':'+outcome]
  for key in keys:strata.setdefault(key,[]).append(r)
 stats={key:{'rows':len(part),'history_insufficient':sum(r['history_insufficient']for r in part),'zero_penalty':sum(r['penalty']==0 for r in part),'capped_penalty':sum(r['penalty']==10 for r in part),'components':{component:distribution([r[component]for r in part])for component in ['native','visibility_score','local_score','residual_px','penalty']}}for key,part in strata.items()}
 (a.output/'component-distributions.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
 failures=sorted((r for r in eligible if failed(r)),key=lambda r:identity(r));write_rows(a.output/'all-evaluation-failures.csv',failures)
 candidates=sorted((r for r in eligible if r['matched_category']=='native_only'and not failed(r)),key=lambda r:(-r['penalty'],r['tie']));cases=[];seen=set()
 for r in candidates:
  trajectory=(r['group_id'],r['point'])
  if trajectory not in seen:cases.append(r);seen.add(trajectory)
  if len(cases)==3:break
 write_rows(a.output/'correct-penalized-cases.csv',cases)
 centers=failures+cases;context={}
 for center in centers:
  label=f"{center['group_id']}:{center['point']}:{center['frame']}"
  for r in bytrajectory[center['archive_index'],center['point']]:
   if abs(r['frame']-center['frame'])<=2:
    copy=context.setdefault(identity(r),dict(r,center_ids=[]));copy['center_ids'].append(label)
 write_rows(a.output/'failure-and-case-context.csv',[context[k]for k in sorted(context)],extra=['center_ids'])
 concentration=[]
 for group in result['evaluation_groups']:
  for point in [None]+sorted({r['point']for r in ev if r['group_id']==group}):
   part=[r for r in eligible if r['group_id']==group and(point is None or r['point']==point)]
   concentration.append({'group':group,'point':'all'if point is None else point,'opportunities':len(part),'all_failures':sum(failed(r)for r in part),'native_matched_failures':sum(failed(r)and identity(r)in nm for r in part),'augmented_matched_failures':sum(failed(r)and identity(r)in am for r in part),'matched_failure_difference':sum(failed(r)and identity(r)in am for r in part)-sum(failed(r)and identity(r)in nm for r in part)})
 with(a.output/'error-concentration.csv').open('x',newline='')as z:
  w=csv.DictWriter(z,list(concentration[0]),lineterminator='\n');w.writeheader();w.writerows(concentration)
 boundary={method:{'last_selected':{k:ordered[1767][k]for k in ['group_id','point','frame','tie',method]},'first_rejected':{k:ordered[1768][k]for k in ['group_id','point','frame','tie',method]}}for method,ordered in [('native',norder),('augmented',aorder)]}
 summary={'status':'completed','scope':'post-hoc descriptive diagnosis; observed evaluation labels, no new confirmatory test or method','source_commit':BASE,'inputs_sha256':inputs,'absolute_tolerance':ATOL,'relative_tolerance':0,'increment_checks':len(differences),'max_increment_absolute_difference':max(differences),'raw_rows':len(rows),'evaluation_opportunities':len(eligible),'policies':summaries,'matched_boundaries':boundary,'correct_penalized_cases':[{k:r[k]for k in ['group_id','point','frame','tie','native','augmented','residual_px','penalty','error_px']}for r in cases],'failure_context_centers':len(failures),'context_rows':len(context),'headroom':{'native_failures':4,'accepts':1768,'maximum_risk_reduction_fraction':4/1768,'best_possible_augmented_minus_native':-4/1768,'unchanged_practical_target':-.02,'failures_reduction_required_for_target':math.ceil(.02*1768),'target_point_estimate_impossible_on_observed_sample':True},'original_paired_ci_unchanged':result['bootstrap']['intervals']['matched_risk_difference'],'bootstrap_redrawn':False,'thresholds_refit':False,'new_inference_or_archive_decodes':0,'raw_history_scope':'all earlier finite coordinates, including rejected, invalid-logit or GT-occluded rows; no future coordinates or GT control history'}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in summary.items()if k not in ['inputs_sha256','matched_boundaries','correct_penalized_cases']},indent=2))

if __name__=='__main__':main()
