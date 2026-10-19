"""Independent saved-probability arithmetic and preservation; no model/fit/scorer imports."""
import json,hashlib,collections,subprocess,math
from pathlib import Path
import numpy as np
b=Path('/workspace/scratch/a349d03a0e0a');c=b/'scratch/0010-headroom-control';o=b/'scratch/0010-headroom-output';r=b/'research-0010-headroom'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def spec(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def ah(a):return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+b'\0'+a.tobytes(order='C')).hexdigest()
def ih(x):return hashlib.sha256(json.dumps(list(x),separators=(',',':')).encode()).hexdigest()
checks=0
def ck(x,msg):
 global checks
 checks+=1
 if not x:raise ValueError(msg)
load=lambda p:json.loads(p.read_text())
f=load(c/'preflight.json');rec=load(o/'receipt.json');saved=load(o/'predictions.json');score=load(o/'scores.json');lock=load(o/'selection-lock.json');seal=load(o/'prediction-seal.json');ex=load(b/'scratch/0010-headroom-supervision/execution.json')
ck(rec['status']=='completed' and rec['stage']=='completed','receipt completed');ck(ex['child_exit_code']==ex['supervisor_exit_code']==ex['raw_wait_status']==0 and not ex['timed_out'],'supervision exit')
ck(ex['wall_limit_seconds']==900 and ex['cpu_affinity']==[0,1,2,3] and ex['address_space_bytes']==8*1024**3,'resources')
ck([rec[k] for k in ['image_calls_attempted','image_calls_completed','images_completed','text_calls','scoring_attempted','scoring_completed']]==[32,32,128,0,1,1],'counts')
for n in ['predictions','features','selection-lock']:
 k=n.replace('-','_')+'_sha256';ck(sha(o/(n+('.npz' if n=='features' else '.json')))==seal[k],'seal '+n)
ck(load(o/'score-started.json')==seal,'score lock');ck(sha(o/'scores.json')==rec['scores_sha256'],'score hash');ck(sha(o/'prediction-seal.json')==rec['prediction_seal_sha256'],'seal hash')
for n,d in ex['logs'].items():ck(spec(b/'scratch/0010-headroom-supervision'/n)==d,'closed supervision log')
config=load(r/'experiments/0010/config.json');classes=config['class_order'];names=lock['names'];ids=lock['ids'];rows=saved['rows'];arms=['zero_shot','native_alpha1','native_alpha4','loo_alpha1','other_out_alpha1'];controls=[a for a in arms if a!='loo_alpha1']
train=(b/'clip-0010-assets/metadata/eurosat-spatial-train.txt').read_text().splitlines();excluded=load(b/'scratch/0010-capability-output/receipt.json')['names']
expected=sorted(set(train)-set(excluded),key=lambda n:(hashlib.sha256(('0010-stata-headroom-v1|'+n).encode()).hexdigest(),n))[:128]
ck(names==expected and len(set(names))==128 and lock['batches']==[i//64 for i in range(128)],'prospective selection')
ck(ids==[hashlib.sha256(('0010-stata-headroom-id-v1|'+n).encode()).hexdigest() for n in names],'opaque ids')
ck([x['id'] for x in rows]==ids and saved['status']=='completed' and saved['arms']==arms,'rows')
y=[classes.index(n.rsplit('_',1)[0]) for n in names];pred={a:[] for a in arms}
for i,row in enumerate(rows):
 ck(row['batch']==i//64 and row['donor_id']==ids[(i//64)*64+(i+1)%64],'donor')
 for a in arms:
  p=row['probabilities'][a];ck(row['statuses'][a]=='ok' and len(p)==10 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in p),'probabilities')
  ck(abs(math.fsum(p)-1)<=2e-6 and sorted(p)[-1]!=sorted(p)[-2],'mass/tie');pred[a].append(max(range(10),key=lambda k:p[k]))
correct={a:[v==t for v,t in zip(pred[a],y)] for a in arms};counts={a:sum(correct[a]) for a in arms};acc={a:v/128 for a,v in counts.items()}
def pair(indices,a):
 win=sum(correct['loo_alpha1'][i] and not correct[a][i] for i in indices);harm=sum(not correct['loo_alpha1'][i] and correct[a][i] for i in indices)
 return dict(n=len(indices),corrections=win,harms=harm,net=win-harm,disagreements=sum(pred['loo_alpha1'][i]!=pred[a][i] for i in indices))
paired={a:pair(list(range(128)),a) for a in controls};batches=[{a:pair(list(range(k*64,(k+1)*64)),a) for a in controls} for k in range(2)]
occupancy=[]
for k in range(2):
 co=collections.Counter(y[k*64:(k+1)*64]);occupancy.extend(co[v] for v in y[k*64:(k+1)*64])
strata={name:{a:pair([i for i,x in enumerate(occupancy) if test(x)],a) for a in controls} for name,test in [('one',lambda x:x==1),('two',lambda x:x==2),('three_plus',lambda x:x>=3)]}
best=max(counts[a] for a in controls);winners=[a for a in controls if counts[a]==best];gates=dict(net_eight_over_best=counts['loo_alpha1']-best>=8,both_batches_positive=all(batches[k][a]['net']>0 for k in range(2) for a in winners),no_singleton_net_harm=all(strata['one'][a]['net']>=0 for a in ['native_alpha1','native_alpha4']))
low=128-best<8 or any(paired[a]['disagreements']<8 for a in winners);decision='CONTINUE_TO_MECHANISM' if all(gates.values()) else ('LOW_HEADROOM' if low else 'NO_GO')
recomputed=dict(n=128,correct=counts,accuracy=acc,strongest_fixed_controls=winners,paired=paired,batches=batches,occupancy_strata=strata,singleton_assessed=1 in occupancy,gates=gates,decision=decision)
for k,v in recomputed.items():ck(v==score[k],'independent score '+k)
ck(len(saved['fit_states'])==132 and saved['successful_native_solver_executions']==264,'fit count')
for k,e in enumerate(saved['fit_states']):
 batch=k//66;j=k%66;bid=ids[batch*64:(batch+1)*64];delete=None if j<2 else bid[j-2];support=[v for v in bid if v!=delete]
 ck(e['status']=='completed' and e['batch']==batch and e['excluded_id']==delete and e['alpha']==(4 if j==1 else 1) and e['support_n']==len(support) and e['support_ids_sha256']==ih(support) and e['tag']==('full-alpha'+str(4 if j==1 else 1) if j<2 else 'delete-'+str(j-2)),'fit lineage')
features={}
with np.load(o/'features.npz',allow_pickle=False) as z, np.load(o/'partial-features.npz',allow_pickle=False) as p:
 ck(set(z.files)=={'image_raw','image_normalized','text_normalized'} and set(p.files)=={'image_raw'},'array schema')
 for n,shape in [('image_raw',(128,512)),('image_normalized',(128,512)),('text_normalized',(10,512))]:
  a=z[n];ck(a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all(),'shape finite');features[n]=dict(shape=list(a.shape),dtype=str(a.dtype),array_sha256=ah(a))
 norms=np.linalg.norm(z['image_raw'],axis=1,keepdims=True);ck((norms>0).all() and np.array_equal(z['image_raw']/norms,z['image_normalized']),'normalization');ck(np.array_equal(z['image_raw'],p['image_raw']),'partial same')
 err=float(np.max(np.abs(np.linalg.norm(z['image_normalized'],axis=1)-1)));ck(err<=2e-5,'unit norms')
 ck(saved['inputs']==dict(features_sha256=ah(z['image_normalized']),prototypes_sha256=ah(z['text_normalized'].T[None].copy()),ids_sha256=ih(ids)),'input source binding')
 with np.load(b/'scratch/0010-capability-output/features.npz',allow_pickle=False) as t:ck(np.array_equal(z['text_normalized'],t['text_normalized']),'reused text')
for rel,d in f['preserved'].items():ck(spec(b/rel)==d,'preservation '+rel)
for repo in ['research-0010-capability','research-0010-cached-admission','research-0010-headroom']:ck(not subprocess.check_output(['git','status','--porcelain'],cwd=b/repo),'clean '+repo)
freeze=load(r/'protocols/0010-real-data-headroom-freeze.json')
for rel,d in freeze['inputs'].items():ck(spec(r/rel)=={k:d[k] for k in ['bytes','sha256']},'freeze unchanged')
newmarkers={p:spec(b/'clip-0010-assets'/p) for p in ['0010-headroom-launch.started.json','0010-headroom-started.json']}
binary={p:spec(o/p) for p in ['features.npz','partial-features.npz']}
report=dict(status='PASS_INDEPENDENT_SAVED_ARITHMETIC_AND_PRESERVATION',checks=checks,recomputed=recomputed,fit_states=132,authored_executions=264,features=features,max_image_unit_norm_error=err,private_npz_not_exported=binary,preserved_count=len(f['preserved']),new_markers=newmarkers,source=freeze['source_commit'],main=f['head'],new_asset_bytes=0,direct_body_unchanged=456087034,historical_all_HTTP='UNKNOWN',limits='Two computation batches are not independent scenes or a population CI; LOO/OTHER peer-set confound remains. No new inference/refit/scorer invocation.',output_files={p.name:spec(p) for p in o.iterdir() if p.is_file()})
(c/'saved-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=report['status'],checks=checks,decision=decision,correct=counts,preserved_count=report['preserved_count'],max_unit_norm_error=err)))
