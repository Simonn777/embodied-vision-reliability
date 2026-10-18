"""Audit saved native outputs, receipts and preservation; no model/GT access."""
import csv,hashlib,json,math,re,statistics,subprocess,sys
from pathlib import Path
import numpy as np
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0005-recovery';C=Path(__file__).parent;N=B/'tapnextpp-recovery';O=B/'scratch/0005-recovered-native-capability'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def inventory(root):
 out={}
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():out[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
 return out
def logs(rec,base,stem):
 assert rec['exit_code']==0 and rec['raw_wait_status']==0
 for s in ['stdout','stderr']:assert sha(base/(stem+s+'.log'))==rec[s+'_sha256']
def main():
 assert not(C/'native-audit.json').exists()and'torch'not in sys.modules
 outer=load(C/'native-execution.json');logs(outer,C,'native.')
 inner=load(O/'outer.json');logs(inner,O,'')
 assert sha(C/'observe_recovery.py')==outer['observer_source_sha256']
 rec=load(O/'run/receipt.json');assert rec['status']=='completed_native_capability'
 assert rec['forwards_attempted']==rec['frames_completed']==len(rec['frames'])==8
 cfg=load(R/'configs/0005-native-capability.json');freeze=load(R/'protocols/0005-native-recovery-freeze.json')
 assert rec['thread_counts']==[4,4] and rec['limits']['affinity']==[0,1,2,3]and rec['limits']['AS_bytes']==8589934592
 assert rec['checkpoint_sha256']=='cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da'
 assert sha(B/'bootstapir-cache/development-000.npz')==rec['development_npz_sha256']
 assert sha(N/'0005-recovered-native-started.json')==rec['marker_sha256']
 assert sha(N/'0005-recovered-supervisor-started.json')==inner['launch_marker_sha256']
 assert rec['provenance']['freeze_sha256']==inner['freeze_sha256']==outer['freeze_sha256']==sha(R/'protocols/0005-native-recovery-freeze.json')
 for stage in ['dependencies','checkpoint']:assert rec['provenance']['preparation_receipt_sha256'][stage]==sha(N/('recovery-'+stage+'.json'))
 for p,h in freeze['files'].items():
  assert sha(R/p)==h
  for prefix in ['HEAD:',':',freeze['source_commit']+':']:assert hashlib.sha256(subprocess.check_output(['git','show',prefix+p],cwd=R)).hexdigest()==h
 for row in rec['imports'].values():assert sha(Path(row['path']))==row['sha256']
 for p,h in rec['source_sha256'].items():
  # Source binding keys are upstream paths under the independent recovery source.
  path=Path(p)if Path(p).is_absolute()else N/'source'/p
  assert sha(path)==h,(p,path)
 expected={'tracks_model_yx':((1,1,8,2),'float32'),'coordinate_logits':((1,1,8,512),'float32'),'visibility_logits':((1,1,8,1),'float32'),'positions_xy':((8,2),'float32'),'visible':((8,),'bool')}
 rows=[];frames=[];qhash=None;floats=0
 for t,fr in enumerate(rec['frames']):
  assert fr['frame']==t and fr['output_file']==f'frame-{t:03d}.npz'
  p=O/'run'/fr['output_file'];assert sha(p)==fr['output_sha256']
  with np.load(p,allow_pickle=False)as z:
   assert set(z.files)==set(expected)
   arrays={k:z[k]for k in z.files}
  for k,(shape,dtype)in expected.items():
   a=arrays[k];assert a.shape==shape and str(a.dtype)==dtype and np.isfinite(a).all()
   if dtype!='bool':floats+=a.size
  assert np.array_equal(arrays['positions_xy'],arrays['tracks_model_yx'][0,0,:,::-1])
  assert np.array_equal(arrays['visible'],arrays['visibility_logits'][0,0,:,0]>0)
  s=fr['state'];assert s['step']==t+1 and len(s['hidden_state'])==12
  q=s['query_points'];assert q['shape']==[1,8,3]and q['bytes']==96 and q['dtype']=='float32'and q['device']=='cpu'and q['finite']is True
  if qhash is None:qhash=q['sha256']
  assert q['sha256']==qhash
  for layer in s['hidden_state']:
   for k,shape in [('rg_lru_state',[1032,768]),('conv1d_state',[1032,3,768])]:
    a=layer[k];assert a['shape']==shape and a['bytes']==math.prod(shape)*4 and a['dtype']=='float32'and a['device']=='cpu'and a['finite']is True and re.fullmatch('[0-9a-f]{64}',a['sha256'])
  for qid in range(8):rows.append([t,qid,*map(float,arrays['positions_xy'][qid]),float(arrays['visibility_logits'][0,0,qid,0]),int(arrays['visible'][qid])])
  frames.append({'frame':t,'sha256':sha(p),'bytes':p.stat().st_size,'track_frame_seconds':fr['track_frame_seconds'],'whole_frame_with_validation_save_seconds':fr['whole_frame_with_validation_save_seconds'],'visible_predictions':int(arrays['visible'].sum()),'arrays':{k:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}for k,a in arrays.items()}})
 assert len(rows)==64 and {(x[0],x[1])for x in rows}=={(f,q)for f in range(8)for q in range(8)}
 with(C/'saved-predictions.csv').open('x',newline='')as f:
  w=csv.writer(f);w.writerow(['frame','query','predicted_x','predicted_y','visibility_logit','predicted_visible']);w.writerows(rows)
 before=load(C/'pre-execution.json');ref=before['old_tracked_baseline_reference'];assert sha(R/ref['path'])==ref['sha256']
 baseline=load(R/ref['path']);baseline.update(before['execution_main_incremental_tracked_hashes'])
 assert len(baseline)==before['execution_main_tracked_count']
 for p,h in baseline.items():assert sha(R/p)==h,p
 inv=inventory(B/'tapnextpp-cache');assert inv==load(R/'results/0005-native-capability/review/postfailure-cache-inventory.json')
 for p,h in before['existing_old_markers'].items():assert sha(B/p)==h['sha256']
 ledger=load(N/'acquisition-ledger.json');assert ledger['new_payload_bytes_observed']==2532282370 and ledger['conservative_cumulative_ceiling_bytes']==7499066782
 # Stored state arrays are hashes only: do not claim to independently recompute them.
 result={'status':'passed_saved_output_checks','scope':'8-frame native capability only; no accuracy/GT/model rerun','frames_completed':8,'forwards_attempted_recorded':8,'grid_rows':len(rows),'finite_float_scalars':int(floats),'finite_boolean_outputs':64,'frames':frames,'first_track_frame_seconds':frames[0]['track_frame_seconds'],'subsequent_track_frame_mean_seconds':statistics.mean(x['track_frame_seconds']for x in frames[1:]),'sum_track_frame_seconds':sum(x['track_frame_seconds']for x in frames),'outer_exit':outer['exit_code'],'inner_exit':inner['exit_code'],'outer_wall_seconds':outer['wall_seconds'],'inner_wall_seconds':inner['wall_seconds'],'adapter_elapsed_seconds':rec['elapsed_seconds'],'peak_rss_kib':rec['peak_rss_kib'],'execution_main':before['execution_main'],'frozen_inputs_verified':len(freeze['files']),'prior_tracked_files_preserved':len(baseline),'prior_results_preserved':before['execution_main_result_count'],'failed_cache_files_preserved':len(inv),'old_baseline_reference':ref,'saved_predictions_sha256':sha(C/'saved-predictions.csv'),'source_state_limit':'Independent reviewer checks stored state step/shape/dtype/finite assertions and hash format/query-hash stability. State tensors were not retained, so their values/hashes cannot be independently recomputed here. The runtime adapter checked and hashed actual tensors.','checkpoint_review_correction':'Initial non-scientific checker used checkpoint.json instead of frozen asset id checkpoint-256.json and exited 1. Its source/logs remain. Version 2 changed only record path lookup to the frozen asset id, verified saved bytes, and exited 0. No preparation/download/model stage was repeated.','new_checkpoint_payload_bytes':2532282370,'prior_current_ceiling_bytes':4966784411,'conservative_cumulative_ceiling_bytes':7499066782,'no_torch_import':True}
 with(C/'native-audit.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:v for k,v in result.items()if k!='frames'},indent=2))
if __name__=='__main__':main()
