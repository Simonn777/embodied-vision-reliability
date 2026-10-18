"""Independent NumPy/stdlib-only archived-output audit; never imports repo scripts.
Usage: python3 check_saved.py REPOSITORY > audit.json
No cloud, network, raw dataset, checkpoint or model access. Source files read as text.
"""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
import numpy as np

HEAD='0436cb438666db9bd0645f131632fc2fafcdf025'
BASE='f0787ab50f5dfc31f8b8a58bdcb165a12d2d83a6'
R=Path(sys.argv[1]).resolve()
D=R/'results/0005-recovered-capability'
N=D/'native'; P=D/'preparation'; C=D/'control'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def require(condition, message):
 if not condition:raise AssertionError(message)
def check_hash(p,digest):require(sha(p)==digest,'hash differs: '+str(p))
def dt(text):return datetime.datetime.fromisoformat(text)
def checked_logs(rec,folder,prefix,exitcode=0):
 require(rec['exit_code']==exitcode and rec['raw_wait_status']==exitcode*256,'exit differs: '+prefix)
 for s in ('stdout','stderr'):check_hash(folder/(prefix+s+'.log'),rec[s+'_sha256'])
 elapsed=(dt(rec.get('ended_utc',rec.get('finished_utc')))-dt(rec['started_utc'])).total_seconds()
 monotonic=rec.get('wall_seconds',rec.get('elapsed_seconds'))
 require(abs(elapsed-monotonic)<0.02,'UTC/perf duration differs: '+prefix)
 return {'exit':exitcode,'raw_wait_status':rec['raw_wait_status'],'wall_seconds':monotonic,'peak_rss_kib':rec['peak_rss_kib']}

require(git('rev-parse','HEAD').decode().strip()==HEAD,'wrong HEAD')
require(git('status','--porcelain')==b'','checkout not clean')
freeze=load(R/'protocols/0005-native-recovery-freeze.json')
require(len(freeze['files'])==16,'freeze count')
check_hash(R/'protocols/0005-native-recovery-freeze.json','916cdabbdc8d5be7dcb0060ca54216457d982a3bdc564c6509c79195e51966ef')
for path,digest in freeze['files'].items():
 check_hash(R/path,digest)
 for ref in (HEAD,BASE,freeze['source_commit'],''):
  require(hashlib.sha256(git('show',ref+':'+path)).hexdigest()==digest,'frozen git bytes differ '+ref+' '+path)
original=load(R/'protocols/0005-native-capability-freeze.json')
require(len(original['files'])==7,'original freeze count')
for p,h in original['files'].items():check_hash(R/p,h)
manifest=load(D/'copy-manifest.json')['files']
require(len(manifest)==74 and len({x['copied'] for x in manifest})==74,'copy count')
for item in manifest:
 p=R/item['copied'];require(p.is_file() and not p.is_symlink(),'copy path')
 require(p.stat().st_size==item['bytes'],'copy size '+item['copied']);check_hash(p,item['sha256'])
require({p.relative_to(R).as_posix() for p in D.rglob('*') if p.is_file()}=={x['copied'] for x in manifest}|{str((D/'README.md').relative_to(R)),str((D/'copy-manifest.json').relative_to(R))},'unlisted archive file')

# Independent historical-object verification, including the baseline's entire hash map.
pre=load(C/'pre-execution.json');ref=pre['old_tracked_baseline_reference'];check_hash(R/ref['path'],ref['sha256'])
baseline=load(R/ref['path']);require(len(baseline)==1769,'old baseline count')
baseline.update(pre['execution_main_incremental_tracked_hashes'])
require(len(baseline)==1832,'merged base tracked count')
require(set(git('ls-tree','-r','--name-only',BASE).decode().splitlines())==set(baseline),'baseline missing tracked paths')
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for path,digest in baseline.items():
 proc.stdin.write((BASE+':'+path+'\n').encode());proc.stdin.flush()
 header=proc.stdout.readline().decode().strip().split()
 require(len(header)==3 and header[1]=='blob','bad historical object')
 remaining=int(header[2]);h=hashlib.sha256()
 while remaining:
  block=proc.stdout.read(min(1024*1024,remaining));require(bool(block),'truncated git object');h.update(block);remaining-=len(block)
 require(proc.stdout.read(1)==b'\n','git framing');require(h.hexdigest()==digest,'baseline object hash '+path)
proc.stdin.close();require(proc.wait()==0,'git batch failed')
oldresults={p:h for p,h in baseline.items() if p.startswith('results/')}
require(len(oldresults)==1651,'prior result count')
for p,h in oldresults.items():check_hash(R/p,h)
modified=[p for p,h in baseline.items() if sha(R/p)!=h]
require(sorted(modified)==['CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-18.md'],'unexpected old file changed')

rec=load(N/'run/receipt.json');outer=load(N/'outer.json');external=load(C/'native-execution.json')
require(rec['status']=='completed_native_capability' and rec['frames_completed']==rec['forwards_attempted']==len(rec['frames'])==8,'completion mismatch')
require(outer['source_sha256']==freeze['files'],'outer freeze map')
for obj in (rec['provenance'],outer,external):require(obj['freeze_sha256']==sha(R/'protocols/0005-native-recovery-freeze.json'),'receipt freeze')
require(rec['adapter_sha256']==sha(R/'scripts/tapnext_native_adapter.py'),'adapter binding')
check_hash(P/'0005-recovered-native-started.json',rec['marker_sha256'])
check_hash(P/'0005-recovered-supervisor-started.json',outer['launch_marker_sha256'])
for stage in ('dependencies','checkpoint'):
 check_hash(P/('recovery-'+stage+'.json'),rec['provenance']['preparation_receipt_sha256'][stage])
require(rec['thread_counts']==[4,4],'thread counts')
for obj in (rec,outer,external):require(obj['limits']['affinity']==[0,1,2,3] and obj['limits']['AS_bytes']==8*1024**3,'resource limits')
require(rec['cgroup_memory_max']=='8589934592','cgroup memory')
for obj in (outer,external):require(obj['command'][:3]==['timeout','--kill-after=5s','1800'],'external timeout')
operations={}
for name in ('verify-before-dependencies','dependencies','verify-before-checkpoint','checkpoint','native','software'):
 d=load(C/(name+'-execution.json'));operations[name]=checked_logs(d,C,name+'.')
 check_hash(C/'observe_recovery.py',d['observer_source_sha256'])
operations['native-child']=checked_logs(outer,N,'')
operations['import']=checked_logs(load(P/'import-check.json'),P,'import-check.')
for name,code in [('dependencies-review',0),('checkpoint-review',1),('checkpoint-review-v2',0),('native-review',1),('native-review-v2',0)]:
 d=load(C/(name+'-execution.json'));operations[name]=checked_logs(d,C,name+'.',code)
 check_hash(C/Path(d['command'][4]).name,d['source_sha256'])
require(external['wall_seconds']>=outer['wall_seconds']>=rec['elapsed_seconds'],'enclosing durations')
require(external['peak_rss_kib']==outer['peak_rss_kib']==rec['peak_rss_kib']==3727996,'rss mismatch')
require(dt(external['started_utc'])<dt(outer['started_utc'])<dt(rec['started_utc'])<dt(rec['finished_utc'])<dt(outer['ended_utc'])<dt(external['ended_utc']),'UTC ordering')

expected={'tracks_model_yx':((1,1,8,2),'float32'),'coordinate_logits':((1,1,8,512),'float32'),'visibility_logits':((1,1,8,1),'float32'),'positions_xy':((8,2),'float32'),'visible':((8,),'bool')}
rows=list(csv.DictReader((C/'saved-predictions.csv').open(newline='')))
require(len(rows)==64,'CSV row count');floats=0;booleans=0;query_hashes=set();layer_hashes=[];forwards=[];frame_whole=[];head_differences=[]
for t,fr in enumerate(rec['frames']):
 require(fr['frame']==t and fr['output_file']==f'frame-{t:03}.npz','frame ids')
 path=N/'run'/fr['output_file'];check_hash(path,fr['output_sha256'])
 with zipfile.ZipFile(path) as z:require(z.testzip() is None and len(z.infolist())==5,'NPZ CRC/keycount')
 with np.load(path,allow_pickle=False) as z:
  require(set(z.files)==set(expected),'array key set');arrays={k:z[k] for k in z.files}
 for k,(shape,dtype) in expected.items():
  a=arrays[k];require(a.shape==shape and str(a.dtype)==dtype and np.isfinite(a).all(),'schema/finite '+k)
  if dtype=='bool':booleans+=a.size
  else:floats+=a.size
 require(np.array_equal(arrays['positions_xy'],arrays['tracks_model_yx'][0,0,:,::-1]),'wrapper axes')
 require(np.array_equal(arrays['visible'],arrays['visibility_logits'][0,0,:,0]>0),'wrapper visibility')
 # Extra mathematical audit only: independently re-evaluate the saved-head formula
 # in float64. This is NOT another model forward or a changed scientific gate.
 logits=arrays['coordinate_logits'][0,0].astype(np.float64).reshape(8,2,256)
 index=np.arange(256);center=logits.argmax(-1)[...,None]
 probs=np.exp((logits-logits.max(-1,keepdims=True))*0.5)*(np.abs(index-center)<=20)
 predicted=(probs*index).sum(-1)/probs.sum(-1)+0.5
 head_differences.append(float(np.max(np.abs(predicted-arrays['tracks_model_yx'][0,0]))))
 for q in range(8):
  row=rows[t*8+q]
  require(int(row['frame'])==t and int(row['query'])==q,'CSV grid')
  require([float(row['predicted_x']),float(row['predicted_y'])]==arrays['positions_xy'][q].astype(float).tolist(),'CSV xy')
  require(float(row['visibility_logit'])==float(arrays['visibility_logits'][0,0,q,0]),'CSV logit')
  require(int(row['predicted_visible'])==int(arrays['visible'][q]),'CSV visibility')
 state=fr['state'];require(state['step']==t+1 and len(state['hidden_state'])==12,'state step/layers')
 for name,value,shape in [('query_points',state['query_points'],[1,8,3])]+[(key,layer[key],shape) for layer in state['hidden_state'] for key,shape in [('rg_lru_state',[1032,768]),('conv1d_state',[1032,3,768])]]:
  require(value['shape']==shape and value['bytes']==math.prod(shape)*4 and value['finite'] is True and value['dtype']=='float32' and value['device']=='cpu' and re.fullmatch('[0-9a-f]{64}',value['sha256']),'state assertion '+name)
 query_hashes.add(state['query_points']['sha256']);layer_hashes.extend(layer['rg_lru_state']['sha256'] for layer in state['hidden_state'])
 forwards.append(fr['track_frame_seconds']);frame_whole.append(fr['whole_frame_with_validation_save_seconds'])
 require(0<forwards[-1]<=frame_whole[-1],'frame timing')
require(floats==33088 and booleans==64 and len(query_hashes)==1,'totals/query stability')
phase_total=sum(rec['phases_seconds'].values());inference=rec['phases_seconds']['inference_loop_with_validation_and_artifacts']
require(sum(forwards)<=sum(frame_whole)<=inference and phase_total<=rec['elapsed_seconds'],'phase containment')

# Cross-bind recovery records and archived manifests; live cloud files are NOT accessed.
deps=load(P/'recovery-dependencies.json');ckpt=load(P/'recovery-checkpoint.json')
require(deps['status']==ckpt['status']=='completed' and deps['new_network_payload_bytes']==0,'preparation status')
for stage,d in [('dependencies',deps),('checkpoint',ckpt)]:
 require(d['preserved_failed_cache'] is True and d['model_constructed'] is False,'preparation boundary')
 for key in ('source_manifest','overlay_manifest','ledger_snapshot'):check_hash(P/d[key]['relative_path'],d[key]['sha256'])
 check_hash(P/'markers'/(stage+'.started.json'),d['marker_sha256'])
 require(d['config_sha256']==sha(R/'configs/0005-native-recovery.json') and d['source_sha256']==sha(R/'scripts/prepare_tapnext_recovery.py'),'preparer binding')
require(deps['import_check']==load(P/'import-check.json'),'import child identity')
require(ckpt['dependency_receipt_sha256']==sha(P/'recovery-dependencies.json'),'checkpoint/dependencies binding')
asset=load(P/'asset-records/checkpoint-256.json');require(ckpt['checkpoint_asset']==asset,'asset receipt')
check_hash(P/'asset-records/checkpoint-256.json',ckpt['checkpoint_asset_record_sha256'])
require(asset['http_payload_attempts']==1 and asset['http_status']==200 and not asset['partial_exists'] and asset['partial_bytes_stored']==0,'checkpoint GET/partial report')
require(asset['payload_bytes_observed']==asset['asset']['bytes']==2532282370 and asset['sha256']==asset['observed_payload_sha256']==rec['checkpoint_sha256']==asset['asset']['sha256'],'checkpoint bytes report')
ledger=load(P/'acquisition-ledger.json');require(ledger==load(P/'acquisition-ledger-checkpoint.json'),'latest ledger')
require(ledger['base_ceiling_bytes']==4823387101+128*1024**2,'inherited base+metadata')
require(ledger['conservative_cumulative_ceiling_bytes']==ledger['base_ceiling_bytes']+sum(ledger['reservations'].values())==7499066782<10*1024**3,'ceiling formula')
require(ledger['payload_bytes_observed']==9179574+2532282370 and ledger['new_payload_bytes_observed']==2532282370,'observed payload')
source=load(P/'source-manifest.json')['files'];overlay=load(P/'overlay-manifest.json')['files']
require(len(source)==5 and len(overlay)==378,'manifest counts')
for module,digest in rec['source_sha256'].items():require(source[module.replace('.','/')+'.py']['sha256']==digest,'upstream hash binding')
oldinv=load(R/pre['failed_cache_inventory_reference']['path']);require(len(oldinv)==407,'failed cache archived manifest count')
for relative,item in overlay.items():require(oldinv['overlay/'+relative]==item,'prior overlay manifest differs')
require('torch' not in sys.modules,'model package imported')
result={'status':'passed_archived_artifact_audit','head':HEAD,'base':BASE,'script_sha256':sha(__file__),
 'freeze_inputs':16,'original_freeze_inputs':7,'copy_manifest_files':74,'historical_base_blobs_checked':1832,'prior_result_bytes_preserved':1651,'intentional_state_updates':modified,
 'frames':8,'CSV_rows_exactly_equal':64,'finite_float_scalars':floats,'boolean_outputs':booleans,'wrapper_array_equal':True,'query_state_hashes':len(query_hashes),'stored_state_steps':list(range(1,9)),'stored_layers_per_step':12,'distinct_rg_lru_hashes':len(set(layer_hashes)),
 'independent_float64_saved_head_max_abs_difference_px':max(head_differences),'saved_head_note':'Numerical diagnostic only; no scientific threshold changed; saved logits only.',
 'sum_forward_seconds':sum(forwards),'first_forward_seconds':forwards[0],'later_mean_forward_seconds':sum(forwards[1:])/7,'sum_frame_artifact_seconds':sum(frame_whole),'phase_sum_seconds':phase_total,'adapter_seconds':rec['elapsed_seconds'],'adapter_unattributed_seconds':rec['elapsed_seconds']-phase_total,'operations':operations,
 'checkpoint_first_GET_recorded_bytes':2532282370,'conservative_ceiling_bytes':ledger['conservative_cumulative_ceiling_bytes'],
 'limitations':['No cloud filesystem or downloaded checkpoint rechecked in this parent audit; those claims are cross-bound saved runtime evidence.','State tensors absent: recomputed schema/hash consistency only, not their numeric values or actual SHA preimages.','Observer parents have no additional durable outer wait receipt.','Eight-frame capability is not efficacy, novelty, real-video transfer or long-clip validation.'],'model_imported':False,'network_or_model_execution':False}
print(json.dumps(result,indent=2,allow_nan=False))
