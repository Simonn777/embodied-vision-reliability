"""Publishable metadata/aggregate evidence only; never copies pixels or weights."""
import datetime,hashlib,json,os,platform,shutil,statistics,subprocess
from pathlib import Path
B=Path('/workspace/scratch/a349d03a0e0a');R=B/'research-0006-screen';C=Path(__file__).parent;N=B/'conditional-residual-0006';D=R/'results/0006-stage0'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 assert not D.exists();pre=read(C/'pre-execution.json');base=read(R/pre['historical_baseline_reference']['path']);base.update(pre['current_main_incremental_tracked_hashes']);assert len(base)==1941
 for p,h in base.items():assert sha(R/p)==h,p
 for p,h in pre['old_cache_markers'].items():assert sha(B/p)==h,p
 freeze=read(R/'protocols/0006-stage0-freeze.json')
 for p,h in freeze['files'].items():
  assert sha(R/p)==h
  for prefix in ['HEAD:',':',freeze['source_commit']+':']:assert hashlib.sha256(subprocess.check_output(['git','show',prefix+p],cwd=R)).hexdigest()==h
 inventory={}
 for p in sorted(N.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():inventory[p.relative_to(N).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
 write(C/'cache-inventory.json',{'scope':'metadata and hashes only; payloads remain in local cache','files':inventory})
 old={}
 for p in sorted((B/'tapnextpp-cache').rglob('*')):
  if p.is_file():old[p.relative_to(B/'tapnextpp-cache').as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
 assert old==read(R/'results/0005-native-capability/review/postfailure-cache-inventory.json')
 scores=read(C/'scores.json');ind=read(C/'independent-scores.json');assert scores['status']==ind['scientific_status']=='stop_this_candidate'
 stages={s:read(C/(s+'-execution.json'))for s in ['tests','acquire','prepare','group-d1','group-d2','group-d3','group-d4','score']}
 for name,x in stages.items():
  assert x['exit_code']==x['raw_wait_status']==0 and x['observer_sha256']==sha(C/'observe_stage.py')
  for stream in ['stdout','stderr']:assert sha(C/(name+'.'+stream+'.log'))==x[stream+'_sha256']
 assert stages['tests']['tests_observed']==66 and stages['tests']['skipped_lines']==0
 methods=[]
 for co in ['white','colored']:
  for m in ['blindspot','safe_blindspot','residual','r2r_alpha05']:
   ss=[r['score']for r in scores['cases']if r['condition']==co and r['method']==m]
   methods.append({'condition':co,'method':m,'pairs':len(ss),'mean_psnr_db':statistics.mean(x['psnr_db']for x in ss),'mean_first_difference_mse':statistics.mean(x['first_difference']['combined']['mse']for x in ss)})
 cpu=next((s.split(':',1)[1].strip()for s in Path('/proc/cpuinfo').read_text().splitlines()if s.startswith('model name')),None)
 summary={'scope':'prospective synthetic-corruption DEVELOPMENT screen; not real-camera denoising or a paper result','AI_assisted':True,'execution_main':pre['execution_main'],'source_commit':pre['source_commit'],'freeze_commit':pre['freeze_commit'],'freeze_sha256':pre['freeze_sha256'],'fits_completed':64,'updates_per_fit':500,'numerical_gate_status':scores['status'],'mean_colored_gain_db':scores['mean_colored_gain_db'],'group_mean_colored_gain_db':scores['group_mean_colored_gain_db'],'gradient_error_ratio':scores['mean_candidate_first_difference_mse']/scores['mean_corresponding_baseline_first_difference_mse'],'gates':scores['gates'],'mean_method_metrics':methods,'colored_pairs':scores['colored_pairs'],'provenance_status':'OPEN_STDOUT_HASH_DISCREPANCY','log_gap':read(C/'log-gap-diagnostic.json'),'strict_grid_review_exit':read(C/'grid-review-execution.json')['exit_code'],'independent_numeric_comparisons':ind['numeric_comparisons'],'independent_max_absolute_difference':ind['max_absolute_numeric_difference'],'stages':[{k:x[k]for k in ['stage','started_utc','ended_utc','exit_code','raw_wait_status','wall_seconds','peak_rss_kib','CPU_user_seconds','CPU_system_seconds']}for x in stages.values()],'sum_group_wall_seconds':sum(stages['group-'+g]['wall_seconds']for g in ['d1','d2','d3','d4']),'fit_seconds_sum':read(C/'postfailure-integrity.json')['fit_seconds_sum'],'cpu_model_observed_after_run':cpu,'runtime':{'python':platform.python_version(),'platform':platform.platform(),'cpu_affinity':[0,1,2,3],'numeric_threads':4,'torch_intra_threads':4,'torch_interop_threads':1,'AS_bytes':8589934592},'new_data_payload_bytes':read(N/'acquisition.json')['data_payload_bytes'],'conservative_acquisition_ceiling_bytes':read(N/'acquisition.json')['conservative_ceiling_bytes'],'prior_tracked_files_preserved':len(base),'prior_results_preserved':pre['current_main_results_count'],'old_failed_cache_files_preserved':len(old),'old_markers_preserved':len(pre['old_cache_markers']),'old_baseline_reference':pre['historical_baseline_reference'],'no_retraining':True,'scorer_invocations':1,'next':'Stop this exact candidate/configuration. Independently review the open log discrepancy and published measurements. No test tuning, repeat fit, manuscript or new experiment is authorized by these results.'}
 write(C/'summary.json',summary)
 D.mkdir();pairs=[]
 for p in sorted(C.iterdir()):
  if p.is_file()and p.name!='scores.json':
   assert p.suffix in ['.py','.json','.log'];pairs.append((p,'control/'+p.name))
 for p in sorted((N/'assets').iterdir()):
  if p.suffix=='.json':pairs.append((p,'acquisition/'+p.name))
 for name in ['acquisition.json','prepare.started.json','prepared.json']:pairs.append((N/name,'inputs/'+name))
 for p in sorted((N/'cases').rglob('*')):
  if p.is_file()and p.suffix in ['.json','.log']:pairs.append((p,'cases/'+p.relative_to(N/'cases').as_posix()))
 mappings=[]
 for src,rel in pairs:
  dst=D/rel;dst.parent.mkdir(parents=True,exist_ok=True);assert src.stat().st_size<1048576
  shutil.copyfile(src,dst);assert sha(dst)==sha(src)
  mappings.append({'source':src.relative_to(B).as_posix(),'copied':dst.relative_to(R).as_posix(),'bytes':dst.stat().st_size,'sha256':sha(dst),'mode':'byte_exact'})
 original=(C/'scores.json').read_bytes();compact=(json.dumps(scores,separators=(',',':'),allow_nan=False)+'\n').encode()
 assert (json.dumps(json.loads(compact),indent=2,allow_nan=False)+'\n').encode()==original and len(compact)<1048576
 (D/'scores.compact.json').write_bytes(compact)
 mappings.append({'source':'scratch/0006-control/scores.json','copied':'results/0006-stage0/scores.compact.json','source_bytes':len(original),'source_sha256':sha(C/'scores.json'),'bytes':len(compact),'sha256':sha(D/'scores.compact.json'),'mode':'lossless_json_compaction','exact_original_reconstruction':"(json.dumps(json.loads(compact_bytes), indent=2, allow_nan=False) + '\\n').encode(); verified byte-identical"})
 write(D/'copy-manifest.json',{'self_excluded':True,'files':mappings})
 print(json.dumps({'published_evidence_files':len(mappings),'evidence_bytes':sum(x['bytes']for x in mappings),'numerical_status':summary['numerical_gate_status'],'provenance_status':summary['provenance_status'],'group_wall_seconds_sum':summary['sum_group_wall_seconds'],'old_results_preserved':summary['prior_results_preserved'],'new_payload_bytes':summary['new_data_payload_bytes'],'all_old_cache_and_data_payloads_excluded':True},indent=2))
if __name__=='__main__':main()
