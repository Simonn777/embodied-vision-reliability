"""Independent stdlib aggregate audit; no model import, labels or evaluation."""
import ast,csv,datetime,fcntl,hashlib,json,math,os,statistics,struct,subprocess
from pathlib import Path
B=Path('results/0002-calibration-completion');F=Path('protocols/0002-freeze-v2.json');NEW=['g-016','g-032','g-035','g-026'];BASE='d6eb151874fbe2afe52651f0049bd2e009afe8d7'
def load(p):return json.loads(Path(p).read_text())
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+str(path)])
def check_time(r):
 dt=(datetime.datetime.fromisoformat(r['ended_at_utc'])-datetime.datetime.fromisoformat(r['started_at_utc'])).total_seconds()
 value=r.get('elapsed_seconds',r.get('wall_seconds'));assert value>=0 and abs(dt-value)<.03
idx=load(B/'index.json');baseline=load(B/'baseline.json');f=load(F);m=load(f['manifest_path']);cal=[g['group_id']for g in m['groups']if g['partition']=='calibration'];ev=[g['group_id']for g in m['groups']if g['partition']=='evaluation']
assert idx['calibration_groups_inferred']==len(cal)==10 and idx['evaluation_groups_inferred']==0 and len(ev)==20
assert idx['evaluation_allocation_order']==ev and idx['new_groups']==NEW and cal==baseline['allocation']
assert idx['execution_main_for_new_batch']==BASE and subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
assert h(F)==idx['canonical_receipt_sha256']=='bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b'
assert F.read_bytes()==blob('origin/main',F)==blob('fbb6dd9208f6ea95e588f410c05c9ac5b58dcb1c',F)
assert len(f['input_sha256'])==29
for path,digest in f['input_sha256'].items():
 assert h(path)==digest
 for ref in [f['input_commit'],BASE,'origin/main']:assert hashlib.sha256(blob(ref,path)).hexdigest()==digest
for path,digest in baseline['historical_prior_group_files'].items():
 assert h(path)==digest and hashlib.sha256(blob(BASE,path)).hexdigest()==digest
assert len(baseline['historical_prior_group_files'])==idx['historical_file_count']==204
assert not any((Path('results/0002-pilot-predictions/v2-reconstructed')/g).exists()for g in ev)
measured=[];postrun=[];previous_end=None
for entry in idx['groups']:
 g=entry['group_id'];vid=entry['archive_index'];a=Path(entry['execution_directory']);o=Path(entry['runner_receipt_path']).parent;r=load(o/'receipt.json');outer=load(a/'outer.json');info=r['videos'][str(vid)];ids=info['point_indices']
 assert r['status']=='completed'and r['partition']=='calibration'and r['group_id']==g and r['freeze_input_commit']==f['input_commit']
 assert type(vid)is int and next(z for z in m['groups']if z['group_id']==g)['members']==[vid]
 assert len(ids)<=2 and ids==sorted(set(ids))and all(type(q)is int and 0<=q<30 for q in ids)
 path=Path(entry['csv_path']);assert h(path)==entry['csv_sha256']==info['raw_sha256'];assert h(o/'receipt.json')==entry['runner_receipt_sha256']
 with path.open(newline='')as z:
  reader=csv.DictReader(z);assert reader.fieldnames==['frame','query','point','x','y','occlusion_logit','expected_dist_logit','frame_seconds'];rows=list(reader)
 seen=set();times={};finite=nonfinite=missing=invalid=0;num=hashlib.sha256()
 for row in rows:
  t,q=int(row['frame']),int(row['query']);assert 0<=t<64 and 0<=q<len(ids)and(t,q)not in seen and int(row['point'])==ids[q];seen.add((t,q))
  values=[float(row[k])for k in['x','y','occlusion_logit','expected_dist_logit']];finite+=sum(math.isfinite(v)for v in values);nonfinite+=sum(not math.isfinite(v)for v in values);missing+=not all(math.isfinite(v)for v in values[:2]);invalid+=not all(math.isfinite(v)for v in values[2:]);num.update(struct.pack('<4f',*values))
  sec=float(row['frame_seconds']);assert math.isfinite(sec)and sec>=0
  if t in times:assert times[t]==sec
  times[t]=sec
 assert seen=={(t,q)for t in range(64)for q in range(len(ids))}and len(rows)==entry['expected_grid_rows']==64*len(ids)
 assert finite==entry['finite_values_recorded']==info['finite_values']
 lat=info['all_frame_seconds'];assert len(lat)==64 and all(math.isfinite(v)and v>=0 for v in lat)
 if ids:assert [times[t]for t in range(64)]==lat
 assert lat[0]==info['first_frame_seconds']and statistics.median(lat[1:])==info['subsequent_median_seconds']
 assert math.fsum(lat)==entry['sum_frame_intervals_once_seconds'] and math.fsum(lat)<=r['elapsed_seconds']<=outer['phases']['infer']['wall_seconds']
 assert outer['phases']['infer']['exit_code']==outer['phases']['infer']['raw_wait_status']==0
 assert r['peak_rss_kib']==entry['peak_rss_kib']==outer['phases']['infer']['peak_rss_kib']
 assert outer['source_commit']==entry['execution_source_commit']
 for path,digest in f['input_sha256'].items():assert hashlib.sha256(blob(entry['execution_source_commit'],path)).hexdigest()==digest
 if g in NEW:
  p=load(a/'preflight.json');v=load(a/'independent-verification.json');sdir=Path('results/0002-pilot-execution')/(g+'-20261017-supervision');s=load(sdir/'supervisor.json');pdir=Path('results/0002-pilot-execution')/(g+'-20261017-postrun')
  assert outer['source_commit']==outer['origin_main']==p['head']==p['origin_main']==BASE
  assert p['status']=='passed'and p['input_sha256']==f['input_sha256']and p['actual_dependency_versions']==f['dependency_versions']and len(p['actual_module_origins'])==16
  assert p['numpy_actual_version']=='2.0.2'and p['torch_actual_version']=='2.10.0+cpu'and p['torch_cuda']is None
  assert p['torch_threads']==r['torch_threads']==[2,2]and p['affinity']==r['affinity']==[0,1]
  assert p['address_space_limit_bytes']==r['limit_as_bytes']==6*1024**3 and set(p['numeric_thread_environment'].values())=={'2'}and len(p['numeric_thread_environment'])==5
  assert s['wrapper_exit_code_observed_by_parent']==s['wrapper_raw_wait_status']==0 and s['status']==outer['status']=='completed'
  assert h(sdir/'supervisor-source.py')==s['supervisor_source_sha256']and h(a/'launcher-source.py')==s['wrapper_source_sha256']==outer['wrapper_sha256']
  for name,digest in s['log_sha256'].items():assert h(sdir/name)==digest
  check_time(s);check_time(outer)
  if previous_end is not None:assert previous_end<=s['started_at_utc']
  for phase,rec in outer['phases'].items():
   assert rec['status']=='completed'and rec['exit_code']==rec['raw_wait_status']==0;check_time(rec)
   for stream in ['stdout','stderr']:assert h(a/(phase+'.'+stream+'.log'))==rec[stream+'_sha256']
  assert v['status']=='passed'and v['raw_csv_sha256']==info['raw_sha256']and v['float32_row_order_numeric_sha256']==num.hexdigest()
  assert v['finite_values']==finite and v['coordinate_missing_rows']==missing and v['logit_invalid_rows']==invalid
  for path,digest in v['artifact_sha256'].items():assert h(path)==digest
  pre=load(a/'prelaunch.json')
  for path,digest in pre['earlier_batch_artifact_sha256'].items():assert h(path)==digest
  phases=['software','verify','publication_audit']if g==NEW[0]else['verify','publication_audit']
  for n,phase in enumerate(phases):
   rec=load(pdir/(phase+'.json'));assert rec['status']=='completed'and rec['child_exit_code_observed_by_parent']==rec['raw_wait_status']==0;check_time(rec)
   source=a/('software-checks-source.py'if phase=='software'else'verification-source.py')if phase!='publication_audit'else pdir/'publication-audit-source.py'
   assert h(source)==rec['child_source_sha256']and h(pdir/'postrun-supervisor-source.py')==rec['parent_source_sha256']
   assert rec['closed_phase_source_log_bindings_verified']=={x:True for x in phases[:n+1]}
   for stream,digest in rec['log_sha256'].items():assert h(pdir/(phase+'.'+stream+'.log'))==digest
   postrun.append({'group':g,'phase':phase,'exit_code':0,'raw_wait_status':0,'receipt_sha256':h(pdir/(phase+'.json'))})
  assert json.loads((pdir/'verify.stdout.log').read_text())=={k:value for k,value in v.items()if k not in('artifact_sha256','input_identity')}
  previous_end=load(pdir/'publication_audit.json')['ended_at_utc']
 measured.append({'group_id':g,'rows':len(rows),'points':ids,'finite_values':finite,'nonfinite_values':nonfinite,'coordinate_missing_rows':missing,'logit_invalid_rows':invalid,'numeric_float32_sha256':num.hexdigest(),'csv_sha256':info['raw_sha256'],'infer_child_seconds':outer['phases']['infer']['wall_seconds'],'peak_rss_kib':r['peak_rss_kib']})
shared=Path(idx['shared_batch_software']['directory']);sw=load(shared/'software-checks.json');probe=load(shared/'pytest_availability.stdout.log');assert probe=={'pytest_available':False,'tests_executed':0}
assert sw['toy_smoke']['exit_code']==0 and idx['shared_batch_software']['toy_frames']==1240
assert load('results/0002-pilot-execution/g-016-20261017-postrun/software.stdout.log')==sw
# Byte checks of current cache, using no deserialization/model/dependency import.
p=load('results/0002-pilot-execution/g-026-20261017/preflight.json');v=load('results/0002-pilot-execution/g-026-20261017/independent-verification.json');cache=Path('scratch/boot-cache')
def cachepath(label):
 assert label.startswith('<cache>/');q=(cache/label[len('<cache>/'):]).resolve();assert q.is_relative_to(cache.resolve());return q
def streamhash(path):
 d=hashlib.sha256()
 with Path(path).open('rb')as z:
  for part in iter(lambda:z.read(1024*1024),b''):d.update(part)
 return d.hexdigest()
for package,info in p['actual_module_origins'].items():
 assert h(cachepath(info['resolved_module_path']))==info['module_entry_sha256']
 raw=(cachepath(info['dist_info_path'])/'RECORD').read_bytes();text=raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n').encode('utf-8');record=v['cache_rehash']['RECORD_digests'][package]
 assert hashlib.sha256(raw).hexdigest()==record['raw_bytes_sha256']and hashlib.sha256(text).hexdigest()==record['universal_newline_text_sha256']==info['installed_RECORD_sha256']
for name,info in p['cached_objects'].items():assert streamhash(cache/name)==info['sha256']and(cache/name).stat().st_size==info['bytes']
for path,digest in p['upstream_module_sha256'].items():assert h(cache/'source'/path)==digest
for key in['numpy_native_module','torch_native_module']:assert cachepath(p[key]).is_file()
markers={}
for g in NEW:
 marker=load(cache/('frozen-v2-'+g.replace('-','')+'-launch.json'));assert marker['status']=='completed';markers[g]=marker
for name in ['frozen-v2-research.lock']+['frozen-v2-'+g.replace('-','')+'.lock'for g in NEW]:
 with(cache/name).open('a')as z:fcntl.flock(z,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(z,fcntl.LOCK_UN)
# Publication/privacy check before the current parent's final receipt is closed.
paths=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines();assert paths and not set(paths)&set(f['input_sha256'])and str(F)not in paths
needles=[str(Path.cwd()).encode(),b'sk-'+b'proj-',b'gh'+b'p_',b'github_'+b'pat_',b'BEGIN '+b'PRIVATE KEY']
for path in paths:
 data=Path(path).read_bytes();assert len(data)<100000 and b'\0'not in data and not any(n in data for n in needles),path
 assert data==subprocess.check_output(['git','show',':'+path])
assert Path('notes/2026-10-17.md').read_bytes().startswith(blob(BASE,'notes/2026-10-17.md'));subprocess.check_call(['git','diff','--cached','--check'])
current=Path('results/0002-pilot-execution/g-026-20261017-postrun');artifact_paths=set()
for g in NEW:
 for base in[Path('results/0002-pilot-execution')/(g+'-20261017'),Path('results/0002-pilot-execution')/(g+'-20261017-supervision'),Path('results/0002-pilot-execution')/(g+'-20261017-postrun'),Path('results/0002-pilot-predictions/v2-reconstructed')/g]:artifact_paths.update(p for p in base.iterdir()if p.is_file())
artifact_paths.update(p for p in B.iterdir()if p.is_file()and p.name!='verification.json')
artifact_paths={p for p in artifact_paths if not(p.parent==current and p.name in['batch_audit.json','batch_audit.stdout.log','batch_audit.stderr.log'])}
result={'status':'passed','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'independent stdlib receipt/CSV/cache/Git/source/log audit; no candidate/model imports, annotation decode, GT errors, confidence scores, threshold fitting or replay','execution_source':BASE,'frozen_files_verified':29,'exact_dependency_versions_per_new_group':16,'historical_files_preserved':len(baseline['historical_prior_group_files']),'new_groups_serially_verified':NEW,'calibration_groups_complete':10,'evaluation_groups_complete':0,'prescribed_prediction_sets':30,'actual_csv_rows':sum(r['rows']for r in measured),'actual_finite_scalars':sum(r['finite_values']for r in measured),'all_groups':measured,'prior_closed_postrun_child_records':postrun,'shared_software_checks':1,'pytest_available':False,'work_tests_executed':0,'toy_frames':1240,'new_download_bytes':0,'acquisition_ceiling_bytes':1482283622,'markers':markers,'next_stage':'20 frozen evaluation-group inferences; not evaluate','first_evaluation_group_not_run':ev[0],'postrun_parent_own_exit_not_further_supervised':True,'cycle_boundary':'Current batch-audit parent receipt/logs and this report are excluded from artifact hashes; the final Git tree binds them after closure. No receipt self-hash or recursive supervisor.','artifact_sha256':{str(p):h(p)for p in sorted(artifact_paths)}}
with(B/'verification.json').open('x')as z:json.dump(result,z,indent=2);z.write('\n')
print(json.dumps({k:value for k,value in result.items()if k not in['artifact_sha256','all_groups','markers','prior_closed_postrun_child_records']},indent=2))
