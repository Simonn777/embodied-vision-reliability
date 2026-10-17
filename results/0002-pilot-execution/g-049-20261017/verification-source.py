"""Independent stdlib record verification; no model, archive/GT or metrics import."""
import ast,csv,datetime,hashlib,json,math,statistics,struct,subprocess
from pathlib import Path
R=Path.cwd();A=Path('results/0002-pilot-execution/g-049-20261017');S=Path('results/0002-pilot-execution/g-049-20261017-supervision');O=Path('results/0002-pilot-predictions/v2-reconstructed/g-049')
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path])
def utc_delta(a,b):return (datetime.datetime.fromisoformat(b)-datetime.datetime.fromisoformat(a)).total_seconds()
x=load(A/'outer.json');p=load(A/'preflight.json');r=load(O/'receipt.json');s=load(S/'supervisor.json');f=load('protocols/0002-freeze-v2.json')
assert x['status']==s['status']==r['status']=='completed'
assert s['wrapper_exit_code_observed_by_parent']==s['wrapper_raw_wait_status']==0
assert h(S/'supervisor-source.py')==s['supervisor_source_sha256']
assert h(A/'launcher-source.py')==s['wrapper_source_sha256']==x['wrapper_sha256']
for name,digest in s['log_sha256'].items():assert h(S/name)==digest
assert list(x['phases'])==['preflight','infer']
for name,v in x['phases'].items():
 assert v['status']=='completed' and v['exit_code']==v['raw_wait_status']==0
 assert abs(utc_delta(v['started_at_utc'],v['ended_at_utc'])-v['wall_seconds'])<.01
 for stream in ['stdout','stderr']:assert h(A/(name+'.'+stream+'.log'))==v[stream+'_sha256']
assert abs(utc_delta(x['started_at_utc'],x['ended_at_utc'])-x['wall_seconds'])<.02
assert abs(utc_delta(s['started_at_utc'],s['ended_at_utc'])-s['elapsed_seconds'])<.02
assert x['wall_seconds']<=s['elapsed_seconds']
parsed=ast.parse((A/'launcher-source.py').read_text())
code=next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREFLIGHT' for t in n.targets))
assert code==(A/'preflight-code.py').read_text()
assert p['actual_dependency_versions']==f['dependency_versions'] and len(p['actual_module_origins'])==16
assert p['numpy_actual_version']=='2.0.2' and p['torch_actual_version']=='2.10.0+cpu' and p['torch_cuda'] is None
assert all(v['module_origin_matches_distribution'] for v in p['actual_module_origins'].values())
assert p['actual_module_origins']['numpy']['RECORD_matches_historical'] is False
assert all(v['RECORD_matches_historical'] for k,v in p['actual_module_origins'].items() if k!='numpy')
assert p['torch_threads']==r['torch_threads']==[2,2]
assert set(p['numeric_thread_environment'].values())=={'2'} and len(p['numeric_thread_environment'])==5
assert p['affinity']==r['affinity']==[0,1]
assert p['address_space_limit_bytes']==r['limit_as_bytes']==6*1024**3
assert p['group']['group_id']==r['group_id']=='g-049' and p['group']['members']==[49] and r['partition']=='evaluation'
assert p['head']==p['origin_main']==x['source_commit']==x['origin_main']=='6bdce8259d196a832a52ec0ab289066ecdf4907c'
assert r['freeze_input_commit']==p['freeze_input_commit']==f['input_commit']=='bbe7428e126a566a62d3c123d7158f120aa66e2b'
assert r['freeze_receipt_sha256']==p['receipt_sha256']==h('protocols/0002-freeze-v2.json')=='bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b'
for name,v in r['actual_modules'].items():assert v['sha256']==p['upstream_module_sha256'][v['relative_path']]
assert r['checkpoint']['sha256']==p['cached_objects']['weights.pt']['sha256']==f['checkpoint_sha256']
assert p['cached_objects']['rgb.zip']['sha256']==f['archive_sha256'] and p['new_download_bytes_this_step']==0
# Rehash cache bytes without imports, unpickling, inference or label access.
cache=Path('scratch/boot-cache')
def cache_path(label):
 assert label.startswith('<cache>/')
 q=(cache/label[len('<cache>/'):]).resolve();assert q.is_relative_to(cache.resolve())
 return q
def stream_hash(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as z:
  for part in iter(lambda:z.read(1024*1024),b''):digest.update(part)
 return digest.hexdigest()
record_digests={}
for package,v in p['actual_module_origins'].items():
 assert h(cache_path(v['resolved_module_path']))==v['module_entry_sha256']
 record_path=cache_path(v['dist_info_path'])/'RECORD';raw_record=record_path.read_bytes()
 # importlib.metadata.read_text uses universal-newline text decoding. The
 # existing preflight/history digest is of that decoded/re-encoded text.
 text_record=raw_record.decode('utf-8').replace('\r\n','\n').replace('\r','\n').encode('utf-8')
 text_digest=hashlib.sha256(text_record).hexdigest()
 assert text_digest==v['installed_RECORD_sha256']
 record_digests[package]={'raw_bytes_sha256':hashlib.sha256(raw_record).hexdigest(),'universal_newline_text_sha256':text_digest,'preflight_text_sha256':v['installed_RECORD_sha256'],'CRLF_count':raw_record.count(b'\r\n')}

for filename,v in p['cached_objects'].items():
 assert (cache/filename).stat().st_size==v['bytes']
 assert stream_hash(cache/filename)==v['sha256']
for path,digest in p['upstream_module_sha256'].items():assert h(cache/'source'/path)==digest
for key in ['numpy_native_module','torch_native_module']:assert cache_path(p[key]).is_file()

assert set(r['videos'])=={'49'} and set(O.iterdir())=={O/'receipt.json',O/'video-049.csv'}
i=r['videos']['49'];ids=i['point_indices'];assert len(ids)<=2 and len(set(ids))==len(ids) and ids==sorted(ids) and all(type(q)is int and 0<=q<30 for q in ids)
assert i['frames']==64 and h(O/'video-049.csv')==i['raw_sha256']
with (O/'video-049.csv').open(newline='') as z:
 reader=csv.DictReader(z);assert reader.fieldnames==['frame','query','point','x','y','occlusion_logit','expected_dist_logit','frame_seconds'];rows=list(reader)
seen=set();times={};finite=nonfinite=coord_missing=logit_invalid=0;numeric=hashlib.sha256()
for row in rows:
 t,q=int(row['frame']),int(row['query']);assert 0<=t<64 and 0<=q<len(ids) and (t,q)not in seen
 assert int(row['point'])==ids[q];seen.add((t,q))
 vals=[float(row[k]) for k in ['x','y','occlusion_logit','expected_dist_logit']]
 finite+=sum(math.isfinite(v)for v in vals);nonfinite+=sum(not math.isfinite(v)for v in vals)
 coord_missing+=not all(math.isfinite(v)for v in vals[:2]);logit_invalid+=not all(math.isfinite(v)for v in vals[2:])
 numeric.update(struct.pack('<4f',*vals));v=float(row['frame_seconds']);assert math.isfinite(v) and v>=0
 if t in times:assert times[t]==v
 times[t]=v
assert seen=={(t,q)for t in range(64)for q in range(len(ids))} and len(rows)==64*len(ids)
lat=i['all_frame_seconds'];assert len(lat)==64 and all(math.isfinite(v)and v>=0 for v in lat)
if ids:assert [times[t]for t in range(64)]==lat
assert lat[0]==i['first_frame_seconds'] and statistics.median(lat[1:])==i['subsequent_median_seconds']
assert finite==i['finite_values'] and math.fsum(lat)<=r['elapsed_seconds']<=x['phases']['infer']['wall_seconds']
assert r['peak_rss_kib']==x['phases']['infer']['peak_rss_kib']
assert len(f['input_sha256'])==29
for path,digest in f['input_sha256'].items():
 assert h(path)==digest
 for ref in [f['input_commit'],x['source_commit'],'HEAD','origin/main']:
  assert hashlib.sha256(blob(ref,path)).hexdigest()==digest
# Compare every prior group result and review file, including documentation.
prelaunch=load(A/'prelaunch.json')
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',x['source_commit']],text=True).splitlines()
preserved={path:hashlib.sha256(blob(x['source_commit'],path)).hexdigest()for path in paths if path.startswith(('results/','docs/','protocols/'))}
for path,digest in prelaunch['historical_prior_group_files'].items():assert preserved[path]==digest
for path,digest in preserved.items():assert h(path)==digest
for prior_path,prior_digest in prelaunch['earlier_batch_artifact_sha256'].items():assert h(prior_path)==prior_digest
manifest=load(f['manifest_path']);ev=[z['group_id']for z in manifest['groups']if z['partition']=='evaluation'];cal=[z['group_id']for z in manifest['groups']if z['partition']=='calibration']
assert ev==['g-031', 'g-015', 'g-001', 'g-005', 'g-038', 'g-004', 'g-021', 'g-010', 'g-014', 'g-007', 'g-011', 'g-049', 'g-028', 'g-025', 'g-019', 'g-006', 'g-037', 'g-008', 'g-027', 'g-024'] and ev[11]=='g-049' and len(cal)==10

report={'status':'passed','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'stdlib CSV/grid/time/Git/provenance consistency; no candidate imports, annotation/archive decode, model replay, GT errors, score or threshold fitting','group':'g-049','archive_index':49,'point_indices':ids,'rows':len(rows),'output_shape':[64,len(ids),4],'finite_values':finite,'nonfinite_values':nonfinite,'coordinate_missing_rows':coord_missing,'logit_invalid_rows':logit_invalid,'raw_csv_sha256':i['raw_sha256'],'float32_row_order_numeric_sha256':numeric.hexdigest(),'first_frame_seconds':lat[0],'subsequent_median_seconds':statistics.median(lat[1:]),'sum_frame_intervals_counted_once_seconds':math.fsum(lat),'runner_wall_seconds':r['elapsed_seconds'],'infer_child_wall_seconds':x['phases']['infer']['wall_seconds'],'wrapper_wall_seconds':x['wall_seconds'],'supervisor_elapsed_seconds':s['elapsed_seconds'],'infer_child_exit_code':x['phases']['infer']['exit_code'],'parent_observed_wrapper_exit_code':s['wrapper_exit_code_observed_by_parent'],'peak_rss_kib':r['peak_rss_kib'],'source_commit':x['source_commit'],'freeze_input_commit':f['input_commit'],'frozen_files_verified':29,'dependency_versions_origins_verified':16,'cache_rehash':{'module_entry_files':16,'installed_RECORD_files':16,'upstream_source_files':3,'archive_and_checkpoint_bytes':True,'native_module_paths_present':True,'RECORD_digest_semantics':'preflight/history digest is universal-newline decoded UTF-8 text; raw-byte digests separately recorded','RECORD_digests':record_digests,'scope':'entry/RECORD/native-path and source/object checks; not hashing every installed package file'},'preserved_prior_group_result_and_review_files':len(preserved),'preserved_historical_sha256':preserved,'query_selection_scope':'indices match unchanged frozen runner receipt; frame0-only selection is established by frozen code and verified input identities, without a second annotation decode','input_identity':manifest['identities'][49],'new_download_bytes':0,'acquisition_ceiling_bytes':1482283622,'calibration_groups_inferred':10,'calibration_groups_allocated':10,'evaluation_groups_inferred':12,'evaluation_groups_allocated':20,'next_evaluation_group':ev[12],'artifact_sha256':{str(p):h(p)for base in [A,S,O]for p in base.iterdir()if p.is_file() and p.name!='independent-verification.json'}}
(A/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ['artifact_sha256','input_identity']},indent=2))
