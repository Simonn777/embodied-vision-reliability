"""Independent stdlib record verification; no model, archive/GT or metrics import."""
import ast,csv,datetime,hashlib,json,math,statistics,struct,subprocess
from pathlib import Path
R=Path.cwd();A=Path('results/0002-pilot-execution/g-013-20261017');S=Path('results/0002-pilot-execution/g-013-20261017-supervision');O=Path('results/0002-pilot-predictions/v2-reconstructed/g-013')
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
assert p['group']['group_id']==r['group_id']=='g-013' and p['group']['members']==[13] and r['partition']=='calibration'
assert p['head']==p['origin_main']==x['source_commit']==x['origin_main']=='5561cf5ee91ac01a9c419ca3a1c938fd43d32a4e'
assert r['freeze_input_commit']==p['freeze_input_commit']==f['input_commit']=='bbe7428e126a566a62d3c123d7158f120aa66e2b'
assert r['freeze_receipt_sha256']==p['receipt_sha256']==h('protocols/0002-freeze-v2.json')=='bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b'
for name,v in r['actual_modules'].items():assert v['sha256']==p['upstream_module_sha256'][v['relative_path']]
assert r['checkpoint']['sha256']==p['cached_objects']['weights.pt']['sha256']==f['checkpoint_sha256']
assert p['cached_objects']['rgb.zip']['sha256']==f['archive_sha256'] and p['new_download_bytes_this_step']==0
assert set(r['videos'])=={'13'} and set(O.iterdir())=={O/'receipt.json',O/'video-013.csv'}
i=r['videos']['13'];ids=i['point_indices'];assert len(ids)<=2 and len(set(ids))==len(ids) and ids==sorted(ids) and all(type(q)is int and 0<=q<30 for q in ids)
assert i['frames']==64 and h(O/'video-013.csv')==i['raw_sha256']
with (O/'video-013.csv').open(newline='') as z:
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
# Preserve every already-published g-033 result/evidence byte, not just its CSV.
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',x['source_commit'],'--','results'],text=True).splitlines()
preserved=[v for v in paths if 'g-033' in v]
for path in preserved:assert Path(path).read_bytes()==blob(x['source_commit'],path)
manifest=load(f['manifest_path']);cal=[g['group_id']for g in manifest['groups']if g['partition']=='calibration']
assert cal[:3]==['g-033','g-013','g-022']
report={'status':'passed','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'stdlib CSV/grid/time/Git/provenance consistency; no candidate imports, annotation/archive decode, model replay, GT errors, score or threshold fitting','group':'g-013','archive_index':13,'point_indices':ids,'rows':len(rows),'output_shape':[64,len(ids),4],'finite_values':finite,'nonfinite_values':nonfinite,'coordinate_missing_rows':coord_missing,'logit_invalid_rows':logit_invalid,'raw_csv_sha256':i['raw_sha256'],'float32_row_order_numeric_sha256':numeric.hexdigest(),'first_frame_seconds':lat[0],'subsequent_median_seconds':statistics.median(lat[1:]),'sum_frame_intervals_counted_once_seconds':math.fsum(lat),'runner_wall_seconds':r['elapsed_seconds'],'infer_child_wall_seconds':x['phases']['infer']['wall_seconds'],'wrapper_wall_seconds':x['wall_seconds'],'supervisor_elapsed_seconds':s['elapsed_seconds'],'infer_child_exit_code':x['phases']['infer']['exit_code'],'parent_observed_wrapper_exit_code':s['wrapper_exit_code_observed_by_parent'],'peak_rss_kib':r['peak_rss_kib'],'source_commit':x['source_commit'],'freeze_input_commit':f['input_commit'],'frozen_files_verified':29,'dependency_versions_origins_verified':16,'preserved_g033_result_files':len(preserved),'query_selection_scope':'indices match unchanged frozen runner receipt; frame0-only selection is established by frozen code and verified input identities, without a second annotation decode','input_identity':manifest['identities'][13],'new_download_bytes':0,'acquisition_ceiling_bytes':1482283622,'calibration_groups_inferred':2,'calibration_groups_allocated':10,'evaluation_groups_inferred':0,'evaluation_groups_allocated':20,'next_calibration_group':cal[2],'artifact_sha256':{str(p):h(p)for base in [A,S,O]for p in base.iterdir()if p.is_file() and p.name!='independent-verification.json'}}
(A/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ['artifact_sha256','input_identity']},indent=2))
