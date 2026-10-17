"""Independent stdlib artifact verification: no model/decoder/score/GT access."""
import ast
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess

root=Path.cwd();run=Path('results/0002-pilot-execution/g-033-20261017-reconstructed-run');recovery=Path('results/0002-pilot-execution/g-033-20261017-recovery');output=Path('results/0002-pilot-predictions/v2-reconstructed/g-033')
h=lambda data:hashlib.sha256(data).hexdigest()
outer=json.loads((run/'outer.json').read_text());pre=json.loads((run/'preflight.json').read_text());receipt=json.loads((output/'receipt.json').read_text());frozen=json.loads(Path('protocols/0002-freeze-v2.json').read_text())
assert outer['status']==receipt['status']=='completed'
assert list(outer['phases'])==['preflight','infer']
for phase,record in outer['phases'].items():
 assert record['exit_code']==0 and record['raw_wait_status']==0
 assert datetime.fromisoformat(record['started_at_utc'])<datetime.fromisoformat(record['ended_at_utc'])
 for stream in ['stdout','stderr']:assert h((run/(phase+'.'+stream+'.log')).read_bytes())==record[stream+'_sha256']
assert h((run/'launcher-source.py').read_bytes())==outer['wrapper_sha256']
module=ast.parse((run/'launcher-source.py').read_text())
code=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREFLIGHT' for t in n.targets))
assert code==(run/'preflight-code.py').read_text()
assert pre['actual_dependency_versions']==frozen['dependency_versions'] and len(pre['actual_module_origins'])==16
assert all(v['module_origin_matches_distribution'] for v in pre['actual_module_origins'].values())
assert pre['numpy_actual_version']=='2.0.2' and pre['torch_actual_version']=='2.10.0+cpu' and pre['torch_cuda'] is None
assert pre['torch_threads']==receipt['torch_threads']==[2,2]
assert all(x=='2' for x in pre['numeric_thread_environment'].values()) and len(pre['affinity'])==2
assert pre['address_space_limit_bytes']==receipt['limit_as_bytes']==6*1024**3
assert pre['group']['group_id']==receipt['group_id']=='g-033' and pre['group']['members']==[33] and receipt['partition']=='calibration'
assert receipt['freeze_input_commit']==pre['freeze_input_commit']==frozen['input_commit']
assert receipt['freeze_receipt_sha256']==pre['receipt_sha256']==h(Path('protocols/0002-freeze-v2.json').read_bytes())
assert set(receipt['videos'])=={'33'} and set(output.iterdir())=={output/'receipt.json',output/'video-033.csv'}
info=receipt['videos']['33'];ids=info['point_indices'];assert ids==[0,1] and info['frames']==64
assert h((output/'video-033.csv').read_bytes())==info['raw_sha256']
rows=list(csv.DictReader((output/'video-033.csv').open()));assert len(rows)==128
seen=set();times={};finite=0;numeric=hashlib.sha256()
for r in rows:
 t,q=int(r['frame']),int(r['query']);assert 0<=t<64 and 0<=q<2 and (t,q) not in seen
 assert int(r['point'])==ids[q];seen.add((t,q))
 values=[float(r[k]) for k in ['x','y','occlusion_logit','expected_dist_logit']]
 finite+=sum(math.isfinite(v) for v in values)
 numeric.update(struct.pack('<4f',*values))
 seconds=float(r['frame_seconds']);assert math.isfinite(seconds) and seconds>=0
 if t in times:assert times[t]==seconds
 times[t]=seconds
assert seen=={(t,q) for t in range(64) for q in range(2)}
latencies=[times[t] for t in range(64)]
assert latencies==info['all_frame_seconds'] and latencies[0]==info['first_frame_seconds']
assert statistics.median(latencies[1:])==info['subsequent_median_seconds']
assert finite==info['finite_values']
assert math.fsum(latencies)<=receipt['elapsed_seconds']<=outer['phases']['infer']['wall_seconds']
assert receipt['peak_rss_kib']==outer['phases']['infer']['peak_rss_kib']
for path,digest in frozen['input_sha256'].items():
 assert h(Path(path).read_bytes())==digest
 for ref in [frozen['input_commit'],'HEAD','origin/main']:
  assert h(subprocess.check_output(['git','show',ref+':'+path]))==digest
assert len(frozen['input_sha256'])==29
selection=json.loads((recovery/'official-wheel-selection.json').read_text());download=json.loads((recovery/'download-attempt2.json').read_text());install=json.loads((recovery/'install.json').read_text())
assert download['status']=='verified' and download['downloaded_bytes']==selection['selected_wheel']['size']==19236173
assert download['actual_sha256']==selection['selected_wheel']['digests']['sha256']
wheel=Path('scratch/boot-cache/numpy-reconstruction-20261017')/selection['selected_wheel']['filename']
assert h(wheel.read_bytes())==download['actual_sha256'] and install['exit_code']==0
assert h((recovery/'install.stdout.log').read_bytes())==install['stdout_sha256']
raw=Path('scratch/g033-install-original.stderr.log').read_bytes();assert h(raw)==install['stderr_sha256']
assert raw.replace(str(root).encode(),b'<repo>')==(recovery/'install.stderr.log').read_bytes()
assert pre['actual_module_origins']['numpy']['RECORD_matches_historical'] is False
assert all(v['RECORD_matches_historical'] for k,v in pre['actual_module_origins'].items() if k!='numpy')
manifest=json.loads(Path(frozen['manifest_path']).read_text());identity=manifest['identities'][33]
result={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'independent standard-library CSV/latency/Git/provenance verification; no candidate imports, archive decoding, model rerun, GT error, scoring or threshold fitting',
 'group':'g-033','archive_index':33,'rows':len(rows),'output_shape':[64,2,4],'finite_values':finite,'raw_csv_sha256':info['raw_sha256'],'float32_numeric_row_order_sha256':numeric.hexdigest(),
 'first_frame_seconds':latencies[0],'subsequent_median_seconds':statistics.median(latencies[1:]),'sum_recorded_frame_intervals_seconds':math.fsum(latencies),
 'runner_wall_seconds':receipt['elapsed_seconds'],'outer_infer_wall_seconds':outer['phases']['infer']['wall_seconds'],'peak_rss_kib':receipt['peak_rss_kib'],
 'source_commit':outer['source_commit'],'freeze_input_commit':frozen['input_commit'],'frozen_input_hashes_verified':29,'version_and_origin_checks':16,
 'raw_model_outputs_not_repeated':True,'calibration_groups_inferred':1,'calibration_groups_allocated':10,'evaluation_groups_inferred':0,'evaluation_groups_allocated':20,
 'initial_query_identity_scope':'point indices match the frozen runner receipt; frame0 selection and normalized-query conversion rely on the unchanged frozen runner/adapter, not an independent second annotation decode',
 'input_identity_from_frozen_manifest':identity,
 'environment_identity_scope':'version-matched reconstruction; NumPy wheel/RECORD newly recorded and different from historical installed RECORD; other15 recorded RECORDs match',
 'research_acquisition_ceiling_bytes':1462956526+selection['metadata_bytes']+download['downloaded_bytes'],
 'artifact_sha256':{str(p):h(p.read_bytes()) for base in [run,output,recovery] for p in base.iterdir() if p.is_file() and p.name!='independent-verification.json'}}
(run/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','input_identity_from_frozen_manifest')},indent=2))
