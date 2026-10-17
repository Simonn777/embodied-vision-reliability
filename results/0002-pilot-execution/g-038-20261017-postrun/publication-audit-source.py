"""Final stdlib publication audit, observed by the post-run parent."""
import datetime,hashlib,json,subprocess
from pathlib import Path
A=Path('results/0002-pilot-execution/g-038-20261017');P=Path('results/0002-pilot-execution/g-038-20261017-postrun')
def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
v=load(A/'independent-verification.json');f=load('protocols/0002-freeze-v2.json')
assert v['status']=='passed' and v['group']=='g-038'
for path,digest in v['artifact_sha256'].items():assert h(path)==digest,path
for path,digest in v['preserved_historical_sha256'].items():assert h(path)==digest,path
for path,digest in f['input_sha256'].items():assert h(path)==digest,path
assert len(f['input_sha256'])==29
assert h('protocols/0002-freeze-v2.json')=='bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b'
for phase,source in [('software','software-checks-source.py'),('verify','verification-source.py')]:
 phase_dir=Path('results/0002-pilot-execution/g-031-20261017-postrun')if phase=='software'else P
 source_dir=Path('results/0002-pilot-execution/g-031-20261017')if phase=='software'else A
 r=load(phase_dir/(phase+'.json'))
 assert r['status']=='completed' and r['child_exit_code_observed_by_parent']==r['raw_wait_status']==0
 assert h(phase_dir/'postrun-supervisor-source.py')==r['parent_source_sha256']
 assert h(source_dir/source)==r['child_source_sha256']
 assert r['closed_phase_source_log_bindings_verified'][phase] is True
 for stream,digest in r['log_sha256'].items():assert h(phase_dir/(phase+'.'+stream+'.log'))==digest
 elapsed=(datetime.datetime.fromisoformat(r['ended_at_utc'])-datetime.datetime.fromisoformat(r['started_at_utc'])).total_seconds()
 assert abs(elapsed-r['elapsed_seconds'])<.02
 assert r['address_space_limit_bytes']==6*1024**3 and r['cpu_affinity']==[0,1]
stdout=load(P/'verify.stdout.log')
for key in ['status','raw_csv_sha256','rows','point_indices','source_commit','frozen_files_verified','preserved_prior_group_result_and_review_files']:assert stdout[key]==v[key]
paths=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
assert paths and not set(paths)&set(f['input_sha256']) and 'protocols/0002-freeze-v2.json'not in paths
needles=[str(Path.cwd()).encode(),b'sk-'+b'proj-',b'gh'+b'p_',b'github_'+b'pat_',b'BEGIN '+b'PRIVATE KEY']
for path in paths:
 b=Path(path).read_bytes();assert len(b)<1024**2 and b'\0'not in b,path
 assert not any(n in b for n in needles),path
 assert b==subprocess.check_output(['git','show',':'+path]),path
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show','HEAD:notes/2026-10-17.md']))
subprocess.check_call(['git','diff','--cached','--check'])
print(json.dumps({'status':'passed','scope':'final staged artifacts plus parent-captured software/verification exits and logs; no model or annotation/metric access','staged_files_before_this_audit_logs':len(paths),'preserved_historical_files':len(v['preserved_historical_sha256']),'frozen_files_unchanged':29,'notes_append_only':True,'postrun_parent_exit_records_verified':['software','verify'],'new_download_bytes':0,'calibration_inferred':10,'evaluation_inferred':5,'next_group_not_run':v['next_evaluation_group']},indent=2))
