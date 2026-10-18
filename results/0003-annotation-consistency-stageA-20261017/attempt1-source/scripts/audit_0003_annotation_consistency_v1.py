"""Frozen prospective runner. This Stage A must not call main on real data."""
import argparse,fcntl,hashlib,json,re,subprocess
from pathlib import Path
import evr.annotation_consistency_v1 as science
import evr.point_geometry_v2 as geometry
import evr.bounded_point_audit as bounded
import evr.rotation_diagnostic_v1 as rotation
import evr.motion_design as design

RECEIPT='protocols/0003-annotation-consistency-freeze-v1.json'
REMOTE='refs/remotes/origin/research/0003-source-history-20261017'
BASE='2826fac713fd998e4edf1666017c4944f98002d3'
PRIOR={
 'protocols/0003-design-freeze-v1.json':'adb1b58b2fdab0e63899c67e30706f14b710a113bd6892eecdc08657b7826a7a',
 'protocols/0003-timing-addendum-freeze-v1.json':'ef05a121bf5ef0f525367ed7bcf3b4a8cd20df66c2be6bb09194fb44a5f6e2ab',
 'protocols/0003-resource-addendum-freeze-v1.json':'0e1b867fad0a3873ffb8cb6ec64335ce5c2c0dbf3ef779efd8cbe5c553452e04',
 'protocols/0003-fortran-audit-freeze-v2.json':'ad6250bb93453b0ff8f0e9ef4e25b853cac99eb37f910ea007fa389f285f2da1',
 'protocols/0003-rotation-diagnostic-freeze-v1.json':'957f830c6a67e6923ed82cba90d1ddf9d74558f03f93763abd82e01aad095820'}
OLD='results/0003-fortran-stageB-20261017/audit/audit.json'
EXCERPT='results/0003-rotation-stageB-20261017/audit/camera-excerpt.json'
DIAG='results/0003-rotation-stageB-20261017/audit/diagnostic.json'
MANDATORY=frozenset([
 'protocols/0003-annotation-consistency-v1.md','configs/0003-annotation-consistency-v1.json',
 'src/evr/annotation_consistency_v1.py','scripts/audit_0003_annotation_consistency_v1.py','scripts/run_0003_annotation_consistency_bounded_v1.py','tests/test_annotation_consistency_v1.py',
 'src/evr/bounded_point_audit.py','src/evr/point_geometry_v2.py','src/evr/rotation_diagnostic_v1.py','src/evr/motion_design.py',
 'src/evr/point_source_history_v1.py','scripts/audit_0003_source_history_v1.py','tests/test_point_source_history_v1.py',
 'protocols/0003-annotation-consistency-candidate-v1.md','configs/0003-annotation-consistency-candidate-v1.json',
 'results/0003-source-history-stageA-20261017/audit/audit.json','results/0003-source-history-stageA-20261017/source-audit-parent.json',
 'results/0003-source-history-stageA-20261017/source-audit.stdout.log','results/0003-source-history-stageA-20261017/source-audit.stderr.log',
 'results/0003-source-history-stageA-20261017/capture_parent.py',
 'protocols/0003-half-cell-analysis-v1.md','configs/0003-half-cell-analysis-v1.json','scripts/half_cell_certificate_v1.py','tests/test_half_cell_certificate_v1.py',
 'results/0003-half-cell-analysis-20261017/input.json','results/0003-half-cell-analysis-20261017/output.json',
 'results/0003-half-cell-analysis-20261017/data-terms.json','results/0003-half-cell-analysis-20261017/pre-execution.json',
 'results/0003-half-cell-analysis-20261017/execution.json','results/0003-half-cell-analysis-20261017/stderr.log',
 'results/0003-half-cell-analysis-20261017/verification-output.json','results/0003-half-cell-analysis-20261017/verification-execution.json',
 'results/0003-half-cell-analysis-20261017/verification.stderr.log','results/0003-half-cell-analysis-20261017/verify_saved.py',
 'results/0003-half-cell-analysis-20261017/verification_parent.py','results/0003-source-history-stageA-20261017/half_cell_parent.py',
 OLD,EXCERPT,DIAG,*PRIOR])


def git(root,*args):
 p=subprocess.run(['git','-C',str(root),*args],capture_output=True)
 if p.returncode:raise ValueError('Git binding unavailable')
 return p.stdout


def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb')as f:
  while chunk:=f.read(1024**2):digest.update(chunk)
 return digest.hexdigest()


def regular(root,name):
 if not isinstance(name,str)or not name or '\0'in name or any(p in ('','..','.')for p in name.split('/'))or Path(name).is_absolute():raise ValueError('safe relative path')
 p=root/name
 if p.is_symlink()or any(x.is_symlink()for x in p.parents)or not p.resolve().is_relative_to(root.resolve())or not p.is_file():raise ValueError('regular input')
 return p


def verify_receipt(root,path,mandatory,remote):
 raw=regular(root,path).read_bytes();r=design.unique_json(raw)
 if set(r)!={'schema','scope','input_commit','inputs_sha256'}or type(r['schema'])is not int or r['schema']!=1 or r['scope']!='DEVELOPMENT_DESCRIPTIVE_ANNOTATION_SELF_CONSISTENCY_ONLY'or type(r['inputs_sha256'])is not dict or set(r['inputs_sha256'])!=mandatory:raise ValueError('complete canonical mandatory set')
 commit=r['input_commit']
 if not isinstance(commit,str)or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('real input commit')
 if path in git(root,'ls-tree','-r','--name-only',commit).decode().splitlines():raise ValueError('receipt self-reference')
 for ref in ['HEAD',remote]:
  if git(root,'show',ref+':'+path)!=raw:raise ValueError('committed/published receipt')
  git(root,'merge-base','--is-ancestor',commit,ref)
 if git(root,'show',':'+path)!=raw:raise ValueError('receipt index')
 for name,d in r['inputs_sha256'].items():
  if not isinstance(d,str)or not re.fullmatch('[0-9a-f]{64}',d)or sha(regular(root,name))!=d:raise ValueError('current input digest')
  for spec in [commit+':'+name,'HEAD:'+name,':'+name,remote+':'+name]:
   if hashlib.sha256(git(root,'show',spec)).hexdigest()!=d:raise ValueError('input blob/index/HEAD/remote')
 return r


def verify_published():
 root=Path.cwd();r=verify_receipt(root,RECEIPT,MANDATORY,REMOTE);prior={}
 for name,d in PRIOR.items():
  if sha(regular(root,name))!=d:raise ValueError('prior receipt identity')
  for ref in ['HEAD:',':','refs/remotes/origin/main:']:
   if hashlib.sha256(git(root,'show',ref+name)).hexdigest()!=d:raise ValueError('prior receipt publication')
  old=design.unique_json((root/name).read_bytes())
  if not old['inputs_sha256']:raise ValueError('empty prior input set')
  git(root,'merge-base','--is-ancestor',old['input_commit'],'refs/remotes/origin/main')
  for p,want in old['inputs_sha256'].items():
   if sha(regular(root,p))!=want:raise ValueError('prior input current digest')
   for spec in [old['input_commit']+':'+p,'HEAD:'+p,':'+p,'refs/remotes/origin/main:'+p]:
    if hashlib.sha256(git(root,'show',spec)).hexdigest()!=want:raise ValueError('prior input Git digest')
   if p in prior and prior[p]!=want:raise ValueError('prior conflict')
   prior[p]=want
 if len(prior)!=38:raise ValueError('complete 38-input prior chain')
 for module,path in [(science,'src/evr/annotation_consistency_v1.py'),(geometry,'src/evr/point_geometry_v2.py'),(bounded,'src/evr/bounded_point_audit.py'),(rotation,'src/evr/rotation_diagnostic_v1.py'),(design,'src/evr/motion_design.py')]:
  if Path(module.__file__).resolve()!=(root/path).resolve()or sha(module.__file__)!=r['inputs_sha256'][path]:raise ValueError('actual imported source identity')
 old=design.unique_json(Path(OLD).read_bytes());diag=design.unique_json(Path(DIAG).read_bytes())
 if old['selected_original_point_indices']!=[0,1]or diag['old_rotation_gate_pass']is not False or (diag['returned_bytes_including_carried'],diag['reserves_including_carried'],diag['directory_reserve_including_carried'],diag['charged_bytes'])!=(20614916,2295240,1480,22910156):raise ValueError('allocation/failure/carry')
 return r


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 r=verify_published();c=design.unique_json(Path('configs/0003-annotation-consistency-v1.json').read_bytes())
 locks=[]
 for name in ['0003-design.lock','0003-sample.lock','0003-annotation-consistency-v1.lock']:
  f=(a.cache.parent/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 for name,want in c['prior_marker_sha256'].items():
  if sha(a.cache/name)!=want:raise ValueError('old marker changed')
 if a.output.exists():raise ValueError('exclusive new output required')
 a.output.mkdir(parents=True,exist_ok=False)
 with(a.cache/'annotation-consistency-v1-started.json').open('x')as f:json.dump({'input_commit':r['input_commit'],'no_retry':True},f)
 budget=science.continued_budget();result={'state':'STARTING','outcome':'STOPPED','counts_complete':False,'interruption_semantics':'incomplete lower bounds or unknown','checks':science.blank_checks(),'input_commit':r['input_commit'],'execution_commit':git(Path.cwd(),'rev-parse','HEAD').decode().strip(),'receipt_sha256':sha(RECEIPT),'scope':'descriptive stored annotation consistency; not rigid geometry qualification','data_terms':c['data_terms']}
 def save():
  result.update(returned_bytes=budget.returned_bytes,reserved_bytes=budget.stream_reserve_bytes,directory_reserve_subset=budget.central_directory_reserved_bytes,charged_bytes=budget.charged_bytes)
  science.summarize(result);geometry.atomic_json(a.output/'audit.json',result)
 save()
 try:
  npz=a.cache/'resource-audit-v1/selected-annotation.npz'
  if npz.is_symlink()or not npz.is_file()or npz.stat().st_size!=c['NPZ_bytes']or sha(npz)!=c['NPZ_sha256']:raise ValueError('same cached NPZ required; no fallback')
  expected=design.unique_json(Path(EXCERPT).read_bytes());diag=design.unique_json(Path(DIAG).read_bytes())['statistics']['double']
  expected_bytes={label:b''.join(bytes.fromhex(v['stored_bytes_hex'])for row in expected['camera_access'][key]['values']for v in row)for label,key in [('K','intrinsics'),('RT','extrinsics')]}
  old_gate={'tau':science.TAU,'RRt_max':diag['maxabs_RRt'],'RtR_max':diag['maxabs_RtR'],'det_minus_one_abs':diag['abs_det_minus_one'],'proper_rotation_pass':False}
  result['state']='NUMERIC_ACCESS_STARTING';save()
  data=science.read_inputs(npz,budget,result,save)
  if budget.charged_bytes!=c['prospective_total_charged_bytes']:raise ValueError('exact planned charge')
  result['state']='DESCRIPTIVE_CHECKS_STARTING';save();science.check_points(data,expected_bytes,old_gate,result,save)
 except Exception as exc:
  result.update(outcome='STOPPED',error_type=type(exc).__name__,reason=str(exc).replace(str(Path.cwd()),'<repo>').replace(str(a.cache),'<cache>'))
 finally:
  result.update(state='COMPLETE',counts_complete=True,interruption_semantics='complete for attempted work');save()
 print(json.dumps(result,indent=2,allow_nan=False));raise SystemExit(0 if result['outcome']=='COMPLETED_DESCRIPTIVE_WITH_ORIGINAL_RIGID_FAILURE'else 1)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print(json.dumps({'state':'COMPLETE','outcome':'PRE_ACCESS_OR_SETUP_STOPPED','error_type':type(exc).__name__}));raise SystemExit(1)
