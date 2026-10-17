"""Future failure-follow-up. Stage A must neither create its receipt nor run main."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
from evr.motion_design import unique_json
from evr.point_geometry_v2 import atomic_json
from evr.rotation_diagnostic_v1 import continued_budget,read_cameras,diagnose

RECEIPT='protocols/0003-rotation-diagnostic-freeze-v1.json'
REMOTE='refs/remotes/origin/research/0003-rotation-diagnostic-20261017'
OLD_AUDIT='results/0003-fortran-stageB-20261017/audit/audit.json'
MANDATORY=frozenset([
 'protocols/0003-rotation-diagnostic-v1.md','configs/0003-rotation-diagnostic-v1.json',
 'src/evr/rotation_diagnostic_v1.py','scripts/run_rotation_diagnostic_v1.py','tests/test_rotation_diagnostic_v1.py',
 'results/0003-rotation-stageA-20261017/source-and-rights.json',
 'protocols/0003-design-freeze-v1.json','protocols/0003-timing-addendum-freeze-v1.json',
 'protocols/0003-resource-addendum-freeze-v1.json','protocols/0003-fortran-audit-freeze-v2.json',
 OLD_AUDIT,'results/0003-fortran-stageA-20261017/source-convention.json'])
PRIOR={
 'protocols/0003-design-freeze-v1.json':'adb1b58b2fdab0e63899c67e30706f14b710a113bd6892eecdc08657b7826a7a',
 'protocols/0003-timing-addendum-freeze-v1.json':'ef05a121bf5ef0f525367ed7bcf3b4a8cd20df66c2be6bb09194fb44a5f6e2ab',
 'protocols/0003-resource-addendum-freeze-v1.json':'0e1b867fad0a3873ffb8cb6ec64335ce5c2c0dbf3ef779efd8cbe5c553452e04',
 'protocols/0003-fortran-audit-freeze-v2.json':'ad6250bb93453b0ff8f0e9ef4e25b853cac99eb37f910ea007fa389f285f2da1'}


def git(root,*args):
 p=subprocess.run(['git','-C',str(root),*args],capture_output=True)
 if p.returncode:raise ValueError('Git binding unavailable')
 return p.stdout


def regular(root,name):
 p=root/name
 if p.is_symlink()or any(q.is_symlink()for q in p.parents)or not p.resolve().is_relative_to(root.resolve())or not p.is_file():raise ValueError('regular safe input required')
 return p


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(2**20):h.update(b)
 return h.hexdigest()


def verify_current_receipt(root,path,mandatory,remote):
 # Narrow binding routine; production wrapper fixes all paths/set/scope. Synthetic
 # Git fixtures exercise this same routine with local objects and no network.
 raw=regular(root,path).read_bytes();r=unique_json(raw)
 if set(r)!={'schema','scope','input_commit','inputs_sha256'}or type(r['schema'])is not int or r['schema']!=1 or r['scope']!='FAILURE_FOLLOWUP_FRAME0_CAMERA_ONLY'or type(r['inputs_sha256'])is not dict or set(r['inputs_sha256'])!=mandatory:raise ValueError('complete canonical mandatory set')
 commit=r['input_commit']
 if not isinstance(commit,str)or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('input SHA')
 if path in git(root,'ls-tree','-r','--name-only',commit).decode().splitlines():raise ValueError('receipt self-reference')
 for ref in ['HEAD',remote]:
  if git(root,'show',ref+':'+path)!=raw:raise ValueError('receipt not committed/published')
  git(root,'merge-base','--is-ancestor',commit,ref)
 if git(root,'show',':'+path)!=raw:raise ValueError('receipt index changed')
 for name,d in r['inputs_sha256'].items():
  if not isinstance(d,str)or not re.fullmatch('[0-9a-f]{64}',d)or sha(regular(root,name))!=d:raise ValueError('current input hash')
  for spec in [commit+':'+name,'HEAD:'+name,':'+name,remote+':'+name]:
   if hashlib.sha256(git(root,'show',spec)).hexdigest()!=d:raise ValueError('input blob/index/HEAD/remote mismatch')
 return r


def verify_prior(root):
 names=set()
 for path,want in PRIOR.items():
  if sha(regular(root,path))!=want:raise ValueError('prior canonical receipt changed')
  r=unique_json((root/path).read_bytes());mapping=r['inputs_sha256']
  if not mapping:raise ValueError('empty prior map')
  for name,d in mapping.items():
   names.add(name)
   if sha(regular(root,name))!=d:raise ValueError('prior current input changed')
   for spec in [r['input_commit']+':'+name,'HEAD:'+name,':'+name,'refs/remotes/origin/main:'+name]:
    if hashlib.sha256(git(root,'show',spec)).hexdigest()!=d:raise ValueError('prior input Git binding')
 if sha(regular(root,OLD_AUDIT))!='dd580c6acfa52ebada5ffb11c75591fd601323bd4e4dd94a8db292d91f66c3a2':raise ValueError('old failure changed')
 return names


def verify_published():
 root=Path.cwd();r=verify_current_receipt(root,RECEIPT,MANDATORY,REMOTE);verify_prior(root)
 old=unique_json((root/OLD_AUDIT).read_bytes())
 if (old['returned_bytes_including_carried'],old['stream_and_directory_reserves_including_carried'],old['implicit_directory_reserved_bytes'],old['charged_bytes'])!=(20614538,2163798,1110,22778336):raise ValueError('exact prior carry binding')
 return r


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 r=verify_published();config=unique_json(Path('configs/0003-rotation-diagnostic-v1.json').read_bytes());locks=[]
 for name in ['0003-design.lock','0003-sample.lock','0003-rotation-diagnostic-v1.lock']:
  f=(a.cache.parent/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if sha(a.cache/'fortran-v2-started.json')!=config['old_marker_sha256']:raise ValueError('old one-shot marker changed')
 if a.output.exists():raise ValueError('exclusive output required')
 a.output.mkdir(parents=True,exist_ok=False)
 with(a.cache/'rotation-diagnostic-v1-started.json').open('x')as f:json.dump({'input_commit':r['input_commit'],'no_retry':True},f)
 budget=continued_budget();result={'status':'running','input_commit':r['input_commit'],'execution_commit':git(Path.cwd(),'rev-parse','HEAD').decode().strip(),'receipt_sha256':sha(RECEIPT),
  'camera_access_state':'NOT_STARTED','counts_complete':False,'interrupted_counter_semantics':'LOWER_BOUNDS_OR_UNKNOWN',
  'data_terms':config['data_excerpt_terms'],'source_member':config['member'],'NPZ_sha256':config['NPZ_sha256'],
  'scope':'only frame0 K9+RT16; distinct diagnostic, not rerun of old audit','old_rotation_gate_pass':None}
 def save():
  result.update(returned_bytes_including_carried=budget.returned_bytes,reserves_including_carried=budget.stream_reserve_bytes,
   directory_reserve_including_carried=getattr(budget,'central_directory_reserved_bytes',0),charged_bytes=budget.charged_bytes,
   new_returned_bytes=budget.returned_bytes-20614538,new_reserves=budget.stream_reserve_bytes-2163798)
  atomic_json(a.output/'diagnostic.json',result)
 save()
 try:
  npz=a.cache/'resource-audit-v1/selected-annotation.npz'
  if npz.is_symlink()or not npz.is_file()or npz.stat().st_size!=config['NPZ_bytes']or sha(npz)!=config['NPZ_sha256']:raise ValueError('cached NPZ mismatch; no download/extract fallback')
  sources=unique_json(Path('results/0003-rotation-stageA-20261017/source-and-rights.json').read_bytes())
  for source in sources['sources']:
   if sha(a.cache/source['cache_name'])!=source['sha256']:raise ValueError('cached source text identity')
  result['camera_access_state']='NUMERIC_CAMERA_ACCESS_STARTING';save()
  entries=read_cameras(npz,budget,result,save)
  excerpt={'data_terms':config['data_excerpt_terms'],'source_member':config['member'],'source_member_sha256':config['NPZ_sha256'],'frame':0,
   'unchanged_numeric_values':True,'scalar_count':25,'camera_access':entries}
  atomic_json(a.output/'camera-excerpt.json',excerpt)
  def matrix(key):return[[float.fromhex(v['float_hex'])for v in row]for row in entries[key]['values']]
  statistics=diagnose(matrix('intrinsics'),matrix('extrinsics'));result['statistics']=statistics
  result['old_rotation_gate_pass']=statistics['original_rotation_gate_pass']
  if result['old_rotation_gate_pass']:raise ValueError('unexpected original rotation pass contradicts fixed prior failure; STOP')
  result['status']='completed_diagnostic_original_rotation_gate_failed'
 except Exception as exc:result.update(status='stopped',error_type=type(exc).__name__,reason=str(exc).replace(str(Path.cwd()),'<repo>'))
 finally:
  result['counts_complete']=result['status']!='running'
  result['interrupted_counter_semantics']='COMPLETE_FOR_ATTEMPTED_WORK'if result['counts_complete']else'LOWER_BOUNDS_OR_UNKNOWN';save()
 print(json.dumps(result,indent=2,allow_nan=False))
 raise SystemExit(0 if result['status']=='completed_diagnostic_original_rotation_gate_failed'else 1)


if __name__=='__main__':main()
