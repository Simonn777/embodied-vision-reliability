"""Stage A source/history audit only; no data readers, source execution or HTTP."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
from evr.point_source_history_v1 import inspect_sources

BASE='2826fac713fd998e4edf1666017c4944f98002d3'
RECEIPTS=['protocols/0003-design-freeze-v1.json','protocols/0003-timing-addendum-freeze-v1.json',
 'protocols/0003-resource-addendum-freeze-v1.json','protocols/0003-fortran-audit-freeze-v2.json',
 'protocols/0003-rotation-diagnostic-freeze-v1.json']
SAVED=['results/0003-fortran-stageB-20261017/audit/audit.json',
 'results/0003-rotation-stageB-20261017/audit/camera-excerpt.json',
 'results/0003-rotation-stageB-20261017/audit/diagnostic.json']
h=lambda raw:hashlib.sha256(raw).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],env={**os.environ,'GIT_NO_LAZY_FETCH':'1'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('exclusive new evidence directory')
 assert git('rev-parse','HEAD').decode().strip()==BASE==git('rev-parse','origin/main').decode().strip()
 maps={};chain={}
 for path in RECEIPTS:
  raw=Path(path).read_bytes();r=json.loads(raw);assert r['inputs_sha256'] and path not in r['inputs_sha256']
  assert path not in git('ls-tree','-r','--name-only',r['input_commit']).decode().splitlines()
  for ref in ['HEAD',':','origin/main']:
   spec=ref+(''if ref==':'else':')+path;assert git('show',spec)==raw
  git('merge-base','--is-ancestor',r['input_commit'],'origin/main')
  for name,d in r['inputs_sha256'].items():
   assert not Path(name).is_absolute()and '..'not in Path(name).parts
   assert h(Path(name).read_bytes())==d
   for spec in [r['input_commit']+':'+name,'HEAD:'+name,':'+name,'origin/main:'+name]:assert h(git('show',spec))==d
   if name in maps:assert maps[name]==d
   maps[name]=d
  chain[path]={'receipt_sha256':h(raw),'input_commit':r['input_commit'],'input_count':len(r['inputs_sha256'])}
 evidence,tree=inspect_sources(a.cache)
 objects=[tree['sha']]+[e['sha']for e in tree['tree']if e['type']=='blob']
 result=subprocess.run(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input='\n'.join(objects)+'\n',text=True,capture_output=True,env={**os.environ,'GIT_NO_LAZY_FETCH':'1'},check=True)
 evidence['local_upstream_objects']={'requested_objects':len(objects),'available_records':[x for x in result.stdout.splitlines()if not x.endswith(' missing')],'missing_count':sum(x.endswith(' missing')for x in result.stdout.splitlines()),'network':'no lazy fetch; existing repository object store only','history_limit':'missing commit prevents revision history inspection; not evidence that upstream has no conversion script'}
 # Read only previously published JSON, never a data archive or metadata member.
 saved={name:json.loads(Path(name).read_text())for name in SAVED}
 old,excerpt,diag=(saved[x]for x in SAVED)
 evidence['saved_facts']={'input_sha256':{name:h(Path(name).read_bytes())for name in SAVED},'selected_scene_member':excerpt['source_member'],'selected_NPZ_sha256':excerpt['source_member_sha256'],'selected_original_point_indices':old['selected_original_point_indices'],'schema':old['schema'],'scalar_count':excerpt['scalar_count'],'proper_rotation_fail':diag['old_rotation_gate_pass']is False,'half_counts':{name:{'total':g['total'],'exact_count':g['exactly_representable_count']}for name,g in diag['statistics']['binary16_exact_representability']['groups'].items()},'single_final_f32_rounding_excluded':diag['statistics']['single_final_f32_rounding']['excluded'],'carried_returned':diag['returned_bytes_including_carried'],'carried_reserved':diag['reserves_including_carried'],'carried_directory_subset':diag['directory_reserve_including_carried'],'charged_bytes':diag['charged_bytes'],'new_numeric_data_access':0}
 evidence['verified_chains']=chain;evidence['verified_distinct_frozen_inputs']=len(maps)
 evidence['runtime']={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_source':BASE,'python':sys.version,'platform':platform.platform(),'CPU':next((x.split(':',1)[1].strip()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('model name')),'unknown'),'affinity':sorted(os.sched_getaffinity(0))}
 evidence['scope']='source text/tree and published saved JSON only; zero archive/NPZ/metadata/RGB opening, no projection/model/new numeric camera decode, zero downloads'
 a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps({'status':'completed_cached_source_audit','sources':evidence['cached_body_count'],'source_bytes':evidence['total_source_and_tree_bytes_read'],'missing_upstream_objects':evidence['local_upstream_objects']['missing_count'],'frozen_inputs':len(maps),'additional_numeric_access':0},indent=2))
if __name__=='__main__':main()
