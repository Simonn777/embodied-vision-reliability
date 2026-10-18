"""Saved-byte preservation and binding audit only; no RNG, solver or data decode."""
import collections,hashlib,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0009-stage0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
def write(name,v):
 with (HERE/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 b=load(HERE/'pre-execution.json');assert not git('status','--porcelain')
 assert git('rev-parse','HEAD').decode().strip()==b['execution_main']
 assert git('rev-parse','origin/main').decode().strip()==b['execution_main']
 tracked={n:sha(REPO/n) for n in git('ls-files','-z').decode().split('\0') if n}
 assert tracked==b['tracked'],'tracked bytes changed'
 trees={}
 for name,expected in b['historical_trees'].items():
  d=ROOT/name;actual={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file() and not p.is_symlink()}
  assert actual==expected,('historical tree changed',name);trees[name]={'count':len(actual),'all_paths_and_bytes_unchanged':True}
 for n,h in b['frozen_files'].items():
  assert sha(REPO/n)==h
  for ref in (':'+n,'HEAD:'+n,b['source_commit']+':'+n):assert hashlib.sha256(git('show',ref)).hexdigest()==h
 receipt='protocols/0009-geometric-stage0-freeze.json'
 assert sha(REPO/receipt)==b['canonical_sha256']
 assert git('show',b['freeze_commit']+':'+receipt)==git('show',':'+receipt)==(REPO/receipt).read_bytes()
 subprocess.run(['git','merge-base','--is-ancestor',b['freeze_commit'],'HEAD'],cwd=REPO,check=True)
 phases={}
 for label in ('preflight','frozen-verify','targeted-tests','checker-self-test','scientific-collection','evaluation','independent-review'):
  d=load(HERE/(label+'.execution.json'));assert d['actual_exit']==d['raw_wait_status']==0
  for suffix,key in (('.command.private.json','command_sha256'),('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256')):assert sha(HERE/(label+suffix))==d[key]
  assert sha(HERE/'capture.py')==d['capture_source_sha256']
  phases[label]={k:d[k] for k in ('actual_exit','raw_wait_status','wall_seconds','user_cpu_seconds','system_cpu_seconds','peak_rss_kib')}
 c=HERE/'collection';s=HERE/'supervision';lock=load(c/'selection-lock.json')
 for n,h in lock['files'].items():assert sha(c/n)==h,('locked file changed',n)
 actual={str(p.relative_to(c)) for p in c.rglob('*') if p.is_file()};assert actual==set(lock['files'])|{'selection-lock.json'}
 external=load(s/'execution.json');binding=load(s/'collection-binding.json')
 assert sha(s/'execution.json')==binding['execution_sha256']
 assert sha(c/'selection-lock.json')==binding['selection_lock']['sha256']
 for n,d in external['logs'].items():assert sha(s/n)==d['sha256'] and (s/n).stat().st_size==d['bytes']
 assert external['child_exit_code']==external['supervisor_exit_code']==external['raw_wait_status']==0
 assert not external['timed_out'] and binding['effective_exit_code']==0
 review=load(HERE/'review.json');evaluation=load(HERE/'evaluation/evaluation.json')
 assert review['review_status']=='PASS' and review['decision']=='NO_GO'
 assert review['selection_lock_sha256']==evaluation['selection_lock_sha256']==sha(c/'selection-lock.json')
 statuses=collections.Counter();tail=collections.Counter();reasons=collections.Counter();failures=[]
 for p in sorted((c/'cases').glob('*.json')):
  case=load(p);statuses[case['status']]+=1
  for h in case['hypotheses']:
   t=h['tail'];tail[t['status']]+=1
   if t['status']=='unsupported':reasons[t['reason']]+=1
  if case['status']!='completed':failures.append(case['case_id'])
 assert sum(statuses.values())==256 and sum(tail.values())==8192
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  hit={Path(a.decode(errors='replace')).name for a in args if a}&{'supervise_0009.py','run_geometric_stage0.py','run_antithetic_training.py','run_antithetic_gradient_screen.py'}
  if hit:active.append({'pid':int(p.name),'scripts':sorted(hit)})
 assert not active,active
 out={'status':'PASS','scope':'Saved bytes, JSON and Git bindings only; no RNG, tail DP or scientific rerun.','execution_main':b['execution_main'],'source_commit':b['source_commit'],'freeze_commit':b['freeze_commit'],'canonical_sha256':b['canonical_sha256'],'tracked_files_unchanged':len(tracked),'prior_result_protocol_files_unchanged':sum(n.startswith(('results/','protocols/')) for n in tracked),'historical_trees':trees,'frozen_files_unchanged':len(b['frozen_files']),'locked_files_verified':len(lock['files']),'case_status_counts':dict(statuses),'tail_hypothesis_status_counts':dict(tail),'unsupported_reasons':dict(reasons),'failed_case_ids':failures,'selection_complete':lock['selection_complete'],'phases':phases,'success_counts':review['success_counts'],'gates':review['gates'],'review':{k:review[k] for k in ('review_status','decision','checks','numeric_comparisons','max_absolute_difference','outer_resource_evidence_pass','case_resource_evidence_pass','reviewer_sha256')},'artifact_sha256':{n:sha(HERE/n) for n in ('pre-execution.json','collection/selection-lock.json','collection/collection.json','supervision/execution.json','supervision/collection-binding.json','evaluation/evaluation.json','review.json')},'postcheck_source_sha256':sha(__file__),'no_active_project_science':True,'new_asset_download_bytes':0,'existing_asset_accounting_gap':b['existing_asset_accounting_gap']}
 write('postcheck.json',out)
 print(json.dumps({'status':'PASS','tracked_preserved':len(tracked),'old_result_protocol_preserved':out['prior_result_protocol_files_unchanged'],'historical_regular_files_preserved':sum(v['count'] for v in trees.values()),'frozen_inputs':15,'case_status':dict(statuses),'tail_status':dict(tail),'postcheck_sha256':sha(HERE/'postcheck.json'),'decision':review['decision']},sort_keys=True))
if __name__=='__main__':main()
