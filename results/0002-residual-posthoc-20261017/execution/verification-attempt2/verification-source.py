"""Independent stdlib set/algebra/byte audit; never imports the diagnostic CLI."""
import ast
import csv
import datetime
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT=Path.cwd();B=Path('results/0002-residual-posthoc-20261017');A=B/'analysis';TOL=1e-12
BASE='83a8971ef0abbbfe1d929f4f77bf3a65c4f5bd6c'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return tuple(int(r[x])for x in ('archive_index','point','frame'))
def table(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def near(a,b):
 if a is None or b is None:assert a is b
 else:assert math.isclose(a,b,rel_tol=0,abs_tol=TOL),(a,b)
def fail(r):return not int(r['coordinate_valid'])or float(r['error_px'])>4

total_checks=0
summary=load(A/'summary.json');baseline=load(B/'baseline.json');binding=load(B/'plan-source-binding.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
assert len(baseline['historical_sha256'])==1066
for path,digest in baseline['historical_sha256'].items():
 assert sha(path)==digest
 assert hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+path])).hexdigest()==digest
 total_checks+=2
assert Path('notes/2026-10-17.md').read_bytes().startswith(subprocess.check_output(['git','show',BASE+':notes/2026-10-17.md']))
frozen=load('protocols/0002-freeze-v2.json');assert len(frozen['input_sha256'])==29
assert sha('protocols/0002-freeze-v2.json')=='bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b'
for path,digest in frozen['input_sha256'].items():
 assert sha(path)==digest
 assert hashlib.sha256(subprocess.check_output(['git','show',frozen['input_commit']+':'+path])).hexdigest()==digest
 total_checks+=2
for path,digest in summary['inputs_sha256'].items():assert sha(path)==digest;total_checks+=1
export=load('results/0002-frozen-evaluation-v2-audit-20261017/row-export.json');index=load('results/0002-evaluation-completion/index.json')
original=[];raw={};prediction_files=0
for g in index['groups']:
 for path,digest in [(g['csv_path'],g['csv_sha256']),(g['runner_receipt_path'],g['runner_receipt_sha256'])]:assert sha(path)==digest;prediction_files+=1
 for r in table(g['csv_path']):
  k=(g['archive_index'],int(r['point']),int(r['frame']));assert k not in raw;raw[k]=r
for p,d in export['sha256'].items():assert sha(p)==d;original.extend(table(p))
rows={key(r):r for r in original};assert len(rows)==len(original)==len(raw)==3840 and prediction_files==60
for g in index['groups']:
 grid={k for k in raw if k[0]==g['archive_index']}
 assert grid=={(g['archive_index'],p,t)for p in g['point_indices']for t in range(g['frames'])}
for k,r in rows.items():
 assert int(r['opportunity'])==(int(r['visible'])and int(r['post_query']))
 assert int(r['post_query'])==(k[2]>0)
 assert (int(r['failure'])==fail(r))if r['failure']!=''else r['error_px']==''
 for v in ['x','y','occlusion_logit','expected_dist_logit']:assert float(r[v])==float(raw[k][v])
 total_checks+=7
eligible={k:r for k,r in rows.items()if r['partition']=='evaluation'and int(r['opportunity'])};assert len(eligible)==2209
result=load('results/0002-frozen-evaluation-v2-20261017/evaluation.json')
selections={}
# Heap selection is separate from the full-sort implementation under audit.
for method in ['native','augmented']:
 col='score_'+method
 ranked=[(float(r[col]),r['tie'],k)for k,r in eligible.items()if int(r['available'])and math.isfinite(float(r[col]))]
 selections['matched',method]={x[2]for x in heapq.nsmallest((4*len(eligible)+4)//5,ranked)}
 boundary=result['thresholds'][method]['threshold']
 selections['operational',method]={k for k,r in eligible.items()if int(r['available'])and(float(r[col]),r['tie'])<=tuple(boundary)}
failkeys={k for k,r in eligible.items()if fail(r)};assert len(failkeys)==72
cats={};allchanged={}
for policy,expected in [('matched',(1768,4,1768,7)),('operational',(1855,8,1826,9))]:
 n=selections[policy,'native'];a=selections[policy,'augmented']
 assert(len(n),len(n&failkeys),len(a),len(a&failkeys))==expected
 cells={'both':n&a,'native_only':n-a,'augmented_only':a-n,'neither':eligible.keys()-(n|a)};cats[policy]=cells
 for name,ks in cells.items():
  assert summary['policies'][policy]['cells'][name]=={'rows':len(ks),'failures':len(ks&failkeys),'correct':len(ks-failkeys)}
 assert len(a)-len(n)==len(cells['augmented_only'])-len(cells['native_only'])
 assert len(a&failkeys)-len(n&failkeys)==len(cells['augmented_only']&failkeys)-len(cells['native_only']&failkeys)
 changed=table(A/('decision-changes-'+policy+'.csv'));allchanged[policy]=changed
 assert len(changed)==len({key(r)for r in changed})and{key(r)for r in changed}==n^a
 for r in changed:
  k=key(r);assert r[policy+'_category']==next(c for c,ks in cells.items()if k in ks)
  assert r['tie']==eligible[k]['tie']
  for method in ['native','augmented']:near(float(r[method]),float(eligible[k]['score_'+method]))
 for c in table(A/'group-point-cells.csv'):
  if c['policy']!=policy:continue
  ks={k for k in cells[c['category']]if rows[k]['group_id']==c['group']and(c['point']=='all'or k[1]==int(c['point']))}
  assert(int(c['rows']),int(c['failures']),int(c['correct']))==(len(ks),len(ks&failkeys),len(ks-failkeys))
 total_checks+=len(changed)+len(cells)
# Algebraically independent second-position difference, valid here after verifying
# every saved trajectory is consecutive and finite; no acceptance/logit/GT gate.
penalties={};residuals={};insufficient={};maxdifference=0
for video,point in sorted({(k[0],k[1])for k in rows}):
 keys=sorted(k for k in rows if k[:2]==(video,point));assert [k[2]for k in keys]==list(range(64))
 xy=[(float(raw[k]['x']),float(raw[k]['y']))for k in keys];assert all(math.isfinite(v)for p in xy for v in p)
 for i,k in enumerate(keys):
  d=0 if i<2 else math.hypot((xy[i][0]-xy[i-1][0])-(xy[i-1][0]-xy[i-2][0]),(xy[i][1]-xy[i-1][1])-(xy[i-1][1]-xy[i-2][1]))
  residuals[k]=d;penalties[k]=min(d/8,10);insufficient[k]=i<2
  delta=float(rows[k]['score_augmented'])-float(rows[k]['score_native']);near(delta,penalties[k]);maxdifference=max(maxdifference,abs(delta-penalties[k]))
  total_checks+=1
assert sum(insufficient.values())==120 and sum(p==0 for p in penalties.values())==120 and not any(p==10 for p in penalties.values())
assert summary['increment_checks']==3840
near(summary['max_increment_absolute_difference'],maxdifference)
for name in ['decision-changes-matched.csv','decision-changes-operational.csv','all-evaluation-failures.csv','correct-penalized-cases.csv','failure-and-case-context.csv']:
 for r in table(A/name):
  k=key(r);near(float(r['residual_px']),residuals[k]);near(float(r['penalty']),penalties[k]);assert bool(int(r['history_insufficient']))==insufficient[k]
  assert r['history_frames']=='|'.join(str(t)for t in range(max(0,k[2]-2),k[2]))
  for policy in ['matched','operational']:
   assert r[policy+'_category']==(next(c for c,s in cats[policy].items()if k in s)if k in eligible else'not_opportunity')
 assert len(table(A/name))==len({key(r)for r in table(A/name)})
assert{key(r)for r in table(A/'all-evaluation-failures.csv')}==failkeys
candidates=heapq.nsmallest(len(cats['matched']['native_only']-failkeys),cats['matched']['native_only']-failkeys,key=lambda k:(-penalties[k],rows[k]['tie']))
cases=[];seen=set()
for k in candidates:
 if k[:2]not in seen:cases.append(k);seen.add(k[:2])
 if len(cases)==3:break
assert[key(r)for r in table(A/'correct-penalized-cases.csv')]==cases
assert cases==[(5,0,21),(11,0,28),(31,0,18)]
expected_context={k for center in failkeys|set(cases)for k in rows if k[:2]==center[:2]and abs(k[2]-center[2])<=2}
context=table(A/'failure-and-case-context.csv');assert len(context)==115 and{key(r)for r in context}==expected_context
for r in context:
 k=key(r);centers={f"{rows[c]['group_id']}:{c[1]}:{c[2]}"for c in failkeys|set(cases)if c[:2]==k[:2]and abs(c[2]-k[2])<=2}
 assert set(r['center_ids'].split('|'))==centers
# Independently recreate all descriptive strata and inclusive stdlib quantiles.
strata={}
for k,r in rows.items():
 partition=r['partition'];labels=['raw:'+partition]
 if int(r['opportunity']):
  outcome='failure'if fail(r)else'correct';labels+=['opportunity:'+partition,'outcome:'+partition+':'+outcome,'group:'+r['group_id']+':'+outcome,'trajectory:'+r['group_id']+':'+str(k[1])+':'+outcome]
  if partition=='evaluation':labels+=['exchange:'+p+':'+next(c for c,s in cats[p].items()if k in s)+':'+outcome for p in cats]
 for label in labels:strata.setdefault(label,[]).append(k)
stored=load(A/'component-distributions.json');assert set(stored)==set(strata)
component_checks=0
for label,ks in strata.items():
 s=stored[label];assert s['rows']==len(ks)and s['history_insufficient']==sum(insufficient[k]for k in ks)and s['zero_penalty']==sum(penalties[k]==0 for k in ks)and s['capped_penalty']==0
 for component,column in [('native','score_native'),('visibility_score','score_visibility'),('local_score','score_local'),('residual_px',None),('penalty',None)]:
  v=[residuals[k]if component=='residual_px'else penalties[k]if component=='penalty'else float(rows[k][column])for k in ks]
  actual=s['components'][component];assert actual['n']==len(v)
  q=statistics.quantiles(v,n=100,method='inclusive')if len(v)>1 else v*99
  expected={'min':min(v),'q25':q[24],'median':statistics.median(v),'q75':q[74],'p90':q[89],'p95':q[94],'max':max(v),'mean':statistics.fmean(v)}
  for name,value in expected.items():near(actual[name],value);component_checks+=1
for r in table(A/'error-concentration.csv'):
 ks={k for k in eligible if rows[k]['group_id']==r['group']and(r['point']=='all'or k[1]==int(r['point']))}
 assert int(r['opportunities'])==len(ks)and int(r['all_failures'])==len(ks&failkeys)
 for method in ['native','augmented']:assert int(r[method+'_matched_failures'])==len(ks&failkeys&selections['matched',method])
assert len({rows[k]['group_id']for k in failkeys})==7 and len({k[:2]for k in failkeys})==8
assert {k[:2]for k in (cats['matched']['native_only']|cats['matched']['augmented_only'])&failkeys}=={(37,1)}
assert sum(sorted([sum(k[0]==v for k in failkeys)for v in {k[0]for k in eligible}],reverse=True)[:3])==63
assert summary['headroom']['failures_reduction_required_for_target']==36
near(summary['headroom']['maximum_risk_reduction_fraction'],4/1768)
assert summary['original_paired_ci_unchanged']==result['bootstrap']['intervals']['matched_risk_difference']
# Closed earlier phases only: the currently running audit's receipt/logs and this
# audit output are excluded from their own hash map, avoiding recursive hashes.
for phase,source in [('tests','tests/test_residual_posthoc.py'),('analysis','scripts/analyze_0002_residuals.py')]:
 p=B/'execution'/(phase+'.json');r=load(p)
 assert r['status']=='completed'and r['child_exit_code_observed_by_parent']==r['raw_wait_status']==0
 assert r['child_source_sha256']==sha(source)and r['parent_source_sha256']==sha(B/'execution/parent-source.py')
 assert r['timeout_seconds']==900 and r['address_space_limit_bytes']==6*1024**3 and len(r['affinity'])==2
 assert r['elapsed_seconds']>0 and r['peak_rss_kib']>0
 assert r['execution_main']==BASE and r['plan_sha256']==binding['plan_sha256']
 for stream,digest in r['log_sha256'].items():assert sha(B/'execution'/(phase+'.'+stream+'.log'))==digest
 assert binding['created_at_utc']<r['started_at_utc']<r['ended_at_utc']
for field,path in [('plan_sha256','docs/analysis/0002-motion-residual-posthoc-plan.md'),('analysis_source_sha256','scripts/analyze_0002_residuals.py'),('tests_source_sha256','tests/test_residual_posthoc.py'),('parent_source_sha256',B/'execution/parent-source.py'),('baseline_sha256',B/'baseline.json')]:assert binding[field]==sha(path)
assert binding['pytest_available']is False and binding['pytest_tests_executed']==0
assert 'Ran 11 tests' in(B/'execution/tests.stderr.log').read_text()and'OK'in(B/'execution/tests.stderr.log').read_text()
report=Path('docs/analysis/0002-motion-residual-posthoc.md').read_text();assert '8 of 40 point trajectories'in report and 'post-hoc'in report and 'not an irreducible error floor'in report
for path in ['scripts/analyze_0002_residuals.py','tests/test_residual_posthoc.py',str(B/'verification-source.py')]:
 tree=ast.parse(Path(path).read_text());modules=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):modules.extend(x.name.split('.')[0]for x in node.names)
  elif isinstance(node,ast.ImportFrom):modules.append(node.module.split('.')[0])
 assert set(modules)<={'argparse','ast','csv','datetime','hashlib','heapq','importlib','json','math','pathlib','statistics','subprocess','sys','tempfile','unittest'}
# Stage is explicit; all new artifacts are small text, without host-private paths.
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z']).split(b'\0');staged=[p.decode()for p in staged if p]
assert staged
for path in staged:
 data=Path(path).read_bytes();assert len(data)<1024**2 and b'\0'not in data
 data.decode('utf-8');assert (b'/' + b'workspace/')not in data and(b'/' + b'root/')not in data and(b'-----BEGIN '+b'PRIVATE KEY')not in data
closed={str(p):sha(p)for p in B.rglob('*')if p.is_file()and p.name not in ['verification.json','verification.stdout.log','verification.stderr.log']}
# execution/verification.json has the same name as output and is excluded above.
out={'status':'passed','scope':'independent heap/set/integer conservation and second-position-difference checks; no diagnostic/evaluator imports, model/archive, refit or bootstrap redraw','source_commit':BASE,'verifier_sha256':sha(__file__),'absolute_tolerance':TOL,'relative_tolerance':0,'historical_files_verified_against_current_and_main':1066,'prediction_files_preserved':prediction_files,'frozen_inputs_current_and_input_commit':29,'review_rows':len(rows),'evaluation_opportunities':len(eligible),'matched_changed_rows':len(allchanged['matched']),'operational_changed_rows':len(allchanged['operational']),'all_failure_rows':len(failkeys),'context_rows':len(context),'component_numeric_comparisons':component_checks,'other_assertion_units':total_checks,'residual_checks':len(penalties),'algebra_max_absolute_difference':maxdifference,'error_groups':7,'error_trajectories':8,'staged_text_files_inspected_before_audit_closure':len(staged),'closed_artifact_sha256':closed,'current_phase_exclusion':'current verification receipt/stdout/stderr and output excluded; actual parent wait closes execution/verification.json afterwards, later Git tree binds all bytes','parent_own_exit_scope':'no more external supervisor receipt'}
with(B/'verification.json').open('x')as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k!='closed_artifact_sha256'},indent=2))
