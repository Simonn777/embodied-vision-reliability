"""Stdlib provenance/publication audit of the sole frozen evaluation; no decode/model."""
import ast,csv,datetime,fcntl,hashlib,json,math,os,subprocess
from pathlib import Path
B=Path('results/0002-frozen-evaluation-v2-audit-20261017');O=Path('results/0002-frozen-evaluation-v2-20261017');S=Path('results/0002-pilot-execution/evaluation-v2-20261017-supervision');P=B/'postrun';BASE='43b64a3936bc05ed05c89894768c630391492cf4'
def load(p):return json.loads(Path(p).read_text())
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def blob(ref,p):return subprocess.check_output(['git','show',ref+':'+str(p)])
def streamhash(p):
 d=hashlib.sha256()
 with Path(p).open('rb')as z:
  for part in iter(lambda:z.read(1024*1024),b''):d.update(part)
 return d.hexdigest()
def timing(r,key):
 elapsed=(datetime.datetime.fromisoformat(r['ended_at_utc'])-datetime.datetime.fromisoformat(r['started_at_utc'])).total_seconds();assert r[key]>=0 and abs(elapsed-r[key])<.03
baseline=load(B/'baseline.json');f=load('protocols/0002-freeze-v2.json');pre=load(B/'preflight.json');outer=load(B/'outer.json');sup=load(S/'supervisor.json');runner=load(O/'receipt.json');result=load(O/'evaluation.json');independent=load(B/'independent-verification.json');export=load(B/'row-export.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
assert pre['head']==pre['origin_main']==outer['source_commit']==outer['origin_main']==BASE
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines();expected={p for p in paths if p.startswith(('results/','docs/','protocols/'))}
assert set(baseline['historical_sha256'])==expected and len(expected)==988
for p,digest in baseline['historical_sha256'].items():assert h(p)==hashlib.sha256(blob(BASE,p)).hexdigest()==digest
assert len(baseline['prediction_sha256'])==60 and len(baseline['grids'])==30
index=load('results/0002-evaluation-completion/index.json');gridrows=0
for e in index['groups']:
 r=load(e['runner_receipt_path']);i=e['archive_index'];ids=r['videos'][str(i)]['point_indices'];rows=list(csv.DictReader(open(e['csv_path'])))
 assert r['mode']=='infer'and r['status']=='completed'and r['partition']==e['partition']and r['group_id']==e['group_id']and r['freeze_input_commit']==f['input_commit']
 assert len(rows)==64*len(ids)==e['expected_grid_rows']and {(int(z['frame']),int(z['query']),int(z['point']))for z in rows}=={(t,q,ids[q])for t in range(64)for q in range(len(ids))}
 for p,digest in [(e['csv_path'],e['csv_sha256']),(e['runner_receipt_path'],e['runner_receipt_sha256'])]:assert h(p)==digest==baseline['prediction_sha256'][p]
 assert load(Path(e['execution_directory'])/'outer.json')['source_commit']==e['execution_source_commit'];gridrows+=len(rows)
assert gridrows==3840
receipt=Path('protocols/0002-freeze-v2.json');assert h(receipt)=='bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b'and receipt.read_bytes()==blob(BASE,receipt)==blob('origin/main',receipt)==blob('fbb6dd9208f6ea95e588f410c05c9ac5b58dcb1c',receipt)
assert f['input_commit']=='bbe7428e126a566a62d3c123d7158f120aa66e2b'and len(f['input_sha256'])==29
for p,digest in f['input_sha256'].items():
 assert h(p)==digest
 for ref in [BASE,'origin/main',f['input_commit']]:assert hashlib.sha256(blob(ref,p)).hexdigest()==digest
 assert subprocess.check_output(['git','show',':'+p])==Path(p).read_bytes()
assert pre['input_sha256']==f['input_sha256']and pre['actual_dependency_versions']==f['dependency_versions']and len(pre['actual_module_origins'])==16
assert pre['numpy_actual_version']=='2.0.2'and pre['torch_actual_version']=='2.10.0+cpu'and pre['torch_cuda']is None and pre['torch_threads']==[2,2]
assert pre['affinity']==runner['affinity']==[0,1]and pre['address_space_limit_bytes']==runner['limit_as_bytes']==6*1024**3
assert len(pre['numeric_thread_environment'])==5 and set(pre['numeric_thread_environment'].values())=={'2'}
cache=Path('scratch/boot-cache');oldpre=load('results/0002-pilot-execution/g-024-20261017/preflight.json');oldv=load('results/0002-pilot-execution/g-024-20261017/independent-verification.json')
def cachepath(label):
 assert label.startswith('<cache>/');p=(cache/label[len('<cache>/'):]).resolve();assert p.is_relative_to(cache.resolve());return p
for package,info in pre['actual_module_origins'].items():
 assert info['module_origin_matches_distribution']and info['metadata_version']==f['dependency_versions'][package]
 assert h(cachepath(info['resolved_module_path']))==info['module_entry_sha256']==oldpre['actual_module_origins'][package]['module_entry_sha256']
 raw=(cachepath(info['dist_info_path'])/'RECORD').read_bytes();normalized=raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n').encode('utf-8')
 assert hashlib.sha256(raw).hexdigest()==info['raw_RECORD_sha256']==oldv['cache_rehash']['RECORD_digests'][package]['raw_bytes_sha256']
 assert hashlib.sha256(normalized).hexdigest()==info['installed_RECORD_sha256']==oldpre['actual_module_origins'][package]['installed_RECORD_sha256']
 assert info['RECORD_matches_historical']==(package!='numpy')
for key in ['numpy_native_module','torch_native_module']:assert cachepath(pre[key]).is_file()and pre[key]==oldpre[key]
for name,info in pre['cached_objects'].items():assert(cache/name).stat().st_size==info['bytes']and streamhash(cache/name)==info['sha256']
assert pre['cached_objects']['rgb.zip']['bytes']==187291581
for p,digest in pre['upstream_module_sha256'].items():assert h(cache/'source'/p)==digest
assert outer['status']==sup['status']==runner['status']=='completed'and runner['mode']==outer['mode']=='evaluate'
assert list(outer['phases'])==['preflight','evaluate']and runner['packages']=={'numpy':'2.0.2'}
assert sup['wrapper_exit_code_observed_by_parent']==sup['wrapper_raw_wait_status']==0
assert h(S/'supervisor-source.py')==sup['supervisor_source_sha256']and h(B/'launcher-source.py')==sup['wrapper_source_sha256']==outer['wrapper_sha256']
for name,digest in sup['log_sha256'].items():assert h(S/name)==digest
timing(outer,'wall_seconds');timing(sup,'elapsed_seconds');timing(runner,'elapsed_seconds')
for name,r in outer['phases'].items():
 assert r['status']=='completed'and r['exit_code']==r['raw_wait_status']==0;timing(r,'wall_seconds')
 for stream in ['stdout','stderr']:assert h(B/(name+'.'+stream+'.log'))==r[stream+'_sha256']
command=(B/'command.txt').read_text().strip();assert command==outer['phases']['evaluate']['command']and 'run_frozen_pilot.py evaluate --cache scratch/boot-cache --predictions results/0002-pilot-predictions/v2-reconstructed --output '+str(O)in command and '--group-id'not in command
assert runner['result_sha256']==h(O/'evaluation.json')==h(B/'independent-result.json')and runner['peak_rss_kib']==outer['phases']['evaluate']['peak_rss_kib']
module=ast.parse((B/'launcher-source.py').read_text());precode=next(ast.literal_eval(z.value)for z in module.body if isinstance(z,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='PREFLIGHT'for t in z.targets));assert precode==(B/'preflight-code.py').read_text()and 'torch.load('not in precode and 'TAPIR('not in precode
assert independent['status']==export['status']=='passed'and independent['absolute_tolerance']==1e-12 and independent['relative_tolerance']==0
assert independent['max_metric_absolute_difference']==independent['max_score_absolute_difference']==0 and independent['numeric_leaves_compared']==26566 and independent['original_ties_checked']==3840
for report in [independent,export]:
 for p,digest in report['sha256'].items():assert h(p)==digest
assert export['files']==30 and export['row_counts']=={'calibration':1280,'evaluation':2560}
group_opps={g:0 for g in result['evaluation_groups']};auditrows=0
for p in export['sha256']:
 assert Path(p).stat().st_size<1024**2
 for r in csv.DictReader(open(p)):
  auditrows+=1
  if r['partition']=='evaluation':group_opps[r['group_id']]+=int(r['opportunity'])
assert auditrows==3840 and sum(group_opps.values())==result['primary_opportunities']==2209
draws=list(csv.DictReader(open(B/'bootstrap-draws.csv')));reps=list(csv.DictReader(open(B/'bootstrap-replicates.csv')));assert len(draws)==len(reps)==2000
samples={k:[]for k in result['effect']}
for i,(draw,rep)in enumerate(zip(draws,reps)):
 assert int(draw['replicate'])==int(rep['replicate'])==i
 ids=[draw['draw_'+str(k)]for k in range(20)];assert all(g in group_opps for g in ids)
 assert sum(group_opps[g]for g in ids)==int(rep['opportunities'])
 for k in samples:
  if rep[k]!='':samples[k].append(float(rep[k]))
for k,values in samples.items():
 values.sort();interval=result['bootstrap']['intervals'][k];assert len(values)==interval['defined_replicates']==2000
 for quantile,target in zip([.025,.975],interval['ci95']):
  position=(len(values)-1)*quantile;i=math.floor(position);actual=values[i]+(values[min(i+1,len(values)-1)]-values[i])*(position-i);assert abs(actual-target)<=1e-12
phases=['export','independent','software'];source={'export':B/'export-source.py','independent':B/'independent-source.py','software':B/'software-checks-source.py'}
for i,phase in enumerate(phases):
 r=load(P/(phase+'.json'));assert r['status']=='completed'and r['child_exit_code_observed_by_parent']==r['raw_wait_status']==0;timing(r,'elapsed_seconds')
 assert r['parent_source_sha256']==h(P/'postrun-supervisor-source.py')and r['child_source_sha256']==h(source[phase])
 assert r['closed_phase_source_log_bindings_verified']=={k:True for k in phases[:i+1]}
 for stream,digest in r['log_sha256'].items():assert h(P/(phase+'.'+stream+'.log'))==digest
assert load(P/'export.stdout.log')=={k:v for k,v in export.items()if k!='sha256'}and load(P/'independent.stdout.log')==independent
sw=load(B/'software-checks.json');assert load(P/'software.stdout.log')==sw and load(B/'pytest_availability.stdout.log')=={'pytest_available':False,'tests_executed':0}and sw['toy_smoke']['exit_code']==0 and 'Completed 1240 scored frames'in(B/'toy_smoke.stdout.log').read_text()
marker=load(cache/'frozen-v2-evaluation-launch.json');assert marker['status']=='completed'and marker['output_path']==str(O)and marker['audit_path']==str(B)
for name in ['frozen-v2-research.lock','frozen-v2-evaluation.lock']:
 with(cache/name).open('a')as z:fcntl.flock(z,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(z,fcntl.LOCK_UN)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines();assert staged and not set(staged)&set(f['input_sha256'])and str(receipt)not in staged
needles=[str(Path.cwd()).encode(),b'sk-'+b'proj-',b'gh'+b'p_',b'github_'+b'pat_',b'BEGIN '+b'PRIVATE KEY']
for p in staged:
 data=Path(p).read_bytes();assert len(data)<1024**2 and b'\0'not in data and not any(k in data for k in needles),p
 assert data==subprocess.check_output(['git','show',':'+p])
assert Path('notes/2026-10-17.md').read_bytes().startswith(blob(BASE,'notes/2026-10-17.md'));subprocess.check_call(['git','diff','--cached','--check'])
artifacts={p for root in [B,O,S]for p in root.rglob('*')if p.is_file()};artifacts={p for p in artifacts if p!=B/'publication-verification.json'and not(p.parent==P and p.name.startswith('publication_audit.'))}
report={'status':'passed','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_source':BASE,'scope':'stdlib Git/CSV/cache/receipt/resource/exit/source-log/privacy audit; no model/import/decode/metric evaluator calls','historical_files_preserved':988,'prediction_files_preserved':60,'prediction_grid_rows':gridrows,'audit_csv_rows':auditrows,'frozen_files_unchanged':29,'exact_cached_dependencies':16,'original_evaluator_invocations':1,'inference_or_weight_loads':0,'evaluation_result_sha256':runner['result_sha256'],'numerical_checks':independent['numeric_leaves_compared'],'bootstrap_draw_rows_verified':2000,'shared_software_checks':1,'work_tests_executed':0,'new_download_or_install_bytes':0,'parent_postrun_own_exit_not_further_observed':True,'hash_boundary':'This verification and current publication-audit parent receipt/logs excluded from their own maps; final Git tree binds closed records, no self-hash or extra recursive supervisor.','artifact_sha256':{str(p):h(p)for p in sorted(artifacts)},'publication_text_sha256':{p:h(p)for p in staged if not p.startswith('results/')}}
with(B/'publication-verification.json').open('x')as z:json.dump(report,z,indent=2);z.write('\n')
print(json.dumps({k:v for k,v in report.items()if k not in ['artifact_sha256','publication_text_sha256']},indent=2))
