"""Read-only audit of archived 0006 evidence; no models, payload arrays or cloud access."""
import ast, datetime, hashlib, itertools, json, math, subprocess
from pathlib import Path
R=Path.cwd()
D=R/'results/0006-stage0'
EXPECTED_HEAD='dabd5b98bb1bc0b8207a7bc491504dbfdc8c378c'
def git(*args): return subprocess.check_output(['git',*args],cwd=R)
def sha(b): return hashlib.sha256(b).hexdigest()
def h(p): return sha(p.read_bytes())
def read(p): return json.loads(p.read_text())
def eq(a,b): assert a==b,(str(a)[:1500],str(b)[:1500])
def close(a,b): assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def stamp(s): return datetime.datetime.fromisoformat(s)
def tree(ref):
 out={}
 for row in git('ls-tree','-r','-z',ref).split(b'\0'):
  if row:
   meta,p=row.split(b'\t',1);mode,typ,obj=meta.split();out[p.decode()]=(mode.decode(),typ.decode(),obj.decode())
 return out
report={'reviewed_head':EXPECTED_HEAD,'scope':'Read-only published file data; no cache payload arrays, cloud or scientific execution','checks':{}}
eq(git('rev-parse','HEAD').decode().strip(),EXPECTED_HEAD)
summary=read(D/'control/summary.json');pre=read(D/'control/pre-execution.json');cfg=read(R/'configs/0006-stage0.json');freeze=read(R/'protocols/0006-stage0-freeze.json');fp='protocols/0006-stage0-freeze.json'
for ref in ['HEAD:',':',summary['freeze_commit']+':',summary['execution_main']+':']:
 eq(sha(git('show',ref+fp)),summary['freeze_sha256'])
eq(h(R/fp),summary['freeze_sha256']);eq(freeze['source_commit'],summary['source_commit']);eq(pre['required_inputs'],freeze['files'])
for p,digest in freeze['files'].items():
 eq(h(R/p),digest)
 for ref in ['HEAD:',':',freeze['source_commit']+':',summary['execution_main']+':']:
  eq(sha(git('show',ref+p)),digest)
eq(len(freeze['files']),13)
for p,digest in pre['old_freeze_receipt_hashes'].items():eq(h(R/p),digest)
report['checks']['frozen_inputs']=13;report['checks']['freeze_receipts']=len(pre['old_freeze_receipt_hashes'])
base=tree(summary['execution_main']);head=tree('HEAD');oldresults={p:v for p,v in base.items() if p.startswith('results/')}
eq(len(oldresults),1741)
for p,v in oldresults.items():
 eq(head[p],v);body=(R/p).read_bytes();eq(hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(),v[2])
changed_old=[p for p in base if head.get(p)!=base[p]]
eq(set(changed_old),{'CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-18.md'})
assert (R/'notes/2026-10-18.md').read_bytes().startswith(git('show',summary['execution_main']+':notes/2026-10-18.md'))
report['checks']['prior_results_bytes_preserved']=len(oldresults);report['checks']['prior_other_changes']=changed_old;report['checks']['note_append_only']=True
manifest=read(D/'copy-manifest.json')['files'];eq(len(manifest),380);eq(len({x['copied'] for x in manifest}),380)
for x in manifest:
 p=R/x['copied'];assert p.is_file() and not p.is_symlink();eq(p.stat().st_size,x['bytes']);eq(h(p),x['sha256'])
 if x['mode']=='lossless_json_compaction':
  original=(json.dumps(read(p),indent=2,allow_nan=False)+'\n').encode();eq(len(original),x['source_bytes']);eq(sha(original),x['source_sha256'])
 else:eq(x['mode'],'byte_exact')
files=[R/p for p in head if p.startswith('results/0006-stage0/')];report['unreviewed_concurrent_additions']=[str(p.relative_to(R)) for p in D.rglob('*') if p.is_file() and str(p.relative_to(R)) not in head];eq({str(p.relative_to(R)) for p in files},{x['copied'] for x in manifest}|{'results/0006-stage0/README.md','results/0006-stage0/copy-manifest.json'})
for p in files:
 assert p.suffix in {'.md','.json','.log','.py'} and not p.is_symlink();p.read_text();assert b'\0' not in p.read_bytes()
report['checks']['copy_manifest_entries']=len(manifest);report['checks']['archived_files']=len(files);report['checks']['no_image_array_weight_payload_files']=True
scores=read(D/'scores.compact.json');inv=read(D/'control/cache-inventory.json')['files'];post=read(D/'control/postfailure-integrity.json');ind=read(D/'control/independent-scores.json')
scored={f"{x['group']}-{x['condition']}-{x['seed']}-{x['method']}":x for x in scores['cases']}
expected={f'{g}-{c}-{s}-{m}' for g,c,s,m in itertools.product(cfg['image_ids'],cfg['conditions'],cfg['seeds'],cfg['methods'])}
for values in [scored,{p.name for p in (D/'cases').iterdir() if p.is_dir()},{p.name[:-15] for p in (D/'cases').glob('*.execution.json')},{p.name[:-13] for p in (D/'cases').glob('*.started.json')}]:eq(set(values),expected)
eq(len(expected),64)
bind={x['case']:x for x in scores['case_bindings']};runtime={x['case']:x for x in scores['runtime_measurements']};postrows={x['case']:x for x in post['case_records']};prepared={x['name']:x for x in read(D/'inputs/prepared.json')['records']};eq(len(prepared),16)
for a in [bind,runtime,postrows]:eq(set(a),expected)
logs=[];paired={};receipts={};cases=[]
for name in sorted(expected):
 p=D/'cases'/name/'receipt.json';rec=read(p);receipts[name]=rec;ex=read(D/'cases'/(name+'.execution.json'));start=read(D/'cases'/(name+'.started.json'));pr=prepared[f"{rec['image_id']}-{rec['condition']}-{rec['seed']}"]
 eq(name,f"{rec['image_id']}-{rec['condition']}-{rec['seed']}-{rec['method']}");eq(ex['case'],name);eq(ex['exit_code'],0);eq(ex['raw_wait_status'],0);eq(ex['timed_out'],False)
 eq(start['command'],ex['command']);assert stamp(start['started_utc'])<=stamp(ex['started_utc'])<stamp(ex['ended_utc']);assert abs((stamp(ex['ended_utc'])-stamp(ex['started_utc'])).total_seconds()-ex['wall_seconds'])<.02
 for opt,val in [('--image-id',rec['image_id']),('--condition',rec['condition']),('--seed',str(rec['seed'])),('--method',rec['method']),('--freeze-commit',summary['freeze_commit'])]:eq(ex['command'][ex['command'].index(opt)+1],val)
 eq(ex['receipt_sha256'],h(p));eq(bind[name]['receipt_sha256'],h(p));eq(bind[name]['execution_sha256'],h(D/'cases'/(name+'.execution.json')));eq(bind[name]['prediction_file_sha256'],rec['files']['prediction.npy']);eq(bind[name]['reference_sha256'],pr['reference_sha256']);eq(runtime[name]['receipt'],rec)
 eq(rec['input_file_sha256'],pr['input_sha256']);eq(inv[f"prepared/{pr['name']}-input.npz"]['sha256'],pr['input_sha256']);eq(inv[f"prepared/{rec['image_id']}-reference.npz"]['sha256'],pr['reference_sha256'])
 eq(postrows[name]['receipt_sha256'],h(p));eq(postrows[name]['execution_sha256'],h(D/'cases'/(name+'.execution.json')));eq(postrows[name]['prediction_file_sha256'],rec['files']['prediction.npy']);eq(postrows[name]['weight_file_sha256'],rec['files']['weights.npz']);eq(postrows[name]['input_sha256'],rec['input_file_sha256'])
 for f,digest in rec['files'].items():eq(inv[f'cases/{name}/{f}']['sha256'],digest)
 for stem in [f'cases/{name}/receipt.json',f'cases/{name}.execution.json',f'cases/{name}.started.json',f'cases/{name}.stdout.log',f'cases/{name}.stderr.log']:
  eq(inv[stem]['sha256'],h(D/stem));eq(inv[stem]['bytes'],(D/stem).stat().st_size)
 for st in ['stdout','stderr']:
  lp=D/'cases'/f'{name}.{st}.log'
  if h(lp)!=ex[st+'_sha256']:logs.append({'case':name,'stream':st,'actual_bytes':lp.stat().st_size,'actual_sha256':h(lp),'recorded_sha256':ex[st+'_sha256']})
 fit=rec['fit'];eq(rec['status'],'completed');eq(fit['status'],'completed');eq(fit['completed_steps'],500);eq(fit['requested_steps'],500);eq(fit['failure_reason'],None);eq(rec['finite_output'],True);eq(rec['clean_input_accessed'],False)
 eq(rec['torch_threads'],4);eq(rec['torch_interop_threads'],1);eq(rec['dtype'],'float32');eq(rec['device'],'cpu');eq(rec['architecture'],[49,64,32,1]);eq(fit['parameter_count'],5313);eq(rec['batch_size'],512)
 assert fit['fit_seconds']<90 and ex['wall_seconds']<120 and ex['peak_rss_kib']<8*1024**2
 eq([x['step'] for x in fit['loss_every_50']],list(range(50,501,50)));assert all(math.isfinite(x['mean_batch_loss']) for x in fit['loss_every_50'])
 pair=(rec['image_id'],rec['condition'],rec['seed']);paired.setdefault(pair,[]).append((fit['initial_weight_sha256'],fit['batch_indices_sha256']))
 if rec['method']=='r2r_alpha05':eq(rec['alpha'],.5);eq(rec['mc_samples'],50);eq(rec['inference']['primary_forward_calls'],750);eq(rec['inference']['direct_forward_calls'],15);eq(fit['augmentation_standard_normal_draws'],12544000);eq(rec['inference']['standard_normal_draws'],36465800)
 else:eq(rec['mc_samples'],1);eq(rec['inference']['primary_forward_calls'],15)
 if rec['method']=='safe_blindspot':eq(rec['safe_blindspot_zero_feature_indices'],[24] if rec['condition']=='white' else [i for i in range(49) if 1<=i//7<=5 and 1<=i%7<=5])
 cases.append(ex)
eq(len(paired),16);assert all(len(v)==4 and len(set(v))==1 for v in paired.values())
for g,s in itertools.product(cfg['image_ids'],cfg['seeds']):
 rs=[receipts[f'{g}-white-{s}-{m}'] for m in ['blindspot','safe_blindspot','residual']]
 for key in ['prediction_sha256']:eq(len({r[key] for r in rs}),1)
 eq(len({r['files']['prediction.npy'] for r in rs}),1);eq(len({r['fit']['weight_sha256'] for r in rs}),1)
eq(len(logs),1);eq(logs[0]['case'],'d3-colored-0-blindspot');eq(logs[0]['stream'],'stdout');eq(logs[0]['actual_bytes'],862)
name=logs[0]['case'];body=(D/'cases'/f'{name}.stdout.log').read_bytes();suffix=(json.dumps({'status':'completed','case':name,'receipt_sha256':h(D/'cases'/name/'receipt.json')},allow_nan=False)+'\n').encode();eq(len(suffix),144);eq(sha(body+suffix),logs[0]['recorded_sha256'])
report['checks'].update(case_grid=64,record_and_inventory_bindings=64,matched_initial_minibatch_sets=16,white_prediction_file_and_array_hash_sets_equal=8,case_stream_hashes_matching=127)
report['open_provenance_defects']=logs;report['checks']['hypothetical_suffix_144B_matches_recorded_hash_not_recovery']=True
stages=['tests','acquire','prepare','group-d1','group-d2','group-d3','group-d4','score'];stageex={}
for n in stages:
 ex=read(D/'control'/f'{n}-execution.json');stageex[n]=ex;eq(ex['exit_code'],0);eq(ex['raw_wait_status'],0);eq(ex['frozen_inputs'],freeze['files']);eq(ex['freeze_sha256'],summary['freeze_sha256']);eq(ex['observer_sha256'],h(D/'control/observe_stage.py'));eq(ex['limits'],{'affinity':[0,1,2,3],'AS_bytes':8589934592});eq(ex['command'][:3],['timeout','--kill-after=5s','1800'])
 for st in ['stdout','stderr']:eq(ex[st+'_sha256'],h(D/'control'/f'{n}.{st}.log'))
eq(stageex['tests']['tests_observed'],66);eq(stageex['tests']['skipped_lines'],0);eq(stageex['tests']['validation_passed'],True)
for prefix,src,obs,code,raw in [('grid-review','check_saved_0006.py','observe_review.py',1,256),('postfailure-review','check_postfailure_0006.py','observe_postfailure.py',0,0),('metrics-review','check_scores_0006.py','observe_scores.py',0,0)]:
 ex=read(D/'control'/f'{prefix}-execution.json');eq(ex['source_sha256'],h(D/'control'/src));eq(ex['observer_sha256'],h(D/'control'/obs));eq(ex['exit_code'],code);eq(ex['raw_wait_status'],raw)
 for st in ['stdout','stderr']:eq(ex[st+'_sha256'],h(D/'control'/f'{prefix}.{st}.log'))
eq(post['status'],'BLOCKED_LOG_PROVENANCE');eq(ind['provenance_status'],'BLOCKED_LOG_PROVENANCE');eq(summary['provenance_status'],'OPEN_STDOUT_HASH_DISCREPANCY')
report['checks']['stage_control_stream_hashes_matching']=22;report['checks']['strict_review_failure_preserved']={'exit_code':1,'raw_wait_status':256}
fit_sum=sum(r['fit']['fit_seconds'] for r in receipts.values());case_sum=sum(x['wall_seconds'] for x in cases);group_sum=sum(stageex[f'group-{g}']['wall_seconds'] for g in cfg['image_ids'])
close(fit_sum,summary['fit_seconds_sum']);close(fit_sum,post['fit_seconds_sum']);close(case_sum,post['case_wall_seconds_sum']);close(group_sum,summary['sum_group_wall_seconds']);eq(max(x['peak_rss_kib'] for x in cases),post['peak_case_rss_kib'])
report['times']={'fit_seconds_sum':fit_sum,'case_wall_seconds_sum':case_sum,'group_wall_seconds_sum':group_sum,'peak_case_rss_kib':max(x['peak_rss_kib'] for x in cases)}
for g in cfg['image_ids']:
 outer=stageex['group-'+g]
 groupcases=sorted([x for x in cases if x['case'].startswith(g+'-')],key=lambda x:x['started_utc']);eq(len(groupcases),16)
 for x in groupcases:assert stamp(outer['started_utc'])<=stamp(x['started_utc'])<stamp(x['ended_utc'])<=stamp(outer['ended_utc'])
 for a,b in zip(groupcases,groupcases[1:]):assert stamp(a['ended_utc'])<=stamp(b['started_utc'])
report['checks']['case_intervals_serial_and_inside_group']=True
acq=read(D/'inputs/acquisition.json');eq(sum(x['bytes'] for x in acq['assets']),68858);eq(acq['conservative_ceiling_bytes'],acq['historical_ceiling_bytes']+acq['new_primary_document_metadata_allowance_bytes']+acq['data_payload_bytes']);eq(acq['conservative_ceiling_bytes'],7532690072)
for x in acq['assets']:
 eq(read(D/'acquisition'/f"{x['id']}.json"),x);assert x['attempts']==1
 eq(inv['assets/'+x['source_url'].rsplit('/',1)[-1]]['sha256'],x['sha256']);eq(inv['assets/'+x['source_url'].rsplit('/',1)[-1]]['bytes'],x['bytes'])
report['checks']['acquisition_record_bytes']=68858
for field in ['mean_colored_gain_db','group_mean_colored_gain_db','gates']:eq(scores[field],summary[field])
eq(scores['status'],'stop_this_candidate');eq(scores['paper_success'],False);eq(scores['development_only'],True);eq(ind['scientific_status'],scores['status']);eq(ind['numeric_comparisons'],80*70+3+4+8)
close(scores['mean_colored_gain_db'],ind['mean_colored_gain_db']);close(scores['mean_candidate_first_difference_mse']/scores['mean_corresponding_baseline_first_difference_mse'],summary['gradient_error_ratio']);eq(ind['scores_file_sha256'],sha((json.dumps(scores,indent=2,allow_nan=False)+'\n').encode()))
imports=[]
for node in ast.walk(ast.parse((D/'control/check_scores_0006.py').read_text())):
 if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
 elif isinstance(node,ast.ImportFrom):imports.append(node.module)
assert not any(x and ('torch' in x or 'conditional_residual' in x) for x in imports)
report['checks']['independent_checker_imports']=imports
report['numerical_claim']={'decision':scores['status'],'mean_colored_gain_db':scores['mean_colored_gain_db'],'gradient_error_ratio':summary['gradient_error_ratio'],'colored_four_gates_fail':all(v is False for k,v in scores['gates'].items() if k.startswith('colored')),'paper_success':False,'limitation':'This provenance audit checks saved aggregate/receipt consistency. Parent separately recalculates block arithmetic. Full cache predictions and weights were not read or independently rehashed here.'}
report['blocking_findings_for_negative_archive']=[]
report['overall']='Archive binding checks passed with exactly one disclosed unresolved case stdout mismatch. Not an all-evidence integrity pass; no positive scientific claim follows.'
Path('/tmp/evr-0006-parent-provenance-check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
