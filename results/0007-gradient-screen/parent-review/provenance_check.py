#!/usr/bin/env python3
"""Offline public-text/Git audit only. No arrays, models, network or repo imports."""
import argparse, collections, datetime, hashlib, json, math, re, struct, subprocess, time
from pathlib import Path
from functools import lru_cache

SNAPSHOT = '5dc6839d272f48bb57ce4a2f02ccf7aa8233478c'
BASE = 'ca7fad653af31f93d770ede11a7f7ffdb5d12ec0'
PREFIX = 'results/0007-gradient-screen/'
CHECKS = 0

def check(condition, reason):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(reason)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def arrhash(values, dtype, shape, pack):
    header = json.dumps({'shape':shape, 'dtype':dtype}, sort_keys=True, separators=(',',':')).encode()
    return sha(header + b'\0' + b''.join(struct.pack(pack, v) for v in values))

def near(a,b):
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def main(root, oldpath):
    start=time.monotonic()
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root)
    def gitfiles(rev):
        return {x.split('\t',1)[1]:x.split()[2] for x in git('ls-tree','-r',rev).decode().splitlines()}
    @lru_cache(None)
    def archive_bytes(rel):return git('show',SNAPSHOT+':'+PREFIX+rel)
    def archive_text(rel):return archive_bytes(rel).decode('utf-8')
    def j(rel):return json.loads(archive_text(rel),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    def fsha(rel):return sha(archive_bytes(rel[len(PREFIX):])) if rel.startswith(PREFIX) else sha((root/rel).read_bytes())
    check(git('rev-parse','HEAD').decode().strip()==SNAPSHOT,'expected publication snapshot')
    tree=gitfiles(SNAPSHOT); basetree=gitfiles(BASE)
    inventory=read(oldpath)
    check(inventory['commit']==BASE,'inventory base')
    old=inventory['git_blobs']
    expected_old={k:v for k,v in basetree.items() if k.startswith(('results/','protocols/'))}
    check(old==expected_old and len(old)==2164,'complete 2164 prior-public inventory')
    for rel,digest in old.items():
        check(tree[rel]==digest and blob((root/rel).read_bytes())==digest,'old bytes '+rel)
    modified=[k for k in basetree if tree.get(k)!=basetree[k]]
    check(set(modified)=={'CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-18.md'},'only three old document edits')
    check((root/'notes/2026-10-18.md').read_bytes().startswith(git('show',BASE+':notes/2026-10-18.md')),'append-only note')
    archive={k for k in tree if k.startswith(PREFIX)}
    check(len(archive)==108,'108 archived public files')
    check(set(tree)-set(basetree)==archive,'no additions beyond result archive')
    working_tree_archive_changes=[]
    for rel in archive:
        p=root/rel; raw=archive_bytes(rel[len(PREFIX):]); text=raw.decode('utf-8')
        check(not p.is_symlink() and blob(raw)==tree[rel],'regular snapshot Git archive '+rel)
        if p.read_bytes()!=raw:working_tree_archive_changes.append(rel)
        check(p.suffix in {'.md','.py','.log','.json'} and b'\0' not in raw,'text-only archive '+rel)
        check(not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|/Users/|/home/oai/',text),'credential/private-path pattern '+rel)
        if p.suffix=='.json':j(rel[len(PREFIX):])
    hashlist=j('artifact-hashes.json')['files']
    check({PREFIX+x for x in hashlist}==archive-{PREFIX+'artifact-hashes.json'},'artifact hash inventory exact')
    for rel,digest in hashlist.items():check(fsha(PREFIX+rel)==digest,'artifact digest '+rel)
    manifest=j('copy-manifest.json'); entries=manifest['entries']
    copied={e['destination']:e for e in entries}
    check(len(copied)==len(entries)==100,'100 unique copy records')
    modes=collections.Counter(e['mode'] for e in entries)
    check(modes=={'byte_exact_copy':79,'path_redaction_not_original_bytes':21},'copy modes')
    for e in entries:
        check(e['destination'] in archive and fsha(e['destination'])==e['destination_sha256'],'copy destination hash '+e['destination'])
        check(not Path(e['source']).is_absolute() and '..' not in Path(e['source']).parts,'relative source')
        if e['mode']=='byte_exact_copy':check(e['source_sha256']==e['destination_sha256'],'equal declared copy digests')
        else:check(e['source_sha256']!=e['destination_sha256'],'transformation distinctly hashed')
    created=sorted(archive-set(copied))
    check(set(created)=={PREFIX+x for x in ['README.md','archive-tool-error.json','artifact-hashes.json','case-index.json','copy-manifest.json','freeze-bindings.json','host-metadata.json','pre-execution-summary.json']},'eight constructed artifacts')
    # All execution hashes bind original-source hashes, not redacted destinations.
    def original(rel):return copied[PREFIX+rel]['source_sha256']
    def verify_exec(folder,label,expected_exit):
        ex=j(folder+'/'+label+'.execution.json')
        check(ex['actual_exit']==expected_exit and ex['raw_wait_status']==expected_exit*256,'exit '+label)
        for suf,key in [('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256'),('.command.recipe.json','command_file_sha256')]:
            check(ex[key]==original(folder+'/'+label+suf),'closed original binding '+label+suf)
        t0=datetime.datetime.fromisoformat(ex['utc_started']);t1=datetime.datetime.fromisoformat(ex['utc_completed'])
        check(t1>t0 and abs((t1-t0).total_seconds()-ex['wall_seconds'])<.005,'clock '+label)
        check(ex['peak_rss_kib']>0 and ex['peak_rss_kib']<8*1024**2,'RSS bound '+label)
        return ex
    freeze=j('freeze-bindings.json'); canonical=read(root/freeze['canonical_freeze_path'])
    check(freeze['execution_main']==BASE and canonical['source_commit']==freeze['source_commit'],'source identity')
    check(fsha(freeze['canonical_freeze_path'])==freeze['canonical_freeze_sha256'],'canonical SHA')
    check(git('show',freeze['canonical_freeze_commit']+':'+freeze['canonical_freeze_path'])==(root/freeze['canonical_freeze_path']).read_bytes(),'canonical committed bytes')
    check(canonical['files']==freeze['required_files'] and len(canonical['files'])==8,'eight freeze files')
    for rel,digest in canonical['files'].items():
        for ref in [freeze['source_commit'],BASE,SNAPSHOT,'']:
            check(sha(git('show',ref+':'+rel))==digest,'source/HEAD/index '+rel)
        check(fsha(rel)==digest,'current frozen bytes '+rel)
    for a,b in [(freeze['source_commit'],freeze['canonical_freeze_commit']),(freeze['canonical_freeze_commit'],BASE),(BASE,SNAPSHOT)]:
        check(subprocess.run(['git','merge-base','--is-ancestor',a,b],cwd=root).returncode==0,'commit ancestry')
    cfg=read(root/'configs/0007-gradient-screen.json');cases=cfg['cases']
    check([c['name'] for c in cases]==['d%d-colored-%d'%(i,s) for i in range(1,5) for s in range(2)],'case order')
    prior=read(root/'results/0006-stage0/control/cache-inventory.json')['files']
    inputlist=[]
    for c in cases:
        for kind in ['input','weights','receipt']:
            rel=c[kind+'_relative']; prev=prior[rel]
            check(prev['sha256']==c[kind+'_sha256'],'published prior cache binding')
            inputlist.append(dict(relative=rel,sha256=prev['sha256'],bytes=prev['bytes']))
    begin=j('study/study.started.json');complete=j('study/study.completed.json');pre=j('pre-execution-summary.json')
    check(inputlist==freeze['input_hashes']==begin['inputs']==complete['inputs_after']==pre['cache_inputs'],'24 before/after claims and prior identity')
    check(begin['source_commit']==complete['source_commit']==freeze['source_commit'],'study source')
    check(begin['freeze_sha256']==freeze['canonical_freeze_sha256'] and begin['freeze_commit']==freeze['canonical_freeze_commit'],'study freeze')
    check(pre['frozen_files']==canonical['files'] and pre['prior_result_files']==2130 and pre['tracked_baseline_count']==2342,'preflight counts')
    scores=j('study/scores.json');summaries={(r['case_index'],q,r['method']):r for r in scores['summaries'] for q in [0 if r['regime']=='A_augmentation' else 1]}
    score_methods=['independent','random','mean','samepatch'];core_methods=['independent','random_antithetic','mean_antithetic','same_patch']
    case_summary=[]; all_record_count=0
    case_index=j('case-index.json')['cases']
    check(len(case_index)==8 and scores['bindings']['source_commit']==freeze['source_commit'] and scores['bindings']['freeze_commit']==freeze['canonical_freeze_commit'],'score source binding')
    for c in cases:
        idx=c['index'];name=c['name'];r=j('cases/'+name+'/receipt.json');ex=verify_exec('case-control',name,0)
        check(complete['cases'][idx]==dict(case=name,**ex),'case/study execution equality')
        check(ex['hard_timeout'] is False and ex['hard_seconds']==120,'case timeout')
        check(ex['receipt_sha256']==fsha(PREFIX+'cases/'+name+'/receipt.json'),'receipt bytes')
        check(scores['bindings']['cases'][idx]==dict(case=name,receipt_sha256=ex['receipt_sha256']),'score receipt binding')
        mark=j('cases/'+name+'/started.json')
        check(mark['case_index']==idx and mark['source_commit']==freeze['source_commit'] and ex['utc_started']<=mark['utc']<=ex['utc_completed'],'case started marker')
        check(j('case-control/'+name+'.stdout.log')==dict(case=name,receipt_sha256=ex['receipt_sha256'],completed=True),'case stdout binds receipt')
        check(r['case_identity']==c and r['case_index']==idx and r['status']=='completed' and r['failure'] is None,'completed case identity')
        check(r['source_sha256']==canonical['files']['scripts/antithetic_gradient_core.py'] and r['source_commit']==freeze['source_commit'],'actual core SHA unchanged')
        check(r['freeze_commit']==freeze['canonical_freeze_commit'] and r['freeze_sha256']==freeze['canonical_freeze_sha256'],'case freeze')
        check(r['weight_unchanged'] is True and r['weight_before_sha256']==r['weight_after_sha256']==c['weight_content_sha256'],'no weight update')
        check(r['optimizer_steps']==0 and r['completed_measurements']==r['expected_measurements']==512,'zero updates/512')
        check(r['parameter_count']==5313 and r['gradient_array_shape']==[2,4,64,5313] and r['batch_size']==512,'grid shape')
        check(r['cpu_affinity']==[0,1,2,3] and r['numeric_threads']==4 and r['interop_threads']==1,'four CPU cores')
        check(r['device']=='cpu' and r['torch_version']=='2.10.0+cpu' and r['numpy_version']=='2.0.2','fixed runtime')
        check(r['clean_reference_accessed'] is False and r['input_files_unchanged']==inputlist[idx*3:idx*3+3],'allowed input claims')
        recipe=j('case-control/'+name+'.command.recipe.json')
        check(recipe['argv'][2]=='case' and recipe['argv'][-2:]==['--case-index',str(idx)] and '--freeze-commit' in recipe['argv'],'case recipe')
        check(recipe['argv'][recipe['argv'].index('--freeze-commit')+1]==freeze['canonical_freeze_commit'],'recipe freeze')
        seen={};plans=r['seed_plans'];order=[]
        for q in range(2):
            plan=plans[q]
            check(plan['master_seeds']==dict(zip(['indices','augmentation','pairing'],[700000+10000*idx+1000*q+s for s in range(3)])),'master seeds')
            for k in ['indices','augmentation','pairing']:
                check(len(plan[k])==64 and all(type(v) is int and 0<=v<2**63-1 for v in plan[k]),'replicate seeds')
            if q==0:check(plan['fixed_index_seed']==plan['indices'][0],'fixed index seed')
            else:check('fixed_index_seed' not in plan,'redraw regime has no fixed index seed')
            for rep in range(64):
                rotation=(idx+rep)%4;methods=list(range(rotation,4))+list(range(rotation))
                order.append(dict(regime=q,replicate=rep,method_indices=methods))
        check(r['execution_order']==order,'balanced method order')
        for a in r['records']:
            key=(a['regime'],a['method_index'],a['replicate']);q,m,k=key
            check(key not in seen and q in range(2) and m in range(4) and k in range(64),'unique grid cell');seen[key]=a
            check(a['status']=='completed' and a['forward_calls']==a['backward_calls']==1 and a['method']==core_methods[m],'one forward/backward')
            check(a['batch_occurrences']==512 and a['gaussian_rows']==(512 if m==0 else 256),'equal work')
            check(a['index_seed']==plans[q]['indices'][0 if q==0 else k] and a['augmentation_seed']==plans[q]['augmentation'][k] and a['pairing_seed']==plans[q]['pairing'][k],'record seeds')
            check(math.isfinite(a['loss']) and a['loss']>=0 and math.isfinite(a['wall_seconds']) and a['wall_seconds']>0 and a['audit_seconds']>=0,'finite loss and clocks')
            check(all(re.fullmatch('[0-9a-f]{64}',a[k]) for k in ['gradient_sha256','base_indices_sha256','used_indices_sha256','epsilon_sha256','pair_order_sha256']),'record hashes')
        check(len(seen)==512 and len(r['records'])==512,'all unique records')
        desired=[(q,m,k) for o in order for q,k in [(o['regime'],o['replicate'])] for m in o['method_indices']]
        check([(a['regime'],a['method_index'],a['replicate']) for a in r['records']]==desired,'record execution order')
        for q in range(2):
            for k in range(64):
                rows=[seen[q,m,k] for m in range(4)]
                check(len({a['base_indices_sha256'] for a in rows})==1,'common base batch')
                check(rows[0]['used_indices_sha256']==rows[0]['base_indices_sha256'],'independent uses base batch unchanged; paired order is permuted')
                check(len({a['epsilon_sha256'] for a in rows[1:]})==1,'paired noise same seeded marginal draw')
            for m in range(4):
                wall=sum(seen[q,m,k]['wall_seconds'] for k in range(64))
                check(near(wall,summaries[idx,q,score_methods[m]]['step_time_sum_seconds']),'score wall sums')
        for key,dtype,packer,field in [('wall_seconds','<f8','<d','walls_sha256'),('loss','<f8','<d','losses_sha256')]:
            values=[seen[q,m,k][key] for q in range(2) for m in range(4) for k in range(64)]
            check(arrhash(values,dtype,[2,4,64],packer)==r[field],'reconstructed record array content '+key)
        check(arrhash([True]*512,'|b1',[2,4,64],'?')==r['completed_mask_sha256'],'completion content digest')
        check(near(sum(a['wall_seconds'] for a in r['records']),r['estimator_wall_seconds']),'estimator total')
        check(near(sum(a['audit_seconds'] for a in r['records']),r['per_measurement_audit_seconds']),'audit total')
        check(0<r['estimator_wall_seconds']+r['per_measurement_audit_seconds']+r['shared_setup_seconds']<=r['total_seconds']<=r['production_entry_total_seconds']<ex['wall_seconds']<120,'nested clocks')
        indexrow=case_index[idx]
        check(indexrow['case']==name and indexrow['receipt_sha256']==ex['receipt_sha256'] and indexrow['gradient_file_sha256']==r['files']['gradients.npy'] and indexrow['gradient_content_sha256']==r['gradient_array_sha256'],'case index evidence identity')
        for method in score_methods:
            check(indexrow['B_variance_by_method'][method]==summaries[idx,1,method]['trace_covariance'] and near(indexrow['B_step_seconds_by_method'][method],summaries[idx,1,method]['step_time_sum_seconds']),'case index scalar linkage')
        case_summary.append(dict(case=name,records=len(seen),case_wall_seconds=ex['wall_seconds'],core_seconds=r['total_seconds'],estimator_seconds=r['estimator_wall_seconds'],peak_rss_kib=ex['peak_rss_kib']))
        all_record_count+=len(seen)
    outer=[]
    for p in sorted((root/PREFIX/'outer-control').glob('*.execution.json')):
        label=p.name.removesuffix('.execution.json');ex=verify_exec('outer-control',label,2 if label=='preflight' else 0)
        check(ex['outer_source_sha256']==original('outer-control/outer.py') and ex['execution_main']==BASE,'outer source/main '+label)
        check(ex['cpu_affinity']==[0,1,2,3] and ex['address_space_bytes']==8*1024**3,'outer resources')
        outer.append(dict(label=label,exit=ex['actual_exit'],raw_wait=ex['raw_wait_status'],wall=ex['wall_seconds']))
    check(len(outer)==10,'ten retained outer wait records')
    integ=j('outer-control/integrity.json');pub=j('outer-control/publication-review.json');replay=j('outer-control/independent-replay.stdout.log')
    check(integ['total_measurements']==all_record_count==4096 and integ['optimizer_updates']==0,'4096 zero-update audit')
    check(fsha(PREFIX+'outer-control/integrity.py')==integ['checker_source_sha256'],'integrity checker SHA')
    check(fsha(PREFIX+'outer-control/publication_review.py')==pub['checker_sha256'],'publication checker SHA')
    check(fsha(PREFIX+'outer-control/independent_checker.py')==replay['checker_sha256']==freeze['independent_checker_sha256'],'independent checker SHA')
    check(replay['status']=='PASS' and replay['checks']==5351 and replay['numeric_checks']==3205 and replay['projection_checks']==576 and replay['failures']==[],'reported independent replay summary')
    check(replay['scores_sha256']==fsha(PREFIX+'study/scores.json') and replay['decision_recomputed']==scores['decision'],'replay score identity')
    for row,c in zip(replay['input_file_hashes'],cases):
        r=j('cases/'+c['name']+'/receipt.json')
        check(row['case']==c['name'] and row['receipt_sha256']==fsha(PREFIX+'cases/'+c['name']+'/receipt.json'),'replay receipt identity')
        check(row['gradients.npy_sha256']==r['files']['gradients.npy'] and row['walls.npy_sha256']==r['files']['walls.npy'],'replay raw-file declarations')
    bmean=sum(summaries[i,1,'mean']['step_time_sum_seconds'] for i in range(8))
    bind=sum(summaries[i,1,'independent']['step_time_sum_seconds'] for i in range(8))
    check(near(bmean/bind,scores['B_total_mean_vs_independent_step_time_ratio']) and bmean/bind<=1.10,'published time gate from per-record sums')
    check(j('outer-control/saved-integrity.stdout.log')==integ and j('outer-control/publication-review.stdout.log')==pub,'checker log/JSON equality')
    check(j('outer-control/frozen-score.stdout.log')['scores_sha256']==fsha(PREFIX+'study/scores.json'),'frozen score output SHA')
    check(scores['decision']=='pass_local_mechanism_screen_only' and all(scores['gates'].values()),'limited numerical pass claim')
    check(j('archive-tool-error.json')['actual_tool_exit']==1 and j('archive-tool-error.json')['scientific_operation_repeated'] is False,'archival failure retained')
    check('Ran 43 tests' in archive_text('outer-control/runtime-tests.stderr.log'),'cloud software-test count claim')
    check('\nOK\n' in archive_text('outer-control/runtime-tests.stderr.log') and 'skipped' not in archive_text('outer-control/runtime-tests.stderr.log').lower(),'cloud no-skip software-test report')
    check(pub['copy_manifest_modes']=={'byte_exact_copy':74,'path_redaction_not_original_bytes':20},'publication review earlier-stage count')
    return dict(status='PASS_PUBLIC_ARTIFACT_AUDIT_WITH_OBSERVABILITY_LIMITS',checks=CHECKS,snapshot=SNAPSHOT,
        prior_public_files_preserved=len(old),prior_results_preserved=sum(k.startswith('results/') for k in old),
        prior_protocol_files_preserved=sum(k.startswith('protocols/') for k in old),archive_files=len(archive),
        copy_modes=dict(modes),constructed_artifacts=created,frozen_files=8,cached_identity_links=24,
        measurements=all_record_count,cases=case_summary,outer_receipt_records=outer,
        archive_file_bytes=sum(len(archive_bytes(k[len(PREFIX):])) for k in archive),
        subsequent_working_tree_archive_changes=sorted(working_tree_archive_changes),
        total_estimator_seconds=sum(r['estimator_seconds'] for r in case_summary),
        total_case_wall_seconds=sum(r['case_wall_seconds'] for r in case_summary),
        study_wall_seconds=complete['wall_seconds'],source_hashes=canonical['files'],
        independent_raw_gradient_replay_performed_locally=False,
        independent_cloud_tool_statuses_observed_here=False,
        limitations=['Private source paths/command bytes and raw gradients unavailable locally; source digests are cross-bound declarations.',
            'Cloud wait4 receipts/logs are artifact evidence. Parent reports only first four preflight/source tool calls directly observed; subsequent tool records could not be retrieved due oversized turn.',
            'Private failed-cache and marker preservation is a cloud artifact assertion, not a local rehash.',
            '0006 stdout gap remains OPEN. No PSNR/training/novelty result follows from 0007.'],
        elapsed_seconds=time.monotonic()-start)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True,type=Path);ap.add_argument('--prior-inventory',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    output=main(a.repo,a.prior_inventory)
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ['status','checks','prior_public_files_preserved','archive_files','measurements','elapsed_seconds']}))
