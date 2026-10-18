#!/usr/bin/env python3
"""Independent saved-public-byte/reaggregation review; no EVR/NumPy/RNG/DP imports."""
import argparse, collections, hashlib, io, json, math, re, shlex, subprocess, tarfile
from pathlib import Path, PurePosixPath

BASE='cd144b90d337a172a892b56d87cd794524605753'
PUBLICATION='c4c89585d4272f081319ae55e9241900611a3c52'
FREEZE='17f417efcccf85047e0935569f87f94b27705e82'
SOURCE='4b11f1de4d6fcc9bef45c8eee824ff8e7241abc0'
FAMILIES=('uniform','clustered_outlier','compact_true','sparse_isolated')
CONTROLS=('raw','msac','cell','excess','ratio'); METHODS=CONTROLS+('tail',)
PHASES=('preflight','frozen-verify','targeted-tests','checker-self-test','scientific-collection','evaluation','independent-review','postcheck')
checks=0

def check(ok,label):
    global checks
    checks+=1
    if not ok: raise ValueError(label)

def digest(b): return hashlib.sha256(b).hexdigest()
def measure(b): return {'bytes':len(b),'sha256':digest(b)}
def load(p): return json.loads(p.read_bytes())
def git(repo,*args): return subprocess.check_output(['git',*args],cwd=repo)
def tree(repo,ref):
    return {r.split(b'\t',1)[1].decode():r.split(b'\t',1)[0].decode().split()[2] for r in git(repo,'ls-tree','-r','-z',ref).split(b'\0') if r}
def blob_id(b): return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def safe(relative):
    p=PurePosixPath(relative)
    return not p.is_absolute() and '\\' not in relative and '..' not in p.parts and p.as_posix()==relative

def review(repo):
    root=repo/'results/0009-stage0'; observed=load(root/'parent-review/tool-observations.json')
    artifacts=load(root/'artifact-hashes.json'); before=tree(repo,BASE); pub=tree(repo,PUBLICATION)
    prefix='results/0009-stage0/'
    original={n[len(prefix):] for n in pub if n.startswith(prefix)}
    check(original==set(artifacts['files'])|{'artifact-hashes.json'},'original 56-file publication set')
    for name in original:
        check(safe(name) and not (root/name).is_symlink(),'safe public path')
        b=(root/name).read_bytes(); check(blob_id(b)==pub[prefix+name],'publication blob '+name)
        if name!='artifact-hashes.json': check(measure(b)==artifacts['files'][name],'artifact hash '+name)
    copies=load(root/'copy-manifest.json')['copies']; derivatives=[]
    for row in copies:
        b=(root/row['public_path']).read_bytes()
        check(measure(b)=={'bytes':row['public_bytes'],'sha256':row['public_sha256']},'copy public identity')
        if row['mode']=='byte-exact':
            check(row['original_bytes']==len(b) and row['original_sha256']==digest(b),'byte-exact declaration')
        else:
            check(row['mode']=='explicit summary derivative' and row['public_path']=='control/pre-execution.summary.json','declared derivative only')
            derivatives.append(row['public_path'])
    check(len(copies)==47 and len(derivatives)==1,'copy count')
    for name,oid in before.items(): check(pub.get(name)==oid,'prior publication blob '+name)
    old=[n for n in before if n.startswith(('results/','protocols/'))]
    for name in old: check(blob_id((repo/name).read_bytes())==before[name],'old working blob '+name)
    check(len(before)==2892 and len(old)==2672,'prior file counts')
    freeze_path='protocols/0009-geometric-stage0-freeze.json'; fb=(repo/freeze_path).read_bytes(); frozen=json.loads(fb)
    check(len(frozen['files'])==15 and frozen['source_commit']==SOURCE,'freeze schema')
    check(fb==git(repo,'show',FREEZE+':'+freeze_path)==git(repo,'show',BASE+':'+freeze_path),'canonical freeze unchanged')
    for name,expected in frozen['files'].items():
        check(digest((repo/name).read_bytes())==expected,'current freeze '+name)
        for ref in (SOURCE,FREEZE,BASE): check(digest(git(repo,'show',ref+':'+name))==expected,'git freeze '+name)
    manifest=load(root/'collection-file-manifest.json'); archive=(root/'collection.tar.gz').read_bytes()
    check(digest(archive)==manifest['archive_sha256'] and archive[4:8]==b'\0'*4,'archive identity/mtime')
    members={}; total=0
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tf:
        for member in tf:
            check(len(members)<515 and member.isfile() and safe(member.name) and member.name.startswith('collection/'),'safe bounded regular tar member')
            relative=member.name[len('collection/'):]
            check(relative not in members and relative in manifest['files'],'unique declared tar member')
            check((member.uid,member.gid,member.mtime,member.mode)==(0,0,0,0o644),'deterministic member metadata')
            check(0<=member.size<=262144,'bounded member size')
            total+=member.size; check(total<=32*1024**2,'bounded archive total')
            stream=tf.extractfile(member); b=stream.read(member.size+1)
            check(len(b)==member.size and measure(b)==manifest['files'][relative],'member hash '+relative); members[relative]=b
    check(set(members)==set(manifest['files']) and len(members)==515 and total==manifest['original_total_bytes']==28906991,'all archive members')
    lock=json.loads(members['selection-lock.json']); collection=json.loads(members['collection.json'])
    check(set(lock['files'])==set(members)-{'selection-lock.json'},'lock coverage')
    for name,sha in lock['files'].items(): check(digest(members[name])==sha,'selection lock '+name)
    binding=collection['binding']
    check(binding==load(root/'evaluation/evaluation.json')['binding'],'collection/evaluation binding')
    check(binding['files']==frozen['files'] and binding['receipt_sha256']==digest(fb) and binding['source_commit']==SOURCE and binding['freeze_commit']==FREEZE and binding['execution_head']==BASE,'source/freeze binding')
    sb=load(root/'supervision/collection-binding.json'); execution=load(root/'supervision/execution.json')
    check(sb['execution_sha256']==digest((root/'supervision/execution.json').read_bytes()) and sb['selection_lock']==measure(members['selection-lock.json']) and sb['effective_exit_code']==0,'supervisor lock binding')
    check(execution['child_exit_code']==execution['raw_wait_status']==execution['supervisor_exit_code']==0 and not execution['timed_out'],'scientific child exit')
    check(execution['cpu_affinity']==[0,1,2,3] and execution['numeric_threads']==4 and execution['address_space_bytes']==8*1024**3 and execution['wall_limit_seconds']==900 and execution['wall_seconds']<900,'science resource receipt')
    check(execution['argv']==load(root/'supervision/command.json')['argv'],'science command')
    for name,value in execution['logs'].items(): check(measure((root/'supervision'/name).read_bytes())==value,'science log '+name)
    phases=[]; check(len(observed['stages'])==8,'eight tool observations')
    for record in observed['stages']:
        check(record['status']=='completed' and record['exitCode']==0 and record['output']['truncated'] is False,'actual tool completion')
        candidates=[]
        for line in record['output']['text'].splitlines():
            try: value=json.loads(line)
            except ValueError: continue
            if isinstance(value,dict) and value.get('label') in PHASES: candidates.append(value)
        check(len(candidates)==1,'one captured phase receipt per tool'); row=candidates[0]; phase=row['label']; phases.append(phase)
        check(row==load(root/f'control/{phase}.execution.json'),'observed receipt byte-value match '+phase)
        command=load(root/f'control/{phase}.command.private.json')
        check(row['command_sha256']==digest((root/f'control/{phase}.command.private.json').read_bytes()),'actual command hash')
        shell=shlex.split(record['command']); check(shell[:2]==['/bin/bash','-lc'] and len(shell)==3,'observed shell shape')
        calls=[shlex.split(line) for line in shell[2].splitlines() if '/capture.py ' in line]; check(len(calls)==1,'observed capture invocation')
        words=calls[0]; pos=next(i for i,w in enumerate(words) if w.endswith('/capture.py'))
        expected_argv=words[pos+3:] if words[pos+2]=='frozen900' else ['timeout','--kill-after=5s',words[pos+2]]+words[pos+3:]
        check(words[pos+1]==phase and command['argv']==expected_argv,'observed command vs archived argv')
        check(row['actual_exit']==row['raw_wait_status']==0 and row['cpu_affinity']==[0,1,2,3] and row['numeric_threads']==4 and row['address_space_bytes']==8*1024**3,'phase exit/resources')
        check(record['durationMs']/1000>=row['wall_seconds'],'distinct tool/wait4 durations')
        for stream in ('stdout','stderr'): check(row[stream+'_sha256']==digest((root/f'control/{phase}.{stream}.log').read_bytes()),'phase stream hash')
        check(row['capture_source_sha256']==digest((root/'control/capture.py').read_bytes()),'capture source')
    check(tuple(phases)==PHASES,'ordered eight phases')
    evaluation=load(root/'evaluation/evaluation.json'); expected={f'{f}-{s:02d}' for f in FAMILIES for s in range(64)}
    check(set(collection['case_order'])==expected and len(collection['case_order'])==256,'complete collection IDs')
    check(json.loads(members['planned-cases.json'])['case_ids']==collection['case_order'],'planned case order')
    check([r['case_id'] for r in evaluation['cases']]==collection['case_order'],'evaluation order')
    counts={f:{m:0 for m in METHODS} for f in FAMILIES}; pairs={f:{m:dict(baseline_success_tail_failure=0,tail_success_baseline_failure=0,both_success=0,both_failure=0) for m in CONTROLS} for f in FAMILIES}
    case_status=collections.Counter(); tail_status=collections.Counter(); unsupported=[]; raw_harms=[]; incorrect=[]
    for outcome in evaluation['cases']:
        case_id=outcome['case_id']; c=json.loads(members[f'cases/{case_id}.json']); labels=json.loads(members[f'labels/{case_id}.json']); family=c['family']; case_status[c['status']]+=1
        check(c['case_id']==case_id and family==outcome['family'] and c['status']==outcome['status'] and c['controls_complete'] is True,'case identity/status')
        check(labels['H_star']==[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]] and labels['true_candidate_id']==4,'saved identity ground truth')
        check(len(c['hypotheses'])==32 and sorted(c['inputs']['candidate_order'])==list(range(32)),'full shared H pool')
        check([h['canonical_id'] for h in c['hypotheses']]==c['inputs']['candidate_order'],'recorded candidate order')
        statuses=[h['tail']['status'] for h in c['hypotheses']]; tail_status.update(statuses)
        check(c['tail_attempted']==sum(x!='not_attempted' for x in statuses),'attempt counts')
        if c['status']=='tail_failed':
            check(statuses.count('unsupported')==1 and c['selections']['tail']=={'selected_id':None,'tied_ids':[]},'failed tail selection')
            index=statuses.index('unsupported'); h=c['hypotheses'][index]
            check(statuses[:index]==['completed']*index and statuses[index+1:]==['not_attempted']*(31-index),'stop after first unsupported')
            check(c['reason']==h['tail']['reason'] and c['reason'].startswith('non-complete component has smaller side > 12:'),'declared unsupported reason')
            unsupported.append({'case_id':case_id,'first_unsupported_hypothesis_record_index':index,'canonical_hypothesis_id':h['canonical_id'],'H':c['inputs']['candidates'][h['canonical_id']],'H_sha256':h['H_sha256'],'reason':h['tail']['reason'],'operations':h['tail']['operations'],'graph_components':h['graph_components'],'tail_attempted':c['tail_attempted']})
        else: check(c['status']=='completed' and statuses==['completed']*32,'complete tail case')
        flags={}
        for method in METHODS:
            selected=c['selections'][method]['selected_id']; result=outcome['methods'][method]
            check(selected==result['selected_id'],'saved selected ID')
            if selected is None:
                check(method=='tail' and result=={'offset':None,'selected_id':None,'success':False,'transfer_error':None},'failure outcome'); success=False
            else:
                H=c['inputs']['candidates'][selected]; check(H[0][:2]==[1.,0.] and H[1][:2]==[0.,1.] and H[2]==[0.,0.,1.],'translation-only matrix')
                offset=[H[0][2],H[1][2]]; error=math.hypot(*offset); success=error<=3
                check(result['offset']==offset and math.isclose(result['transfer_error'],error,rel_tol=1e-12,abs_tol=1e-10) and result['success']==success,'translation outcome')
            flags[method]=success; counts[family][method]+=success
        if flags['raw'] and not flags['tail']: raw_harms.append(case_id)
        if c['status']=='completed' and not flags['tail']: incorrect.append(case_id)
        for method in CONTROLS:
            key='both_success' if flags[method] and flags['tail'] else 'both_failure' if not flags[method] and not flags['tail'] else 'baseline_success_tail_failure' if flags[method] else 'tail_success_baseline_failure'
            pairs[family][method][key]+=1
    check(counts==evaluation['success_counts'] and pairs==evaluation['paired'],'independent counts and paired table')
    best=max(CONTROLS,key=lambda m:counts['clustered_outlier'][m]); gain=counts['clustered_outlier']['tail']-counts['clustered_outlier'][best]
    harms={f:{m:pairs[f][m]['baseline_success_tail_failure'] for m in CONTROLS} for f in ('uniform','compact_true','sparse_isolated')}
    gates={'coverage_complete':case_status=={'completed':256},'primary_strongest_control':best,'primary_net_gain':gain,'primary_gain_pass':gain>=8,'insufficient_primary_headroom':counts['clustered_outlier'][best]>56,'paired_harms_15':harms,'paired_harm_pass':all(v<=2 for row in harms.values() for v in row.values())}
    gates['decision']='GO' if gates['coverage_complete'] and gates['primary_gain_pass'] and gates['paired_harm_pass'] else 'NO_GO'
    for k,v in gates.items(): check(evaluation['gates'][k]==v,'gate '+k)
    description=load(root/'saved-case-description.json'); unsupported.sort(key=lambda x:x['case_id']); ids=[x['case_id'] for x in unsupported]
    check(unsupported==description['unsupported_cases'] and raw_harms==description['raw_success_tail_failure_ids'] and sorted(raw_harms)==ids==description['overlap_with_unsupported'] and description['raw_success_tail_failure_not_unsupported']==[] and description['unsupported_not_raw_success_tail_failure']==[],'exact twelve failure/harm overlap')
    check(incorrect==['sparse_isolated-23'] and [r['case_id'] for r in description['completed_but_incorrect_tail']]==incorrect,'single completed incorrect case')
    wrong=json.loads(members['cases/sparse_isolated-23.json']); saved_wrong=description['completed_but_incorrect_tail'][0]
    check(saved_wrong['all_six_saved_selections']==wrong['selections'] and all(v['selected_id']==13 for v in wrong['selections'].values()),'completed incorrect all-six ID13')
    compact=load(root/'control/compact-execution.json'); check(compact['external_supervision']==execution and compact['postcheck_execution']==load(root/'control/postcheck.execution.json'),'compact execution links')
    tests=load(root/'control/tests.json'); check(tests['passed'] and tests['tests_run']==73 and tests['skips']==tests['failures']==tests['errors']==0 and tests['linux_test_count']==7,'archived artificial test counts')
    check(case_status=={'completed':244,'tail_failed':12} and tail_status=={'completed':7985,'unsupported':12,'not_attempted':195},'status totals')
    post=load(root/'control/postcheck.json'); check(post['case_status_counts']==dict(case_status) and post['tail_hypothesis_status_counts']==dict(tail_status),'archived postcheck totals')
    replay=load(root/'control/review.json'); check(replay['review_status']=='PASS' and replay['decision']=='NO_GO' and replay['success_counts']==counts and replay['gates']==evaluation['gates'],'prior saved replay report')
    patterns={'credential':r'(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)','signed_query':r'https?://[^\s"<>]*[?&](?:X-Amz-Signature|X-Goog-Signature|sig|token)=','personal_home':r'/(?:Users|home)/[A-Za-z0-9_.-]+/'}
    findings=[]
    text_inputs={n:(root/n).read_bytes() for n in original if not n.endswith('.gz')}; text_inputs.update({'archive:'+n:b for n,b in members.items()});text_inputs['parent-review/tool-observations.json']=(root/'parent-review/tool-observations.json').read_bytes()
    for name,b in text_inputs.items():
        text=b.decode('utf-8')
        for kind,pattern in patterns.items():
            if re.search(pattern,text,re.I): findings.append({'file':name,'kind':kind})
    check(not findings,'privacy pattern review')
    return {'review_status':'PASS','scope':'Public byte identities, saved selected translation IDs and outcome reaggregation only; no RNG, DP, geometry rescoring or production evaluator run.','publication_commit':PUBLICATION,'execution_head':BASE,'checks':checks,'published_files':len(original),'copy_rows':len(copies),'summary_derivatives':derivatives,'archive_members':len(members),'archive_original_bytes':total,'prior_commit_blobs_unchanged':len(before),'old_result_protocol_working_blobs_unchanged':len(old),'frozen_inputs_unchanged':15,'canonical_freeze_sha256':digest(fb),'tool_observations_matched':phases,'tool_observations_sha256':digest((root/'parent-review/tool-observations.json').read_bytes()),'success_counts':counts,'paired':pairs,'gates':gates,'case_status_counts':dict(case_status),'tail_status_counts':dict(tail_status),'unsupported_ids':ids,'raw_success_tail_failure_ids':sorted(raw_harms),'completed_incorrect_ids':incorrect,'privacy_pattern_findings':findings,'limitations':['Private cloud-cache preservation is supported by the archived cloud postcheck, not independently rehashed locally.','Copy source digests are archived declarations; unavailable private originals were not reopened.','The 20KB observation file was supplied from parent tool retrieval; this review matches its eight records and does not claim a new cloud observation.','Publication commands were not freshly observed: failed shell push exit128 is archived and Connector publication/remote verification is separately reported by parent.','No exact-tail counting, RNG replay, input-coordinate score reconstruction or new science was performed.'],'reviewer_sha256':digest(Path(__file__).read_bytes())}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3]);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    result=review(args.repo.resolve());args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'review_status':result['review_status'],'decision':result['gates']['decision'],'checks':result['checks']}))
