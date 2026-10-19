"""Independent saved JSON arithmetic only. No production imports, NumPy, models or RNG."""
import argparse, hashlib, json, math, re, subprocess
from collections import Counter
from pathlib import Path

ARMS=('zero_shot','native_alpha1','native_alpha4','loo_alpha1','other_out_alpha1')
CONTROLS=('zero_shot','native_alpha1','native_alpha4','other_out_alpha1')
CLASSES=('AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake')

def require(ok,reason):
    if not ok: raise ValueError(reason)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def ih(ids):return digest(json.dumps(list(ids),separators=(',',':')).encode())
def read(path):
    require(path.is_file() and not path.is_symlink(),'missing/nonregular '+str(path))
    return json.loads(path.read_text())
def check_file(path,spec):
    require(path.is_file() and not path.is_symlink(),'file kind '+str(path));raw=path.read_bytes()
    got={'bytes':len(raw),'sha256':digest(raw),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
    require(all(v is None or got[k]==v for k,v in spec.items()),'file identity '+str(path))
def top1(p):
    require(type(p) is list and len(p)==10 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in p),'finite 10-way probability')
    require(abs(sum(p)-1)<=2e-6,'probability mass')
    best=max(p);require(p.count(best)==1,'unique argmax')
    return p.index(best)
def reconstruct(saved,labels):
    # This gate precedes all label access. No partial-cohort accuracy.
    if saved.get('status')!='completed':return {'status':'INCOMPLETE_UNSCORED'}
    require(saved['arms']==list(ARMS) and len(saved['rows'])==128,'fixed complete grid')
    rows=saved['rows'];ids=[r['id'] for r in rows]
    require(len(set(ids))==128 and all(re.fullmatch('[a-f0-9]{64}',q) for q in ids),'opaque IDs')
    require(saved['inputs']['ids_sha256']==ih(ids),'ID hash')
    events=saved['fit_states'];require(len(events)==132 and saved['successful_native_solver_executions']==264,'132/264 coverage')
    for b in range(2):
        group=ids[64*b:64*b+64]
        for j,e in enumerate(events[66*b:66*b+66]):
            excluded=None if j<2 else group[j-2];support=[q for q in group if q!=excluded]
            require(e['status']=='completed' and e['batch']==b and e['excluded_id']==excluded and e['support_n']==len(support) and e['support_ids_sha256']==ih(support),'fit lineage')
            require(e['alpha']==(4 if j==1 else 1) and e['tag']==('full-alpha'+str(e['alpha']) if j<2 else 'delete-'+str(j-2)),'fit tag/alpha')
    pred=[]
    for i,r in enumerate(rows):
        require(r['batch']==i//64 and r['donor_id']==ids[(i//64)*64+(i+1)%64],'donor/batch')
        require(set(r['statuses'])==set(ARMS) and all(r['statuses'][a]=='ok' for a in ARMS) and set(r['probabilities'])==set(ARMS),'all 640 valid')
        pred.append({a:top1(r['probabilities'][a]) for a in ARMS})
    cost=saved['costs_seconds'];keys={'zero_shot','fits_with_parity','one_way_readouts','predictor_wall_before_final_save'}
    require(set(cost)==keys and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in cost.values()),'cost fields')
    require(cost['predictor_wall_before_final_save']>=sum(cost[k] for k in keys if k!='predictor_wall_before_final_save'),'cost accounting')
    y=list(labels);require(len(y)==128 and all(type(v) is int and 0<=v<10 for v in y),'labels')
    correct={a:[pred[i][a]==y[i] for i in range(128)] for a in ARMS};counts={a:sum(correct[a]) for a in ARMS}
    best=max(counts[a] for a in CONTROLS);co_best=[a for a in CONTROLS if counts[a]==best]
    def pairs(index,a):
        corrections=sum(correct['loo_alpha1'][i] and not correct[a][i] for i in index)
        harms=sum(correct[a][i] and not correct['loo_alpha1'][i] for i in index)
        return dict(n=len(index),corrections=corrections,harms=harms,net=corrections-harms,disagreements=sum(pred[i]['loo_alpha1']!=pred[i][a] for i in index))
    paired={a:pairs(list(range(128)),a) for a in CONTROLS}
    batches=[{a:pairs(list(range(b*64,b*64+64)),a) for a in CONTROLS} for b in range(2)]
    occ=[]
    for b in range(2):
        freq=Counter(y[b*64:b*64+64]);occ.extend(freq[v] for v in y[b*64:b*64+64])
    strata={}
    for label,test in [('one',lambda n:n==1),('two',lambda n:n==2),('three_plus',lambda n:n>=3)]:
        idx=[i for i,n in enumerate(occ) if test(n)];strata[label]={a:pairs(idx,a) for a in CONTROLS}
    gates=dict(net_eight_over_best=counts['loo_alpha1']-best>=8,both_batches_positive=all(batches[b][a]['net']>0 for b in range(2) for a in co_best),no_singleton_net_harm=all(strata['one'][a]['net']>=0 for a in ('native_alpha1','native_alpha4')))
    insensitive=128-best<8 or any(paired[a]['disagreements']<8 for a in co_best)
    decision='CONTINUE_TO_MECHANISM' if all(gates.values()) else 'LOW_HEADROOM' if insensitive else 'NO_GO'
    summary=dict(scope='development headroom only; no independent-scene confidence claim',decision=decision,n=128,correct=counts,accuracy={a:counts[a]/128 for a in ARMS},strongest_fixed_controls=co_best,paired=paired,batches=batches,occupancy_strata=strata,singleton_assessed=1 in occ,gates=gates,costs_seconds=cost)
    return dict(status='COMPLETE_RECONSTRUCTED',summary=summary,argmax_by_arm={a:[p[a] for p in pred] for a in ARMS},occupancy_per_row=occ)

def review(output,supervision,repo,freeze_commit,execution_commit,observations=None):
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
    freeze_path='protocols/0010-real-data-headroom-freeze.json';raw=git('show',freeze_commit+':'+freeze_path);f=json.loads(raw);source=f['source_commit']
    require(git('rev-list','--parents','-n','1',freeze_commit).decode().split()==[freeze_commit,source],'freeze parent')
    require(git('diff-tree','--no-commit-id','--name-only','-r',source,freeze_commit).decode().splitlines()==[freeze_path],'receipt-only commit')
    require(git('show',execution_commit+':'+freeze_path)==raw,'execution receipt')
    for n,s in f['inputs'].items():
        b=git('show',source+':'+n)
        require(len(b)==s['bytes'] and digest(b)==s['sha256'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==s['git_blob_sha1'],'source input '+n)
        require(git('show',execution_commit+':'+n)==b,'execution input '+n)
    plan_raw=git('show',source+':experiments/0010-headroom/plan.json');plan=json.loads(plan_raw)
    require(digest(plan_raw)==f['plan_sha256'] and plan['admitted'] is True,'admitted frozen plan')
    e=read(supervision/'execution.json');cmd=read(supervision/'command.json')
    require(all(e[k]==v for k,v in cmd.items()),'supervision command closure')
    for n,s in e['logs'].items():check_file(supervision/n,s)
    require(e['cpu_affinity']==[0,1,2,3] and e['numeric_threads']==4 and e['address_space_bytes']==8589934592 and e['wall_limit_seconds']==900,'fixed resource limits')
    argv=e['argv'];require(argv[2]=='_child' and argv[1].endswith('/experiments/0010-headroom/run.py'),'supervised frozen child')
    require(argv[argv.index('--plan-sha256')+1]==f['plan_sha256'],'supervision plan SHA')
    recorded_output=argv[argv.index('--output')+1]
    actual={};actual_status='NOT_SUPPLIED'
    if observations:
        ob=read(observations);records=ob.get('commands',[ob.get('item')])
        records=[x for x in records if x]
        matches=[]
        for c in records:
            for line in c.get('output',{}).get('text','').splitlines():
                try:v=json.loads(line)
                except (ValueError,TypeError):continue
                if isinstance(v,dict) and v.get('command_sha256'):
                    actual.setdefault('captured_records',[]).append(v)
            if recorded_output in c.get('command','') and 'launch' in c.get('command',''):matches.append(c)
        actual_status='SUPPLIED_REQUIRES_CAPTURE_MAPPING';actual['launch_tool_candidates']=[{k:c.get(k) for k in ('id','status','exitCode','durationMs')} for c in matches]
    receipt=read(output/'receipt.json') if (output/'receipt.json').is_file() else None
    success=e['child_exit_code']==e['supervisor_exit_code']==e['raw_wait_status']==0 and not e['timed_out'] and not e['supervisor_interrupted'] and all(e[k] is None for k in ('failure_type','failure_stage','reap_error'))
    result=dict(scope='saved-public arithmetic/provenance only; no refit/model/raw-array replay',freeze_sha256=digest(raw),source_commit=source,freeze_inputs=len(f['inputs']),execution_commit=execution_commit,outer_actual_observation_status=actual_status,actual_observation_summary=actual,supervised_exit=e['supervisor_exit_code'])
    if not success or not receipt or receipt.get('status')!='completed':
        result.update(status='INCOMPLETE_UNSCORED',accuracy_computed=False,retained_score_artifact_present=(output/'scores.json').exists(),scoring_attempted_if_recorded=receipt.get('scoring_attempted') if receipt else None);return result
    require([receipt[k] for k in ('image_calls_attempted','image_calls_completed','images_completed','text_calls','scoring_attempted','scoring_completed')]==[32,32,128,0,1,1],'fixed model/score counts')
    for n,k in [('selection-lock.json','selection_lock_sha256'),('prediction-seal.json','prediction_seal_sha256'),('scores.json','scores_sha256')]:require(digest((output/n).read_bytes())==receipt[k],'receipt '+n)
    lock=read(output/'selection-lock.json');seal=read(output/'prediction-seal.json')
    require(seal==read(output/'score-started.json'),'seal before score token')
    require(seal['all_arms']==640 and seal['features_sha256']==receipt['features_sha256'] and seal['selection_lock_sha256']==receipt['selection_lock_sha256'],'complete seal binding')
    require(seal['predictions_sha256']==digest((output/'predictions.json').read_bytes()),'prediction seal hash')
    selection=json.loads(git('show',source+':experiments/0010-headroom/selection.json'))['names']
    excluded=json.loads(git('show',source+':experiments/0010/selection.json'))['names']
    train=git('show',source+':experiments/0010/bundle/metadata/eurosat-spatial-train.txt').decode().splitlines()
    expected=sorted(set(train)-set(excluded),key=lambda n:(hashlib.sha256(('0010-stata-headroom-v1|'+n).encode()).digest(),n))[:128]
    require(selection==expected==lock['names'] and len(set(expected))==128 and not set(expected)&set(excluded),'128 fixed/excluded8 selector')
    ids=[digest(('0010-stata-headroom-id-v1|'+n).encode()) for n in expected]
    require(lock['ids']==ids and lock['batches']==[i//64 for i in range(128)],'locked opaque ID/batch order')
    require(lock['evidence']['class_order']==list(CLASSES),'class mapping fixed')
    require(lock['evidence']['inputs']=={k:v['sha256'] for k,v in plan['inputs'].items()},'lock plan bindings')
    require(len(lock['members'])==len(receipt['members'])==128,'JPEG member coverage')
    hashes=[]
    for i,(a,b) in enumerate(zip(lock['members'],receipt['members'])):
        require(Path(a['name']).name==expected[i] and b['id']==ids[i] and b['bytes']==a['bytes'],'member order/length')
        hashes.append(b['sha256'])
    require(len(set(hashes))==128,'unique new image bytes')
    old_probe=next(v['public_evidence_path'] for k,v in f['external_inputs'].items() if k=='capability_receipt')
    probe=json.loads(git('show',source+':'+old_probe))
    require(not set(hashes)&{x['sha256'] for x in probe['members']},'no capability duplicate bytes')
    saved=read(output/'predictions.json');require(saved['status']=='completed','no partial scoring')
    require([r['id'] for r in saved['rows']]==ids,'prediction rows follow lock')
    derived=reconstruct(saved,[CLASSES.index(n.rsplit('_',1)[0]) for n in expected])
    require(derived['summary']==read(output/'scores.json'),'independent summary equals saved score')
    result.update(status='PASS_PUBLIC_SAVED_ARITHMETIC',accuracy_computed=True,summary=derived['summary'],argmax_by_arm=derived['argmax_by_arm'],occupancy_per_row=derived['occupancy_per_row'],model_calls=[32,32,128,0],fit_states=132,native_executions=264,features_raw_available_locally=(output/'features.npz').is_file())
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output-root','supervision-root','repo','freeze-commit','execution-commit','report'):p.add_argument('--'+name,required=True)
    p.add_argument('--observations');a=p.parse_args()
    result=review(Path(a.output_root),Path(a.supervision_root),Path(a.repo),a.freeze_commit,a.execution_commit,Path(a.observations) if a.observations else None)
    with Path(a.report).open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:result[k] for k in ('status','accuracy_computed')}))
