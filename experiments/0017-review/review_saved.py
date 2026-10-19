#!/usr/bin/env python3
"""Independent saved-only audit: no production imports, HDF5, or estimators.

Only reference-rotation SVD uses NumPy. Angles, choices and trapezoids use
scalar Python arithmetic. Saved geometric scores and depth counts are bound
inputs: their private coordinate-dependent construction cannot be recomputed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS=('e_norm','e_ts','depth_scale','depth_affine')
SCENES=('florence_cathedral','lincoln_memorial','london_bridge','milan_cathedral','sagrada_familia')
SIZES=(10675000,10678136,10486136,10636784,10636488)
SALT='evr0017:development:2026-10-19\0'
BASE_URL='https://vision.maths.lth.se/viktor/posebench/relative/monodepth/'
ANGLE_ATOL=3e-6
AUC_ATOL=1e-7
SOURCE_PATHS={
    'experiments/0017-depth-information/'+name for name in
    ('common.py','predict_pair.py','run_diagnostic.py','evaluate_saved.py','acquire_and_allocate.py')
} | {'protocols/0017-depth-information.md','protocols/0017-depth-assets.md',
     'configs/0017-libraries.json','experiments/0013-pose-headroom/evaluate_sealed.py'}


def need(condition, message):
    if not condition: raise ValueError(message)


def finite_tree(value):
    if isinstance(value,float): need(math.isfinite(value),'Nonfinite JSON')
    elif isinstance(value,list):
        for item in value: finite_tree(item)
    elif isinstance(value,dict):
        for item in value.values(): finite_tree(item)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load(path):
    def unique(pairs):
        out={}
        for key,value in pairs:
            need(key not in out,'Duplicate JSON key'); out[key]=value
        return out
    value=json.loads(Path(path).read_text(),object_pairs_hook=unique)
    finite_tree(value)
    return value


def safe(folder,name):
    need(isinstance(name,str) and Path(name).name==name,'Unsafe filename')
    path=Path(folder)/name
    need(not path.is_symlink() and path.resolve().parent==Path(folder).resolve(),'Unsafe symlink')
    return path


def near(actual,expected,label,atol=1e-9):
    need(type(actual) in (int,float) and type(expected) in (int,float)
         and math.isfinite(actual) and math.isfinite(expected)
         and math.isclose(actual,expected,abs_tol=atol,rel_tol=1e-12),label+' differs')


def compare(actual,expected,label,atol=1e-9):
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),label+' keys differ')
        for key in expected: compare(actual[key],expected[key],label+'.'+key,atol)
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),label+' length differs')
        for i,(a,b) in enumerate(zip(actual,expected)): compare(a,b,label+str(i),atol)
    elif type(expected) in (int,float): near(actual,expected,label,atol)
    else: need(type(actual) is type(expected) and actual==expected,label+' differs')


def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))


def det(a):
    return math.fsum((a[0][0]*a[1][1]*a[2][2],a[0][1]*a[1][2]*a[2][0],a[0][2]*a[1][0]*a[2][1],
                     -a[0][2]*a[1][1]*a[2][0],-a[0][1]*a[1][0]*a[2][2],-a[0][0]*a[1][2]*a[2][1]))


def matrix(value):
    need(isinstance(value,list) and len(value)==3 and all(isinstance(row,list) and len(row)==3 for row in value),'Matrix shape')
    need(all(type(v) in (int,float) and math.isfinite(v) for row in value for v in row),'Matrix finite')
    return value


def rotation(value):
    r=matrix(value)
    need(all(abs(dot(r[i],r[j])-float(i==j))<=1e-6 for i in range(3) for j in range(3)),'Rotation orthogonality')
    need(abs(det(r)-1)<=1e-6,'Rotation determinant')
    return r


def unit(value):
    if isinstance(value,list) and len(value)==1 and isinstance(value[0],list): value=value[0]
    elif isinstance(value,list) and len(value)==3 and all(isinstance(v,list) and len(v)==1 for v in value): value=[v[0] for v in value]
    need(isinstance(value,list) and len(value)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in value),'Translation shape/finite')
    norm=math.hypot(*value)
    need(math.isfinite(norm) and norm>0,'Translation norm')
    return [v/norm for v in value],norm


def degrees(cosine): return math.degrees(math.acos(max(-1.,min(1.,cosine))))


def reference(saved):
    raw=matrix(saved['R_original']); u,s,vt=np.linalg.svd(np.array(raw,dtype=float))
    u,vt=u.tolist(),vt.tolist()
    r=[[math.fsum(u[i][k]*vt[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    need(min(s)>0 and det(r)>0,'Improper/rank-deficient reference projection'); rotation(r)
    t,norm=unit(saved['t_original'])
    compare(saved['R_projected'],r,'GT projected R',2e-12)
    compare(saved['t_unit'],t,'GT t unit',2e-12)
    near(saved['translation_original_norm'],norm,'GT t norm')
    compare(saved['singular_values'],s.tolist(),'GT singular values',2e-12)
    near(saved['projected_determinant'],det(r),'GT determinant',2e-12)
    near(saved['projection_frobenius_change'],math.hypot(*(r[i][j]-raw[i][j] for i in range(3) for j in range(3))),'GT projection change',2e-12)
    need(saved['projection']=='U @ Vt, no determinant-sign correction','GT projection convention')
    return r,t


def errors(estimate,gt):
    r=rotation(estimate['R']); t,_=unit(estimate['t'])
    er=degrees((math.fsum(r[i][j]*gt[0][i][j] for i in range(3) for j in range(3))-1)/2)
    et=degrees(dot(t,gt[1])); folded=min(et,180-et)
    return {'rotation_degrees':er,'translation_degrees':et,'max_degrees':max(er,et),
            'translation_signfolded_degrees':folded,'max_signfolded_degrees':max(er,folded)}


def auc(values,threshold):
    need(values and threshold>0 and all(math.isfinite(v) and v>=0 for v in values),'AUC input')
    previous_x=previous_y=0.; areas=[]
    for index,x in enumerate(sorted(values),1):
        if x>=threshold: break
        y=index/len(values)
        areas.append((x-previous_x)*(y+previous_y)/2)
        previous_x,previous_y=x,y
    areas.append((threshold-previous_x)*previous_y)
    return math.fsum(areas)/threshold


def expected_options(arm):
    out={'ransac':{'min_iterations':100,'max_iterations':1000,'dyn_num_trials_mult':3.,'success_prob':.9999,
        'seed':0,'progressive_sampling':False,'max_prosac_iterations':100000,'score_initial_model':False},'bundle':{}}
    if arm.startswith('e_'): out.update(max_error=2.,tangent_sampson=arm=='e_ts',real_focal_check=False)
    else: out.update(max_errors=[16.,2.],estimate_shift=arm=='depth_affine',weight_sampson=1.)
    return out


def verify_files(repo,manifest_path,seal_path,evaluation_path):
    repo,manifest_path,seal_path,evaluation_path=map(Path,(repo,manifest_path,seal_path,evaluation_path))
    m,s,e=load(manifest_path),load(seal_path),load(evaluation_path)
    mh,sh=digest(manifest_path),digest(seal_path)
    sources={name:digest(repo/name) for name in SOURCE_PATHS}
    need(m['schema']=='evr0017.development.v1' and m['selection_salt']==SALT,'Manifest schema/salt')
    need(m['development_forever']==list(SCENES) and m['coordinate_convention']=='authored_raw_pixels','Development scope')
    need(m['source_sha256']=={p:sources[p] for p in ('experiments/0017-depth-information/acquire_and_allocate.py','protocols/0017-depth-assets.md')},'Allocation source binding')
    assets=[{'scene':n,'filename':n+'_roma_moge.h5','url':BASE_URL+n+'_roma_moge.h5','bytes':size} for n,size in zip(SCENES,SIZES)]
    need(m['assets_sha256']==hashlib.sha256(json.dumps(assets,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'Asset declaration binding')
    expected=[]; verified_assets={}
    need([v['scene'] for v in m['scenes']]==list(SCENES),'Scene inventory')
    for scene,asset in zip(m['scenes'],assets):
        need(all(scene[k]==v for k,v in asset.items()),'Scene metadata')
        keys=scene['root_keys']; need(keys==sorted(set(keys)) and len(keys)>=16,'Root-key inventory')
        chosen=sorted(keys,key=lambda k:(hashlib.sha256((SALT+scene['scene']+'\0'+k).encode()).hexdigest(),k))[:16]
        need(scene['selected_pair_keys']==chosen,'Salted selection')
        verified_assets[asset['filename']]={'bytes':asset['bytes'],'sha256':scene['sha256']}
        for key in chosen:
            expected.append({'pair_index':len(expected),'scene':scene['scene'],'key':key,'filename':scene['filename'],
                             'dataset_sha256':scene['sha256'],'dataset_bytes':scene['bytes']})
    need(m['selected_pairs']==expected and len(expected)==80,'Flat80 allocation')
    library=load(repo/'configs/0017-libraries.json')['current']
    need(library['version']=='3.0.0','Library version')
    need(s['schema']=='evr0017.seal.v1' and s['status']=='completed' and s['error'] is None,'Seal status')
    need(s['source_sha256']==sources and s['manifest_sha256']==mh and s['library']==library,'Seal bindings')
    need(s['verified_assets']==verified_assets,'Verified asset metadata')
    integrity=safe(seal_path.parent,'integrity.json')
    need(digest(integrity)==s['integrity_sha256'],'Integrity receipt hash')
    need(load(integrity)=={'manifest_sha256':mh,'source_sha256':sources,'library':library,'verified_assets':verified_assets},'Integrity receipt contents')
    need(e['schema']=='evr0017.evaluation.v1' and e['status']=='completed' and e['pair_count']==80,'Evaluation status')
    need(e['manifest_sha256']==mh and e['seal_sha256']==sh and e['source_sha256']==sources,'Evaluation bindings')
    need(len(s['rows'])==320,'Missing sealed arms')
    scheduled=[(i,a) for i in range(80) for a in ARMS[i%4:]+ARMS[:i%4]]
    need([(r['pair_index'],r['arm']) for r in s['rows']]==scheduled,'Rotating arm schedule differs')
    records={}; shared={}; hashes={'integrity.json':s['integrity_sha256']}
    for row in s['rows']:
        i,arm=row['pair_index'],row['arm']; item=(i,arm)
        need(type(i) is int and 0<=i<80 and arm in ARMS and item not in records,'Unknown/duplicate arm')
        identity=expected[i]
        need(row['scene']==identity['scene'] and row['key']==identity['key'],'Seal row identity')
        paths={kind:safe(seal_path.parent,row[field]) for kind,field in (('final','path'),('raw','raw_path'),('process','process_path'))}
        for kind,field in (('final','sha256'),('raw','raw_sha256'),('process','process_sha256')):
            need(digest(paths[kind])==row[field],'Changed '+kind+' file'); hashes[paths[kind].name]=row[field]
        raw,final,process=(load(paths[k]) for k in ('raw','final','process'))
        log=safe(seal_path.parent,process['log_path']); need(digest(log)==process['log_sha256'],'Changed process log')
        hashes[log.name]=process['log_sha256']
        need(process['raw_worker_path']==row['raw_path'] and process['raw_worker_sha256']==row['raw_sha256'],'Process/raw linkage')
        bound={'schema':'evr0017.prediction.v1','pair_index':i,'scene':identity['scene'],'key':identity['key'],
            'filename':identity['filename'],'arm':arm,'manifest_sha256':mh,'source_sha256':sources,
            'dataset_sha256_declared':identity['dataset_sha256'],'dataset_sha256_verified':identity['dataset_sha256'],
            'dataset_bytes':identity['dataset_bytes'],'library':library,'options':expected_options(arm)}
        need(all(raw.get(k)==v for k,v in bound.items()),'Raw worker bindings')
        termination,code=process['termination'],process['exit_code']
        if raw['status']=='completed' and termination is None and code==0: status='completed'
        elif raw['status']=='error' and raw.get('failure_kind')=='estimator' and termination is None and code==1: status='estimator_error'
        elif raw['status']=='started' and raw.get('phase')=='estimator' and termination=='pair_timeout': status='timeout'
        elif raw['status']=='started' and raw.get('phase')=='estimator' and termination is None and code!=0: status='estimator_error'
        else: raise ValueError('Noneligible worker/process state')
        timings=dict(raw.get('timings',{}))
        for key in ('input_seconds','call_seconds','score_seconds'): timings.setdefault(key,None)
        timings['worker_total_seconds']=timings.get('charged_total_seconds'); timings['charged_total_seconds']=process['wall_seconds']
        reconstructed={**raw,'status':status,'process_receipt':process,'timings':timings}
        need(final==reconstructed and row['status']==status,'Final normalization differs')
        inputs={k:final[k] for k in ('input_arrays','calibration','input_diagnostics')}
        need(i not in shared or shared[i]==inputs,'Different inputs across four arms'); shared[i]=inputs
        if status=='completed':
            n=final['input_arrays']['x1']['shape'][0]
            need(type(n) is int and n>0 and final['mask_length']==n,'Mask/input length')
            packed=bytes.fromhex(final['native_inliers_packbits_big_hex'])
            need(len(packed)==(n+7)//8,'Mask packed length')
            bits=[(byte>>shift)&1 for byte in packed for shift in range(7,-1,-1)]
            need(not any(bits[n:]) and sum(bits[:n])==final['kept_count'],'Mask padding/count')
            need(hashlib.sha256(bytes(bits[:n])).hexdigest()==final['mask_sha256'],'Mask hash')
            score=final['final_scores']['pixel_msac']; need(type(score) in (int,float) and 0<=score<=n,'Saved score range')
            if arm.startswith('depth_'):
                geometry=final['geometry']; need(all(geometry[k]==final['estimate'][k] for k in ('R','t')),'Raw depth geometry differs')
                need(all(type(geometry[k]) in (int,float) for k in ('scale','shift1','shift2')),'Depth parameter types')
                need(final['depth_diagnostics']['all']['denominator']==n and final['depth_diagnostics']['returned_mask']['denominator']==final['kept_count'],'Depth count denominator')
        records[item]=final
    need(len(records)==320,'Incomplete Cartesian product')
    return m,s,e,records,{'manifest_sha256':mh,'seal_sha256':sh,'evaluation_sha256':digest(evaluation_path),
                        'source_sha256':sources,'prediction_and_process_file_sha256':hashes}


def summary(outcomes):
    values=[o['penalized_max_degrees'] for o in outcomes]
    folded=[o['penalized_max_signfolded_degrees'] for o in outcomes]
    successful=[o['penalized_max_degrees'] for o in outcomes if o['status']=='completed']
    return {'pair_count':len(outcomes),'status_counts':dict(Counter(o['status'] for o in outcomes)),
        'auc_fraction':{str(t):auc(values,t) for t in (5,10,20)},
        'signfolded_auc_fraction':{str(t):auc(folded,t) for t in (5,10,20)},
        'count_strictly_below':{str(t):sum(v<t for v in values) for t in (5,10,20)},
        'success_only_auc_fraction':{str(t):auc(successful,t) for t in (5,10,20)} if successful else None}


def outcome(record,gt):
    if record['status']=='completed':
        er=errors(record['estimate'],gt)
        return {'status':'completed','error':er,'penalized_max_degrees':er['max_degrees'],
                'penalized_max_signfolded_degrees':er['max_signfolded_degrees']}
    need(not record.get('estimate'),'Failed record retains a pose')
    return {'status':record['status'],'error':None,'penalized_max_degrees':180.,'penalized_max_signfolded_degrees':180.}


def choice(arm,outcomes):
    if arm is not None: return {**outcomes[arm],'arm':arm}
    return {'status':'unavailable','arm':None,'error':None,'penalized_max_degrees':180.,'penalized_max_signfolded_degrees':180.}


def arithmetic(identities,records,evaluation):
    rows=evaluation['rows']; need(len(rows)==80,'Evaluation needs all80')
    need([r['pair_index'] for r in rows]==list(range(80)),'Evaluation row order')
    methods=list(ARMS)+['pixel_msac_external','oracle_geometry','oracle_depth','oracle_all']
    recomputed=[]; oracle_discrepancies=[]
    comparison_totals={d:{g:Counter() for g in ARMS[:2]} for d in ARMS[2:]}
    for i,row in enumerate(rows):
        ident=identities[i]; need(row['scene']==ident['scene'] and row['key']==ident['key'],'Evaluation identity')
        gt=reference(row['ground_truth']); bank={a:records[(i,a)] for a in ARMS}
        outcomes={a:outcome(bank[a],gt) for a in ARMS}
        compare(row['outcomes'],outcomes,'Independent pose errors',ANGLE_ATOL)
        successful=[a for a in ARMS if outcomes[a]['status']=='completed']
        selected=min(successful,key=lambda a:(bank[a]['final_scores']['pixel_msac'],ARMS.index(a))) if successful else None
        choices={'pixel_msac_external':choice(selected,outcomes)}
        for name,arms in (('oracle_geometry',ARMS[:2]),('oracle_depth',ARMS[2:]),('oracle_all',ARMS)):
            eligible=[a for a in arms if a in successful]
            # Exact identity must follow the saved-error arithmetic; scalar-angle
            # rounding is separately bounded and never relaxes deployed choices.
            saved_best=min(eligible,key=lambda a:(row['outcomes'][a]['penalized_max_degrees'],ARMS.index(a))) if eligible else None
            independent=min(eligible,key=lambda a:(outcomes[a]['penalized_max_degrees'],ARMS.index(a))) if eligible else None
            need(row['choices'][name]['arm']==saved_best,'Saved oracle not exact argmin')
            if saved_best!=independent:
                regret=outcomes[saved_best]['penalized_max_degrees']-outcomes[independent]['penalized_max_degrees']
                need(0<=regret<=ANGLE_ATOL,'Independent oracle regret exceeds tolerance')
                oracle_discrepancies.append({'pair_index':i,'oracle':name,'saved_arm':saved_best,'scalar_arm':independent,'regret_degrees':regret})
            choices[name]=choice(saved_best,outcomes)
        need(row['choices']['pixel_msac_external']['arm']==selected,'Deployed score argmin differs')
        compare(row['choices'],choices,'Choice outcomes',ANGLE_ATOL)
        costs={}
        for a in ARMS:
            timing=bank[a]['timings']; compare(row['timings_by_arm'][a],timing,'Copied timings')
            for key in ('input_seconds','call_seconds','score_seconds','charged_total_seconds'):
                need(key in timing,'Missing timing')
                if timing[key] is None: need(bank[a]['status']!='completed' and key!='charged_total_seconds','Unknown completed/charged time')
                else: need(type(timing[key]) in (int,float) and timing[key]>=0,'Invalid time')
            need(math.fsum(timing[k] or 0 for k in ('input_seconds','call_seconds','score_seconds'))<=timing['charged_total_seconds']+1e-6,'Uncharged work')
            costs[a]=timing['charged_total_seconds']
        near(row['selector_charged_seconds'],math.fsum(costs.values()),'Selector charges every arm')
        comparisons={}; mechanism={}
        for d in ARMS[2:]:
            comparisons[d]={}; de=outcomes[d]['penalized_max_degrees']
            for g in ARMS[:2]:
                ge=outcomes[g]['penalized_max_degrees']
                comp={'depth_minus_geometry_degrees':de-ge,'strict_win':de<ge,'strict_loss':de>ge,'exact_tie':de==ge,
                    'rescue_below5':ge>=5 and de<5,'damage_above5':ge<5 and de>=5,
                    'material_benefit':ge-de>=1 or (ge>=5 and de<5),'material_harm':de-ge>=1 or (ge<5 and de>=5),
                    'both_completed':outcomes[d]['status']==outcomes[g]['status']=='completed'}
                comparisons[d][g]=comp
                for k,v in comp.items():
                    if type(v) is bool: comparison_totals[d][g][k]+=int(v)
            support=bank[d]['depth_diagnostics']['returned_mask'] if outcomes[d]['status']=='completed' else {'denominator':0,'native_branch_inactive_count':0}
            n,inactive=support['denominator'],support['native_branch_inactive_count']
            need(type(n) is int and n>=0 and type(inactive) is int and 0<=inactive<=2*n,'Inactive support counts')
            gs=[outcomes[g]['penalized_max_degrees'] for g in ARMS[:2]]
            harm=all(de-ge>=1 for ge in gs) or (de>=5 and all(ge<5 for ge in gs))
            mechanism[d]={'returned_mask_denominator':n,'native_branch_inactive_count':inactive,
                'joint_material_harm_vs_both':harm,'qualifies':outcomes[d]['status']=='completed'
                and all(outcomes[g]['status']=='completed' for g in ARMS[:2]) and harm and n>0 and inactive>0}
        compare(row['comparisons'],comparisons,'Pairwise benefit/harm',ANGLE_ATOL)
        compare(row['mechanism'],mechanism,'Mechanism conditions')
        recomputed.append({'scene':ident['scene'],'methods':{**outcomes,**choices},'costs':costs,'mechanism':mechanism})
    def summaries(subset): return {m:summary([r['methods'][m] for r in subset]) for m in methods}
    full=summaries(recomputed)
    scene={s:summaries([r for r in recomputed if r['scene']==s]) for s in SCENES}
    loso={s:summaries([r for r in recomputed if r['scene']!=s]) for s in SCENES}
    need(all(scene[s]['e_norm']['pair_count']==16 and loso[s]['e_norm']['pair_count']==64 for s in SCENES),'Scene/LOSO denominators')
    compare(evaluation['summaries'],full,'All80 summaries',AUC_ATOL)
    compare(evaluation['scene_summaries'],scene,'Scene16 summaries',AUC_ATOL)
    compare(evaluation['leave_one_scene_out_descriptive'],loso,'LOSO64 summaries',AUC_ATOL)
    compare(evaluation['comparisons_summary'],{d:{g:dict(v) for g,v in groups.items()} for d,groups in comparison_totals.items()},'Comparison totals')
    f=lambda name:full[name]['auc_fraction']['5']
    headroom=100*(f('oracle_all')-f('oracle_geometry'))
    affected={d:[s for s in SCENES if any(r['scene']==s and r['mechanism'][d]['qualifies'] for r in recomputed)] for d in ARMS[2:]}
    decisions={'additional_depth_oracle_headroom_pp':headroom,'selector_necessary_headroom_pass':headroom>=2.,
        'oracle_all_minus_best_single_pp':100*(f('oracle_all')-max(f(a) for a in ARMS)),
        'omission_mechanism_scenes_by_depth_arm':affected,'omission_mechanism_trigger_by_arm':{d:len(affected[d])>=2 for d in ARMS[2:]},
        'new_method_improvement_established':False,'manuscript_admitted':False}
    compare(evaluation['decisions'],decisions,'Prospective decisions',100*AUC_ATOL)
    costs={a:math.fsum(r['costs'][a] for r in recomputed) for a in ARMS}
    compare(evaluation['charged_seconds_by_arm'],costs,'Arm costs')
    near(evaluation['external_selector_charged_seconds'],math.fsum(costs.values()),'External total cost')
    return {'decisions':decisions,'independent_summaries':full,'independent_scene_summaries':scene,
            'independent_leave_one_scene_out':loso,'charged_seconds_by_arm':costs,
            'oracle_scalar_roundoff_discrepancies':oracle_discrepancies}


def audit(repo,manifest,seal,evaluation):
    m,s,e,records,bindings=verify_files(repo,manifest,seal,evaluation)
    computed=arithmetic(m['selected_pairs'],records,e)
    # Detect changes while the saved-only audit ran.
    for name,sha in bindings['source_sha256'].items(): need(digest(Path(repo)/name)==sha,'Source changed during audit')
    need(digest(evaluation)==bindings['evaluation_sha256'] and digest(seal)==bindings['seal_sha256'] and digest(manifest)==bindings['manifest_sha256'],'Bound input changed during audit')
    for name,sha in bindings['prediction_and_process_file_sha256'].items(): need(digest(safe(Path(seal).parent,name))==sha,'Record changed during audit')
    return {'schema':'evr0017.saved-review.v1','status':'passed','scope':'Saved-only independent arithmetic and receipt audit',
        'review_source_sha256':digest(__file__),'pair_count':80,'arm_record_count':320,
        'angle_tolerance_degrees':ANGLE_ATOL,'auc_fraction_tolerance':AUC_ATOL,
        'limitations':['No private cache, images, estimator, or production evaluation code imported.',
            'Geometric pixel MSAC and depth inactivity/reprojection counts are hash-bound saved inputs; private coordinates are unavailable, so these values are not recomputed.',
            'Reference poses are independently processed from the evaluation JSON, not re-read from the original cache.',
            'Saved library identity is checked against the pinned config; this auditor does not reopen the private native binary or cached data files.'],
        'bindings':bindings,**computed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','manifest','seal','evaluation','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    # Exclusive receipt, retained even if verification fails.
    with args.output.open('x+') as stream:
        report={'schema':'evr0017.saved-review.v1','status':'started'}
        stream.write(json.dumps(report)+'\n'); stream.flush()
        try: report=audit(args.repo,args.manifest,args.seal,args.evaluation)
        except BaseException as exc:
            report={'schema':'evr0017.saved-review.v1','status':'failed','error':type(exc).__name__+': '+str(exc),
                    'review_source_sha256':digest(__file__)}
            stream.seek(0); stream.write(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n'); stream.truncate(); stream.flush()
            raise
        stream.seek(0); stream.write(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n'); stream.truncate(); stream.flush()
    print(json.dumps({'status':'passed','pair_count':80,'arm_record_count':320}))


if __name__=='__main__': main()
