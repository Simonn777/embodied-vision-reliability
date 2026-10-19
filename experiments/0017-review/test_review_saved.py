"""Artificial independent-review fixtures; imports no production module."""
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import review_saved as audit

I=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))


def oracle_summary(outcomes):
    # Independent test reference via explicitly assembled CDF and NumPy trapezoid.
    def area(values,t):
        sorted_values=sorted(values)
        inside=[(v,(i+1)/len(values)) for i,v in enumerate(sorted_values) if v<t]
        x=[0.]+[v for v,_ in inside]+[float(t)]
        y=[0.]+[r for _,r in inside]+[inside[-1][1] if inside else 0.]
        return float(np.trapezoid(y,x))/t
    errors=[v['penalized_max_degrees'] for v in outcomes]
    folded=[v['penalized_max_signfolded_degrees'] for v in outcomes]
    successes=[v['penalized_max_degrees'] for v in outcomes if v['status']=='completed']
    return {'pair_count':len(outcomes),'status_counts':dict(Counter(v['status'] for v in outcomes)),
        'auc_fraction':{str(t):area(errors,t) for t in (5,10,20)},
        'signfolded_auc_fraction':{str(t):area(folded,t) for t in (5,10,20)},
        'count_strictly_below':{str(t):sum(v<t for v in errors) for t in (5,10,20)},
        'success_only_auc_fraction':{str(t):area(successes,t) for t in (5,10,20)} if successes else None}


def arithmetic_fixture():
    identities=[]; records={}; rows=[]
    for i in range(80):
        scene=audit.SCENES[i//16]; key='key-%02d'%(i%16)
        identities.append({'pair_index':i,'scene':scene,'key':key})
        outcomes={}; timings={}
        for a,theta,cost,score in zip(audit.ARMS,(2.2,3.4,7.1 if i<32 else 4.8,1.1),(1.,2.,3.,4.),(.2,.1,.3,.4)):
            status='timeout' if i==79 else ('estimator_error' if i==78 and a=='depth_affine' else 'completed')
            if i==0 and a=='e_norm': score=.1
            t=[math.cos(math.radians(theta)),math.sin(math.radians(theta)),0.]
            record={'status':status,'estimate':{'R':I,'t':t} if status=='completed' else None,
                'final_scores':{'pixel_msac':score},'timings':{'input_seconds':.1,'call_seconds':.1 if status=='completed' else None,
                'score_seconds':.1 if status=='completed' else None,'charged_total_seconds':cost},
                'depth_diagnostics':{'all':{'denominator':8},'returned_mask':{'denominator':8,
                    'native_branch_inactive_count':1 if a=='depth_scale' and i<32 else 0}} if a.startswith('depth_') else None}
            records[(i,a)]=record; timings[a]=record['timings']
            er={'rotation_degrees':0.,'translation_degrees':theta,'max_degrees':theta,
                'translation_signfolded_degrees':theta,'max_signfolded_degrees':theta} if status=='completed' else None
            outcomes[a]={'status':status,'error':er,'penalized_max_degrees':theta if er else 180.,'penalized_max_signfolded_degrees':theta if er else 180.}
        def selected(a):
            return {**outcomes[a],'arm':a} if a else {'status':'unavailable','arm':None,'error':None,
                'penalized_max_degrees':180.,'penalized_max_signfolded_degrees':180.}
        choices={'pixel_msac_external':selected(None if i==79 else ('e_norm' if i==0 else 'e_ts')),
            'oracle_geometry':selected(None if i==79 else 'e_norm'),
            'oracle_depth':selected(None if i==79 else ('depth_scale' if i==78 else 'depth_affine')),
            'oracle_all':selected(None if i==79 else ('e_norm' if i==78 else 'depth_affine'))}
        comparisons={}; mechanism={}
        for d in audit.ARMS[2:]:
            de=outcomes[d]['penalized_max_degrees']; comparisons[d]={}
            for g in audit.ARMS[:2]:
                ge=outcomes[g]['penalized_max_degrees']
                comparisons[d][g]={'depth_minus_geometry_degrees':de-ge,'strict_win':de<ge,'strict_loss':de>ge,'exact_tie':de==ge,
                    'rescue_below5':ge>=5 and de<5,'damage_above5':ge<5 and de>=5,
                    'material_benefit':ge-de>=1 or (ge>=5 and de<5),'material_harm':de-ge>=1 or (ge<5 and de>=5),
                    'both_completed':outcomes[d]['status']==outcomes[g]['status']=='completed'}
            success=outcomes[d]['status']=='completed'
            # For timeout/all failed, d=g=180, so there is no joint harm.
            harm=all(de-outcomes[g]['penalized_max_degrees']>=1 for g in audit.ARMS[:2]) or (de>=5 and all(outcomes[g]['penalized_max_degrees']<5 for g in audit.ARMS[:2]))
            inactive=1 if success and d=='depth_scale' and i<32 else 0
            mechanism[d]={'returned_mask_denominator':8 if success else 0,'native_branch_inactive_count':inactive,
                'joint_material_harm_vs_both':harm,'qualifies':bool(success and i!=79 and harm and inactive)}
        gt={'R_original':I,'t_original':[1.,0.,0.],'R_projected':I,'t_unit':[1.,0.,0.],
            'translation_original_norm':1.,'projection':'U @ Vt, no determinant-sign correction',
            'projection_frobenius_change':0.,'singular_values':[1.,1.,1.],'projected_determinant':1.}
        rows.append({'pair_index':i,'scene':scene,'key':key,'ground_truth':gt,'outcomes':outcomes,'choices':choices,
            'comparisons':comparisons,'mechanism':mechanism,'timings_by_arm':timings,'selector_charged_seconds':10.})
    methods=list(audit.ARMS)+['pixel_msac_external','oracle_geometry','oracle_depth','oracle_all']
    def summarise(items):
        return {m:oracle_summary([r['outcomes'][m] if m in audit.ARMS else r['choices'][m] for r in items]) for m in methods}
    full=summarise(rows)
    comparisons={d:{g:{key:sum(r['comparisons'][d][g][key] for r in rows) for key in
        ('strict_win','strict_loss','exact_tie','rescue_below5','damage_above5','material_benefit','material_harm','both_completed')}
        for g in audit.ARMS[:2]} for d in audit.ARMS[2:]}
    headroom=100*(full['oracle_all']['auc_fraction']['5']-full['oracle_geometry']['auc_fraction']['5'])
    evaluation={'rows':rows,'summaries':full,'scene_summaries':{s:summarise([r for r in rows if r['scene']==s]) for s in audit.SCENES},
        'leave_one_scene_out_descriptive':{s:summarise([r for r in rows if r['scene']!=s]) for s in audit.SCENES},
        'comparisons_summary':comparisons,'charged_seconds_by_arm':dict(zip(audit.ARMS,(80.,160.,240.,320.))),
        'external_selector_charged_seconds':800.,'decisions':{'additional_depth_oracle_headroom_pp':headroom,
        'selector_necessary_headroom_pass':True,'oracle_all_minus_best_single_pp':100*(full['oracle_all']['auc_fraction']['5']-max(full[a]['auc_fraction']['5'] for a in audit.ARMS)),
        'omission_mechanism_scenes_by_depth_arm':{'depth_scale':list(audit.SCENES[:2]),'depth_affine':[]},
        'omission_mechanism_trigger_by_arm':{'depth_scale':True,'depth_affine':False},'new_method_improvement_established':False,'manuscript_admitted':False}}
    return identities,records,evaluation


def file_fixture(root):
    identities,records,evaluation=arithmetic_fixture(); predictions=root/'predictions'; predictions.mkdir()
    for name in audit.SOURCE_PATHS:
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('artificial source '+name)
    library={'version':'3.0.0','binary_sha256':'a'*64}; write(root/'configs/0017-libraries.json',{'current':library})
    sources={name:audit.digest(root/name) for name in audit.SOURCE_PATHS}
    scenes=[]; pairs=[]; assets=[]
    for name,size in zip(audit.SCENES,audit.SIZES):
        filename=name+'_roma_moge.h5'; keys=['key-%02d'%j for j in range(16)]
        chosen=sorted(keys,key=lambda k:(hashlib.sha256((audit.SALT+name+'\0'+k).encode()).hexdigest(),k))
        asset={'scene':name,'filename':filename,'url':audit.BASE_URL+filename,'bytes':size}; assets.append(asset)
        scenes.append({**asset,'sha256':'b'*64,'root_keys':keys,'selected_pair_keys':chosen})
        for key in chosen: pairs.append({'pair_index':len(pairs),'scene':name,'key':key,'filename':filename,'dataset_sha256':'b'*64,'dataset_bytes':size})
    manifest={'schema':'evr0017.development.v1','selection_salt':audit.SALT,'development_forever':list(audit.SCENES),
        'coordinate_convention':'authored_raw_pixels','assets_sha256':hashlib.sha256(json.dumps(assets,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'source_sha256':{p:sources[p] for p in ('experiments/0017-depth-information/acquire_and_allocate.py','protocols/0017-depth-assets.md')},
        'scenes':scenes,'selected_pairs':pairs}
    mp=root/'manifest.json'; write(mp,manifest); mh=audit.digest(mp)
    verified={s['filename']:{'bytes':s['bytes'],'sha256':s['sha256']} for s in scenes}
    write(predictions/'integrity.json',{'manifest_sha256':mh,'source_sha256':sources,'library':library,'verified_assets':verified})
    sealed=[]
    for i in range(80):
        evaluation['rows'][i]['key']=pairs[i]['key']
        for arm in audit.ARMS[i%4:]+audit.ARMS[:i%4]:
            r=copy.deepcopy(records[(i,arm)]); identity=pairs[i]; status=r['status']; stem='pair-%02d-%s'%(i,arm)
            r.update(schema='evr0017.prediction.v1',pair_index=i,scene=identity['scene'],key=identity['key'],filename=identity['filename'],arm=arm,
                manifest_sha256=mh,source_sha256=sources,dataset_sha256_declared='b'*64,dataset_sha256_verified='b'*64,
                dataset_bytes=identity['dataset_bytes'],library=library,options=audit.expected_options(arm),
                input_arrays={'x1':{'shape':[8,2]},'x2':{'shape':[8,2]},'depth1':{'shape':[8]},'depth2':{'shape':[8]}},
                input_diagnostics={'count':8},calibration={'manufactured':True},returned_ransac_stats={'iterations':100})
            if status=='completed':
                r.update(phase='complete',mask_length=8,kept_count=8,native_inliers_packbits_big_hex='ff',mask_sha256=hashlib.sha256(b'\1'*8).hexdigest())
                if arm.startswith('depth_'): r['geometry']={**r['estimate'],'scale':1.,'shift1':0.,'shift2':0.}
            outer=r['timings']['charged_total_seconds']; r['timings']['charged_total_seconds']=.5 if status=='completed' else None
            raw=copy.deepcopy(r)
            if status=='timeout': raw.update(status='started',phase='estimator')
            elif status=='estimator_error': raw.update(status='error',phase='error',failure_kind='estimator')
            rawpath=predictions/(stem+'.worker.json'); write(rawpath,raw)
            log=predictions/(stem+'.log'); log.write_text('artificial log')
            process={'exit_code':0 if status=='completed' else (-9 if status=='timeout' else 1),
                'termination':'pair_timeout' if status=='timeout' else None,'wall_seconds':outer,
                'raw_worker_path':rawpath.name,'raw_worker_sha256':audit.digest(rawpath),
                'log_path':log.name,'log_sha256':audit.digest(log)}
            processpath=predictions/(stem+'.process.json'); write(processpath,process)
            final={**raw,'status':status,'process_receipt':process,'timings':{**raw['timings'],'worker_total_seconds':raw['timings']['charged_total_seconds'],'charged_total_seconds':outer}}
            evaluation['rows'][i]['timings_by_arm'][arm]=final['timings']
            finalpath=predictions/(stem+'.json'); write(finalpath,final)
            sealed.append({'pair_index':i,'scene':identity['scene'],'key':identity['key'],'arm':arm,'path':finalpath.name,'sha256':audit.digest(finalpath),'status':status,
                'raw_path':rawpath.name,'raw_sha256':audit.digest(rawpath),'process_path':processpath.name,'process_sha256':audit.digest(processpath)})
    seal={'schema':'evr0017.seal.v1','status':'completed','error':None,'rows':sealed,'source_sha256':sources,'manifest_sha256':mh,
        'library':library,'verified_assets':verified,'integrity_sha256':audit.digest(predictions/'integrity.json')}
    sp=predictions/'seal.json'; write(sp,seal)
    evaluation.update(schema='evr0017.evaluation.v1',status='completed',pair_count=80,manifest_sha256=mh,seal_sha256=audit.digest(sp),source_sha256=sources)
    ep=root/'evaluation.json'; write(ep,evaluation)
    return mp,sp,ep


class ReviewTests(unittest.TestCase):
    def test_manual_angles_directed_signfolded_and_rotation(self):
        r=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        e=audit.errors({'R':r,'t':[-1.,0.,0.]},(I,[1.,0.,0.]))
        self.assertEqual(e['rotation_degrees'],90.); self.assertEqual(e['translation_degrees'],180.)
        self.assertEqual(e['translation_signfolded_degrees'],0.); self.assertEqual(e['max_signfolded_degrees'],90.)

    def test_exact_trapezoids_boundary_and_failure_penalty(self):
        self.assertEqual(audit.auc([0.,5.],5),.5)
        self.assertEqual(audit.auc([5.],5),0.)
        self.assertEqual(audit.auc([1.,3.],5),.75)
        self.assertEqual(audit.auc([180.]*80,20),0.)

    def test_full_arithmetic_80_scene16_loso64_and_mechanism(self):
        ids,records,ev=arithmetic_fixture(); result=audit.arithmetic(ids,records,ev)
        self.assertTrue(result['decisions']['selector_necessary_headroom_pass'])
        self.assertTrue(result['decisions']['omission_mechanism_trigger_by_arm']['depth_scale'])
        self.assertEqual(result['independent_scene_summaries'][audit.SCENES[0]]['e_norm']['pair_count'],16)
        self.assertEqual(result['independent_leave_one_scene_out'][audit.SCENES[0]]['e_norm']['pair_count'],64)

    def test_bad_deployed_choice_and_missing_failure_cost_rejected(self):
        for mutation in ('choice','cost','gate'):
            ids,records,ev=arithmetic_fixture()
            if mutation=='choice': ev['rows'][0]['choices']['pixel_msac_external']['arm']='depth_affine'
            elif mutation=='cost': ev['external_selector_charged_seconds']-=10
            else: ev['decisions']['omission_mechanism_trigger_by_arm']['depth_affine']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): audit.arithmetic(ids,records,ev)

    def test_reference_projection_and_invalid_rotation(self):
        _,_,ev=arithmetic_fixture(); gt=copy.deepcopy(ev['rows'][0]['ground_truth'])
        gt['R_original']=[[2.,0.,0.],[0.,1.,0.],[0.,0.,1.]]; gt['singular_values']=[2.,1.,1.]; gt['projection_frobenius_change']=1.
        self.assertEqual(audit.reference(gt)[0],I)
        with self.assertRaises(ValueError): audit.rotation([[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])

    def test_complete_file_chain_with_artificial_320_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); paths=file_fixture(root)
            result=audit.audit(root,*paths)
            self.assertEqual(result['status'],'passed'); self.assertEqual(result['arm_record_count'],320)
            self.assertEqual(len(result['bindings']['prediction_and_process_file_sha256']),1281)

    def test_source_output_process_log_and_seal_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); mp,sp,ep=file_fixture(root); seal=audit.load(sp); row=seal['rows'][0]
            targets=[root/'experiments/0017-depth-information/common.py',sp.parent/row['path'],sp.parent/row['process_path'],sp.parent/'pair-00-e_norm.log']
            for path in targets:
                original=path.read_bytes(); path.write_bytes(original+b' ')
                with self.subTest(path=path.name),self.assertRaises(ValueError): audit.verify_files(root,mp,sp,ep)
                path.write_bytes(original)
            changed=copy.deepcopy(seal); changed['rows'].pop(); write(sp,changed)
            # Rebinding evaluation to the changed seal must not hide an incomplete cohort.
            ev=audit.load(ep); ev['seal_sha256']=audit.digest(sp); write(ep,ev)
            with self.assertRaises(ValueError): audit.verify_files(root,mp,sp,ep)

    def test_json_nan_duplicate_and_unsafe_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            for text in ('{"x": NaN}','{"x":1,"x":2}'):
                path.write_text(text)
                with self.assertRaises(ValueError): audit.load(path)
            with self.assertRaises(ValueError): audit.safe(directory,'../outside')


if __name__=='__main__': unittest.main()
