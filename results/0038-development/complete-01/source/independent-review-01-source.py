"""Independently recompute J from sealed packed masks and pinned annotation bytes.

No production metric/readout imports, model, RGB, fitting or new prediction.
Requires a clean completed D38 outer before numerical annotation decoding.
Old five controls remain identified sealed scalar references; new four readouts
are rescored from binary pixels. Original-resolution resize was checked by the
producer; this auditor does not replay a neural decoder or interpolation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

COHORT = (('4691_cut_guava',86),('817_cut_onion',90),('9675_fold_clothes',201),('1934_tear_film',69))
FIXED = ('B','B+M1','B+M2','B+M1+M2','fixed_negative')
NEW = ('early_mean','flat_histories','late_mean','output_ensemble')
ARMS = FIXED + NEW
CONTROLS = tuple(a for a in ARMS if a != 'late_mean')
POLICIES = ('author_background','void_ignore')
KEYS = ('primary','void_ignore_sensitivity')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def require(ok,message):
    if not ok:raise ValueError(message)


def pixel_counts(truth,prediction,np,ignore):
    valid=(truth!=255) if ignore else np.ones(truth.shape,dtype=bool)
    fg=truth==1
    intersection=int(np.count_nonzero(prediction & fg & valid))
    union=int(np.count_nonzero((prediction | fg) & valid))
    return intersection,union,intersection/union if union else 1.


def run(spec_path):
    spec=read(spec_path);out=Path(spec['output']);require(not out.exists(),'fresh independent review')
    out.mkdir(parents=True)
    began=time.perf_counter()
    record=dict(schema='evr0038.independent-pixel-review.v1',status='started',passed=False,errors=[],checks=0,
        spec_sha256=sha(spec_path),source_and_input_hashes={},reviewed_frames=0,label_decodes=0,
        model_calls=0,network_calls=0,offline_attempts=[],compatibility_probes=[],maximum_difference=0.,
        independent_new_readout_pixel_scores=True,old_controls_are_sealed_scalar_references=True,
        decoder_or_interpolation_replayed=False,rows=[])
    def check(ok,label):
        record['checks']+=1
        if not ok:
            record['errors'].append(label)
            raise ValueError(label)
    def number(a,b,label):
        check(type(a) in (int,float) and math.isfinite(a),'finite '+label)
        delta=abs(a-b);record['maximum_difference']=max(record['maximum_difference'],delta)
        check(delta<=1e-12,label)
    def bound(desc):
        p=Path(desc['path'])
        check(p.is_absolute() and p.is_file() and not p.is_symlink(),'regular bound file')
        check(sha(p)==desc['sha256'],'file hash '+str(p))
        if 'bytes' in desc:check(p.stat().st_size==desc['bytes'],'file size')
        record['source_and_input_hashes'][str(p)]=desc['sha256']
        return p
    def bj(desc):return read(bound(desc))
    def artifact(root,desc):
        rel=Path(desc['path']);p=(root/rel).resolve()
        check(not rel.is_absolute() and '..' not in rel.parts and p.is_relative_to(root.resolve()),'safe artifact')
        return bound(dict(desc,path=str(p)))
    try:
        check(sha(__file__)==spec['self_sha256'],'review source identity')
        check(sys.flags.no_site==1 and str(Path(sys.executable).absolute())==spec['interpreter'],'qualified interpreter')
        check(all(os.environ.get(k)=='4' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')),'thread envelope')
        ws=bj(spec['worker_spec']);ospec=bj(spec['worker_outer_spec']);outer=bj(spec['worker_outer'])
        completed=bj(spec['worker_receipt']);root=Path(ws['output'])
        check(outer['schema']=='evr0038.authorized-exit-checked-operation.v1' and outer['status']=='completed'
            and outer['exit_code']==0 and outer['termination'] is None and not outer['remaining_descendant_pids']
            and not outer['cleanup_errors'] and outer['wall_seconds']<3600
            and outer['peak_group_rss_kib']<=8388608,'whole scientific outer clean')
        check(outer['spec_sha256']==spec['worker_outer_spec']['sha256']
            and outer['output_sha256']==spec['worker_receipt']['sha256']
            and ospec['worker_spec_sha256']==spec['worker_spec']['sha256']
            and outer['expected_commit']==ospec['commit']==spec['expected_commit'],'complete outer linkage')
        check(completed['status']=='completed' and completed['phase']=='completed' and completed['eligible'] is True
            and completed['spec_sha256']==spec['worker_spec']['sha256']
            and completed['source_sha256']==ws['source_sha256']
            and completed['completed_frames']==446 and completed['completed_sequences']==4
            and completed['all_predictions_sealed_before_gt'] is True and not completed['offline_attempts'],
            'complete producer and pre-GT seal')
        check(completed['resource_exception']==outer['resource_exception']==ws['resource_exception']
            and completed['operation_wall_seconds']==outer['hard_timeout_seconds']==3600,'same authorized envelope')
        for p,h in ws['source_sha256'].items():bound(dict(path=p,sha256=h))
        check(outer['supervisor_sha256']==ws['source_sha256'][str(Path(ws['repo'])/'experiments/0038-late-memory-readout/supervise_authorized.py')],
              'exact authorized supervisor')
        check(outer['monitor_sha256']=='4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd','unchanged monitor')
        for name,digest in outer['log_sha256'].items():bound(dict(path=str(Path(spec['worker_outer']['path']).parent/name),sha256=digest))
        seal=bj(dict(path=str(root/'prediction-seal.json'),sha256=completed['prediction_seal_sha256']))
        prediction=bj(dict(path=str(root/'prediction-receipt.json'),sha256=seal['receipt_sha256']))
        report=bj(dict(path=str(root/'evaluation.json'),sha256=completed['evaluation_sha256']))
        check(seal['complete'] is True and seal['frames']==prediction['frames']==446
            and seal['outputs']==prediction['outputs']==[r['output'] for r in prediction['rows']]
            and prediction['source_sha256']==ws['source_sha256'],'complete archive seal')
        expected=[(n,i) for n,c in COHORT for i in range(c)]
        check([(r['sequence'],r['frame']) for r in prediction['rows']]==expected,'all446 ordered positions')
        check(prediction['model_initial_hash']==prediction['model_final_hash']==completed['model_final_hash'],'unchanged model')
        check(prediction['noninitial_label_decodes']==0 and prediction['initial_label_decodes']==4,'prediction input scope')
        for seq,(name,count) in zip(prediction['sequences'],COHORT):
            check(seq['sequence']==name and seq['frames']==seq['completed_frames']==count
                and seq['status']=='completed' and seq['final_state_exact'] is True
                and seq['final_native_state_sha256']==seq['expected_final_native_state_sha256'],'complete native state')
            side=count-2
            expected_calls={'native':dict(image_encoder=count,mask_decoder=count,memory_encoder=count,memory_attention=count-1,pointer_projection=count),
                'candidate_writes':dict(image_encoder=0,mask_decoder=0,memory_encoder=2*side,memory_attention=0,pointer_projection=2*side),
                'readouts':dict(image_encoder=0,mask_decoder=5*side,memory_encoder=0,memory_attention=8*side,pointer_projection=5*side)}
            for phase,kinds in expected_calls.items():
                for kind,n in kinds.items():
                    check(seq['calls'][phase][kind+'_intent']==seq['calls'][phase][kind+'_completed']==n,'complete logical work')
        oldspec=bj(ws['d35_spec']);d35=bj(ws['d35_result']);prior_receipt=bj(ws['d37_receipt'])
        prior_root=Path(ws['d37_receipt']['path']).parent
        prior_desc=next(x for x in prior_receipt['outputs'] if x['path']=='report.json')
        d37=read(artifact(prior_root,prior_desc))
        annotations={}
        for field in ('initial_acquisition','metric_acquisition'):
            acq=bj(oldspec[field]);base=Path(oldspec[field]['path']).parent
            check(acq['status']=='completed','complete annotation acquisition')
            for row in acq['members']:
                key=(row['sequence'],row['frame_index']);check(key not in annotations,'unique annotation')
                annotations[key]=dict(path=str(base/row['saved_relative']),sha256=row['sha256'],bytes=row['bytes'])
        check(set(annotations)==set(expected),'complete pinned annotation population')
        guard_path=bound(spec['guard'])
        definition=importlib.util.spec_from_file_location('review_offline_guard',guard_path)
        guard=importlib.util.module_from_spec(definition);definition.loader.exec_module(guard)
        guard.install_guard(record['offline_attempts'],record['compatibility_probes'])
        import site
        site.main()
        import numpy as np
        from PIL import Image
        # Independent miniature pixel oracle includes255 false positives and empty union.
        truth=np.asarray([[1,0],[255,0]],dtype=np.uint8);pred=np.asarray([[1,0],[1,0]],dtype=bool)
        check(pixel_counts(truth,pred,np,False)==(1,2,.5),'manufactured background convention')
        check(pixel_counts(truth,pred,np,True)==(1,1,1.),'manufactured ignore convention')
        check(pixel_counts(np.zeros((1,1),np.uint8),np.zeros((1,1),bool),np,False)==(0,0,1.),'manufactured empty union')
        current={key:{x['sequence']:x for x in report[key]['per_sequence']} for key in KEYS}
        old={key:{x['sequence']:x for x in d35[key]['per_sequence']} for key in KEYS}
        prior={key:{x['sequence']:x for x in d37[key]['per_sequence']} for key in KEYS}
        raw={key:{n:{a:[] for a in ARMS} for n,_ in COHORT} for key in KEYS}
        for row in prediction['rows']:
            name,i=row['sequence'],row['frame'];shape=tuple(row['shape']);pixels=math.prod(shape)
            check(all(row['native_parity'][k] is True for k in ('bank_exact','output_exact','input_exact','controls_exact','stored_write_exact')),'all native parities')
            check(row['readouts']['identity_fastpath_used'] is False,'no identity cost shortcut')
            path=artifact(root,row['output'])
            with np.load(path,allow_pickle=False) as archive:
                check(set(archive.files)=={'low_logits','packed_masks','object_ids','shape'},'exact archive keys')
                packed=archive['packed_masks'];low=archive['low_logits']
                check(packed.dtype==np.uint8 and packed.shape==(4,1,(pixels+7)//8)
                    and low.dtype==np.float32 and low.shape==(4,1,256,256) and np.isfinite(low).all()
                    and archive['object_ids'].tolist()==[1] and archive['shape'].tolist()==list(shape),'complete finite archive')
            bits=np.unpackbits(packed,axis=-1,bitorder='big')
            check(not bits[...,pixels:].any(),'zero padding')
            gtpath=bound(annotations[(name,i)])
            with Image.open(gtpath) as image:truth=np.asarray(image)
            check(truth.dtype==np.uint8 and truth.shape==shape,'annotation dimensions')
            record['label_decodes']+=1;counts={}
            for key,policy in zip(KEYS,POLICIES):
                counts[key]={}
                for arm in FIXED:
                    ref=prior[key][name]['methods'][arm]['per_object'][0]['frame_J'][i]
                    if arm!='fixed_negative':number(ref,old[key][name]['methods'][arm]['per_object'][0]['frame_J'][i],'sealed original control')
                    number(current[key][name]['methods'][arm]['per_object'][0]['frame_J'][i],ref,'current control equality')
                    raw[key][name][arm].append(ref)
                for a,arm in enumerate(NEW):
                    mask=bits[a,0,:pixels].reshape(shape).astype(bool)
                    intersect,union,value=pixel_counts(truth,mask,np,policy=='void_ignore')
                    counts[key][arm]=[intersect,union]
                    number(current[key][name]['methods'][arm]['per_object'][0]['frame_J'][i],value,'independent pixel J')
                    raw[key][name][arm].append(value)
            record['rows'].append(dict(sequence=name,frame=i,counts=counts))
            record['reviewed_frames']+=1
        aggregates={};per_sequence={}
        for key,policy in zip(KEYS,POLICIES):
            per_sequence[key]={};j_count=tail_count=0
            for name,count in COHORT:
                interior=list(range(1,count-1));tail=interior[3*len(interior)//4:]
                j_count+=len(interior);tail_count+=len(tail)
                check(current[key][name]['frame_positions']==dict(J=interior,Jtr=tail),'author endpoint positions')
                per_sequence[key][name]={}
                for arm in ARMS:
                    values={metric:math.fsum(raw[key][name][arm][i] for i in positions)/len(positions)
                            for metric,positions in (('J',interior),('Jtr',tail))}
                    per_sequence[key][name][arm]=values
                    for metric,value in values.items():number(current[key][name]['methods'][arm][metric],value,'sequence average')
            check((j_count,tail_count)==(438,110),'complete scoring denominators')
            aggregates[key]={a:{m:math.fsum(per_sequence[key][n][a][m] for n,_ in COHORT)/4
                for m in ('J','Jtr')} for a in ARMS}
            for a in ARMS:
                for m in ('J','Jtr'):number(report[key]['methods'][a][m],aggregates[key][a][m],'equal-clip aggregate')
            methods=aggregates[key]
            clauses={f'over_every_control_3pp_{m}':methods['late_mean'][m]-max(methods[a][m] for a in CONTROLS)>=.03 for m in ('J','Jtr')}
            clauses['three_clips_positive_Jtr_over_union']=sum(per_sequence[key][n]['late_mean']['Jtr']>per_sequence[key][n]['B+M1+M2']['Jtr'] for n,_ in COHORT)>=3
            clauses['no_clip_native_loss_over_0_5pp']=all(per_sequence[key][n]['late_mean'][m]-per_sequence[key][n]['B'][m]>=-.005 for n,_ in COHORT for m in ('J','Jtr'))
            decision=report['continuation' if key=='primary' else 'sensitivity_continuation']
            check(decision['clauses']==clauses and decision['numerical_passed']==all(clauses.values()),'independent unchanged decision')
        check(record['reviewed_frames']==record['label_decodes']==446 and not record['offline_attempts'],'complete offline pixel review')
        for p,h in record['source_and_input_hashes'].items():check(sha(p)==h,'unchanged reviewed input')
        check(sha(__file__)==spec['self_sha256'],'review source unchanged')
        record.update(status='completed',passed=True,aggregates=aggregates,per_sequence=per_sequence,
            numerical_passed=report['continuation']['numerical_passed'])
    except BaseException as exc:
        record.update(status='failed',passed=False,error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        record['wall_seconds']=time.perf_counter()-began
        save(out/'review.json',record)
        save(out/'receipt.json',dict(schema='evr0038.independent-review-receipt.v1',status=record['status'],
            passed=record['passed'],checks=record['checks'],errors=record['errors'],
            spec_sha256=record['spec_sha256'],review_sha256=sha(out/'review.json'),wall_seconds=record['wall_seconds']))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True)
    result=run(p.parse_args().spec)
    print(json.dumps({k:result[k] for k in ('status','passed','reviewed_frames','checks','maximum_difference','wall_seconds')}))
