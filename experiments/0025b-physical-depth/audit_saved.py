"""Independent saved-array arithmetic audit, not independent-team replication.

No generator, predictor, evaluator or geometry-core scientific function is
imported. common.py is used only for source/resource/JSON provenance checks.
GT is decoded only after both clean outer receipts and all saved outputs verify.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time
import zlib

import numpy as np

VIEWS=("rs_x_positive","rs_x_negative","rs_y_positive","rs_y_negative")
ARMS=("rs_row","rs_mid","gs_transport","rs_scale_only")
ATOL=1e-10
GEOMETRY_RTOL=1e-10
FIT_RTOL=1e-8


def need(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def raw_sha(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def local(folder,name):
    p=folder/name
    need(not p.is_symlink() and p.resolve().is_relative_to(folder.resolve()),'unsafe saved path')
    return p


def verify_file(folder,entry):
    p=local(folder,entry['filename'])
    need(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'saved file hash/size differs')
    return p


def cache(folder,binding):
    p=verify_file(folder,binding)
    with np.load(p,allow_pickle=False) as loaded:
        need(set(loaded.files)==set(binding['arrays']),'saved array key set')
        result={key:loaded[key].copy() for key in loaded.files}
    for key,a in result.items():
        need(binding['arrays'][key]==dict(shape=list(a.shape),dtype=str(a.dtype),raw_c_order_sha256=raw_sha(a)),
             'raw array hash/shape/dtype: '+key)
    return result


def compare(observed,expected,label,report,rtol=GEOMETRY_RTOL):
    a,b=np.asarray(observed),np.asarray(expected)
    if a.shape!=b.shape:
        report['failures'].append(label+': shape');return
    if a.dtype.kind=='b' or b.dtype.kind=='b' or (a.dtype.kind in 'iu' and b.dtype.kind in 'iu'):
        if not np.array_equal(a,b):report['failures'].append(label+': exact discrete mismatch')
        return
    if not (np.array_equal(np.isnan(a),np.isnan(b)) and np.array_equal(np.isposinf(a),np.isposinf(b))
            and np.array_equal(np.isneginf(a),np.isneginf(b))):
        report['failures'].append(label+': finite/nonfinite pattern');return
    valid=np.isfinite(a)&np.isfinite(b)
    if not np.any(valid):return
    difference=np.abs(a[valid]-b[valid])
    bound=ATOL+rtol*np.abs(b[valid])
    ratio=float(np.max(difference/bound))
    report['maximum_absolute_difference']=max(report['maximum_absolute_difference'],float(np.max(difference)))
    report['maximum_tolerance_ratio']=max(report['maximum_tolerance_ratio'],ratio)
    if ratio>1.:report['failures'].append(label+': tolerance exceeded')


def geometry(view):
    """Independent Rodrigues matrices on the fixed integer lattice only."""
    i,j=np.indices((64,64),dtype=np.int64)
    u=((2*j*447+63)//126).reshape(-1);v=((2*i*335+63)//126).reshape(-1)
    train=((i+j)%2==0).reshape(-1)
    axis=np.array([1.,0.,0.]) if view.startswith('rs_x') else np.array([0.,1.,0.])
    axis*=1. if view.endswith('positive') else -1.
    x,y,z=axis
    S=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    angles=(v/335.-.5)*math.radians(12.)
    Q=(np.eye(3)[None,:,:]+np.sin(angles)[:,None,None]*S[None,:,:]
       +(1.-np.cos(angles))[:,None,None]*(S@S)[None,:,:])
    rays=np.column_stack(((u-223.5)/481.2,(v-167.5)/481.2,np.ones(4096)))
    middle=np.einsum('nji,nj->ni',Q,rays,optimize=False)
    h=middle[:,2]
    need(np.all(np.isfinite(h)&(h>0)),'independent geometric h')
    normalized=middle[:,:2]/h[:,None]
    source=normalized*np.array([481.2,480.])+np.array([319.5,239.5])
    gs=normalized*481.2+np.array([223.5,167.5])
    return dict(u=u,v=v,train=train,eval=~train,h=h,source_xy=source,canonical_gs_xy=gs)


def neighbors(image,xy):
    height,width=image.shape
    inside=np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<width-1)&(xy[:,1]>=0)&(xy[:,1]<height-1)
    safe=np.where(inside[:,None],xy,0.)
    base=np.floor(safe).astype(int);fraction=safe-base
    x,y=base.T;dx,dy=fraction.T
    values=np.column_stack((image[y,x],image[y,x+1],image[y+1,x],image[y+1,x+1])).astype(np.float64)
    weights=np.column_stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy))
    return values,weights,inside


def reconstruct(truth,prediction,view_index):
    g=geometry(VIEWS[view_index-1]);values,w,inside=neighbors(truth,g['source_xy'])
    _,_,gs_inside=neighbors(np.zeros((336,448)),g['canonical_gs_xy'])
    positive=np.isfinite(values).all(axis=1)&(values>0).all(axis=1)
    spread=np.full(4096,np.nan)
    spread[positive]=(np.max(values[positive],axis=1)-np.min(values[positive],axis=1))/np.min(values[positive],axis=1)
    support=inside&positive&(spread<=.02)&gs_inside
    mid=np.full(4096,np.nan);mid[support]=1./np.sum(w[support]/values[support],axis=1)
    row=mid/g['h']
    dr=prediction['view_%d_depth'%view_index][g['v'],g['u']].astype(float)
    dr[~np.isfinite(dr)]=np.nan
    gv,gw,gi=neighbors(prediction['view_0_depth'],g['canonical_gs_xy'])
    good=gi&np.isfinite(gv).all(axis=1)
    dg=np.full(4096,np.nan);dg[good]=np.sum(gw[good]*gv[good],axis=1)
    designs=dict(rs_row=np.column_stack((dr,np.ones(4096))),rs_mid=np.column_stack((dr/g['h'],1./g['h'])),
                 gs_transport=np.column_stack((dg/g['h'],1./g['h'])),rs_scale_only=dr[:,None])
    counts=dict(total=4096,support=int(support.sum()),train=int((support&g['train']).sum()),
                eval=int((support&g['eval']).sum()),source_positive_four=int((inside&positive).sum()),
                canonical_gs_inbounds=int(gs_inside.sum()))
    available=counts['support']>=2048 and counts['train']>=256 and counts['eval']>=256
    return dict(g,support=support,counts=counts,available=available,source_mid_z=mid,row_z=row,designs=designs)


def closed_fit(X,y):
    """Independent weighted-centered affine formula; SVD is rank-only."""
    if not (np.isfinite(X).all() and np.isfinite(y).all()):return dict(status='unavailable',reason='nonfinite_training')
    norms=np.sqrt(np.sum(X*X,axis=0))
    if not np.all(np.isfinite(norms)&(norms>0)):return dict(status='unavailable',reason='invalid_norm')
    s=np.linalg.svd(X/norms,compute_uv=False)
    threshold=max(X.shape)*np.finfo(float).eps*s[0]
    rank=int((s>threshold).sum())
    details=dict(norms=norms,singular_values=s,rank=rank,threshold=threshold)
    if rank!=X.shape[1]:return dict(details,status='unavailable',reason='rank_deficient')
    a=X.astype(np.longdouble);target=y.astype(np.longdouble)
    if X.shape[1]==1:
        beta=np.array([np.sum(a[:,0]*target)/np.sum(a[:,0]**2)],dtype=float)
    else:
        c=a[:,1];x=a[:,0]/c;t=target/c;w=c*c
        mx=np.sum(w*x)/np.sum(w);my=np.sum(w*t)/np.sum(w)
        slope=np.sum(w*(x-mx)*(t-my))/np.sum(w*(x-mx)**2)
        beta=np.array([slope,my-slope*mx],dtype=float)
    details['coefficients']=beta
    if not np.isfinite(beta).all() or beta[0]<=0:return dict(details,status='unavailable',reason='nonpositive_or_nonfinite_scale')
    predicted=np.sum(a*beta.astype(np.longdouble),axis=1).astype(float)
    if not np.all(np.isfinite(predicted)&(predicted>0)):return dict(details,status='unavailable',reason='invalid_training_prediction')
    return dict(details,status='usable',prediction=predicted)


def distribution(a):
    a=np.asarray(a,dtype=float)
    need(a.ndim==1 and np.isfinite(a).all(),'independent finite distribution')
    if a.size==0:return dict(count=0,mean=None,median=None,quantiles=None)
    ordered=np.sort(a)
    def quantile(q):
        k=(len(ordered)-1)*q;lo=int(math.floor(k));hi=int(math.ceil(k))
        return float((1-(k-lo))*ordered[lo]+(k-lo)*ordered[hi])
    return dict(count=len(a),mean=math.fsum(a.tolist())/len(a),median=quantile(.5),
                quantiles={str(q):quantile(q) for q in (0.,.25,.5,.75,.9,.95,1.)})


def compare_summary(saved,expected,label,report):
    need(saved['count']==expected['count'],label+' count')
    for key in ('mean','median'):
        if expected[key] is None:need(saved[key] is None,label+' empty')
        else:compare(saved[key],expected[key],label+'.'+key,report,FIT_RTOL)
    if expected['quantiles'] is not None:
        need(set(saved['quantiles'])==set(expected['quantiles']),label+' quantile keys')
        for q,value in expected['quantiles'].items():compare(saved['quantiles'][q],value,label+'.q'+q,report,FIT_RTOL)


def inspect_case(case,saved,prediction,truth,report):
    index,view=case['frame_index'],case['view_index'];label='%02d/%s'%(index,VIEWS[view-1])
    r=reconstruct(truth,prediction,view)
    need(case['view']==VIEWS[view-1] and case['support']['counts']==r['counts'],label+' support counts')
    need((case['support']['status']=='usable')==r['available'],label+' support status')
    for key in ('support','h','source_xy','canonical_gs_xy','source_mid_z','row_z'):
        compare(saved[key],r[key],label+'.'+key,report)
    for key in ('u','v','train','eval'):compare(saved['lattice_'+key],r[key],label+'.lattice_'+key,report)
    compare_summary(case['physical_mid_row_separation'],distribution(np.abs(r['source_mid_z'][r['support']]-r['row_z'][r['support']])/r['row_z'][r['support']]),label+'.separation',report)
    means={}
    for arm in ARMS:
        X=r['designs'][arm];row=case['arms'][arm];use=r['support']&r['train'];hold=r['support']&r['eval']
        compare(saved[arm+'_design'],X,label+'.'+arm+'.design',report)
        fit=closed_fit(X[use],r['row_z'][use]) if r['available'] else dict(status='unavailable',reason='support')
        need(row['fit']['training_count']==int(use.sum()),label+' training denominator')
        if (row['fit']['status']=='usable')!=(fit['status']=='usable'):
            report['failures'].append(label+'.'+arm+': independent fit availability');continue
        for key,mapped in (('coefficients','coefficients'),('column_norms','norms'),('singular_values','singular_values'),('rank_threshold','threshold')):
            if key in row['fit'] and mapped in fit:compare(row['fit'][key],fit[mapped],label+'.'+arm+'.'+key,report,FIT_RTOL)
        if 'rank' in row['fit'] and 'rank' in fit:need(row['fit']['rank']==fit['rank'],label+' rank')
        need(row['evaluation']['holdout_count']==int(hold.sum()),label+' heldout denominator')
        compare(saved[arm+'_holdout_indices'],np.flatnonzero(hold),label+'.'+arm+'.indices',report)
        available=fit['status']=='usable' and np.isfinite(X[hold]).all()
        predicted=None
        if available:
            predicted=np.sum(X[hold].astype(np.longdouble)*fit['coefficients'].astype(np.longdouble),axis=1).astype(float)
            available=bool(np.all(np.isfinite(predicted)&(predicted>0)))
        if (row['evaluation']['status']=='usable')!=available:
            report['failures'].append(label+'.'+arm+': independent evaluation availability')
        if available:
            errors=np.abs(predicted-r['row_z'][hold])/r['row_z'][hold]
            for key,value in (('predicted_row_z',predicted),('truth_row_z',r['row_z'][hold]),
                              ('residual_row_z',predicted-r['row_z'][hold]),('absolute_relative_error',errors)):
                compare(saved[arm+'_'+key],value,label+'.'+arm+'.'+key,report,FIT_RTOL)
            stats=distribution(errors);means[arm]=stats['mean']
            compare_summary(row['evaluation']['errors'],stats,label+'.'+arm+'.errors',report)
            compare(row['evaluation']['mean_absrel'],stats['mean'],label+'.'+arm+'.mean',report,FIT_RTOL)
        else:means[arm]=None
    # Model masks cannot modify support; verify their descriptive counts only.
    for n,name in ((0,'gs'),(view,'rs')):
        depth=prediction['view_%d_depth'%n];mask=prediction['view_%d_mask'%n];m=case['raw_model_anomalies'][name]
        need(m['mask_used_to_filter'] is False and m['full_image_model_mask_true']==int(mask.sum()),label+' mask descriptive')
        need(m['full_image_nonfinite']==int((~np.isfinite(depth)).sum()),label+' nonfinite count')
    return dict(frame_index=index,scene=case['scene'],view_index=view,means=means,support_usable=r['available'])


def main_audit(args,report):
    spec=importlib.util.spec_from_file_location('_evr0025b_audit_common',Path(__file__).with_name('common.py'))
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    need(Path(args.repo_root).resolve()==c.ROOT.resolve(),'repo root differs')
    bindings=c.verify_sources(args.expected_commit)
    p,e=Path(args.prediction_dir),Path(args.evaluation_dir)
    po=c.verify_outer(args.prediction_supervision,'predict',args.expected_commit,p,'--output-dir')
    eo=c.verify_outer(args.evaluation_supervision,'evaluate',args.expected_commit,e,'--output-dir')
    flags=c.command_flags(eo['command'])
    need(c.expand(flags['--prediction-dir'])==p.resolve() and c.expand(flags['--prediction-supervision'])==Path(args.prediction_supervision).resolve(),'evaluation input command')
    pr,er=c.read_json(p/'receipt.json'),c.read_json(e/'receipt.json')
    need(pr['status']=='completed' and pr['eligible'] and er['status']=='completed','complete inner receipts')
    need(pr['expected_commit']==er['expected_commit']==args.expected_commit,'inner source identity')
    need(pr['depth_forwards_attempted']==pr['depth_forwards_completed']==80 and len(pr['frames'])==16,'complete predictions')
    need(er['frame_count']==16 and er['case_count']==64 and er['source_poses_read'] is False,'complete evaluation')
    need(er['gt_decode_attempted']==er['gt_decode_completed']==16 and er['gt_value_access_attempted'] is True,'complete GT decode accounting')
    need(pr['gt_values_decoded'] is False and er['gt_values_decoded'] is True,'GT phase declaration')
    need(all(pr[k]==0 for k in ('feature_forwards','matcher_forwards','native_calls')) and pr['offline_attempts']==[],'prediction operation scope')
    c.verify_source_bytes(pr['all_source_sha256']);c.verify_source_bytes(er['source_sha256'],bindings)
    need(er['prediction_bindings']['receipt_sha256']==sha(p/'receipt.json') and er['prediction_bindings']['seal_sha256']==sha(p/'seal.json')
         and er['prediction_bindings']['supervision_sha256']==sha(args.prediction_supervision),'evaluation prediction binding')
    need(er['prediction_bindings']['supervision_log_sha256']=={name:po[name+'_sha256'] for name in ('stdout','stderr')},
         'evaluation prediction log binding')
    need(sha(p/'seal.json')==pr['seal_sha256'] and sha(e/'result-seal.json')==er['result_seal_sha256'],'seal hashes')
    ps,es=c.read_json(p/'seal.json'),c.read_json(e/'result-seal.json')
    expected_p={'selection.json','rgb-inputs.json'}|{'frame-%02d.%s'%(i,x) for i in range(16) for x in ('json','npz')}
    expected_e={'pre-GT-barrier.json','GT-inputs.json','aggregate.json'}|{'truth-%02d.json'%i for i in range(16)}|{
        'frame-%02d-%s.%s'%(i,v,x) for i in range(16) for v in VIEWS for x in ('json','npz')}
    for folder,seal,names in ((p,ps,expected_p),(e,es,expected_e)):
        need(seal['expected_commit']==args.expected_commit and len(seal['outputs'])==len(names)
             and {x['filename'] for x in seal['outputs']}==names,'complete output seal')
        for entry in seal['outputs']:verify_file(folder,entry)
    need(ps['source_sha256']==pr['all_source_sha256'] and ps['acquisition_bindings']==pr['acquisition_bindings']==er['acquisition_bindings'],'source/acquisition chain')
    need(ps['frontend_bindings']==pr['frontend_bindings'],'frontend chain')
    selected=c.read_json(p/'selection.json')['selected'];need([x['index'] for x in selected]==list(range(16)),'selected cohort')
    rgb=c.read_json(p/'rgb-inputs.json');need(len(rgb['images'])==16 and rgb['all_hashes_recorded_before_pixel_decode'] is True,'pre-RGB inventory')
    predictions=[];cases=[]
    for i,row in enumerate(selected):
        need(row['scene']==('living_room' if i<8 else 'office'),'scene allocation')
        frame=c.read_json(p/('frame-%02d.json'%i));need(frame==pr['frames'][i] and frame['status']=='completed','frame receipt binding')
        need(all(frame[k]==v for k,v in row.items()),'frame identity')
        verify_file(p,rgb['images'][i]);a=cache(p,frame['cache'])
        need(set(a)=={'view_%d_%s'%(v,k) for v in range(5) for k in ('depth','mask','intrinsics')},'raw prediction keys')
        need(len(frame['views'])==5 and [v['name'] for v in frame['views']]==['gs',*VIEWS]
             and all(v['status']=='completed' for v in frame['views']),'complete ordered views')
        expected_k=np.array([[481.2/448,0.,.5],[0.,481.2/336,.5],[0.,0.,1.]])
        for v in range(5):
            depth,mask,K=[a['view_%d_%s'%(v,k)] for k in ('depth','mask','intrinsics')]
            need(depth.shape==mask.shape==(336,448) and depth.dtype==np.float32 and mask.dtype==np.bool_,'prediction shapes/dtypes')
            need(K.shape==(3,3) and np.isfinite(K).all() and np.all(np.abs(K-expected_k)<=1e-5*np.abs(expected_k)),'known returned K')
        predictions.append(a)
        for v,name in enumerate(VIEWS,1):
            case=c.read_json(e/('frame-%02d-%s.json'%(i,name)))
            need(case['frame_index']==i and case['view_index']==v and case['scene']==row['scene'] and set(case['arms'])==set(ARMS),'complete case identity/arms')
            cases.append((case,cache(e,case['arrays'])))
    gt=c.read_json(e/'GT-inputs.json');barrier=c.read_json(e/'pre-GT-barrier.json')
    need(sha(e/'GT-inputs.json')==er['GT_inputs_sha256'] and sha(e/'pre-GT-barrier.json')==er['pre_GT_barrier_sha256'],'GT barrier hashes')
    need(gt['all_hashes_and_IHDRs_recorded_before_numeric_decode'] is True and gt['gt_values_decoded'] is False
         and barrier['gt_values_decoded'] is False and barrier['verified_frames']==16 and barrier['verified_forwards']==80,'pre-GT barrier declarations')
    need(barrier['prediction_bindings']==er['prediction_bindings'] and barrier['source_sha256']==er['source_sha256'],'pre-GT barrier bindings')
    need([x['index'] for x in gt['images']]==list(range(16)),'GT population')
    for i,item in enumerate(gt['images']):
        path=verify_file(e,item);need(item['member']==selected[i]['depth_member']['name'],'selected GT member')
        header=path.read_bytes()[:33]
        need(len(header)==33 and header[:16]==b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR','GT PNG header')
        need(zlib.crc32(header[12:29])&0xffffffff==int.from_bytes(header[29:33],'big'),'GT header CRC')
        w,h,bits,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',header[16:29])
        need((w,h,bits,color,compression,filtering)==(640,480,16,0,0,0) and interlace in (0,1),'GT PNG format')
        need(sha(path)==item['sha256'] and hashlib.sha256(header).hexdigest()==item['png']['ihdr_sha256'],'GT preflight hash')
    report['pre_GT_complete_barrier_verified']=True
    from PIL import Image
    independently=[]
    for i,item in enumerate(gt['images']):
        with Image.open(local(e,item['filename'])) as im:
            need(im.format=='PNG' and im.size==(640,480) and im.getexif().get(274,1)==1,'GT decode identity')
            stored=np.asarray(im).copy()
        need(stored.shape==(480,640) and stored.dtype.kind in 'iu' and np.all((stored>=0)&(stored<=65535)),'GT integer decode')
        metadata=c.read_json(e/('truth-%02d.json'%i));need(raw_sha(stored.astype(np.uint16))==metadata['decoded_uint16_sha256'],'GT decoded hash')
        need(all(metadata[k]==v for k,v in selected[i].items()) and metadata['encoded_sha256']==item['sha256']
             and metadata['member']==item['member'] and metadata['source_poses_read'] is False,'GT selected identity')
        truth=stored.astype(np.float64)/5000.
        for case,arrays in cases[4*i:4*i+4]:independently.append(inspect_case(case,arrays,predictions[i],truth,report))
    saved=c.read_json(e/'aggregate.json');need(sha(e/'aggregate.json')==er['aggregate_sha256'],'aggregate binding')
    complete=all(x['support_usable'] and all(x['means'].get(a) is not None for a in ('rs_row','gs_transport')) for x in independently)
    need(saved['primary_complete']==complete and saved['primary_status']==('available' if complete else 'unavailable'),'primary complete denominator')
    need(saved['case_count']==64 and len(saved['scenes'])==2,'aggregate coverage')
    scene_rows=[];gate=complete
    for n,name in enumerate(('living_room','office')):
        rows=[r for r in independently if r['scene']==name];s=saved['scenes'][n]
        need(len(rows)==32 and s['scene']==name and s['case_count']==32,'scene population')
        for arm in ARMS:
            vals=[r['means'][arm] for r in rows if r['means'].get(arm) is not None]
            desc=s['descriptive_arms'][arm]
            need(desc['coverage']==len(vals) and desc['fixed_denominator']==32,'descriptive coverage')
            compare(desc['values'],vals,name+'.'+arm+'.ordered_values',report,FIT_RTOL)
            compare_summary(desc['distribution'],distribution(vals),name+'.'+arm,report)
        if complete:
            errors=distribution([r['means']['rs_row'] for r in rows])
            deltas=distribution([r['means']['rs_row']-r['means']['gs_transport'] for r in rows])
            compare_summary(s['primary_metrics']['rs_row'],errors,name+'.primary',report)
            compare_summary(s['primary_metrics']['paired_rs_minus_gs'],deltas,name+'.paired',report)
            absolute,excess=errors['median']<=.10,deltas['median']<=.02
            need(s['absolute_error_gate']==absolute and s['paired_excess_gate']==excess,'exact scene gate')
            gate=gate and absolute and excess
            scene_rows.append(dict(scene=name,median_rs=errors['median'],median_paired_excess=deltas['median']))
        else:need(s['primary_metrics'] is None and not s['absolute_error_gate'] and not s['paired_excess_gate'],'no partial primary')
    for arm in ARMS:need(saved['arm_availability'][arm]==sum(r['means'].get(arm) is not None for r in independently),'all64 arm coverage')
    need(saved['gate_passed']==er['gate_passed']==bool(gate),'aggregate gate')
    need(not saved['pose_gain_established'] and not saved['manuscript_admitted'],'interpretation scope')
    c.verify_source_bytes(bindings)
    c.verify_outer(args.prediction_supervision,'predict',args.expected_commit,p,'--output-dir')
    c.verify_outer(args.evaluation_supervision,'evaluate',args.expected_commit,e,'--output-dir')
    report.update(case_count=64,arm_records=256,independent_primary_complete=complete,independent_gate_passed=bool(gate),scenes=scene_rows,
        bindings=dict(prediction_receipt=sha(p/'receipt.json'),evaluation_receipt=sha(e/'receipt.json'),
                      prediction_outer=sha(args.prediction_supervision),evaluation_outer=sha(args.evaluation_supervision)),
        method='Rodrigues lattice geometry + inverse-depth GT + longdouble centered affine OLS; SVD only for rank. Metadata/resource helpers shared, scientific functions independent.')


def functional_self_check():
    """Tiny manufactured formula check; never reads a dataset or saved study."""
    x=np.linspace(-3.,2.,13);c=np.linspace(.7,1.3,13);X=np.column_stack((c*x,c));y=c*(1.7*x+6.)
    f=closed_fit(X,y);need(f['status']=='usable','centered fit self-check')
    need(np.allclose(f['coefficients'],[1.7,6.],rtol=1e-13,atol=1e-13),'weighted centered coefficients')
    s=closed_fit(x[:,None],2.*x);need(s['status']=='unavailable','nonpositive physical depth rejected')
    g=geometry('rs_x_positive');source=np.ones((480,640))*4.
    prediction={'view_1_depth':np.ones((336,448))*4.,'view_0_depth':np.ones((336,448))*4.}
    r=reconstruct(source,prediction,1)
    need(np.allclose(r['row_z'][r['support']],4./r['h'][r['support']]),'inverse depth and h direction')
    need(distribution([0.,1.,2.,3.])['median']==1.5,'independent quantile interpolation')
    return dict(status='passed',scope='four compact manufactured arithmetic checks only')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check',action='store_true')
    for name in ('prediction-dir','evaluation-dir','prediction-supervision','evaluation-supervision','repo-root','output-dir'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--expected-commit')
    args=p.parse_args()
    if args.self_check:
        print(json.dumps(functional_self_check()));return
    need(all(getattr(args,k.replace('-','_')) is not None for k in
        ('prediction-dir','evaluation-dir','prediction-supervision','evaluation-supervision','repo-root','output-dir','expected-commit')),'all audit inputs required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    report=dict(schema='evr0025b.independent-saved-audit.v1',status='started',failures=[],maximum_absolute_difference=0.,maximum_tolerance_ratio=0.,
        pre_GT_complete_barrier_verified=False,source_sha256=sha(Path(__file__)),expected_commit=args.expected_commit,
        tolerances=dict(atol=ATOL,geometry_rtol=GEOMETRY_RTOL,fit_rtol=FIT_RTOL),
        no_original_model_or_fitting_pipeline_rerun=True,independent_closed_form_fits_recomputed=True)
    try:
        main_audit(args,report)
        report['status']='passed' if not report['failures'] else 'failed'
    except BaseException as error:
        report.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        report['wall_seconds']=time.perf_counter()-start
        with (args.output_dir/'audit.json').open('x') as out:json.dump(report,out,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','case_count','arm_records','independent_gate_passed','maximum_tolerance_ratio','wall_seconds')}))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':
    main()
