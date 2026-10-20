"""Independent saved-only Stage A reconstruction. No core/fixture/run imports.

Uses the supervisor only for frozen-source proof. Scientific geometry is rebuilt
with matrix power series (absolute 1e-10 check). Verified saved matrices are
factored with NumPy SVD for numerical checks; this shares the original numerical
backend and is not an algorithmically independent factorization. Alternate QR
parameter differences are descriptive because weak directions amplify roundoff.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ATOL=RTOL=1e-8
METRIC_TOL=1e-10
GEOMETRY_TOL=1e-10
POINTS=np.array([[-1.8,-1.2,3.2],[1.5,-1.1,4.5],[-1.3,.9,2.8],[1.9,1.2,5.2],
                 [.2,-.6,2.3],[-.4,1.6,4.1],[.8,.4,3.7],[-1.6,.2,5.5]])
K=np.array([[620.,0.,639.5],[0.,600.,479.5],[0.,0.,1.]])
ARMS=('D4','D8','N8','A8')


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def compare(a,b,label,report,metric=False,geometry=False):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    require(a.shape==b.shape,label+': shape')
    require(np.isfinite(a).all() and np.isfinite(b).all(),label+': nonfinite')
    if not a.size:return
    d=np.abs(a-b);tol=GEOMETRY_TOL if geometry else METRIC_TOL if metric else ATOL+RTOL*np.abs(b)
    ratio=float(np.max(d/tol));report['max_absolute_difference']=max(report['max_absolute_difference'],float(d.max()))
    report['max_tolerance_ratio']=max(report['max_tolerance_ratio'],ratio)
    if ratio>1:report['failures'].append(label)


def angular(t,truth):
    a,b=np.asarray(t,dtype=float),np.asarray(truth,dtype=float)
    a=a/math.hypot(*a);b=b/math.hypot(*b)
    # Independently implemented stable directed angle, including both endpoints.
    return math.degrees(math.atan2(math.hypot(*np.cross(a,b)),float(np.dot(a,b))))


def rotation_integral(w,s):
    """Fixed 32-term matrix series; declared manufactured angles are <1 rad."""
    x,y,z=np.asarray(w,dtype=float);W=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    require(abs(s)*math.hypot(x,y,z)<1.,'outside declared series domain')
    Q=np.eye(3);J=s*np.eye(3);term=np.eye(3)
    for k in range(1,33):
        term=term@(s*W)/k;Q+=term;J+=s*term/(k+1)
    return Q,J


def form(data,ids,arm):
    R=np.asarray(data['R']);rb=[];Bs=[]
    for k in (1,2):
        rr=[];bb=[]
        for ray,s in zip(data['r%d'%k],data['s%d'%k]):
            Q,J=rotation_integral(data['w%d'%k],s);rr.append(Q.T@ray);bb.append(Q.T@J)
        rb.append(np.array(rr));Bs.append(np.array(bb))
    a,b=rb;B,D=Bs;c=a@R.T;RB=np.einsum('ab,nbc->nac',R,B,optimize=False)
    mat=[];rhs=[]
    for i in ids:
        if arm=='N8':
            n=np.cross(b[i],c[i]);mat.append(np.r_[n,-n@RB[i],n@D[i]])
        else:
            columns=[-np.eye(3)]
            if arm!='A8':columns += [RB[i],-D[i]]
            columns += [-c[i,:,None],float(data['d2'][i])*b[i,:,None],b[i,:,None]]
            mat.append(np.column_stack(columns));rhs.extend(float(data['d1'][i])*c[i])
    return np.array(mat).reshape(-1,9 if arm=='N8' else 6 if arm=='A8' else 12),None if arm=='N8' else np.array(rhs),(c,b,RB,D)


def decomposition(A,y):
    norms=np.array([math.hypot(*v) for v in A.T]);scales=np.ones(A.shape[1]);np.divide(1,norms,out=scales,where=norms>0)
    eq=A*scales;u,s,vt=np.linalg.svd(eq,full_matrices=True)
    rtol=max(A.shape)*np.finfo(float).eps;rank=int((s>rtol*s[0]).sum());weak=int((s>1e-8*s[0]).sum())
    alternate=None
    if y is None:gamma=vt[-1]
    elif rank:
        # Same LAPACK/NumPy backend is deliberately disclosed; geometry above is independent.
        uy=np.einsum('ij,i->j',u[:,:rank],y,optimize=False)
        gamma=np.einsum('ji,j->i',vt[:rank],uy/s[:rank],optimize=False)
        if rank==A.shape[1]:
            q,r=np.linalg.qr(eq,mode='reduced');alternate=scales*np.linalg.solve(r,q.T@y)
    else:gamma=np.zeros(A.shape[1])
    return dict(norms=norms,scales=scales,s=s,vt=vt,rank=rank,weak=weak,rtol=rtol,x=scales*gamma,alternate=alternate)


def null_check(saved,dec,t,label,report):
    tn=math.hypot(*t);compare(saved['translation_norm'],tn,label+'.t_norm',report)
    for name,rank in [('numerical',dec['rank']),('weak',dec['weak'])]:
        item=saved[name];N=np.asarray(item['basis_scaled'],dtype=float)
        own=dec['vt'][rank:].T;require(item['nullity']==own.shape[1],label+' nullity')
        compare(N@N.T,own@own.T,label+'.'+name+'.projector',report)
        physical=dec['scales'][:,None]*N
        compare(item['basis_physical'],physical,label+'.physical',report)
        if tn>0:
            unit=np.array(t)/tn;functional=(np.eye(3)-np.outer(unit,unit))@physical[:3]/tn
            compare(item['translation_direction_map'],functional,label+'.functional',report)
            compare(item['translation_direction_frobenius'],math.hypot(*functional.ravel()),label+'.functional_norm',report)
        else:require(item['translation_direction_map'] is None and item['translation_direction_frobenius'] is None,label+' undefined functional')


def truth_null_check(s,record,truth,label,report):
    diagnostic=s.get('clean_truth_task_null_diagnostic')
    if record['regime']=='clean':
        require(diagnostic is not None,label+' missing clean truth-null diagnostic')
        tt=np.asarray(truth['t']);nt=math.hypot(*tt);P=(np.eye(3)-np.outer(tt/nt,tt/nt))/nt
        for kind in ('numerical','weak'):
            B=np.asarray(s['task_null'][kind]['basis_physical']);mapping=P@B[:3]
            compare(diagnostic[kind]['map'],mapping,label+'.truth_null_map',report)
            compare(diagnostic[kind]['frobenius'],math.hypot(*mapping.ravel()),label+'.truth_null_norm',report)
    else:require(diagnostic is None,label+' unexpected truth-null use')


def sign_checks(saved,g,x,label,report):
    c,b,RB,D=g;counts=[];all_regular=True
    for index,sign in enumerate((1,-1)):
        h=saved['hypotheses'][index];require(h['sign']==sign and len(h['points'])==8,label+' sign order')
        good=0;sign_regular=True
        for i in range(8):
            M=np.column_stack((c[i],-b[i]));k=sign*(x[:3]-RB[i]@x[3:6]+D[i]@x[6:9])
            dec=decomposition(M,-k);z=dec['x'];p=h['points'][i]
            require(p['rank']==dec['rank'],label+' triangulation rank')
            compare([p['depth1'],p['depth2']],z,label+'.triangulated_depth',report)
            compare(p['singular_values'],dec['s'],label+'.triangulation_spectrum',report)
            compare(p['residual_norm'],math.hypot(*(M@z+k)),label+'.triangulation_residual',report)
            positive=bool(dec['rank']==2 and np.all(z>0));good+=positive
            require(p['positive']==positive,label+' triangulation positivity')
            regular=bool(dec['rank']==2 and np.isfinite(z).all());sign_regular &= regular
            require(p['status']==('valid' if regular else 'rank_deficient'),label+' triangulation status')
        require(h['positive_count']==good and h['all_triangulations_valid']==sign_regular,label+' sign counts');counts.append(good)
        all_regular &= sign_regular
    admitted=[s for s,count in zip((1,-1),counts) if count>=5]
    chosen=admitted[0] if all_regular and len(admitted)==1 else None
    require(saved['selected_sign']==chosen and saved['required_positive']==5 and saved['valid']==(chosen is not None) and saved['all_triangulations_valid']==all_regular,label+' cheirality rule')
    return chosen


def solve_check(data,record,truth,label,report):
    arm=record['arm'];ids=[int(x) for x in record['indices'].split('-')];s=record['solver']
    A,y,g=form(data,ids,arm);compare(s['A'],A,label+'.matrix',report,geometry=True)
    if y is None:require(s['y'] is None,label+' homogeneous RHS')
    else:compare(s['y'],y,label+'.RHS',report,geometry=True)
    require(s['indices']==ids and s['pool_size']==8,label+' indices')
    # Geometry is checked first; factor original saved bytes to isolate conditioning.
    A=np.asarray(s['A'],dtype=float);y=None if y is None else np.asarray(s['y'],dtype=float)
    d=decomposition(A,y)
    require(not str(s.get('reason','')).startswith('numerical_error:'),label+' original numerical failure cannot be independently certified')
    for key,value in [('column_norms',d['norms']),('column_scales',d['scales']),('singular_values',d['s']),('rank_rtol',d['rtol']),('weak_rtol',1e-8)]:compare(s[key],value,label+'.'+key,report)
    require(s['rank']==d['rank'] and s['weak_rank']==d['weak'] and s['right_vectors_shape']==[A.shape[1]]*2 and s['full_matrices'] is True,label+' ranks/full Vt')
    V=np.asarray(s['Vt']);compare(V@V.T,np.eye(A.shape[1]),label+'.orthogonal_V',report)
    # Compare projectors, not arbitrary signs/bases inside repeated singular spaces.
    raw=np.asarray(s.get('unnormalized_solution',s['solution']),dtype=float)
    if arm=='N8':
        tn=math.hypot(*raw[:3])
        if tn==0:
            require(s['status']==record['status']=='unavailable' and s['reason']==record['reason']=='undefined_translation' and s['parameters'] is None and record['raw_angle_degrees'] is None and record['error_degrees']==180.,label+' undefined translation')
            null_check(s['task_null'],d,raw[:3],label,report)
            truth_null_check(s,record,truth,label,report)
            return dict(case=record['case'],regime=record['regime'],arm=arm,status='unavailable',error=180.)
        if d['rank']==8:
            own=d['x'];own*=1 if np.dot(own,raw)>=0 else -1
            compare(raw,own,label+'.homogeneous_solution',report)
        else:
            gamma=raw/d['scales'];N=d['vt'][d['rank']:].T
            compare(gamma,N@(N.T@gamma),label+'.nonunique_null_solution',report)
        x=raw/tn;sign=sign_checks(s['cheirality'],g,x,label,report)
        if sign is not None:x*=sign
        compare(s['solution'],x,label+'.signed_solution',report)
        reason='rank_not_eight' if d['rank']!=8 else 'unresolved_cheirality_sign' if sign is None else None
        params=dict(t=x[:3],v1=x[3:6],v2=x[6:9],b1=None,a2=None,b2=None)
    else:
        x=np.asarray(s['solution']);compare(x,d['x'],label+'.saved_matrix_SVD_solution',report)
        if d['alternate'] is not None:
            difference=float(np.max(np.abs(x-d['alternate'])))
            report['alternate_QR_max_parameter_difference']=max(report.get('alternate_QR_max_parameter_difference',0.),difference)
        off=3 if arm=='A8' else 9;v1=np.zeros(3) if arm=='A8' else x[3:6];v2=np.zeros(3) if arm=='A8' else x[6:9]
        params=dict(t=x[:3],v1=v1,v2=v2,b1=x[off],a2=x[off+1],b2=x[off+2])
        z1=np.asarray(data['d1'])+params['b1'];z2=params['a2']*np.asarray(data['d2'])+params['b2']
        ch=s['cheirality'];positive=(z1>0)&(z2>0);valid=bool(params['a2']>0 and positive.all())
        compare(ch['depth1'],z1,label+'.z1',report);compare(ch['depth2'],z2,label+'.z2',report)
        require(ch['pool_size']==8 and ch['finite_count']==8 and ch['positive_count']==int(positive.sum()) and ch['scale_positive']==bool(params['a2']>0) and ch['valid']==valid and ch['fitting_count']==len(ids) and ch['fitting_finite_count']==len(ids) and ch['fitting_positive_count']==int(positive[ids].sum()),label+' affine positivity')
        reason='rank_deficient' if d['rank']!=A.shape[1] else 'undefined_translation' if math.hypot(*x[:3])==0 else 'nonpositive_affine_scale_or_pool_depth' if not valid else None
    for k,v in params.items():
        if v is None:require(s['parameters'][k] is None,label+' no affine input N8')
        else:compare(s['parameters'][k],v,label+'.parameter_'+k,report)
    require(s['status']==record['status']==('valid' if reason is None else 'unavailable') and s['reason']==record['reason']==reason,label+' validity')
    residual=A@x-(0 if y is None else y);rr=s['residuals']
    compare(rr['vector'],residual,label+'.residual',report);compare(rr['normal_equation_vector'],A.T@residual,label+'.normal_equation',report)
    compare(rr['norm'],math.hypot(*residual),label+'.residual_norm',report)
    null_check(s['task_null'],d,raw[:3],label,report)
    truth_null_check(s,record,truth,label,report)
    raw_angle=angular(x[:3],truth['t']) if math.hypot(*x[:3])>0 else None
    if raw_angle is not None:compare(record['raw_angle_degrees'],raw_angle,label+'.angle',report)
    else:require(record['raw_angle_degrees'] is None,label+' undefined angle')
    error=raw_angle if reason is None else 180.
    compare(record['error_degrees'],error,label+'.reported_error',report)
    return dict(case=record['case'],regime=record['regime'],arm=arm,status=s['status'],error=error)


def input_check(item,reference,rng,report):
    label=item['case']+'/'+item['regime']+'/'+str(item['realization']);clean=item['clean'];truth=item['truth'];data=item['data']
    require(item['source_sha256']=='0b058405fbc151b8765dafdb4e4a81f8c7e48f515da3d217b2df33b9a9565683' and item['source_relative']=='experiments/0025-rs-identifiability/check_manufactured.py',label+' generator source')
    compare(clean['points'],POINTS,label+'.fixed_points',report);compare(clean['K'],K,label+'.K',report)
    require(clean['width']==1280 and clean['height']==960,label+' sensor size')
    Rg,_=rotation_integral([.20,-.13,.11],1.);w1=np.array([2.1,-1.7,.9]);w2=np.array([-1.4,.75,1.8]);v1=np.array([.8,-.35,.42]);v2=np.array([-.55,.44,.2]);t=np.array([.42,-.16,.25]);R=Rg
    if item['case']=='N':R=rotation_integral([.02,-.013,.011],1.)[0]
    if item['case']=='C':R,J=rotation_integral(w1,.12);t=J@v1;w2=w1;v2=v1
    if item['case']=='E':R=np.eye(3);t=np.array([.3,0.,0.]);w1=w2=v1=v2=np.zeros(3)
    for k,v in dict(R=R,t=t,w1=w1,w2=w2,v1=v1,v2=v2,b1=.18,a1=1.,a2=1.17,b2=-.09).items():compare(truth[k],v,label+'.truth_'+k,report)
    pix=np.asarray(clean['pixels']);times=np.asarray(clean['times']);z=np.asarray(clean['axial_depths']);rays=np.asarray(clean['rays']);raw=np.asarray(clean['raw_depths'])
    compare(clean['readouts'],[.04,.04],label+'.readouts',report)
    compare(times,.04*(pix[:,:,1]-479.5)/959.,label+'.implicit_row',report)
    compare(raw,(z-np.array([.18,-.09])[:,None])/np.array([1.,1.17])[:,None],label+'.raw_depth',report)
    for camera,(rotation,translation,w,v) in enumerate([(np.eye(3),np.zeros(3),w1,v1),(R,t,w2,v2)]):
        for i in range(8):
            Q,J=rotation_integral(w,times[camera,i]);point=Q@(rotation@POINTS[i]+translation)+J@v
            compare(point,z[camera,i]*rays[camera,i],label+'.physical_projection',report)
            compare(pix[camera,i],(K@rays[camera,i])[:2],label+'.pixel_projection',report)
    require(np.all(z>0) and np.all((pix[:,:,0]>=0)&(pix[:,:,0]<=1279)&(pix[:,:,1]>=0)&(pix[:,:,1]<=959)),label+' physical visibility')
    if item['regime']=='clean':
        require(item['draws'] is None,label+' clean draws');observed=pix;observed_depth=raw;inputR=R;inputw=[w1,w2]
    else:
        draws=item['draws'];uv=(pix+.5)/np.array([1280,960]);pattern=np.sin(2*np.pi*uv[:,:,0])+np.cos(2*np.pi*uv[:,:,1]);pattern-=pattern.mean(axis=1,keepdims=True);pattern/=np.sqrt(np.mean(pattern**2,axis=1,keepdims=True))
        if item['regime']=='P':
            require(draws['rng_state_before']==rng.bit_generator.state,label+' draw order')
            generated=(rng.standard_normal((2,8,2)),rng.standard_normal(2),rng.standard_normal((2,8)),rng.standard_normal((3,3)))
            for k,v in zip(('pixel_standard_normal','coherent_standard_normal','epsilon_standard_normal','angular_axes_standard_normal'),generated):require(np.array_equal(draws[k],v),label+' exact seeded '+k)
            require(draws['rng_state_after']==rng.bit_generator.state,label+' RNG after')
            reference[(item['case'],item['realization'])]=draws
        else:require(draws==reference[(item['case'],item['realization'])],label+' P/S identical draws')
        observed=pix+.5*np.asarray(draws['pixel_standard_normal']);axes=np.asarray(draws['angular_axes_standard_normal']);axes/=np.linalg.norm(axes,axis=1)[:,None]
        relative=.03*(pattern*np.asarray(draws['coherent_standard_normal'])[:,None]+np.asarray(draws['epsilon_standard_normal']))/math.sqrt(2.)
        observed_depth=raw+z/np.array([1.,1.17])[:,None]*relative
        compare(draws['relative_depth_error'],relative,label+'.depth_error',report)
        if item['regime']=='P':inputR=R;inputw=[w1,w2]
        else:inputR=rotation_integral(math.radians(.1)*axes[0],1.)[0]@R;inputw=[w1+math.radians(.5)*axes[1],w2+math.radians(.5)*axes[2]]
    compare(item['observed_pixels'],observed,label+'.observed_pixels',report)
    for i in range(2):
        rr=np.column_stack(((observed[i,:,0]-639.5)/620.,(observed[i,:,1]-479.5)/600.,np.ones(8)))
        for k,v in [('r',rr),('s',.04*(observed[i,:,1]-479.5)/959.),('d',observed_depth[i]),('w',inputw[i])]:compare(data[k+str(i+1)],v,label+'.input_'+k+str(i+1),report)
    compare(data['R'],inputR,label+'.input_R',report)


def group_metrics(rows):
    values=np.array([r['error'] for r in rows]);valid=np.array([r['status']=='valid' for r in rows]);use=valid&(values<=5.)
    return dict(attempted=len(rows),available=int(valid.sum()),useful_count=int(use.sum()),useful_yield=float(use.mean()),
                median_error_degrees=float(np.median(values)),p90_error_degrees=float(np.quantile(values,.9)),maximum_error_degrees=float(values.max()),
                empirical_cdf_integral_auc5_percent=float(100*np.maximum(0,1-values/5.).mean()),
                recall={str(q):float(np.mean(valid&(values<=q))) for q in (1,2,5,10,20,90,180)})


def audit(args,report):
    import supervise
    bindings=supervise.verify_frozen(args.expected_commit,[HERE/n for n in supervise.OWNED]+[supervise.PROTOCOL,supervise.LEGACY])
    outer=load(args.outer_receipt);folder=Path(args.input_dir)
    require(outer['schema']=='evr0025c.supervision.v1' and outer['operation']=='noise' and outer['expected_commit']==args.expected_commit and outer['source_sha256']==bindings,'outer source identity')
    require(outer['status']=='completed' and outer['exit_code']==0 and outer['termination'] is None and outer['remaining_descendant_pids']==[] and outer['cleanup_errors']==[],'clean outer required')
    require(outer['hard_timeout_seconds']==1800 and 0<=outer['wall_seconds']<1800 and outer['group_rss_limit_kib']==8*1024*1024 and 0<=outer['peak_group_rss_kib']<=8*1024*1024,'outer resource contract')
    require(outer['thread_environment']=={k:'4' for k in supervise.THREAD_KEYS},'outer thread contract')
    for name in ('stdout','stderr'):require(outer[name+'_sha256']==digest(Path(args.outer_receipt).parent/(name+'.log')),'outer log hash')
    expected_command=['$PYTHON','$REPOSITORY/experiments/0025c-rs-noise/run.py','--expected-commit',args.expected_commit,'--output-dir']
    require(outer['command'][:5]==expected_command and len(outer['command'])==6,'outer command shape')
    expanded=outer['command'][5].replace('$REPOSITORY',str(ROOT)).replace('$PRIVATE',str(ROOT.parent));require(Path(expanded).resolve()==folder.resolve(),'outer input location')
    receipt=load(folder/'receipt.json');seal=load(folder/'seal.json')
    require(receipt['status']=='completed' and receipt['expected_commit']==seal['expected_commit']==args.expected_commit and receipt['source_sha256']==bindings,'inner seal/source')
    require(receipt['case_attempted']==receipt['case_completed']==132 and receipt['solver_attempted']==receipt['solver_completed']==9636,'complete attempts')
    require(receipt['model_calls']==receipt['real_data_reads']==0 and digest(folder/'seal.json')==receipt['seal_sha256'],'scope/seal hash')
    names={f'case-{i:03d}.json' for i in range(132)}|{'aggregate.json','all-solves.csv','started.json'}
    require(len(seal['records'])==len(names) and {e['path'] for e in seal['records']}==names,'seal coverage')
    for entry in seal['records']:require(digest(folder/entry['path'])==entry['sha256'],'sealed bytes changed')
    schedule=[(case,r,i) for case in ('G','N','C','E') for r,i in [('clean',-1)]+[(r,i) for i in range(16) for r in ('P','S')]]
    rng=np.random.Generator(np.random.PCG64(20261019));reference={};rows=[]
    tasks=[('D4','-'.join(map(str,x))) for x in itertools.combinations(range(8),4)]+[(a,'0-1-2-3-4-5-6-7') for a in ('D8','N8','A8')]
    csvrows=list(csv.DictReader((folder/'all-solves.csv').open()));require(len(csvrows)==9636,'CSV count')
    for i,(case,regime,realization) in enumerate(schedule):
        record=load(folder/f'case-{i:03d}.json');item=record['input']
        require((item['case'],item['regime'],item['realization'])==(case,regime,realization),'cohort order')
        input_check(item,reference,rng,report)
        if regime=='clean':
            A,y,_=form(item['data'],list(range(8)),'D8');tr=item['truth']
            beta=np.r_[tr['t'],tr['v1'],tr['v2'],tr['b1'],tr['a2'],tr['b2']]
            residual=float(np.max(np.abs(A@beta-y)));require(residual<=1e-10,'independent clean affine truth equation')
            compare(record['clean_check']['affine_truth_max_abs_residual'],residual,'clean physical residual',report,geometry=True)
            require(record['clean_check']['pass_gate'] is True,'clean truth flag')
            C,_,_=form(item['data'],list(range(8)),'N8')
            require(np.max(np.abs(C@beta[:9]))<=1e-10,'independent clean depth-free truth equation')
        else:require(record['clean_check'] is None,'no noisy truth admission')
        require([(r['arm'],r['indices']) for r in record['solves']]==tasks,'complete solve order')
        for j,r in enumerate(record['solves']):
            require((r['record_index'],r['case'],r['regime'],r['realization'])==(i,case,regime,realization),'row identity')
            csvrow=csvrows[i*73+j]
            for k,v in csvrow.items():
                require(k in r,'CSV extra field')
                if r[k] is None:require(v=='','CSV null')
                elif isinstance(r[k],(float,int)):require(float(v)==r[k],'CSV numeric')
                else:require(v==r[k],'CSV categorical')
            rows.append(solve_check(item['data'],r,item['truth'],f'{i}/{j}',report))
    result=load(folder/'aggregate.json');groups={}
    for r in rows:groups.setdefault((r['case'],r['regime'],r['arm']),[]).append(r)
    require(len(result['groups'])==len(groups)==48,'summary groups')
    for saved in result['groups']:
        key=tuple(saved[k] for k in ('case','regime','arm'));m=group_metrics(groups[key])
        for k,v in m.items():
            if isinstance(v,dict):
                for q,value in v.items():compare(saved[k][q],value,'metric.'+k+q,report,True)
            else:compare(saved[k],v,'metric.'+k,report,True)
    passed=True
    for saved,(case,regime) in zip(result['decisions'],itertools.product(('G','N','C'),('P','S'))):
        m=group_metrics(groups[(case,regime,'D4')]);ok=m['attempted']==1120 and m['useful_count']>=144
        require(saved==dict(case=case,regime=regime,attempted=m['attempted'],useful_count=m['useful_count'],pass_gate=ok),'gate group')
        passed &= ok
    require(len(result['decisions'])==6 and result['technical_pass'] is True and result['utility_gate_pass']==receipt['utility_gate_pass']==passed,'complete gate')
    require(all(digest(ROOT/path)==h for path,h in bindings.items()),'source changed during audit')
    report.update(case_count=132,solve_count=9636,independent_gate_passed=bool(passed),
                  input_receipt_sha256=digest(folder/'receipt.json'),seal_sha256=digest(folder/'seal.json'),outer_sha256=digest(args.outer_receipt))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('expected-commit','input-dir','output-dir','outer-receipt'):p.add_argument('--'+name,required=True)
    args=p.parse_args();output=Path(args.output_dir).resolve();require(not output.exists() and ROOT not in output.parents and output!=ROOT,'fresh private output')
    output.mkdir(parents=True);began=time.perf_counter()
    report=dict(schema='evr0025c.saved-audit.v1',status='started',failures=[],max_absolute_difference=0.,max_tolerance_ratio=0.,
                expected_commit=args.expected_commit,source_sha256=digest(__file__),tolerances=dict(atol=ATOL,rtol=RTOL,metrics=METRIC_TOL,geometry_absolute=GEOMETRY_TOL),
                scope='Independent saved-input geometry and metric reconstruction; verified saved matrices use the same NumPy SVD numerical backend. Alternate QR discrepancies descriptive. Not independent-team or algorithmically independent factorization replication. No model, original fixture generation or original solver imported.')
    try:
        audit(args,report);report['status']='passed' if not report['failures'] else 'failed'
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        report['wall_seconds']=time.perf_counter()-began
        with (output/'audit.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','case_count','solve_count','independent_gate_passed','max_absolute_difference','max_tolerance_ratio','wall_seconds')}))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
