"""Shared 0016 bindings and explicitly defined final-pose diagnostic scores."""
import hashlib, importlib.util, json, os
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('baseline0013',ROOT/'experiments/0013-pose-headroom/predict_pair.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
sha256=base.sha256
json_safe=base.json_safe
BACKENDS=('old_norm','current_norm','current_ts')
SEEDS=(0,1,2,3)
SALT='evr0016:development:2026-10-19\0'
MANIFEST=ROOT/'configs/0016-development-manifest.json'
LIBRARIES=ROOT/'configs/0016-libraries.json'
PROTOCOL=ROOT/'protocols/0016-search-headroom.md'
SOURCE_FILES=('common.py','allocate.py','predict_pair.py','run_diagnostic.py','evaluate_saved.py')

def write_new(path, value):
    encoded=json.dumps(json_safe(value),sort_keys=True,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x') as f:
        f.write(encoded); f.flush(); os.fsync(f.fileno())

def source_hashes():
    files=[HERE/name for name in SOURCE_FILES]+[PROTOCOL,LIBRARIES,
        ROOT/'experiments/0013-pose-headroom/predict_pair.py',
        ROOT/'experiments/0013-pose-headroom/run_baseline.py',
        ROOT/'experiments/0013-pose-headroom/evaluate_sealed.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}

def load_manifest(path):
    m=json.loads(Path(path).read_text()); keys=m['selected_pair_keys']
    old=json.loads((ROOT/'configs/0012-development-manifest.json').read_text())
    if (m.get('schema')!='evr0016.development.v1' or len(keys)!=64 or len(set(keys))!=64
        or m.get('development_scene')!='0015' or m.get('development_scene_pairs')!=600
        or m.get('coordinate_convention')!='raw_pixels' or m.get('held_out_scenes')!=['0022']
        or m.get('excluded_manifest_sha256')!=sha256(ROOT/'configs/0012-development-manifest.json')
        or not isinstance(m.get('root_name_inventory_sha256'),str)
        or len(m['root_name_inventory_sha256'])!=64
        or any(not k.startswith('0015-') for k in keys)
        or set(keys)&set(old['selected_pair_keys'])
        or m.get('selection_salt')!=SALT or m.get('dataset_sha256')!=old['dataset_sha256']
        or m.get('dataset_bytes')!=old['dataset_bytes']):
        raise ValueError('Invalid 0016 development allocation')
    if keys!=sorted(keys,key=lambda k:hashlib.sha256((SALT+k).encode()).hexdigest()):
        raise ValueError('Allocation order changed')
    return m

def options(backend,seed):
    if backend not in BACKENDS or type(seed) is not int or seed not in SEEDS: raise ValueError('Unknown backend/seed')
    if backend=='old_norm': return {**base.RANSAC_OPTIONS,'seed':seed}
    return {'max_error':1.0,'tangent_sampson':backend=='current_ts','real_focal_check':False,
        'ransac':{'min_iterations':1000,'max_iterations':100000,'dyn_num_trials_mult':3.0,
                  'success_prob':0.9999,'seed':seed,'progressive_sampling':False,
                  'max_prosac_iterations':100000,'score_initial_model':False},'bundle':{}}

def validate_pose(estimate):
    R=base.finite_array(estimate['R'],(3,3),'R'); t=base.finite_array(estimate['t'],(3,),'t')
    scale=float(np.max(np.abs(t)))
    norm=0. if scale==0 else float(np.linalg.norm(t/scale))
    if (not np.allclose(np.einsum('ij,kj->ik',R,R,optimize=False),np.eye(3),rtol=0,atol=1e-6)
            or abs(np.linalg.det(R)-1)>1e-6 or norm==0 or not np.isfinite(norm)):
        raise ValueError('Invalid pose')
    return R,(t/scale)/norm

def rays_and_jac(x,K):
    x=np.asarray(x,dtype=float); K=base.finite_array(K,(3,3),'K')
    if x.ndim!=2 or x.shape[1]!=2 or len(x)==0 or not np.isfinite(x).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('Invalid pixel coordinates or camera')
    v=np.column_stack(((x[:,0]-K[0,2])/K[0,0],(x[:,1]-K[1,2])/K[1,1],np.ones(len(x))))
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        norm=np.linalg.norm(v,axis=1)
    if not np.isfinite(norm).all() or np.any(norm<=0): raise ValueError('Invalid ray norm')
    d=v/norm[:,None]
    j=(np.eye(3)[None,:,:]-np.einsum('ni,nj->nij',d,d,optimize=False))[:,:,:2]/norm[:,None,None]
    j[:,:,0]/=K[0,0]; j[:,:,1]/=K[1,1]
    return v,d,j

def prepare_geometry(x1,x2,calibration):
    K1=np.asarray(calibration['K1']); K2=np.asarray(calibration['K2'])
    v1,d1,j1=rays_and_jac(x1,K1); v2,d2,j2=rays_and_jac(x2,K2)
    threshold=calibration['normalized_epipolar_threshold']
    if len(x1)!=len(x2) or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('Invalid paired geometry or threshold')
    return {'p1':np.column_stack((x1,np.ones(len(x1)))),'p2':np.column_stack((x2,np.ones(len(x2)))),
        'v1':v1,'v2':v2,'d1':d1,'d2':d2,'j1':j1,'j2':j2,
        'invK1':np.linalg.inv(K1),'invK2':np.linalg.inv(K2),
        'normalized_threshold':calibration['normalized_epipolar_threshold']}

def _final_scores(estimate,g):
    """All scores computed on FINAL poses. Pixel score is our common selector.

    The normalized and unit-bearing/Jacobian TS objectives are independently
    evaluated diagnostic columns; no returned pre-refinement score selects poses.
    The algebraic cheirality test follows PoseLib's positive depth tolerance .01
    after normalizing t. Zero denominators receive saturated cost rather than NaN.
    """
    R,t=validate_pose(estimate)
    tx=np.array([[0.,-t[2],t[1]],[t[2],0.,-t[0]],[-t[1],t[0],0.]])
    E=np.einsum('ij,jk->ik',tx,R,optimize=False)
    rd=np.einsum('ij,nj->ni',R,g['d1'],optimize=False)
    a=-np.einsum('ni,ni->n',rd,g['d2'],optimize=False)
    b1=-np.einsum('ni,i->n',rd,t,optimize=False); b2=np.einsum('ni,i->n',g['d2'],t,optimize=False)
    mind=.01*(1-a*a)
    valid=(b1-a*b2>mind)&(b2-a*b1>mind)
    def loss(A,p1,p2,threshold,j1=None,j2=None):
        l2=np.einsum('ij,nj->ni',A,p1,optimize=False)
        l1=np.einsum('ji,nj->ni',A,p2,optimize=False)
        c=np.einsum('ni,ni->n',p2,l2,optimize=False)
        if j1 is None: u,w=l1[:,:2],l2[:,:2]
        else:
            u=np.einsum('nij,ni->nj',j1,l1,optimize=False)
            w=np.einsum('nij,ni->nj',j2,l2,optimize=False)
        denom=np.einsum('ni,ni->n',u,u,optimize=False)+np.einsum('ni,ni->n',w,w,optimize=False)
        cost=np.ones(len(c)); good=valid&(denom>0)
        cost[good]=np.minimum(1.,(c[good]*c[good])/(denom[good]*threshold*threshold))
        return float(cost.sum()),int(np.count_nonzero(cost<1))
    F=np.einsum('ij,jk,kl->il',g['invK2'].T,E,g['invK1'],optimize=False)
    pixel,pn=loss(F,g['p1'],g['p2'],1.)
    normalized,nn=loss(E,g['v1'],g['v2'],g['normalized_threshold'])
    tangent,tn=loss(E,g['d1'],g['d2'],1.,g['j1'],g['j2'])
    return {'pixel_msac':pixel,'normalized_msac':normalized,'tangent_msac':tangent,
        'pixel_inliers':pn,'normalized_inliers':nn,'tangent_inliers':tn,'cheiral_count':int(valid.sum())}

def final_scores(estimate,g):
    for value in g.values():
        if not np.isfinite(value).all(): raise ValueError('Nonfinite geometry')
    if g['normalized_threshold']<=0: raise ValueError('Invalid normalized threshold')
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        result=_final_scores(estimate,g)
    return json_safe(result)

def verify_seal(manifest_path,seal_path):
    m=load_manifest(manifest_path); seal_path=Path(seal_path); s=json.loads(seal_path.read_text())
    if s.get('status')!='completed' or s.get('manifest_sha256')!=sha256(manifest_path) or s.get('source_sha256')!=source_hashes():
        raise ValueError('Seal/source binding mismatch')
    rows=s.get('rows',[])
    expected={(b,i,k) for b in BACKENDS for i,k in enumerate(m['selected_pair_keys'])}
    if len(rows)!=192 or {(r['backend'],r['pair_index'],r['key']) for r in rows}!=expected:
        raise ValueError('Incomplete/duplicate prediction cohort')
    records=[]
    for row in rows:
        name=row['filename']
        if Path(name).name!=name: raise ValueError('Unsafe sealed path')
        p=seal_path.parent/name
        if p.is_symlink() or p.parent.resolve()!=seal_path.parent.resolve(): raise ValueError('Sealed path is a symlink')
        if sha256(p)!=row['sha256']: raise ValueError('Prediction hash mismatch')
        r=json.loads(p.read_text())
        if any(r.get(k)!=row[k] for k in ('backend','pair_index','key')) or r.get('status')!='completed':
            raise ValueError('Prediction identity/status mismatch')
        if r.get('manifest_sha256')!=s['manifest_sha256'] or r.get('source_sha256')!=s['source_sha256']:
            raise ValueError('Prediction input binding mismatch')
        expected_library=json.loads(LIBRARIES.read_text())['old' if r['backend']=='old_norm' else 'current']
        if r.get('library')!=expected_library: raise ValueError('Prediction library mismatch')
        if len(r.get('candidates',[]))!=4: raise ValueError('Missing terminal candidates')
        for seed,candidate in zip(SEEDS,r['candidates']):
            if type(candidate.get('seed')) is not int or candidate['seed']!=seed or candidate.get('options')!=options(r['backend'],seed):
                raise ValueError('Candidate seed/options mismatch')
        process_path=seal_path.parent/(name+'.process.json')
        if process_path.is_symlink() or sha256(process_path)!=row.get('process_sha256'): raise ValueError('Process receipt differs')
        process=json.loads(process_path.read_text())
        if process.get('exit_code')!=0 or process.get('termination'): raise ValueError('Failed process')
        records.append(r)
    return m,s,records
