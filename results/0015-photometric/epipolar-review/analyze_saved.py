"""Post-hoc GT epipolar compatibility, not correspondence error or method selection."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def camera_K(camera):
    fx, fy, cx, cy = camera['params']
    return np.array([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])

def target_distance(F, p, q):
    x = np.column_stack([p,np.ones(len(p))])
    y = np.column_stack([q,np.ones(len(q))])
    line = np.einsum('ij,nj->ni',F,x,optimize=False)
    denominator = np.sqrt(np.einsum('ni,ni->n',line[:,:2],line[:,:2],optimize=False))
    if np.any(denominator <= 1e-15):
        raise ValueError('degenerate GT epipolar line')
    return np.abs(np.einsum('ni,ni->n',line,y,optimize=False))/denominator

def summary(values):
    return {'count':len(values),'mean_px':float(np.mean(values)),
            'median_px':float(np.median(values)), 'rms_px':float(np.sqrt(np.mean(values*values))),
            'max_px':float(np.max(values))}

def main():
    # Exact artificial horizontal epipolar fixture before any scientific saved records.
    F=np.array([[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    assert np.allclose(target_distance(F,np.array([[1.,2.],[3.,4.]]),np.array([[90.,2.],[-20.,7.]])),[0.,3.])
    truth_path=REPO/'results/0013-baseline-calibration-compatibility/evaluation.json'
    if sha(truth_path) != 'f8fe46392810b78d485a8dd90ae1849eae6491eaba620d2c9454b6c0f15cb5a4':
        raise ValueError('original labels changed')
    seal_path=HERE.parent/'photometric/seal.json'
    seal=json.loads(seal_path.read_text())
    assert seal['complete'] and len(seal['rows']) == 16
    truth=json.loads(truth_path.read_text())
    rows, all_before, all_after = [],[],[]
    for i,(entry, gt_row) in enumerate(zip(seal['rows'],truth['rows'])):
        path=seal_path.parent/entry['path']
        assert sha(path) == entry['sha256']
        r=json.loads(path.read_text())
        assert r['pair_index']==i and r['key']==gt_row['key']
        queries=[q for q in r['queries'] if q['admitted']]
        p=np.array([q['original_source_cache'] for q in queries])
        q0=np.array([q['original_target_cache'] for q in queries])
        q1=np.array([q['target_cache'] for q in queries])
        gt=gt_row['ground_truth']
        R=np.array(gt['R_projected']);t=np.array(gt['t_unit'])
        x,y,z=t;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        F=np.linalg.inv(camera_K(r['calibration']['camera2'])).T @ cross @ R @ np.linalg.inv(camera_K(r['calibration']['camera1']))
        d0=target_distance(F,p,q0);d1=target_distance(F,p,q1)
        all_before.extend(d0);all_after.extend(d1)
        rows.append({'pair_index':i,'key':r['key'],'count':len(queries),
            'before':summary(d0),'after':summary(d1),
            'improved_points':int(np.sum(d1<d0-1e-9)), 'worsened_points':int(np.sum(d1>d0+1e-9)),
            'point_rows':[{'index':q['index'],'before_px':float(a),'after_px':float(b)} for q,a,b in zip(queries,d0,d1)]})
    a=np.array(all_before);b=np.array(all_after)
    result={'schema':'evr0015.posthoc-epipolar.v1','script_sha256':sha(Path(__file__)),
        'baseline_evaluation_sha256':sha(truth_path),'photometric_seal_sha256':sha(seal_path),
        'scope':'Post-hoc descriptive saved-output diagnostic; all admitted points; no new images/H5/poses.',
        'metric':'Absolute target-image pixel distance to fixed GT epipolar line through fixed original source point; authored native cameras and saved projected GT rotation.',
        'not_correspondence_truth':True,'no_population_independence_claim':True,
        'before':summary(a),'after':summary(b),
        'improved_points':int(np.sum(b<a-1e-9)),'worsened_points':int(np.sum(b>a+1e-9)),
        'macro_pair_mean_before_px':float(np.mean([r['before']['mean_px'] for r in rows])),
        'macro_pair_mean_after_px':float(np.mean([r['after']['mean_px'] for r in rows])),
        'pairs_with_lower_mean_distance':sum(r['after']['mean_px']<r['before']['mean_px'] for r in rows),
        'rows':rows}
    with (HERE/'analysis.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
