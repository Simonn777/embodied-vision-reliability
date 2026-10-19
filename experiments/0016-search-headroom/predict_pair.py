"""Four fixed seeds of one official backend, no reference-pose access."""
import argparse,json,sys,time
from pathlib import Path
import common as c
np=c.np

def library_metadata(backend):
    folder=Path(backend.__file__).resolve().parent; binary=Path(backend._core.__file__).resolve()
    return {'version':backend.__version__,'binary_name':binary.name,'binary_bytes':binary.stat().st_size,
        'binary_sha256':c.sha256(binary),'python_files':{p.name:c.sha256(p) for p in sorted(folder.glob('*.py'))}}

def invoke(backend,x1,x2,cal,kind,seed):
    opt=c.options(kind,seed)
    if kind=='old_norm':
        return backend.estimate_relative_pose(x1,x2,cal['camera1'],cal['camera2'],opt,{},initial_pose=None)
    return backend.estimate_relative_pose(x1,x2,cal['camera1'],cal['camera2'],opt,initial_pose=None)

def run(data,manifest_path,output,kind,index):
    # Create one-shot output before import or scientific data access.
    with Path(output).open('x+') as stream:
        r={'status':'started','schema':'evr0016.prediction.v1','backend':kind,'pair_index':index,
            'phase':'binding','candidates':[]}
        def save(): c.base.write_record(stream,r)
        save()
        try:
            m=c.load_manifest(manifest_path)
            if kind not in c.BACKENDS or type(index) is not int or not 0<=index<64: raise ValueError('Invalid work item')
            r.update(key=m['selected_pair_keys'][index],manifest_sha256=c.sha256(manifest_path),source_sha256=c.source_hashes())
            import poselib,h5py
            r['library']=library_metadata(poselib)
            expected=json.loads(c.LIBRARIES.read_text())['old' if kind=='old_norm' else 'current']
            if r['library']!=expected: raise ValueError('Installed library differs from build pin')
            r['phase']='inputs'; save(); start=time.perf_counter()
            with h5py.File(data,'r') as h: x1,x2,cal=c.base.read_pair(h,r['key'])
            with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                g=c.prepare_geometry(x1,x2,cal)
            r['input_seconds']=time.perf_counter()-start
            r['calibration']=c.json_safe(cal)
            r['input_array_sha256']={name:__import__('hashlib').sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name,value in [('x1',x1),('x2',x2)]}
            charged=r['input_seconds']
            for seed in c.SEEDS:
                row={'seed':seed,'options':c.options(kind,seed),'status':'started'}
                r['candidates'].append(row); r['phase']='estimator'; save()
                start=time.perf_counter()
                try:
                    pose,info=invoke(poselib,x1,x2,cal,kind,seed)
                except Exception as exc:
                    row.update(status='estimator_error',error=type(exc).__name__+': '+str(exc),call_seconds=time.perf_counter()-start,score_seconds=0.)
                    charged+=row['call_seconds']; row['charged_end_seconds']=charged; save(); continue
                row['call_seconds']=time.perf_counter()-start; r['phase']='validate_and_rescore'; save()
                start=time.perf_counter()
                estimate={'R':np.asarray(pose.R),'t':np.asarray(pose.t)}
                c.validate_pose(estimate)
                row['estimate']=c.json_safe(estimate)
                row['final_scores']=c.final_scores(estimate,g)
                stats=dict(info); mask=np.asarray(stats.pop('inliers'))
                if mask.shape!=(10000,) or not np.isin(mask,[False,True]).all(): raise ValueError('Invalid returned mask')
                row['native_inliers_packbits_big_hex']=np.packbits(mask.astype(np.uint8),bitorder='big').tobytes().hex()
                row['returned_ransac_stats']=c.json_safe(stats)
                row['returned_stats_semantics']='Pre-final-refinement native stats; not used for candidate selection'
                row['score_seconds']=time.perf_counter()-start
                charged+=row['call_seconds']+row['score_seconds']
                row.update(charged_end_seconds=charged,status='completed'); save()
            r.update(status='completed',phase='complete',charged_total_seconds=charged); save(); return 0
        except BaseException as exc:
            r.update(status='error',error=type(exc).__name__+': '+str(exc)); save(); return 1

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--backend',choices=c.BACKENDS,required=True); p.add_argument('--pair-index',type=int,required=True)
    a=p.parse_args(); return run(a.data,a.manifest,a.output,a.backend,a.pair_index)
if __name__=='__main__': sys.exit(main())
