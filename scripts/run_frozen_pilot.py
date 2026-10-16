"""Future exact-group inference/evaluation. Requires published freeze on main.

This session must not invoke infer/evaluate on held-out partitions. Use --help or
the pure fixture suite only until an independently reviewed freeze has merged.
"""
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from evr.bootstapir_dev import sha256,download_object,verify_module_paths,stream
from evr.pilot_frozen import verify_freeze,load_verified_dataset,verify_video,query_ids,prediction_rows
from evr.pilot_metrics import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['infer','evaluate'])
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',type=Path,default=Path('protocols/0002-freeze-v1.json'))
    p.add_argument('--group-id');p.add_argument('--predictions',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    keys=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']
    if any(os.environ.get(k)!='2' for k in keys):raise ValueError('numeric threads must be 2')
    root=Path(__file__).resolve().parents[1];started=time.perf_counter()
    rec={'mode':a.mode,'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
         'affinity':sorted(os.sched_getaffinity(0)),'limit_as_bytes':6*1024**3,'timeout_seconds':900,
         'packages':{'numpy':np.__version__}}
    def save():
        rec.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (a.output/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
    save()
    try:
        frozen=verify_freeze(root,a.freeze);rec['freeze_input_commit']=frozen['input_commit']
        rec['freeze_receipt_sha256']=sha256(a.freeze)
        manifest=json.loads((root/frozen['manifest_path']).read_text())
        if not manifest['targets_met']:raise ValueError('allocation target gate')
        dataset=load_verified_dataset(a.cache,manifest)
        if a.mode=='infer':
            groups=[g for g in manifest['groups'] if g['group_id']==a.group_id and g['partition'] in ('calibration','evaluation')]
            if len(groups)!=1:raise ValueError('group must be allocated calibration/evaluation')
            group=groups[0]
            if len(group['members'])>4:raise ValueError('group exceeds four-video resource gate; do not split it')
            rec['group_id']=group['group_id'];rec['partition']=group['partition'];rec['videos']={}
            import torch
            if torch.__version__!='2.10.0+cpu' or torch.version.cuda is not None:raise ValueError('patched CPU torch gate')
            rec['actual_modules']=verify_module_paths(a.cache/'source')
            from tapnet.torch.tapir_model import TAPIR
            for name,info in rec['actual_modules'].items():
                actual=Path(sys.modules[name].__file__).resolve()
                if actual!=(a.cache/'source'/info['relative_path']).resolve() or sha256(actual)!=info['sha256']:raise ValueError('actual import mismatch')
            torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20261016)
            np.random.seed(20261016);torch.use_deterministic_algorithms(True)
            rec['packages']['torch']=torch.__version__;rec['torch_threads']=[torch.get_num_threads(),torch.get_num_interop_threads()]
            rec['checkpoint']=download_object(a.cache,'weights.pt')
            if rec['checkpoint']['sha256']!=frozen['checkpoint_sha256']:raise ValueError('checkpoint freeze mismatch')
            model=TAPIR(pyramid_level=1,use_casual_conv=True);assert model.use_casual_conv
            model.load_state_dict(torch.load(a.cache/'weights.pt',map_location='cpu',weights_only=True),strict=True)
            model.eval().requires_grad_(False)
            for i in group['members']:
                e=dataset[i];verify_video(e,manifest['identities'][i]);ids=query_ids(e)
                frames=e['video'][:64]
                if frames.shape!=(64,256,256,3):raise ValueError('fixed 64-frame gate')
                qs=[{'arrival':0,'xy':e['points'][j,0].tolist()} for j in ids]
                file=a.output/f'video-{i:03}.csv'
                with file.open('x',newline='') as f:
                    w=csv.writer(f,lineterminator='\n');w.writerow(['frame','query','point','x','y','occlusion_logit','expected_dist_logit','frame_seconds'])
                    def append(t,values,seconds):
                        for q,v in enumerate(values):w.writerow([t,q,ids[q],*map(float,v),seconds])
                        f.flush()
                    # Preserve nonfinite estimator outputs as missingness for the
                    # evaluator rather than silently clipping/dropping a video.
                    raw,times=stream(model,frames,qs,chunk_size=8,on_frame=append,allow_nonfinite=True)
                rec['videos'][str(i)]={'point_indices':ids,'frames':64,'raw_sha256':sha256(file),
                    'first_frame_seconds':times[0],'subsequent_median_seconds':float(np.median(times[1:])),
                    'all_frame_seconds':times,'finite_values':int(np.isfinite(raw).sum())}
                save()
        else:
            if a.predictions is None:raise ValueError('complete frozen predictions required')
            cal,cgroups=prediction_rows(a.predictions,manifest,'calibration',dataset,frozen['input_commit'])
            ev,egroups=prediction_rows(a.predictions,manifest,'evaluation',dataset,frozen['input_commit'])
            config=json.loads((root/'configs/0002-exploratory-v1.json').read_text())
            result=evaluate(cal,ev,egroups,repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'])
            result['calibration_groups']=cgroups;result['evaluation_groups']=egroups
            (a.output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            rec['result_sha256']=sha256(a.output/'evaluation.json')
        rec['status']='completed'
    except Exception as e:
        rec.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc().replace(str(root),'<repo>'))
    finally:save()
    return 0 if rec['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
