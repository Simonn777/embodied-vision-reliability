"""Serial metadata audit and video-0-only 64-frame CPU development measurement."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
from evr.bootstapir_dev import (NumpyOnlyUnpickler, compare, download_object, sha256,
    stream, verify_development, verify_module_paths, validation_status)
from evr.pilot_identity import allocate, digest_array


def utc(): return datetime.now(timezone.utc).isoformat()


def reserve_output(path):
    # Refuse any reused directory, including an empty one; no receipt overwritten.
    path.mkdir(parents=True, exist_ok=False)


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["metadata","dev64","repeat64"])
    p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference",type=Path);a=p.parse_args()
    reserve_output(a.output)
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
        if os.environ.get(k)!="2": raise ValueError("thread environment must be 2")
    start=time.perf_counter();root=Path(__file__).resolve().parents[1]
    sources=["scripts/run_pilot_preflight.py","src/evr/bootstapir_dev.py","src/evr/pilot_identity.py",
             "protocols/0002-preflight-64-and-identity.md","configs/0002-preflight.json"]
    receipt={"phase":a.phase,"started_at_utc":utc(),"status":"running",
      "python_version":platform.python_version(),"platform":platform.platform(),
      "meminfo_start_kib":{s.split(':')[0]:int(s.split()[1]) for s in open('/proc/meminfo') if s.startswith(('MemTotal:','MemAvailable:'))},
      "cgroup_cpu_max":Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
      "disk_start_bytes":dict(zip(('total','used','free'),shutil.disk_usage(a.cache))),
      "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "file_sha256":{s:sha256(root/s) for s in sources},"affinity":sorted(os.sched_getaffinity(0)),
      "limit_as_bytes":6*1024**3,"hard_timeout_seconds":900,
      "cpu_model":next(s.split(":",1)[1].strip() for s in open('/proc/cpuinfo') if s.startswith('model name')),
      "cgroup_memory_max":Path('/sys/fs/cgroup/memory.max').read_text().strip(),
      "command":"env -i PATH=\"$PATH\" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_pilot_preflight.py "+a.phase+" --cache "+str(a.cache)+" --output "+str(a.output)+((' --reference '+str(a.reference)) if a.reference else '')}
    def save():
        receipt.update(ended_at_utc=utc(),elapsed_seconds=time.perf_counter()-start,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        for source in sources:
            actual=subprocess.check_output(['git','hash-object',source],text=True).strip()
            published=subprocess.check_output(['git','rev-parse','HEAD:'+source],text=True).strip()
            if actual!=published:raise ValueError('runtime source differs from recorded commit: '+source)
        if a.phase=='metadata':
            receipt['archive']=download_object(a.cache,'rgb.zip')
            entries=[];frames_all=[]
            with zipfile.ZipFile(a.cache/'rgb.zip') as z:
                infos=z.infolist()
                if sum(x.file_size for x in infos)>4*1024**3: raise ValueError('archive size gate')
                if any(Path(x.filename).is_absolute() or '..' in Path(x.filename).parts for x in infos): raise ValueError('ZIP path gate')
                members=[x for x in infos if x.filename.endswith('.pkl')]
                if len(members)!=1: raise ValueError('pickle member gate')
                with z.open(members[0]) as f: data=NumpyOnlyUnpickler(f).load()
                if not isinstance(data,list) or len(data)!=50: raise ValueError('50-example list required')
                with (a.output/'frame-hashes.csv').open('w',newline='') as out:
                    w=csv.writer(out,lineterminator='\n');w.writerow(['archive_index','frame','rgb_sha256'])
                    for i,example in enumerate(data):
                        if set(example)!={'video','points','occluded'}: raise ValueError('unexpected schema; audit scene IDs explicitly')
                        video=example['video'];points=example['points'];occ=example['occluded']
                        item={'archive_index':i,'schema_keys':sorted(example),'arrays':{}}
                        for key,array in example.items():
                            if array.dtype.hasobject: raise ValueError('object array forbidden')
                            item['arrays'][key]={'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest_array(array)}
                        item['video_sha256']=item['arrays']['video']['sha256']
                        item['prefix64_sha256']=digest_array(video[:64]);item['prefix_length']=min(64,len(video))
                        hashes=[digest_array(frame) for frame in video];frames_all.append(hashes)
                        for t,h in enumerate(hashes):w.writerow([i,t,h])
                        out.flush();entries.append(item)
            allocation=allocate(entries,frames_all)
            manifest={'scope':'identity/metadata/exact-RGB duplicate audit only; no model inference, errors or reliability scores for videos 1..49',
              'archive':receipt['archive'],'identities':entries,**allocation,
              'frame_hashes_sha256':sha256(a.output/'frame-hashes.csv'),
              'known_shared_assets':'triplet-4 object set and front-left camera; scene/episode relationships unknown',
              'near_duplicates_excluded':False,'heldout_model_inference_performed':False}
            (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            receipt['manifest_sha256']=sha256(a.output/'manifest.json');receipt['counts']=allocation['counts']
        else:
            import torch
            if torch.__version__!='2.10.0+cpu' or torch.version.cuda is not None:raise ValueError('patched CPU torch required')
            receipt['upstream_modules']=verify_module_paths(a.cache/'source')
            from tapnet.torch.tapir_model import TAPIR
            for name,info in receipt['upstream_modules'].items():
                actual=Path(sys.modules[name].__file__).resolve()
                if actual!=(a.cache/'source'/info['relative_path']).resolve() or sha256(actual)!=info['sha256']:raise ValueError('actual module mismatch')
            torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20261016)
            np.random.seed(20261016);torch.use_deterministic_algorithms(True)
            receipt['packages']={k:importlib.metadata.version(k) for k in ['torch','numpy','dm-tree','einshape']}
            receipt['torch_threads']=[torch.get_num_threads(),torch.get_num_interop_threads()]
            receipt['checkpoint']=download_object(a.cache,'weights.pt')
            arrays,receipt['input_identity']=verify_development(a.cache/'development-000.npz',root/'results/0002-development/decode/development-manifest.json')
            frames=arrays['video'][:64]
            if frames.shape!=(64,256,256,3):raise ValueError('64-frame 256px gate')
            ids=np.flatnonzero(~arrays['occluded'][:,0])[:2].tolist()
            if len(ids)!=2:raise ValueError('two frame-zero-visible development queries required')
            queries=[{'arrival':0,'xy':arrays['points'][i,0].tolist()} for i in ids]
            receipt['point_indices']=ids;receipt['query_selection']='first two visible at frame 0 only';receipt['queries']=queries
            model=TAPIR(pyramid_level=1,use_casual_conv=True);assert model.use_casual_conv
            model.load_state_dict(torch.load(a.cache/'weights.pt',weights_only=True,map_location='cpu'),strict=True)
            model.eval().requires_grad_(False);assert not model.training and not any(x.requires_grad for x in model.parameters())
            outputs={};checks={}
            def run(name,images,qs):
                with (a.output/(name+'-frames.csv')).open('x',newline='') as f:
                    w=csv.writer(f,lineterminator='\n');w.writerow(['frame','query','x','y','occlusion_logit','expected_dist_logit','frame_seconds'])
                    def append(t,values,latency):
                        for q,v in enumerate(values):w.writerow([t,q,*map(float,v),latency])
                        f.flush()
                    begin=time.perf_counter();values,times=stream(model,images,qs,chunk_size=8,on_frame=append)
                outputs[name]=values;receipt.setdefault('runs',{})[name]={'wall_seconds':time.perf_counter()-begin,'shape':list(values.shape),
                    'first_frame_seconds':times[0],'subsequent_median_seconds':float(np.median(times[1:])),
                    'frame_seconds_sum':sum(times),'finite_values':int(np.isfinite(values).sum()),
                    'raw_csv_sha256':sha256(a.output/(name+'-frames.csv'))}
                save();print(name,receipt['runs'][name]['wall_seconds'],flush=True)
                return values
            batch=run('batch64',frames,queries)
            if a.phase=='dev64':
                single=run('single16',frames[:16],queries[:1]);checks['single_batch_prefix']=compare(single[:,0],batch[:16,0])
                # Earlier batch is a fixed recorded input; compare only if query IDs match.
                from check_bootstapir_measurements import load_runs
                old=load_runs(root/'results/0002-development/validated/raw-outputs.csv')
                oldids=json.loads((root/'results/0002-development/validated/measure-receipt.json').read_text())['point_indices']
                if ids!=oldids:raise ValueError('prior prefix query IDs differ; no invented comparison')
                checks['old16_batch_prefix']=compare(old['batch_16'],batch[:16])
            else:
                if a.reference is None:raise ValueError('repeat reference required')
                rows=list(csv.DictReader(open(a.reference/'batch64-frames.csv')))
                ref=np.array([[float(r[k]) for k in ['x','y','occlusion_logit','expected_dist_logit']] for r in rows],np.float32).reshape(64,2,4)
                checks['independent64_replay']=compare(ref,batch);receipt['reference_csv_sha256']=sha256(a.reference/'batch64-frames.csv')
            receipt['checks']=checks;receipt['status']=validation_status(checks)
        if receipt['status']=='running':receipt['status']='completed'
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc().replace(str(root),'<repo>'))
        print(receipt['traceback'],file=sys.stderr)
    finally:save()
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
