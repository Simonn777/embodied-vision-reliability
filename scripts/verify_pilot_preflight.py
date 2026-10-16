"""Independent raw-number and graph/allocation recomputation (no model)."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import os
import time
import zipfile
import numpy as np
from evr.bootstapir_dev import NumpyOnlyUnpickler,sha256


def arrayhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def raw(path,Q):
    rows=list(csv.DictReader(open(path)));out=np.full((len(rows)//Q,Q,4),np.nan,np.float32);seen=set();times={}
    for r in rows:
        t,q=int(r['frame']),int(r['query'])
        if (t,q) in seen:raise ValueError('duplicate raw row')
        seen.add((t,q));out[t,q]=[float(r[k]) for k in ['x','y','occlusion_logit','expected_dist_logit']]
        latency=float(r['frame_seconds'])
        if t in times:assert times[t]==latency
        times[t]=latency
    assert len(seen)==out.shape[0]*Q and np.isfinite(out).all()
    return out,[times[t] for t in range(len(times))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('results/0002-preflight'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    manifest=json.loads((a.root/'identity/manifest.json').read_text());inventory=list(csv.DictReader(open(a.root/'identity/frame-hashes.csv')))
    assert sha256(a.root/'identity/frame-hashes.csv')==manifest['frame_hashes_sha256']
    by_video={i:[] for i in range(50)}
    for r in inventory:
        i=int(r['archive_index']);assert int(r['frame'])==len(by_video[i]);by_video[i].append(r['rgb_sha256'])
    # Independently build graph using intersections; do not call candidate allocate.
    adjacency={i:set() for i in range(50)}
    for i in range(50):
        for j in range(i):
            same=any(manifest['identities'][i][k]==manifest['identities'][j][k] for k in ['video_sha256','prefix64_sha256'])
            if same or set(by_video[i])&set(by_video[j]):adjacency[i].add(j);adjacency[j].add(i)
    components=[];visited=set()
    for i in range(50):
        if i in visited:continue
        stack=[i];group=[]
        while stack:
            j=stack.pop()
            if j in visited:continue
            visited.add(j);group.append(j);stack.extend(adjacency[j]-visited)
        components.append(sorted(group))
    actual={tuple(g['members']):g for g in manifest['groups']};assert set(actual)=={tuple(g) for g in components}
    ordered=sorted(components,key=lambda g:hashlib.sha256(('0002|20261019|g-'+'-'.join(f'{i:03}' for i in g)).encode()).hexdigest())
    counts=dict(development=0,calibration=0,evaluation=0,reserve=0)
    for group in ordered:
        partition='development' if 0 in group else 'calibration' if counts['calibration']<10 else 'evaluation' if counts['evaluation']<20 else 'reserve'
        assert actual[tuple(group)]['partition']==partition;counts[partition]+=len(group)
    assert counts==manifest['counts']
    # Recompute every array and frame digest directly from official cached bytes.
    assert sha256(a.cache/'rgb.zip')==manifest['archive']['sha256']
    with zipfile.ZipFile(a.cache/'rgb.zip') as z:
        member=next(x for x in z.namelist() if x.endswith('.pkl'))
        with z.open(member) as f:data=NumpyOnlyUnpickler(f).load()
    verified=0
    for i,e in enumerate(data):
        identity=manifest['identities'][i]
        for k,v in e.items():
            assert arrayhash(v)==identity['arrays'][k]['sha256'];assert list(v.shape)==identity['arrays'][k]['shape'];assert str(v.dtype)==identity['arrays'][k]['dtype'];verified+=1
        assert arrayhash(e['video'][:64])==identity['prefix64_sha256']
        assert [arrayhash(frame) for frame in e['video']]==by_video[i]
    batch,bt=raw(a.root/'dev64/batch64-frames.csv',2);single,st=raw(a.root/'dev64/single16-frames.csv',1);repeat,rt=raw(a.root/'repeat64/batch64-frames.csv',2)
    diffs={'independent64':float(np.max(np.abs(batch-repeat))),'single_batch16':float(np.max(np.abs(batch[:16,0]-single[:,0])))}
    assert max(diffs.values())<=1e-4
    for phase,name,times in [('dev64','batch64',bt),('dev64','single16',st),('repeat64','batch64',rt)]:
        receipt=json.loads((a.root/phase/'receipt.json').read_text());r=receipt['runs'][name]
        assert receipt['status']=='completed' and r['first_frame_seconds']==times[0]
        assert r['subsequent_median_seconds']==float(np.median(times[1:]))
        assert abs(r['frame_seconds_sum']-sum(times))<1e-9 and sum(times)<=r['wall_seconds']
    result={'status':'completed','started_at_utc':started,'ended_at_utc':datetime.now(timezone.utc).isoformat(),
      'elapsed_seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'array_digests_recomputed':verified,'prefix_digests_recomputed':50,'frame_digests_recomputed':len(inventory),
      'graph_components':len(components),'counts':counts,'raw_max_differences':diffs,'atol':1e-4,'rtol':0,
      'heldout_inference_or_errors':False,'scope':'independent algorithms for graph/allocation and numeric/latency checks, same trusted restricted decoder and host'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
