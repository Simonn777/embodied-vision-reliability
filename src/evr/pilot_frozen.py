"""Frozen-input gates and exact prediction-grid loading; no network/torch import."""
import csv
import json
import importlib.metadata
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from evr.bootstapir_dev import sha256, download_object, NumpyOnlyUnpickler
from evr.pilot_identity import digest_array
from evr.pilot_metrics import rows_from_arrays


def verify_freeze(root,receipt_path,require_main=True):
    root=Path(root);receipt=json.loads(Path(receipt_path).read_text())
    if receipt.get('status')!='frozen_exploratory_v1':raise ValueError('not a frozen exploratory receipt')
    for package,version in receipt.get('dependency_versions',{}).items():
        if importlib.metadata.version(package)!=version:raise ValueError('frozen dependency version changed: '+package)
    commit=receipt['input_commit']
    subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=root,check=True,capture_output=True)
    if require_main:
        subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'],cwd=root,check=True,capture_output=True)
    for path,digest in receipt['input_sha256'].items():
        p=Path(path)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe frozen path')
        if sha256(root/p)!=digest:raise ValueError('frozen input changed: '+path)
    return receipt


def load_verified_dataset(cache,manifest):
    identity=download_object(cache,'rgb.zip')
    if identity['sha256']!=manifest['archive']['sha256']:raise ValueError('archive differs from allocation')
    with zipfile.ZipFile(Path(cache)/'rgb.zip') as z:
        infos=z.infolist()
        if sum(x.file_size for x in infos)>4*1024**3 or any(Path(x.filename).is_absolute() or '..' in Path(x.filename).parts for x in infos):raise ValueError('ZIP gate')
        files=[x for x in infos if x.filename.endswith('.pkl')]
        if len(files)!=1:raise ValueError('pickle count')
        with z.open(files[0]) as f:data=NumpyOnlyUnpickler(f).load()
    if not isinstance(data,list) or len(data)!=len(manifest['identities']):raise ValueError('dataset length')
    return data


def verify_video(example,identity):
    if set(example)!=set(identity['arrays']):raise ValueError('video schema mismatch')
    for key,info in identity['arrays'].items():
        a=example[key]
        if list(a.shape)!=info['shape'] or str(a.dtype)!=info['dtype'] or digest_array(a)!=info['sha256']:raise ValueError('video identity mismatch')
    if digest_array(example['video'][:64])!=identity['prefix64_sha256']:raise ValueError('prefix identity mismatch')


def query_ids(example):
    return np.flatnonzero(~example['occluded'][:,0])[:2].astype(int).tolist()


def prediction_rows(directory,manifest,partition,dataset,freeze_commit):
    root=Path(directory);rows=[];groups=[g for g in manifest['groups'] if g['partition']==partition]
    for group in groups:
        folder=root/group['group_id'];receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['status']!='completed' or receipt['freeze_input_commit']!=freeze_commit or receipt['group_id']!=group['group_id']:raise ValueError('incomplete/unbound prediction group')
        for i in group['members']:
            identity=manifest['identities'][i];example=dataset[i];verify_video(example,identity)
            ids=query_ids(example);info=receipt['videos'][str(i)];path=folder/f'video-{i:03}.csv'
            if info['point_indices']!=ids or info['frames']!=64 or sha256(path)!=info['raw_sha256']:raise ValueError('prediction identity/grid gate')
            raw=np.full((64,len(ids),4),np.nan,np.float64);seen=set()
            for r in csv.DictReader(open(path)):
                t,q=int(r['frame']),int(r['query'])
                if not (0<=t<64 and 0<=q<len(ids)) or (t,q) in seen or int(r['point'])!=ids[q]:raise ValueError('duplicate/outside prediction row')
                seen.add((t,q));raw[t,q]=[float(r[k]) for k in ['x','y','occlusion_logit','expected_dist_logit']]
            if len(seen)!=64*len(ids):raise ValueError('incomplete prediction grid')
            rows.extend(rows_from_arrays(i,identity['video_sha256'],group['group_id'],raw,ids,example['points'],example['occluded']))
    return rows,[g['group_id'] for g in groups]
