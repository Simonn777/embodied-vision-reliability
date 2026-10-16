import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from evr.pilot_frozen import verify_freeze,verify_video,prediction_rows
from evr.pilot_identity import digest_array
from evr.bootstapir_dev import sha256


def identity(example,index):
    return {'archive_index':index,'video_sha256':digest_array(example['video']),
      'prefix64_sha256':digest_array(example['video'][:64]),
      'arrays':{k:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256':digest_array(a)} for k,a in example.items()}}


def test_freeze_gates_changed_files_and_requires_published_main(tmp_path):
    subprocess.run(['git','init','-q'],cwd=tmp_path,check=True)
    (tmp_path/'code.py').write_text('fixed')
    subprocess.run(['git','add','code.py'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],cwd=tmp_path,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tmp_path,text=True).strip()
    f=tmp_path/'receipt.json';f.write_text(json.dumps({'status':'frozen_exploratory_v1','input_commit':commit,'input_sha256':{'code.py':sha256(tmp_path/'code.py')}}))
    verify_freeze(tmp_path,f,require_main=False)
    with pytest.raises(subprocess.CalledProcessError):verify_freeze(tmp_path,f)
    (tmp_path/'code.py').write_text('changed')
    with pytest.raises(ValueError,match='frozen input changed'):verify_freeze(tmp_path,f,require_main=False)


def test_future_prediction_grid_and_zero_query_group_retained(tmp_path):
    a={'video':np.zeros((64,256,256,3),np.uint8),'points':np.zeros((1,64,2),np.float32),'occluded':np.zeros((1,64),bool)}
    b={k:v.copy() for k,v in a.items()};b['occluded'][:]=True
    manifest={'identities':[identity(a,0),identity(b,1)],'groups':[{'group_id':'a','members':[0],'partition':'evaluation'},{'group_id':'empty','members':[1],'partition':'evaluation'}]}
    for group,e,ids in [('a',a,[0]),('empty',b,[])]:
        folder=tmp_path/group;folder.mkdir();i=0 if group=='a' else 1;file=folder/f'video-{i:03}.csv'
        with file.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['frame','query','point','x','y','occlusion_logit','expected_dist_logit'])
            if ids:
                for t in range(64):w.writerow([t,0,0,0,0,-1,-1])
        (folder/'receipt.json').write_text(json.dumps({'status':'completed','freeze_input_commit':'fixed','group_id':group,'videos':{str(i):{'point_indices':ids,'frames':64,'raw_sha256':sha256(file)}}}))
    rows,frame=prediction_rows(tmp_path,manifest,'evaluation',[a,b],'fixed')
    assert len(rows)==64 and frame==['a','empty']
    (tmp_path/'a/video-000.csv').write_text('tampered')
    with pytest.raises(ValueError,match='prediction identity'):prediction_rows(tmp_path,manifest,'evaluation',[a,b],'fixed')


def test_array_or_prefix_identity_change_rejected():
    e={'video':np.zeros((64,256,256,3),np.uint8),'points':np.zeros((1,64,2),np.float32),'occluded':np.zeros((1,64),bool)}
    i=identity(e,0);e['video'][0,0,0,0]=1
    with pytest.raises(ValueError,match='video identity'):verify_video(e,i)
