"""Exercise prediction-grid->score->calibration->evaluation->bootstrap on fixtures."""
import csv
import json
from pathlib import Path
import tempfile
import numpy as np
from evr.bootstapir_dev import sha256
from evr.pilot_identity import digest_array
from evr.pilot_frozen import prediction_rows
from evr.pilot_metrics import evaluate


def run_fixture():
    a={'video':np.zeros((64,256,256,3),np.uint8),'points':np.broadcast_to([.25,.5],(1,64,2)).copy(),'occluded':np.zeros((1,64),bool)}
    data=[{k:v.copy() for k,v in a.items()} for _ in range(3)];data[2]['occluded'][:]=True
    ids=[]
    for i,e in enumerate(data):
        ids.append({'archive_index':i,'video_sha256':digest_array(e['video']),'prefix64_sha256':digest_array(e['video']),
          'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest_array(v)} for k,v in e.items()}})
    manifest={'identities':ids,'groups':[{'group_id':'cal','members':[0],'partition':'calibration'},{'group_id':'eval','members':[1],'partition':'evaluation'},{'group_id':'empty','members':[2],'partition':'evaluation'}]}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for i,g in enumerate(manifest['groups']):
            folder=root/g['group_id'];folder.mkdir();file=folder/f'video-{i:03}.csv';points=[] if i==2 else [0]
            with file.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['frame','query','point','x','y','occlusion_logit','expected_dist_logit'])
                if points:
                    for t in range(64):w.writerow([t,0,0,64+(5 if t%5==0 else 0),128,-1,1 if t%5==0 else -1])
            (folder/'receipt.json').write_text(json.dumps({'status':'completed','freeze_input_commit':'fixture','group_id':g['group_id'],'videos':{str(i):{'point_indices':points,'frames':64,'raw_sha256':sha256(file)}}}))
        cal,_=prediction_rows(root,manifest,'calibration',data,'fixture');ev,groups=prediction_rows(root,manifest,'evaluation',data,'fixture')
        result=evaluate(cal,ev,groups,repeats=2000,seed=20261018)
        assert result['matched']['native']['reachable'] and result['matched']['augmented']['reachable']
        assert result['bootstrap']['allocated_groups']==['eval','empty']
        return {'scope':'synthetic fixture only; not a tracking research result','all_pipeline_stages_executed':True,'result':result}


if __name__=='__main__':print(json.dumps(run_fixture(),indent=2,allow_nan=False))
