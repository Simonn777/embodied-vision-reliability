"""Manufactured 48-point API qualification only; never loads cached assets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import common as c
import predict_pair as worker
np=c.np


def fixture():
    rng=np.random.default_rng(20261023)
    X=rng.uniform([-2.,-1.,4.],[2.,1.,9.],(48,3))
    angle=.09
    R=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
    t=np.array([.6,.03,.02]); Y=(R@X.T).T+t
    K=np.array([[800.,0.,320.],[0.,800.,240.],[0.,0.,1.]])
    def project(points):
        pixels=(K@points.T).T
        return pixels[:,:2]/pixels[:,2:]
    camera={'model':'PINHOLE','width':640,'height':480,'params':[800.,800.,320.,240.]}
    pair={'x1':project(X),'x2':project(Y),'depth1':X[:,2].copy(),'depth2':Y[:,2].copy(),
          'calibration':{'K1':K,'K2':K,'camera1':dict(camera),'camera2':dict(camera)}}
    return pair, R, t


def run(output):
    # Exclusive receipt before importing a native estimator.
    with Path(output).open('x+') as stream:
        receipt={'schema':'evr0017.native-qualification.v1','status':'started',
                 'scope':'Manufactured 48-point scene; no scientific assets','calls':[]}
        c.write_record(stream,receipt)
        try:
            import poselib
            receipt['library']=c.verify_library(poselib)
            receipt['source_sha256']={name:c.sha256(c.HERE/name) for name in ('common.py','predict_pair.py','qualify_native.py')}
            receipt['library_config_sha256']=c.sha256(c.LIBRARIES)
            pair,R,t=fixture(); receipt['fixture_input_arrays']=worker.input_hashes(pair)
            receipt['fixture_R']=R.tolist(); receipt['fixture_t']=t.tolist()
            for iterations in (2,4):
                for arm in c.ARMS:
                    options=c.options(arm)
                    options['ransac']['min_iterations']=iterations
                    options['ransac']['max_iterations']=iterations
                    item={'arm':arm,'iterations':iterations,'options':options,'status':'started'}
                    receipt['calls'].append(item); c.write_record(stream,receipt)
                    started=time.perf_counter()
                    result,info=worker.invoke(poselib,pair,arm,options)
                    item['call_seconds']=time.perf_counter()-started
                    item.update(worker.unpack_result(result,info,pair,arm))
                    if item['returned_ransac_stats'].get('iterations') != iterations:
                        raise ValueError('Native iteration count differs from the fixed artificial qualification request')
                    predicted_R=np.array(item['estimate']['R']); predicted_t=np.array(item['estimate']['t'])
                    predicted_t/=np.linalg.norm(predicted_t)
                    item['rotation_error_degrees']=float(np.degrees(np.arccos(np.clip((np.trace(predicted_R@R.T)-1)/2,-1,1))))
                    item['translation_error_degrees']=float(np.degrees(np.arccos(np.clip(predicted_t@(t/np.linalg.norm(t)),-1,1))))
                    item['status']='completed'; c.write_record(stream,receipt)
            receipt['status']='completed'; c.write_record(stream,receipt)
            print(json.dumps({'status':'completed','artificial_calls':len(receipt['calls'])})); return 0
        except BaseException as exc:
            receipt.update(status='error',error=type(exc).__name__+': '+str(exc)); c.write_record(stream,receipt)
            print(receipt['error'],file=sys.stderr); return 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    sys.exit(run(parser.parse_args().output))
