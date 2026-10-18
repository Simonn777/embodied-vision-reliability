#!/usr/bin/env python3
"""Read public Git blobs and invoke the prewritten independent output verifier."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

EXPECTED='f49dbb6a0382aecd02b76dda62ba44fc37aeba2aef128384774ca7c90ec11a0b'
VERIFIER=Path('/tmp/0003-half-certificate-output-verifier.py')
PREFIX=Path('/tmp/0003-half-output-native')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--candidate',required=True)
    args=parser.parse_args()
    assert sha(VERIFIER.read_bytes())==EXPECTED
    root='results/0003-half-cell-analysis-20261017/'
    paths={key:root+filename for key,filename in [('input','input.json'),('claimed-output','output.json'),('data-terms','data-terms.json')]}
    def get(ref,path):return subprocess.check_output(['git','-C',args.repo,'show',ref+':'+path])
    raw={key:get(args.candidate,path) for key,path in paths.items()}
    original_publication='398446b158435abfa96603d22aea0f9ad68b8d28'
    assert all(get(original_publication,paths[key])==raw[key] for key in raw)
    excerpt_path='results/0003-rotation-stageB-20261017/audit/camera-excerpt.json'
    excerpt_raw=get(args.candidate,excerpt_path)
    excerpt=json.loads(excerpt_raw)
    original=json.loads(raw['input'])
    expected_hex=''.join(excerpt['camera_access']['extrinsics']['values'][i][j]['stored_bytes_hex'] for i in range(3) for j in range(3))
    assert original=={'R_hex':expected_hex,'byte_order':'little'}
    for key,data in raw.items():Path(str(PREFIX)+'-'+key+'.json').write_bytes(data)
    command=[sys.executable,'-I',str(VERIFIER),'--original-input',str(PREFIX)+'-input.json','--claimed-output',str(PREFIX)+'-claimed-output.json']
    started=datetime.now(timezone.utc).isoformat()
    start=time.perf_counter()
    child=subprocess.run(command,capture_output=True)
    elapsed=time.perf_counter()-start
    ended=datetime.now(timezone.utc).isoformat()
    stdout_path=Path(str(PREFIX)+'-stdout.json')
    stderr_path=Path(str(PREFIX)+'-stderr.log')
    stdout_path.write_bytes(child.stdout);stderr_path.write_bytes(child.stderr)
    receipt={
        'reviewed_candidate':args.candidate,'original_result_publication':original_publication,
        'command':command,'started_utc':started,'ended_utc':ended,'elapsed_seconds':elapsed,
        'observed_child_returncode':child.returncode,
        'verifier_source_sha256':sha(VERIFIER.read_bytes()),'launcher_source_sha256':sha(Path(__file__).read_bytes()),
        'published_inputs':{key:{'path':paths[key],'sha256':sha(raw[key])} for key in raw},
        'source_camera_excerpt':{'path':excerpt_path,'sha256':sha(excerpt_raw),'R9_stored_bytes_equal_input':True},
        'published_input_output_terms_unchanged_from_original_result_commit':True,
        'stdout_sha256':sha(child.stdout),'stderr_sha256':sha(child.stderr),
        'data_terms':json.loads(raw['data-terms']),
        'boundary':'Only prewritten independent output verifier executed on published JSON. No proposal generation, candidate code import, dataset/cache/archive/model/projection access. Receipt observes verifier child exit, not an additional outer supervisor exit.'}
    receipt_path=Path(str(PREFIX)+'-execution.json')
    save_json(receipt_path,receipt)
    assert child.returncode==0 and child.stderr==b''
    verified=json.loads(child.stdout)
    report={
        'review_status':'APPROVE_accurate_inconclusive_publication',
        'reviewed_candidate':args.candidate,'original_result_publication':original_publication,
        'verified_result':verified,
        'verifier_source_sha256':sha(VERIFIER.read_bytes()),'launcher_source_sha256':sha(Path(__file__).read_bytes()),
        'input_sha256':sha(raw['input']),'claimed_output_sha256':sha(raw['claimed-output']),
        'independent_output_sha256':sha(child.stdout),'execution_receipt_sha256':sha(receipt_path.read_bytes()),
        'data_terms_sha256':sha(raw['data-terms']),
        'source_camera_excerpt_sha256':sha(excerpt_raw),
        'original_R9_binding_verified':True,
        'failed_cell_indices':[item['index'] for item in verified['cell_checks'] if not item['membership']],
        'limits':['The fixed witness is exact SO(3), but one half-cell failure prevents a compatibility certificate.',
                  'An inconclusive single proposal does not exclude every possible SO(3) witness and does not establish actual generation provenance.',
                  'Original strict rotation gate and results remain unchanged; no hypothetical matrix may enter geometry/data qualification.',
                  'Independent checker does not regenerate the quaternion formula or rerun the proposal; source/execution binding is reviewed separately.'],
        'data_terms':json.loads(raw['data-terms'])}
    save_json(Path(str(PREFIX)+'-review.json'),report)
    print(json.dumps({'review_status':report['review_status'],'independently_expected_status':verified['independently_expected_status'],
                      'cells_satisfied':verified['cells_satisfied'],'exact_SO3':verified['exact_SO3'],
                      'failed_cell_indices':report['failed_cell_indices'],'elapsed_seconds':elapsed,
                      'observed_child_returncode':child.returncode},indent=2))

if __name__=='__main__':main()
