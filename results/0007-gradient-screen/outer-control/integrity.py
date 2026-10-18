"""Independent saved-evidence checks; never reads scientific input arrays."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0007-gradient';OUT=ROOT/'antithetic-gradient-0007';CONTROL=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':a.dtype.str},sort_keys=True,separators=(',',':')).encode('ascii')+b'\0');h.update(memoryview(a).cast('B'));return h.hexdigest()
def main():
    before=read(CONTROL/'pre-execution.json');cfg=read(REPO/'configs/0007-gradient-screen.json')
    started=read(OUT/'study.started.json');complete=read(OUT/'study.completed.json')
    assert not (OUT/'study.failed.json').exists()
    assert started['inputs']==complete['inputs_after']==before['cache_inputs']
    assert started['source_commit']==complete['source_commit']==before['source_commit']
    assert started['freeze_commit']==before['freeze_commit'] and started['freeze_sha256']==before['freeze_sha256']
    for label in ('scientific-run','runtime-tests','frozen-verify'):
        ex=read(CONTROL/(label+'.execution.json'));assert ex['actual_exit']==ex['raw_wait_status']==0
        for suffix,key in (('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256'),('.command.private.json','command_file_sha256')):assert sha(CONTROL/(label+suffix))==ex[key]
    tests=(CONTROL/'runtime-tests.stderr.log').read_text();assert 'Ran 43 tests' in tests and '\nOK\n' in tests and 'skipped' not in tests.lower()
    rows=[]
    for c,item in enumerate(cfg['cases']):
        folder=OUT/'cases'/item['name'];r=read(folder/'receipt.json');p=OUT/'control'/item['name'];ex=read(p.with_suffix('.execution.json'))
        assert complete['cases'][c]=={'case':item['name'],**ex}
        assert ex['actual_exit']==ex['raw_wait_status']==0 and ex['hard_timeout'] is False
        for suffix,key in (('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256'),('.command.private.json','command_file_sha256')):assert sha(p.with_suffix(suffix))==ex[key]
        assert ex['receipt_sha256']==sha(folder/'receipt.json')
        stdout=json.loads(p.with_suffix('.stdout.log').read_text());assert stdout=={'case':item['name'],'receipt_sha256':sha(folder/'receipt.json'),'completed':True}
        assert r['case_identity']==item and r['status']=='completed' and r['failure'] is None
        assert r['source_commit']==before['source_commit'] and r['freeze_commit']==before['freeze_commit'] and r['freeze_sha256']==before['freeze_sha256']
        assert r['weight_unchanged'] is True and r['weight_before_sha256']==r['weight_after_sha256']==item['weight_content_sha256']
        assert r['optimizer_steps']==0 and r['completed_measurements']==r['expected_measurements']==512
        assert r['source_sha256']==before['frozen_files']['scripts/antithetic_gradient_core.py']
        a={}
        for name,shape,dtype,key in (('gradients',(2,4,64,5313),'float32','gradient_array_sha256'),('walls',(2,4,64),'float64','walls_sha256'),('losses',(2,4,64),'float64','losses_sha256'),('completed',(2,4,64),'bool','completed_mask_sha256')):
            path=folder/(name+'.npy');assert not path.is_symlink() and sha(path)==r['files'][name+'.npy']
            a[name]=np.load(path,allow_pickle=False,mmap_mode='r')
            assert a[name].shape==shape and a[name].dtype==np.dtype(dtype) and np.isfinite(a[name]).all() and ash(a[name])==r[key]
        assert a['completed'].all() and (a['walls']>0).all()
        seen=set()
        for record in r['records']:
            pos=(record['regime'],record['method_index'],record['replicate']);assert pos not in seen;seen.add(pos)
            assert record['gradient_sha256']==ash(a['gradients'][pos]) and record['loss']==a['losses'][pos] and record['wall_seconds']==a['walls'][pos]
            assert record['forward_calls']==record['backward_calls']==1 and record['status']=='completed'
        assert len(seen)==512 and len(r['records'])==512
        cmd=read(p.with_suffix('.command.private.json'));assert cmd['cwd']==str(REPO)
        assert cmd['argv'][2]=='case' and cmd['argv'][-1]==str(c)
        rows.append({'case':item['name'],'measurements':len(seen),'gradient_file_sha256':r['files']['gradients.npy'],'gradient_content_sha256':r['gradient_array_sha256'],'receipt_sha256':sha(folder/'receipt.json'),'case_wall_seconds':ex['wall_seconds'],'peak_rss_kib':ex['peak_rss_kib'],'log_bindings_passed':True,'all_completed':True,'parameters_unchanged':True})
    for item in before['cache_inputs']:assert sha(ROOT/'conditional-residual-0006'/item['relative'])==item['sha256']
    for name,h in before['tracked'].items():assert sha(REPO/name)==h,name
    for name,h in before['old_failed_cache'].items():assert sha(ROOT/'tapnextpp-cache'/name)==h,name
    for name,h in before['old_markers'].items():assert sha(ROOT/name)==h,name
    result={'status':'PASS','cases':rows,'total_measurements':sum(r['measurements'] for r in rows),'optimizer_updates':0,'cached_inputs_preserved':len(before['cache_inputs']),'prior_tracked_preserved':len(before['tracked']),'prior_result_files_preserved':sum(n.startswith('results/') for n in before['tracked']),'old_failed_cache_files_preserved':len(before['old_failed_cache']),'old_marker_files_preserved':len(before['old_markers']),'baseline_sha256':sha(CONTROL/'pre-execution.json'),'checker_source_sha256':sha(__file__),'new_asset_bytes':0,'no_clean_reference_or_old_prediction_access':True}
    with (CONTROL/'integrity.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
