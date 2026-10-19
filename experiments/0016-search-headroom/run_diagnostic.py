"""Serial bounded label-blind supervision and global prediction seal."""
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import common as c
import predict_pair as worker
spec=importlib.util.spec_from_file_location('supervisor0013',c.ROOT/'experiments/0013-pose-headroom/run_baseline.py')
# The preserved supervisor imports its sibling by its historical module name.
sys.modules['predict_pair']=c.base
prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
sys.modules['predict_pair']=worker

def run(data,manifest_path,out,old_runtime,current_runtime):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); rows=[]; error=None; hashes=c.source_hashes(); mh=c.sha256(manifest_path)
    c.write_new(out/'started.json',{'source_sha256':hashes,'manifest_sha256':mh,'timeout_seconds':1800,'per_worker_seconds':90,'threads':1,'rss_limit_kib':8*1024*1024})
    try:
        m=c.load_manifest(manifest_path)
        if Path(data).stat().st_size!=m['dataset_bytes'] or c.sha256(data)!=m['dataset_sha256']: raise ValueError('Data binding differs')
        libraries=json.loads(c.LIBRARIES.read_text())
        # Rotate backend order by pair index to reduce fixed order timing effects.
        for i,key in enumerate(m['selected_pair_keys']):
            order=c.BACKENDS[i%3:]+c.BACKENDS[:i%3]
            for backend in order:
                remaining=1800-(time.perf_counter()-started)
                if remaining<=0: raise TimeoutError('Global budget exhausted')
                name='pair-%02d-%s.json'%(i,backend)
                runtime=old_runtime if backend=='old_norm' else current_runtime
                # Current runtime obtains h5py from the old isolated runtime, after current PoseLib.
                os.environ['PYTHONPATH']=os.pathsep.join([str(runtime),str(old_runtime)])
                result=prior.run_child([sys.executable,str(c.HERE/'predict_pair.py'),'--data',str(Path(data).resolve()),
                    '--manifest',str(Path(manifest_path).resolve()),'--output',str((out/name).resolve()),
                    '--backend',backend,'--pair-index',str(i)],out/(name+'.log'),remaining)
                c.write_new(out/(name+'.process.json'),result)
                if result['exit_code']!=0 or result['termination']: raise RuntimeError('Incomplete worker '+name)
                r=json.loads((out/name).read_text())
                expected={'status':'completed','key':key,'pair_index':i,'backend':backend,'manifest_sha256':mh,
                          'source_sha256':hashes,'library':libraries['old' if backend=='old_norm' else 'current']}
                if any(r.get(k)!=v for k,v in expected.items()): raise ValueError('Worker binding differs '+name)
                if [v['seed'] for v in r['candidates']]!=list(c.SEEDS): raise ValueError('Missing candidates')
                rows.append({'filename':name,'sha256':c.sha256(out/name),'pair_index':i,'key':key,'backend':backend,
                             'process_sha256':c.sha256(out/(name+'.process.json'))})
            print(json.dumps({'completed_pairs':i+1,'workers':len(rows),'wall_seconds':time.perf_counter()-started}),flush=True)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
    c.write_new(out/'seal.json',{'schema':'evr0016.seal.v1','status':'completed' if error is None else 'ineligible',
        'error':error,'rows':rows,'source_sha256':hashes,'manifest_sha256':mh,'wall_seconds':time.perf_counter()-started})
    if error: print(error,file=sys.stderr); return 1
    c.verify_seal(manifest_path,out/'seal.json'); return 0

def main():
    p=argparse.ArgumentParser()
    for name in ('data','manifest','output-dir','old-runtime','current-runtime'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); return run(a.data,a.manifest,a.output_dir,a.old_runtime,a.current_runtime)
if __name__=='__main__': sys.exit(main())
