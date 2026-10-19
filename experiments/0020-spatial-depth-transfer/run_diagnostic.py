"""Serial fixed-bank scoring and an 80-pair pre-label seal; technical errors stop."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import common as c
import predict_pair as worker

PAIR_TIMEOUT=30.
TOTAL_TIMEOUT=600.
RSS_LIMIT_KIB=8*1024*1024
THREAD_VARIABLES=c.base_runner.THREAD_VARIABLES
bound_file=c.bound_file


def run_child(command,log_path,remaining_seconds,environment):
    started=time.perf_counter(); deadline=started+min(PAIR_TIMEOUT,remaining_seconds)
    termination,peak_rss,watchdog_error=None,0,None
    process=None; code=None
    with Path(log_path).open('xb') as log:
        try:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            while process.poll() is None:
                if time.perf_counter()>=deadline:
                    termination='pair_timeout' if remaining_seconds>=PAIR_TIMEOUT else 'total_timeout';break
                measured=subprocess.run(['/bin/ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True,timeout=2)
                value=measured.stdout.strip()
                if value.isdigit():
                    peak_rss=max(peak_rss,int(value))
                    if peak_rss>RSS_LIMIT_KIB: termination='rss_limit';break
                time.sleep(.1)
        except BaseException as exc:
            termination='launch_error' if process is None else 'watchdog_error'
            watchdog_error=type(exc).__name__
        finally:
            if process is not None:
                if process.poll() is None:
                    try: process.kill()
                    except ProcessLookupError: pass
                code=process.wait(timeout=5)
    elapsed=time.perf_counter()-started
    if termination is None and elapsed>min(PAIR_TIMEOUT,remaining_seconds):
        termination='pair_timeout_at_exit' if remaining_seconds>=PAIR_TIMEOUT else 'total_timeout_at_exit'
    return {'exit_code':code,'termination':termination,'wall_seconds':elapsed,
            'sampled_peak_rss_kib':peak_rss,'watchdog_error':watchdog_error,
            'rss_sampling':'worker PID ps at about 0.1-second intervals; peaks may be missed'}


def wrap_record(raw,expected,process,candidates):
    c.json_safe(raw)
    if any(raw.get(k)!=v for k,v in expected.items()): raise ValueError('Worker identity or binding differs')
    if raw.get('status')!='completed' or process['exit_code']!=0 or process['termination'] is not None:
        raise ValueError('Technical interruption or incomplete scoring blocks interpretation')
    peak=process.get('sampled_peak_rss_kib')
    if (type(peak) is not int or not 0<=peak<=RSS_LIMIT_KIB or process.get('watchdog_error') is not None
        or not c.np.isfinite(process['wall_seconds']) or not 0<=process['wall_seconds']<=PAIR_TIMEOUT):
        raise ValueError('Worker resource receipt exceeds protocol limits')
    c.check_inputs(raw,candidates)
    c.validate_scores(raw['scores'],candidates,raw['key'])
    historical=sum(r['timings']['charged_total_seconds'] for r in candidates)
    if raw.get('candidate_generation_seconds')!=historical: raise ValueError('Candidate-bank cost differs')
    timings=dict(raw['timings']);timings['charged_total_seconds']=process['wall_seconds']
    keys=('binding_seconds','input_seconds','score_seconds','output_seconds','worker_total_seconds','charged_total_seconds')
    if any(timings.get(k) is None or not c.np.isfinite(timings[k]) or timings[k]<0 for k in keys):
        raise ValueError('Unavailable or invalid completed timing')
    components=sum(timings[k] for k in keys[:4])
    if timings['worker_total_seconds']+1e-6<components or process['wall_seconds']+1e-6<timings['worker_total_seconds']:
        raise ValueError('Timing components exceed enclosing time')
    return {**raw,'timings':timings,'process_receipt':process}


def _verify_payload(manifest_path,baseline_seal,seal,folder):
    manifest,bank,binding=c.baseline_bundle(manifest_path,baseline_seal)
    sources,mh,library=c.source_hashes(),c.sha256(manifest_path),c.library_config()['current']
    if (seal.get('schema')!='evr0020.seal.v1' or seal.get('status')!='completed' or seal.get('error') is not None
        or seal.get('source_sha256')!=sources or seal.get('manifest_sha256')!=mh
        or seal.get('library')!=library or seal.get('baseline_binding')!=binding):
        raise ValueError('Incomplete or changed scoring seal')
    if (type(seal.get('wall_seconds')) not in (int,float) or not c.np.isfinite(seal['wall_seconds'])
        or not 0<=seal['wall_seconds']<=TOTAL_TIMEOUT):
        raise ValueError('Scoring cohort exceeds its wall-time budget')
    assets={r['filename']:{'bytes':r['dataset_bytes'],'sha256':r['dataset_sha256']} for r in manifest['selected_pairs']}
    integrity={'manifest_sha256':mh,'source_sha256':sources,'library':library,'baseline_binding':binding,'verified_assets':assets}
    path=bound_file(folder,'integrity.json')
    if c.sha256(path)!=seal['integrity_sha256'] or json.loads(path.read_text())!=integrity or seal['verified_assets']!=assets:
        raise ValueError('Integrity receipt differs')
    rows=seal.get('rows',[])
    if [r.get('pair_index') for r in rows]!=list(range(80)): raise ValueError('Incomplete or reordered 80-pair cohort')
    records=[];fits=0
    for row in rows:
        i=row['pair_index']; paths={k:bound_file(folder,row[k]) for k in ('path','raw_path','process_path')}
        for key,digest in (('path','sha256'),('raw_path','raw_sha256'),('process_path','process_sha256')):
            if c.sha256(paths[key])!=row[digest]: raise ValueError('Sealed scoring file changed')
        process=json.loads(paths['process_path'].read_text())
        if (process['raw_worker_path']!=row['raw_path'] or process['raw_worker_sha256']!=row['raw_sha256']
            or c.sha256(bound_file(folder,process['log_path']))!=process['log_sha256']):
            raise ValueError('Process/raw/log binding differs')
        raw=json.loads(paths['raw_path'].read_text())
        final=wrap_record(raw,worker.identity(manifest['selected_pairs'][i],mh,sources,library,binding),process,bank[i])
        if final!=json.loads(paths['path'].read_text()) or row['status']!='completed': raise ValueError('Final scoring record differs')
        if any(row[k]!=final[k] for k in ('scene','key')): raise ValueError('Seal identity differs')
        fits+=c.fit_count(final['scores']);records.append(final)
    if fits>3200 or seal.get('fit_attempts')!=fits: raise ValueError('Fit accounting/cap differs')
    return manifest,seal,records


def verify_seal(manifest_path,baseline_seal,seal_path):
    path=Path(seal_path);bound_file(path.parent,path.name)
    return _verify_payload(manifest_path,baseline_seal,json.loads(path.read_text()),path.parent)


def run(data_dir,manifest_path,baseline_seal,output_dir,dependency_runtime):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();rows=[];assets={};sources={};library={};binding={};mh=None;ih=None;error=None;fits=0
    c.write_new(out/'started.json',{'schema':'evr0020.operation.v1','status':'started','pair_timeout_seconds':PAIR_TIMEOUT,
        'total_timeout_seconds':TOTAL_TIMEOUT,'rss_limit_kib':RSS_LIMIT_KIB,'threads':1,'max_fits_per_pair':40,'max_total_fits':3200})
    try:
        manifest,bank,binding=c.baseline_bundle(manifest_path,baseline_seal)
        sources,mh,library=c.source_hashes(),c.sha256(manifest_path),c.library_config()['current']
        for row in manifest['selected_pairs']:
            if row['filename'] in assets:continue
            path=bound_file(data_dir,row['filename'])
            if path.stat().st_size!=row['dataset_bytes'] or c.sha256(path)!=row['dataset_sha256']:
                raise ValueError('Dataset size/hash differs before scoring')
            assets[row['filename']]={'bytes':row['dataset_bytes'],'sha256':row['dataset_sha256']}
        c.write_new(out/'integrity.json',{'manifest_sha256':mh,'source_sha256':sources,'library':library,
                                        'baseline_binding':binding,'verified_assets':assets})
        ih=c.sha256(out/'integrity.json')
        env=dict(os.environ);env.update({k:'1' for k in THREAD_VARIABLES});env['PYTHONPATH']=str(Path(dependency_runtime).resolve())
        for row in manifest['selected_pairs']:
            i=row['pair_index'];remaining=TOTAL_TIMEOUT-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('Global scoring budget exhausted')
            stem=f'pair-{i:02d}'
            raw_path,pp,fp=(out/(stem+s) for s in ('.worker.json','.process.json','.json'))
            log=out/(stem+'.log')
            command=[sys.executable,str(c.HERE/'predict_pair.py'),'--data',str(bound_file(data_dir,row['filename']).resolve()),
                '--manifest',str(Path(manifest_path).resolve()),'--baseline-seal',str(Path(baseline_seal).resolve()),
                '--output',str(raw_path.resolve()),'--pair-index',str(i)]
            process=run_child(command,log,remaining,env)
            process.update(raw_worker_path=raw_path.name,raw_worker_sha256=c.sha256(raw_path) if raw_path.exists() else None,
                           log_path=log.name,log_sha256=c.sha256(log))
            c.write_new(pp,process)
            raw=json.loads(raw_path.read_text())
            final=wrap_record(raw,worker.identity(row,mh,sources,library,binding),process,bank[i])
            fits+=c.fit_count(final['scores'])
            if fits>3200:raise ValueError('Cohort fit cap exceeded')
            c.write_new(fp,final)
            rows.append({'pair_index':i,'scene':row['scene'],'key':row['key'],'status':'completed','path':fp.name,'sha256':c.sha256(fp),
                'raw_path':raw_path.name,'raw_sha256':c.sha256(raw_path),'process_path':pp.name,'process_sha256':c.sha256(pp)})
            print(json.dumps({'completed_pairs':len(rows),'fit_attempts':fits,'wall_seconds':time.perf_counter()-started}),flush=True)
        if c.source_hashes()!=sources or c.sha256(manifest_path)!=mh:raise ValueError('Source/manifest changed during scoring')
        for name,expected in assets.items():
            path=bound_file(data_dir,name)
            if path.stat().st_size!=expected['bytes'] or c.sha256(path)!=expected['sha256']:raise ValueError('Dataset changed during scoring')
        if time.perf_counter()-started>TOTAL_TIMEOUT:raise TimeoutError('Global scoring budget exhausted')
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        for private in (str(Path(data_dir).resolve()),str(out.resolve()),str(c.ROOT.resolve())):error=error.replace(private,'<local-path>')
    completed=len(rows)==80 and error is None
    seal={'schema':'evr0020.seal.v1','status':'completed' if completed else 'ineligible','error':error,'rows':rows,
          'source_sha256':sources,'manifest_sha256':mh,'library':library,'baseline_binding':binding,
          'verified_assets':assets,'integrity_sha256':ih,'fit_attempts':fits,'wall_seconds':time.perf_counter()-started}
    if completed:
        verify_started=time.perf_counter()
        try:
            _verify_payload(manifest_path,baseline_seal,seal,out)
            if time.perf_counter()-started>TOTAL_TIMEOUT:raise TimeoutError('Global budget exhausted during seal verification')
        except BaseException as exc:
            completed=False;seal.update(status='ineligible',error=type(exc).__name__+': '+str(exc))
        seal['seal_verification_seconds']=time.perf_counter()-verify_started
    seal['wall_seconds']=time.perf_counter()-started
    c.write_new(out/'seal.json',seal)
    if not completed:print(seal['error'],file=sys.stderr)
    return 0 if completed else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','manifest','baseline-seal','output-dir','dependency-runtime'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();return run(a.data_dir,a.manifest,a.baseline_seal,a.output_dir,a.dependency_runtime)

if __name__=='__main__':sys.exit(main())
