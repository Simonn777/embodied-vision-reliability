"""Serial, bounded 80 x 4 prediction supervision; no reference-pose access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import common as c
import predict_pair as worker

PAIR_TIMEOUT = 90.
TOTAL_TIMEOUT = 1800.
RSS_LIMIT_KIB = 8*1024*1024
THREAD_VARIABLES = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def run_child(command, log_path, remaining_seconds, environment):
    started = time.perf_counter()
    deadline = started+min(PAIR_TIMEOUT, remaining_seconds)
    termination, peak_rss, watchdog_error = None, 0, None
    with Path(log_path).open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                if time.perf_counter() >= deadline:
                    termination = 'pair_timeout' if remaining_seconds >= PAIR_TIMEOUT else 'total_timeout'
                    break
                measured = subprocess.run(['/bin/ps','-o','rss=','-p',str(process.pid)],
                                          capture_output=True, text=True, timeout=2)
                value = measured.stdout.strip()
                if value.isdigit():
                    peak_rss = max(peak_rss, int(value))
                    if peak_rss > RSS_LIMIT_KIB:
                        termination = 'rss_limit'; break
                time.sleep(.1)
        except Exception as exc:
            termination, watchdog_error = 'watchdog_error', type(exc).__name__
        finally:
            if process.poll() is None:
                process.kill()
            code = process.wait(timeout=5)
    return {'exit_code':code, 'termination':termination, 'wall_seconds':time.perf_counter()-started,
            'sampled_peak_rss_kib':peak_rss, 'watchdog_error':watchdog_error,
            'rss_sampling':'ps at about 0.1-second intervals; peaks may be missed'}


def identity(row, arm, manifest_hash, sources, library):
    return {**{k:row[k] for k in ('pair_index','scene','key','filename')}, 'arm':arm,
            'schema':worker.SCHEMA, 'manifest_sha256':manifest_hash, 'source_sha256':sources,
            'dataset_sha256_declared':row['dataset_sha256'], 'dataset_sha256_verified':row['dataset_sha256'],
            'dataset_bytes':row['dataset_bytes'], 'options':c.options(arm), 'library':library}


def check_record(record, expected):
    if any(record.get(k) != value for k,value in expected.items()):
        raise ValueError('Worker identity/source/options/library binding mismatch')
    c.json_safe(record)
    if record.get('status') == 'completed':
        c.validate_pose(record['estimate'])
        n = record['input_arrays']['x1']['shape'][0]
        if record.get('mask_length') != n:
            raise ValueError('Returned mask length mismatch')
        packed = bytes.fromhex(record['native_inliers_packbits_big_hex'])
        if len(packed) != (n+7)//8:
            raise ValueError('Returned packed mask length mismatch')
        bits = c.np.unpackbits(c.np.frombuffer(packed,dtype=c.np.uint8),bitorder='big')
        if bits[n:].any(): raise ValueError('Nonzero mask padding')
        mask = bits[:n].astype(c.np.bool_)
        if int(mask.sum()) != record['kept_count'] or hashlib.sha256(mask.tobytes()).hexdigest() != record['mask_sha256']:
            raise ValueError('Returned mask hash/count mismatch')
        score = record['final_scores']['pixel_msac']
        if not 0 <= score <= n: raise ValueError('Invalid final score')
        if record['arm'].startswith('depth_'):
            geometry = record['geometry']
            if any(geometry[k] != record['estimate'][k] for k in ('R','t')):
                raise ValueError('Raw geometry/estimate differs')
            for name in ('scale','shift1','shift2'): float(geometry[name])
            if record['depth_diagnostics'] is None: raise ValueError('Missing depth diagnostics')


def wrap_record(raw, expected, process):
    check_record(raw, expected)
    status = raw.get('status')
    if status == 'completed' and process['exit_code'] == 0 and process['termination'] is None:
        final_status = 'completed'
    elif status == 'error' and raw.get('failure_kind') == 'estimator' and process['exit_code'] == 1 and process['termination'] is None:
        final_status = 'estimator_error'
    elif status == 'started' and raw.get('phase') == 'estimator':
        if process['termination'] == 'pair_timeout': final_status = 'timeout'
        elif process['termination'] is None and process['exit_code'] != 0: final_status = 'estimator_error'
        else: raise ValueError('Resource interruption is not evaluation eligible')
    else:
        raise ValueError('Non-estimator failure or inconsistent worker exit')
    timings = {**raw.get('timings', {})}
    timings.setdefault('input_seconds', None)
    timings.setdefault('call_seconds', None)
    timings.setdefault('score_seconds', None)
    timings['worker_total_seconds'] = timings.get('charged_total_seconds')
    timings['charged_total_seconds'] = process['wall_seconds']
    return {**raw, 'status':final_status, 'process_receipt':process, 'timings':timings}


def bound_file(folder, filename):
    if not isinstance(filename,str) or Path(filename).name != filename:
        raise ValueError('Unsafe record path')
    path = Path(folder)/filename
    if path.is_symlink() or path.resolve().parent != Path(folder).resolve():
        raise ValueError('Unsafe record symlink')
    return path


def _verify_payload(manifest_path, seal, folder):
    manifest = c.load_manifest(manifest_path)
    sources, mh = c.source_hashes(), c.sha256(manifest_path)
    library = json.loads(c.LIBRARIES.read_text())['current']
    if (seal.get('schema') != 'evr0017.seal.v1' or seal.get('status') != 'completed'
        or seal.get('source_sha256') != sources or seal.get('manifest_sha256') != mh or seal.get('library') != library):
        raise ValueError('Incomplete or changed seal bindings')
    integrity_path = bound_file(folder, 'integrity.json')
    if c.sha256(integrity_path) != seal['integrity_sha256']: raise ValueError('Integrity receipt differs')
    integrity = json.loads(integrity_path.read_text())
    if integrity != {'manifest_sha256':mh,'source_sha256':sources,'library':library,'verified_assets':seal['verified_assets']}:
        raise ValueError('Integrity bindings differ')
    wanted_assets = {r['filename']:{'bytes':r['dataset_bytes'],'sha256':r['dataset_sha256']} for r in manifest['selected_pairs']}
    if seal['verified_assets'] != wanted_assets: raise ValueError('Asset verification differs')
    wanted = {(r['pair_index'],arm) for r in manifest['selected_pairs'] for arm in c.ARMS}
    rows, records, seen, shared = seal.get('rows',[]), [], set(), {}
    if len(rows) != 320: raise ValueError('Incomplete 320-arm cohort')
    for row in rows:
        item = (row['pair_index'], row['arm'])
        if item not in wanted or item in seen: raise ValueError('Duplicate or unknown arm')
        seen.add(item)
        manifest_row = manifest['selected_pairs'][item[0]]
        if any(row[k] != manifest_row[k] for k in ('scene','key')): raise ValueError('Seal row identity mismatch')
        path = bound_file(folder, row['path'])
        raw_path = bound_file(folder, row['raw_path'])
        process_path = bound_file(folder, row['process_path'])
        for file, key in ((path,'sha256'),(raw_path,'raw_sha256'),(process_path,'process_sha256')):
            if c.sha256(file) != row[key]: raise ValueError('Sealed file hash differs')
        process = json.loads(process_path.read_text())
        log_path = bound_file(folder, process['log_path'])
        if c.sha256(log_path) != process['log_sha256']: raise ValueError('Sealed process log differs')
        if process.get('raw_worker_sha256') != row['raw_sha256'] or process.get('raw_worker_path') != row['raw_path']:
            raise ValueError('Process/raw linkage differs')
        expected = identity(manifest['selected_pairs'][item[0]], item[1], mh, sources, library)
        reconstructed = wrap_record(json.loads(raw_path.read_text()), expected, process)
        record = json.loads(path.read_text())
        if record != reconstructed or record['status'] != row['status']:
            raise ValueError('Final record classification differs')
        inputs = {k:record[k] for k in ('input_arrays','calibration','input_diagnostics')}
        if item[0] in shared and shared[item[0]] != inputs:
            raise ValueError('Four arms did not use identical input bindings')
        shared[item[0]] = inputs
        records.append(record)
    records.sort(key=lambda r:(r['pair_index'],c.ARMS.index(r['arm'])))
    return manifest, seal, records


def verify_seal(manifest_path, seal_path):
    seal_path = Path(seal_path)
    bound_file(seal_path.parent, seal_path.name)
    return _verify_payload(manifest_path, json.loads(seal_path.read_text()), seal_path.parent)


def run(data_dir, manifest_path, output_dir, current_runtime, dependency_runtime):
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter(); rows=[]; error=None; mh=None; sources={}; library={}; assets={}; integrity_hash=None
    c.write_new(out/'started.json', {'schema':'evr0017.operation.v1','status':'started',
        'pair_timeout_seconds':PAIR_TIMEOUT,'total_timeout_seconds':TOTAL_TIMEOUT,'rss_limit_kib':RSS_LIMIT_KIB,'threads':1})
    try:
        manifest = c.load_manifest(manifest_path); mh=c.sha256(manifest_path); sources=c.source_hashes()
        library=json.loads(c.LIBRARIES.read_text())['current']
        for row in manifest['selected_pairs']:
            if row['filename'] in assets: continue
            path=bound_file(data_dir,row['filename'])
            if path.stat().st_size != row['dataset_bytes'] or c.sha256(path) != row['dataset_sha256']:
                raise ValueError('Dataset content binding differs')
            assets[row['filename']]={'bytes':row['dataset_bytes'],'sha256':row['dataset_sha256']}
        c.write_new(out/'integrity.json',{'manifest_sha256':mh,'source_sha256':sources,'library':library,'verified_assets':assets})
        integrity_hash=c.sha256(out/'integrity.json')
        environment=dict(os.environ)
        environment.update({name:'1' for name in THREAD_VARIABLES})
        environment['PYTHONPATH']=os.pathsep.join((str(Path(current_runtime).resolve()),str(Path(dependency_runtime).resolve())))
        for pair in manifest['selected_pairs']:
            index=pair['pair_index']; offset=index%4
            for arm in c.ARMS[offset:]+c.ARMS[:offset]:
                remaining=TOTAL_TIMEOUT-(time.perf_counter()-started)
                if remaining<=0: raise TimeoutError('Global cohort budget exhausted')
                stem='pair-%02d-%s'%(index,arm); raw_path=out/(stem+'.worker.json')
                command=[sys.executable,str(c.HERE/'predict_pair.py'),'--data',str(bound_file(data_dir,pair['filename']).resolve()),
                    '--manifest',str(Path(manifest_path).resolve()),'--output',str(raw_path.resolve()),'--arm',arm,'--pair-index',str(index)]
                log_path=out/(stem+'.log')
                process=run_child(command,log_path,remaining,environment)
                process.update(raw_worker_path=raw_path.name,raw_worker_sha256=c.sha256(raw_path) if raw_path.exists() else None,
                               log_path=log_path.name,log_sha256=c.sha256(log_path))
                process_path=out/(stem+'.process.json'); c.write_new(process_path,process)
                raw=json.loads(raw_path.read_text())
                final=wrap_record(raw,identity(pair,arm,mh,sources,library),process)
                final_path=out/(stem+'.json'); c.write_new(final_path,final)
                rows.append({'pair_index':index,'scene':pair['scene'],'key':pair['key'],'arm':arm,'path':final_path.name,
                    'sha256':c.sha256(final_path),'status':final['status'],'raw_path':raw_path.name,'raw_sha256':c.sha256(raw_path),
                    'process_path':process_path.name,'process_sha256':c.sha256(process_path)})
            print(json.dumps({'completed_pairs':index+1,'arms':len(rows),'wall_seconds':time.perf_counter()-started}),flush=True)
        if c.source_hashes()!=sources or c.sha256(manifest_path)!=mh: raise ValueError('Source/manifest changed')
        for filename, expected in assets.items():
            path=bound_file(data_dir,filename)
            if path.stat().st_size!=expected['bytes'] or c.sha256(path)!=expected['sha256']:
                raise ValueError('Dataset changed during prediction')
        if time.perf_counter()-started > TOTAL_TIMEOUT: raise TimeoutError('Global cohort budget exhausted')
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        for private in (str(Path(data_dir).resolve()),str(out.resolve()),str(c.ROOT.resolve())):
            error=error.replace(private,'<local-path>')
    completed=len(rows)==320 and error is None
    seal={'schema':'evr0017.seal.v1','status':'completed' if completed else 'ineligible',
        'error':error,'rows':rows,'manifest_sha256':mh,'source_sha256':sources,'library':library,'verified_assets':assets,
        'integrity_sha256':integrity_hash,'wall_seconds':time.perf_counter()-started}
    if completed:
        try:
            _verify_payload(manifest_path,seal,out)
        except BaseException as exc:
            completed=False
            seal.update(status='ineligible',error=type(exc).__name__+': '+str(exc))
    c.write_new(out/'seal.json',seal)
    if not completed: print(seal['error'],file=sys.stderr)
    return 0 if completed else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','manifest','output-dir','current-runtime','dependency-runtime'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    return run(args.data_dir,args.manifest,args.output_dir,args.current_runtime,args.dependency_runtime)


if __name__=='__main__':
    sys.exit(main())
