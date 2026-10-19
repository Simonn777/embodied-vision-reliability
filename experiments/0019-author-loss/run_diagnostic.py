"""Serial 80 x 2 author-loss control with full-cohort pre-final parity; no GT."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import common as c
import predict_pair as worker

PAIR_TIMEOUT = 90.
TOTAL_TIMEOUT = 1800.
RSS_LIMIT_KIB = 8*1024*1024
THREAD_VARIABLES = c.base_runner.THREAD_VARIABLES
run_child = c.base_runner.run_child
bound_file = c.bound_file


def wrap_record(raw, expected, process, baseline):
    c.base_runner.check_record(raw, expected)
    c.check_inputs(raw, baseline)
    status, code, termination = raw.get('status'), process['exit_code'], process['termination']
    if status == 'completed' and code == 0 and termination is None:
        final_status = 'completed'
        parity = c.check_parity(raw, baseline)
        if raw.get('prefinal_parity') != parity:
            raise ValueError('Worker pre-final parity receipt differs')
    elif (status == 'error' and raw.get('failure_kind') == 'estimator'
          and raw.get('failed_phase') == 'estimator' and code == 1 and termination is None):
        final_status = 'estimator_error'
    elif status == 'started' and raw.get('phase') == 'estimator':
        if termination == 'pair_timeout': final_status = 'timeout'
        elif termination is None and code != 0: final_status = 'estimator_error'
        else: raise ValueError('Resource interruption is not evaluation eligible')
    else:
        raise ValueError('Non-estimator failure or inconsistent worker exit')
    if final_status != 'completed':
        parity = {'available': False, 'passed': None, 'reason': final_status,
                  'final_geometry_constrained': False}
    timings = dict(raw['timings'])
    timings['charged_total_seconds'] = process['wall_seconds']
    if not c.np.isfinite(process['wall_seconds']) or process['wall_seconds'] < 0:
        raise ValueError('Invalid charged process time')
    for key in ('binding_seconds', 'input_seconds', 'call_seconds', 'score_seconds', 'worker_total_seconds'):
        value = timings.get(key)
        if value is None:
            if final_status == 'completed' or key in ('binding_seconds', 'input_seconds'):
                raise ValueError('Required completed timing unavailable')
        elif not c.np.isfinite(value) or value < 0:
            raise ValueError('Invalid timing component')
        timings.setdefault(key, None)
    measured = sum(timings[k] or 0. for k in ('binding_seconds', 'input_seconds', 'call_seconds', 'score_seconds'))
    whole = timings['worker_total_seconds']
    if (whole is not None and (whole+1e-6 < measured or process['wall_seconds']+1e-6 < whole)) or process['wall_seconds']+1e-6 < measured:
        raise ValueError('Timing components exceed enclosing wall time')
    return {**raw, 'status': final_status, 'prefinal_parity': parity,
            'timings': timings, 'process_receipt': process}


def _verify_payload(manifest_path, baseline_seal, seal, folder):
    manifest, baseline, baseline_binding = c.baseline_bundle(manifest_path, baseline_seal)
    mh, sources = c.sha256(manifest_path), c.source_hashes()
    library = c.library_config()['current']
    if (seal.get('schema') != 'evr0019.seal.v1' or seal.get('status') != 'completed'
        or seal.get('error') is not None or seal.get('source_sha256') != sources
        or seal.get('manifest_sha256') != mh or seal.get('library') != library
        or seal.get('baseline_binding') != baseline_binding):
        raise ValueError('Incomplete or changed author-loss seal bindings')
    expected_assets = {r['filename']: {'bytes': r['dataset_bytes'], 'sha256': r['dataset_sha256']}
                       for r in manifest['selected_pairs']}
    expected_integrity = {'manifest_sha256': mh, 'source_sha256': sources, 'library': library,
                          'baseline_binding': baseline_binding, 'verified_assets': expected_assets}
    ip = bound_file(folder, 'integrity.json')
    if c.sha256(ip) != seal['integrity_sha256'] or json.loads(ip.read_text()) != expected_integrity:
        raise ValueError('Integrity receipt differs')
    if seal['verified_assets'] != expected_assets:
        raise ValueError('Verified asset declarations differ')
    expected_order = [(i, a) for i in range(80) for a in c.ARMS[i % 2:]+c.ARMS[:i % 2]]
    rows = seal.get('rows', [])
    if [(r.get('pair_index'), r.get('arm')) for r in rows] != expected_order:
        raise ValueError('Incomplete or reordered 160-replay cohort')
    records = []
    for row in rows:
        i, arm = row['pair_index'], row['arm']
        files = {k: bound_file(folder, row[k]) for k in ('path', 'raw_path', 'process_path')}
        for key, digest in (('path', 'sha256'), ('raw_path', 'raw_sha256'), ('process_path', 'process_sha256')):
            if c.sha256(files[key]) != row[digest]:
                raise ValueError('Sealed replay file changed')
        process = json.loads(files['process_path'].read_text())
        if (process['raw_worker_path'] != row['raw_path'] or process['raw_worker_sha256'] != row['raw_sha256']
            or c.sha256(bound_file(folder, process['log_path'])) != process['log_sha256']):
            raise ValueError('Process/raw/log binding differs')
        expected = worker.identity(manifest['selected_pairs'][i], arm, mh, sources, library, baseline_binding)
        raw = json.loads(files['raw_path'].read_text())
        final = wrap_record(raw, expected, process, baseline[i, arm])
        if final != json.loads(files['path'].read_text()) or row['status'] != final['status']:
            raise ValueError('Final replay record differs')
        if any(row[k] != final[k] for k in ('scene', 'key')):
            raise ValueError('Seal row linkage differs')
        records.append(final)
    completed = sum(r['status'] == 'completed' for r in records)
    if (seal.get('all_completed_prefinal_parity_passed') is not True
        or seal.get('completed_calls') != completed or seal.get('native_failure_calls') != 160-completed):
        raise ValueError('Cohort pre-final parity coverage differs')
    records.sort(key=lambda r: (r['pair_index'], c.ARMS.index(r['arm'])))
    return manifest, seal, records


def verify_seal(manifest_path, baseline_seal, seal_path):
    path = Path(seal_path)
    bound_file(path.parent, path.name)
    return _verify_payload(manifest_path, baseline_seal, json.loads(path.read_text()), path.parent)


def run(data_dir, manifest_path, baseline_seal, output_dir, runtime, dependency_runtime):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, assets, sources, library, baseline_binding = [], {}, {}, {}, {}
    mh, integrity_hash, error = None, None, None
    c.write_new(out/'started.json', {'schema': 'evr0019.operation.v1', 'status': 'started',
        'pair_timeout_seconds': PAIR_TIMEOUT, 'total_timeout_seconds': TOTAL_TIMEOUT,
        'rss_limit_kib': RSS_LIMIT_KIB, 'threads': 1})
    try:
        manifest, baseline, baseline_binding = c.baseline_bundle(manifest_path, baseline_seal)
        mh, sources = c.sha256(manifest_path), c.source_hashes()
        library = c.library_config()['current']
        for row in manifest['selected_pairs']:
            if row['filename'] in assets:
                continue
            path = bound_file(data_dir, row['filename'])
            if path.stat().st_size != row['dataset_bytes'] or c.sha256(path) != row['dataset_sha256']:
                raise ValueError('Dataset hash/size differs before replay')
            assets[row['filename']] = {'bytes': row['dataset_bytes'], 'sha256': row['dataset_sha256']}
        c.write_new(out/'integrity.json', {'manifest_sha256': mh, 'source_sha256': sources, 'library': library,
            'baseline_binding': baseline_binding, 'verified_assets': assets})
        integrity_hash = c.sha256(out/'integrity.json')
        environment = dict(os.environ)
        environment.update({name: '1' for name in THREAD_VARIABLES})
        environment['PYTHONPATH'] = os.pathsep.join((str(Path(runtime).resolve()), str(Path(dependency_runtime).resolve())))
        for pair in manifest['selected_pairs']:
            i = pair['pair_index']
            for arm in c.ARMS[i % 2:]+c.ARMS[:i % 2]:
                remaining = TOTAL_TIMEOUT-(time.perf_counter()-started)
                if remaining <= 0:
                    raise TimeoutError('Global replay budget exhausted')
                stem = f'pair-{i:02d}-{arm}'
                raw_path, process_path, final_path = (out/(stem+suffix) for suffix in ('.worker.json', '.process.json', '.json'))
                command = [sys.executable, str(c.HERE/'predict_pair.py'), '--data', str(bound_file(data_dir, pair['filename']).resolve()),
                    '--manifest', str(Path(manifest_path).resolve()), '--baseline-seal', str(Path(baseline_seal).resolve()),
                    '--output', str(raw_path.resolve()), '--arm', arm, '--pair-index', str(i)]
                log_path = out/(stem+'.log')
                process = run_child(command, log_path, remaining, environment)
                process.update(raw_worker_path=raw_path.name, raw_worker_sha256=c.sha256(raw_path) if raw_path.exists() else None,
                               log_path=log_path.name, log_sha256=c.sha256(log_path))
                c.write_new(process_path, process)
                raw = json.loads(raw_path.read_text())
                final = wrap_record(raw, worker.identity(pair, arm, mh, sources, library, baseline_binding), process, baseline[i, arm])
                c.write_new(final_path, final)
                rows.append({'pair_index': i, 'scene': pair['scene'], 'key': pair['key'], 'arm': arm,
                    'path': final_path.name, 'sha256': c.sha256(final_path), 'status': final['status'],
                    'raw_path': raw_path.name, 'raw_sha256': c.sha256(raw_path),
                    'process_path': process_path.name, 'process_sha256': c.sha256(process_path)})
            print(json.dumps({'completed_pairs': i+1, 'replays': len(rows),
                              'wall_seconds': time.perf_counter()-started}), flush=True)
        if c.source_hashes() != sources or c.sha256(manifest_path) != mh:
            raise ValueError('Sources or manifest changed')
        for filename, expected in assets.items():
            path = bound_file(data_dir, filename)
            if path.stat().st_size != expected['bytes'] or c.sha256(path) != expected['sha256']:
                raise ValueError('Dataset changed during replay')
        if time.perf_counter()-started > TOTAL_TIMEOUT:
            raise TimeoutError('Global replay budget exhausted')
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
        for private in (str(Path(data_dir).resolve()), str(out.resolve()), str(c.ROOT.resolve())):
            error = error.replace(private, '<local-path>')
    completed = len(rows) == 160 and error is None
    seal = {'schema': 'evr0019.seal.v1', 'status': 'completed' if completed else 'ineligible',
            'error': error, 'rows': rows, 'manifest_sha256': mh, 'source_sha256': sources,
            'library': library, 'baseline_binding': baseline_binding, 'verified_assets': assets,
            'integrity_sha256': integrity_hash,
            'completed_calls': sum(r['status'] == 'completed' for r in rows),
            'native_failure_calls': sum(r['status'] != 'completed' for r in rows),
            'all_completed_prefinal_parity_passed': completed, 'wall_seconds': time.perf_counter()-started}
    if completed:
        verification_started = time.perf_counter()
        try:
            _verify_payload(manifest_path, baseline_seal, seal, out)
            if time.perf_counter()-started > TOTAL_TIMEOUT:
                raise TimeoutError('Global replay budget exhausted during final verification')
        except BaseException as exc:
            completed = False
            seal.update(status='ineligible', all_completed_prefinal_parity_passed=False, error=type(exc).__name__+': '+str(exc))
        seal['seal_verification_seconds'] = time.perf_counter()-verification_started
    seal['wall_seconds'] = time.perf_counter()-started
    c.write_new(out/'seal.json', seal)
    if not completed:
        print(seal['error'], file=sys.stderr)
    return 0 if completed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'manifest', 'baseline-seal', 'output-dir', 'runtime', 'dependency-runtime'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    return run(args.data_dir, args.manifest, args.baseline_seal, args.output_dir, args.runtime, args.dependency_runtime)


if __name__ == '__main__':
    sys.exit(main())
