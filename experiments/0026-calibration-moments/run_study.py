"""One supervised 80-pair 0026 prediction operation; never reads reference poses."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import study_io as io

THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def run_child(command, log_path, deadline, env):
    start = time.perf_counter()
    limit = min(deadline, start+20.)
    termination, peak = None, 0
    with Path(log_path).open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            while child.poll() is None:
                if time.perf_counter() >= limit:
                    termination = 'pair_timeout' if start+20. <= deadline else 'operation_timeout'
                    break
                sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)],
                                        capture_output=True, text=True, timeout=2)
                if sample.returncode != 0 or not sample.stdout.strip().isdigit():
                    if child.poll() is not None:
                        break
                    raise RuntimeError('RSS supervision sample failed for a running child')
                if sample.stdout.strip().isdigit():
                    peak = max(peak, int(sample.stdout.strip()))
                    if peak > 8*1024*1024:
                        termination = 'rss_limit'
                        break
                time.sleep(.1)
        except BaseException:
            termination = 'supervision_error'
            raise
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = child.wait(timeout=5)
    return {'exit_code': code, 'termination': termination, 'wall_seconds': time.perf_counter()-start,
            'sampled_peak_rss_kib': peak, 'rss_sampling': 'child ps every about0.1s; peak may be missed'}


def run(args):
    start = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    status, error, commit = 'error', None, None
    def deadline_check():
        if time.perf_counter()-start >= 1800.:
            raise TimeoutError('whole-operation deadline exceeded')
    try:
        commit = io.require_committed(args.repo)
        sources = io.source_hashes()
        c, manifest, baseline, prior = io.verify_prior(args.repo)
        for scene in manifest['scenes']:
            deadline_check()
            path = Path(args.data_dir)/scene['filename']
            if path.stat().st_size != scene['bytes'] or io.digest(path) != scene['sha256']:
                raise ValueError('dataset opaque binding mismatch')
        native = {'library_sha256': io.digest(args.native_library),
                  'receipt_sha256': io.digest(args.native_receipt)}
        job = {'schema': 'evr0026.job.v1', 'commit': commit, 'source_sha256': sources,
               'prior': prior, 'native': native, 'selected_pairs': manifest['selected_pairs']}
        io.write_new(out/'job.json', job)
        environment = dict(os.environ)
        environment.update({name: '4' for name in THREAD_VARS})
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        environment['PYTHONPATH'] = os.pathsep.join(str(Path(p).resolve()) for p in args.runtime)
        preparation_seconds = time.perf_counter()-start
        for index in range(80):
            deadline_check()
            name = f'pair-{index:02d}'
            command = [sys.executable, '-B', str(io.HERE/'predict_pair.py'), '--job', str(out/'job.json'),
                       '--repo', str(Path(args.repo).resolve()), '--data-dir', str(Path(args.data_dir).resolve()),
                       '--native-library', str(Path(args.native_library).resolve()),
                       '--native-receipt', str(Path(args.native_receipt).resolve()), '--index', str(index),
                       '--output', str(out/(name+'.json'))]
            process = run_child(command, out/(name+'.log'), start+1800., environment)
            io.write_new(out/(name+'.process.json'), process)
            if process['exit_code'] != 0 or process['termination'] is not None:
                # Preserve timeout/raw logs; no partial cohort evaluation or retry.
                raise RuntimeError('pair did not complete: '+name+': '+str(process))
            path = out/(name+'.json')
            result = json.loads(path.read_text())
            if result['status'] != 'completed' or result['pair_index'] != index or result['source_sha256'] != sources:
                raise ValueError('completed pair identity/source differs')
            records.append({'pair_index': index, 'path': path.name, 'sha256': io.digest(path),
                            'process': name+'.process.json', 'process_sha256': io.digest(out/(name+'.process.json')),
                            'log': name+'.log', 'log_sha256': io.digest(out/(name+'.log'))})
            records[-1].update(partial=name+'.partial.json', partial_sha256=io.digest(out/(name+'.partial.json')))
            print(json.dumps({'completed_pairs': len(records), 'wall_seconds': time.perf_counter()-start}), flush=True)
        if sources != io.source_hashes() or io.require_committed(args.repo) != commit:
            raise ValueError('source changed during operation')
        # Recheck complete historical bank and opaque files before sealing.
        if io.verify_prior(args.repo)[3] != prior:
            raise ValueError('historical bank changed during operation')
        for scene in manifest['scenes']:
            deadline_check()
            if io.digest(Path(args.data_dir)/scene['filename']) != scene['sha256']:
                raise ValueError('opaque input changed during operation')
        if io.digest(args.native_library) != native['library_sha256'] or io.digest(args.native_receipt) != native['receipt_sha256']:
            raise ValueError('native binary/build receipt changed during operation')
        deadline_check()
        io.write_new(out/'seal.json', {'schema': 'evr0026.seal.v1', 'status': 'completed',
                     'source_commit': commit, 'source_sha256': sources, 'job_sha256': io.digest(out/'job.json'),
                     'rows': records, 'preparation_seconds': preparation_seconds})
        deadline_check()
        status = 'completed'
        return 0
    except BaseException as exc:
        status = 'error'
        error = type(exc).__name__+': '+str(exc)
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        io.write_new(out/'outer-receipt.json', {'schema': 'evr0026.outer.v1', 'status': status,
                     'error': error, 'source_commit': commit, 'completed_pairs': len(records),
                     'exit_code': 0 if status == 'completed' else 1,
                     'wall_seconds': time.perf_counter()-start,
                     'max_child_rss_native_bytes_macos': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                     'operation_limit_seconds': 1800, 'pair_limit_seconds': 20,
                     'thread_environment': {name: '4' for name in THREAD_VARS},
                     'seal_sha256': io.digest(out/'seal.json') if (out/'seal.json').exists() else None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'data-dir', 'native-library', 'native-receipt', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--runtime', action='append', type=Path, required=True)
    sys.exit(run(parser.parse_args()))
