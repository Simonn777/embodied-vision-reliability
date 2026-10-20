"""Frozen-source generation and truth-free prediction; no work on import.

Generation is a distinct process. Prediction consumes only observation JSONL
and generation metadata; it never opens the truth file. Evaluation is separate.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
NATIVE_SHA256 = '3228d488ca11f966378b6b7657ee6ebc5716da474c7ae88fccff7ab433a6981d'
ROWS = 138


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode(data):
    def reject(value):
        raise ValueError('nonfinite JSON token: '+value)
    return json.loads(data, object_pairs_hook=_unique, parse_constant=reject)


def save_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)+b'\n')


def save_progress(path, value):
    # Each completed JSONL record is immutable; only the active-row checkpoint
    # is replaced, including before a native attempt that may abort the process.
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.pending')
    with temporary.open('wb') as stream:
        stream.write(canonical(value)+b'\n')
        stream.flush()
    os.replace(temporary, path)


def verify_source(repo, expected_commit, source_root=HERE):
    repo, source_root = Path(repo).resolve(), Path(source_root).resolve()
    if not re.fullmatch('[0-9a-f]{40}', expected_commit):
        raise ValueError('expected full commit SHA')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != expected_commit:
        raise ValueError('HEAD differs from frozen source commit')
    names = sorted(p.name for p in source_root.glob('*.py'))+['protocol.md']
    if not all(name in names for name in ('pipeline.py', 'fixtures.py', 'study_core.py',
                                         'tangent_solver.py', 'controls.py', 'evaluate.py')):
        raise ValueError('incomplete scientific source set')
    hashes = {}
    for name in names:
        path = source_root/name
        relative = path.relative_to(repo).as_posix()
        data = path.read_bytes()
        committed = subprocess.check_output(['git', '-C', str(repo), 'show',
                                             expected_commit+':'+relative])
        if data != committed:
            raise ValueError('source differs from frozen commit: '+name)
        hashes[name] = sha(data)
    return hashes


def verify_current(hashes, source_root=HERE):
    for name, digest in hashes.items():
        if sha((Path(source_root)/name).read_bytes()) != digest:
            raise ValueError('scientific source changed during operation: '+name)


def environment():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                platform=platform.platform(), thread_environment={key: os.environ.get(key)
                for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})


def generate(repo, commit, output):
    sources = verify_source(repo, commit)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    job = dict(schema='evr0027.generation-job.v1', status='started',
               expected_commit=commit, source_sha256=sources, environment=environment())
    save_new(output/'generation-job.json', job)
    # This is the only production import of the physical fixture generator.
    import fixtures
    if any(name in sys.modules for name in ('study_core', 'tangent_solver', 'poselib')):
        raise RuntimeError('generation process imported a pose estimator')
    count = 0
    with (output/'inputs.jsonl').open('xb') as inputs, (output/'truth.jsonl').open('xb') as truth:
        for record in fixtures.iter_cases():
            if record['row_id'] != count or count >= ROWS:
                raise ValueError('generation row order/cardinality mismatch')
            if set(record['observation']) != fixtures.OBSERVATION_KEYS:
                raise ValueError('generation observation allowlist')
            inputs.write(canonical(dict(row_id=count, observation=record['observation']))+b'\n')
            truth.write(canonical(dict(row_id=count, truth=record['truth'],
                                       provenance=record['provenance']))+b'\n')
            count += 1
        inputs.flush(); truth.flush()
    if count != ROWS:
        raise ValueError('incomplete generation')
    verify_current(sources)
    manifest = dict(schema='evr0027.generation.v1', status='completed', row_count=count,
                    expected_commit=commit, source_sha256=sources,
                    generation_job_sha256=sha((output/'generation-job.json').read_bytes()),
                    inputs_sha256=sha((output/'inputs.jsonl').read_bytes()),
                    truth_sha256=sha((output/'truth.jsonl').read_bytes()),
                    elapsed_seconds=time.perf_counter()-start,
                    numpy_version=fixtures.np.__version__)
    save_new(output/'generation.json', manifest)
    return manifest


def load_inputs(path):
    data = Path(path).read_bytes()
    lines = data.splitlines()
    if len(lines) != ROWS or not all(line.strip() for line in lines):
        raise ValueError('complete 138-row input required')
    rows = [decode(line) for line in lines]
    for i, row in enumerate(rows):
        if (set(row) != {'row_id', 'observation'} or type(row['row_id']) is not int
                or row['row_id'] != i):
            raise ValueError('input row keys/order mismatch')
    return data, rows


def load_native(runtime):
    runtime = Path(runtime).resolve()
    binary = runtime/'poselib/_core.cpython-39-darwin.so'
    if sha(binary.read_bytes()) != NATIVE_SHA256:
        raise ValueError('native binary differs before import')
    sys.path.insert(0, str(runtime))
    native = importlib.import_module('poselib')
    core = importlib.import_module('poselib._core')
    if Path(core.__file__).resolve() != binary or not callable(native.monodepth_pose_3pt):
        raise ValueError('unexpected loaded native API')
    return native, binary


def predict(repo, commit, generation, runtime, output):
    sources = verify_source(repo, commit)
    generation, output = Path(generation), Path(output)
    manifest_bytes = (generation/'generation.json').read_bytes()
    manifest = decode(manifest_bytes)
    if (manifest.get('schema') != 'evr0027.generation.v1'
            or manifest.get('status') != 'completed' or manifest.get('row_count') != ROWS
            or manifest.get('expected_commit') != commit or manifest.get('source_sha256') != sources):
        raise ValueError('generation binding mismatch')
    generation_job = (generation/'generation-job.json').read_bytes()
    if sha(generation_job) != manifest['generation_job_sha256']:
        raise ValueError('generation job changed')
    inputs_bytes, rows = load_inputs(generation/'inputs.jsonl')
    if sha(inputs_bytes) != manifest['inputs_sha256']:
        raise ValueError('generation inputs changed')
    # Only a digest is copied from generation metadata. No truth file is opened.
    if not re.fullmatch('[0-9a-f]{64}', manifest.get('truth_sha256', '')):
        raise ValueError('missing truth digest in generation metadata')
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    job = dict(schema='evr0027.prediction-job.v1', status='started',
               expected_commit=commit, source_sha256=sources,
               generation_sha256=sha(manifest_bytes),
               generation_job_sha256=sha(generation_job),
               inputs_sha256=sha(inputs_bytes), truth_sha256=manifest['truth_sha256'],
               native_sha256=NATIVE_SHA256, environment=environment())
    save_new(output/'job.json', job)
    import study_core
    if 'fixtures' in sys.modules:
        raise RuntimeError('prediction process imported physical truth generator')
    # Validate complete input coverage before any numerical/native call.
    for row in rows:
        study_core.validate_observation(row['observation'])
    native, binary = load_native(runtime)
    count = 0
    with (output/'predictions.jsonl').open('xb') as predictions:
        for row in rows:
            observation_hash = sha(canonical(row['observation']))
            def progress(result):
                save_progress(output/'active-row.json', dict(
                    row_id=row['row_id'], input_sha256=observation_hash, result=result))
            result = study_core.process(row['observation'], native, progress=progress)
            if result['status'] != 'completed':
                raise ValueError('incomplete study result')
            predictions.write(canonical(dict(row_id=row['row_id'],
                input_sha256=observation_hash, result=result))+b'\n')
            predictions.flush()
            count += 1
    if count != ROWS:
        raise ValueError('incomplete prediction')
    verify_current(sources)
    if (sha(binary.read_bytes()) != NATIVE_SHA256
            or sha((generation/'inputs.jsonl').read_bytes()) != job['inputs_sha256']
            or sha((generation/'generation.json').read_bytes()) != job['generation_sha256']
            or sha((generation/'generation-job.json').read_bytes()) != job['generation_job_sha256']):
        raise ValueError('bound input/runtime changed during prediction')
    seal = dict(schema='evr0027.seal.v1', status='completed', row_count=count,
                expected_commit=commit, source_sha256=sources,
                inputs_sha256=job['inputs_sha256'], truth_sha256=job['truth_sha256'],
                predictions_sha256=sha((output/'predictions.jsonl').read_bytes()),
                job_sha256=sha((output/'job.json').read_bytes()),
                elapsed_seconds=time.perf_counter()-start)
    save_new(output/'seal.json', seal)
    return seal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('generate', 'predict'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', type=Path)
    parser.add_argument('--native-runtime', type=Path)
    args = parser.parse_args()
    if args.operation == 'generate':
        if args.generation or args.native_runtime:
            parser.error('generation must not receive predictor inputs')
        record = generate(args.repo, args.expected_commit, args.output)
    else:
        if not args.generation or not args.native_runtime:
            parser.error('prediction requires generation and native runtime')
        record = predict(args.repo, args.expected_commit, args.generation,
                         args.native_runtime, args.output)
    print(json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
