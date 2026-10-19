"""Fresh observer build from the previously pinned archive; no estimator calls.

All paths are explicit, all stage outputs exclusive, and dependencies are reused
without network or installation. Logs/receipts retain failed attempts. The caller
archives portable receipts after removing local path prefixes from commands/logs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

SOURCE = 'PoseLib-6a636b1d84273d5e3ba97676463476569611572a'
ARCHIVE_SHA = '3198f87d1e012d493a5cd0fecd1e8b33b20e6dccc14841a581d1f71c8c36237d'
SDK = '/Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk'
COMPILER = '/Library/Developer/CommandLineTools/usr/bin/c++'
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def dependencies(old):
    result = {}
    for name in ('inputs/cmake/data', 'inputs/pybind11/include', 'inputs/pybind11/share', 'eigen-install'):
        if not (old/name).is_dir():
            raise ValueError('Missing dependency directory: '+name)
        result[name] = inventory(old/name)
        if not result[name]:
            raise ValueError('Empty dependency directory: '+name)
    return result


def compare_flags(old, output, source):
    for relative in ('PoseLib/CMakeFiles/PoseLib.dir/flags.make',
                     'pybind/CMakeFiles/pyposelib.dir/flags.make'):
        original = (old/'pose-build'/relative).read_text()
        current = (output/'pose-build'/relative).read_text()
        original = original.replace(str(old/'inputs'/SOURCE), '<SOURCE>').replace(str(old/'pose-build'), '<BUILD>')
        current = current.replace(str(source), '<SOURCE>').replace(str(output/'pose-build'), '<BUILD>')
        if original != current:
            raise ValueError('Effective compiler flags changed: '+relative)


def stage(name, command, output, env, deadline):
    start = time.monotonic()
    peak, termination = 0, None
    save(output/(name+'.command.json'), command)
    child, code = None, None
    with (output/(name+'.log')).open('xb') as log:
        try:
            if time.monotonic() > deadline:
                raise TimeoutError('Build operation deadline reached before launch')
            child = subprocess.Popen(command, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                raw = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,rss='],
                                     capture_output=True, text=True, timeout=3, check=True)
                rows = [list(map(int, r.split())) for r in raw.stdout.splitlines() if len(r.split()) == 3]
                descendants = {child.pid}
                while True:
                    new = descendants | {pid for pid, parent, _ in rows if parent in descendants}
                    if new == descendants:
                        break
                    descendants = new
                peak = max(peak, sum(rss for pid, _, rss in rows if pid in descendants))
                if peak > 8*1024*1024:
                    termination = 'rss_limit'
                    break
                if time.monotonic() > deadline:
                    termination = 'build_total_1200s_timeout'
                    break
                time.sleep(.25)
        except BaseException as exc:
            termination = type(exc).__name__+': '+str(exc)
        finally:
            if child is not None:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                code = child.wait(timeout=5)
            if time.monotonic() > deadline and termination is None:
                termination = 'build_total_1200s_timeout'
            log.flush()
            save(output/(name+'.receipt.json'), {
                'stage': name, 'exit_code': code, 'termination': termination,
                'seconds': time.monotonic()-start, 'peak_group_rss_kib': peak,
                'log_sha256': sha(output/(name+'.log')),
                'command_sha256': sha(output/(name+'.command.json'))})
    print(json.dumps({'stage': name, 'exit_code': code, 'termination': termination}), flush=True)
    if code != 0 or termination:
        raise RuntimeError('Build stage failed: '+name)


def run(old, output):
    began = time.monotonic()
    deadline = began+1200
    old, output = old.resolve(), output.resolve()
    archive = old/'poselib.tar.gz'
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError('Official source archive hash mismatch')
    if output.exists():
        raise ValueError('Build destination must not exist; preserve any previous attempt')
    compiler = subprocess.check_output([COMPILER, '--version'], text=True).splitlines()[:3]
    if compiler[0] != 'Apple clang version 21.0.0 (clang-2100.3.34.2)' or not Path(SDK).is_dir():
        raise ValueError('Original compiler/SDK unavailable')
    output.mkdir(parents=True)
    save(output/'started.json', {'schema': 'evr0018.build.v1', 'archive_sha256': sha(archive),
         'builder_sha256': sha(__file__), 'python': sys.version, 'compiler': compiler,
         'python_executable_sha256': sha(sys.executable),
         'max_jobs': 2, 'rss_limit_kib': 8*1024*1024, 'timeout_seconds': 1200})
    dependency_inventory = dependencies(old)
    save(output/'dependency-inventory.json', dependency_inventory)
    original_inventory = inventory(old/'inputs'/SOURCE)
    save(output/'original-source-inventory.json', original_inventory)
    inputs = output/'inputs'
    inputs.mkdir()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            p = Path(member.name)
            if p.is_absolute() or '..' in p.parts or p.parts[0] != SOURCE or not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected archive member: '+member.name)
        bundle.extractall(inputs)
    source = inputs/SOURCE
    if inventory(source) != original_inventory:
        raise ValueError('Archive extraction differs from immutable original source')
    env = {k: v for k, v in os.environ.items() if
           not k.startswith('CMAKE_') and k not in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'SDKROOT', 'MACOSX_DEPLOYMENT_TARGET')}
    env.update(PYTHONPATH=str(old/'inputs'), CMAKE_BUILD_PARALLEL_LEVEL='2',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    cmake = str(old/'inputs/cmake/data/bin/cmake')
    stage('instrument', [sys.executable, str(Path(__file__).with_name('instrument_source.py')),
          '--source', str(source), '--receipt', str(output/'instrumentation.json')], output, env, deadline)
    stage('pose-config', [cmake, '-S', str(source), '-B', str(output/'pose-build'),
          '-DCMAKE_BUILD_TYPE=Release', '-DPYTHON_PACKAGE=ON', '-DGENERATE_STUBS=OFF',
          '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTS=OFF', '-DMARCH_NATIVE=OFF',
          '-DCMAKE_CXX_COMPILER='+COMPILER, '-DCMAKE_OSX_SYSROOT='+SDK,
          '-DEigen3_DIR='+str(old/'eigen-install/share/eigen3/cmake'),
          '-DPython_EXECUTABLE='+sys.executable], output, env, deadline)
    compare_flags(old, output, source)
    stage('pose-build', [cmake, '--build', str(output/'pose-build'), '-j', '2'], output, env, deadline)
    if inventory(old/'inputs'/SOURCE) != original_inventory:
        raise ValueError('Original source changed during observer build')
    if dependencies(old) != dependency_inventory:
        raise ValueError('Shared build dependency files changed')
    destination = output/'runtime/poselib'
    destination.mkdir(parents=True)
    for item in (source/'pyposelib').iterdir():
        if item.is_file():
            shutil.copy2(item, destination/item.name)
    binaries = list((output/'pose-build/pybind').glob('_core*.so'))
    if len(binaries) != 1:
        raise ValueError('Expected exactly one native binary')
    shutil.copy2(binaries[0], destination/binaries[0].name)
    if time.monotonic() > deadline:
        raise TimeoutError('Build operation exceeded1200s')
    save(output/'completed.json', {'schema': 'evr0018.build.v1', 'status': 'completed',
         'source_inventory': inventory(source), 'runtime_inventory': inventory(destination),
         'seconds': time.monotonic()-began, 'dependency_inventory_unchanged': True,
         'scientific_native_calls': 0, 'original_source_unchanged': True})
    print('Isolated observer runtime ready; no estimator invoked', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.prior_preparation, args.output)
