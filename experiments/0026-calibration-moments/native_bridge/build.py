"""Build only the separate bridge; never runs a refiner or imports NumPy/PoseLib."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
COMPILER=Path('/Library/Developer/CommandLineTools/usr/bin/c++')
SDK=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk')
ABI_VERSION=260001


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def inventory(folder):
    files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    if not files: raise ValueError('Empty dependency tree: '+str(folder))
    return files


def preflight(project_root):
    project_root=Path(project_root).resolve()
    lock=json.loads((HERE/'input-lock.json').read_text())
    if lock.get('schema')!='evr0026.native-input-lock.v1': raise ValueError('Bad dependency lock')
    for rel, expected in lock['files'].items():
        if sha(project_root/rel)!=expected: raise ValueError('Pinned input changed: '+rel)
    for rel, expected in lock['trees'].items():
        actual=inventory(project_root/rel)
        digest=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=expected['sha256'] or len(actual)!=expected['file_count']:
            raise ValueError('Pinned header tree changed: '+rel)
    if sha(COMPILER)!=lock['compiler_sha256'] or not SDK.is_dir(): raise ValueError('Compiler/SDK mismatch')
    return lock


def build(output_dir,project_root=PROJECT):
    project_root=Path(project_root).resolve()
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=False)
    receipt={'schema':'evr0026.native-build.v1','status':'started','abi_version':ABI_VERSION,
             'exit_code':None,'library_sha256':None,'source_sha256':{},'command':None,
             'project_root':str(project_root),
             'build_amendment':'After preserved build-01 unused-parameter failures in upstream camera_models.h, use -isystem for dependency headers; bridge retains -Wall -Wextra -Werror. No source/archive or algorithm change.'}
    start=time.perf_counter(); child=None
    def save():
        temporary=output/'receipt.tmp'
        temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
        temporary.replace(output/'receipt.json')
    save()
    try:
        lock=preflight(project_root)
        sources={name:sha(HERE/name) for name in ('build.py','refiner.cc','bridge.py','input-lock.json','NOTICE.md')}
        source=project_root/lock['source_directory']; eigen=project_root/lock['eigen_directory']; generated=project_root/lock['generated_directory']
        library=output/'libevr0026_refiner.dylib'
        command=[str(COMPILER),'-std=gnu++17','-O3','-DNDEBUG','-arch','arm64','-isysroot',str(SDK),
            '-fPIC','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror','-dynamiclib',
            '-Wl,-undefined,error','-isystem',str(source),'-isystem',str(generated),'-isystem',str(eigen),
            str(HERE/'refiner.cc'),str(project_root/lock['archive']),'-o',str(library)]
        receipt.update(source_sha256=sources,bridge_python_sha256=sources['bridge.py'],
                       inputs=lock,command=command,python=sys.version,python_executable=sys.executable)
        save()
        env=dict(os.environ)
        for name in ('CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','LIBRARY_PATH','CPATH','CPLUS_INCLUDE_PATH',
                     'SDKROOT','MACOSX_DEPLOYMENT_TARGET','ARCHFLAGS'):
            env.pop(name,None)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[name]='1'
        with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            receipt['exit_code']=child.wait(timeout=180)
        if receipt['exit_code']!=0: raise RuntimeError('Compiler returned nonzero exit')
        if preflight(project_root)!=lock or any(sha(HERE/k)!=v for k,v in sources.items()):
            raise ValueError('Dependency/source changed during build')
        receipt.update(status='completed',library_name=library.name,library_bytes=library.stat().st_size,
                       library_sha256=sha(library),native_calls=0)
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc),
                       exit_code=child.returncode if child else None,native_calls=0)
        return 1
    finally:
        receipt['wall_seconds']=time.perf_counter()-start
        receipt['log_sha256']={name:sha(output/name) for name in ('stdout.log','stderr.log') if (output/name).exists()}
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--project-root',default=str(PROJECT),help='Asset workspace root; pass explicitly after relocating source')
    args=parser.parse_args()
    sys.exit(build(args.output_dir,args.project_root))
