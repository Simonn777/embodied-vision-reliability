#!/usr/bin/env python3
"""Frozen collect/evaluate entry points. This file itself performs no outer resource supervision."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = 'protocols/0009-geometric-stage0-freeze.json'
CONFIG_PATH = 'configs/0009-geometric-stage0.json'
FROZEN_PATHS = (
    'src/evr/__init__.py', 'src/evr/geometric_verification.py',
    'src/evr/geometry_evaluation.py', 'src/evr/exact_assignment_tail.py',
    'src/evr/geometric_stage0.py', 'src/evr/geometric_stage0_evaluation.py',
    'scripts/run_geometric_stage0.py', 'scripts/supervise_0009.py',
    'scripts/check_geometric_stage0.py', CONFIG_PATH,
    'protocols/0009-geometric-stage0.md', 'tests/test_geometric_stage0.py',
    'tests/test_geometric_verification.py', 'tests/test_exact_assignment_tail.py',
    'tests/test_0009_supervisor.py',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git','-C',str(root),*args], stderr=subprocess.PIPE)


def verify_freeze(freeze_commit, root=ROOT):
    root = Path(root).resolve()
    if not re.fullmatch(r'[0-9a-f]{40}',freeze_commit):
        raise ValueError('freeze commit must be an exact lowercase 40-hex SHA')
    canonical = git('show',freeze_commit+':'+RECEIPT,root=root)
    receipt = json.loads(canonical)
    source = receipt.get('source_commit','')
    if receipt.get('schema') != 'evr0009.freeze.v1' or not re.fullmatch(r'[0-9a-f]{40}',source):
        raise ValueError('invalid canonical freeze schema/source')
    files = receipt.get('files',{})
    if set(files) != set(FROZEN_PATHS) or not all(re.fullmatch(r'[0-9a-f]{64}',v) for v in files.values()):
        raise ValueError('freeze must bind the exact mandatory input set')
    parents = git('rev-list','--parents','-n','1',freeze_commit,root=root).decode().split()
    if parents != [freeze_commit,source]:
        raise ValueError('freeze must be a distinct single-parent commit after source')
    delta = git('diff-tree','--no-commit-id','--name-status','-r',freeze_commit,root=root).decode().splitlines()
    if delta != ['A\t'+RECEIPT]:
        raise ValueError('freeze commit must add only the canonical receipt')
    git('merge-base','--is-ancestor',freeze_commit,'HEAD',root=root)
    for relative in (*FROZEN_PATHS, RECEIPT):
        path = root / relative
        if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():
            raise ValueError('frozen path must be a regular file without symlink parents: '+relative)
        current = path.read_bytes()
        index = git('show',':'+relative,root=root)
        head = git('show','HEAD:'+relative,root=root)
        if relative == RECEIPT:
            if not (current == index == head == canonical):
                raise ValueError('canonical receipt differs from committed/index/current bytes')
        elif not (sha(current) == sha(index) == sha(head) ==
                  sha(git('show',source+':'+relative,root=root)) == files[relative]):
            raise ValueError('frozen input differs: '+relative)
    return {'freeze_commit': freeze_commit, 'source_commit': source,
            'receipt_sha256': sha(canonical), 'files': files,
            'execution_head': git('rev-parse','HEAD',root=root).decode().strip()}


def load_verified_modules(binding):
    if any(name == 'evr' or name.startswith('evr.') for name in sys.modules):
        raise RuntimeError('evr modules must not be preloaded')
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0,str(ROOT/'src'))
    modules = {}
    names = ('evr','evr.geometric_verification','evr.geometry_evaluation',
             'evr.exact_assignment_tail','evr.geometric_stage0','evr.geometric_stage0_evaluation')
    for name in names:
        relative = ('src/evr/__init__.py' if name == 'evr' else
                    'src/evr/' + name.rsplit('.',1)[1] + '.py')
        path = ROOT/relative
        source = path.read_bytes()
        if sha(source) != binding['files'][relative]:
            raise RuntimeError('frozen source bytes changed before execution: '+relative)
        spec = importlib.util.spec_from_file_location(name,path,
                    submodule_search_locations=[str(ROOT/'src/evr')] if name=='evr' else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Execute the hashed source itself, never a timestamp-valid compiled cache.
        exec(compile(source,str(path),'exec'),module.__dict__)
        if name in ('evr.geometric_stage0','evr.geometric_stage0_evaluation'):
            modules[name.rsplit('.',1)[1]] = module
    for name, module in list(sys.modules.items()):
        if name == 'evr' or name.startswith('evr.'):
            origin = getattr(module,'__file__',None)
            if not origin:
                raise RuntimeError('missing evr module source origin')
            resolved = Path(origin).resolve()
            try:
                relative = resolved.relative_to(ROOT).as_posix()
            except ValueError as exc:
                raise RuntimeError('evr origin outside frozen repo') from exc
            if relative not in binding['files'] or sha(resolved.read_bytes()) != binding['files'][relative]:
                raise RuntimeError('unbound scientific module origin: '+name)
    return modules


def write_new(path, value):
    data = (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as output:
        output.write(data)
    return sha(data)


def new_output(path):
    path = path.resolve()
    path.mkdir(parents=False,exist_ok=False)
    return path


def require_collect_resources():
    if sys.platform != 'linux' or not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('production collection requires Linux external supervision')
    affinity = sorted(os.sched_getaffinity(0))
    if not 1 <= len(affinity) <= 4:
        raise RuntimeError('collection affinity must contain at most four CPUs')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if not (0 < soft <= hard <= 8589934592):
        raise RuntimeError('collection requires finite soft/hard RLIMIT_AS <= 8 GiB')
    names = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
    threads = {name: os.environ.get(name,'') for name in names}
    if any(not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 4
           for value in threads.values()):
        raise RuntimeError('all numerical thread environment variables must be integers 1..4')
    return {'affinity':affinity,'rlimit_as_soft':soft,'rlimit_as_hard':hard,
            'thread_environment':threads}


def resource_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'user_cpu_seconds': usage.ru_utime, 'system_cpu_seconds': usage.ru_stime,
            'peak_rss': usage.ru_maxrss,
            'peak_rss_unit': 'bytes' if sys.platform == 'darwin' else 'KiB'}


def collect(output, binding, core, resource_preflight):
    import numpy as np
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    before = resource_snapshot()
    (output/'cases').mkdir()
    (output/'labels').mkdir()
    order = core.case_order()
    manifest, inventory, serialization = {}, [], 0.
    manifest['planned-cases.json'] = write_new(output/'planned-cases.json',
        {'case_ids':[f'{f}-{s:02d}' for f,s in order],
         'interruption_rule':'Absent case files are unattempted; no lock means incomplete.'})
    invalid = False
    for family, s in order:
        case_id = f'{family}-{s:02d}'
        if invalid:
            rec = {'case_id':case_id,'family':family,'seed_index':s,'status':'unattempted',
                   'reason':'previous study-validity failure','controls_complete':False,
                   'hypotheses':[],'selections':{},'tail_attempted':0}
            labels = None
        else:
            # Seeds, Generator creation, coordinates and all hashes are inside the case timer.
            seeds = {}
            def build():
                seeds.update(core.stream_seeds(family,s))
                rngs = {name: np.random.Generator(np.random.PCG64(value)) for name,value in seeds.items()}
                return core.construct_case(family,s,rngs)
            rec, labels = core.execute_case(build)
            rec.update({'case_id':case_id,'family':family,'seed_index':s,'rng_seeds':seeds})
            invalid = rec['status'] == 'invalid'
        stamp = time.perf_counter()
        relative = f'cases/{case_id}.json'
        manifest[relative] = write_new(output/relative,rec)
        if labels is not None:
            relative = f'labels/{case_id}.json'
            manifest[relative] = write_new(output/relative,labels)
        serialization += time.perf_counter()-stamp
        inventory.append({'case_id':case_id,'status':rec['status']})
    after = resource_snapshot()
    summary = {'schema':'evr0009.collection.v1','binding':binding,
               'command': ['run_geometric_stage0.py','collect','--freeze-commit',binding['freeze_commit'],
                           '--output','<new-output-directory>'],
               'numpy_version':np.__version__,'python_version':sys.version.split()[0],
               'started_utc':started_utc,'finished_utc':datetime.now(timezone.utc).isoformat(),
               'case_order':[f'{f}-{s:02d}' for f,s in order], 'inventory':inventory,
               'selection_complete':all(r['status']=='completed' for r in inventory),
               'case_serialization_seconds':serialization,
               'collection_wall_seconds_before_final_serialization':time.perf_counter()-started,
               'resource_before':before,'resource_after':after,'resource_preflight':resource_preflight,
               'outer_900s_4cpu_8gib_supervision':'must be independently bound by external supervisor',
               'scope':'Synthetic development selection only; GT labels saved separately and not scored.'}
    manifest['collection.json'] = write_new(output/'collection.json',summary)
    lock = {'schema':'evr0009.selection-lock.v1','freeze_commit':binding['freeze_commit'],
            'files':manifest,'selection_complete':summary['selection_complete'],
            'self_hash_boundary':'This lock excludes itself; parent supervisor must bind its raw bytes.'}
    write_new(output/'selection-lock.json',lock)
    return 1 if invalid or any(r['status'] in ('incomplete','unattempted') for r in inventory) else 0


def evaluate(input_dir, output, binding, evaluator):
    input_dir = input_dir.resolve()
    if input_dir == output or input_dir in output.parents or output in input_dir.parents:
        raise ValueError('evaluation output must not overlap locked collection')
    raw_lock = (input_dir/'selection-lock.json').read_bytes()
    lock = json.loads(raw_lock)
    if lock.get('schema') != 'evr0009.selection-lock.v1' or lock.get('freeze_commit') != binding['freeze_commit']:
        raise ValueError('selection lock schema/freeze mismatch')
    files = lock.get('files',{})
    for relative, digest in files.items():
        path = input_dir/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
            raise ValueError('invalid selection-lock path')
        if sha(path.read_bytes()) != digest:
            raise ValueError('locked artifact changed: '+relative)
    actual = {p.relative_to(input_dir).as_posix() for p in input_dir.rglob('*') if p.is_file()}
    if actual != set(files)|{'selection-lock.json'}:
        raise ValueError('collection inventory differs from lock')
    collection = json.loads((input_dir/'collection.json').read_bytes())
    if collection['binding'] != binding:
        # Later publication commits are permitted, with identical scientific source/freeze.
        left, right = dict(collection['binding']), dict(binding)
        left.pop('execution_head',None); right.pop('execution_head',None)
        if left != right:
            raise ValueError('collection source binding differs')
    expected = collection['case_order']
    canonical_ids = [f'{f}-{s:02d}' for s in range(64)
                     for f in evaluator.FAMILIES[s%4:]+evaluator.FAMILIES[:s%4]]
    if expected != canonical_ids or len(expected) != len(set(expected)):
        raise ValueError('collection case allocation differs')
    records, labels = {}, {}
    for case_id in expected:
        relative = 'cases/'+case_id+'.json'
        if relative not in files:
            raise ValueError('missing locked case record')
        records[case_id] = json.loads((input_dir/relative).read_bytes())
        if records[case_id].get('case_id') != case_id:
            raise ValueError('case ID mismatch')
        relative = 'labels/'+case_id+'.json'
        if relative in files:
            labels[case_id] = json.loads((input_dir/relative).read_bytes())
    start = time.perf_counter()
    outcome = evaluator.evaluate_cases(records,labels,expected_ids=expected)
    outcome['binding'] = binding
    outcome['selection_lock_sha256'] = sha(raw_lock)
    outcome['evaluation_seconds_before_serialization'] = time.perf_counter()-start
    write_new(output/'evaluation.json',outcome)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('verify-only','collect','evaluate'))
    parser.add_argument('--freeze-commit',required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--input',type=Path)
    args = parser.parse_args(argv)
    binding = verify_freeze(args.freeze_commit)
    if args.command == 'verify-only':
        print(json.dumps(binding,sort_keys=True))
        return 0
    if args.output is None or (args.command=='evaluate' and args.input is None):
        parser.error('scientific command needs --output; evaluate also needs --input')
    if args.command == 'evaluate':
        left, right = args.input.resolve(), args.output.resolve()
        if left == right or left in right.parents or right in left.parents:
            raise ValueError('evaluation output must not overlap locked collection')
    preflight = require_collect_resources() if args.command == 'collect' else None
    # Reserve output before any RNG, scientific import, artifact input or computation.
    output = new_output(args.output)
    try:
        modules = load_verified_modules(binding)
        core = modules['geometric_stage0']
        config = json.loads((ROOT/CONFIG_PATH).read_bytes())
        if config != core.CONFIG:
            raise ValueError('config differs from fixed implementation constants')
        if args.command == 'collect':
            return collect(output,binding,core,preflight)
        return evaluate(args.input,output,binding,modules['geometric_stage0_evaluation'])
    except BaseException as exc:
        write_new(output/'failure.json',{'status':'failed','error_type':type(exc).__name__,
                                        'message':str(exc),'binding':binding})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
