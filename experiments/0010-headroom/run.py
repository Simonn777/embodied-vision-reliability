#!/usr/bin/env python3
"""Prospective source-bound launcher; no filled/admitted plan is distributed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

SUPERVISOR_SHA = '2d306a8813d36c207eb276a187a298864965bffbe79c1dd463d9482f3bf7d77f'


def checked(path, expected):
    path = Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)): raise ValueError('symlink input')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('source/input SHA mismatch')
    return raw


def load(name, path, sha):
    raw = checked(path, sha); m = ModuleType(name); m.__file__ = str(Path(path).resolve())
    sys.modules[name] = m
    exec(compile(raw, m.__file__, 'exec'), m.__dict__)
    return m


def reference(plan, name, base):
    ref = plan['inputs'][name]; path = Path(ref['path'])
    path = path if path.is_absolute() else base/path
    return path, ref['sha256']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('launch','_child'))
    for name in ('plan','plan-sha256','assetdir','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--supervision'); parser.add_argument('--python')
    args = parser.parse_args(argv)
    base = Path(args.plan).resolve().parent; here = Path(__file__).resolve().parent
    plan = json.loads(checked(args.plan,args.plan_sha256))
    if plan['schema'] != '0010-headroom-prospective-v1' or plan['admitted'] is not True: raise ValueError('separate headroom admission required')
    if set(plan['driver_files']) != {'run.py','native.py','driver.py'}: raise ValueError('driver source set')
    for name,sha in plan['driver_files'].items(): checked(here/name,sha)
    paths = [Path(args.assetdir).resolve(),Path(args.output).resolve()]
    if args.supervision: paths.append(Path(args.supervision).resolve())
    if any(a == b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[:i]): raise ValueError('overlapping cache/output/supervision')
    cfg_path,sha = reference(plan,'capability_config',base); cfg = json.loads(checked(cfg_path,sha))
    ids = cfg['cpu_ids']
    if type(ids) is not list or len(ids) != 4 or any(type(i) is not int or i < 0 for i in ids) or ids != sorted(set(ids)): raise ValueError('four distinct CPU IDs')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key] = '4'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'; os.environ['PYTHONNOUSERSITE'] = '1'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.stage == 'launch':
        if not args.python or not args.supervision: parser.error('launch requires --python and --supervision')
        if Path(args.output).exists() or Path(args.supervision).exists(): raise ValueError('fresh outputs required')
        spath,ssha = reference(plan,'supervisor',base)
        if ssha != SUPERVISOR_SHA: raise ValueError('supervisor source changed')
        supervisor = load('_headroom_external_supervisor',spath,ssha)
        with (Path(args.assetdir)/'0010-headroom-launch.started.json').open('x') as f:
            json.dump({'plan_sha256':args.plan_sha256,'output':str(paths[1]),'supervision':str(paths[2])},f)
        command = [str(Path(args.python).resolve()),str(Path(__file__).resolve()),'_child','--plan',str(Path(args.plan).resolve()),
                   '--plan-sha256',args.plan_sha256,'--assetdir',str(paths[0]),'--output',str(paths[1])]
        receipt = supervisor.supervise(command,args.supervision,seconds=900,cpu_ids=ids)
        return receipt['supervisor_exit_code'] if type(receipt['supervisor_exit_code']) is int else 125
    if not sys.platform.startswith('linux') or sorted(os.sched_getaffinity(0)) != ids: raise ValueError('supervised Linux required')
    import resource
    if resource.getrlimit(resource.RLIMIT_AS) != (8*1024**3,8*1024**3): raise ValueError('external memory bound required')
    if Path('/sys/fs/cgroup/memory.max').read_text().strip() != str(8*1024**3): raise ValueError('finite 8 GiB cgroup required')
    driver = load('driver',here/'driver.py',plan['driver_files']['driver.py'])
    native = load('native',here/'native.py',plan['driver_files']['native.py'])
    hpath,hsha = reference(plan,'headroom',base)
    if hsha != native.HEADROOM_SHA: raise ValueError('headroom implementation changed')
    headroom = load('_reviewed_headroom',hpath,hsha)
    ops = native.Native(plan,base,args.assetdir,headroom)
    receipt = driver.run(args.output,Path(args.assetdir)/'0010-headroom-started.json',ops,headroom)
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
