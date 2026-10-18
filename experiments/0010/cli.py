#!/usr/bin/env python3
"""Prospective fixed 0010 stages; requires separate published admission, no installer."""
import argparse, hashlib, json, os, resource, sys
from pathlib import Path
from types import ModuleType

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PINS={
    'acquire.py':'412c6cddba8fcd32241026611edb627c76690bf8dcc93195d6360a4ef24b2d75',
    'run_capability.py':'532ef917a87e976dfd1bc0fd1858c16ea8a4b5bebaa9569add017e394f6a7ba6',
    'supervise_0009.py':'2d306a8813d36c207eb276a187a298864965bffbe79c1dd463d9482f3bf7d77f',
}
JSON_PINS={'dependencies.json': '5557cef2a9b663c7249589031eff410750329270de75b9403cfa3a7a0b1ad578', 'downloads.json': '8977bcb4af7ae30fd4ef61585395d067ad1e0dc42ac44710ff10f0096b1f3bc6', 'layout.json': 'c0510b9634817858fed2ec3a4e7d83c1ce690b58e6d41aaa39c57a2ca9d9b79e'}  # Fixed local JSON; no runtime URL discovery.


def checked_bytes(path, expected):
    path=Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('unsafe input file')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('input hash mismatch')
    return raw


def load(name):
    path=(ROOT/'scripts' if name=='supervise_0009.py' else HERE)/name
    raw=checked_bytes(path,PINS[name]); module=ModuleType('_0010_'+path.stem)
    module.__file__=str(path); exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def fixed_json(name): return json.loads(checked_bytes(HERE/name,JSON_PINS[name]))


def checked_config(path, sha):
    cfg=json.loads(checked_bytes(path,sha)); load('run_capability.py').validate_config(cfg)
    return cfg


def disjoint(paths):
    paths=[Path(p).resolve() for p in paths]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[:i]):
        raise ValueError('overlapping writable/source paths')


def bundle_inputs(cfg, layout, cap):
    for relative,source in layout.items():
        if source['kind']=='bundle':
            path=cap.safe_file(HERE/'bundle',source['path']); got=cap.identity(path)
            if any(v is not None and got[k]!=v for k,v in cfg['assets'][relative].items()):
                raise ValueError('bundled input identity mismatch')


def materialize(cache, assetdir, cfg, layout, cap, *, reserved=False):
    """Fresh fixed layout by no-overwrite hard links, only after all identities pass."""
    cache,assetdir=Path(cache).resolve(),Path(assetdir).resolve()
    disjoint([cache,assetdir,HERE/'bundle'])
    if set(layout)!=set(cfg['assets']): raise ValueError('layout asset set')
    sources={}; identities={}
    for relative,source in layout.items():
        if source['kind'] not in ('bundle','cache'): raise ValueError('layout source kind')
        root=HERE/'bundle' if source['kind']=='bundle' else cache
        path=cap.safe_file(root,source['path']); got=cap.identity(path)
        if any(v is not None and got[k]!=v for k,v in cfg['assets'][relative].items()):
            raise ValueError('layout asset identity mismatch')
        # Capability relative paths are fixed by validate_config before this call.
        sources[relative]=path; identities[relative]=got
    if reserved:
        if {p.name for p in assetdir.iterdir()}!={'0010-layout-started.json'}: raise ValueError('reserved layout changed')
    else:
        assetdir.mkdir(parents=True,exist_ok=False)
    for relative,source in sources.items():
        destination=assetdir/relative; destination.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,destination,follow_symlinks=False)
    cap.verify_assets(assetdir,cfg)
    with (assetdir/'layout-receipt.json').open('x') as f:
        json.dump({'status':'completed','assets':identities,'layout':layout},f,indent=2)
    return identities


def child_acquire(mode, cache, assetdir, cfg):
    cap=load('run_capability.py')
    if not sys.platform.startswith('linux') or sorted(os.sched_getaffinity(0))!=cfg['cpu_ids']:
        raise ValueError('supervised Linux affinity required')
    if resource.getrlimit(resource.RLIMIT_AS)!=(8*1024**3,8*1024**3): raise ValueError('supervised memory bound required')
    cap.cgroup_memory_limit()
    name='dependencies.json' if mode=='wheels' else 'downloads.json'
    layout=fixed_json('layout.json')
    if mode=='assets':
        bundle_inputs(cfg,layout,cap); disjoint([cache,assetdir,HERE/'bundle'])
        Path(assetdir).mkdir(parents=True,exist_ok=False)
        with (Path(assetdir)/'0010-layout-started.json').open('x') as f: json.dump({'status':'reserved_before_acquisition'},f)
    load('acquire.py').acquire_all(fixed_json(name),cache)
    if mode=='assets': materialize(cache,assetdir,cfg,layout,cap,reserved=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('wheels','assets','probe','_wheels','_assets'))
    for name in ('config','config-sha256'): parser.add_argument('--'+name,required=True)
    for name in ('cache','assetdir','output','supervision','python'): parser.add_argument('--'+name)
    args=parser.parse_args(argv); cfg=checked_config(args.config,args.config_sha256)
    mode=args.stage.lstrip('_')
    if mode in ('wheels','assets') and not args.cache: parser.error('--cache required')
    if mode in ('assets','probe') and not args.assetdir: parser.error('--assetdir required')
    if mode=='probe' and not args.output: parser.error('--output required')
    if args.stage.startswith('_'):
        child_acquire(mode,args.cache,args.assetdir,cfg); return 0
    if not args.supervision or not args.python: parser.error('--supervision and --python required')
    writable=[args.supervision]+([args.cache] if args.cache else [])+([args.assetdir] if args.assetdir else [])+([args.output] if args.output else [])
    disjoint(writable+[HERE/'bundle'])
    if mode=='probe':
        checked_bytes(HERE/'run_capability.py',PINS['run_capability.py'])
        command=[args.python,str(HERE/'run_capability.py'),'--assetdir',args.assetdir,'--config',args.config,
                 '--config-sha256',args.config_sha256,'--selection',str(HERE/'selection.json'),'--output',args.output]
    else:
        command=[args.python,str(Path(__file__).resolve()),'_'+mode,'--config',args.config,
                 '--config-sha256',args.config_sha256,'--cache',args.cache]
        if args.assetdir: command+=['--assetdir',args.assetdir]
    receipt=load('supervise_0009.py').supervise(command,args.supervision,
                        seconds=300 if mode=='probe' else 900,cpu_ids=cfg['cpu_ids'])
    code=receipt['supervisor_exit_code']
    return code if type(code) is int else 125


if __name__=='__main__': raise SystemExit(main())
