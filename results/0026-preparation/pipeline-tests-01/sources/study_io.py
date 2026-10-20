"""0026 source bindings, legacy input verification and JSON; no scientific reads."""
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MANIFEST_SHA = 'a034601021b4baa08a6cc04e91e951ce27dc061d0c0556cf7f1ac141d6a66e34'
TC_SEAL_SHA = 'fb2537ea13b52ac1cc3d2f26fb84053e7fe2da62dedef3b857d3dd8718c9b38e'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def jsonable(value):
    """Explicit nonfinite diagnostics, never pretend they are finite numbers."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: jsonable(v) for k, v in value.items()}
    raise TypeError('unsupported result type: '+type(value).__name__)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(jsonable(value), stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def aliases(mapping):
    missing = object()
    old = {k: sys.modules.get(k, missing) for k in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def module(name, path, dependencies=None):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    with aliases(dependencies or {}):
        spec.loader.exec_module(result)
    return result


def legacy(repo):
    folder = Path(repo)/'experiments/0019-author-loss'
    c = module('evr0026_legacy_common', folder/'common.py')
    worker = module('evr0026_legacy_worker', folder/'predict_pair.py', {'common': c})
    runner = module('evr0026_legacy_runner', folder/'run_diagnostic.py',
                    {'common': c, 'predict_pair': worker})
    return c, runner


def verify_prior(repo):
    repo = Path(repo)
    manifest = repo/'configs/0017-development-manifest.json'
    seal = repo/'results/0019-author-loss/predictions/seal.json'
    original = repo/'results/0017-depth/predictions/seal.json'
    if digest(manifest) != MANIFEST_SHA or digest(seal) != TC_SEAL_SHA:
        raise ValueError('historical manifest/TC seal binding differs')
    c, runner = legacy(repo)
    m, s, records = runner.verify_seal(manifest, original, seal)
    bank = {r['pair_index']: r for r in records if r['arm'] == 'depth_affine'}
    if set(bank) != set(range(80)) or any(r['status'] != 'completed' for r in bank.values()):
        raise ValueError('incomplete historical TC-affine bank')
    bindings = {'manifest': digest(manifest), 'TC_seal': digest(seal),
                'original_seal': digest(original), 'legacy_sources': c.source_hashes(),
                'baselines': {str(i): digest(seal.parent/f'pair-{i:02d}-depth_affine.json')
                              for i in range(80)}}
    return c, m, bank, bindings


def source_hashes():
    paths = sorted(HERE.glob('*.py')) + [HERE/'protocol.md']
    paths += sorted((HERE/'native_bridge').glob('*.py'))
    paths += sorted((HERE/'native_bridge').glob('*.cpp'))
    paths += sorted((HERE/'native_bridge').glob('*.cc'))
    paths += [HERE/'native_bridge/input-lock.json']
    return {str(path.relative_to(HERE)): digest(path) for path in paths}


def require_committed(repo):
    repo = Path(repo).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    for name, expected in source_hashes().items():
        path = (HERE/name).resolve().relative_to(repo)
        data = subprocess.check_output(['git', 'show', commit+':'+str(path)], cwd=repo)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('current experiment source differs from committed source: '+name)
    return commit
