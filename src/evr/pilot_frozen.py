"""Frozen-input gates and exact prediction-grid loading; no network/torch import."""
import csv
import hashlib
import re
import json
import importlib.metadata
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from evr.bootstapir_dev import sha256, download_object, NumpyOnlyUnpickler, UPSTREAM, UPSTREAM_HASHES
from evr.pilot_identity import digest_array
from evr.pilot_metrics import rows_from_arrays


CANONICAL_RECEIPT = 'protocols/0002-freeze-v2.json'
CANONICAL_MANIFEST = 'results/0002-preflight/identity/manifest.json'
CANONICAL_CONFIG = 'configs/0002-exploratory-v2.json'
CANONICAL_PROTOCOL = 'protocols/0002-exploratory-frozen-v2.md'
MANDATORY_DEPENDENCIES = {'torch': '2.10.0+cpu',
 'numpy': '2.0.2',
 'dm-tree': '0.1.9',
 'einshape': '1.0',
 'absl-py': '2.3.1',
 'attrs': '25.3.0',
 'wrapt': '1.17.3',
 'sympy': '1.14.0',
 'filelock': '3.32.3',
 'networkx': '3.6.1',
 'jinja2': '3.1.6',
 'fsspec': '2026.7.0',
 'mpmath': '1.3.0',
 'MarkupSafe': '3.0.3',
 'typing-extensions': '4.16.0',
 'setuptools': '78.1.0'}
PREREQUISITE_FILES = ('results/0002-preflight/freeze-v2-validation.json',
 'results/0002-preflight-recording/dev64/receipt.json',
 'results/0002-preflight-recording/repeat64/receipt.json',
 'results/0002-preflight-recording/independent-verification.json',
 'results/0002-preflight/fixture-frozen-v1.json',
 'results/0002-preflight/local-validation.json')
MANDATORY_INPUTS = frozenset(('results/0002-preflight/freeze-v2-validation.json',
 'protocols/0002-exploratory-frozen-v2.md',
 'configs/0002-exploratory-v2.json',
 'results/0002-preflight/identity/manifest.json',
 'results/0002-preflight/identity/frame-hashes.csv',
 'results/0002-preflight/dependencies.json',
 'results/0002-development/transfer-ledger.json',
 'requirements/0002-torch-cpu.txt',
 'requirements/0002-small-runtime.txt',
 'pyproject.toml',
 'docs/0002-exposure-and-scope.md',
 'src/evr/__init__.py',
 'src/evr/bootstapir_dev.py',
 'src/evr/pilot_identity.py',
 'src/evr/pilot_metrics.py',
 'src/evr/pilot_frozen.py',
 'scripts/run_frozen_pilot.py',
 'scripts/run_pilot_preflight.py',
 'scripts/verify_pilot_preflight.py',
 'scripts/pilot_fixture_check.py',
 'tests/test_pilot_identity.py',
 'tests/test_pilot_metrics.py',
 'tests/test_pilot_frozen.py',
 'tests/test_bootstapir_dev.py',
 'results/0002-preflight-recording/dev64/receipt.json',
 'results/0002-preflight-recording/repeat64/receipt.json',
 'results/0002-preflight-recording/independent-verification.json',
 'results/0002-preflight/fixture-frozen-v1.json',
 'results/0002-preflight/local-validation.json'))

def _git(root, *args):
    try:
        return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ValueError('required Git publication/object gate failed') from exc


def _blob(root, revision, path):
    # Reject symlinks/submodules, ambiguous paths and absent objects, not just
    # whatever bytes `git show` happens to resolve.
    tree = _git(root, 'ls-tree', revision, '--', path).decode().strip()
    fields = tree.split()
    if len(fields) != 4 or fields[0:2] != ['100644', 'blob'] or fields[3] != path:
        raise ValueError('required regular committed blob missing: ' + path)
    return _git(root, 'show', revision + ':' + path), fields[2]


def _current_file(root, path, head_oid):
    p = root / path
    if p.resolve() != p.absolute() or not p.is_file():
        raise ValueError('noncanonical/symlink frozen file: ' + path)
    index = _git(root, 'ls-files', '--stage', '--', path).decode().strip().split()
    if index != ['100644', head_oid, '0', path]:
        raise ValueError('untracked/staged frozen file: ' + path)
    return p.read_bytes()


def _json(data):
    def unique(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=unique)


def verify_freeze(root, receipt_path):
    """Require a canonical, committed, main-published v2 receipt and full inputs.

    Local refs are the publication evidence; fetch origin/main before execution.
    There is deliberately no flag bypassing publication or dependency gates.
    """
    root = Path(root).resolve()
    supplied = Path(receipt_path)
    if not supplied.is_absolute():
        supplied = root / supplied
    if supplied.absolute() != root / CANONICAL_RECEIPT or supplied.resolve() != supplied.absolute():
        raise ValueError('canonical v2 receipt required; v1 is superseded')
    head_bytes, oid = _blob(root, 'HEAD', CANONICAL_RECEIPT)
    published, _ = _blob(root, 'origin/main', CANONICAL_RECEIPT)
    current = _current_file(root, CANONICAL_RECEIPT, oid)
    if current != head_bytes or current != published:
        raise ValueError('receipt differs from committed/published version')
    receipt = _json(current)
    if type(receipt.get('version')) is not int or receipt.get('version') != 2 or receipt.get('status') != 'frozen_exploratory_v2':
        raise ValueError('canonical v2 receipt status required')
    bound_paths = {'manifest_path': CANONICAL_MANIFEST, 'config_path': CANONICAL_CONFIG,
                   'protocol_path': CANONICAL_PROTOCOL, 'receipt_path': CANONICAL_RECEIPT,
                   'dependency_identity_receipt': 'results/0002-preflight/dependencies.json'}
    if any(receipt.get(k) != v for k, v in bound_paths.items()):
        raise ValueError('canonical manifest/config/protocol/receipt path binding failed')
    hashes = receipt.get('input_sha256')
    deps = receipt.get('dependency_versions')
    if not isinstance(hashes, dict) or set(hashes) != MANDATORY_INPUTS:
        raise ValueError('complete mandatory input map required')
    if deps != MANDATORY_DEPENDENCIES:
        raise ValueError('complete fixed dependency map required')
    prerequisites = receipt.get('prerequisite_receipt_sha256')
    if not isinstance(prerequisites, dict) or set(prerequisites) != set(PREREQUISITE_FILES):
        raise ValueError('complete prerequisite map required')
    if any(prerequisites[p] != hashes[p] for p in PREREQUISITE_FILES):
        raise ValueError('prerequisite hash binding failed')
    commit = receipt.get('input_commit')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('full input commit SHA required')
    _git(root, 'merge-base', '--is-ancestor', commit, 'HEAD')
    _git(root, 'merge-base', '--is-ancestor', commit, 'origin/main')
    # Inputs must predate the first v2 receipt; never put the receipt into its
    # own frozen input tree or map, including by reusing a later commit.
    if _git(root, 'ls-tree', commit, '--', CANONICAL_RECEIPT).strip():
        raise ValueError('input tree already contains v2 receipt; circular freeze')
    for path, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid frozen SHA256: ' + path)
        blob, _ = _blob(root, commit, path)
        head, head_oid = _blob(root, 'HEAD', path)
        working = _current_file(root, path, head_oid)
        if any(hashlib.sha256(data).hexdigest() != digest for data in (blob, head, working)):
            raise ValueError('input commit/current frozen hash mismatch: ' + path)
    config = _json((root / CANONICAL_CONFIG).read_bytes())
    if (config.get('version') != 2 or config.get('manifest_path') != CANONICAL_MANIFEST
            or config.get('protocol_path') != CANONICAL_PROTOCOL
            or config.get('freeze_receipt_path') != CANONICAL_RECEIPT):
        raise ValueError('frozen config path/version binding failed')
    if (receipt.get('upstream_commit') != UPSTREAM
            or receipt.get('upstream_module_sha256') != UPSTREAM_HASHES
            or config.get('upstream_commit') != UPSTREAM
            or receipt.get('checkpoint_sha256') != '87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f'
            or receipt.get('archive_sha256') != 'ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc'
            or any(receipt.get(k) != config.get(k) for k in ('checkpoint_sha256', 'archive_sha256'))):
        raise ValueError('upstream/model/archive identity binding failed')
    manifest = _json((root / CANONICAL_MANIFEST).read_bytes())
    if manifest.get('archive', {}).get('sha256') != receipt.get('archive_sha256'):
        raise ValueError('manifest archive binding failed')
    metadata = _json((root / bound_paths['dependency_identity_receipt']).read_bytes())
    if {k: v.get('version') for k, v in metadata.get('packages', {}).items()} != MANDATORY_DEPENDENCIES:
        raise ValueError('dependency provenance map mismatch')
    for package, version in MANDATORY_DEPENDENCIES.items():
        if importlib.metadata.version(package) != version:
            raise ValueError('frozen dependency version changed: ' + package)
    return receipt


def load_verified_dataset(cache,manifest):
    identity=download_object(cache,'rgb.zip')
    if identity['sha256']!=manifest['archive']['sha256']:raise ValueError('archive differs from allocation')
    with zipfile.ZipFile(Path(cache)/'rgb.zip') as z:
        infos=z.infolist()
        if sum(x.file_size for x in infos)>4*1024**3 or any(Path(x.filename).is_absolute() or '..' in Path(x.filename).parts for x in infos):raise ValueError('ZIP gate')
        files=[x for x in infos if x.filename.endswith('.pkl')]
        if len(files)!=1:raise ValueError('pickle count')
        with z.open(files[0]) as f:data=NumpyOnlyUnpickler(f).load()
    if not isinstance(data,list) or len(data)!=len(manifest['identities']):raise ValueError('dataset length')
    return data


def verify_video(example,identity):
    if set(example)!=set(identity['arrays']):raise ValueError('video schema mismatch')
    for key,info in identity['arrays'].items():
        a=example[key]
        if list(a.shape)!=info['shape'] or str(a.dtype)!=info['dtype'] or digest_array(a)!=info['sha256']:raise ValueError('video identity mismatch')
    if digest_array(example['video'][:64])!=identity['prefix64_sha256']:raise ValueError('prefix identity mismatch')


def query_ids(example):
    return np.flatnonzero(~example['occluded'][:,0])[:2].astype(int).tolist()


def prediction_rows(directory,manifest,partition,dataset,freeze_commit):
    root=Path(directory);rows=[];groups=[g for g in manifest['groups'] if g['partition']==partition]
    for group in groups:
        folder=root/group['group_id'];receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['status']!='completed' or receipt['freeze_input_commit']!=freeze_commit or receipt['group_id']!=group['group_id']:raise ValueError('incomplete/unbound prediction group')
        for i in group['members']:
            identity=manifest['identities'][i];example=dataset[i];verify_video(example,identity)
            ids=query_ids(example);info=receipt['videos'][str(i)];path=folder/f'video-{i:03}.csv'
            if info['point_indices']!=ids or info['frames']!=64 or sha256(path)!=info['raw_sha256']:raise ValueError('prediction identity/grid gate')
            raw=np.full((64,len(ids),4),np.nan,np.float64);seen=set()
            for r in csv.DictReader(open(path)):
                t,q=int(r['frame']),int(r['query'])
                if not (0<=t<64 and 0<=q<len(ids)) or (t,q) in seen or int(r['point'])!=ids[q]:raise ValueError('duplicate/outside prediction row')
                seen.add((t,q));raw[t,q]=[float(r[k]) for k in ['x','y','occlusion_logit','expected_dist_logit']]
            if len(seen)!=64*len(ids):raise ValueError('incomplete prediction grid')
            rows.extend(rows_from_arrays(i,identity['video_sha256'],group['group_id'],raw,ids,example['points'],example['occluded']))
    return rows,[g['group_id'] for g in groups]
