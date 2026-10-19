#!/usr/bin/env python3
"""One-shot frozen depth-cache acquisition and names/shapes-only allocation.

No dataset value, attribute value, image, depth, coordinate or pose is read.
Partial downloads and durable failure receipts are retained, without retries.
"""
import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'protocols/0017-depth-assets.md'
BASE_URL = 'https://vision.maths.lth.se/viktor/posebench/relative/monodepth/'
SALT = 'evr0017:development:2026-10-19\0'
SCENE_SIZES = (
    ('florence_cathedral', 10675000), ('lincoln_memorial', 10678136),
    ('london_bridge', 10486136), ('milan_cathedral', 10636784),
    ('sagrada_familia', 10636488),
)
ASSETS = tuple({'scene': scene, 'filename': scene + '_roma_moge.h5',
                'url': BASE_URL + scene + '_roma_moge.h5', 'bytes': size}
               for scene, size in SCENE_SIZES)
MAX_NETWORK_BYTES = 53112544
REQUEST_SECONDS = 60
METADATA_SECONDS = 30
CHUNK_BYTES = 65536
REQUIRED_FIELDS = {'x1', 'x2', 'K1', 'K2', 'depth1', 'depth2', 'camera1', 'camera2', 'R', 't'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write_new(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect refused: URL is frozen')


@contextmanager
def deadline(seconds):
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'existing alarm conflicts with bounded acquisition')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('bounded acquisition phase exceeded %d seconds' % seconds)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def select(scene, root_keys):
    require(scene in dict(SCENE_SIZES), 'unknown development scene')
    require(len(root_keys) >= 16 and len(root_keys) == len(set(root_keys)), 'scene needs at least16 distinct pairs')
    require(all(isinstance(key, str) and key and '/' not in key and '\0' not in key for key in root_keys), 'invalid pair key')
    return sorted(root_keys, key=lambda key: (
        hashlib.sha256((SALT + scene + '\0' + key).encode('utf-8')).hexdigest(), key))[:16]


def inspect_metadata(path, h5_module=None):
    """Traverse hard links and metadata only; never access Dataset[...]/attrs."""
    if h5_module is None:
        import h5py as h5_module
    nodes = 0
    def children(group, prefix, groups, datasets, depth):
        nonlocal nodes
        require(depth <= 8, 'unexpected metadata nesting/cycle')
        names = sorted(group.keys())
        for name in names:
            nodes += 1
            require(nodes <= 200000, 'metadata node bound exceeded')
            require(isinstance(name, str) and name and '/' not in name and '\0' not in name, 'invalid field name')
            link = group.get(name, getlink=True)
            require(isinstance(link, h5_module.HardLink), 'external/soft HDF5 link refused')
            obj = group[name]  # Resolves an object handle, never a dataset value.
            relative = prefix + name
            if isinstance(obj, h5_module.Dataset):
                datasets[relative] = {'shape': None if obj.shape is None else [int(n) for n in obj.shape],
                                      'dtype': str(obj.dtype)}
            elif isinstance(obj, h5_module.Group):
                groups.append(relative)
                children(obj, relative + '/', groups, datasets, depth + 1)
            else:
                raise ValueError('unexpected HDF5 object kind')
    with h5_module.File(path, 'r') as handle:
        keys = sorted(handle.keys())
        require(16 <= len(keys) <= 10000 and len(set(keys)) == len(keys), 'unexpected pair count')
        metadata = []
        for key in keys:
            require(isinstance(key, str) and key and '/' not in key and '\0' not in key, 'invalid root key')
            require(isinstance(handle.get(key, getlink=True), h5_module.HardLink), 'root external/soft link refused')
            group = handle[key]
            require(isinstance(group, h5_module.Group), 'root pair must be a group')
            require(REQUIRED_FIELDS <= set(group.keys()), 'pair is missing expected field names')
            groups, datasets = [], {}
            children(group, '', groups, datasets, 0)
            metadata.append({'key': key, 'groups': groups, 'datasets': datasets})
    return {'root_keys': keys, 'pairs_metadata': metadata,
            'schema_summary': {'pair_count': len(keys), 'dataset_metadata_count': sum(len(p['datasets']) for p in metadata),
                               'root_object_kind': 'group', 'all_links_hard': True,
                               'dataset_values_read': 0, 'attribute_values_read': 0}}


def download(asset, target, receipt, item, save, opener):
    """One GET; read at most the exact frozen Content-Length, in bounded chunks."""
    item.update(status='requesting', body_bytes=0, body_bytes_are_exact=True, sha256=None)
    receipt['requested_body_upper_bound'] += asset['bytes']
    require(receipt['requested_body_upper_bound'] <= MAX_NETWORK_BYTES, 'requested payload exceeds frozen bound')
    save()
    request = urllib.request.Request(asset['url'], headers={'Accept-Encoding': 'identity'})
    start = time.monotonic()
    digest = hashlib.sha256()
    try:
        with target.open('xb') as output:
            def accept(data):
                require(isinstance(data, bytes), 'HTTP body chunk must be bytes')
                item['body_bytes'] += len(data)
                receipt['network_body_bytes'] += len(data)
                require(item['body_bytes'] <= asset['bytes']
                        and receipt['network_body_bytes'] <= MAX_NETWORK_BYTES, 'HTTP body exceeded bounded read')
                output.write(data)
                digest.update(data)
            try:
                with deadline(REQUEST_SECONDS), opener(request, timeout=REQUEST_SECONDS) as response:
                    item['http_status'] = response.status
                    item['content_length'] = response.headers.get('Content-Length')
                    item['last_modified'] = response.headers.get('Last-Modified')
                    item['etag'] = response.headers.get('ETag')
                    require(response.status == 200, 'expected HTTP200; body not read')
                    require(response.geturl() == asset['url'], 'response URL differs; body not read')
                    require(response.headers.get('Content-Encoding', 'identity') == 'identity', 'encoded response refused')
                    require(response.headers.get('Content-Length') == str(asset['bytes']), 'Content-Length differs; body not read')
                    require(response.headers.get('Transfer-Encoding') is None, 'transfer encoding refused')
                    item['status'] = 'downloading'
                    save()
                    while item['body_bytes'] < asset['bytes']:
                        maximum = min(CHUNK_BYTES, asset['bytes'] - item['body_bytes'])
                        try:
                            data = response.read(maximum)
                        except http.client.IncompleteRead as exc:
                            accept(exc.partial)
                            raise ValueError('incomplete HTTP body') from exc
                        except BaseException:
                            item['body_bytes_are_exact'] = False
                            receipt['network_body_bytes_are_exact'] = False
                            raise
                        require(bool(data), 'short HTTP body')
                        accept(data)
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                output.flush()
                os.fsync(output.fileno())
                item['sha256'] = digest.hexdigest()
                item['stored_bytes'] = output.tell()
        require(target.stat().st_size == asset['bytes'], 'stored size differs')
        item['status'] = 'downloaded'
    finally:
        item['elapsed_seconds'] = time.monotonic() - start
        save()


def acquire(cache_dir, receipt_path, manifest_path, opener=None, h5_module=None):
    """All output locations are exclusive; a failed invocation never auto-retries."""
    cache_dir, receipt_path, manifest_path = map(Path, (cache_dir, receipt_path, manifest_path))
    targets = [cache_dir / asset['filename'] for asset in ASSETS]
    paths = [receipt_path, manifest_path, *targets]
    require(len({str(p.resolve()) for p in paths}) == len(paths), 'output paths overlap')
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError('preserve existing output: ' + str(path))
    require(PROTOCOL.is_file(), 'asset protocol must exist before acquisition')
    require(sum(a['bytes'] for a in ASSETS) == MAX_NETWORK_BYTES, 'asset length sum differs from payload cap')
    require(all(a['url'] == BASE_URL + a['filename'] and a['url'].startswith('https://') for a in ASSETS), 'asset URL changed')
    source = {'experiments/0017-depth-information/acquire_and_allocate.py': sha256(__file__),
              'protocols/0017-depth-assets.md': sha256(PROTOCOL)}
    assets_hash = hashlib.sha256(canonical(ASSETS)).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('x+') as stream:
        receipt = {'schema': 'evr0017.depth-acquisition.v1', 'status': 'started',
                   'source_sha256': source, 'assets_sha256': assets_hash,
                   'assets': list(ASSETS), 'requests': [], 'network_body_bytes': 0,
                   'network_body_bytes_are_exact': True, 'requested_body_upper_bound': 0,
                   'maximum_network_body_bytes': MAX_NETWORK_BYTES, 'automatic_retries': 0,
                   'dataset_values_read': 0, 'attribute_values_read': 0,
                   'development_forever': [a['scene'] for a in ASSETS]}
        started = time.monotonic()
        def save():
            encoded = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n'
            stream.seek(0); stream.write(encoded); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        save()
        try:
            opener = opener or urllib.request.build_opener(NoRedirect()).open
            for asset, target in zip(ASSETS, targets):
                item = {'scene': asset['scene'], 'filename': asset['filename'], 'url': asset['url'],
                        'expected_bytes': asset['bytes']}
                receipt['requests'].append(item)
                download(asset, target, receipt, item, save, opener)
            # No HDF5 is opened until every frozen payload has completed.
            receipt['status'] = 'metadata_allocation'; save()
            scenes, selected_pairs = [], []
            for asset, target, item in zip(ASSETS, targets, receipt['requests']):
                with deadline(METADATA_SECONDS):
                    metadata = inspect_metadata(target, h5_module)
                require(target.stat().st_size == asset['bytes'] and sha256(target) == item['sha256'],
                        'downloaded bytes changed during metadata inspection')
                selected = select(asset['scene'], metadata['root_keys'])
                scene = {**asset, 'sha256': item['sha256'], **metadata, 'selected_pair_keys': selected}
                scenes.append(scene)
                item['metadata_status'] = 'completed'; item['pair_count'] = len(metadata['root_keys'])
                save()
                for key in selected:
                    selected_pairs.append({'pair_index': len(selected_pairs), 'scene': asset['scene'], 'key': key,
                        'filename': asset['filename'], 'dataset_sha256': item['sha256'], 'dataset_bytes': asset['bytes']})
            require(len(selected_pairs) == 80, 'allocation must contain80 pairs')
            require(source['protocols/0017-depth-assets.md'] == sha256(PROTOCOL)
                    and source['experiments/0017-depth-information/acquire_and_allocate.py'] == sha256(__file__), 'source changed during acquisition')
            manifest = {'schema': 'evr0017.development.v1', 'source_sha256': source, 'assets_sha256': assets_hash,
                        'selection_salt': SALT, 'selection': '16 lowest (salted SHA256,key) per fixed scene; names only',
                        'pairs_per_scene': 16, 'pair_count': 80, 'scenes': scenes, 'selected_pairs': selected_pairs,
                        'development_forever': [a['scene'] for a in ASSETS],
                        'coordinate_convention': 'authored_raw_pixels', 'matcher': 'roma', 'depth_label': 'moge',
                        'checkpoint_identity_verified': False, 'dataset_values_read': 0, 'attribute_values_read': 0}
            write_new(manifest_path, manifest)
            receipt.update(status='completed', manifest_sha256=sha256(manifest_path), selected_pairs=80,
                           elapsed_seconds=time.monotonic() - started)
            save()
            return receipt, manifest
        except BaseException as exc:
            receipt.update(status='failed', error=type(exc).__name__ + ': ' + str(exc),
                           elapsed_seconds=time.monotonic() - started, manifest_written=manifest_path.exists())
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir', 'receipt', 'manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    acquire(args.cache_dir, args.receipt, args.manifest)


if __name__ == '__main__':
    main()
