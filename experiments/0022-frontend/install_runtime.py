"""Verify acquired 0022 assets and install wheels only into a fresh private target."""
import argparse
import hashlib
from importlib import metadata, util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'moge': ('microsoft/MoGe', '925b8ed835a7a9cdb7578ba15c658a0afc969030'),
    'utils3d': ('EasternJournalist/utils3d', '3fab839f0be9931dac7c8488eb0e1600c236e183'),
    'lightglue': ('cvg/LightGlue', 'eb42fee2d71449efb0aa5c10549752b5d75384d8'),
}


def load_helper(name):
    spec = util.spec_from_file_location('evr0022_' + name, Path(__file__).with_name(name + '.py'))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_batch(directory, manifest_path, wheels=False):
    directory, manifest_path = Path(directory), Path(manifest_path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('asset directory must be a real directory')
    manifest = read_json(manifest_path)
    acquire = load_helper('acquire')
    cap = acquire.validate(manifest)
    ledger_path = directory / 'ledger.json'
    if ledger_path.is_symlink():
        raise ValueError('symlink ledger')
    ledger = read_json(ledger_path)
    canonical = canonical_sha256(manifest)
    if ledger.get('schema') != 1 or ledger.get('status') != 'completed' or ledger.get('manifest_sha256') != canonical:
        raise ValueError('acquisition ledger is incomplete or belongs to another manifest')
    if ledger.get('body_cap_bytes') != cap or type(ledger.get('actual_body_bytes')) is not int or not 0 <= ledger['actual_body_bytes'] <= cap:
        raise ValueError('invalid acquisition budget record')
    records = ledger.get('assets')
    if not isinstance(records, list) or len(records) != len(manifest['assets']):
        raise ValueError('acquisition asset count mismatch')
    by_id = {}
    for row in records:
        if not isinstance(row, dict) or row.get('id') in by_id:
            raise ValueError('duplicate acquisition record')
        by_id[row.get('id')] = row
    if set(by_id) != {a['id'] for a in manifest['assets']}:
        raise ValueError('acquisition asset identity mismatch')
    hashes = {}
    for asset in manifest['assets']:
        row = by_id[asset['id']]
        tofu = asset.get('digest_mode') == 'first_download_sha256'
        allowed = {'downloaded_size_checked_first_hash'} if tofu else {'downloaded_verified', 'reused_verified'}
        if row.get('status') not in allowed or not re.fullmatch('[0-9a-f]{64}', str(row.get('sha256', ''))):
            raise ValueError('asset has no completed hash binding')
        if wheels and (not asset['filename'].endswith('.whl') or 'sha256' not in asset or tofu):
            raise ValueError('wheel must have a publisher SHA256')
        path = directory / asset['filename']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != asset['bytes']:
            raise ValueError('asset size/type mismatch: ' + asset['filename'])
        sha, git = acquire.digests(asset['bytes'])
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                sha.update(block); git.update(block)
        if not acquire.matches(asset, (sha, git)) or sha.hexdigest() != row['sha256']:
            raise ValueError('asset digest mismatch: ' + asset['filename'])
        hashes[asset['filename']] = sha.hexdigest()
    return manifest, {'manifest_sha256': sha256(manifest_path),
                      'manifest_canonical_sha256': canonical,
                      'acquisition_ledger_sha256': sha256(ledger_path),
                      'asset_sha256': hashes}


def source_layout(manifest):
    rows = manifest.get('source_layout')
    if not isinstance(rows, list) or not rows:
        raise ValueError('missing source layout')
    assets = {a['filename']: a for a in manifest['assets']}
    required = {name for name, asset in assets.items() if 'git_blob_sha1' in asset}
    seen, destinations = set(), set()
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid source layout row')
        prefix, rel, filename = row.get('install_prefix'), row.get('relative_path'), row.get('filename')
        if prefix not in PINS or not isinstance(rel, str) or '\\' in rel:
            raise ValueError('unknown or unsafe source layout')
        path = PurePosixPath(rel)
        if path.is_absolute() or str(path) != rel or any(not re.fullmatch('[A-Za-z0-9_.-]+', p) or p in ('.', '..') for p in path.parts):
            raise ValueError('unsafe source relative path')
        if filename not in required or filename in seen or filename != prefix + '--' + rel.replace('/', '--'):
            raise ValueError('source filename mapping mismatch')
        repository, revision = PINS[prefix]
        if row.get('repository') != repository or row.get('revision') != revision:
            raise ValueError('source revision mismatch')
        if assets[filename]['url'] != 'https://raw.githubusercontent.com/' + repository + '/' + revision + '/' + rel:
            raise ValueError('source URL/layout mismatch')
        destination = 'sources/' + prefix + '/' + rel
        if destination in destinations:
            raise ValueError('duplicate source destination')
        seen.add(filename); destinations.add(destination)
        result.append({'filename': filename, 'relative_path': destination, 'bytes': assets[filename]['bytes']})
    if seen != required or {row['install_prefix'] for row in rows} != set(PINS):
        raise ValueError('source layout is not complete')
    return result


def inventory(directory, destination):
    hashes, sizes = {}, {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in installed runtime')
        if path.is_file():
            key = path.relative_to(destination).as_posix()
            hashes[key], sizes[key] = sha256(path), path.stat().st_size
    return hashes, sizes


def verify_acquisition_chain(args, supervise):
    """A completed batch alone cannot authorize a failed outer operation."""
    operation_path = (Path(args.acquisition_operation) if args.acquisition_operation else
                      Path(args.source_assets).parent / 'operation.json').resolve()
    supervision_path = Path(args.acquisition_supervision).resolve()
    operation, outer = read_json(operation_path), read_json(supervision_path)
    commit = operation.get('source_commit')
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('invalid acquisition commit')
    if operation.get('schema') != 'evr0022.asset-operation.v1' or operation.get('status') != 'completed':
        raise ValueError('acquisition operation is incomplete')
    if (outer.get('schema') != 'evr0022.supervision.v1' or outer.get('operation') != 'acquisition'
            or outer.get('status') != 'completed' or type(outer.get('exit_code')) is not int
            or outer['exit_code'] != 0 or outer.get('termination') is not None
            or outer.get('expected_commit') != commit or outer.get('remaining_descendant_pids') != []
            or outer.get('cleanup_errors') != []):
        raise ValueError('acquisition supervision was not cleanly completed')
    elapsed, peak = outer.get('wall_seconds'), outer.get('peak_group_rss_kib')
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed < 1800
            or type(peak) is not int or not 0 <= peak <= 8 * 1024 * 1024
            or outer.get('hard_timeout_seconds') != 1800
            or outer.get('group_rss_limit_kib') != 8 * 1024 * 1024
            or outer.get('thread_environment') != {key: '4' for key in supervise.THREAD_KEYS}):
        raise ValueError('invalid acquisition resource receipt')
    paths = ['configs/0022-source-model-assets.json', 'configs/0022-wheel-assets.json',
             'protocols/0022-traceable-frontend-assets.md',
             'experiments/0022-frontend/acquire.py', 'experiments/0022-frontend/acquire_operation.py']
    frozen = {path: hashlib.sha256(subprocess.check_output(
        ['git', 'show', commit + ':' + path], cwd=ROOT)).hexdigest() for path in paths}
    if operation.get('bindings') != frozen:
        raise ValueError('acquisition inner source binding differs from its commit')
    frozen_outer = {path: frozen[path] for path in paths if path.startswith('experiments/')}
    helper = 'experiments/0022-frontend/supervise.py'
    frozen_outer[helper] = hashlib.sha256(subprocess.check_output(
        ['git', 'show', commit + ':' + helper], cwd=ROOT)).hexdigest()
    if outer.get('source_sha256') != frozen_outer:
        raise ValueError('acquisition outer source binding differs from its commit')
    if (sha256(args.source_manifest) != frozen[paths[0]] or sha256(args.wheel_manifest) != frozen[paths[1]]):
        raise ValueError('current manifests differ from the acquisition freeze')
    command = outer.get('command')
    if not isinstance(command, list) or len(command) < 2 or command[:2] != [
            '$PYTHON', '$REPOSITORY/experiments/0022-frontend/acquire_operation.py']:
        raise ValueError('acquisition child command mismatch')
    supervise.require_child_commit(command, commit, True)
    output_values = []
    for index, part in enumerate(command):
        if part == '--output' and index + 1 < len(command):
            output_values.append(command[index + 1])
        elif part.startswith('--output='):
            output_values.append(part.split('=', 1)[1])
    if len(output_values) != 1:
        raise ValueError('missing/duplicate acquisition output argument')
    output = output_values[0].replace('$REPOSITORY', str(ROOT)).replace('$PRIVATE', str(ROOT.parent))
    if not Path(output).is_absolute():
        output = str(ROOT / output)
    if Path(output).resolve() != operation_path.parent:
        raise ValueError('acquisition command does not identify supplied operation')
    log_hashes = {}
    for name in ('stdout', 'stderr'):
        path = supervision_path.parent / (name + '.log')
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing acquisition log')
        log_hashes[name] = sha256(path)
        if outer.get(name + '_sha256') != log_hashes[name]:
            raise ValueError('acquisition log hash mismatch')
    lines = (supervision_path.parent / 'stdout.log').read_text().splitlines()
    final_lines = [line for line in lines if line.strip()]
    expected_final = {key: operation[key] for key in ('status', 'wall_seconds', 'batches')}
    if not final_lines or json.loads(final_lines[-1]) != expected_final:
        raise ValueError('acquisition final log does not bind the inner operation')
    batches = operation.get('batches')
    if not isinstance(batches, list) or len(batches) != 2:
        raise ValueError('incomplete acquisition batches')
    ledgers = {}
    for row, relative_manifest, dirname, supplied in zip(batches, paths[:2],
            ('source-model-assets', 'wheel-assets'), (args.source_assets, args.wheel_assets)):
        if (row.get('manifest') != relative_manifest or row.get('directory') != dirname
                or row.get('status') != 'completed'
                or (operation_path.parent / dirname).resolve() != Path(supplied).resolve()):
            raise ValueError('acquisition batch path/status mismatch')
        path = Path(supplied) / 'ledger.json'
        if row.get('ledger_sha256') != sha256(path):
            raise ValueError('acquisition batch ledger changed')
        if row.get('actual_body_bytes') != read_json(path).get('actual_body_bytes'):
            raise ValueError('acquisition batch byte accounting mismatch')
        ledgers[dirname] = sha256(path)
    return {'acquisition_operation_sha256': sha256(operation_path),
            'acquisition_supervision_sha256': sha256(supervision_path),
            'acquisition_source_commit': commit, 'acquisition_log_sha256': log_hashes,
            'acquisition_batch_ledger_sha256': ledgers}


def run(args):
    supervise = load_helper('supervise')
    source_manifest_path = Path(args.source_manifest).resolve()
    wheel_manifest_path = Path(args.wheel_manifest).resolve()
    bindings = supervise.verify_frozen(args.expected_commit, [Path(__file__), Path(__file__).with_name('acquire.py'),
        Path(__file__).with_name('supervise.py'), source_manifest_path, wheel_manifest_path,
        ROOT / 'protocols/0022-traceable-frontend-assets.md'])
    destination = Path(args.destination).absolute()
    if destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('third-party runtime must remain outside the public repository')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    receipt_path = destination / 'install-receipt.json'
    record = {'schema': 'evr0022.install.v1', 'status': 'started',
              'expected_commit': args.expected_commit, 'source_sha256': bindings,
              'destination': '$DESTINATION', 'python_version': sys.version,
              'phase': 'verify_assets'}
    supervise.save(receipt_path, record)
    began = time.perf_counter()
    interrupted = None
    try:
        record.update(verify_acquisition_chain(args, supervise))
        sources, source_binding = verify_batch(args.source_assets, source_manifest_path)
        wheels, wheel_binding = verify_batch(args.wheel_assets, wheel_manifest_path, wheels=True)
        if (source_binding['acquisition_ledger_sha256'] != record['acquisition_batch_ledger_sha256']['source-model-assets']
                or wheel_binding['acquisition_ledger_sha256'] != record['acquisition_batch_ledger_sha256']['wheel-assets']):
            raise ValueError('acquisition ledger changed during installation admission')
        for prefix, binding in [('source', source_binding), ('wheel', wheel_binding)]:
            record.update({prefix + '_' + key: value for key, value in binding.items()})
        layout = source_layout(sources)
        record['phase'] = 'copy_sources'; supervise.save(receipt_path, record)
        for row in layout:
            original = Path(args.source_assets) / row['filename']
            target = destination / row['relative_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            with original.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            row['sha256'] = sha256(target)
            if row['sha256'] != source_binding['asset_sha256'][row['filename']] or target.stat().st_size != row['bytes']:
                raise ValueError('copied source changed')
        record['source_layout'] = layout
        runtime = destination / 'runtime'
        runtime.mkdir()
        command = [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install',
                   '--no-index', '--no-deps', '--no-compile', '--target', str(runtime)]
        command += [str((Path(args.wheel_assets) / a['filename']).resolve()) for a in wheels['assets']]
        record['command'] = [supervise.display(v.replace(str(destination), '$DESTINATION')) for v in command]
        record['phase'] = 'pip_install'; supervise.save(receipt_path, record)
        env = dict(os.environ)
        env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX='1', PIP_NO_INPUT='1',
                   PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='')
        with (destination / 'install-stdout.log').open('xb') as stdout, (destination / 'install-stderr.log').open('xb') as stderr:
            outcome = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)
        record['pip_exit_code'] = outcome.returncode
        if outcome.returncode != 0:
            raise RuntimeError('pip failed with exit code ' + str(outcome.returncode))
        record['phase'] = 'verify_installed'
        installed, sizes = inventory(destination / 'sources', destination)
        runtime_files, runtime_sizes = inventory(runtime, destination)
        installed.update(runtime_files); sizes.update(runtime_sizes)
        for row in layout:
            if installed[row['relative_path']] != row['sha256']:
                raise ValueError('installed source changed during pip')
        distributions = []
        for dist in metadata.distributions(path=[str(runtime)]):
            matches = [f for f in (dist.files or []) if str(f).endswith('.dist-info/METADATA')]
            if len(matches) != 1:
                raise ValueError('missing or ambiguous installed distribution metadata')
            path = Path(dist.locate_file(matches[0])).resolve()
            relative = path.relative_to(destination).as_posix()
            distributions.append({'name': dist.metadata['Name'], 'version': dist.version,
                                  'metadata_path': relative, 'metadata_sha256': installed[relative]})
        normalize = lambda value: re.sub('[-_.]+', '-', value).lower()
        expected_distributions = {(normalize(a['filename'].split('-')[0]), a['filename'].split('-')[1])
                                  for a in wheels['assets']}
        actual_distributions = {(normalize(d['name']), d['version']) for d in distributions}
        if len(distributions) != len(wheels['assets']) or actual_distributions != expected_distributions:
            raise ValueError('installed distribution identities differ from verified wheels')
        record.update(status='completed', phase='completed', installed_files=installed,
                      installed_file_bytes=sizes, installed_distributions=distributions)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=supervise.display(error))
        if not isinstance(error, Exception):
            interrupted = error
    finally:
        record['wall_seconds'] = time.perf_counter() - began
        for name in ('install-stdout.log', 'install-stderr.log'):
            if (destination / name).is_file():
                record[name.replace('-', '_').replace('.', '_') + '_sha256'] = sha256(destination / name)
        supervise.save(receipt_path, record)
    if interrupted is not None:
        raise interrupted
    return 0 if record['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-assets', required=True)
    parser.add_argument('--wheel-assets', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--wheel-manifest', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--acquisition-operation', help='Defaults to source-assets parent/operation.json')
    parser.add_argument('--acquisition-supervision', required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
