"""Draft 0029 offline installer. No work occurs on import.

Before execution copy byte-identically into the repository, freeze its source,
protocol and explicit sdist-review policy, and use the existing resource monitor.
No scientific inputs/models are opened or invoked. DAM is copied, not qualified.
"""
import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time

SELF = 'experiments/0029-memory-information/install_runtime.py'
ACQUIRE = 'experiments/0029-memory-information/acquire_assets.py'
HELPER = 'experiments/0022-frontend/acquire.py'
ACQ_PROTOCOL = 'protocols/0029-runtime-assets.md'
SUPERVISOR = 'experiments/0027-affine-tangent/supervise.py'
MONITOR = 'experiments/0025c-rs-noise/supervise.py'
MANIFESTS = ('configs/0029-sam2-assets.json', 'configs/0029-runtime-assets.json')
BATCHES = ('sam2-assets', 'runtime-assets')
OLD_MANIFESTS = ('configs/0022-source-model-assets.json', 'configs/0022-wheel-assets.json')
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
TOOLS = {'pip': '25.2', 'setuptools': '80.9.0', 'wheel': '0.45.1'}

# Runs under -S: the audit hook is installed BEFORE site.main() processes .pth.
# pip's only admitted child is its fixed PEP517 Python helper, wrapped in this
# same launcher. This is a Python/process guard, not an OS sandbox/certificate.
GUARD = r'''import os, pathlib, runpy, subprocess, sys
_here = pathlib.Path(__file__).resolve()
_python = pathlib.Path(sys.executable).absolute()
_mode, _target, *_args = sys.argv[1:]
_real_popen = subprocess.Popen

def audit(event, args):
    if event.startswith('socket.') or event in ('urllib.Request', 'os.system', 'os.posix_spawn', 'os.posix_spawnp') or event.startswith(('os.exec', 'os.spawn')):
        raise PermissionError('offline guard: ' + event)
    if event == 'subprocess.Popen':
        executable, argv = args[:2]
        if not isinstance(argv, (list, tuple)) or len(argv) < 5 or pathlib.Path(str(executable)).absolute() != _python or argv[:4] != [str(_python), '-S', str(_here), 'script']:
            raise PermissionError('unapproved subprocess')
sys.addaudithook(audit)

def guarded_popen(argv, *args, **kwargs):
    if kwargs.get('shell') or kwargs.get('executable') is not None or not isinstance(argv, (list, tuple)) or len(argv) != 4:
        raise PermissionError('only explicit PEP517 Python hook subprocess admitted')
    script = pathlib.Path(str(argv[1])).resolve()
    expected = pathlib.Path(sys.prefix) / 'lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py'
    if pathlib.Path(str(argv[0])).absolute() != _python or script != expected.resolve() or argv[2] not in ('get_requires_for_build_wheel', 'prepare_metadata_for_build_wheel', 'build_wheel'):
        raise PermissionError('unapproved build helper')
    child = [str(_python), '-S', str(_here), 'script', str(script), *argv[2:]]
    env = dict(kwargs.get('env') or os.environ)
    for key, value in {'PIP_NO_INDEX':'1', 'PIP_NO_INPUT':'1', 'PIP_DISABLE_PIP_VERSION_CHECK':'1', 'PIP_CONFIG_FILE':os.devnull, 'PYTHONNOUSERSITE':'1', 'PYTHONDONTWRITEBYTECODE':'1'}.items():
        env[key] = value
    env.pop('PYTHONHOME', None)
    env['PYTHONPATH'] = ''
    kwargs['env'] = env
    return _real_popen(child, *args, **kwargs)
subprocess.Popen = guarded_popen
import site
site.main()
if sys.version_info[:3] != (3, 12, 13):
    raise RuntimeError('exact CPython3.12.13 required')
# Bootstrap pip alone from an already verified wheel; no dependency discovery.
_bootstrap = os.environ.get('EVR_BOOTSTRAP_PIP', '')
if _bootstrap:
    sys.path.insert(0, _bootstrap)
sys.argv = [_target, *_args]
if _mode == 'module':
    runpy.run_module(_target, run_name='__main__', alter_sys=True)
elif _mode == 'script':
    runpy.run_path(_target, run_name='__main__')
else:
    raise ValueError('invalid guarded invocation mode')
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=bad)


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def safe_relative(value):
    require(isinstance(value, str) and '\\' not in value, 'invalid relative path')
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and all(re.fullmatch(r'[A-Za-z0-9_.+-]+', x) and x not in ('.', '..') for x in p.parts), 'unsafe relative path')
    return p


def file_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing/symlink file: ' + str(path))
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def git_bytes(root, commit, relative):
    require(re.fullmatch(r'[0-9a-f]{40}', str(commit)) is not None, 'invalid commit')
    safe_relative(relative)
    return subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=root, timeout=15)


def frozen(root, commit, paths):
    result = {}
    for relative in paths:
        expected = git_bytes(root, commit, relative)
        path = Path(__file__) if relative == SELF else root / relative
        require(not path.is_symlink() and path.read_bytes() == expected, 'source differs from freeze: ' + relative)
        result[relative] = hashlib.sha256(expected).hexdigest()
    return result


def display(value, root, out, prior, acq):
    text = str(value)
    for path, alias in [(out, '$INSTALLATION'), (prior, '$PRIOR_ASSETS'), (acq, '$ACQUISITION'), (root, '$REPOSITORY'), (root.parent, '$PRIVATE')]:
        text = text.replace(str(path), alias)
    return text


def verify_acquisition(root, acq, outer_path, spec_path, manifests):
    operation = read_json(acq / 'operation.json'); outer = read_json(outer_path); spec = read_json(spec_path)
    commit = operation.get('source_commit')
    paths = (*MANIFESTS, ACQ_PROTOCOL, HELPER, ACQUIRE)
    frozen_hashes = {p: hashlib.sha256(git_bytes(root, commit, p)).hexdigest() for p in paths}
    require(operation.get('schema') == 'evr0029.asset-operation.v1' and operation.get('status') == 'completed' and operation.get('bindings') == frozen_hashes, 'incomplete/unbound acquisition')
    require(operation.get('asset_or_dependency_only') is True and operation.get('benchmark_payload_access') is False, 'acquisition scope')
    for path in MANIFESTS:
        require(sha(root / path) == frozen_hashes[path], 'acquired manifest changed')
    require(outer.get('schema') == 'evr0026.operation.v1' and outer.get('status') == 'completed' and type(outer.get('exit_code')) is int and outer['exit_code'] == 0 and outer.get('termination') is None and outer.get('remaining_descendant_pids') == [] and outer.get('cleanup_errors') == [], 'unclean acquisition outer')
    require(outer.get('expected_commit') == commit and outer.get('spec_sha256') == sha(spec_path) and outer.get('output_sha256') == sha(acq / 'operation.json'), 'outer binding mismatch')
    require(outer.get('supervisor_sha256') == hashlib.sha256(git_bytes(root, commit, SUPERVISOR)).hexdigest() and outer.get('monitor_sha256') == hashlib.sha256(git_bytes(root, commit, MONITOR)).hexdigest(), 'monitor source mismatch')
    require(outer.get('hard_timeout_seconds') == 1800 and outer.get('group_rss_limit_kib') == 8388608 and outer.get('thread_environment') == {k: '4' for k in THREAD_KEYS} and number(outer.get('wall_seconds'), 0, 1800) and type(outer.get('peak_group_rss_kib')) is int and 0 <= outer['peak_group_rss_kib'] <= 8388608, 'resource receipt mismatch')
    require(spec.get('commit') == commit and Path(spec['repo']).resolve() == root and Path(spec['operation_output']).resolve() == outer_path.parent and Path(spec['result_file']).resolve() == acq / 'operation.json' and spec.get('timeout') == 1800, 'spec identity mismatch')
    require(all(spec.get('environment', {}).get(k) == '4' for k in THREAD_KEYS), 'spec thread mismatch')
    command = spec.get('command')
    require(isinstance(command, list) and len(command) == 9 and all(isinstance(x, str) for x in command), 'unexpected acquisition command')
    expected_tail = ['-B', str(root / ACQUIRE), '--repo-root', str(root), '--expected-commit', commit, '--outputdir', str(acq)]
    require(command[1:] == expected_tail, 'acquisition command arguments mismatch')
    observed = outer.get('command')
    expected = [x.replace(str(root), '$REPOSITORY').replace(str(root.parent), '$PRIVATE') for x in command]
    require(isinstance(observed, list) and observed[1:] == expected[1:] and observed[0] in (expected[0], '$PYTHON'), 'outer command mismatch')
    logs = {}
    for name in ('stdout.log', 'stderr.log'):
        logs[name] = file_record(outer_path.parent / name)['sha256']
    require(outer.get('log_sha256') == logs, 'outer logs changed')
    lines = (outer_path.parent / 'stdout.log').read_text().splitlines()
    nonempty = [line for line in lines if line.strip()]
    require(nonempty and json.loads(nonempty[-1]) == {k: operation[k] for k in ('status', 'wall_seconds', 'batches')}, 'acquisition stdout not bound to operation')
    batches = operation.get('batches')
    require(isinstance(batches, list) and len(batches) == 2, 'acquisition batches incomplete')
    total = 0
    for row, name, manifest_path, manifest in zip(batches, BATCHES, MANIFESTS, manifests):
        ledger_path = acq / name / 'ledger.json'; ledger = read_json(ledger_path)
        require(row.get('directory') == name and row.get('manifest') == manifest_path and row.get('status') == 'completed' and row.get('ledger_sha256') == sha(ledger_path), 'batch receipt mismatch')
        verify_ledger(ledger, manifest)
        require(row.get('actual_body_bytes') == ledger['actual_body_bytes'], 'batch byte mismatch')
        total += ledger['actual_body_bytes']
    require(operation.get('actual_body_bytes') == total and operation.get('body_cap_bytes') == 300000000 and number(operation.get('wall_seconds'), 0, 1800), 'acquisition total mismatch')
    return {'commit': commit, 'operation_sha256': sha(acq / 'operation.json'), 'outer_sha256': sha(outer_path), 'spec_sha256': sha(spec_path), 'log_sha256': logs, 'batch_ledger_sha256': {n: sha(acq / n / 'ledger.json') for n in BATCHES}, 'source_sha256': frozen_hashes}


def verify_ledger(ledger, manifest):
    require(ledger.get('schema') == 1 and ledger.get('status') == 'completed' and ledger.get('manifest_sha256') == canonical(manifest), 'incomplete/wrong asset ledger')
    cap = manifest['body_cap_bytes']; body = ledger.get('actual_body_bytes')
    require(ledger.get('body_cap_bytes') == cap and type(body) is int and 0 <= body <= cap, 'asset ledger budget')
    rows = ledger.get('assets'); requests = ledger.get('requests')
    require(isinstance(rows, list) and len(rows) == len(manifest['assets']) and [r.get('id') for r in rows] == [a['id'] for a in manifest['assets']], 'asset ledger order/identity')
    require(isinstance(requests, list) and all(type(r.get('bytes_read')) is int and r['bytes_read'] >= 0 and r.get('status') in ('complete', 'redirected') for r in requests), 'request ledger status/accounting')
    require(sum(r['bytes_read'] for r in requests) == body and sum(r.get('new_body_bytes', -1) for r in rows) == body, 'asset byte sum mismatch')
    for row, asset in zip(rows, manifest['assets']):
        allowed = ('downloaded_size_checked_first_hash',) if asset.get('digest_mode') == 'first_download_sha256' else ('downloaded_verified', 'reused_verified')
        require(row.get('status') in allowed and re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256'))) is not None, 'asset incomplete')


def verify_asset(path, asset, expected_ledger_sha=None):
    rec = file_record(path)
    require(rec['bytes'] == asset['bytes'] and ('sha256' not in asset or rec['sha256'] == asset['sha256']) and (expected_ledger_sha is None or rec['sha256'] == expected_ledger_sha), 'asset size/SHA mismatch: ' + asset['id'])
    if 'git_blob_sha1' in asset:
        h = hashlib.sha1(b'blob ' + str(rec['bytes']).encode() + b'\0')
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
        require(h.hexdigest() == asset['git_blob_sha1'], 'Git blob mismatch')
    return rec


def admitted_inputs(root, acq, prior, manifests):
    new = {}; reused = {}; bindings = {}
    for directory, manifest in zip(BATCHES, manifests):
        ledger = read_json(acq / directory / 'ledger.json'); verify_ledger(ledger, manifest)
        for asset, row in zip(manifest['assets'], ledger['assets']):
            path = acq / directory / asset['filename']; rec = verify_asset(path, asset, row['sha256'])
            new[asset['id']] = (asset, path); bindings[directory + '/' + asset['filename']] = rec
    old = [read_json(root / p) for p in OLD_MANIFESTS]
    for directory, manifest in zip(('source-model-assets', 'wheel-assets'), old):
        verify_ledger(read_json(prior / directory / 'ledger.json'), manifest)
    prior_op = read_json(prior / 'operation.json')
    require(prior_op.get('schema') == 'evr0022.asset-operation.v1' and prior_op.get('status') == 'completed', 'old input operation incomplete')
    require(all(row.get('status') == 'completed' and row.get('ledger_sha256') == sha(prior / row['directory'] / 'ledger.json') for row in prior_op.get('batches', [])) and len(prior_op.get('batches', [])) == 2, 'old input ledgers changed')
    for index, directory in enumerate(('source-model-assets', 'wheel-assets')):
        selected = manifests[index]['non_downloaded_inputs']
        expected = selected['lightglue_source']['source_manifest_sha256'] if index == 0 else selected['source_manifest_sha256']
        require(sha(root / OLD_MANIFESTS[index]) == expected, 'old manifest mismatch')
        assets = {a['id']: a for a in old[index]['assets']}
        ledger = {x['id']: x for x in read_json(prior / directory / 'ledger.json')['assets']}
        entries = selected['model_assets'] if index == 0 else selected['wheels']
        for row in entries:
            asset = assets[row['id']]; require(asset['filename'] == row['filename'] and asset['bytes'] == row['bytes'], 'reuse identity mismatch')
            if index == 0: require(sha(prior / directory / 'ledger.json') == row['source_receipt_sha256'], 'TOFU source receipt changed')
            path = prior / directory / row['filename']; rec = verify_asset(path, row, ledger[row['id']]['sha256'])
            reused[row['id']] = (asset, path); bindings['prior/' + directory + '/' + row['filename']] = rec
        if index == 0:
            for layout in selected['lightglue_source']['source_layout']:
                asset = next(a for a in old[0]['assets'] if a['filename'] == layout['filename'])
                path = prior / directory / asset['filename']; rec = verify_asset(path, asset, ledger[asset['id']]['sha256'])
                reused[asset['id']] = (asset, path); bindings['prior/' + directory + '/' + asset['filename']] = rec
    return new, reused, bindings


def inspect_reviewed_sdists(review, assets):
    require(review.get('schema') == 'evr0029.sdist-build-review.v1' and review.get('approved') is True and review.get('network') == 'disabled' and review.get('build_tool_versions') == TOOLS, 'missing explicit sdist build approval')
    expected = {k: v for k, v in assets.items() if v[0].get('kind') == 'sdist'}
    rows = review.get('sdists'); require(isinstance(rows, list) and len(rows) == 2 and {r.get('id') for r in rows} == set(expected), 'sdist review identities')
    for row in rows:
        asset, path = expected[row['id']]
        require(row.get('approved') is True and row.get('sha256') == asset['sha256'] and row.get('filename') == asset['filename'] and isinstance(row.get('review_note'), str) and row['review_note'].strip(), 'sdist not approved at exact hash')
        root = str(safe_relative(row['archive_root'])); files = row.get('files'); dirs = row.get('directories')
        require(isinstance(files, dict) and files and isinstance(dirs, list) and len(dirs) == len(set(dirs)), 'complete archive inventory required')
        require(isinstance(row.get('build_hooks'), list) and root + '/setup.py' in row['build_hooks'] and all(x in files for x in row['build_hooks']), 'reviewed build hooks required')
        observed = {}; directories = []
        with tarfile.open(path, mode='r:gz') as archive:
            for member in archive.getmembers():
                name = member.name.rstrip('/') if member.isdir() else member.name
                safe_relative(name); require(name == root or name.startswith(root + '/'), 'archive escapes declared root')
                require(name not in observed and name not in directories, 'duplicate archive member')
                if member.isdir(): directories.append(name); continue
                require(member.isfile() and member.size <= 8 * 1024 * 1024 and name in files, 'unapproved archive member/type')
                stream = archive.extractfile(member); require(stream is not None, 'missing archive stream')
                data = stream.read(8 * 1024 * 1024 + 1)
                observed[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                require(len(data) == member.size and observed[name] == files[name], 'reviewed archive bytes changed')
        require(observed == files and sorted(directories) == sorted(dirs) and sum(x['bytes'] for x in observed.values()) <= 8 * 1024 * 1024, 'archive review incomplete/oversized')
    return rows


def inventory(root):
    result = {}
    paths = [root / 'offline_guard.py']
    for name in ('sources', 'weights', 'venv', 'built-wheels'):
        directory = root / name
        require(directory.is_dir() and not directory.is_symlink(), 'installed root missing/symlink')
        paths.extend(directory.rglob('*'))
    for path in sorted(paths):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            require(rel.startswith('venv/bin/python') and path.resolve().is_file(), 'unexpected installed symlink')
            result[rel] = {'type': 'symlink', 'target': os.readlink(path), 'resolved_sha256': sha(path.resolve())}
        elif path.is_file(): result[rel] = {'type': 'file', **file_record(path)}
    return result


def installed_metadata_path(dist, site):
    """Return this installed distribution's own top-level METADATA.

    RECORD also inventories vendored distributions. Those nested records remain
    in the complete runtime inventory but are not separate installed packages.
    """
    candidates = []
    for entry in dist.files or ():
        parts = PurePosixPath(str(entry)).parts
        if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == "METADATA":
            candidates.append(entry)
    if len(candidates) != 1:
        raise ValueError("installed METADATA missing/ambiguous")
    path = Path(dist.locate_file(candidates[0]))
    if (path.is_symlink() or not path.is_file()
            or path.parent.is_symlink() or path.parent.parent.resolve() != Path(site).resolve()):
        raise ValueError("installed METADATA outside direct site-packages distribution")
    if path.read_text(encoding="utf-8") != dist.read_text("METADATA"):
        raise ValueError("installed METADATA identity mismatch")
    return path

def run(args):
    root = Path(args.repo_root).resolve(); acq = Path(args.acquisition).resolve(); prior = Path(args.prior_assets).resolve(); out = Path(args.destination).absolute()
    require(not out.exists() and not out.is_symlink() and root not in out.resolve().parents and out.resolve() != root, 'destination must be fresh and private')
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic(); record = {'schema': 'evr0029.install.v1', 'status': 'started', 'expected_commit': args.expected_commit, 'phase': 'source_admission', 'commands': [], 'scientific_input_access': False, 'model_invocations': 0}
    def save():
        record['wall_seconds'] = time.monotonic() - began
        pending = out / 'install-receipt.pending'; pending.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n'); os.replace(pending, out / 'install-receipt.json')
    def command(name, argv, environment, cwd=None):
        row = {'name': name, 'command': [display(x, root, out, prior, acq) for x in argv], 'status': 'started'}
        record['commands'].append(row); save(); start = time.monotonic()
        try:
            with (out / (name + '-stdout.log')).open('xb') as stdout, (out / (name + '-stderr.log')).open('xb') as stderr:
                result = subprocess.run(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, timeout=max(0.01, 1800 - (time.monotonic() - began)), check=False)
            row['exit_code'] = result.returncode; require(result.returncode == 0, name + ' failed')
            row['status'] = 'completed'
        except BaseException as error:
            row.update(status='failed', error_type=type(error).__name__); raise
        finally:
            row['wall_seconds'] = time.monotonic() - start
            row['logs'] = {k: file_record(out / (name + '-' + k + '.log')) for k in ('stdout', 'stderr') if (out / (name + '-' + k + '.log')).is_file()}; save()
    save()
    try:
        protocol = Path(args.protocol).resolve().relative_to(root).as_posix(); review_path = Path(args.build_review).resolve(); review_rel = review_path.relative_to(root).as_posix()
        source_paths = [SELF, *MANIFESTS, *OLD_MANIFESTS, protocol, review_rel]
        record['source_sha256'] = frozen(root, args.expected_commit, source_paths)
        manifests = [read_json(root / p) for p in MANIFESTS]
        record['acquisition_bindings'] = verify_acquisition(root, acq, Path(args.acquisition_supervision).resolve(), Path(args.acquisition_spec).resolve(), manifests)
        new, reused, bindings = admitted_inputs(root, acq, prior, manifests); record['input_files'] = bindings
        review = read_json(review_path); inspect_reviewed_sdists(review, new)
        record['build_review_sha256'] = sha(review_path); record['phase'] = 'copy_unchanged_sources'; save()
        copies = []
        for layout in manifests[0]['source_layout']:
            filename = layout['filename']; asset, src = new[filename]
            expected = 'sources/' + layout['install_prefix'] + '/' + layout['relative_path']
            require(layout.get('install_path') == expected and asset.get('install_path') == expected and layout['revision'] == asset['revision'] and layout['repository'] == asset['repository'], 'source layout mismatch')
            copies.append((src, str(safe_relative(expected)), asset))
        for layout in manifests[0]['non_downloaded_inputs']['lightglue_source']['source_layout']:
            asset, src = reused[layout['filename']]
            require(layout['install_prefix'] == 'lightglue' and layout['revision'] == 'eb42fee2d71449efb0aa5c10549752b5d75384d8' and layout['repository'] == 'cvg/LightGlue', 'LightGlue source pin')
            copies.append((src, str(safe_relative('sources/lightglue/' + layout['relative_path'])), asset))
        for key, (asset, src) in {**new, **reused}.items():
            if asset.get('kind') == 'model_checkpoint' or key in ('superpoint_v1.pth', 'superpoint_lightglue.pth'):
                copies.append((src, str(safe_relative('weights/' + asset['filename'])), asset))
        require(len({dst for _, dst, _ in copies}) == len(copies), 'duplicate copied destinations')
        for src, dst, asset in copies:
            target = out / dst; target.parent.mkdir(parents=True, exist_ok=True)
            with src.open('rb') as source, target.open('xb') as dest: shutil.copyfileobj(source, dest, 1024 * 1024)
            verify_asset(target, asset, sha(src))
        record['copied_files'] = {dst: file_record(out / dst) for _, dst, _ in copies}
        guard = out / 'offline_guard.py'; guard.write_text(GUARD); record['guard_sha256'] = sha(guard)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_', 'UV_', 'EVR_'))}
        env.update({k: '4' for k in THREAD_KEYS}); env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='', PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX='1', PIP_NO_INPUT='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1')
        tmp = out / 'tmp'; tmp.mkdir(); env['TMPDIR'] = str(tmp)
        python = str(Path(args.python).absolute()); venv = out / 'venv'
        record['base_python'] = {'path': python, 'resolved_path': str(Path(python).resolve()), **file_record(Path(python).resolve())}
        record['phase'] = 'create_venv'; save()
        command('create-venv', [python, '-S', str(guard), 'module', 'venv', '--without-pip', str(venv)], env)
        py = str(venv / 'bin/python'); wheel_assets = [(a, p) for a, p in [*new.values(), *reused.values()] if a['filename'].endswith('.whl')]
        pipwheel = next(path for asset, path in wheel_assets if asset['id'] == 'pip'); env['EVR_BOOTSTRAP_PIP'] = str(pipwheel)
        record['phase'] = 'install_verified_wheels'; save()
        command('install-wheels', [py, '-S', str(guard), 'module', 'pip', '--disable-pip-version-check', 'install', '--no-index', '--no-deps', '--no-compile', '--no-cache-dir', *[str(path) for _, path in wheel_assets]], env)
        env.pop('EVR_BOOTSTRAP_PIP'); site = venv / 'lib/python3.12/site-packages'
        buildroot = out / 'built-wheels'; buildroot.mkdir(); built = []
        for row in review['sdists']:
            asset, src = new[row['id']]; destination = buildroot / asset['id']; destination.mkdir()
            record['phase'] = 'build_' + asset['id']; save()
            command('build-' + asset['id'], [py, '-S', str(guard), 'module', 'pip', '--disable-pip-version-check', 'wheel', '--use-pep517', '--no-build-isolation', '--no-deps', '--no-index', '--no-cache-dir', '--wheel-dir', str(destination), str(src)], env)
            wheels = list(destination.iterdir()); require(len(wheels) == 1 and wheels[0].suffix == '.whl' and not wheels[0].is_symlink(), 'build must emit exactly one wheel')
            built.append({'id': asset['id'], 'source_sha256': asset['sha256'], 'wheel': wheels[0].relative_to(out).as_posix(), **file_record(wheels[0])})
        record['built_wheels'] = built; record['phase'] = 'install_reviewed_builds'; save()
        command('install-built', [py, '-S', str(guard), 'module', 'pip', '--disable-pip-version-check', 'install', '--no-index', '--no-deps', '--no-compile', '--no-cache-dir', *[str(out / x['wheel']) for x in built]], env)
        distributions = []
        for dist in metadata.distributions(path=[str(site)]):
            path = installed_metadata_path(dist, site)
            distributions.append({'name': dist.metadata['Name'], 'version': dist.version, 'metadata': path.relative_to(out).as_posix(), **file_record(path)})
        normalize = lambda s: re.sub(r'[-_.]+', '-', s).lower()
        expected = {(normalize(a['name']), a['version']) for a in manifests[1]['assets'] + manifests[1]['non_downloaded_inputs']['wheels']}
        require(len(distributions) == len(expected) and {(normalize(d['name']), d['version']) for d in distributions} == expected, 'installed versions differ from exact closure')
        record['installed_distributions'] = distributions
        require(frozen(root, args.expected_commit, source_paths) == record['source_sha256'], 'source changed during install')
        require(verify_acquisition(root, acq, Path(args.acquisition_supervision).resolve(), Path(args.acquisition_spec).resolve(), manifests) == record['acquisition_bindings'], 'acquisition chain changed')
        require(admitted_inputs(root, acq, prior, manifests)[2] == bindings, 'input assets changed')
        require({dst: file_record(out / dst) for _, dst, _ in copies} == record['copied_files'] and sha(guard) == record['guard_sha256'], 'installed source/weights/guard changed')
        record['installed_inventory'] = inventory(out)
        require(time.monotonic() - began < 1800, 'total installation wall limit')
        record.update(status='completed', phase='completed', dam_qualified=False, native_inference_qualified=False)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=display(error, root, out, prior, acq)); raise
    finally:
        save()
        print(json.dumps({k: record[k] for k in ('status', 'phase', 'wall_seconds')}, allow_nan=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo-root', 'expected-commit', 'protocol', 'acquisition', 'acquisition-supervision', 'acquisition-spec', 'prior-assets', 'build-review', 'destination'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--python', default='/opt/homebrew/bin/python3.12')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
