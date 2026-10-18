"""Two explicit preparation stages; never constructs or runs a model."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class PrepFailure(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value, exclusive=False):
    path = Path(path)
    target = path if exclusive else path.with_name(path.name + '.tmp')
    with target.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if not exclusive:
        target.replace(path)


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise PrepFailure('Unsafe relative path')
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise PrepFailure('Symlink in preparation path')
    return current


def error_record(error):
    value = {'type': type(error).__name__}
    if isinstance(error, PrepFailure):
        value['reason'] = str(error)
    if isinstance(error, OSError):
        value['errno'] = error.errno
    if isinstance(error, urllib.error.HTTPError):
        value['http_status'] = error.code
    return value  # Never print signed redirect URLs, response bodies or environment secrets.


@contextmanager
def deadline(seconds):
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] != 0:
        raise PrepFailure('Unexpected active in-process deadline')
    def expired(signum, frame):
        raise TimeoutError('Bounded operation exceeded its wall deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 5
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.username or parsed.password:
            raise PrepFailure('Rejected non-HTTPS or credential-bearing redirect')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def default_open(url, timeout):
    # build_opener retains the normal proxy discovery and default HTTPS CA checks.
    opener = urllib.request.build_opener(HTTPSOnlyRedirect())
    return opener.open(urllib.request.Request(url, method='GET'), timeout=timeout)


def validate_config(config):
    if config['schema_version'] != 1:
        raise PrepFailure('Unknown preparation schema')
    budget = config['budget']; assets = config['assets']
    if (budget['historical_ceiling_bytes'] != 4823387101 or
        budget['metadata_allowance_bytes'] != 128 * 1024**2 or
        budget['limit_bytes'] != 10 * 1024**3):
        raise PrepFailure('Unexpected inherited resource budget')
    kinds = [asset['kind'] for asset in assets]
    if kinds.count('source') != 5 or kinds.count('wheel') != 3 or kinds.count('checkpoint') != 1 or len(kinds) != 9:
        raise PrepFailure('Expected five sources, three wheels, one checkpoint')
    if len({a['id'] for a in assets}) != 9 or len({a['relative_path'] for a in assets}) != 9:
        raise PrepFailure('Repeated asset identity or path')
    for asset in assets:
        if not asset['id'].replace('-', '').isalnum():
            raise PrepFailure('Unsafe asset id')
        parsed = urllib.parse.urlsplit(asset['url'])
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.fragment or parsed.query:
            raise PrepFailure('Asset URL must use ordinary HTTPS without credentials/fragments')
        if type(asset['bytes']) is not int or asset['bytes'] < 1:
            raise PrepFailure('Invalid expected length')
        digest = asset['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise PrepFailure('Invalid SHA256')
        rel = PurePosixPath(asset['relative_path'])
        expected_dir = {'source': 'source', 'wheel': 'wheels', 'checkpoint': 'weights'}[asset['kind']]
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != expected_dir:
            raise PrepFailure('Unexpected asset destination')
    transfer = config['transfer']
    if transfer != {'read_timeout_seconds': 60, 'per_asset_timeout_seconds': 840, 'chunk_bytes': 1048576}:
        raise PrepFailure('Transfer bounds changed')
    planned = budget['historical_ceiling_bytes'] + budget['metadata_allowance_bytes'] + sum(a['bytes'] + 1 for a in assets)
    if planned >= budget['limit_bytes']:
        raise PrepFailure('Conservative cumulative acquisition budget exceeded')
    return planned


def acquire(asset, root, ledger, ledger_path, transfer, open_url=default_open):
    """One logical HTTPS transfer (normal HTTPS redirects allowed), never retry."""
    final = safe_path(root, asset['relative_path'])
    partial = final.with_name(final.name + '.partial')
    record_path = safe_path(root, 'asset-records/' + asset['id'] + '.json')
    for path in (partial, record_path):
        if path.exists() or path.is_symlink():
            raise PrepFailure('Existing partial/attempt: preserve it and do not retry')
    final.parent.mkdir(parents=True, exist_ok=True); record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {'asset': asset, 'status': 'starting', 'started_utc': utc(), 'payload_bytes_observed': 0,
              'http_payload_attempts': 0, 'config_sha256': ledger['config_sha256'],
              'source_sha256': ledger['source_sha256']}
    save(record_path, record, exclusive=True)
    start = time.monotonic(); h = hashlib.sha256()
    try:
        reserved = sum(ledger['reservations'].values()) + asset['bytes'] + 1
        if ledger['base_ceiling_bytes'] + reserved > ledger['limit_bytes']:
            raise PrepFailure('Cumulative reservation would exceed budget')
        if asset['id'] in ledger['reservations']:
            raise PrepFailure('Prior reservation means an earlier attempt exists')
        ledger['reservations'][asset['id']] = asset['bytes'] + 1
        ledger['conservative_cumulative_ceiling_bytes'] = ledger['base_ceiling_bytes'] + reserved
        save(ledger_path, ledger)
        if final.exists():
            if not final.is_file() or final.stat().st_size != asset['bytes'] or sha(final) != asset['sha256']:
                raise PrepFailure('Existing final asset mismatch; nothing overwritten')
            record['status'] = 'reused_verified'; record['sha256'] = asset['sha256']
        else:
            record['http_payload_attempts'] = 1; save(record_path, record)
            with partial.open('xb') as stream, deadline(transfer['per_asset_timeout_seconds']):
                with open_url(asset['url'], timeout=transfer['read_timeout_seconds']) as response:
                    if response.status != 200:
                        raise PrepFailure('Unexpected HTTP status')
                    content_length = response.headers.get('Content-Length')
                    if content_length is not None and int(content_length) != asset['bytes']:
                        raise PrepFailure('HTTP length differs from frozen asset')
                    record['http_status'] = response.status
                    record['response_content_length'] = content_length
                    record['final_https_host'] = urllib.parse.urlsplit(response.geturl()).hostname
                    while True:
                        remaining = asset['bytes'] + 1 - record['payload_bytes_observed']
                        block = response.read(min(transfer['chunk_bytes'], remaining))
                        if not block:
                            break
                        record['payload_bytes_observed'] += len(block)
                        ledger['payload_bytes_observed'] += len(block)
                        h.update(block); stream.write(block); stream.flush()
                        save(ledger_path, ledger); save(record_path, record)
                        if record['payload_bytes_observed'] > asset['bytes']:
                            raise PrepFailure('Body exceeded frozen length; at most one sentinel byte read')
                stream.flush(); os.fsync(stream.fileno())
                if record['payload_bytes_observed'] != asset['bytes'] or h.hexdigest() != asset['sha256']:
                    raise PrepFailure('Truncated body or SHA256 mismatch; partial retained')
                if partial.stat().st_size != asset['bytes'] or sha(partial) != asset['sha256']:
                    raise PrepFailure('Stored bytes failed verification')
                os.link(partial, final)  # Atomic publication refuses an existing destination.
                partial.unlink()
            record['status'] = 'downloaded_verified'; record['sha256'] = asset['sha256']
    except BaseException as error:
        record['status'] = 'failed'; record['error'] = error_record(error)
        raise
    finally:
        record['finished_utc'] = utc(); record['elapsed_seconds'] = time.monotonic() - start
        record['partial_bytes_stored'] = partial.stat().st_size if partial.exists() else 0
        record['observed_payload_sha256'] = h.hexdigest()
        save(record_path, record)
    return record


def verify_source_tree(source_root, files):
    """Reject every unbound entry/initializer and ambiguous namespace before import."""
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir() or len(files) != 5:
        raise PrepFailure('Expected the exact five-file independent source tree')
    dirs = {str(parent) for name in files for parent in Path(name).parents if str(parent) != '.'}
    observed = set()
    for path in source_root.rglob('*'):
        name = path.relative_to(source_root).as_posix()
        if path.is_symlink():
            raise PrepFailure('Source symlink is not a frozen input')
        if path.is_dir() and name in dirs:
            continue
        if not path.is_file() or name not in files or sha(path) != files[name]:
            raise PrepFailure('Unexpected or changed source entry: ' + name)
        observed.add(name)
    if observed != set(files):
        raise PrepFailure('Pinned source file missing')
    if any(name == 'tapnet' or name.startswith('tapnet.') for name in sys.modules):
        raise PrepFailure('TAPNet must not already be imported')
    spec = PathFinder.find_spec('tapnet', sys.path)
    wanted = (source_root / 'tapnet').resolve()
    if (spec is None or spec.loader is not None or spec.origin is not None
            or [Path(p).resolve() for p in spec.submodule_search_locations or []] != [wanted]):
        raise PrepFailure('Only the frozen TAPNet namespace may resolve')


def file_manifest(overlay):
    if overlay.is_symlink() or not overlay.is_dir():
        raise PrepFailure("Overlay root must be a real directory")
    files = {}
    for path in sorted(overlay.rglob('*')):
        if path.is_symlink():
            raise PrepFailure('Symlink in overlay')
        if path.is_file():
            files[str(path.relative_to(overlay))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        elif not path.is_dir():
            raise PrepFailure('Non-regular overlay entry')
    if not files or not any(name.endswith('.dist-info/RECORD') for name in files):
        raise PrepFailure('Overlay metadata/RECORD missing')
    return files


def child(command, prefix, env, timeout):
    record = {'command': command, 'started_utc': utc()}; start = time.monotonic(); process = None
    try:
        with prefix.with_suffix('.stdout.log').open('xb') as out, prefix.with_suffix('.stderr.log').open('xb') as err:
            process = subprocess.Popen(command, env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                with deadline(timeout):
                    _, status, usage = os.wait4(process.pid, 0)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(process.pid, 0)
                record['terminated_by_parent'] = True
            process.returncode = os.waitstatus_to_exitcode(status)
            record.update(exit_code=process.returncode, raw_wait_status=status, peak_rss_kib=usage.ru_maxrss)
        record.update(elapsed_seconds=time.monotonic() - start, finished_utc=utc(),
                      stdout_sha256=sha(prefix.with_suffix('.stdout.log')), stderr_sha256=sha(prefix.with_suffix('.stderr.log')))
        save(prefix.with_suffix('.json'), record, exclusive=True)
        if record['exit_code'] != 0 or record.get('terminated_by_parent'):
            raise PrepFailure('Installation/import child failed; see preserved child receipt')
        return record
    finally:
        if process is not None and process.returncode is None:
            process.poll()


PROBE = r'''
import hashlib,importlib,importlib.metadata,importlib.util,json,pathlib,sys
old,overlay,source=map(pathlib.Path,sys.argv[1:4]);pins=json.loads(sys.argv[4]);files=json.loads(sys.argv[5]);result={}
helper_path=pathlib.Path(sys.argv[6])
if hashlib.sha256(helper_path.read_bytes()).hexdigest()!=sys.argv[7]:raise RuntimeError('Preparation source changed before probe')
spec=importlib.util.spec_from_file_location('_evr_preparation_guard',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.verify_source_tree(source,files)
import torch,numpy,torchvision,einops,PIL
for name,module in [('torch',torch),('numpy',numpy),('torchvision',torchvision),('einops',einops),('Pillow',PIL)]:
 version=importlib.metadata.version(name); path=pathlib.Path(module.__file__).resolve()
 expected=old/'runtime-fixed' if name in ('torch','numpy') else overlay
 if version!=pins[name] or str(module.__version__)!=pins[name] or not path.is_relative_to(expected):raise RuntimeError('Import version/path mismatch: '+name)
 result[name]={'version':version,'module_version':str(module.__version__),'path':str(path)}
if torch.version.cuda is not None:raise RuntimeError('Expected CPU-only Torch')
helper.verify_source_tree(source,files)
for name in ['tapnet.tapnext.tapnext_torch','tapnet.tapnext.tapnext_lru_modules','tapnet.tapnext.pscan','tapnet.tapnextpp.votsp2026.model','tapnet.tapnextpp.votsp2026.utils']:
 module=importlib.import_module(name);path=pathlib.Path(module.__file__).resolve()
 if path!=source/(name.replace('.','/')+'.py'):raise RuntimeError('Source import mismatch')
 result[name]={'path':str(path)}
print(json.dumps({'status':'passed','no_model_constructed_or_forward_called':True,'imports':result},indent=2))
'''


def external_timeout(signum, frame):
    raise TimeoutError('External preparation deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'config-sha256', 'old-cache', 'new-cache', 'stage'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--repository', default=os.getcwd())
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, external_timeout)
    if args.stage not in ('dependencies', 'checkpoint'):
        raise PrepFailure('Unknown stage')
    config_path = Path(args.config).resolve()
    if sha(config_path) != args.config_sha256:
        raise PrepFailure('Configuration SHA does not match frozen input')
    cfg = json.loads(config_path.read_text())['preparation']; planned = validate_config(cfg)
    repo = Path(args.repository).resolve(); old = Path(args.old_cache).resolve()
    root_arg = Path(args.new_cache).absolute(); root = root_arg.resolve()
    if root_arg.is_symlink() or root == old or root.is_relative_to(old) or root.parent != old.parent:
        raise PrepFailure('New cache must be an independent non-symlink sibling of old cache')
    historical = safe_path(repo, cfg['budget']['historical_ledger_path'])
    if sha(historical) != cfg['budget']['historical_ledger_sha256'] or json.loads(historical.read_text())['conservative_cumulative_ceiling_bytes'] != cfg['budget']['historical_ceiling_bytes']:
        raise PrepFailure('Historical cumulative ledger differs')
    if sys.platform != 'linux' or platform.python_version() != cfg['runtime']['python']:
        raise PrepFailure('Runtime must match the cloud preflight')
    for rel, digest in cfg['runtime']['old_init_sha256'].items():
        if sha(safe_path(old, rel)) != digest:
            raise PrepFailure('Old runtime identity changed')
    affinity = [0, 1, 2, 3]
    if not set(affinity).issubset(os.sched_getaffinity(0)):
        raise PrepFailure('The frozen CPUs 0,1,2,3 must all be allowed')
    os.sched_setaffinity(0, affinity); resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3,) * 2)
    root.mkdir(exist_ok=True); source_sha = sha(__file__)
    source_files = {a['relative_path'].removeprefix('source/'): a['sha256'] for a in cfg['assets'] if a['kind'] == 'source'}
    isolated_paths = [str(root/'source'), str(root/'overlay')]
    sys.path[:] = isolated_paths + [p for p in sys.path if str(Path(p).resolve()) not in isolated_paths]
    lock_path = safe_path(root, 'preparation.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = safe_path(root, 'markers/' + args.stage + '.started.json')
        receipt_path = safe_path(root, 'preparation-' + args.stage + '.json')
        if marker.exists() or receipt_path.exists():
            raise PrepFailure('This stage has already been attempted; do not retry')
        marker.parent.mkdir(parents=True, exist_ok=True)
        receipt = {'schema_version': 1, 'stage': args.stage, 'status': 'running',
                   'config_sha256': args.config_sha256, 'source_sha256': source_sha,
                   'checkout_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                   'started_utc': utc(), 'assets': [], 'affinity': affinity, 'AS_bytes': 8 * 1024**3}
        save(marker, {**receipt, 'pid': os.getpid()}, exclusive=True)
        receipt['marker_sha256'] = sha(marker); save(receipt_path, receipt, exclusive=True)
        ledger_path = safe_path(root, 'acquisition-ledger.json')
        try:
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                if ledger['config_sha256'] != args.config_sha256 or ledger['source_sha256'] != source_sha:
                    raise PrepFailure('Existing new-cache ledger is bound to other inputs')
            else:
                base = cfg['budget']['historical_ceiling_bytes'] + cfg['budget']['metadata_allowance_bytes']
                ledger = {'config_sha256': args.config_sha256, 'source_sha256': source_sha,
                          'base_ceiling_bytes': base, 'planned_ceiling_bytes': planned,
                          'limit_bytes': cfg['budget']['limit_bytes'], 'reservations': {},
                          'payload_bytes_observed': 0, 'conservative_cumulative_ceiling_bytes': base}
                save(ledger_path, ledger, exclusive=True)
            if args.stage == 'checkpoint':
                deps_path = safe_path(root, 'preparation-dependencies.json'); deps = json.loads(deps_path.read_text())
                if deps['status'] != 'completed' or deps['config_sha256'] != args.config_sha256 or deps['source_sha256'] != source_sha:
                    raise PrepFailure('Successful bound dependency stage is required first')
                manifest = safe_path(root, deps['overlay_manifest']['relative_path'])
                if sha(manifest) != deps['overlay_manifest']['sha256'] or file_manifest(root/'overlay') != json.loads(manifest.read_text())['files']:
                    raise PrepFailure('Installed overlay changed')
                for asset in cfg['assets']:
                    if asset['kind'] != 'checkpoint':
                        p = safe_path(root, asset['relative_path'])
                        if p.stat().st_size != asset['bytes'] or sha(p) != asset['sha256']:
                            raise PrepFailure('A prepared dependency/source changed')
                verify_source_tree(root/'source', source_files)
                receipt['dependency_receipt_sha256'] = sha(deps_path)
                receipt['overlay_manifest'] = deps['overlay_manifest']
            for asset in cfg['assets']:
                if (asset['kind'] == 'checkpoint') != (args.stage == 'checkpoint'):
                    continue
                receipt['assets'].append(acquire(asset, root, ledger, ledger_path, cfg['transfer']))
                save(receipt_path, receipt)
            if args.stage == 'dependencies':
                overlay = safe_path(root, 'overlay')
                if overlay.exists():
                    raise PrepFailure('Existing overlay must not be overwritten')
                temporary = safe_path(root, 'tmp'); temporary.mkdir(exist_ok=False)
                env = {'PATH': os.environ['PATH'], 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                       'TMPDIR': str(temporary), 'PYTHONPATH': ':'.join(map(str, [overlay, old/'runtime-fixed', old/'runtime', root/'source']))}
                for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
                    env[name] = '4'
                wheels = [str(safe_path(root, a['relative_path'])) for a in cfg['assets'] if a['kind'] == 'wheel']
                command = [sys.executable, '-m', 'pip', '--isolated', 'install', '--no-deps', '--no-index', '--no-compile', '--no-cache-dir', '--disable-pip-version-check', '--target', str(overlay), *wheels]
                receipt['installation'] = child(command, root/'install', env, cfg['install_timeout_seconds'])
                before = file_manifest(overlay)
                pins = {**cfg['runtime']['old_packages'], **cfg['runtime']['overlay_packages']}
                verify_source_tree(root/'source', source_files)
                command = [sys.executable, '-c', PROBE, str(old), str(overlay), str(root/'source'), json.dumps(pins), json.dumps(source_files), str(Path(__file__).resolve()), source_sha]
                receipt['import_check'] = child(command, root/'import-check', env, cfg['import_timeout_seconds'])
                verify_source_tree(root/'source', source_files)
                if file_manifest(overlay) != before:
                    raise PrepFailure('Overlay changed during import-only check')
                manifest = {'schema_version': 1, 'config_sha256': args.config_sha256, 'source_sha256': source_sha, 'files': before}
                manifest_path = safe_path(root, 'overlay-manifest.json'); save(manifest_path, manifest, exclusive=True)
                receipt['overlay_manifest'] = {'relative_path': 'overlay-manifest.json', 'sha256': sha(manifest_path)}
            for rel, digest in cfg['runtime']['old_init_sha256'].items():
                if sha(safe_path(old, rel)) != digest:
                    raise PrepFailure('Old runtime import files changed')
            receipt['status'] = 'completed'
        except BaseException as error:
            receipt.update(status='failed', error=error_record(error))
        finally:
            receipt['finished_utc'] = utc()
            if ledger_path.exists():
                snapshot = safe_path(root, 'acquisition-ledger-' + args.stage + '.json')
                save(snapshot, json.loads(ledger_path.read_text()), exclusive=True)
                receipt['ledger_sha256'] = sha(snapshot)
                receipt['ledger_snapshot'] = {'relative_path': snapshot.name, 'sha256': sha(snapshot)}
            save(receipt_path, receipt)
        print(json.dumps({'status': receipt['status'], 'stage': args.stage, 'receipt': str(receipt_path)}, indent=2))
        return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
