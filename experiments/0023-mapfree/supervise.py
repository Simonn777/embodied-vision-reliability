"""Supervise one frozen 0023 operation.

Derived from our 0022 supervisor (SHA256
76c7291004d181150a2d1a0ea5d287fffec3cb41f07645ea5bd1409ef241ace5).
Only operation names, schema, and frozen-helper bindings are changed; process,
resource, environment, cleanup, log and receipt accounting remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 1800.0
RSS_LIMIT_KIB = 8 * 1024 * 1024
SCRIPTS = {'acquisition': 'acquire_operation.py', 'frontend': 'frontend.py',
           'prediction': 'predict.py', 'evaluation': 'evaluate.py'}
HELPERS = {
    'acquisition': ('range_inputs.py', 'propose_plan.py'),
    'frontend': ('range_inputs.py', 'propose_plan.py', 'acquire_operation.py'),
    'prediction': ('cache_inputs.py', 'study.py', 'geometry_ba.py', 'field_ba.py', 'conditional.py', 'covariance.py'),
    'evaluation': (),
}
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def display(value):
    return str(value).replace(str(sys.executable), '$PYTHON').replace(
        str(ROOT), '$REPOSITORY').replace(str(ROOT.parent), '$PRIVATE')


def verify_frozen(commit, paths):
    """Bind owned sources, without requiring unrelated HEAD/worktree equality."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('expected_commit must be a full Git commit SHA')
    resolved = subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'],
                                       cwd=ROOT, text=True).strip()
    if resolved != commit:
        raise ValueError('expected commit does not resolve exactly')
    result = {}
    for path in paths:
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix()
        expected = subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=ROOT)
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError('source differs from reviewed commit: ' + rel)
        result[rel] = hashlib.sha256(actual).hexdigest()
    return result


def snapshot(parent, known_groups=()):
    sample = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss='],
                            capture_output=True, text=True, timeout=3, check=True)
    rows = []
    for line in sample.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError('malformed process sample')
        row = tuple(map(int, fields))
        if row[3] < 0:
            raise ValueError('negative RSS sample')
        rows.append(row)
    descendants = {parent}
    while True:
        new = descendants | {pid for pid, ppid, pgid, _ in rows
                             if ppid in descendants or pgid in known_groups}
        if new == descendants:
            break
        descendants = new
    selected = [row for row in rows if row[0] in descendants]
    groups = {pgid for _, _, pgid, _ in selected if pgid in descendants}
    groups.discard(os.getpgrp())
    return selected, groups


def save(path, record):
    pending = path.with_suffix('.pending')
    with pending.open('w') as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(pending, path)


def require_child_commit(command, expected, required):
    values = []
    for index, part in enumerate(command):
        if part == '--expected-commit':
            if index + 1 == len(command):
                raise ValueError('missing child expected commit value')
            values.append(command[index + 1])
        elif part.startswith('--expected-commit='):
            values.append(part.split('=', 1)[1])
    if (required and not values) or len(values) > 1 or (values and values[0] != expected):
        raise ValueError('child expected commit must match supervisor')


def run(args):
    operation = args.operation
    command = json.loads(args.command_json)
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or '\0' in x for x in command):
        raise ValueError('command-json must be a nonempty string argument list')
    if command[0] == sys.executable:
        command = command[1:]
    if not command:
        raise ValueError('missing operation script')
    script = Path(command[0])
    script = (ROOT / script).resolve() if not script.is_absolute() else script.resolve()
    allowed = ROOT / 'experiments/0023-mapfree' / SCRIPTS[operation]
    if script != allowed.resolve():
        raise ValueError('operation script is not allowlisted')
    require_child_commit(command, args.expected_commit, True)
    frozen_paths = [Path(__file__), script]
    frozen_paths.extend(script.parent / name for name in HELPERS[operation])
    bindings = verify_frozen(args.expected_commit, frozen_paths)
    output = Path(args.output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    command = [sys.executable, str(script), *command[1:]]
    env = dict(os.environ)
    env.update({key: '4' for key in THREAD_KEYS})
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='')
    record = {'schema': 'evr0023.supervision.v1', 'status': 'started', 'operation': operation,
              'expected_commit': args.expected_commit, 'source_sha256': bindings,
              'command': [display(v) for v in command], 'cwd': '$REPOSITORY',
              'python_version': platform.python_version(), 'thread_environment': {k: env[k] for k in THREAD_KEYS},
              'hard_timeout_seconds': TIMEOUT, 'group_rss_limit_kib': RSS_LIMIT_KIB,
              'rss_sampling': 'All observed descendants and retained process groups, approximately every 0.1s; transient peaks may be missed'}
    receipt = output / 'receipt.json'
    save(receipt, record)
    began = time.perf_counter()
    child = None
    code = None
    termination = None
    peak = 0
    groups = set()
    remaining = []
    cleanup_errors = []
    interrupted = None
    with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout,
                                     stderr=stderr, start_new_session=True)
            groups.add(child.pid)
            while True:
                rows, fresh_groups = snapshot(child.pid, groups)
                groups.update(fresh_groups)
                peak = max(peak, sum(row[3] for row in rows))
                if peak > RSS_LIMIT_KIB:
                    termination = 'group_rss_limit'; break
                if time.perf_counter() - began >= TIMEOUT:
                    termination = 'timeout'; break
                if child.poll() is not None:
                    if any(row[0] != child.pid for row in rows):
                        termination = 'descendants_remain_after_parent_exit'
                    break
                time.sleep(.1)
        except BaseException as error:
            termination = type(error).__name__ + ': ' + display(error)
            if not isinstance(error, Exception):
                interrupted = error
        finally:
            if child is not None:
                if child.poll() is None or termination is not None or child.returncode != 0:
                    try:
                        os.killpg(child.pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        cleanup_errors.append('stop_group: ' + type(error).__name__)
                    try:
                        rows, fresh_groups = snapshot(child.pid, groups)
                        groups.update(fresh_groups)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                    for group in groups:
                        try:
                            os.killpg(group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            cleanup_errors.append('kill_group: ' + type(error).__name__)
                try:
                    code = child.wait(timeout=5)
                except (OSError, subprocess.SubprocessError) as error:
                    cleanup_errors.append(type(error).__name__)
                    termination = termination or 'child_wait_failed'
                try:
                    rows, _ = snapshot(child.pid, groups)
                    remaining = [row[0] for row in rows if row[0] != child.pid]
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    termination = termination or 'cleanup_sample_error: ' + type(error).__name__
            stdout.flush(); stderr.flush()
    elapsed = time.perf_counter() - began
    if elapsed >= TIMEOUT and termination is None:
        termination = 'timeout_at_exit'
    record.update(status='completed' if code == 0 and termination is None and not remaining and not cleanup_errors else 'failed',
                  exit_code=code, termination=termination, wall_seconds=elapsed,
                  peak_group_rss_kib=peak, remaining_descendant_pids=remaining,
                  cleanup_errors=cleanup_errors,
                  stdout_sha256=sha256(output / 'stdout.log'), stderr_sha256=sha256(output / 'stderr.log'))
    save(receipt, record)
    if interrupted is not None:
        raise interrupted
    return 0 if record['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=tuple(SCRIPTS), required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--command-json', required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
