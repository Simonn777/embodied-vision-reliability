"""Infrastructure-only check: no annotation/archive opening or scientific runner main."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

ROOT = Path.cwd()
OUT = ROOT / 'results/0003-cache-readiness-20261018'
# Publication redaction only: the executed snapshot had the runtime-local cache
# location as a literal. See source-publication-note.json; do not claim byte identity.
CACHE = Path(os.environ['POINTODYSSEY_CACHE'])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    tracked = git('ls-files').splitlines()
    baseline = {name: digest(ROOT / name) for name in tracked}
    (OUT / 'tracked-baseline.json').write_text(json.dumps(baseline, indent=2, sort_keys=True) + '\n')
    checks = {
        'expected_cache_directory': CACHE.is_dir(),
        'selected_annotation': (CACHE / 'resource-audit-v1/selected-annotation.npz').is_file(),
        'prior_fortran_marker': (CACHE / 'fortran-v2-started.json').is_file(),
        'prior_rotation_marker': (CACHE / 'rotation-diagnostic-v1-started.json').is_file(),
        'new_operation_marker_visible': (CACHE / 'annotation-consistency-v1-started.json').exists(),
        'tracked_operation_output_exists': (ROOT / 'results/0003-annotation-consistency-v1-execution').exists(),
    }
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            # Never publish arbitrary commands, environment variables or credential values.
            names = [Path(a.decode(errors='replace')).name for a in args]
            relevant = [n for n in names if n.startswith(('audit_0003_', 'run_0003_', 'run_frozen_pilot.py')) and n.endswith('.py')]
            if relevant:
                processes.append({'pid': int(entry.name), 'recognized_research_scripts': relevant})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    command = ['timeout', '900', 'python', '-c',
        "import audit_0003_annotation_consistency_v1 as r; import evr,json; "
        "x=r.verify_published(); print(json.dumps({'published_gate':'PASS',"
        "'mandatory_count':len(x['inputs_sha256']),'prior_receipts':len(r.PRIOR),"
        "'prior_distinct_inputs':38,'input_commit':x['input_commit'],"
        "'receipt_sha256':r.sha(r.RECEIPT),'actual_package_initializer_sha256':r.sha(evr.__file__),"
        "'dataset_or_runner_main_called':False},sort_keys=True))"]
    env = {'PATH': os.environ['PATH'], 'PYTHONPATH': 'src:scripts', 'GIT_NO_LAZY_FETCH': '1'}
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '2'
    affinity = sorted(os.sched_getaffinity(0))[:2]
    assert len(affinity) == 2
    def limits():
        os.sched_setaffinity(0, affinity)
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    record = {
        'status': 'STARTING_INFRASTRUCTURE_ONLY', 'execution_commit': git('rev-parse', 'HEAD'),
        'origin_main': git('rev-parse', 'origin/main'), 'command': command,
        'environment': {**env, 'PATH': 'inherited runtime PATH; value omitted'},
        'affinity': affinity, 'AS_bytes': 6 * 1024**3, 'timeout_seconds': 900,
        'python': platform.python_version(), 'platform': platform.platform(),
        'CPU_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),
        'cache_stat_checks': checks, 'visible_relevant_processes': processes,
        'process_scope': 'this host visible /proc only; neither schedule history nor cross-host idleness proof',
        'marker_absence_interpretation': 'unknown prior execution state when cache directory itself is missing; never recreate or delete markers',
        'script_sha256': digest(Path(__file__)), 'tracked_baseline_count': len(baseline),
        'source_sha256': json.loads(Path('protocols/0003-annotation-consistency-freeze-v1.json').read_text())['inputs_sha256'],
        'data_access': 'stat existence checks only; no archive/NPZ/metadata/RGB or numeric payload opened',
        'new_downloads_bytes': 0, 'model_calls': 0, 'geometry_checks': 0,
        'boundary': 'one parent observes gate child wait4; parent has no further supervisor exit receipt',
    }
    (OUT / 'pre-execution.json').write_text(json.dumps(record, indent=2) + '\n')
    start = time.perf_counter()
    record['started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (OUT / 'gate.stdout.log').open('xb') as stdout, (OUT / 'gate.stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, preexec_fn=limits)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
    record.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-start, observed_exit_code=child.returncode, raw_wait_status=status,
        peak_rss_kib=usage.ru_maxrss, CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
        log_sha256={name: digest(OUT / name) for name in ('gate.stdout.log', 'gate.stderr.log')},
        status='BLOCKED_CACHE_NOT_VISIBLE_NO_SCIENTIFIC_ATTEMPT' if not all(checks[k] for k in ('expected_cache_directory', 'selected_annotation', 'prior_fortran_marker', 'prior_rotation_marker')) else 'REQUIRES_MANUAL_STATE_REVIEW')
    assert all(digest(ROOT / name) == value for name, value in baseline.items())
    record['tracked_baseline_byte_preserved'] = len(baseline)
    (OUT / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('status','observed_exit_code','raw_wait_status','wall_seconds','peak_rss_kib','tracked_baseline_byte_preserved')}, indent=2))
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
