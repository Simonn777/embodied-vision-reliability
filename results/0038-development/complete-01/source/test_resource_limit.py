"""Manufactured specifications, clocks and processes; never launch a child."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name):
    definition = importlib.util.spec_from_file_location('d38_test_' + name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(value)
    return value


limit = load('resource_limit')
supervisor = load('supervise_authorized')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def fixture(root):
    auth = dict(path=str(HERE / 'resource-authorization-01.json'), sha256=limit.AUTHORIZATION_SHA)
    worker = dict(repo=str(ROOT), native_root=str(root / 'native'), output=str(root / 'complete-01'),
        operation_id=limit.OPERATION, operation_wall_seconds=3600, resource_exception=auth,
        launch_admitted=True, preparation_only=False)
    wp = root / 'worker.json'
    save(wp, worker)
    outer = dict(repo=str(ROOT), commit='manufactured', timeout=3600, operation_id=limit.OPERATION,
        resource_exception=auth, worker_spec_sha256=sha(wp), operation_output=str(root / 'complete-supervision-01'),
        result_file=str(root / 'complete-01/receipt.json'),
        command=[str(root / 'native/installation/venv/bin/python'), '-B', '-S', str(HERE / 'run_complete.py'), str(wp)],
        environment={k: '4' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    return worker, outer


class ResourceTests(unittest.TestCase):
    def test_default_and_exact_authorized_limits(self):
        self.assertIsNone(limit.admit(dict(operation_wall_seconds=1800)))
        with tempfile.TemporaryDirectory() as temp:
            worker, outer = fixture(Path(temp).resolve())
            self.assertEqual(limit.admit(worker), worker['resource_exception'])
            self.assertEqual(limit.admit(outer, supervision=True), worker['resource_exception'])

    def test_over_limit_wrong_scope_or_unbound_exception_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            worker, outer = fixture(Path(temp).resolve())
            for changes in (dict(operation_wall_seconds=3601), dict(operation_wall_seconds=float('nan')),
                            dict(operation_id='another-run'), dict(resource_exception={}),
                            dict(output=str(Path(temp).resolve() / 'complete-02'))):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    limit.admit({**worker, **changes})
            outer['worker_spec_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'worker spec changed'):
                limit.admit(outer, supervision=True)

    def test_old_lifecycle_and_monitor_are_unchanged(self):
        old = ROOT / 'experiments/0029-memory-information/supervise_exit_checked.py'
        self.assertEqual(sha(old), '8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8')
        self.assertEqual((HERE / 'supervise_authorized.py').read_text().split('    def sample(label):', 1)[1],
                         old.read_text().split('    def sample(label):', 1)[1])
        self.assertEqual(sha(ROOT / supervisor.MONITOR),
                         '4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd')

    def controlled(self, case):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            worker, outer = fixture(root)
            path = root / 'outer.json'
            save(path, outer)
            clock = SimpleNamespace(now=0.)
            events, samples = [], []
            class Child:
                pid = 91001
                returncode = 0 if case in ('success', 'leak') else None
                def poll(self):
                    return self.returncode
                def wait(self, timeout):
                    return self.returncode
            child = Child()
            def send(pgid, sig):
                events.append((pgid, sig))
                if sig == signal.SIGKILL:
                    child.returncode = -9 if child.returncode is None else child.returncode
            def snapshot(pid, groups):
                if case == 'leak' and not any(sig == signal.SIGKILL for _, sig in events):
                    rows = [(91002, 91001, 91001, 64)]
                elif child.returncode is None:
                    rows = [(91001, 90000, 91001, supervisor.RSS_LIMIT + 1 if case == 'rss' else 128)]
                else:
                    rows = []
                samples.append(rows)
                return rows, {91001} if rows else set()
            monitor = SimpleNamespace(THREAD_KEYS=tuple(outer['environment']), display=str,
                sha256=sha, save=save, snapshot=snapshot)
            def sleep(_):
                clock.now = 3600.
            with patch.object(supervisor.subprocess, 'Popen', return_value=child) as popen, \
                 patch.object(supervisor.os, 'getpgid', return_value=91001), \
                 patch.object(supervisor.os, 'getpgrp', return_value=90000), \
                 patch.object(supervisor.os, 'killpg', side_effect=send), \
                 patch.object(supervisor.time, 'perf_counter', side_effect=lambda: clock.now), \
                 patch.object(supervisor.time, 'sleep', side_effect=sleep):
                record = supervisor.run_spec(path, monitor=monitor)
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(record['hard_timeout_seconds'], 3600)
            self.assertEqual(record['group_rss_limit_kib'], 8388608)
            self.assertEqual(record['resource_exception'], worker['resource_exception'])
            self.assertEqual(record['remaining_descendant_pids'], [])
            return record, events

    def test_complete_success_requires_clean_post_exit(self):
        record, events = self.controlled('success')
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['post_exit_snapshot']['rows'], [])
        self.assertEqual(events, [])

    def test_exact_3600_timeout_is_failure_and_kills(self):
        record, events = self.controlled('timeout')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['termination'], 'timeout')
        self.assertEqual(record['wall_seconds'], 3600.)
        self.assertIn((91001, signal.SIGKILL), events)

    def test_rss_limit_is_unchanged_and_kills(self):
        record, events = self.controlled('rss')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['termination'], 'group_rss_limit')
        self.assertIn((91001, signal.SIGKILL), events)

    def test_live_post_exit_descendant_is_not_clean_success(self):
        record, events = self.controlled('leak')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['termination'], 'descendants_remain_after_fresh_exit_sample')
        self.assertIn((91001, signal.SIGKILL), events)


if __name__ == '__main__':
    unittest.main()
