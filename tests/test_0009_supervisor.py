"""Supervisor safety tests: fake wait/launch faults plus Linux toy processes only.

No collector, geometry generator, model, data, network, or scientific result.
Linux integration tests are explicitly skipped on macOS.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/supervise_0009.py'
spec = importlib.util.spec_from_file_location('evr_0009_supervisor_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
USAGE = SimpleNamespace(ru_utime=0.01, ru_stime=0.02, ru_maxrss=4096)


@contextlib.contextmanager
def fake_linux():
    with mock.patch.object(m.sys, 'platform', 'linux'), mock.patch.object(
            m.os, 'sched_getaffinity', return_value={2, 3, 4, 5, 6}, create=True):
        yield


def assert_logs(test, folder, receipt):
    test.assertEqual(json.loads((folder / 'execution.json').read_text()), receipt)
    for name in ('stdout.log', 'stderr.log'):
        data = (folder / name).read_bytes()
        test.assertEqual(receipt['logs'][name], {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})


class PortableSupervisorTests(unittest.TestCase):
    def test_collection_binding_preserves_receipt_and_binds_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            collection, outer = root / 'collection', root / 'outer'
            collection.mkdir(); outer.mkdir()
            raw = b'{"supervisor_exit_code":0,"child_exit_code":0}\n'
            (outer / 'execution.json').write_bytes(raw)
            lock = b'{"synthetic_lock":true}\n'
            (collection / 'selection-lock.json').write_bytes(lock)
            r = m.bind_collection(collection, outer)
            self.assertEqual(r['effective_exit_code'], 0)
            self.assertEqual(r['selection_lock'], {'bytes': len(lock), 'sha256': hashlib.sha256(lock).hexdigest()})
            self.assertEqual(r['execution_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual((outer / 'execution.json').read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                m.bind_collection(collection, outer)

    def test_missing_or_symlink_lock_never_accepts_child_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name, child, supervisor, symlink in (
                    ('absent-success', 0, 0, False), ('absent-failure', 7, 7, False),
                    ('absent-timeout', -9, 124, False), ('symlink', 0, 0, True)):
                with self.subTest(name=name):
                    collection, outer = root / (name+'-collection'), root / (name+'-outer')
                    collection.mkdir(); outer.mkdir()
                    raw = json.dumps({'supervisor_exit_code':supervisor,'child_exit_code':child}).encode()
                    (outer / 'execution.json').write_bytes(raw)
                    if symlink:
                        (root / 'real-lock').write_text('{}')
                        (collection / 'selection-lock.json').symlink_to(root / 'real-lock')
                    r = m.bind_collection(collection, outer)
                    self.assertIsNone(r['selection_lock'])
                    self.assertEqual(r['effective_exit_code'], 125 if child == 0 else supervisor)
                    self.assertEqual((outer / 'execution.json').read_bytes(), raw)

    def test_nonlinux_refused_without_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'out'
            with mock.patch.object(m.sys, 'platform', 'darwin'):
                with self.assertRaisesRegex(RuntimeError, 'requires Linux'):
                    m.supervise(['unused'], path)
            self.assertFalse(path.exists())

    def test_invalid_limits_and_affinity_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp, fake_linux():
            path = Path(tmp) / 'out'
            for deadline in (0, -1, 901, float('inf'), float('nan'), True, '1'):
                with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                    m.supervise(['unused'], path, seconds=deadline)
            for cpus in ([], [2, 2], [True], [2.0], [0], [2, 3, 4, 5, 6]):
                with self.subTest(cpus=cpus), self.assertRaises(ValueError):
                    m.supervise(['unused'], path, cpu_ids=cpus)
            for argv in ([], 'shell command', [1]):
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    m.supervise(argv, path)
            self.assertFalse(path.exists())

    def test_launch_error_retains_evidence_without_fake_wait_usage(self):
        mask_before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        with tempfile.TemporaryDirectory() as tmp, fake_linux(), mock.patch.object(
                m.subprocess, 'Popen', side_effect=FileNotFoundError('private text')):
            folder = Path(tmp) / 'out'
            r = m.supervise(['unavailable-executable'], folder)
            self.assertEqual(r['failure_type'], 'FileNotFoundError')
            self.assertEqual(r['failure_stage'], 'launch')
            self.assertEqual(r['supervisor_exit_code'], 125)
            self.assertIsNone(r['child_exit_code'])
            self.assertIsNone(r['raw_wait_status'])
            self.assertIsNone(r['max_rss_kib'])
            self.assertFalse(r['supervisor_interrupted'])
            self.assertNotIn('private text', (folder / 'execution.json').read_text())
            self.assertTrue((folder / 'command.json').is_file())
            assert_logs(self, folder, r)
        self.assertEqual(mask_before, signal.pthread_sigmask(signal.SIG_BLOCK, set()))

    def test_wait_fault_is_not_mislabeled_interrupt_or_success(self):
        child = SimpleNamespace(pid=123456, returncode=None)
        with tempfile.TemporaryDirectory() as tmp, fake_linux(), mock.patch.object(
                m.subprocess, 'Popen', return_value=child), mock.patch.object(
                m.os, 'wait4', side_effect=[RuntimeError('private detail'), (child.pid, 9, USAGE)]), mock.patch.object(
                m.os, 'killpg') as kill:
            folder = Path(tmp) / 'out'
            r = m.supervise(['unused'], folder)
            self.assertEqual(r['failure_type'], 'RuntimeError')
            self.assertEqual(r['failure_stage'], 'supervision')
            self.assertFalse(r['supervisor_interrupted'])
            self.assertEqual(r['supervisor_exit_code'], 125)
            self.assertEqual(r['child_exit_code'], -9)
            kill.assert_called_once_with(child.pid, signal.SIGKILL)
            assert_logs(self, folder, r)

    def test_sigterm_cleanup_and_handlers_restored(self):
        child = SimpleNamespace(pid=123456, returncode=None)
        prior = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT)}
        with tempfile.TemporaryDirectory() as tmp, fake_linux(), mock.patch.object(
                m.subprocess, 'Popen', return_value=child), mock.patch.object(
                m.os, 'wait4', side_effect=[m._ReceivedSignal(signal.SIGTERM), (child.pid, 9, USAGE)]), mock.patch.object(
                m.os, 'killpg') as kill:
            r = m.supervise(['unused'], Path(tmp) / 'out')
            self.assertTrue(r['supervisor_interrupted'])
            self.assertEqual(r['interrupt_signal'], signal.SIGTERM)
            self.assertEqual(r['supervisor_exit_code'], 143)
            self.assertEqual(r['child_exit_code'], -9)
            kill.assert_called_once_with(child.pid, signal.SIGKILL)
        self.assertEqual(prior, {n: signal.getsignal(n) for n in prior})

    def test_baseexception_is_preserved_after_cleanup_and_receipt(self):
        child = SimpleNamespace(pid=123456, returncode=None)
        with tempfile.TemporaryDirectory() as tmp, fake_linux(), mock.patch.object(
                m.subprocess, 'Popen', return_value=child), mock.patch.object(
                m.os, 'wait4', side_effect=[SystemExit(7), (child.pid, 9, USAGE)]), mock.patch.object(
                m.os, 'killpg') as kill:
            folder = Path(tmp) / 'out'
            with self.assertRaises(SystemExit) as caught:
                m.supervise(['unused'], folder)
            self.assertEqual(caught.exception.code, 7)
            r = json.loads((folder / 'execution.json').read_text())
            self.assertEqual(r['failure_type'], 'SystemExit')
            self.assertIsNone(r['supervisor_exit_code'])
            self.assertEqual(r['child_exit_code'], -9)
            kill.assert_called_once_with(child.pid, signal.SIGKILL)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux-only subprocess/resource integration; not run on macOS')
class LinuxSupervisorIntegration(unittest.TestCase):
    def run_toy(self, source, *, seconds=3.0):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name) / 'out'
        r = m.supervise([sys.executable, '-c', source], folder, seconds=seconds)
        assert_logs(self, folder, r)
        return folder, r

    def test_exit_zero_and_stream_hashes(self):
        folder, r = self.run_toy("import sys;print('stdout');print('stderr',file=sys.stderr)")
        self.assertEqual(r['raw_wait_status'], 0)
        self.assertEqual(r['child_exit_code'], 0)
        self.assertEqual(r['supervisor_exit_code'], 0)
        self.assertFalse(r['timed_out'])
        self.assertIsNone(r['failure_type'])
        self.assertEqual((folder / 'stdout.log').read_bytes(), b'stdout\n')
        self.assertEqual((folder / 'stderr.log').read_bytes(), b'stderr\n')
        self.assertGreater(r['max_rss_kib'], 0)

    def test_exit_seven_is_not_success(self):
        _, r = self.run_toy('raise SystemExit(7)')
        self.assertEqual(r['child_exit_code'], 7)
        self.assertEqual(r['supervisor_exit_code'], 7)
        self.assertEqual(r['raw_wait_status'], 7 << 8)
        self.assertFalse(r['timed_out'])

    def test_resource_and_affinity_are_actual_child_limits(self):
        source = """import json,os,resource
names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
import signal
print(json.dumps({'affinity':sorted(os.sched_getaffinity(0)),
'as':resource.getrlimit(resource.RLIMIT_AS),'core':resource.getrlimit(resource.RLIMIT_CORE),
'pid':os.getpid(),'pgrp':os.getpgrp(),'threads':{k:os.environ[k] for k in names},
'signal_mask':sorted(signal.pthread_sigmask(signal.SIG_BLOCK,set()))}))
"""
        folder, r = self.run_toy(source)
        self.assertEqual(r['supervisor_exit_code'], 0)
        actual = json.loads((folder / 'stdout.log').read_text())
        self.assertEqual(actual['affinity'], sorted(os.sched_getaffinity(0))[:4])
        self.assertEqual(actual['affinity'], r['cpu_affinity'])
        self.assertEqual(actual['as'], [8 * 1024**3] * 2)
        self.assertEqual(actual['core'], [0, 0])
        self.assertEqual(actual['pid'], actual['pgrp'])
        self.assertTrue(all(v == '4' for v in actual['threads'].values()))
        self.assertEqual(actual['signal_mask'], sorted(signal.pthread_sigmask(signal.SIG_BLOCK, set())))

    def test_hard_timeout_kills_sleeping_process_group(self):
        source = """import subprocess,sys,time,os,json
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
print(json.dumps({'pid':os.getpid(),'grandchild':p.pid}),flush=True)
time.sleep(30)
"""
        folder, r = self.run_toy(source, seconds=0.5)
        self.assertTrue(r['timed_out'])
        self.assertEqual(r['supervisor_exit_code'], 124)
        self.assertEqual(r['child_exit_code'], -signal.SIGKILL)
        self.assertGreaterEqual(r['wall_seconds'], 0.5)
        self.assertLess(r['wall_seconds'], 3.0)
        pids = json.loads((folder / 'stdout.log').read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pids['pid'], 0)
        # Grandchild can await init reaping as a zombie; it must not be running.
        for _ in range(50):
            path = Path('/proc') / str(pids['grandchild']) / 'stat'
            if not path.exists() or path.read_text().split()[2] == 'Z':
                break
            time.sleep(0.01)
        else:
            self.fail('grandchild remained alive after process-group timeout')

    def test_existing_output_is_never_overwritten(self):
        folder, r = self.run_toy("print('first')")
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with self.assertRaises(FileExistsError):
            m.supervise([sys.executable, '-c', "print('second')"], folder, seconds=1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_actual_missing_executable_preserves_launch_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'out'
            r = m.supervise([str(Path(tmp) / 'missing')], folder, seconds=1)
            self.assertEqual(r['supervisor_exit_code'], 125)
            self.assertEqual(r['failure_stage'], 'launch')
            self.assertIsNone(r['child_exit_code'])
            assert_logs(self, folder, r)

    def test_actual_sigterm_reaps_child_and_returns_143(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'out'
            toy = "import os,time;print(os.getpid(),flush=True);time.sleep(30)"
            launcher = ("import importlib.util,json,sys;"
                        "s=importlib.util.spec_from_file_location('sv',sys.argv[1]);"
                        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                        "r=m.supervise([sys.executable,'-c',sys.argv[3]],sys.argv[2]);"
                        "raise SystemExit(r['supervisor_exit_code'])")
            proc = subprocess.Popen([sys.executable, '-c', launcher, str(SCRIPT), str(folder), toy],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                until = time.monotonic() + 3
                log = folder / 'stdout.log'
                while time.monotonic() < until:
                    if log.is_file() and log.stat().st_size:
                        break
                    time.sleep(0.01)
                else:
                    self.fail('toy child did not signal startup')
                child_pid = int(log.read_text())
                proc.send_signal(signal.SIGTERM)
                _, err = proc.communicate(timeout=3)
                self.assertEqual(proc.returncode, 143, err.decode())
                r = json.loads((folder / 'execution.json').read_text())
                self.assertEqual(r['interrupt_signal'], signal.SIGTERM)
                self.assertEqual(r['supervisor_exit_code'], 143)
                self.assertEqual(r['child_exit_code'], -signal.SIGKILL)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()


if __name__ == '__main__':
    unittest.main()
