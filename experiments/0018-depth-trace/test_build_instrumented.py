"""Mocked build supervision only: never extract, compile, or call an estimator."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import build_instrumented as build


class BuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def read_receipt(self):
        receipt = json.loads((self.root/'fake.receipt.json').read_text())
        self.assertEqual(receipt['command_sha256'], build.sha(self.root/'fake.command.json'))
        self.assertEqual(receipt['log_sha256'], build.sha(self.root/'fake.log'))
        return receipt

    def test_launch_failure_preserves_receipt_without_any_process(self):
        with mock.patch.object(build.subprocess, 'Popen', side_effect=OSError('artificial launch failure')), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): build.stage('fake', ['not-executed'], self.root, {}, time.monotonic()+100)
        receipt = self.read_receipt()
        self.assertIsNone(receipt['exit_code'])
        self.assertIn('OSError', receipt['termination'])

    def test_kill_race_still_waits_and_preserves_timeout(self):
        child = mock.Mock(pid=123, poll=mock.Mock(return_value=None), wait=mock.Mock(return_value=0))
        # Start before the deadline; cross it after observing an artificial process tree.
        with mock.patch.object(build.time, 'monotonic', side_effect=[0., 0., 2., 2., 2.]), \
             mock.patch.object(build.subprocess, 'Popen', return_value=child), \
             mock.patch.object(build.subprocess, 'run', return_value=mock.Mock(stdout='123 1 1\n')), \
             mock.patch.object(build.os, 'killpg', side_effect=ProcessLookupError('already exited')) as kill, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): build.stage('fake', ['not-executed'], self.root, {}, 1.)
        kill.assert_called_once(); child.wait.assert_called_once_with(timeout=5)
        self.assertIn('timeout', self.read_receipt()['termination'])

    def test_watchdog_timeout_kills_group_and_waits(self):
        child = mock.Mock(pid=123, poll=mock.Mock(return_value=None), wait=mock.Mock(return_value=-9))
        with mock.patch.object(build.subprocess, 'Popen', return_value=child), \
             mock.patch.object(build.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ps', 3)), \
             mock.patch.object(build.os, 'killpg') as kill, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): build.stage('fake', ['not-executed'], self.root, {}, time.monotonic()+100)
        kill.assert_called_once(); child.wait.assert_called_once_with(timeout=5)
        self.assertIn('TimeoutExpired', self.read_receipt()['termination'])

    def test_expired_deadline_prevents_launch(self):
        with mock.patch.object(build.subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): build.stage('fake', ['not-executed'], self.root, {}, time.monotonic()-1)
        launch.assert_not_called()
        self.assertIn('TimeoutError', self.read_receipt()['termination'])

    def test_artificial_dependency_inventory_detects_mutation(self):
        for directory in ('inputs/cmake/data', 'inputs/pybind11/include', 'inputs/pybind11/share', 'eigen-install'):
            folder = self.root/directory; folder.mkdir(parents=True)
            (folder/'artificial.txt').write_text('initial')
        before = build.dependencies(self.root)
        (self.root/'inputs/pybind11/include/artifical-new.h').write_text('new')
        self.assertNotEqual(before, build.dependencies(self.root))
        (self.root/'inputs/pybind11/include/artifical-new.h').unlink()
        (self.root/'eigen-install/artificial.txt').write_text('changed')
        self.assertNotEqual(before, build.dependencies(self.root))

    def test_dependency_inventory_rejects_missing_or_empty_roots(self):
        with self.assertRaisesRegex(ValueError, 'Missing dependency directory'):
            build.dependencies(self.root)
        for directory in ('inputs/cmake/data', 'inputs/pybind11/include', 'inputs/pybind11/share', 'eigen-install'):
            folder = self.root/directory; folder.mkdir(parents=True)
            (folder/'artificial.txt').write_text('initial')
        (self.root/'inputs/pybind11/share/artificial.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Empty dependency directory'):
            build.dependencies(self.root)

    def test_effective_flag_comparison_allows_only_expected_path_relocations(self):
        old, output = self.root/'old', self.root/'new'
        source = output/'inputs'/build.SOURCE
        for name in ('PoseLib/CMakeFiles/PoseLib.dir/flags.make', 'pybind/CMakeFiles/pyposelib.dir/flags.make'):
            p = old/'pose-build'/name; q = output/'pose-build'/name
            p.parent.mkdir(parents=True); q.parent.mkdir(parents=True)
            original = 'CXX_FLAGS=-O3 -DNDEBUG\nSOURCE='+str(old/'inputs'/build.SOURCE)+'\nBUILD='+str(old/'pose-build')
            p.write_text(original)
            q.write_text(original.replace(str(old/'inputs'/build.SOURCE), str(source)).replace(str(old/'pose-build'), str(output/'pose-build')))
        build.compare_flags(old, output, source)
        target = output/'pose-build/PoseLib/CMakeFiles/PoseLib.dir/flags.make'
        target.write_text(target.read_text().replace('-O3', '-ffast-math -O3'))
        with self.assertRaises(ValueError): build.compare_flags(old, output, source)


if __name__ == '__main__': unittest.main()
