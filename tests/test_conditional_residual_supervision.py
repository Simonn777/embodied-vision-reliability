"""Artificial freeze/re-entry/interface checks only; no science or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_conditional_residual_stage0.py'
spec=importlib.util.spec_from_file_location('evr_stage0_control_fixture',SCRIPT)
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
SOURCE='a'*40
FREEZE='b'*40


class SupervisionTests(unittest.TestCase):
    def make_freeze(self,root):
        files={}
        for name in control.FROZEN_PATHS:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name+'\n')
            files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        freeze={'source_commit':SOURCE,'files':files}
        fp=root/'protocols/0006-stage0-freeze.json';fp.write_text(json.dumps(freeze))
        return fp,freeze

    def git(self,root):
        def output(command,cwd):
            ref,name=command[-1].split(':',1)
            self.assertIn(ref,(SOURCE,FREEZE));self.assertEqual(cwd,root)
            return (root/name).read_bytes()
        return output

    def test_complete_commit_bound_freeze_and_data_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fp,f=self.make_freeze(root)
            with mock.patch.object(control,'ROOT',root),mock.patch.object(control.subprocess,'check_output',side_effect=self.git(root)):
                self.assertEqual(control.verify_freeze(FREEZE),f)
                (root/control.FROZEN_PATHS[0]).write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'Frozen file changed'):control.verify_freeze(FREEZE)

    def test_shrunken_receipt_fails_even_when_committed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fp,f=self.make_freeze(root);f['files'].pop(control.FROZEN_PATHS[0]);fp.write_text(json.dumps(f))
            with mock.patch.object(control,'ROOT',root),mock.patch.object(control.subprocess,'check_output',side_effect=self.git(root)):
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):control.verify_freeze(FREEZE)

    def test_changed_receipt_and_non_sha_ref_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fp,f=self.make_freeze(root)
            with mock.patch.object(control,'ROOT',root),mock.patch.object(control.subprocess,'check_output',return_value=b'other bytes'):
                with self.assertRaisesRegex(RuntimeError,'Canonical'):control.verify_freeze(FREEZE)
                with self.assertRaisesRegex(RuntimeError,'explicit'):control.verify_freeze('main')

    def test_existing_case_rejected_before_import_or_training(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(control,'load_frozen_module',side_effect=AssertionError('trainer must not load')) as loader:
                with self.assertRaises(FileExistsError):control.case(SimpleNamespace(output=Path(td)),{}, {})
                loader.assert_not_called()

    def test_incomplete_case_marker_prevents_child(self):
        with tempfile.TemporaryDirectory() as td:
            cache=Path(td);folder=cache/'cases';folder.mkdir()
            (folder/'d1-white-0-blindspot.started.json').write_text('{}')
            args=SimpleNamespace(cache=cache,image_id='d1')
            cfg={'image_ids':['d1'],'conditions':['white'],'seeds':[0],'methods':['blindspot']}
            with mock.patch.object(control.subprocess,'Popen',side_effect=AssertionError('no child')):
                with self.assertRaisesRegex(RuntimeError,'Incomplete case'):control.group(args,cfg,{})

    def test_budget_exhaustion_does_not_start_child(self):
        with tempfile.TemporaryDirectory() as td:
            args=SimpleNamespace(cache=Path(td),image_id='d1')
            cfg={'image_ids':['d1'],'conditions':['white'],'seeds':[0],'methods':['blindspot'],'group_timeout_seconds':1,'case_timeout_seconds':120}
            with mock.patch.object(control.subprocess,'Popen',side_effect=AssertionError('no child')):
                self.assertEqual(control.group(args,cfg,{})['status'],'partial_time_budget')

    def test_explicit_scientific_module_ignores_search_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fp,f=self.make_freeze(root);relative='scripts/conditional_residual.py'
            (root/relative).write_text('FIXTURE_VALUE = 37\n');f['files'][relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest();fp.write_text(json.dumps(f))
            with mock.patch.object(control,'ROOT',root):
                module=control.load_frozen_module('_evr_fixture',relative)
                self.assertEqual(module.FIXTURE_VALUE,37)
                (root/relative).write_text('raise AssertionError("changed must never execute")\n')
                with self.assertRaisesRegex(RuntimeError,'Unbound'):control.load_frozen_module('_evr_fixture2',relative)


if __name__=='__main__':unittest.main()
