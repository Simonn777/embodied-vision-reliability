"""Synthetic file/process checks; no cached images or scientific gradients."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('gradient_supervisor',Path(__file__).resolve().parents[1]/'scripts/run_antithetic_gradient_screen.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SupervisionTests(unittest.TestCase):
    def test_exact_hashes_and_read_allowlist(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);case={}
            for kind in ('input','weights','receipt'):
                p=root/(kind+'.bin');p.write_bytes(kind.encode())
                case[kind+'_relative']=p.name;case[kind+'_sha256']=m.digest(p)
            (root/'clean-reference.npz').mkdir() # would fail if indiscriminately opened
            self.assertEqual(len(m.check_inputs(root,{'cases':[case]})),3)
            (root/'input.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'identity differs'):
                m.check_inputs(root,{'cases':[case]})

    def test_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'x').write_bytes(b'x');(root/'link').symlink_to(root/'x')
            case={'input_relative':'link','input_sha256':m.digest(root/'x')}
            with self.assertRaisesRegex(RuntimeError,'indirect'):
                m.check_inputs(root,{'cases':[case]})

    def test_exclusive_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'marker.json';m.write(p,{'value':1})
            with self.assertRaises(FileExistsError):m.write(p,{'value':2})
            self.assertEqual(m.read(p),{'value':1})

    def test_actual_exit_and_closed_log_hash(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'command'
            row=m.supervise([sys.executable,'-c','import sys; print("complete",flush=True); sys.exit(7)'],out,5)
            self.assertEqual(row['actual_exit'],7);self.assertFalse(row['hard_timeout'])
            self.assertEqual(row['stdout_sha256'],m.digest(out.with_suffix('.stdout.log')))
            self.assertEqual(out.with_suffix('.stdout.log').read_text(),'complete\n')

    def test_hard_timeout_preserves_partial(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'command'
            row=m.supervise([sys.executable,'-c','import time; print("started",flush=True); time.sleep(30)'],out,.4)
            self.assertTrue(row['hard_timeout']);self.assertEqual(row['actual_exit'],-9)
            self.assertLess(row['wall_seconds'],3)
            self.assertEqual(out.with_suffix('.stdout.log').read_text(),'started\n')

    def test_freeze_requires_full_commit(self):
        with self.assertRaisesRegex(RuntimeError,'full canonical'):
            m.verify_freeze('main')

class EvidenceTests(unittest.TestCase):
    def test_saved_evidence_rejects_tampering_and_failed_complete_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'cases'/'test';out.mkdir(parents=True)
            control=root/'control'/'test';control.parent.mkdir()
            item={'name':'test','index':0,'weight_content_sha256':'weights'}
            frozen={'source_commit':'source','files':{'scripts/antithetic_gradient_core.py':'core'}}
            for name in ('gradients.npy','walls.npy','losses.npy','completed.npy'):
                (out/name).write_bytes(name.encode())
            rec={'completed_measurements':512,'expected_measurements':512,'optimizer_steps':0,
                 'parameter_count':5313,'source_sha256':'core','freeze_sha256':'frozen',
                 'status':'completed','weight_unchanged':True,'weight_before_sha256':'weights',
                 'weight_after_sha256':'weights','case_identity':item,'source_commit':'source',
                 'freeze_commit':'commit','files':{p.name:m.digest(p) for p in out.iterdir()}}
            for suffix in ('.stdout.log','.stderr.log','.command.private.json'):
                control.with_suffix(suffix).write_bytes(suffix.encode())
            def bind():
                (out/'receipt.json').write_text(json.dumps(rec))
                ex={'actual_exit':0,'raw_wait_status':0,'hard_timeout':False,
                    'receipt_sha256':m.digest(out/'receipt.json'),
                    'stdout_sha256':m.digest(control.with_suffix('.stdout.log')),
                    'stderr_sha256':m.digest(control.with_suffix('.stderr.log')),
                    'command_file_sha256':m.digest(control.with_suffix('.command.private.json'))}
                control.with_suffix('.execution.json').write_text(json.dumps(ex))
                return {'case':'test',**ex}
            original_digest=m.digest
            def digest(path):
                return 'frozen' if Path(path)==m.ROOT/m.FREEZE else original_digest(path)
            row=bind()
            with mock.patch.object(m,'digest',side_effect=digest):
                self.assertEqual(m.validate_case_evidence(root,item,frozen,'commit',row),rec)
                control.with_suffix('.stdout.log').write_bytes(b'lost bytes')
                with self.assertRaisesRegex(RuntimeError,'execution, logs'):
                    m.validate_case_evidence(root,item,frozen,'commit',row)
                row=bind();rec['status']='failed';row=bind()
                with self.assertRaisesRegex(RuntimeError,'integrity failed'):
                    m.validate_case_evidence(root,item,frozen,'commit',row)
                rec['status']='completed';rec['weight_after_sha256']='mutated';row=bind()
                with self.assertRaisesRegex(RuntimeError,'integrity failed'):
                    m.validate_case_evidence(root,item,frozen,'commit',row)
                rec['weight_after_sha256']='weights';row=bind()
                (out/'gradients.npy').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'saved gradient'):
                    m.validate_case_evidence(root,item,frozen,'commit',row)

if __name__=='__main__':unittest.main()
