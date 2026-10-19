"""Only fake core / tiny artificial arrays; never import Torch or run StatA."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_admission as a


class Clock:
    def __init__(self, steps): self.now=0.; self.steps=iter(steps)
    def __call__(self): self.now+=next(self.steps); return self.now


class Fake:
    def __init__(self, fail=None, parity=True): self.calls=[]; self.fail=fail; self.parity=parity
    def fit(self,x,p,*,ids,alpha,source_path):
        self.calls.append(('fit',len(x),tuple(ids),alpha,str(source_path)))
        if len(self.calls)==self.fail: raise ValueError('fake_failure')
        return dict(status='ok',reference_bitwise_equal=self.parity,native_solver_executions=2)
    def heldout_readout(self,state,x,*,query_id,source_path):
        self.calls.append(('heldout',query_id,str(source_path)))
        if len(self.calls)==self.fail: raise ValueError('fake_failure')
        return np.array([.8,.2])


class Tests(unittest.TestCase):
    def setUp(self):
        self.x=np.arange(28).reshape(7,4); self.p=np.ones((1,4,2)); self.ids=tuple('abcdefg')
        self.record={'attempts':[]}; self.saved=[]
    def save(self): self.saved.append(copy.deepcopy(self.record))
    def run_fake(self,backend,seconds=(.1,.2,.3)):
        a.measure(backend,self.x,self.p,self.ids,Path('upstream/StatA.py'),self.record,self.save,Clock([0,seconds[0],0,seconds[1],0,seconds[2]]))
    def test_exact_three_calls_order_alpha_and_deleted_query(self):
        b=Fake(); self.run_fake(b)
        self.assertEqual([c[0] for c in b.calls],['fit','heldout','fit','fit'])
        self.assertEqual([(c[1],c[3]) for c in b.calls if c[0]=='fit'],[(6,1),(7,1),(7,4)])
        self.assertEqual(b.calls[0][2],self.ids[1:]); self.assertEqual(b.calls[1][1],self.ids[0])
        self.assertEqual([r['phase'] for r in self.record['attempts']],['T63','T641','T644'])
        self.assertEqual(sum(r['native_solver_executions'] for r in self.record['attempts']),6)
        self.assertTrue(any(s['attempts'][0]['status']=='attempted' for s in self.saved))
    def test_projection_is_not_measured_quality(self):
        self.run_fake(Fake()); self.assertAlmostEqual(self.record['projected_seconds'],20.7)
        self.assertEqual(self.record['status'],'completed'); self.assertTrue(self.record['resource_admitted'])
    def test_projection_limit_failure_no_rerun(self):
        b=Fake(); self.run_fake(b,(2.,2.,2.))
        self.assertEqual(self.record['projected_seconds'],396.)
        self.assertEqual(self.record['status'],'resource_not_admitted'); self.assertEqual(len(b.calls),4)
    def test_each_failure_retains_attempt_and_stops(self):
        for at,expected in [(1,1),(2,1),(3,2),(4,3)]:
            with self.subTest(at=at):
                self.record={'attempts':[]}; b=Fake(fail=at)
                with self.assertRaises(ValueError): self.run_fake(b)
                self.assertEqual(len(b.calls),at); self.assertEqual(len(self.record['attempts']),expected)
                self.assertEqual(self.record['attempts'][-1]['status'],'failed')
                self.assertTrue(self.record['attempts'][-1]['native_solver_execution_count_unknown'])
                self.assertIn('seconds',self.record['attempts'][-1])
    def test_parity_failure_cannot_pass(self):
        b=Fake(parity=False)
        with self.assertRaisesRegex(ValueError,'fit_parity'): self.run_fake(b)
        self.assertEqual(len(self.record['attempts']),1)
    def test_tiny_array_generator_reproducible_order_and_norm(self):
        x,p=a.make_arrays(np,19,n=7,d=5,k=3); x2,p2=a.make_arrays(np,19,n=7,d=5,k=3)
        self.assertTrue(np.array_equal(x,x2)); self.assertTrue(np.array_equal(p,p2))
        self.assertEqual(x.dtype,np.float32); self.assertEqual(p.shape,(1,5,3))
        np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(p[0],axis=0),1,atol=1e-6)
        self.assertEqual(a.array_hash(x),a.array_hash(x.copy()))
    def test_load_transform_single_and_auditable_without_torch(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fake.py'; p.write_text('from pathlib import Path\n'+a.PATH_LINE+'\nvalue=7\n')
            m,r=a.load_exact('_fake_admission_probe',p,portable=True)
            self.assertEqual(m.UPSTREAM_PATH,Path(t)/'upstream/StatA.py')
            self.assertEqual(r['exact_path_substitutions'],1); self.assertEqual(r['original_sha256'],a.sha(p.read_bytes()))
            expected=p.read_text().replace(a.PATH_LINE,a.PORTABLE_LINE,1)
            self.assertEqual(r['executed_sha256'],a.sha(expected.encode()))
            p.write_text(p.read_text()+'\n'+a.PATH_LINE)
            with self.assertRaisesRegex(ValueError,'portable_path_count'): a.load_exact('_must_not_load',p,portable=True)
    def test_test_thread_transform_only_once(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fake.py'; p.write_text('class Torch:\n def set_num_interop_threads(self,n): self.n=n\ntorch=Torch()\ntorch.set_num_interop_threads(1)\n')
            m,r=a.load_exact('_fake_interop',p,test_threads=True)
            self.assertEqual(m.torch.n,4); self.assertEqual(r['exact_test_interop_substitutions'],1)
    def test_source_manifest_missing_or_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as t,patch.object(a,'ROOT',Path(t)):
            p=Path(t)/'source.py'; p.write_text('x=1\n')
            m=Path(t)/'manifest.json'; m.write_text(json.dumps({'files':{'source.py':a.sha(p.read_bytes())}}))
            digest=a.sha(m.read_bytes()); a.verify_bundle(digest)
            with self.assertRaisesRegex(ValueError,'manifest_sha256'): a.verify_bundle('0'*64)
            p.write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'bundle_file'): a.verify_bundle(digest)
    def test_no_torch_or_stata_runtime_loaded(self):
        self.assertFalse(any(k=='torch' or k.startswith('torch.') for k in sys.modules))
        self.assertNotIn('stata_cpu',sys.modules)


if __name__=='__main__': unittest.main(verbosity=2)
