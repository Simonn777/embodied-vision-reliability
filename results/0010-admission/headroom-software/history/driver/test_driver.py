"""Software-only fake fits/encoders. No Torch, real pixels/features or network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import driver
import native
import run as launcher


def load_fixture(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sys.path.insert(0,'/tmp/evr-0010-headroom-prototype')
import headroom
fake = load_fixture('headroom_fake_fixtures','/tmp/evr-0010-headroom-prototype/test_headroom.py')


class Ops:
    def __init__(self,out, *, fail=None):
        self.out=out; self.fail=fail; self.events=[]; self.reads=[]; self.encodes=0; self.label_calls=0
        self.runtime_record={'kind':'artificial fake; no ML runtime'}; self.b=fake.FakeBackend(fail_at=2 if fail=='fit' else None)
        self.names=['AnnualCrop_'+str(i)+'.jpg' for i in range(128)]
    def admit(self):
        self.events.append('admit')
        raw=[('fake bytes '+str(i)).encode() for i in range(128)]
        if self.fail=='duplicate': raw[1]=raw[0]
        self.raw=raw
        return dict(names=self.names, text_normalized=fake.fixture()[1][0].T.copy(),
                    members=[dict(name=n,bytes=len(r),crc32=0) for n,r in zip(self.names,raw)],
                    capability_member_sha256=[hashlib.sha256(('capfake'+str(i)).encode()).hexdigest() for i in range(8)], evidence={'artificial':True})
    def load_encoder(self):
        self.events.append('load')
        assert (self.out/'selection-lock.json').is_file()
    def read_member(self,j): self.reads.append(j); return self.raw[j]
    def preprocess(self,raw): self.events.append('preprocess'); assert len(raw)==4; return raw
    def encode(self,raw):
        self.encodes+=1; self.events.append('encode')
        if self.fail=='encoder': raise RuntimeError('fake error')
        x=np.zeros((4,512),np.float32); x[:,0]=1
        if self.fail=='zero': x[:]=0
        if self.fail=='nonfinite': x[0,0]=np.nan
        return x
    def release_encoder(self): self.events.append('release')
    def backend(self): assert 'release' in self.events; self.events.append('backend'); return self.b
    def labels(self,names):
        self.label_calls+=1; self.events.append('labels')
        assert names==self.names and (self.out/'prediction-seal.json').is_file()
        seal=json.loads((self.out/'prediction-seal.json').read_text())
        assert seal['predictions_sha256']==driver.digest(self.out/'predictions.json')
        assert json.loads((self.out/'predictions.json').read_text())['status']=='completed'
        return np.zeros(128,dtype=int)
    def close(self):
        self.events.append('close')
        if self.fail=='close': raise RuntimeError('fake close')


class Flow(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.out=self.root/'out'; self.marker=self.root/'started'
    def tearDown(self): self.tmp.cleanup()
    def execute(self,fail=None):
        ops=Ops(self.out,fail=fail)
        r=driver.run(self.out,self.marker,ops,headroom)
        return r,ops
    def test_32_once_only_batches_and_132_states_full_flow(self):
        r,o=self.execute()
        self.assertEqual(r['status'],'completed'); self.assertEqual(o.reads,list(range(128)))
        self.assertEqual((r['image_calls_attempted'],r['image_calls_completed'],r['images_completed'],r['text_calls']),(32,32,128,0))
        self.assertEqual(len(o.b.fits),132); self.assertEqual(len(o.b.heldout_calls),128); self.assertEqual(len(o.b.native_calls),384)
        self.assertEqual(r['scoring_attempted'],1); self.assertEqual(o.label_calls,1)
        self.assertLess(o.events.index('release'),o.events.index('backend')); self.assertLess(o.events.index('backend'),o.events.index('labels'))
        seal=json.loads((self.out/'prediction-seal.json').read_text()); self.assertEqual(seal['all_arms'],640)
        self.assertEqual(seal['features_sha256'],driver.digest(self.out/'features.npz'))
        z=np.load(self.out/'features.npz',allow_pickle=False)
        self.assertEqual(z['image_raw'].shape,(128,512)); self.assertEqual(z['text_normalized'].shape,(10,512)); z.close()
    def test_incomplete_fits_never_access_labels(self):
        r,o=self.execute('fit'); self.assertEqual(r['status'],'incomplete'); self.assertEqual(o.label_calls,0)
        self.assertFalse((self.out/'prediction-seal.json').exists()); self.assertFalse((self.out/'scores.json').exists())
        self.assertEqual(len(json.loads((self.out/'predictions.json').read_text())['rows']),128)
    def test_false_complete_with_missing_arm_is_not_sealed(self):
        original=headroom.predict
        def corrupt(*args):
            result=original(*args); path=Path(args[-1]); result['rows'][0]['statuses']['loo_alpha1']='not_attempted'
            path.write_text(json.dumps(result)); return result
        with mock.patch.object(headroom,'predict',side_effect=corrupt): r,o=self.execute()
        self.assertEqual(r['status'],'incomplete'); self.assertEqual(o.label_calls,0)
        self.assertFalse((self.out/'prediction-seal.json').exists())
    def test_nonfinite_outputs_saved_before_refusal(self):
        r,o=self.execute('nonfinite'); self.assertEqual(r['status'],'incomplete'); self.assertEqual(o.encodes,1)
        self.assertEqual(o.label_calls,0); self.assertFalse(o.b.fits)
        with np.load(self.out/'partial-features.npz') as z: self.assertTrue(np.isnan(z['image_raw'][0,0]))
    def test_zero_norm_never_next_batch(self):
        r,o=self.execute('zero'); self.assertEqual(r['images_completed'],4); self.assertEqual(o.encodes,1); self.assertEqual(r['status'],'incomplete')
    def test_encode_exception_retains_attempt_not_completion(self):
        r,o=self.execute('encoder'); self.assertEqual((r['image_calls_attempted'],r['image_calls_completed']),(1,0)); self.assertEqual(o.label_calls,0)
    def test_duplicate_bytes_stop_before_forward(self):
        r,o=self.execute('duplicate'); self.assertEqual(r['status'],'incomplete'); self.assertEqual(o.encodes,0); self.assertEqual(o.reads,[0,1])
    def test_new_output_does_not_bypass_persistent_marker(self):
        r,o=self.execute('duplicate'); before=self.marker.read_bytes()
        second=self.root/'other'; ops=Ops(second)
        with self.assertRaises(FileExistsError): driver.run(second,self.marker,ops,headroom)
        self.assertFalse(ops.events); self.assertEqual(self.marker.read_bytes(),before)
    def test_close_failure_retains_receipt_but_is_not_complete(self):
        r,o=self.execute('close'); self.assertEqual(r['status'],'incomplete'); self.assertEqual(r['close_failure_type'],'RuntimeError')
        self.assertEqual(json.loads((self.out/'receipt.json').read_text())['status'],'incomplete')
    def test_seal_tamper_during_scoring_is_not_complete(self):
        o=Ops(self.out); old=o.labels
        def labels(names):
            y=old(names); p=self.out/'predictions.json'; p.write_text(p.read_text()+' '); return y
        o.labels=labels; r=driver.run(self.out,self.marker,o,headroom)
        self.assertEqual(r['status'],'incomplete'); self.assertIn('changed during scoring',r['failure'])


class Admission(unittest.TestCase):
    def fixture(self):
        p={'timings':{'identities':1.,'runtime_import':1.,'load':1.}}
        for i in range(8):
            p['timings']['image_'+str(i)+'_preprocess']=.1; p['timings']['image_'+str(i)+'_encode']=.1
        t=dict(mode='timing',status='completed',resource_admitted=True,actual_torch_threads=[4,4],inputs_unchanged=True,
               inputs={'shape_features':[64,512],'shape_prototypes':[1,512,10]},runtime={'torch':'fake','numpy':'fake'},
               attempts=[dict(phase=phase,alpha=alpha,status='completed',seconds=.1,native_solver_executions=2,native_solver_execution_count_unknown=False) for phase,alpha in [('T63',1),('T641',1),('T644',4)]],projected_seconds=19.8)
        e=dict(supervisor_exit_code=0,child_exit_code=0,timed_out=False,address_space_bytes=8*1024**3,cpu_affinity=[0,1,2,3],numeric_threads=4,wall_seconds=1.)
        c=dict(cpu_ids=[0,1,2,3],runtime_versions={'torch':'fake','numpy':'fake'})
        return p,t,e,c
    def test_admission_arithmetic_and_four_by_four_threads(self):
        p,t,e,c=self.fixture(); r=native.timing_admission(p,t,e,c)
        self.assertAlmostEqual(r['solver_projection_seconds'],19.8); self.assertAlmostEqual(r['encoder_projection_seconds'],42.9)
    def test_missing_failed_or_excessive_measurement_refuses(self):
        mutations=[lambda p,t,e,c:t.update(status='pending'),lambda p,t,e,c:t.update(actual_torch_threads=[4,1]),
                   lambda p,t,e,c:t['attempts'][0].update(seconds=4),lambda p,t,e,c:p['timings'].update(load=400),
                   lambda p,t,e,c:e.update(timed_out=True),lambda p,t,e,c:t['attempts'][0].update(native_solver_execution_count_unknown=True),
                   lambda p,t,e,c:e.update(wall_seconds=float('nan'))]
        for f in mutations:
            p,t,e,c=self.fixture(); f(p,t,e,c)
            with self.subTest(f=f),self.assertRaises(driver.Invalid): native.timing_admission(p,t,e,c)
    def test_selection_order_exclusion_and_no_replacements(self):
        train=['AnnualCrop_'+str(i)+'.jpg' for i in range(160)]; excluded=train[:8]
        got=driver.selected_names(train,excluded)
        self.assertEqual(got,driver.selected_names(train[::-1],excluded)); self.assertFalse(set(got)&set(excluded))
        self.assertEqual(len(set(driver.ids_for(got))),128)
        with self.assertRaises(driver.Invalid): driver.selected_names(train[:130],excluded)
        with self.assertRaises(driver.Invalid): driver.selected_names(train+[train[0]],excluded)
    def test_input_file_replacement_and_symlink_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()/'x'; p.write_text('fake'); ref={'path':'x','sha256':driver.digest(p)}
            self.assertEqual(native.verified_path(ref,Path(d).resolve()),p)
            p.write_text('changed')
            with self.assertRaises(driver.Invalid): native.verified_path(ref,Path(d).resolve())
            p.unlink(); p.symlink_to(Path(d)/'no-file')
            with self.assertRaises(driver.Invalid): native.verified_path(ref,Path(d).resolve())
    def test_no_production_imports_or_actual_fits(self):
        self.assertFalse(any(x=='torch' or x.startswith('torch.') for x in sys.modules))


class Launcher(unittest.TestCase):
    def test_unadmitted_plan_cannot_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()/'plan.json'; p.write_text(json.dumps({'schema':'0010-headroom-prospective-v1','admitted':False}))
            with self.assertRaisesRegex(ValueError,'separate headroom admission'):
                launcher.main(['launch','--plan',str(p),'--plan-sha256',driver.digest(p),'--assetdir',d,'--output',str(Path(d)/'out')])
    def test_fixed_supervisor_contract_and_launch_marker(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d).resolve(); assets=base/'assets'; assets.mkdir(); cfg=base/'cfg'; cfg.write_text(json.dumps({'cpu_ids':[0,1,2,3]}))
            here=Path(__file__).parent
            plan={'schema':'0010-headroom-prospective-v1','admitted':True,'driver_files':{n:driver.digest(here/n) for n in ['run.py','native.py','driver.py']},
                  'inputs':{'capability_config':{'path':'cfg','sha256':driver.digest(cfg)},'supervisor':{'path':'fake-supervisor','sha256':native.SUPERVISOR_SHA}}}
            p=base/'plan.json'; p.write_text(json.dumps(plan)); supervisor=mock.Mock(); supervisor.supervise.return_value={'supervisor_exit_code':0}
            args=['launch','--plan',str(p),'--plan-sha256',driver.digest(p),'--assetdir',str(assets),'--output',str(base/'output'),'--supervision',str(base/'supervision'),'--python',sys.executable]
            with mock.patch.object(launcher,'load',return_value=supervisor):
                self.assertEqual(launcher.main(args),0)
                _,kw=supervisor.supervise.call_args; self.assertEqual(kw,{'seconds':900,'cpu_ids':[0,1,2,3]})
                command=supervisor.supervise.call_args.args[0]; self.assertIn('_child',command); self.assertIn(str(p),command)
                with self.assertRaises(FileExistsError): launcher.main(args)
                self.assertEqual(supervisor.supervise.call_count,1)


if __name__=='__main__': unittest.main(verbosity=2)
