"""No real inputs. Independent MC oracle and tiny endpoint restore fixtures."""
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import antithetic_training_core as core
import antithetic_training_inference as inference

HAS_TORCH=importlib.util.find_spec('torch') is not None
WIDTHS=(49,4,3,1)


def artificial_state():
    rng=np.random.default_rng(81);result={}
    for layer,left,right in zip((0,2,4),WIDTHS[:-1],WIDTHS[1:]):
        result[str(layer)+'.weight']=rng.normal(0,.03,(right,left)).astype(np.float32)
        result[str(layer)+'.bias']=rng.normal(.1,.02,right).astype(np.float32)
    return result


class NumPyFixtures(unittest.TestCase):
    def test_import_lazy_and_fixed_public_signature(self):
        helper=Path(core.__file__).stem
        script='import sys,importlib;core=importlib.import_module('+repr(helper)+');sys.modules["antithetic_training_core"]=core;import antithetic_training_inference;assert "torch" not in sys.modules'
        env=dict(os.environ,PYTHONPATH=str(Path(core.__file__).resolve().parent))
        r=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parent,env=env,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(list(inspect.signature(inference.predict_endpoint).parameters),['noisy','covariance','state','case_index'])

    def test_new_independent_seed_exact_bytes_and_case_range(self):
        for case in range(8):
            expected=int.from_bytes(hashlib.sha256(('evr0008|'+str(case)+'|inference|0').encode('ascii')).digest()[:8],'little')&((1<<63)-1)
            self.assertEqual(inference.inference_seed(case),expected)
            self.assertNotIn(expected,core.step_seeds(case,1).values())
        for bad in (True,-1,8,1.0):
            with self.assertRaises(ValueError):inference.inference_seed(bad)

    def test_strict_weight_schema_shape_dtype_finite_and_owned_copy(self):
        s=artificial_state();o=inference._weights(s,WIDTHS)
        self.assertEqual(core.state_dict_sha256(s),core.state_dict_sha256(o))
        o['0.bias'][0]+=1;self.assertNotEqual(core.state_dict_sha256(s),core.state_dict_sha256(o))
        bads=[]
        bad=s.copy();bad.pop('0.bias');bads.append(bad)
        bad=s.copy();bad['extra']=np.zeros(1,dtype=np.float32);bads.append(bad)
        for val in (s['0.weight'].astype(np.float64),s['0.weight'][:3],np.full_like(s['0.weight'],np.nan)):
            bad=s.copy();bad['0.weight']=val;bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):inference._weights(bad,WIDTHS)

    def test_production_call_and_draw_arithmetic(self):
        self.assertEqual(((122*122+1023)//1024)*50,750)
        self.assertEqual(122*122*49*50,36465800)


@unittest.skipUnless(HAS_TORCH,'Torch unavailable; no installation attempted')
class TorchFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.t=core._torch_runtime()

    def record(self):return {'forward_calls':0,'forward_calls_attempted':0,'standard_normal_draws':0}

    def test_independent_chunk_draw_oracle_and_float64_mean(self):
        t=self.t
        features=t.arange(5*49,dtype=t.float32).reshape(5,49)/100
        factor=t.eye(49,dtype=t.float32)*.3
        class Center(t.nn.Module):
            def forward(self,x):return x[:,24:25]
        record=self.record()
        observed=inference._mc_predict(t,Center(),features,factor,seed=44,chunk_size=2,mc_samples=3,record=record)
        gen=t.Generator().manual_seed(44);expected=[]
        for start in (0,2,4):
            x=features[start:start+2];outputs=[]
            for _ in range(3):
                e=t.randn(x.shape,generator=gen,dtype=t.float32)
                outputs.append((x+.5*(e@factor.T))[:,24].numpy().astype(np.float64))
            expected.extend(np.mean(outputs,axis=0).astype(np.float32))
        np.testing.assert_array_equal(observed,np.array(expected,dtype=np.float32))
        self.assertEqual(record,{'forward_calls':9,'forward_calls_attempted':9,'standard_normal_draws':735})

    def test_no_interpolation_clipping_or_extra_direct_forward(self):
        t=self.t;features=t.zeros((5,49),dtype=t.float32)
        features[:,24]=t.tensor([-2.,0.,.3,1.,2.])
        class Center(t.nn.Module):
            def forward(self,x):return x[:,24:25]
        r=self.record()
        got=inference._mc_predict(t,Center(),features,t.zeros((49,49)),seed=4,chunk_size=2,mc_samples=2,record=r)
        np.testing.assert_array_equal(got,features[:,24].numpy())
        self.assertEqual(r['forward_calls'],6)

    def test_float64_accumulation_prevents_float32_cancellation(self):
        t=self.t
        class Cycling(t.nn.Module):
            def __init__(self):super().__init__();self.index=0
            def forward(self,x):
                value=(2**25,1.,-(2**25))[self.index];self.index+=1
                return t.full((x.shape[0],1),value,dtype=t.float32)
        got=inference._mc_predict(t,Cycling(),t.zeros((1,49)),t.zeros((49,49)),seed=1,chunk_size=1,mc_samples=3,record=self.record())
        self.assertEqual(got[0],np.float32(1/3))

    def test_endpoint_restore_preserves_parameters_rng_and_no_optimizer(self):
        t=self.t;s=artificial_state();before=core.state_dict_sha256(s)
        p=np.random.default_rng(82).normal(size=(6,49)).astype(np.float32)
        rng=t.get_rng_state().clone()
        kwargs=dict(case_index=1,widths=WIDTHS,chunk_size=4,mc_samples=3,output_shape=(2,3))
        with mock.patch.object(t.optim,'Adam',side_effect=AssertionError('optimizer forbidden')):
            first=inference._predict_patches(p,np.eye(49,dtype=np.float64)*.1,s,**kwargs)
            second=inference._predict_patches(p,np.eye(49,dtype=np.float64)*.1,s,**kwargs)
        self.assertTrue(t.equal(rng,t.get_rng_state()))
        self.assertEqual(first['status'],'completed')
        np.testing.assert_array_equal(first['prediction'],second['prediction'])
        self.assertEqual(first['prediction'].shape,(2,3));self.assertEqual(first['prediction'].dtype,np.dtype('float32'))
        r=first['receipt'];self.assertTrue(r['weight_unchanged'])
        self.assertEqual(r['weight_content_sha256'],before);self.assertEqual(r['weight_after_sha256'],before)
        self.assertEqual(core.state_dict_sha256(s),before)
        self.assertEqual(r['forward_calls'],6);self.assertEqual(r['standard_normal_draws'],882)
        self.assertEqual(r['direct_forward_calls'],0);self.assertEqual(r['optimizer_steps'],0)
        self.assertEqual(r['prediction_sha256'],core.array_sha256(first['prediction']))

    def test_same_seed_draws_for_different_locked_endpoint_states(self):
        t=self.t;p=np.zeros((3,49),dtype=np.float32);states=[]
        for bias in (.1,.2):
            s={k:np.zeros_like(v) for k,v in artificial_state().items()};s['4.bias'][0]=bias;states.append(s)
        seen=[];original=inference._mc_predict
        def capture(torch,model,features,factor,**kwargs):
            seen.append(kwargs['seed']);return original(torch,model,features,factor,**kwargs)
        with mock.patch.object(inference,'_mc_predict',side_effect=capture):
            results=[inference._predict_patches(p,np.eye(49,dtype=np.float64),s,case_index=0,widths=WIDTHS,chunk_size=2,mc_samples=2) for s in states]
        self.assertEqual(seen,[inference.inference_seed(0)]*2)
        for r,s in zip(results,states):np.testing.assert_array_equal(r['prediction'],np.full(3,s['4.bias'][0],dtype=np.float32))

    def test_bad_output_shape_nonfinite_fail_without_returning_partial(self):
        t=self.t;original=core._new_model
        for mode in ('shape','nan'):
            def build(*a):
                model=original(*a)
                if mode=='shape':model.forward=lambda x:t.zeros((x.shape[0],2),dtype=t.float32)
                else:model.forward=lambda x:t.full((x.shape[0],1),float('nan'),dtype=t.float32)
                return model
            with self.subTest(mode=mode),mock.patch.object(core,'_new_model',side_effect=build):
                r=inference._predict_patches(np.zeros((2,49),dtype=np.float32),np.eye(49,dtype=np.float64),artificial_state(),case_index=0,widths=WIDTHS,chunk_size=2,mc_samples=2)
                self.assertEqual(r['status'],'failed');self.assertIsNone(r['prediction'])
                self.assertEqual(r['receipt']['forward_calls'],1)
                self.assertTrue(r['receipt']['weight_unchanged'])

    def test_weight_mutation_is_rejected(self):
        t=self.t;original=core._new_model
        def build(*a):
            model=original(*a);forward=model.forward
            def bad(x):
                model[4].bias.add_(.01)
                return forward(x)
            model.forward=bad;return model
        with mock.patch.object(core,'_new_model',side_effect=build):
            r=inference._predict_patches(np.zeros((2,49),dtype=np.float32),np.eye(49,dtype=np.float64),artificial_state(),case_index=0,widths=WIDTHS,chunk_size=2,mc_samples=2)
        self.assertEqual(r['status'],'failed');self.assertIsNone(r['prediction'])
        self.assertFalse(r['receipt']['weight_unchanged'])
        self.assertEqual(r['receipt']['failure']['type'],'WeightMutation')


if __name__=='__main__':unittest.main(verbosity=2)
