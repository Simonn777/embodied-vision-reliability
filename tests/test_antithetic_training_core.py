"""Artificial fixtures only: tiny private MLPs and virtual training clocks."""
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import antithetic_training_core as core

HAS_TORCH = importlib.util.find_spec('torch') is not None
WIDTHS = (49,4,3,1)


class Clock:
    def __init__(self):self.now=0.0
    def __call__(self):return self.now


class NumPyFixtures(unittest.TestCase):
    def test_lazy_import_and_public_api(self):
        result=subprocess.run([sys.executable,'-c','import sys;import antithetic_training_core;assert "torch" not in sys.modules'],cwd=Path(__file__).resolve().parents[1]/'scripts',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(list(inspect.signature(core.fit_trajectory).parameters),['noisy','covariance','method','case_index'])

    def test_exact_new_seed_strings_and_original_initialization(self):
        self.assertEqual(core.step_seeds(0,1),{'indices':8816578677438601998,'augmentation':2532799512774255878,'pairing':713688446443204181})
        self.assertEqual([core.init_seed(i) for i in range(8)],[620001,620011,620101,620111,620201,620211,620301,620311])
        self.assertNotEqual(core.step_seeds(0,1),core.step_seeds(0,2))
        self.assertNotEqual(core.step_seeds(0,1),core.step_seeds(1,1))
        for case,step in ((True,1),(8,1),(0,0),(0,100001),(0,1.5)):
            with self.assertRaises(ValueError):core.step_seeds(case,step)

    def test_covariance_and_patch_validation(self):
        c=np.eye(49,dtype=np.float64)*.3
        out,f,error=core.validate_covariance(c)
        np.testing.assert_array_equal(c,out)
        np.testing.assert_array_equal(f,np.linalg.cholesky(c).astype(np.float32))
        self.assertGreater(error,0)
        asym=c.copy();asym[0,1]=.1
        singular=c.copy();singular[0,0]=0
        for bad in (c.astype(np.float32),c[:48],asym,singular,np.full_like(c,np.nan)):
            with self.assertRaises(ValueError):core.validate_covariance(bad)
        y=np.arange(128*128,dtype=np.float32).reshape(128,128)
        p=core.extract_patches(y)
        np.testing.assert_array_equal(p[:,24],y[3:-3,3:-3].ravel())
        for bad in (y.astype(np.float64),y[:127],np.full_like(y,np.inf)):
            with self.assertRaises(ValueError):core.extract_patches(bad)

    def test_content_hash_identity_and_dtype_binding(self):
        x=np.arange(4,dtype=np.float32)
        self.assertNotEqual(core.array_sha256(x),core.array_sha256(x.astype(np.float64)))
        s={'b':x,'a':x+1}
        self.assertEqual(core.state_dict_sha256(s),core.state_dict_sha256(dict(reversed(list(s.items())))))
        copied={k:v.copy() for k,v in s.items()};copied['a'][0]+=1
        self.assertNotEqual(core.state_dict_sha256(s),core.state_dict_sha256(copied))


@unittest.skipUnless(HAS_TORCH,'Torch unavailable; no installation attempted')
class TorchFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.t=core._torch_runtime()

    def run_tiny(self,method='independent',updates=3,target=2.0,overrun=.05,step_cost=1.0,copy_cost=0.0,max_steps=20,max_case=90.0,serialization_cost=0.0):
        clock=Clock();step_original=self.t.optim.Adam.step;snap_original=core._snapshot;dumps_original=core.json.dumps
        def step(optimizer,*args,**kwargs):
            out=step_original(optimizer,*args,**kwargs);clock.now+=step_cost;return out
        def snapshot(model):
            out=snap_original(model);clock.now+=copy_cost;return out
        def dumps(value,*args,**kwargs):
            result=dumps_original(value,*args,**kwargs)
            if isinstance(value,dict) and value.get('schema')=='0008-training-core-v1':clock.now+=serialization_cost
            return result
        p=np.random.default_rng(35).normal(0,.1,(7,49)).astype(np.float32)
        with mock.patch.object(self.t.optim.Adam,'step',step),mock.patch.object(core,'_snapshot',snapshot),mock.patch.object(core.json,'dumps',dumps):
            return core._fit_patches(p,np.eye(49,dtype=np.float64)*.1,method,case_index=0,widths=WIDTHS,update_limit=updates,wall_target=target,wall_overrun=overrun,batch_size=32,max_steps=max_steps,max_case_seconds=max_case,clock=clock)

    def test_actual_production_prefix_and_batch_pairing(self):
        t=self.t;seeds=core.step_seeds(0,1)
        full=t.randn((512,49),generator=t.Generator().manual_seed(seeds['augmentation']))
        half=t.randn((256,49),generator=t.Generator().manual_seed(seeds['augmentation']))
        self.assertTrue(t.equal(full[:256],half))
        p=t.arange(7*49,dtype=t.float32).reshape(7,49)/100
        batches={m:core._batch(t,p,t.eye(49),m,32,seeds) for m in core.METHODS}
        idx=batches['independent'][2]
        for m,b in batches.items():
            self.assertTrue(t.equal(b[2],idx))
            rows=p[b[3]]
            noise=(b[0]-rows)/.5
            self.assertTrue(t.allclose(b[1],rows[:,24]-2*noise[:,24],atol=2e-6))
            if m!='independent':
                self.assertTrue(t.equal(b[5],batches['independent'][5][:16]))
                self.assertTrue(t.allclose(noise[::2],-noise[1::2],atol=1e-6))
        self.assertTrue(t.equal(batches['same_patch'][3],idx[:16].repeat_interleave(2)))
        means=p[batches['mean_antithetic'][3]].numpy().mean(axis=1,dtype=np.float64)
        self.assertTrue(np.all(means[:-1]<=means[1:]))

    def test_wall_before_fixed_updates_preserves_detached_snapshot(self):
        r=self.run_tiny()
        self.assertEqual(r['status'],'completed')
        wall=r['endpoints']['wall5s'];fixed=r['endpoints']['updates500']
        self.assertEqual(wall['receipt']['step'],2);self.assertEqual(fixed['receipt']['step'],3)
        self.assertNotEqual(wall['receipt']['state_dict_sha256'],fixed['receipt']['state_dict_sha256'])
        self.assertEqual(core.state_dict_sha256(wall['state_dict_np']),wall['receipt']['state_dict_sha256'])
        with self.assertRaises(ValueError):wall['state_dict_np']['0.bias'][0]=1
        self.assertEqual(r['receipt']['optimizer_steps_executed'],3)
        self.assertEqual(r['receipt']['completed_updates'],3)
        self.assertEqual(r['receipt']['optimizer']['initial_state_entries'],0)
        json.dumps(r['receipt'],allow_nan=False)

    def test_fixed_updates_before_wall_uses_same_single_trajectory(self):
        r=self.run_tiny(updates=2,target=3.0)
        self.assertEqual(r['endpoints']['updates500']['receipt']['step'],2)
        self.assertEqual(r['endpoints']['wall5s']['receipt']['step'],3)
        self.assertEqual(r['receipt']['optimizer_steps_executed'],3)
        self.assertEqual(r['receipt']['drawn_batches'],3)
        self.assertEqual(r['receipt']['inference_calls'],0)

    def test_same_update_checkpoint_copy_shared_and_charged(self):
        r=self.run_tiny(updates=2,target=2.0,copy_cost=.03125)
        self.assertEqual(r['status'],'completed')
        a=r['endpoints']['updates500'];b=r['endpoints']['wall5s']
        self.assertIs(a['state_dict_np'],b['state_dict_np'])
        self.assertTrue(a['receipt']['shared_snapshot']);self.assertTrue(b['receipt']['shared_snapshot'])
        self.assertEqual(b['receipt']['training_elapsed_seconds'],2.03125)
        self.assertEqual(b['receipt']['boundary_elapsed_seconds'],2.0)
        self.assertEqual(b['receipt']['update_completion_elapsed_seconds'],2.0)
        self.assertEqual(r['receipt']['optimizer_steps_executed'],2)

    def test_fixed_checkpoint_copy_crossing_target_cannot_relabel_update(self):
        for tolerance,expected in ((.05,'failed'),(1.1,'completed')):
            with self.subTest(tolerance=tolerance):
                r=self.run_tiny(updates=2,target=2.0,step_cost=.99,copy_cost=.03125,overrun=tolerance)
                self.assertEqual(r['status'],expected)
                fixed=r['endpoints']['updates500']['receipt'];wall=r['endpoints']['wall5s']['receipt']
                self.assertEqual(fixed['step'],2)
                self.assertLess(fixed['update_completion_elapsed_seconds'],2.0)
                self.assertGreater(fixed['training_elapsed_seconds'],2.0)
                self.assertEqual(wall['step'],3)
                self.assertGreater(wall['update_completion_elapsed_seconds'],2.0)
                self.assertFalse(wall['shared_snapshot'])
                self.assertEqual(r['receipt']['optimizer_steps_executed'],3)

    def test_final_serialization_cannot_escape_cooperative_case_cap(self):
        r=self.run_tiny(updates=2,target=2.0,serialization_cost=100.0)
        self.assertEqual(r['status'],'failed')
        self.assertIn('serialization',r['receipt']['failure']['message'])
        self.assertEqual(r['receipt']['receipt_serialization_seconds'],100.0)
        self.assertEqual(r['receipt']['training_wall_seconds'],2.0)
        self.assertEqual(len(r['endpoints']),2)

    def test_copy_overrun_invalidates_entire_trajectory_and_keeps_evidence(self):
        r=self.run_tiny(updates=4,target=2.0,copy_cost=.0625)
        self.assertEqual(r['status'],'failed')
        self.assertIn('overrun',r['receipt']['failure']['message'])
        self.assertEqual(r['receipt']['completed_updates'],2)
        self.assertNotIn('updates500',r['endpoints'])
        self.assertFalse(r['endpoints']['wall5s']['receipt']['valid'])
        self.assertEqual(r['endpoints']['wall5s']['receipt']['overrun_seconds'],.0625)

    def test_exact_time_tolerance_is_inclusive(self):
        r=self.run_tiny(updates=2,target=2.0,copy_cost=.03125,overrun=.03125)
        self.assertEqual(r['status'],'completed')

    def test_maximum_steps_and_cooperative_case_are_fail_closed(self):
        for kwargs in ({'updates':4,'target':10.0,'max_steps':2},{'updates':4,'target':2.0,'max_case':1.0}):
            with self.subTest(kwargs=kwargs):
                r=self.run_tiny(**kwargs)
                self.assertEqual(r['status'],'failed')
                self.assertLess(r['receipt']['optimizer_steps_executed'],4)
                self.assertEqual(r['receipt']['failure']['type'],'TimeoutError')

    def test_common_initialization_and_streams_across_all_four(self):
        before=self.t.get_rng_state().clone()
        results=[self.run_tiny(m,updates=2,target=2.0) for m in core.METHODS]
        self.assertTrue(self.t.equal(before,self.t.get_rng_state()))
        self.assertEqual(len({r['receipt']['initial_weight_sha256'] for r in results}),1)
        for key in ('base_indices','gaussian_prefix','step_seed_tuple'):
            self.assertEqual(len({r['receipt']['stream_sha256'][key] for r in results}),1)
        repeat=self.run_tiny('mean_antithetic',updates=2,target=2.0)
        self.assertEqual(repeat['receipt']['final_weight_sha256'],results[2]['receipt']['final_weight_sha256'])
        self.assertNotEqual(results[0]['receipt']['initial_weight_sha256'],results[0]['receipt']['final_weight_sha256'])

    def test_original_seed_initialization_and_adam_hyperparameters(self):
        seed=core.init_seed(0)
        model=core._new_model(self.t,seed,WIDTHS)
        with self.t.random.fork_rng(devices=[]):
            self.t.manual_seed(seed)
            reference=self.t.nn.Sequential(self.t.nn.Linear(49,4),self.t.nn.ReLU(),self.t.nn.Linear(4,3),self.t.nn.ReLU(),self.t.nn.Linear(3,1))
        one=core._snapshot(model)[1];two=core._snapshot(reference)[1]
        self.assertEqual(one,two)
        r=self.run_tiny(updates=1,target=1.0)
        self.assertEqual(r['receipt']['optimizer'],{'name':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.0,'amsgrad':False,'foreach':False,'initial_state_entries':0})

    def test_fixed_50_loss_records_without_extra_forwards(self):
        count=[0];original=core._new_model
        def build(*a):
            model=original(*a);model.register_forward_hook(lambda *_:count.__setitem__(0,count[0]+1));return model
        with mock.patch.object(core,'_new_model',side_effect=build):
            r=self.run_tiny(updates=50,target=50.0,max_steps=50,max_case=90)
        self.assertEqual(count[0],50)
        self.assertEqual([v['step'] for v in r['receipt']['loss_every_50']],[50])
        self.assertEqual(r['receipt']['loss_every_50'][0]['mean_batch_loss'],r['receipt']['last_loss'])
        self.assertEqual(r['receipt']['optimizer_steps_executed'],50)

    def test_nonfinite_loss_stops_before_optimizer(self):
        t=self.t;original=core._batch
        def bad(*a):
            values=list(original(*a));values[1]=t.full_like(values[1],float('inf'));return tuple(values)
        with mock.patch.object(core,'_batch',side_effect=bad):r=self.run_tiny()
        self.assertEqual(r['status'],'failed')
        self.assertEqual(r['receipt']['optimizer_steps_executed'],0)
        self.assertEqual(r['receipt']['drawn_batches'],1)
        self.assertEqual(r['receipt']['completed_updates'],0)
        self.assertEqual(r['endpoints'],{})


if __name__=='__main__':unittest.main(verbosity=2)
