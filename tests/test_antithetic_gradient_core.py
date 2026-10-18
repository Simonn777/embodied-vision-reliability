"""Artificial/offline fixtures only. No saved weights, images, or science calls."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
spec = importlib.util.spec_from_file_location('antithetic_gradient_core', Path(__file__).resolve().parents[1]/'scripts/antithetic_gradient_core.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)

HAS_TORCH = importlib.util.find_spec('torch') is not None
WIDTHS = (49, 4, 3, 1)


def state(widths=WIDTHS, zeros=False):
    rng = np.random.default_rng(73)
    result = {}
    for layer, left, right in zip((0, 2, 4), widths[:-1], widths[1:]):
        result[str(layer) + '.weight'] = (np.zeros((right, left)) if zeros else rng.normal(0, .03, (right, left))).astype(np.float32)
        result[str(layer) + '.bias'] = (np.zeros(right) if zeros else rng.normal(.1, .01, right)).astype(np.float32)
    return result


class NumPyFixtures(unittest.TestCase):
    def test_no_torch_import_for_module(self):
        root = Path(__file__).resolve().parents[1]/'scripts'
        script = 'import sys; import antithetic_gradient_core; assert "torch" not in sys.modules'
        result = subprocess.run([sys.executable, '-c', script], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_production_api_has_only_case_override(self):
        self.assertEqual(list(inspect.signature(core.run_case).parameters), ['noisy', 'covariance', 'weights', 'case_index'])
        self.assertEqual(inspect.signature(core.run_case).parameters['case_index'].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(sum(a.size for a in state((49,64,32,1)).values()), 5313)

    def test_patch_order_center_and_owned_copy(self):
        y = np.arange(128*128, dtype=np.float32).reshape(128,128)
        patches = core.extract_patches(y)
        self.assertEqual(patches.shape, (122*122,49))
        np.testing.assert_array_equal(patches[0], y[:7,:7].ravel())
        np.testing.assert_array_equal(patches[123], y[1:8,1:8].ravel())
        np.testing.assert_array_equal(patches[:,24], y[3:-3,3:-3].ravel())
        patches[0,0] = -1
        self.assertEqual(y[0,0], 0)
        for bad in (y.astype(np.float64), y[:127], np.full_like(y,np.nan)):
            with self.assertRaises(ValueError): core.extract_patches(bad)

    def test_raw_mean_order_float64_and_stable_occurrence_ties(self):
        rows = np.array([[3]*49, [1]*49, [1]*49, [2]*49], dtype=np.float32)
        np.testing.assert_array_equal(core.mean_pair_order(rows), [1,2,3,0])
        # Float32 accumulation would lose the small residual amid cancellation.
        rows[:] = 0
        rows[0,:3] = [2**25, 1, -2**25]
        rows[1,:3] = [2**25, 2, -2**25]
        rows[2,:3] = [2**25, 0, -2**25]
        rows[3,:3] = [2**25, 1, -2**25]
        np.testing.assert_array_equal(core.mean_pair_order(rows), [2,0,3,1])
        for bad in (rows[:3], rows.astype(np.float64), rows[:,:48]):
            with self.assertRaises(ValueError): core.mean_pair_order(bad)

    def test_covariance_cholesky_rounding_is_recorded_without_repair(self):
        c = np.eye(49,dtype=np.float64)*.3
        out, f, error = core.validate_covariance(c)
        np.testing.assert_array_equal(c,out)
        np.testing.assert_array_equal(f,np.linalg.cholesky(c).astype(np.float32))
        self.assertGreater(error,0)
        self.assertAlmostEqual(error,abs(float(f[0,0])**2-.3))
        self.assertFalse(np.shares_memory(out,c))

    def test_covariance_invalid_and_singular_rejected(self):
        c = np.eye(49,dtype=np.float64)
        asymmetric=c.copy(); asymmetric[0,1]=1e-6
        singular=c.copy(); singular[0,0]=0
        indefinite=c.copy(); indefinite[0,1]=indefinite[1,0]=2
        nonfinite=c.copy(); nonfinite[1,1]=np.inf
        for bad in (c.astype(np.float32),c[:48],asymmetric,singular,indefinite,nonfinite):
            with self.subTest(dtype=bad.dtype,shape=bad.shape):
                with self.assertRaises(ValueError): core.validate_covariance(bad)

    def test_state_schema_hash_order_mutation_and_copy(self):
        s=state(); copied=core._validate_weights(s,WIDTHS)
        self.assertEqual(core.state_dict_sha256(s),core.state_dict_sha256(dict(reversed(list(s.items())))))
        before=core.state_dict_sha256(s)
        copied['0.bias'][0]+=1
        self.assertEqual(core.state_dict_sha256(s),before)
        self.assertNotEqual(core.state_dict_sha256(copied),before)
        for key,value in [('wrong',np.zeros(1,dtype=np.float32)),('0.bias',np.zeros(4,dtype=np.float64)),('4.bias',np.array([np.nan],dtype=np.float32))]:
            bad=s.copy();bad[key]=value
            with self.assertRaises(ValueError): core._validate_weights(bad,WIDTHS)

    def test_array_hash_binds_shape_dtype_and_bytes(self):
        a=np.arange(4,dtype=np.float32)
        self.assertNotEqual(core.array_sha256(a),core.array_sha256(a.reshape(2,2)))
        self.assertNotEqual(core.array_sha256(a),core.array_sha256(a.astype(np.float64)))
        self.assertEqual(core.array_sha256(a),core.array_sha256(a.copy()))

    def test_deadline_exact_boundary_and_bad_integer(self):
        now=[0.0]; d=core._Deadline(90,lambda:now[0])
        now[0]=89.999;d.check()
        now[0]=90
        with self.assertRaises(core.DeadlineExceeded):d.check()
        for bad in (True,1.5,-1,8):
            with self.assertRaises(ValueError):core._integer(bad,'case',0,8)


@unittest.skipUnless(HAS_TORCH,'Torch is unavailable; no installation attempted')
class TorchFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch=core._torch_runtime()

    def test_actual_production_normal_prefix_coupling(self):
        t=self.torch
        for seed in (0, 713800, 2**62+31):
            full=t.randn((512,49),generator=t.Generator().manual_seed(seed),dtype=t.float32)
            half=t.randn((256,49),generator=t.Generator().manual_seed(seed),dtype=t.float32)
            self.assertTrue(t.equal(full[:256],half), 'CPU normal kernel did not preserve required production prefix')
            # Tiny fixture sizes also multiples of the normal kernel vector width.
            small=t.randn((32,49),generator=t.Generator().manual_seed(seed),dtype=t.float32)
            tiny=t.randn((16,49),generator=t.Generator().manual_seed(seed),dtype=t.float32)
            self.assertTrue(t.equal(small[:16],tiny))

    def estimates(self, zero=False):
        t=self.torch
        p=np.zeros((7,49),dtype=np.float32) if zero else np.random.default_rng(1).normal(size=(7,49)).astype(np.float32)
        model=core._new_model(t,state(zeros=zero),WIDTHS)
        before=core.state_dict_sha256(core._export_state(model))
        calls=[];hook=model.register_forward_hook(lambda *_:calls.append(1))
        results={m:core._estimate_gradient(t,model,t.from_numpy(p),t.eye(49),method=m,batch_size=32,index_seed=10,augmentation_seed=11,pairing_seed=12,capture_debug=True) for m in core.METHODS}
        hook.remove()
        self.assertEqual(len(calls),4)
        self.assertEqual(core.state_dict_sha256(core._export_state(model)),before)
        return results

    def test_pairing_augmentation_and_with_replacement_semantics(self):
        results=self.estimates()
        debug={m:r[1]['debug'] for m,r in results.items()}
        base=debug['independent']['base_indices']
        self.assertLess(len(np.unique(base)),len(base))
        for m in core.METHODS:
            np.testing.assert_array_equal(debug[m]['base_indices'],base)
        for m in core.METHODS[1:]:
            np.testing.assert_array_equal(debug[m]['epsilon'],debug['independent']['epsilon'][:16])
            np.testing.assert_array_equal(debug[m]['noise'][::2],-debug[m]['noise'][1::2])
        np.testing.assert_array_equal(debug['same_patch']['used_indices'],np.repeat(base[:16],2))
        for m in ('random_antithetic','mean_antithetic'):
            np.testing.assert_array_equal(np.sort(debug[m]['order']),np.arange(32))
        means=np.mean(debug['mean_antithetic']['rows'],axis=1,dtype=np.float64)
        self.assertTrue(np.all(means[:-1]<=means[1:]))

    def test_zero_model_manual_r2r_loss_and_bias_gradient(self):
        results=self.estimates(zero=True)
        for m,(gradient,record) in results.items():
            noise=record['debug']['noise'][:,24]
            self.assertAlmostEqual(record['loss'],float(np.mean(4*noise.astype(np.float64)**2)),places=5)
            self.assertAlmostEqual(float(gradient[-1]),float(4*np.mean(noise,dtype=np.float64)),places=6)
            np.testing.assert_array_equal(gradient[:-1],0)
            if m!='independent':self.assertLess(abs(float(gradient[-1])),1e-6)
        self.assertGreater(abs(float(results['independent'][0][-1])),1e-3)

    def test_full_tiny_grid_replay_weight_identity_and_rng_isolation(self):
        p=np.random.default_rng(8).normal(size=(9,49)).astype(np.float32)
        t=self.torch;before=t.get_rng_state().clone()
        args=dict(case_index=1,replicates=2,batch_size=32,widths=WIDTHS)
        one=core._run_patches(p,np.eye(49,dtype=np.float64),state(),**args)
        two=core._run_patches(p,np.eye(49,dtype=np.float64),state(),**args)
        self.assertTrue(t.equal(before,t.get_rng_state()))
        self.assertEqual(one['gradients'].shape,(2,4,2,219))
        self.assertEqual(one['gradients'].dtype,np.dtype('float32'))
        self.assertEqual(one['walls'].dtype,np.dtype('float64'))
        self.assertTrue(one['completed'].all())
        self.assertTrue(np.isfinite(one['gradients']).all())
        np.testing.assert_array_equal(one['gradients'],two['gradients'])
        r=one['receipt'];self.assertEqual(r['status'],'completed');self.assertEqual(r['completed_measurements'],16)
        self.assertTrue(r['weight_unchanged']);self.assertEqual(r['optimizer_steps'],0)
        json.dumps(r,allow_nan=False)
        for regime in range(2):
            for rep in range(2):
                recs=[x for x in r['records'] if x['regime']==regime and x['replicate']==rep]
                self.assertEqual(len({x['base_indices_sha256'] for x in recs}),1)
                self.assertEqual(len({x['augmentation_seed'] for x in recs}),1)
        a=[x for x in r['records'] if x['regime']==0]
        self.assertEqual(len({x['base_indices_sha256'] for x in a}),1)
        random_a=[x for x in a if x['method']=='random_antithetic']
        self.assertNotEqual(random_a[0]['pair_order_sha256'],random_a[1]['pair_order_sha256'])
        b=[x for x in r['records'] if x['regime']==1 and x['method']=='independent']
        self.assertNotEqual(b[0]['base_indices_sha256'],b[1]['base_indices_sha256'])
        self.assertEqual(r['execution_order'][0]['method_indices'],[1,2,3,0])
        self.assertEqual(r['execution_order'][1]['method_indices'],[2,3,0,1])

    def test_completed_over_budget_gradient_is_preserved_but_case_fails(self):
        class Clock:
            now=0.0
            def __call__(self):return self.now
        clock=Clock();original=core._new_model
        def build(*args):
            model=original(*args)
            model.register_forward_hook(lambda *_:setattr(clock,'now',2.0))
            return model
        with mock.patch.object(core,'_new_model',side_effect=build):
            result=core._run_patches(np.zeros((3,49),dtype=np.float32),np.eye(49,dtype=np.float64),state(),case_index=0,replicates=2,batch_size=32,widths=WIDTHS,deadline_seconds=1,clock=clock)
        self.assertEqual(result['receipt']['status'],'failed')
        self.assertEqual(result['receipt']['failure']['type'],'DeadlineExceeded')
        self.assertEqual(int(result['completed'].sum()),1)
        self.assertTrue(np.isfinite(result['gradients'][result['completed']]).all())
        self.assertTrue(np.isnan(result['gradients'][~result['completed']]).all())
        self.assertTrue(result['receipt']['weight_unchanged'])
        self.assertEqual(result['walls'][0,0,0],2.0)


if __name__=='__main__':unittest.main(verbosity=2)
