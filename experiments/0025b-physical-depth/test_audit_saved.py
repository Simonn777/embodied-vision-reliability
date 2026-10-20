"""Small manufactured checks of the independent audit; no study data access."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
import audit_saved as a


class SavedAuditChecks(unittest.TestCase):
    def test_centered_weighted_formula_not_lstsq(self):
        x=np.linspace(-3.,2.,19);c=np.linspace(.6,1.4,19)
        X=np.column_stack((c*x,c));y=c*(1.7*x+6.)
        with mock.patch.object(np.linalg,'lstsq',side_effect=AssertionError('must not solve by lstsq')):
            result=a.closed_fit(X,y)
        self.assertEqual(result['status'],'usable')
        np.testing.assert_allclose(result['coefficients'],[1.7,6.],rtol=1e-13,atol=1e-13)
        self.assertEqual(result['rank'],2)

    def test_scale_and_invalid_fit_states(self):
        x=np.array([1.,2.,4.,8.])
        np.testing.assert_allclose(a.closed_fit(x[:,None],3*x)['coefficients'],[3.])
        self.assertEqual(a.closed_fit(np.ones((4,2)),x)['reason'],'rank_deficient')
        self.assertEqual(a.closed_fit(np.column_stack((10-x,np.ones(4))),x)['status'],'unavailable')
        bad=x.copy();bad[1]=np.nan
        self.assertEqual(a.closed_fit(bad[:,None],x)['status'],'unavailable')

    def test_geometry_planar_inverse_truth_and_model_independent_support(self):
        y,x=np.indices((480,640),dtype=float)
        source=1./(.03*(x-319.5)/481.2-.02*(y-239.5)/480.+.25)
        for view in range(1,5):
            raw={'view_%d_depth'%view:np.ones((336,448))*4.,'view_0_depth':np.ones((336,448))*4.}
            r=a.reconstruct(source,raw,view)
            middle_x=(r['source_xy'][:,0]-319.5)/481.2
            middle_y=(r['source_xy'][:,1]-239.5)/480.
            expected=1./(.03*middle_x-.02*middle_y+.25)/r['h']
            np.testing.assert_allclose(r['row_z'][r['support']],expected[r['support']],rtol=2e-15,atol=2e-15)
            raw['view_%d_depth'%view][:]=np.nan
            other=a.reconstruct(source,raw,view)
            np.testing.assert_array_equal(r['support'],other['support'])
            self.assertEqual(r['counts'],other['counts'])

    def test_exact_lattice_and_quantiles(self):
        g=a.geometry('rs_x_positive')
        self.assertEqual(g['u'][0],0);self.assertEqual(g['u'][-1],447)
        self.assertEqual(g['v'][0],0);self.assertEqual(g['v'][-1],335)
        self.assertEqual(int(g['train'].sum()),2048)
        d=a.distribution([0.,1.,4.,9.])
        self.assertEqual(d['mean'],3.5);self.assertEqual(d['median'],2.5)
        self.assertEqual(d['quantiles']['0.25'],.75)
        self.assertEqual(d['quantiles']['0.75'],5.25)

    def test_failed_outer_blocks_array_and_GT_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            common=NS(ROOT=root,verify_sources=lambda _: {},verify_outer=mock.Mock(side_effect=ValueError('outer failed')))
            spec=NS(loader=NS(exec_module=lambda _:None))
            args=NS(repo_root=root,expected_commit='a'*40,prediction_dir=root/'p',evaluation_dir=root/'e',
                    prediction_supervision=root/'po',evaluation_supervision=root/'eo')
            with mock.patch.object(a.importlib.util,'spec_from_file_location',return_value=spec), \
                 mock.patch.object(a.importlib.util,'module_from_spec',return_value=common), \
                 mock.patch.object(np,'load',side_effect=AssertionError('arrays must not open')):
                with self.assertRaisesRegex(ValueError,'outer failed'):
                    a.main_audit(args,{})
            common.verify_outer.assert_called_once()

    def test_npz_raw_hash_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);path=folder/'fake.npz';array=np.arange(6,dtype=np.float32).reshape(2,3)
            np.savez(path,depth=array)
            binding=dict(filename='fake.npz',bytes=path.stat().st_size,sha256=a.sha(path),
                arrays={'depth':dict(shape=[2,3],dtype='float32',raw_c_order_sha256=a.raw_sha(array))})
            np.testing.assert_array_equal(a.cache(folder,binding)['depth'],array)
            binding['arrays']['depth']['raw_c_order_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'raw array hash'):
                a.cache(folder,binding)

    def test_numeric_comparison_keeps_nonfinite_patterns_and_failures(self):
        r=dict(failures=[],maximum_absolute_difference=0.,maximum_tolerance_ratio=0.)
        a.compare([1.,np.nan],[1.,np.nan],'equal',r)
        self.assertEqual(r['failures'],[])
        a.compare([1.,np.inf],[1.,np.nan],'invalid',r)
        a.compare([2.],[1.],'different',r)
        self.assertEqual(len(r['failures']),2)
        self.assertGreater(r['maximum_tolerance_ratio'],1.)


if __name__=='__main__':
    unittest.main()
