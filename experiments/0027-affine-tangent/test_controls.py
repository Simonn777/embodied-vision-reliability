"""Prepared manufactured/fake-API checks. No model or scientific input access."""
import json
from types import SimpleNamespace
import unittest
import numpy as np
from controls import solve_scale_only, solve_scale_only_ordered, solve_points_native


def witness():
    return dict(r1=np.array([0.,0.,1.]),r2=np.array([.5,0.,1.]),
        E1=np.array([[1.,0.],[0.,1.],[0.,0.]]),
        E2=np.array([[1.,0.],[0.,1.],[0.,0.]]),
        d1=1.,d2=1.,g1=np.array([1.,0.]),g2=np.array([2.,8.]),
        A=np.array([[.5,-1.],[0.,1./3.]]))


def solution(R=None,t=None,a=1.,b=0.,shift2=0.):
    return SimpleNamespace(pose=SimpleNamespace(
        R=np.eye(3) if R is None else R,
        t=np.array([.1,.2,.3]) if t is None else t),
        scale=a,shift1=b,shift2=shift2)


def native_result(values,depth2=1.):
    rays=np.array([[0.,0.,1.],[.1,0.,1.],[0.,.1,1.]])
    native=SimpleNamespace(monodepth_pose_3pt=lambda *args:values)
    return solve_points_native(rays,rays,np.ones(3),np.full(3,depth2),native)


class ControlTests(unittest.TestCase):
    def test_scale_only_exact_axial_witness(self):
        for solve in (solve_scale_only,solve_scale_only_ordered):
            with self.subTest(method=solve.__name__):
                result=solve(**witness())
                self.assertEqual(result['status'],'completed')
                c=result['candidates'][0]
                R=np.array([[2.,-2.,1.],[2.,1.,-2.],[1.,2.,2.]])/3.
                np.testing.assert_allclose(c['R'],R,rtol=0.,atol=1e-12)
                np.testing.assert_allclose(c['t'],[1./6.,2./3.,1./3.],rtol=0.,atol=1e-12)
                self.assertAlmostEqual(c['a'],1.,places=12)
                self.assertTrue(c['pose_usable'])
                self.assertLess(c['tangent_squared_residual'],1e-24)
                json.dumps(result,allow_nan=False)

    def test_ordered_basis_explicitly_privileges_second_column(self):
        data=witness()
        data.update(r1=np.array([0.,0.,1.]),r2=np.array([0.,0.,1.]),
                    E1=np.array([[2.,0.],[0.,1.],[0.,0.]]),
                    E2=np.array([[1.,.5],[0.,1.],[0.,0.]]),
                    g1=np.zeros(2),g2=np.zeros(2),A=np.eye(2))
        c=solve_scale_only_ordered(**data)['candidates'][0]
        expected=np.array([[2./np.sqrt(5.),1./np.sqrt(5.),0.],
                           [-1./np.sqrt(5.),2./np.sqrt(5.),0.],[0.,0.,1.]])
        np.testing.assert_allclose(c['R'],expected,rtol=0.,atol=1e-12)

    def test_raw_scale_gauge_preserves_rotation_and_direction(self):
        for solve in (solve_scale_only,solve_scale_only_ordered):
            data=witness();base=solve(**data)['candidates'][0]
            data.update(d1=3.,d2=7.,g1=3.*data['g1'],g2=7.*data['g2'])
            c=solve(**data)['candidates'][0]
            np.testing.assert_allclose(c['R'],base['R'],rtol=0.,atol=1e-12)
            np.testing.assert_allclose(c['t'],3.*np.array(base['t']),rtol=0.,atol=1e-12)
            self.assertAlmostEqual(c['a'],3./7.,places=12)

    def test_input_contract_and_scale_only_nonpositive_depth(self):
        for change in [dict(r1=[0.,0.,2.]),dict(E1=np.ones((3,2))),
                       dict(A=np.eye(2,dtype=complex)),dict(d1=True)]:
            data=witness();data.update(change)
            with self.subTest(change=list(change)),self.assertRaises(ValueError):
                solve_scale_only(**data)
        data=witness();data['d1']=-1.
        self.assertEqual(solve_scale_only(**data)['reason'],'nonpositive_raw_depth')
        rays=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
        with self.assertRaises(ValueError):
            solve_points_native(rays.astype(complex),rays,np.ones(3),np.ones(3),
                                SimpleNamespace(monodepth_pose_3pt=lambda *a:[]))

    def test_native_zero_roots_and_unexpected_api_error(self):
        result=native_result([])
        self.assertEqual(result['native_returned'],0)
        self.assertEqual(result['usable_count'],0)
        rays=np.array([[0.,0.,1.],[.1,0.,1.],[0.,.1,1.]])
        def broken(*args):
            raise RuntimeError('fake API exception')
        with self.assertRaisesRegex(RuntimeError,'fake API exception'):
            solve_points_native(rays,rays,np.ones(3),np.ones(3),
                                SimpleNamespace(monodepth_pose_3pt=broken))

    def test_native_finite_nonphysical_and_improper_outputs_retained(self):
        result=native_result([solution(R=np.diag([-1.,1.,1.])),
                              solution(a=-1.),solution(t=np.zeros(3))])
        self.assertEqual(result['native_returned'],3)
        self.assertEqual(result['usable_count'],0)
        c=result['candidates']
        self.assertFalse(c[0]['proper_rotation'])
        self.assertFalse(c[1]['physical'])
        self.assertFalse(c[2]['translation_direction_defined'])
        json.dumps(result,allow_nan=False)

    def test_native_raw_nonfinite_and_derived_overflow_null_serialized(self):
        bad=np.eye(3);bad[0,0]=float('nan')
        result=native_result([solution(R=bad),solution(a=1e308)],depth2=2.)
        self.assertEqual(result['native_returned'],2)
        self.assertEqual(result['usable_count'],0)
        self.assertIsNone(result['candidates'][0]['R'][0][0])
        derived=result['candidates'][1]
        self.assertEqual(derived['status'],'invalid')
        self.assertEqual(derived['reason'],'nonfinite_or_unresolved_derived_arithmetic')
        self.assertEqual(derived['target_depths'],[None,None,None])
        json.dumps(result,allow_nan=False)

    def test_scale_only_overflow_is_explicit_numerical_failure(self):
        data=witness();data['d1']=1e308
        result=solve_scale_only(**data)
        self.assertEqual(result['status'],'numerical_failure')
        self.assertEqual(result['candidates'],[])
        json.dumps(result,allow_nan=False)

    def test_tiny_nonzero_translation_is_not_squared_norm_zero(self):
        data=witness();data.update(r2=np.array([1e-300,0.,1.]),
                                   g1=np.zeros(2),g2=np.zeros(2),A=np.eye(2))
        c=solve_scale_only(**data)['candidates'][0]
        self.assertTrue(c['translation_direction_defined'])
        self.assertTrue(c['pose_usable'])
        self.assertEqual(c['t'],[1e-300,0.,0.])
        result=native_result([solution(t=np.array([1e-300,0.,0.]))])
        self.assertEqual(result['usable_count'],1)
        json.dumps(result,allow_nan=False)


if __name__=='__main__':
    unittest.main()
