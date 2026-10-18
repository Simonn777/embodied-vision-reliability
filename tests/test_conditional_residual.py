"""Artificial algebra/indexing tests only. No images, model or scientific scores."""
import math
import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import conditional_residual as cr


class ConditionalResidualTests(unittest.TestCase):
    def test_default_autocorrelation_normalization_and_dtype(self):
        one_d = {0: 1, 1: 2/3, 2: 1/6}
        for dtype in (np.float32, np.float64):
            c = cr.covariance_from_kernel(cr.default_kernel(dtype), sigma=2)
            self.assertEqual(c.dtype, np.dtype(dtype))
            expected = np.array([4*one_d.get(abs(y),0)*one_d.get(abs(x),0)
                                 for y,x in cr.OFFSETS])
            np.testing.assert_allclose(c[:,cr.CENTER], expected, atol=0,
                                       rtol=8*np.finfo(dtype).eps)
            self.assertAlmostEqual(float(c[24,24]),4,places=5)

    def test_asymmetric_kernel_preserves_signed_axis_lags(self):
        c = cr.covariance_from_kernel(np.array([[1,2,0],[0,0,1]],dtype=np.float64))
        index = {offset:i for i,offset in enumerate(cr.OFFSETS)}
        self.assertEqual(c[index[(1,1)],24],2)
        self.assertEqual(c[index[(1,-1)],24],0)
        self.assertEqual(c[24,24],6)  # No silent L2 normalization.

    def test_valid_border_shape_and_manual_patch_formula(self):
        for dtype in (np.float32,np.float64):
            y = np.arange(90,dtype=dtype).reshape(9,10)
            c = cr.covariance_from_kernel(cr.default_kernel(dtype))
            features=cr.residual_features(y,c);centers=cr.valid_centers(y.shape)
            self.assertEqual(features.shape,(12,48));self.assertEqual(features.dtype,y.dtype)
            self.assertEqual(centers[[0,-1]].tolist(),[[3,3],[5,6]])
            manual=np.array([[y[row+dy,col+dx] - c[j,24]/c[24,24]*y[row,col]
                              for j,(dy,dx) in zip(cr.NEIGHBORS,cr.FEATURE_OFFSETS)]
                             for row,col in centers],dtype=dtype)
            np.testing.assert_array_equal(features,manual)
            np.testing.assert_allclose(features,np.einsum('ij,kj->ik',cr.extract_valid_patches(y),cr.linear_operator(c),optimize=False),
                                       rtol=1e-6,atol=1e-5)

    def test_every_noncenter_impulse_and_center_subtraction_sign(self):
        c=cr.covariance_from_kernel(cr.default_kernel())
        for channel,(dy,dx) in enumerate(cr.FEATURE_OFFSETS):
            image=np.zeros((7,7));image[dy+3,dx+3]=1
            expected=np.zeros((1,48));expected[0,channel]=1
            np.testing.assert_array_equal(cr.residual_features(image,c),expected)
        image=np.zeros((7,7));image[3,3]=1
        np.testing.assert_array_equal(cr.residual_features(image,c)[0],-c[list(cr.NEIGHBORS),24]/c[24,24])

    def test_operator_crosscovariance_and_nonindependent_channels(self):
        for dtype in (np.float32,np.float64):
            c=cr.covariance_from_kernel(cr.default_kernel(dtype),sigma=3)
            stats=cr.analytic_statistics(c)
            tolerance=16*np.finfo(dtype).eps*float(c[24,24])
            np.testing.assert_allclose(stats['cross_covariance_with_center'],0,atol=tolerance,rtol=0)
            residual=stats['residual_noise_covariance']
            j=list(cr.NEIGHBORS)
            expected=c[np.ix_(j,j)]-np.outer(c[j,24],c[24,j])/c[24,24]
            np.testing.assert_allclose(residual,expected,atol=tolerance,rtol=0)
            self.assertGreater(np.max(np.abs(residual-np.diag(np.diag(residual)))),0.1)
            self.assertGreaterEqual(np.linalg.eigvalsh(residual.astype(float))[0],-tolerance)

    def test_wrong_center_sign_and_missing_variance_normalization_leak(self):
        c=cr.covariance_from_kernel(cr.default_kernel(),sigma=2)
        a=cr.linear_operator(c);j=cr.FEATURE_OFFSETS.index((0,1))
        wrong_sign=a.copy();wrong_sign[:,24]*=-1
        leak=cr.analytic_statistics(c,wrong_sign)['cross_covariance_with_center']
        self.assertAlmostEqual(float(leak[j]),16/3)
        wrong_gain=a.copy();wrong_gain[:,24]=-c[list(cr.NEIGHBORS),24]
        leak=cr.analytic_statistics(c,wrong_gain)['cross_covariance_with_center']
        self.assertAlmostEqual(float(leak[j]),-8)
        wrong_std=a.copy();wrong_std[:,24]=-c[list(cr.NEIGHBORS),24]/math.sqrt(c[24,24])
        leak=cr.analytic_statistics(c,wrong_std)['cross_covariance_with_center']
        self.assertAlmostEqual(float(leak[j]),-8/3)

    def test_spatial_mix_of_separately_centered_maps_reintroduces_leak(self):
        c=cr.covariance_from_kernel(cr.default_kernel())
        a=cr.linear_operator(c);j=cr.FEATURE_OFFSETS.index((0,1))
        self.assertAlmostEqual(cr.analytic_statistics(c)['cross_covariance_with_center'][j],0)
        # Same feature at center i+(0,1): n[i+(0,2)] - (2/3)n[i+(0,1)].
        # Neither raw term is n[i], yet Cov(feature,n[i])=1/6-4/9=-5/18.
        neighbor=np.zeros(49);neighbor[cr.OFFSETS.index((0,2))]=1
        neighbor[cr.OFFSETS.index((0,1))]=-2/3
        mixed=a.copy();mixed[j]=(a[j]+neighbor)/2  # Spatial [1/2,1/2] filter.
        self.assertAlmostEqual(float(np.sum(neighbor*c[:,24])),-5/18)
        self.assertAlmostEqual(cr.analytic_statistics(c,mixed)['cross_covariance_with_center'][j],-5/36)

    def test_bad_covariance_rejected_not_repaired(self):
        good=np.eye(49,dtype=np.float64)
        bad=[np.eye(48),good.astype(np.int64),np.zeros((49,49))]
        nan=good.copy();nan[0,0]=np.nan;bad.append(nan)
        negative=good.copy();negative[24,24]=-1;bad.append(negative)
        asym=good.copy();asym[0,1]=0.2;bad.append(asym)
        indefinite=good.copy();indefinite[0,0]=-1;bad.append(indefinite)
        for c in bad:
            with self.subTest(dtype=str(c.dtype),shape=c.shape):
                with self.assertRaises(ValueError):cr.linear_operator(c)

    def test_bad_images_kernel_sigma_and_dtype_mismatch(self):
        c=np.eye(49,dtype=np.float64)
        for y in (np.zeros((6,7)),np.zeros((7,6)),np.zeros((7,7,1)),
                  np.zeros((7,7),dtype=np.int32),np.full((7,7),np.inf)):
            with self.assertRaises(ValueError):cr.residual_features(y,c)
        with self.assertRaises(ValueError):cr.residual_features(np.zeros((7,7),np.float32),c)
        for sigma in (0,-1,np.nan,np.inf,True,[1]):
            with self.assertRaises(ValueError):cr.covariance_from_kernel(cr.default_kernel(),sigma)
        for kernel in (np.zeros((3,3)),np.ones(3),np.empty((0,3)),np.full((1,1),np.nan)):
            with self.assertRaises(ValueError):cr.covariance_from_kernel(kernel)

    def test_independent_noise_reduces_to_literal_blind_spot_patch(self):
        c=cr.covariance_from_kernel(np.ones((1,1),dtype=np.float64),sigma=5)
        image=np.arange(49,dtype=np.float64).reshape(7,7)
        np.testing.assert_array_equal(cr.residual_features(image,c),image.reshape(1,49)[:,cr.NEIGHBORS])
        np.testing.assert_array_equal(cr.analytic_statistics(c)['cross_covariance_with_center'],np.zeros(48))


if __name__=='__main__':unittest.main(verbosity=2)
