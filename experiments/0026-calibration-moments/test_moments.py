"""Small manufactured arithmetic checks only; no estimator or scientific data."""
import dataclasses
import unittest
from unittest.mock import patch
import numpy as np
import moments as m


def fixture():
    rng = np.random.default_rng(260026)
    r1 = np.column_stack((rng.uniform(-.5, .5, (11, 2)), np.ones(11)))
    r2 = np.column_stack((rng.uniform(-.4, .6, (11, 2)), np.ones(11)))
    d1, d2 = rng.uniform(1., 4., 11), rng.uniform(.8, 3.7, 11)
    w = np.array((1., .3, 0., 2., .8, 1., .1, 0., .6, 1.1, .4))
    return r1, r2, d1, d2, w


def edges_reference(data, theta):
    r1, r2, d1, d2, w = data
    b, a, c = theta
    H = np.zeros((6, 6)); cost = 0.; grad = np.zeros(3); hess = np.zeros((3, 3))
    for i in range(len(w)):
        for j in range(i):
            U = d1[i]*r1[i] - d1[j]*r1[j]
            P = r1[i] - r1[j]
            V = d2[i]*r2[i] - d2[j]*r2[j]
            Q = r2[i] - r2[j]
            coeff = np.array((U@U, 2*(U@P), P@P, -(V@V), -2*(V@Q), -(Q@Q)))
            H += w[i]*w[j]*np.outer(coeff, coeff)
            X, Y = U+b*P, a*V+c*Q
            e = X@X-Y@Y
            de = np.array((2*(X@P), -2*(Y@V), -2*(Y@Q)))
            he = np.array(((2*(P@P), 0., 0.), (0., -2*(V@V), -2*(V@Q)),
                           (0., -2*(V@Q), -2*(Q@Q))))
            cost += w[i]*w[j]*e*e
            grad += 2*w[i]*w[j]*e*de
            hess += 2*w[i]*w[j]*(np.outer(de, de)+e*he)
    return H, cost, grad, hess


def point_reference(data, theta):
    r1, r2, d1, d2, w = data
    b, a, c = theta
    X, Y = (d1+b)[:, None]*r1, (a*d2+c)[:, None]*r2
    mx, my = np.average(X, axis=0, weights=w), np.average(Y, axis=0, weights=w)
    S = sum(wi*np.outer(y-my, x-mx) for x, y, wi in zip(X, Y, w))
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3); D[2, 2] = np.linalg.det(U@Vt)
    R = U@D@Vt
    t = my-R@mx
    cost = sum(wi*np.sum((y-R@x-t)**2) for x, y, wi in zip(X, Y, w))
    return cost, R, t


class TestMoments(unittest.TestCase):
    def test_H_cost_gradient_hessian_against_explicit_edges(self):
        data = fixture(); moments = m.build_moments(*data)
        for theta in ((.2, 1.3, -.1), (-.6, .7, .8), (0., 0., 0.)):
            H, cost, g, h = edges_reference(data, theta)
            actual = m.distance_objective(moments, *theta)
            np.testing.assert_allclose(moments.H, H, rtol=2e-13, atol=2e-12)
            np.testing.assert_allclose(actual["cost_raw"], cost, rtol=2e-13, atol=2e-12)
            np.testing.assert_allclose(actual["gradient"], g, rtol=3e-13, atol=4e-12)
            np.testing.assert_allclose(actual["hessian"], h, rtol=3e-13, atol=4e-12)

    def test_profile_against_explicit_corrected_points(self):
        data = fixture(); moments = m.build_moments(*data)
        for theta in ((.2, 1.3, -.1), (-.6, .7, .8)):
            cost, R, t = point_reference(data, theta)
            actual = m.procrustes_objective(moments, *theta)
            np.testing.assert_allclose(actual["cost_raw"], cost, rtol=2e-13, atol=1e-13)
            np.testing.assert_allclose(actual["R"], R, atol=2e-13)
            np.testing.assert_allclose(actual["t"], t, atol=2e-13)
            self.assertAlmostEqual(np.linalg.det(actual["R"]), 1., places=13)

    def test_positive_per_image_affine_unit_change(self):
        r1, r2, d1, d2, w = fixture()
        theta = (.2, 1.3, -.1); b, a, c = theta
        s1, t1, s2, t2 = 2.3, -.7, .6, .2
        transformed = (s1*b-t1, s1*a/s2, s1*c-s1*a*t2/s2)
        old = m.build_moments(r1, r2, d1, d2, w)
        new = m.build_moments(r1, r2, s1*d1+t1, s2*d2+t2, w)
        D, D2 = m.distance_objective(old, *theta), m.distance_objective(new, *transformed)
        P, P2 = m.procrustes_objective(old, *theta), m.procrustes_objective(new, *transformed)
        np.testing.assert_allclose(D2["cost_raw"], s1**4*D["cost_raw"], rtol=1e-12)
        np.testing.assert_allclose(P2["cost_raw"], s1**2*P["cost_raw"], rtol=1e-12)
        np.testing.assert_allclose(P2["R"], P["R"], atol=2e-13)
        np.testing.assert_allclose(P2["t"], s1*P["t"], atol=3e-13)

    def test_duplicate_vertex_with_split_weight_is_same_measure(self):
        data = fixture(); old = m.build_moments(*data)
        split = [np.concatenate((v, v[:1]), axis=0) for v in data]
        split[-1][0], split[-1][-1] = data[-1][0]*.25, data[-1][0]*.75
        new = m.build_moments(*split)
        np.testing.assert_allclose(new.H, old.H, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(new.C, old.C, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(new.mean, old.mean, rtol=2e-13, atol=2e-13)

    def test_zero_weights_support_and_single_point(self):
        data = fixture(); result = m.build_moments(*data)
        keep = data[-1] > 0
        reduced = m.build_moments(*(v[keep] for v in data))
        self.assertEqual(result.input_count, 11); self.assertEqual(result.support_count, 9)
        np.testing.assert_array_equal(result.H, reduced.H)
        huge = [v.copy() for v in data]
        huge[0][2] = 1e300; huge[2][2] = 1e300
        np.testing.assert_array_equal(m.build_moments(*huge).H, result.H)
        single = m.build_moments(*(v[:1] for v in data))
        np.testing.assert_array_equal(single.C, np.zeros((12, 12)))
        self.assertEqual(m.distance_objective(single, .2, 1.3, -.1)["cost_raw"], 0.)
        self.assertEqual(m.procrustes_objective(single, .2, 1.3, -.1)["cost_raw"], 0.)

    def test_reflection_cannot_be_zero_cost_proper_rotation(self):
        X = np.array(((0., 0., 2.), (1., 0., 2.), (0., 1., 2.),
                      (.2, .3, 3.), (-.4, .2, 2.5)))
        Y = X*np.array((-1., 1., 1.))
        moments = m.build_moments(X/X[:, 2, None], Y/Y[:, 2, None], X[:, 2], Y[:, 2])
        distance = m.distance_objective(moments, 0., 1., 0.)
        point = m.procrustes_objective(moments, 0., 1., 0.)
        self.assertLess(abs(distance["cost_raw"]), 1e-12)
        self.assertGreater(point["cost_raw"], .1)
        self.assertLess(abs(point["orthogonal_cost_raw"]), 1e-12)
        self.assertEqual(point["determinant_correction"], -1.)
        self.assertAlmostEqual(np.linalg.det(point["R"]), 1., places=13)

    def test_signed_cost_not_silently_clamped(self):
        moments = m.build_moments(*fixture())
        # Inject a tiny signed moment perturbation solely to test reporting;
        # this is not claimed to be a physically realizable negative objective.
        H = np.zeros((6, 6)); H[0, 0] = -np.finfo(float).eps
        perturbed = dataclasses.replace(moments, H=H)
        self.assertEqual(m.distance_objective(perturbed, 0., 1., 0.)["cost_raw"], H[0, 0])
        r1, _, d1, _, w = fixture()
        same = m.build_moments(r1, r1, d1, d1, w)
        original_svd = np.linalg.svd
        def rounded_svd(*args, **kwargs):
            U, s, Vt = original_svd(*args, **kwargs)
            return U, s*(1.+8*np.finfo(float).eps), Vt
        with patch.object(m.np.linalg, "svd", side_effect=rounded_svd):
            point = m.procrustes_objective(same, 0., 1., 0.)
        self.assertLess(point["cost_raw"], 0.)
        self.assertEqual(point["cost_raw"], point["trace_sum"]-2*point["proper_trace"])

    def test_validation_and_true_overflow(self):
        data = fixture()
        cases = []
        for index, value in ((0, np.ones((11, 2))), (2, np.ones((11, 1))),
                             (4, -np.ones(11)), (4, np.zeros(11))):
            bad = list(data); bad[index] = value; cases.append(bad)
        bad = [v.copy() for v in data]; bad[2][2] = np.nan; cases.append(bad)
        bad = [v.copy() for v in data]; bad[0][0] = 0.; cases.append(bad)
        bad = list(data); bad[2] = np.ones(11, dtype=complex); cases.append(bad)
        for bad in cases:
            with self.subTest(shapes=[np.shape(x) for x in bad]):
                with self.assertRaises(ValueError): m.build_moments(*bad)
        huge = [v.copy() for v in data]; huge[0][0] = 1e200; huge[2][0] = 1e200
        with self.assertRaises(FloatingPointError): m.build_moments(*huge)
        moments = m.build_moments(*data)
        for fn in (m.distance_objective, m.procrustes_objective):
            with self.assertRaises(ValueError): fn(moments, np.nan, 1., 0.)
            with self.assertRaises((FloatingPointError, ValueError)): fn(moments, 1e300, 1e300, 1e300)

    def test_stored_moments_readonly(self):
        moments = m.build_moments(*fixture())
        for array in (moments.H, moments.C, moments.mean):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError): array.flat[0] = 1.


if __name__ == "__main__":
    unittest.main(verbosity=2)
