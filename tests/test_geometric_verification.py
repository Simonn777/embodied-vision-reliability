"""Only hand-specified 2-5 point fixtures and the required fixed metric grid."""
import math
import unittest

import numpy as np

from evr.geometric_verification import (project, matched_residuals, pairwise_radius_A,
                           marginal_cell_counts, score_hypothesis)
from evr.geometry_evaluation import evaluation_grid, evaluate_transfer


class GeometryTests(unittest.TestCase):
    def test_identity(self):
        xy = np.array([[0, 0], [512, 512]], dtype=np.float64)
        np.testing.assert_array_equal(project(np.eye(3), xy), xy)

    def test_translation(self):
        xy = np.array([[0, 0], [2, 3], [100, 200]], dtype=np.float64)
        h = np.array([[1, 0, 3], [0, 1, -4], [0, 0, 1]])
        np.testing.assert_allclose(project(h, xy), [[3,-4], [5,-1], [103,196]])

    def test_perspective_and_global_scale(self):
        xy = np.array([[0,0], [2,0], [0,2]], dtype=np.float64)
        h = np.array([[1,0,0], [0,1,0], [.5,0,1]])
        expected = np.array([[0,0], [1,0], [0,2]])
        for scale in (1., -7., 1e120, 1e-120):
            np.testing.assert_allclose(project(scale*h, xy), expected, rtol=1e-14, atol=0)

    def test_horizon_and_near_cancellation_rejected(self):
        h = [[1,0,0], [0,1,0], [1,0,-1]]
        for x in (1., 1.+1e-14):
            with self.assertRaisesRegex(ValueError, 'horizon'):
                project(h, [[x,0], [2,1]])
        self.assertTrue(np.isfinite(project(h, [[1+1e-7,0], [2,1]])).all())

    def test_projection_validation(self):
        points = [[0.,0.], [1.,1.]]
        for h in (np.zeros((3,3)), np.eye(2), [[1,0,0],[0,1,0],[0,0,np.inf]]):
            with self.assertRaises(ValueError):
                project(h, points)
        for xy in ([[0,np.nan],[1,1]], [1,2], [[True,False],[False,True]], [['0','1'],['2','3']]):
            with self.assertRaises(ValueError):
                project(np.eye(3), xy)

    def test_pairwise_radius_manual_rectangular_board_inclusive_boundary(self):
        x = [[0,0], [3,0]]
        y = [[0,0], [1,0], [3,1]]
        np.testing.assert_array_equal(pairwise_radius_A(x,y,1), [[1,1,0],[0,0,1]])

    def test_diagonal_residual_uses_euclidean_distance(self):
        actual = matched_residuals([[0,0],[10,10]], [[3,4],[10,12]])
        np.testing.assert_array_equal(actual,[5,2])

    def test_analytic_scores_and_radius_boundary(self):
        x = [[0,0],[10,0],[20,0]]
        y = [[0,0],[11,0],[20,2]]
        out = score_hypothesis(np.eye(3),x,y,2)
        np.testing.assert_array_equal(out.compatibility,np.eye(3,dtype=bool))
        np.testing.assert_array_equal(out.matched_residuals,[0,1,2])
        self.assertEqual(out.raw_count,3)
        self.assertEqual(out.msac,1.75)
        self.assertEqual(out.cell_weighted,1)
        self.assertEqual(out.null_mean,1)
        self.assertEqual(out.excess,2)
        self.assertEqual(out.ratio,3)

    def test_off_diagonal_compatibility_changes_null_not_raw_count(self):
        out = score_hypothesis(np.eye(3),[[0,0],[10,0]],[[10,0],[0,0]],1)
        np.testing.assert_array_equal(out.compatibility,[[0,1],[1,0]])
        self.assertEqual((out.raw_count,out.msac,out.cell_weighted),(0,0,0))
        self.assertEqual((out.null_mean,out.excess,out.ratio),(1,-1,0))

    def test_zero_null_mean_and_tiny_tau_are_finite(self):
        out = score_hypothesis(np.eye(3),[[0,0],[10,0]],[[100,100],[110,110]],1e-300)
        self.assertFalse(out.compatibility.any())
        self.assertEqual((out.raw_count,out.msac,out.cell_weighted,out.null_mean,out.excess,out.ratio),
                         (0,0,0,0,0,0))

    def test_floor_clip_cell_boundary_counts(self):
        xy = [[-1,0],[0,63.999],[64,64],[512,512],[1e10,-1]]
        np.testing.assert_array_equal(marginal_cell_counts(xy),[2,2,1,1,1])

    def test_source_density_is_not_computed_after_h(self):
        x = [[63,0],[65,0]]
        h = [[1,0,1],[0,1,0],[0,0,1]]
        out = score_hypothesis(h,x,x,2)
        np.testing.assert_array_equal(out.source_cell_counts,[1,1])
        np.testing.assert_array_equal(out.target_cell_counts,[1,1])
        self.assertEqual(out.cell_weighted,2)
        np.testing.assert_array_equal(marginal_cell_counts(project(h,x)),[2,2])

    def test_source_and_target_marginals_both_used(self):
        out = score_hypothesis(np.eye(3),[[10,0],[20,0],[70,0]],[[10,0],[70,0],[80,0]],100)
        np.testing.assert_array_equal(out.source_cell_counts,[2,2,1])
        np.testing.assert_array_equal(out.target_cell_counts,[1,2,2])
        self.assertAlmostEqual(out.cell_weighted,0.5+math.sqrt(2))

    def test_verification_and_tau_validation(self):
        xy = [[0,0],[1,1]]
        for tau in (0,-1,np.nan,np.inf,True,[1]):
            with self.assertRaises(ValueError):
                score_hypothesis(np.eye(3),xy,xy,tau)
        with self.assertRaises(ValueError):
            score_hypothesis(np.eye(3),xy,xy+[[2,2]],1)
        with self.assertRaises(ValueError):
            score_hypothesis(np.eye(3),np.empty((0,2)),np.empty((0,2)),1)
        with self.assertRaises(ValueError):
            matched_residuals(xy,xy+[[2,2]])

    def test_derived_nonfinite_distances_rejected(self):
        x, y = [[1e308,0],[0,0]], [[-1e308,0],[0,0]]
        with self.assertRaises(OverflowError):
            matched_residuals(x,y)
        with self.assertRaises(OverflowError):
            pairwise_radius_A(x,y,1)

    def test_fixed_grid_identity_and_translation_metric(self):
        grid = evaluation_grid()
        self.assertEqual(grid.shape,(81,2))
        np.testing.assert_array_equal(grid[0],[0,0])
        np.testing.assert_array_equal(grid[-1],[512,512])
        self.assertEqual(len(np.unique(grid,axis=0)),81)
        self.assertEqual(evaluate_transfer(np.eye(3),np.eye(3)),0)
        translated = [[1,0,3],[0,1,-4],[0,0,1]]
        self.assertAlmostEqual(evaluate_transfer(translated,np.eye(3)),5)

    def test_fixed_grid_perspective_metric_independent_scalar_formula(self):
        h = [[1,0,0],[0,1,0],[1/512,0,1]]
        distances = []
        for y in range(0,513,64):
            for x in range(0,513,64):
                d = 1+x/512
                distances.append(math.hypot(x/d-x,y/d-y))
        self.assertAlmostEqual(evaluate_transfer(h,np.eye(3)),sum(distances)/81)

    def test_fixed_grid_horizon_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError,'horizon'):
            evaluate_transfer([[1,0,0],[0,1,0],[1,0,-256]],np.eye(3))

    def test_inputs_unchanged_and_outputs_float64(self):
        h, x = np.eye(3), np.array([[0.,0.],[64.,64.]])
        y = x.copy()
        out = score_hypothesis(h,x,y,1)
        self.assertEqual(project(h,x).dtype,np.dtype('float64'))
        self.assertEqual(out.matched_residuals.dtype,np.dtype('float64'))
        np.testing.assert_array_equal(h,np.eye(3))
        np.testing.assert_array_equal(x,y)


if __name__ == '__main__':
    unittest.main(verbosity=2)
