"""Manufactured query/prior contracts; no real inputs or native pose estimator."""
import unittest
from unittest import mock
import numpy as np
import conditional as c
import field_ba as f
from test_field_ba import fixture


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        images, edges, poses = fixture()
        self.query = edges[0]
        problem = f.FieldProblem(images, edges[1:], poses)
        self.ref = dict(problem=problem, state=problem.initial)

    def test_alternative_preserves_common_scale_and_all_field_values(self):
        r = f.rotation_exp([.2, -.1, .02]); t = np.array([3., 1., .1])
        w = c.alternative_chart(self.ref, "B", (r, t))
        problem = self.ref["problem"]; d = np.zeros(problem.dimension); d[problem.query_columns("B")] = w
        state = problem.retract(self.ref["state"], d)
        np.testing.assert_allclose(state["poses"]["B"][0], r, atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(state["poses"]["B"][1]), np.linalg.norm(self.ref["state"]["poses"]["B"][1]))
        for key in state["theta"]: np.testing.assert_array_equal(state["theta"][key], self.ref["state"]["theta"][key])

    def test_one_prior_one_query_and_both_starts_retained(self):
        factor = np.eye(17); calls = []
        def fake_solver(evaluate, x0, **kw):
            residual, jac = evaluate(x0); calls.append((residual.copy(), jac.copy()))
            qr, qj = f.query_residual(self.ref, self.query, x0)
            np.testing.assert_allclose(residual, np.r_[qr, x0])
            np.testing.assert_allclose(jac, np.vstack((qj, factor)))
            return dict(x=x0.copy(), residual=residual, converged=True,
                        termination_reason="manufactured", accepted_steps=0, evaluations=1, seconds=0.)
        with mock.patch.object(f, "solve_trf", side_effect=fake_solver):
            result = c.fit_conditional(self.ref, self.query, factor,
                                      self.ref["state"]["poses"]["B"])
        self.assertEqual(len(calls), 2); self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["information_count"], dict(reference_prior=1, query_edge=1))
        self.assertTrue(result["observability"]["observable"])

    def test_real_small_query_solver_and_null_precision(self):
        result = c.fit_conditional(self.ref, self.query, np.zeros((0, 17)), options=dict(max_evaluations=3))
        self.assertTrue(np.isfinite(result["x"]).all())
        self.assertLess(result["attempts"][0]["objective"], 1e-20)
        self.assertEqual(result["information_count"]["reference_prior"], 1)

    def test_invalid_rotation_and_singular_chart_are_not_projected_into_success(self):
        with self.assertRaises(ValueError):
            c.alternative_chart(self.ref, "B", (np.diag([1., 1., 2.]), np.ones(3)))
        w=np.zeros(17); w[0]=2*np.pi
        result=c.pose_observability(self.ref,self.query,w,np.eye(17))
        self.assertFalse(result["observable"])
        self.assertEqual(result["reason"], "singular fixed rotation chart")

    def test_large_translation_cannot_hide_unobservable_direction(self):
        from scipy.sparse import csr_matrix
        problem=self.ref['problem']
        self.ref['state']['poses']['B']=(np.eye(3),np.array([0.,0.,1e12]))
        factor=np.delete(np.eye(17),3,axis=0)
        empty_j=csr_matrix((0,problem.dimension))
        with mock.patch.object(problem,'local',return_value=(None,None,empty_j,None)):
            result=c.pose_observability(self.ref,self.query,np.zeros(17),factor)
        self.assertFalse(result['observable'])
        self.assertGreater(result['direction_null_ratio'],.1)
        self.assertEqual(result['rotation_null_ratio'],0.)

    def test_empty_query_keeps_reference_prior_and_has_no_fake_measurements(self):
        empty=dict(self.query,ids_i=np.array([],dtype=int),ids_j=np.array([],dtype=int))
        result=c.fit_conditional(self.ref,empty,np.eye(17),options=dict(max_evaluations=3))
        self.assertEqual(result["attempts"][0]["objective"],0.)
        self.assertTrue(result["observability"]["observable"])
        failed=c.fit_conditional(self.ref,empty,np.zeros((0,17)),options=dict(max_evaluations=3))
        self.assertFalse(failed["observability"]["observable"])
        joint=c.fit_joint(self.ref,empty,options=dict(max_evaluations=3))
        self.assertEqual(len(joint["attempts"][0]["problem"].edges),len(self.ref["problem"].edges))


if __name__ == "__main__": unittest.main()
