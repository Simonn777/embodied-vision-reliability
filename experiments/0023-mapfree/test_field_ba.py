"""Fixed manufactured geometry only; no scientific assets/native estimator."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import numpy as np

spec = importlib.util.spec_from_file_location("_evr0023_field_test", Path(__file__).with_name("field_ba.py"))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def fixture():
    # One fixed nonplanar configuration, no seed or fixture search.
    k = np.array([[140., 0., 127.5], [0., 135., 95.5], [0., 0., 1.]])
    z = np.linspace(.85, 1.65, 24)
    points = np.column_stack((.21*np.sin(np.arange(24)*1.31),
                              .17*np.cos(np.arange(24)*.71), z))
    poses = {"A0": (np.eye(3), np.zeros(3)),
             "B": (f.rotation_exp([.025, -.04, .015]), np.array([.14, -.025, .015])),
             "C": (f.rotation_exp([-.02, .03, -.01]), np.array([-.08, .03, .01]))}
    images = {}
    for key, (r, t) in poses.items():
        p = np.einsum("ij,nj->ni", r, points)+t
        xy = p[:, :2]/p[:, 2, None]*[k[0, 0], k[1, 1]]+[k[0, 2], k[1, 2]]
        images[key] = dict(K=k.copy(), width=256, height=192, xy=xy,
                           depth=p[:, 2].copy(), theta=np.array([1., 0, 0, 0, 0, 0]))
    edges = [dict(i=i, j=j, ids_i=np.arange(24), ids_j=np.arange(24))
             for i, j in [("A0", "B"), ("B", "C"), ("A0", "C")]]
    return images, edges, poses


def finite_difference(fun, x, eps=1e-6):
    result = []
    for j in range(len(x)):
        a = x.copy(); b = x.copy(); a[j] += eps; b[j] -= eps
        result.append((fun(a)-fun(b))/(2*eps))
    return np.column_stack(result)


class FieldTests(unittest.TestCase):
    def test_fixed_capped_loss_and_exact_derivative(self):
        errors = np.array([[0., 0.], [.001, -.002], [3., 4.], [30., -20.], [np.nan, np.inf]])
        valid = np.array([True, True, True, True, False])
        result = f.robust_blocks(errors, valid)
        expected = [np.log1p(min(float(x*x+y*y), 256)/4) if ok else np.log(65.)
                    for (x, y), ok in zip(errors, valid)]
        np.testing.assert_allclose(np.sum(result["residual"]**2, axis=1), expected, atol=1e-14)
        for idx in [1, 2, 3]:
            numerical = finite_difference(lambda x:f.robust_blocks(x[None], np.array([True]))["residual"][0], errors[idx])
            np.testing.assert_allclose(result["transform"][idx], numerical, atol=2e-9, rtol=2e-7)
        self.assertEqual(result["sqrt_weight"][3], 0); self.assertEqual(result["sqrt_weight"][4], 0)
        np.testing.assert_array_equal(result["transform"][4], 0)
        self.assertEqual(result["squared_pixel_errors"][4], 256.)
        self.assertEqual(result["invalid_count"], 1)

    def test_analytic_full_chart_against_independent_differences(self):
        images, edges, poses = fixture(); problem = f.FieldProblem(images, edges, poses)
        delta = .003*np.sin(np.arange(problem.dimension)*.7)
        for key in ("B", "C"):
            delta[problem.layout[key]["pose"][:3]] = [.025, -.015, .01]
        r, j = problem.evaluate(delta)
        numerical = finite_difference(lambda x:problem.evaluate(x)[0], delta)
        self.assertTrue(np.isfinite(r).all())
        np.testing.assert_allclose(j.toarray(), numerical, rtol=2e-5, atol=4e-7)

    def test_projection_and_same_camera_point_indices_match_scalar_formula(self):
        images, edges, poses = fixture(); problem = f.FieldProblem(images, edges, poses)
        theta = problem.initial["theta"]["A0"]
        theta[1:] = [.02, -.03, .01, -.005, .004]
        r, _, _, meta = problem.local(problem.initial)
        expected = []
        for e in edges:
            for i, j in [(e["i"], e["j"]), (e["j"], e["i"])]:
                for index in range(24):
                    image = images[i]; xy = image["xy"][index]; d = image["depth"][index]
                    u = 2*(xy[0]+.5)/image["width"]-1; v = 2*(xy[1]+.5)/image["height"]-1
                    depth = np.dot([d, u*d, v*d, 1., u, v], problem.initial["theta"][i])
                    ray = np.linalg.solve(image["K"], [*xy, 1.])
                    ri, ti = poses[i]; rj, tj = poses[j]
                    p = rj @ np.linalg.solve(ri, depth*ray-ti)+tj
                    q = images[j]["K"] @ p; error = q[:2]/q[2]-images[j]["xy"][index]
                    expected.append(float(error @ error))
        np.testing.assert_allclose(meta["squared_pixel_errors"], expected, atol=1e-12)
        np.testing.assert_allclose(float(r@r), sum(np.log1p(min(x, 256)/4) for x in expected), atol=1e-12)

    def test_scale_gauge_is_real_and_anchor_removes_it(self):
        images, edges, poses = fixture(); problem = f.FieldProblem(images, edges, poses)
        reference = problem.local(problem.initial)[0]
        state = {"poses": {k:(r.copy(), 2.3*t) for k,(r,t) in poses.items()},
                 "theta": {k:2.3*v.copy() for k,v in problem.initial["theta"].items()}}
        np.testing.assert_allclose(problem.local(state)[0], reference, atol=1e-12)
        images["A0"]["theta"][0] = 2.3
        with self.assertRaisesRegex(ValueError, "anchor scale"):
            f.FieldProblem(images, edges, poses)

    def test_local_pose_log_roundtrip_and_query_seventeen_columns(self):
        images, edges, poses = fixture(); problem = f.FieldProblem(images, edges, poses)
        base = {"problem": problem, "state": problem.initial}
        keep = problem.query_columns("B")
        self.assertEqual(len(keep), 17)
        w = .002*np.cos(np.arange(17)); w[:3] = [.15, -.06, .09]
        delta = np.zeros(problem.dimension); delta[keep] = w
        other = {"state": problem.retract(problem.initial, delta)}
        np.testing.assert_allclose(f.state_delta(base, other)[keep], w, atol=1e-14)
        # Near-pi absolute orientations do not require subtracting rotvecs.
        rb = f.rotation_exp([3.13, 0., 0.]); ro = f.rotation_exp([.02, -.01, .015]) @ rb
        np.testing.assert_allclose(f.rotation_log(ro @ rb.T), [.02, -.01, .015], atol=1e-14)
        r, j = f.query_residual(base, edges[0], w)
        numerical = finite_difference(lambda x:f.query_residual(base, edges[0], x)[0], w)
        np.testing.assert_allclose(j, numerical, atol=4e-7, rtol=3e-5)

    def test_invalid_depth_stays_capped_with_zero_information(self):
        images, edges, poses = fixture(); problem = f.FieldProblem(images, edges, poses)
        problem.initial["theta"]["A0"][3] = -10.
        _, _, information, meta = problem.local(problem.initial)
        self.assertEqual(meta["invalid_count"], 48)
        np.testing.assert_array_equal(meta["squared_pixel_errors"][:24], 256.)
        np.testing.assert_array_equal(information[:48].toarray(), 0.)

    def test_flagged_missing_depth_preserves_all_matches_and_reverse_direction(self):
        images, edges, poses = fixture()
        images["A0"]["depth_valid"] = np.ones(24, dtype=bool)
        images["A0"]["depth_valid"][[0, 7]] = False
        images["A0"]["depth"][[0, 7]] = [np.nan, -np.inf]
        problem = f.FieldProblem(images, edges, poses)
        r, j, information, meta = problem.local(problem.initial)
        self.assertEqual(len(r), 3*2*24*2); self.assertEqual(meta["invalid_count"], 4)
        self.assertTrue(np.isfinite(r).all()); self.assertTrue(np.isfinite(j.data).all())
        for block in [0, 7, 96, 103]:
            self.assertEqual(meta["squared_pixel_errors"][block], 256.)
            np.testing.assert_array_equal(information[2*block:2*block+2].toarray(), 0.)
        self.assertLess(meta["squared_pixel_errors"][24], 1e-20)  # reverse source B remains valid

    def test_finite_state_projection_overflow_is_full_cap_not_nan_jacobian(self):
        images, edges, poses = fixture()
        poses["B"] = (np.eye(3), np.array([1.2e308, 0., 0.]))
        poses["C"] = (np.eye(3), np.array([-1.2e308, 0., 0.]))
        problem = f.FieldProblem(images, [edges[1]], poses)
        # B->C and C->B overflow transformed X; all state entries are finite.
        with np.errstate(all="raise"):
            r, j, information, meta = problem.local(problem.initial)
        self.assertEqual(meta["invalid_count"], 48)
        np.testing.assert_array_equal(meta["squared_pixel_errors"], 256.)
        self.assertTrue(np.isfinite(r).all()); np.testing.assert_array_equal(j.toarray(), 0.)
        np.testing.assert_array_equal(information.toarray(), 0.)
        self.assertAlmostEqual(float(r@r), 48*f.CAP_COST)
        problem.initial["poses"]["B"][1][0] = np.inf
        with self.assertRaisesRegex(ValueError, "state translation"):
            problem.local(problem.initial)

    def test_solver_retains_last_accepted_not_rejected_trial(self):
        from scipy.sparse import csr_matrix
        def evaluate(x):
            return np.array([x[0]-1., 2*(x[0]-1.)]), csr_matrix([[1.], [2.]])
        def manufactured_scipy(fun, x0, jac, **kwargs):
            fun(x0); jac(x0)
            fun(np.array([.5])); jac(np.array([.5]))
            fun(np.array([20.]))  # rejected trial, next call hits budget
            fun(np.array([10.]))
        with mock.patch("scipy.optimize.least_squares", side_effect=manufactured_scipy):
            result = f.solve_trf(evaluate, np.array([0.]), max_evaluations=3)
        np.testing.assert_array_equal(result["x"], [.5])
        self.assertEqual(result["evaluations"], 3); self.assertEqual(result["accepted_steps"], 1)
        self.assertFalse(result["converged"]); self.assertEqual(result["termination_reason"], "evaluation_limit")

    def test_real_scipy_limit_and_manufactured_fit(self):
        from scipy.sparse import csr_matrix
        def evaluate(x):
            return np.array([x[0]*x[0]-1., .2*(x[0]-1.)]), csr_matrix([[2*x[0]], [.2]])
        result = f.solve_trf(evaluate, np.array([3.]), max_accepted=1)
        self.assertEqual(result["accepted_steps"], 1); self.assertFalse(result["converged"])
        self.assertEqual(result["termination_reason"], "accepted_step_limit")
        np.testing.assert_array_equal(result["residual"], evaluate(result["x"])[0])
        images, edges, poses = fixture()
        poses["B"] = (f.rotation_exp([.002, -.003, .001]) @ poses["B"][0], poses["B"][1]+[.002, .001, -.001])
        initial = f.FieldProblem(images, edges, poses).local(f.FieldProblem(images, edges, poses).initial)[0]
        fitted = f.fit_field(images, edges, poses, options={"max_accepted": 3, "max_evaluations": 10})
        self.assertLess(fitted["objective"], float(initial@initial))
        self.assertLessEqual(fitted["evaluations"], 10); self.assertLessEqual(fitted["accepted_steps"], 3)
        np.testing.assert_array_equal(fitted["poses"]["A0"][0], np.eye(3))
        np.testing.assert_array_equal(fitted["poses"]["A0"][1], np.zeros(3))
        self.assertEqual(fitted["theta"]["A0"][0], 1.)
        np.testing.assert_array_equal(f.jacobian_at(fitted).toarray(), fitted["robustJ"].toarray())

    def test_nonfinite_or_invalid_input_never_silently_repaired(self):
        images, edges, poses = fixture(); images["B"]["depth"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            f.FieldProblem(images, edges, poses)
        with self.assertRaisesRegex(ValueError, "finite"):
            f.robust_blocks(np.array([[np.inf, 0.]]), np.array([True]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
