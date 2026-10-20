"""Prepared deterministic arithmetic tests; no data/model/random search."""
import json
import unittest
import numpy as np
from tangent_solver import solve_tangent, EPS


def witness():
    return dict(r1=np.array((0., 0., 1.)), r2=np.array((.5, 0., 1.)),
        E1=np.array(((1., 0.), (0., 1.), (0., 0.))),
        E2=np.array(((1., 0.), (0., 1.), (0., 0.))), d1=1., d2=1.,
        g1=np.array((1., 0.)), g2=np.array((2., 8.)),
        A=np.array(((.5, -1.), (0., 1./3.))))


class TangentSolverTests(unittest.TestCase):
    def test_exact_physical_witness_and_all_roots(self):
        result = solve_tangent(**witness())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["admissible_candidate_count"], 1)
        self.assertEqual(result["pose_usable_count"], 1)
        self.assertEqual(len(result["root_records"]), 4)
        self.assertEqual(sum(x.get("reason") == "complex_root" for x in result["root_records"]), 2)
        self.assertEqual(sum(x.get("reason") == "nonpositive_corrected_depth" for x in result["root_records"]), 1)
        candidate = result["candidates"][0]
        np.testing.assert_allclose([candidate[k] for k in ("b", "a", "c")], [0., 1., 0.], rtol=0., atol=256*EPS)
        R = np.array(((2., -2., 1.), (2., 1., -2.), (1., 2., 2.)))/3.
        np.testing.assert_allclose(candidate["R"], R, rtol=0., atol=256*EPS)
        np.testing.assert_allclose(candidate["t"], [1./6., 2./3., 1./3.], rtol=0., atol=256*EPS)
        self.assertLess(candidate["gram_residual"]["relative"], 256*EPS)
        self.assertGreater(candidate["rotation_determinant"], 0.)
        self.assertIsNone(result["root_selection"])
        self.assertFalse(result["fallback_used"])
        json.dumps(result, allow_nan=False)

    def test_exact_gram_and_quartic_coefficients(self):
        result = solve_tangent(**witness())
        M = np.array(((2., 1., 1./4.), (0., -4./3., -1./2.), (1., 14./9., 10./9.)))
        np.testing.assert_allclose(result["matrices"]["M"], M, rtol=0., atol=32*EPS)
        self.assertAlmostEqual(float(np.linalg.det(M)), -85./54., places=14)
        # ((1+b)^2-1)*(2655*(1+b)^2+3465)/7225, ascending.
        expected = np.array((0., 12240., 16740., 10620., 2655.))/7225.
        np.testing.assert_allclose(result["polynomial"]["coefficients_ascending"], expected,
                                   rtol=128*EPS, atol=128*EPS)
        self.assertEqual(result["polynomial"]["degree"], 4)

    def test_target_principal_point_reports_branch_limitation(self):
        data = witness(); data["r2"] = np.array((0., 0., 1.))
        result = solve_tangent(**data)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "singular_M_generic_branch")
        self.assertLess(result["M_diagnostics"]["rank"], 3)
        self.assertEqual(result["root_records"], [])

    def test_flat_target_depth_and_singular_warp_are_explicit(self):
        data = witness(); data["g2"] = np.zeros(2)
        result = solve_tangent(**data)
        self.assertEqual(result["reason"], "singular_M_generic_branch")
        data = witness(); data["A"][1] = 0.
        result = solve_tangent(**data)
        self.assertEqual(result["reason"], "rank_deficient_input_differential")
        self.assertEqual(result["admissible_candidate_count"], 0)

    def test_physical_flat_source_gauge_is_not_a_unique_solver_result(self):
        # Same center/R/t as witness, but source surface z=1. Target differential
        # is obtained analytically by perspective projection of R*[e1,e2].
        data = witness(); data["g1"] = np.zeros(2)
        data["g2"] = np.array((-2./5., 4./5.))
        data["A"] = np.array(((1./2., -1.), (2./3., 1./3.)))
        result = solve_tangent(**data)
        self.assertEqual(result["M_diagnostics"]["rank"], 3)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "vanished_or_numerically_unresolved_polynomial")
        self.assertEqual(result["candidates"], [])

    def test_invalid_input_rejected_without_coercion(self):
        changes = [dict(r1=[0., 0., 2.]), dict(g1=[1.]), dict(d1=float("nan")),
                   dict(A=np.eye(2, dtype=complex)), dict(E1=np.ones((3, 2)))]
        for changed in changes:
            with self.subTest(changed=list(changed)):
                data = witness(); data.update(changed)
                with self.assertRaises(ValueError):
                    solve_tangent(**data)

    def test_true_arithmetic_overflow_retained_as_failure(self):
        data = witness(); data["d1"] = 1e308
        result = solve_tangent(**data)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(result["admissible_candidate_count"], 0)
        self.assertIn("failure", result)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
