"""Prepared artificial-only checks. No frozen random cohort is iterated here."""
import copy
import json
import unittest
from unittest import mock
import numpy as np
import fixtures
import study_core


def artificial_observation():
    K = np.array(fixtures.K_VALUES)
    xy = np.array([(319.5+x, 239.5+y) for y in (-3., 0., 3.) for x in (-3., 0., 3.)])
    d = np.full(9, 2.)
    other = xy+np.array((25., 0.))
    derivative = fixtures.fit_derivatives(xy, other, d, d)
    return dict(K=K.tolist(), xy1=xy.tolist(), xy2=other.tolist(),
                d1=d.tolist(), d2=d.tolist(), center_index=4, derivatives=derivative)


def physical_candidate():
    return dict(R=np.eye(3).tolist(), t=[.1, 0., 0.], a=1., b=0., c=0., pose_usable=True)


class FixtureTests(unittest.TestCase):
    def test_lazy_iterator_does_not_generate_on_creation(self):
        with mock.patch.object(fixtures, "_physical", side_effect=AssertionError("must remain deferred")):
            iterator = fixtures.iter_cases()
            self.assertIs(iter(iterator), iterator)
            iterator.close()

    def test_physical_projection_and_per_pixel_derivative(self):
        # Clean exact witness only; never requests fixed noisy draws.
        observation, truth = fixtures.make_clean("G", "shifted", "exact")
        X, Y, R, t = map(np.asarray, (truth["X"], truth["Y"], truth["R"], truth["t"]))
        np.testing.assert_allclose(Y, X@R.T+t, atol=2e-15, rtol=2e-15)
        derivatives = observation["derivatives"]
        np.testing.assert_allclose(derivatives["A"], [[.5, -1.], [0., 1./3.]], atol=2e-15)
        np.testing.assert_allclose(derivatives["g1"], [1./500., 0.], atol=2e-18)
        np.testing.assert_allclose(derivatives["g2"], np.array([2., 8.])/(500.*1.3), atol=2e-17)
        self.assertEqual(set(observation), fixtures.OBSERVATION_KEYS)
        self.assertNotIn("R", observation)
        json.dumps(observation, allow_nan=False)

    def test_center_fixed_ols_keeps_measured_center(self):
        xy = np.array([(x, y) for y in (-2., 0., 2.) for x in (-2., 0., 2.)])
        A = np.array([[1.2, -.3], [.1, .8]])
        target = xy@A.T+[7., -4.]
        g1, g2 = np.array([.2, -.4]), np.array([.1, .3])
        d1, d2 = xy@g1+5., target@g2+8.
        result = fixtures.fit_derivatives(xy, target, d1, d2)
        self.assertEqual(result["status"], "completed")
        np.testing.assert_allclose(result["A"], A, atol=2e-15)
        np.testing.assert_allclose(result["g1"], g1, atol=2e-15)
        np.testing.assert_allclose(result["g2"], g2, atol=2e-15)
        np.testing.assert_array_equal(result["fit"]["source_differences"][4], [0., 0.])
        np.testing.assert_array_equal(result["fit"]["target_differences"][4], [0., 0.])

    def test_rank_failure_is_timed_and_preserved(self):
        result = fixtures.fit_derivatives(np.zeros((9, 2)), np.zeros((9, 2)),
                                         np.ones(9), np.ones(9))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "rank_deficient_neighborhood")
        self.assertGreaterEqual(result["elapsed_seconds"], 0.)
        self.assertIsNone(result["A"])

    def test_input_label_barrier(self):
        observation = artificial_observation()
        observation["truth"] = {"R": np.eye(3).tolist()}
        with self.assertRaises(ValueError):
            study_core.process(observation, object())
        observation = artificial_observation()
        observation["derivatives"]["fit"]["gt"] = 1
        with self.assertRaises(ValueError):
            study_core.process(observation, object())

    def test_all_nine_positive_guard_and_behind_camera_cap(self):
        o = study_core.validate_observation(artificial_observation())
        candidate = physical_candidate()
        good = study_core.validate_and_score(candidate, o)
        self.assertTrue(good["pose_usable"])
        self.assertLess(good["score"], 1e-24)
        modified = dict(o, d1=o["d1"].copy())
        modified["d1"][0] = -1.
        invalid = study_core.validate_and_score(candidate, modified)
        self.assertFalse(invalid["pose_usable"])
        self.assertEqual(invalid["all_nine_depths"]["positive_count"], 8)
        candidate["t"] = [0., 0., -3.]
        capped = study_core.validate_and_score(candidate, o)
        self.assertEqual(capped["forward"]["invalid_projection_count"], 9)
        self.assertEqual(capped["forward"]["terms"], [1.]*9)
        self.assertGreaterEqual(capped["score"], .5)

    def test_fake_orchestration_all_calls_ties_and_shared_support(self):
        observation = artificial_observation()
        candidate = physical_candidate()
        calls = []
        def point_control(**args):
            calls.append(copy.deepcopy(args))
            return dict(status="completed", candidates=[copy.deepcopy(candidate)])
        def tangent_control(**args):
            return dict(status="completed", candidates=[copy.deepcopy(candidate)])
        updates = []
        with mock.patch.object(study_core.tangent_solver, "solve_tangent", tangent_control), \
             mock.patch.object(study_core.controls, "solve_scale_only", tangent_control), \
             mock.patch.object(study_core.controls, "solve_scale_only_ordered", tangent_control), \
             mock.patch.object(study_core.controls, "solve_points_native", point_control):
            result = study_core.process(observation, "fake native", progress=updates.append)
        self.assertEqual(len(calls), 12)
        self.assertEqual(result["call_counts"], dict(native_attempted=12, native_executed=12))
        for arm in study_core.ARMS:
            self.assertEqual(result["arms"][arm]["selected_index"], 0)
            self.assertLess(result["arms"][arm]["selected"]["score"], 1e-24)
        self.assertEqual(len(result["arms"]["S"]["attempts"]), 2)
        self.assertEqual(len(result["arms"]["D8"]["attempts"]), 8)
        all_ids = set(i for triangle in study_core.TRIANGLES for i in triangle)
        self.assertEqual(all_ids, set(range(9)))
        self.assertTrue(any(x["active_attempt"] is not None for x in updates))
        self.assertIsNone(result["active_attempt"])
        json.dumps(result, allow_nan=False)

    def test_virtual_support_failure_cannot_be_resurrected(self):
        o = study_core.validate_observation(artificial_observation())
        candidate = physical_candidate()
        candidate["pose_usable"] = False
        candidate["physical"] = False
        row = study_core.validate_and_score(candidate, o)
        self.assertEqual(row["all_nine_depths"]["positive_count"], 9)
        self.assertFalse(row["pose_usable"])
        self.assertEqual(row["reason"], "solver_reported_nonusable_candidate")

    def test_failed_common_fit_skips_derivative_arms_but_runs_eight_raw_triples(self):
        observation = artificial_observation()
        observation["derivatives"] = fixtures.fit_derivatives(
            np.zeros((9, 2)), np.zeros((9, 2)), np.ones(9), np.ones(9))
        with mock.patch.object(study_core.controls, "solve_points_native",
                               return_value=dict(status="completed", candidates=[])) as native:
            result = study_core.process(observation, object())
        self.assertEqual(native.call_count, 8)
        self.assertEqual(result["call_counts"]["native_executed"], 8)
        self.assertTrue(all(not a["native_called"] for a in result["arms"]["V"]["attempts"]))
        self.assertTrue(all(a["native_called"] for a in result["arms"]["D8"]["attempts"]))
        self.assertTrue(all(a["status"] == "unavailable" for a in result["arms"].values()))
        self.assertEqual(sum(len(a["attempts"]) for a in result["arms"].values()), 15)

    def test_unexpected_native_abi_error_propagates(self):
        unavailable = lambda **kw: dict(status="unavailable", candidates=[])
        updates = []
        with mock.patch.object(study_core.tangent_solver, "solve_tangent", unavailable), \
             mock.patch.object(study_core.controls, "solve_scale_only", unavailable), \
             mock.patch.object(study_core.controls, "solve_scale_only_ordered", unavailable), \
             mock.patch.object(study_core.controls, "solve_points_native",
                               side_effect=TypeError("fake ABI mismatch")):
            with self.assertRaisesRegex(TypeError, "ABI mismatch"):
                study_core.process(artificial_observation(), object(), progress=updates.append)
        native_intents = [row["arms"]["V"]["attempts"][-1] for row in updates
                          if row["arms"]["V"]["attempts"]
                          and row["arms"]["V"]["attempts"][-1].get("native_called")]
        self.assertTrue(native_intents)
        self.assertEqual(native_intents[0]["call_state"], "invocation_intent")
        self.assertEqual(native_intents[-1]["status"], "technical_failure")


if __name__ == "__main__":
    unittest.main()
