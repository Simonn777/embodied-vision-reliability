"""Manufactured arrays with fake native interfaces; no model/native/data calls."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import study_core as sc


def fixture():
    X = np.array([[-.8, -.5, 3.], [.7, -.3, 3.6], [-.5, .8, 4.2],
                  [.9, .6, 3.2], [.1, -.8, 4.8], [-.3, .2, 5.1],
                  [.4, .9, 5.5], [-.9, .1, 4.4]])
    angle = .16
    R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                  [-np.sin(angle), 0., np.cos(angle)]])
    t = np.array([.4, -.15, .25])
    Y = X @ R.T+t
    n1 = np.array([[.2, .3, -1.], [-.4, .2, -1.], [.3, -.2, -.8],
                   [.7, .1, -.6], [-.1, -.6, -.7], [.4, .5, -.8],
                   [-.6, .3, -.9], [.2, -.4, -.9]])
    n1 /= np.linalg.norm(n1, axis=1)[:, None]
    K1 = np.array([[510., 0., 319.5], [0., 490., 239.5], [0., 0., 1.]])
    K2 = np.array([[470., 0., 300.5], [0., 530., 250.5], [0., 0., 1.]])
    pair = dict(pair_index=0, scene="manufactured",
                xy1=(X/X[:, 2:] @ K1.T)[:, :2], xy2=(Y/Y[:, 2:] @ K2.T)[:, :2],
                K1=K1, K2=K2, width1=640, height1=480, width2=640, height2=480,
                original_valid1=np.ones(8, bool), original_valid2=np.ones(8, bool),
                normal1=n1, normal2=n1 @ R.T,
                normal_valid1=np.ones(8, bool), normal_valid2=np.ones(8, bool),
                comparator_valid=True)
    pair["original_valid1"][7] = False
    pair["normal_valid2"][6] = False
    for side, depth in (("1", X[:, 2]), ("2", Y[:, 2]/1.7)):
        shifts = {"LS": .08 if side == "1" else -.03,
                  "IRLS": .03 if side == "1" else .04,
                  "ANG": -.02 if side == "1" else .06}
        pair["depths"+side] = dict(A=depth*1.2, B=depth*.9, C=depth.copy(),
                                    FULL=depth.copy(),
                                    **{k: depth+b for k, b in shifts.items()})
        pair["correction"+side] = {
            k: dict(status="completed", admissible=True, b=b) for k, b in shifts.items()}
    return pair, R, t


class RefinementError(RuntimeError):
    def __init__(self, status):
        super().__init__("manufactured native error")
        self.status = status

    def as_record(self):
        return dict(error=str(self), status=self.status,
                    raw=dict(native_called=int(self.status != 1)), nonfinite_values=[])


class Backend:
    def __init__(self, R, t, snapshots=None):
        self.R, self.t, self.snapshots = R.copy(), t.copy(), snapshots
        self.calls = []
        self.mask = np.ones(8, bool)
        self.error = None

    def estimate_relative_pose(self, x1, x2, camera1, camera2, options, *, initial_pose):
        if self.snapshots is not None:
            assert self.snapshots[-1]["native_E"]["call_state"] == "invocation_intent"
        self.calls.append(dict(x1=x1.copy(), x2=x2.copy(), camera1=camera1,
                               camera2=camera2, options=copy.deepcopy(options),
                               initial_pose=initial_pose))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(R=self.R.copy(), t=self.t.copy()), {
            "inliers": self.mask.copy(), "iterations": 1000}


class Refiner:
    RefinementError = RefinementError

    def __init__(self, snapshots=None):
        self.calls, self.snapshots = [], snapshots
        self.error_status = None
        self.program_error = None
        self.physical = True

    def refine(self, x1, x2, d1, d2, R, t, a, b, c, focal1, focal2, *, max_iterations):
        if self.snapshots is not None:
            assert self.snapshots[-1]["affine"]["attempts"][-1]["call_state"] == "invocation_intent"
        self.calls.append(dict(x1=x1.copy(), x2=x2.copy(), d1=d1.copy(), d2=d2.copy(),
                               R=np.asarray(R).copy(), t=np.asarray(t).copy(),
                               a=a, b=b, c=c, focal1=focal1, focal2=focal2,
                               max_iterations=max_iterations))
        if self.program_error is not None:
            raise self.program_error
        if self.error_status is not None:
            raise RefinementError(self.error_status)
        return dict(schema="evr0026.native-refinement.v1",
                    geometry=dict(R=np.asarray(R).tolist(), t=np.asarray(t).tolist(),
                                  a=float(a), b=float(b), c=float(c)),
                    output_checks=dict(physical_admissible=self.physical),
                    timings=dict(native_call_seconds=0., outer_call_seconds=0.))


class StudyCoreTests(unittest.TestCase):
    def test_all_arms_one_support_full_scores_and_native_options(self):
        pair, R, t = fixture()
        snapshots = []
        backend, refiner = Backend(R, t, snapshots), Refiner(snapshots)
        result = sc.process_pair(pair, backend, refiner, emit=snapshots.append)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(tuple(result["arms"]), sc.ARMS)
        self.assertEqual(result["support_indices"], list(range(6)))
        self.assertEqual(result["original_inlier_mask"], [True]*8)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["options"], sc.E_OPTIONS)
        self.assertIsNone(backend.calls[0]["initial_pose"])
        self.assertEqual(backend.calls[0]["camera1"]["model"], "PINHOLE")
        self.assertEqual(backend.calls[0]["x1"].shape, (8, 2))
        self.assertEqual(len(result["similarity_attempts"]), 7)
        for row in result["similarity_attempts"]:
            self.assertEqual(row["result"]["support_count"], 6)
        self.assertEqual(result["normal_attempt"]["support_count"], 6)
        self.assertEqual(result["normal_refinement"]["fit"]["support_count"], 6)
        self.assertEqual(len(refiner.calls), 5)
        for call in refiner.calls:
            self.assertEqual(call["d1"].shape, (6,))
            self.assertEqual(call["max_iterations"], 100)
            self.assertEqual(call["focal1"], 500.)
            self.assertEqual(call["focal2"], 500.)
        for collection in (result["normal_refinement"]["selector"]["attempts"],
                           result["affine"]["selector"]["attempts"]):
            for score in collection:
                self.assertEqual(score["support_count"], 8)
        self.assertEqual(result["calls"], dict(estimate_invocation_intents=1, estimate_returned=1,
                                               refine_invocation_intents=5, refine_returned=5))
        # Earlier callback objects do not acquire later arms.
        self.assertEqual(snapshots[0]["arms"], {})
        json.dumps(result, allow_nan=False)

    def test_affine_seeds_are_in_original_C_depth_gauge(self):
        pair, R, t = fixture()
        refiner = Refiner()
        result = sc.process_pair(pair, Backend(R, t), refiner)
        self.assertEqual([s["label"] for s in result["affine"]["seeds"]], list(sc.AFFINE_SEED_ORDER))
        for label, call in zip(sc.AFFINE_SEED_ORDER, refiner.calls):
            np.testing.assert_array_equal(call["d1"], pair["depths1"]["C"][:6])
            np.testing.assert_array_equal(call["d2"], pair["depths2"]["C"][:6])
            np.testing.assert_allclose(call["x1"],
                                       (np.c_[pair["xy1"][:6], np.ones(6)] @ np.linalg.inv(pair["K1"]).T)[:, :2])
            if label in sc.CORRECTIONS:
                self.assertEqual(call["b"], pair["correction1"][label]["b"])
                self.assertEqual(call["c"], call["a"]*pair["correction2"][label]["b"])
            else:
                self.assertEqual(call["b"], 0.)
                self.assertEqual(call["c"], 0.)
        self.assertAlmostEqual(np.linalg.norm(refiner.calls[0]["t"]), 1.)

    def test_bad_corrected_depth_does_not_shrink_common_support(self):
        pair, R, t = fixture()
        pair["depths1"]["LS"][2] = np.nan
        result = sc.process_pair(pair, Backend(R, t), Refiner())
        self.assertEqual(result["support_indices"], list(range(6)))
        self.assertEqual(result["arms"]["LS"]["status"], "fallback")
        self.assertEqual(result["arms"]["ANG"]["status"], "completed")
        self.assertNotIn("LS", [s["label"] for s in result["affine"]["seeds"]])
        self.assertEqual(result["similarity_attempts"][3]["result"]["support_count"], 6)

    def test_uninformative_ANG_and_invalid_FULL_remain_fallback(self):
        pair, R, t = fixture()
        pair["correction1"]["ANG"].update(status="uninformative", admissible=True, b=0.)
        pair["depths1"]["ANG"] = pair["depths1"]["C"].copy()
        pair["comparator_valid"] = False
        result = sc.process_pair(pair, Backend(R, t), Refiner())
        self.assertEqual(result["arms"]["ANG"]["status"], "fallback")
        self.assertEqual(result["arms"]["FULL"]["status"], "fallback")
        self.assertFalse(result["comparator_valid"])
        self.assertNotIn("ANG", [s["label"] for s in result["affine"]["seeds"]])
        row = next(x for x in result["similarity_attempts"] if x["arm"] == "ANG")
        self.assertEqual(row["result"]["seconds"], 0.)

    def test_support_below_four_and_unavailable_E_keep_all_arms(self):
        pair, R, t = fixture()
        pair["original_valid1"][:5] = False
        refiner = Refiner()
        result = sc.process_pair(pair, Backend(R, t), refiner)
        self.assertEqual(tuple(result["arms"]), sc.ARMS)
        self.assertTrue(all(result["arms"][k]["status"] == "fallback" for k in sc.ARMS[1:]))
        self.assertEqual(refiner.calls, [])
        pair, R, t = fixture()
        backend = Backend(R, np.zeros(3))
        result = sc.process_pair(pair, backend, Refiner())
        self.assertEqual(len(result["arms"]), 11)
        self.assertTrue(all(a["status"] == "unavailable" for a in result["arms"].values()))
        self.assertEqual(result["calls"]["estimate_returned"], 1)

    def test_expected_refinement_failures_are_kept_without_retry(self):
        for status in (1, 3):
            with self.subTest(status=status):
                pair, R, t = fixture()
                refiner = Refiner()
                refiner.error_status = status
                result = sc.process_pair(pair, Backend(R, t), refiner)
                self.assertEqual(len(refiner.calls), 5)
                self.assertTrue(all(a["status"] == "unavailable" for a in result["affine"]["attempts"]))
                self.assertEqual(result["arms"]["AFFINE"]["status"], "fallback")
                self.assertEqual(result["affine"]["selector"]["selected_index"], 0)

    def test_nonphysical_native_outputs_are_not_candidates(self):
        pair, R, t = fixture()
        refiner = Refiner()
        refiner.physical = False
        result = sc.process_pair(pair, Backend(R, t), refiner)
        self.assertEqual(len(result["affine"]["candidates"]), 1)
        self.assertEqual(result["arms"]["AFFINE"]["status"], "fallback")
        self.assertEqual(result["calls"]["refine_returned"], 5)

    def test_unexpected_native_error_propagates_with_durable_intent(self):
        for error in (RefinementError(2), ValueError("ABI signature mismatch")):
            with self.subTest(error=type(error).__name__):
                pair, R, t = fixture()
                snapshots = []
                refiner = Refiner(snapshots)
                refiner.program_error = error
                with self.assertRaises(type(error)):
                    sc.process_pair(pair, Backend(R, t, snapshots), refiner, emit=snapshots.append)
                self.assertEqual(len(refiner.calls), 1)
                self.assertEqual(snapshots[-1]["status"], "technical_failure")
                self.assertEqual(snapshots[-1]["failed_phase"], "affine_native_E")
                self.assertTrue(any(
                    s["affine"]["attempts"] and s["affine"]["attempts"][-1]["call_state"] == "invocation_intent"
                    for s in snapshots))

    def test_backend_error_or_malformed_mask_is_not_scientific_fallback(self):
        pair, R, t = fixture()
        for bad in ("error", "mask"):
            backend, refiner, snapshots = Backend(R, t), Refiner(), []
            if bad == "error":
                backend.error = RuntimeError("native backend failed")
            else:
                backend.mask = np.ones(7, bool)
            with self.assertRaises((RuntimeError, ValueError)):
                sc.process_pair(pair, backend, refiner, emit=snapshots.append)
            self.assertEqual(snapshots[-1]["status"], "technical_failure")
            self.assertEqual(refiner.calls, [])

    def test_normal_refinement_bank_never_adds_unmodified_E_incumbent(self):
        pair, R, t = fixture()
        fake = dict(status="completed", attempts=[
            dict(status="unavailable", label="NORMAL", reason="manufactured"),
            dict(status="unavailable", label="E", reason="manufactured")])
        with patch.object(sc.pc, "refine_joint", return_value=fake):
            result = sc.process_pair(pair, Backend(R, t), Refiner())
        self.assertEqual(result["normal_refinement"]["candidates"], [])
        self.assertEqual(result["arms"]["NORMAL_REFINE"]["status"], "fallback")
        self.assertEqual(result["normal_refinement"]["selector"]["selected_index"], None)

    def test_checkpoint_failure_prevents_first_native_call(self):
        pair, R, t = fixture()
        backend, refiner = Backend(R, t), Refiner()
        def broken(_):
            raise OSError("manufactured persistence failure")
        with self.assertRaises(sc.ProgressError):
            sc.process_pair(pair, backend, refiner, emit=broken)
        self.assertEqual(backend.calls, [])
        self.assertEqual(refiner.calls, [])


if __name__ == "__main__":
    unittest.main()
