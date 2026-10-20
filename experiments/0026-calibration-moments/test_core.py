"""Manufactured orchestration only: no SciPy/native/data/label calls.

All numerical providers are injected or patched. These tests check support,
failure, selection and accounting contracts, not scientific efficacy.
"""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import core


class FakeBase:
    def __init__(self):
        self.scored_counts = []
        self.diagnostic_counts = []

    @staticmethod
    def json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: FakeBase.json_safe(v) for k, v in value.items()}
        return value

    @staticmethod
    def prepare_geometry(x1, x2, calibration):
        return {"h1": np.column_stack((x1/100., np.ones(len(x1)))),
                "h2": np.column_stack((x2/200., np.ones(len(x2)))),
                "params1": np.array((100., 120., 0., 0.)),
                "params2": np.array((200., 240., 0., 0.))}

    @staticmethod
    def validate_pose(estimate):
        R, t = np.asarray(estimate["R"]), np.asarray(estimate["t"])
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError("invalid fake pose")
        if np.max(np.abs(t)) == 0.:
            raise ValueError("zero fake translation")
        return R, t/np.linalg.norm(t)

    def final_scores(self, estimate, geometry):
        n = len(geometry["h1"])
        self.scored_counts.append(n)
        # An intentionally transparent fake selector; no geometric claim.
        return {"pixel_msac": float(estimate["t"][0]), "pixel_threshold": 2.,
                "count": n, "pixel_inliers": n, "cheiral_count": n,
                "degenerate_count": 0, "nonfinite_arithmetic_count": 0}

    def depth_diagnostics(self, geometry, inputs, d1, d2, mask):
        self.diagnostic_counts.append((len(d1), int(mask.sum())))
        return {"all": {"denominator": len(d1),
                "nonpositive_depth1": int(np.sum(d1+geometry["shift1"] <= 0.))},
                "returned_mask": {"denominator": int(mask.sum())}}


class RefinementError(RuntimeError):
    def __init__(self, status):
        super().__init__("manufactured native status " + str(status))
        self.status = status
        self.raw = {"native_called": 0 if status == 1 else 1,
                    "completed": 0, "cost": float("nan") if status == 3 else 0.}


class FakeNative:
    def __init__(self, status=None, negative_output=False):
        self.calls = []
        self.status = status
        self.negative_output = negative_output

    def refine(self, x1, x2, d1, d2, R, t, a, b, c, f1, f2, max_iterations=100):
        self.calls.append({"x1": x1.copy(), "x2": x2.copy(), "d1": d1.copy(),
            "d2": d2.copy(), "R": R.copy(), "t": t.copy(), "parameters": [b, a, c],
            "focals": [f1, f2], "max_iterations": max_iterations})
        if self.status is not None:
            raise RefinementError(self.status)
        geometry = {"R": R.copy(), "t": t.copy(), "scale": float(a),
                    "shift1": float(b), "shift2": float(c/a)}
        if self.negative_output:
            geometry.update(scale=-1., shift1=-100.)
        return {"schema": "evr0026.native-refinement.v1", "geometry": geometry,
            "bundle_stats": {"iterations": 1, "cost": .2},
            "output_checks": {"native_called": True, "completed": True,
                              "physical_admissible": not self.negative_output},
            "timings": {"native_call_seconds": .001}}


def fixture(pair_index=3, kept=8):
    n = 10
    pair = {"x1": np.arange(n*2, dtype=float).reshape(n, 2),
            "x2": np.arange(n*2, dtype=float).reshape(n, 2)+3.,
            "depth1": np.linspace(2., 3., n), "depth2": np.linspace(3., 4., n),
            "calibration": {"camera1": "PINHOLE", "camera2": "PINHOLE",
                            "K1": np.eye(3), "K2": np.eye(3)}}
    mask = np.arange(n) < kept
    base = FakeBase()
    geometry = {"R": np.eye(3).tolist(), "t": [3., 0., 0.],
                "scale": 1.2, "shift1": .1, "shift2": .2/1.2}
    estimate = {"R": geometry["R"], "t": geometry["t"]}
    record = {"pair_index": pair_index, "status": "completed", "arm": "depth_affine",
        "calibration": base.json_safe(pair["calibration"]), "estimate": estimate,
        "geometry": geometry, "mask_length": n, "kept_count": kept,
        "native_inliers_packbits_big_hex": np.packbits(mask, bitorder="big").tobytes().hex(),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "final_scores": base.final_scores(estimate, base.prepare_geometry(pair["x1"], pair["x2"], pair["calibration"])),
        "timings": {"historical_process_seconds": 1.}}
    base.scored_counts.clear()
    return pair, record, base, mask


def optimization(model, seed, m1, m2, objective):
    final = {"parameters": seed.copy(), "cost_raw": -.00001,
             "gradient": np.zeros(3), "gradient_physical": np.zeros(3),
             "coordinates": np.zeros(3)}
    return {"status": "completed", "objective": objective,
            "optimizer": {"success": True}, "final_evaluation": final,
            "last_feasible": copy.deepcopy(final), "objective_evaluations": 2}


class Providers:
    def __init__(self, optimize=optimization, lift_status="completed"):
        self.optimize = optimize
        self.lift_status = lift_status
        self.weights = []

    def build(self, r1, r2, d1, d2, weights):
        self.weights.append(weights.copy())
        return SimpleNamespace(H=np.eye(6), C=np.eye(12), mean=np.zeros(12),
            weight_sum=float(weights.sum()), input_count=len(weights),
            support_count=int(np.count_nonzero(weights)))

    def lift(self, H, m1, m2, max_newton):
        if max_newton != 40:
            raise AssertionError("changed relaxation budget")
        return {"status": self.lift_status,
                "physical": {"status": "feasible", "parameters": [.4, 1.4, .3]},
                "dual": {"condition": float("inf")}}

    @staticmethod
    def recover(model, b, a, c):
        return {"R": np.eye(3), "t": np.array((a, 0., 0.)), "cost_raw": -.0001}

    def __enter__(self):
        self.patches = [patch.object(core.moments, "build_moments", side_effect=self.build),
            patch.object(core.lift_relaxation, "solve_relaxation", side_effect=self.lift),
            patch.object(core.controls, "optimize_calibration", side_effect=self.optimize),
            patch.object(core.moments, "procrustes_objective", side_effect=self.recover)]
        self.mocks = [p.start() for p in self.patches]
        return self

    def __exit__(self, *args):
        for p in reversed(self.patches):
            p.stop()


class CoreTests(unittest.TestCase):
    def test_fixed_support_slots_scoring_ties_costs_and_progress(self):
        pair, baseline, base, mask = fixture()
        native, snapshots = FakeNative(), []
        with Providers() as providers:
            result = core.process_pair(pair, baseline, native, base, 3, progress=snapshots.append)
        self.assertEqual(result["execution_order"], list(core.SLOTS[3:]+core.SLOTS[:3]))
        self.assertEqual(set(result["attempts"]), set(core.SLOTS))
        self.assertEqual(len(native.calls), 7)
        self.assertEqual(providers.mocks[1].call_count, 1)
        self.assertEqual(providers.mocks[2].call_count, 6)
        np.testing.assert_array_equal(providers.weights[0], mask.astype(float))
        self.assertEqual(base.scored_counts, [10]*8)
        for call in native.calls:
            np.testing.assert_array_equal(call["x1"], pair["x1"][mask]/100.)
            np.testing.assert_array_equal(call["d1"], pair["depth1"][mask])
            self.assertEqual(call["max_iterations"], 100)
            self.assertEqual(call["focals"], [110., 220.])
        self.assertEqual(result["selectors"]["G_lift"]["selected_attempt"], "G_lift")
        self.assertEqual(result["selectors"]["F_all"]["selected_attempt"], "F_neutral")
        self.assertEqual(result["selectors"]["continued_tc"]["selected_attempt"], "incumbent")
        for name, selector in result["selectors"].items():
            self.assertEqual(selector["members"], list(core.UNIONS[name]))
            expected = sum(result["preparations"][p]["seconds"] for p in selector["prerequisites"])
            expected += sum(result["attempts"][a]["timings"]["total_seconds"]
                            for a in selector["members"] if a != "incumbent")
            self.assertAlmostEqual(selector["charged_seconds"], expected, places=14)
        self.assertEqual(result["incumbent"]["selected_attempt"], "incumbent")
        self.assertEqual(snapshots[0]["phase"], "before_moments")
        self.assertEqual(snapshots[-1]["phase"], "slot_complete:G_incumbent")
        self.assertEqual(len(snapshots[0]["attempts"]), 0)  # detached snapshot
        self.assertTrue(any(x["phase"] == "before_native:G_lift" for x in snapshots))
        self.assertTrue(result["nonfinite_observations"])  # explicit inf condition, not zero
        json.dumps(result, allow_nan=False)

    def test_small_support_retains_baseline_without_any_numerical_provider(self):
        pair, baseline, base, _ = fixture(kept=3)
        native = FakeNative()
        with Providers() as providers:
            result = core.process_pair(pair, baseline, native, base, 3)
        self.assertEqual(len(native.calls), 0)
        self.assertTrue(all(m.call_count == 0 for m in providers.mocks))
        self.assertTrue(all(a["status"] == "unavailable" for a in result["attempts"].values()))
        self.assertTrue(all(s["selected_attempt"] == "incumbent" for s in result["selectors"].values()))

    def test_nonpositive_incumbent_and_neutral_are_unavailable_not_dropped(self):
        pair, baseline, base, _ = fixture()
        pair["depth1"][0] = -.05
        baseline["geometry"]["shift1"] = -.2
        native = FakeNative()
        with Providers():
            result = core.process_pair(pair, baseline, native, base, 3)
        for name in ("continued_tc", "G_incumbent", "F_incumbent", "G_neutral", "F_neutral"):
            self.assertEqual(result["attempts"][name]["status"], "unavailable")
        self.assertEqual(len(native.calls), 2)
        self.assertEqual(result["incumbent"]["status"], "completed")

    def test_failed_lift_never_uses_present_rounding_as_fallback(self):
        pair, baseline, base, _ = fixture()
        native = FakeNative()
        with Providers(lift_status="failed"):
            result = core.process_pair(pair, baseline, native, base, 3)
        for name in ("G_lift", "F_lift"):
            self.assertEqual(result["attempts"][name]["status"], "unavailable")
        self.assertEqual(len(native.calls), 5)
        self.assertEqual(result["selectors"]["G_lift"]["selected_attempt"], "incumbent")

    def test_nonconvergence_retains_last_evaluation_without_refinement(self):
        pair, baseline, base, _ = fixture()
        def failed(model, seed, m1, m2, objective):
            result = optimization(model, seed, m1, m2, objective)
            if objective == "G":
                result["status"] = "failed"
                result["optimizer"]["success"] = False
            return result
        native = FakeNative()
        with Providers(optimize=failed):
            result = core.process_pair(pair, baseline, native, base, 3)
        self.assertEqual(len(native.calls), 4)
        for name in ("G_neutral", "G_incumbent", "G_lift"):
            row = result["attempts"][name]
            self.assertEqual(row["status"], "failed")
            self.assertIsNotNone(row["optimizer"]["last_feasible"])
            self.assertIsNone(row["native"])

    def test_postnative_nonpositive_depth_is_diagnostic_and_still_scored(self):
        pair, baseline, base, _ = fixture()
        with Providers():
            result = core.process_pair(pair, baseline, FakeNative(negative_output=True), base, 3)
        for row in result["attempts"].values():
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["depth_diagnostics"]["all"]["nonpositive_depth1"], 10)
            self.assertFalse(row["native"]["output_checks"]["physical_admissible"])
        self.assertEqual(base.scored_counts, [10]*8)

    def test_mask_and_incumbent_score_integrity_fail_before_fits(self):
        for mutation in ("padding", "hash", "count", "score"):
            with self.subTest(mutation=mutation):
                pair, baseline, base, _ = fixture()
                if mutation == "padding":
                    baseline["native_inliers_packbits_big_hex"] = "ff01"
                elif mutation == "hash":
                    baseline["mask_sha256"] = "0"*64
                elif mutation == "count":
                    baseline["kept_count"] = 7
                else:
                    baseline["final_scores"]["pixel_msac"] += 1e-8
                with Providers() as providers, self.assertRaises(ValueError):
                    core.process_pair(pair, baseline, FakeNative(), base, 3)
                self.assertTrue(all(m.call_count == 0 for m in providers.mocks))

    def test_native_status_domains_and_programming_faults(self):
        for status, expected in ((1, "unavailable"), (3, "failed")):
            with self.subTest(status=status):
                pair, baseline, base, _ = fixture()
                with Providers():
                    result = core.process_pair(pair, baseline, FakeNative(status=status), base, 3)
                self.assertTrue(all(row["status"] == expected for row in result["attempts"].values()))
                self.assertTrue(all(s["selected_attempt"] == "incumbent" for s in result["selectors"].values()))
                self.assertEqual(result["timings"]["native_attempt_count"], 0 if status == 1 else 7)
                json.dumps(result, allow_nan=False)
        pair, baseline, base, _ = fixture()
        with Providers(), self.assertRaises(RefinementError):
            core.process_pair(pair, baseline, FakeNative(status=2), base, 3)
        def programming_fault(*args):
            raise ValueError("wrong internal matrix shape")
        with Providers(optimize=programming_fault), self.assertRaisesRegex(ValueError, "wrong internal"):
            core.process_pair(pair, baseline, FakeNative(), base, 3)

    def test_progress_failure_propagates_and_prior_slots_are_detached(self):
        pair, baseline, base, _ = fixture()
        snapshots = []
        def progress(snapshot):
            snapshots.append(snapshot)
            if snapshot["phase"] == "before_optimizer:F_neutral":
                raise FloatingPointError("manufactured storage failure")
        with Providers(), self.assertRaises(core.ProgressError):
            core.process_pair(pair, baseline, FakeNative(), base, 3, progress=progress)
        complete = [x for x in snapshots if x["phase"] == "slot_complete:G_lift"]
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0]["attempts"]["G_lift"]["status"], "completed")
        self.assertNotIn("F_neutral", complete[0]["attempts"])


if __name__ == "__main__":
    unittest.main()
