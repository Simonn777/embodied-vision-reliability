"""Manufactured-only tests. No real images, HDF5, labels or native estimator."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("score0020_test", Path(__file__).with_name("score.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture(n=240):
    rng = np.random.default_rng(2020)
    X = np.column_stack((rng.uniform(-1.5, 1.5, n), rng.uniform(-1., 1., n), rng.uniform(4., 9., n)))
    angle = .08
    R = np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                  [-math.sin(angle), 0., math.cos(angle)]])
    t = np.array([.8, .04, .09])
    Y = np.einsum("ij,nj->ni", R, X, optimize=False) + t
    cal, coords = {}, []
    for j, (P, p) in enumerate(((X, [500., 510., 320., 240.]), (Y, [600., 620., 300., 250.])), 1):
        cal["camera"+str(j)] = {"model": "PINHOLE", "width": 640, "height": 480, "params": p}
        cal["K"+str(j)] = [[p[0], 0., p[2]], [0., p[1], p[3]], [0., 0., 1.]]
        coords.append(P[:, :2]/P[:, 2, None]*p[:2]+p[2:])
    d1, d2 = (X[:, 2]-2.)/1.7, (Y[:, 2]+3.)/.6
    pose = {"R": R, "t": t}
    return coords[0], coords[1], d1, d2, cal, pose


def bank(x1, x2, cal, poses):
    g = s.base.prepare_geometry(x1, x2, cal)
    return [{"arm": arm, "status": "completed", "estimate": copy.deepcopy(pose),
             "final_scores": s.base.final_scores(pose, g)} for arm, pose in zip(s.ARMS, poses)]


def call(args, poses=None):
    x1, x2, d1, d2, cal, pose = args
    return s.score_pair("manufactured", x1, x2, d1, d2, cal,
                        bank(x1, x2, cal, [pose]*4 if poses is None else poses))


def scalar_geometry_reference(pose, x1, x2, cal):
    R = np.asarray(pose["R"])
    t = np.asarray(pose["t"])/np.linalg.norm(pose["t"])
    E = np.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]]) @ R
    K1, K2 = np.asarray(cal["K1"]), np.asarray(cal["K2"])
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    values = []
    for x, y in zip(x1, x2):
        p, q = np.r_[x, 1.], np.r_[y, 1.]
        h1, h2 = np.linalg.solve(K1, p), np.linalg.solve(K2, q)
        u, v = h1/np.linalg.norm(h1), h2/np.linalg.norm(h2)
        ru = R @ u
        a, b1, b2 = -float(ru @ v), -float(ru @ t), float(v @ t)
        cheiral = b1-a*b2 > .01*(1-a*a) and b2-a*b1 > .01*(1-a*a)
        l1, l2 = F.T @ q, F @ p
        denom = float(l1[:2] @ l1[:2] + l2[:2] @ l2[:2])
        num = float(q @ F @ p)
        values.append(min(num*num/(4*denom), 1.) if cheiral and denom > 0 else 1.)
    return values


class GeometryTests(unittest.TestCase):
    def test_port_matches_frozen_aggregate_and_independent_scalar(self):
        x1, x2, _, _, cal, pose = fixture()
        x2[:, 1] += np.linspace(0., 5., len(x2))
        g = s.base.prepare_geometry(x1, x2, cal)
        values, meta = s.pixel_costs(pose, g)
        self.assertEqual(meta, s.base.final_scores(pose, g))
        np.testing.assert_allclose(values, scalar_geometry_reference(pose, x1, x2, cal), atol=1e-11, rtol=1e-11)
        pose["t"] *= -1
        wrong, _ = s.pixel_costs(pose, g)
        np.testing.assert_array_equal(wrong, np.ones(len(x1)))

    def test_sealed_score_bank_and_nonfinite_input_barriers(self):
        x1, x2, d1, d2, cal, pose = fixture()
        original = bank(x1, x2, cal, [pose]*4)
        for mutation in (lambda b: b.reverse(), lambda b: b.pop(),
                         lambda b: b[0].update(status="error"),
                         lambda b: b[2]["final_scores"].update(pixel_msac=2.),
                         lambda b: b[3]["final_scores"].update(count=True),
                         lambda b: b[1]["estimate"].update(t=[0., 0., 0.])):
            broken = copy.deepcopy(original)
            mutation(broken)
            with self.assertRaises(ValueError):
                s.score_pair("k", x1, x2, d1, d2, cal, broken)
        for invalid in (np.full(len(d1), np.nan), np.ones((len(d1), 1))):
            with self.assertRaises(ValueError):
                s.score_pair("k", x1, x2, invalid, d2, cal, original)


class FittingTests(unittest.TestCase):
    def test_unexpected_valueerror_is_a_technical_failure(self):
        x1, x2, d1, d2, cal, pose = fixture()
        g = s.base.prepare_geometry(x1, x2, cal)
        R, t = s.base.validate_pose(pose)
        z1, _ = s.normalize_depth(d1)
        z2, _ = s.normalize_depth(d2)
        with patch.object(np.linalg, "svd", side_effect=ValueError("artificial shape fault")):
            with self.assertRaisesRegex(ValueError, "artificial shape fault"):
                s.fit_calibration(R, t, g, z1, z2, np.ones(len(d1), bool), np.ones(len(d1)))
        with patch.object(np, "mean", side_effect=ValueError("artificial normalization fault")):
            with self.assertRaisesRegex(ValueError, "artificial normalization fault"):
                s.normalize_depth(d1)

    def test_exact_affine_calibration_and_direct_reference(self):
        x1, x2, d1, d2, cal, pose = fixture()
        g = s.base.prepare_geometry(x1, x2, cal)
        R, t = s.base.validate_pose(pose)
        z1, m1 = s.normalize_depth(d1)
        z2, m2 = s.normalize_depth(d2)
        weights = np.linspace(1., 2., len(d1))
        support = np.arange(len(d1)) % 3 != 0
        fit = s.fit_calibration(R, t, g, z1, z2, support, weights)
        self.assertEqual(fit["status"], "completed")
        norm = np.linalg.norm(pose["t"])
        expected = [1.7*m1["rms"]/norm, (1.7*m1["mean"]+2.)/norm,
                    .6*m2["rms"]/norm, (.6*m2["mean"]-3.)/norm]
        np.testing.assert_allclose(fit["coefficients"], expected, atol=1e-10, rtol=1e-10)
        rows, rhs = [], []
        for i in np.flatnonzero(support):
            h1, h2 = g["h1"][i], g["h2"][i]
            Ai = np.column_stack((z1[i]*(R @ h1), R @ h1, -z2[i]*h2, -h2))
            rows.extend(math.sqrt(weights[i])*Ai)
            rhs.extend(-math.sqrt(weights[i])*t)
        reference = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0]
        np.testing.assert_allclose(fit["coefficients"], reference, atol=1e-10, rtol=1e-10)

    def test_rank_slope_and_numerical_failure_are_full_cost(self):
        x1, x2, d1, d2, cal, pose = fixture()
        g = s.base.prepare_geometry(x1, x2, cal)
        R, t = s.base.validate_pose(pose)
        z1, _ = s.normalize_depth(d1)
        z2, _ = s.normalize_depth(d2)
        n = len(d1)
        rank = s.fit_calibration(R, t, g, np.zeros(n), np.zeros(n), np.ones(n, bool), np.ones(n))
        self.assertEqual(rank["reason"], "rank_deficient")
        negative = s.fit_calibration(R, -t, g, z1, z2, np.ones(n, bool), np.ones(n))
        self.assertEqual(negative["reason"], "nonpositive_fitted_slope")
        empty = s.fit_calibration(R, t, g, z1, z2, np.zeros(n, bool), np.ones(n))
        self.assertEqual(empty["reason"], "insufficient_fit_support")
        with patch.object(np.linalg, "svd", side_effect=np.linalg.LinAlgError("artificial")):
            failed = s.fit_calibration(R, t, g, z1, z2, np.ones(n, bool), np.ones(n))
        self.assertEqual(failed["reason"], "svd_failure")
        for fit in (rank, negative, empty, failed):
            costs, meta = s.validation_costs(fit, R, t, g, z1, z2, np.arange(n))
            np.testing.assert_array_equal(costs, np.ones(n))
            self.assertEqual(meta["fit_failed_direction_count"], 2*n)
            json.dumps(fit, allow_nan=False)

    def test_invalid_source_and_transformed_directions_keep_denominator(self):
        x1, x2, _, _, cal, _ = fixture(4)
        g = s.base.prepare_geometry(x1, x2, cal)
        fit = {"status": "completed", "reason": None, "coefficients": [1., 0., 1., 0.]}
        z1, z2 = np.array([-1., 0., 2., 1e308]), np.array([2., 2., -1., 1e308])
        costs, meta = s.validation_costs(fit, np.eye(3), np.array([0., 0., -3.]), g, z1, z2, np.arange(4))
        self.assertEqual(meta["direction_denominator"], 8)
        self.assertEqual(meta["forward"]["source_invalid_count"], 2)
        self.assertEqual(meta["reverse"]["source_invalid_count"], 1)
        self.assertGreaterEqual(meta["forward"]["transformed_nonpositive_or_nonfinite_z_count"], 3)
        self.assertTrue(np.isfinite(costs).all())
        self.assertTrue(np.all((costs >= 0) & (costs <= 1)))
        # Isolate true arithmetic overflow from invalid-input rejection.
        g["h1"][:, 0] = 10.
        _, overflow = s.validation_costs(fit, np.eye(3), np.array([1., 0., 0.]), g, z1, z2, np.arange(4))
        self.assertGreaterEqual(overflow["forward"]["transformed_nonfinite_count"], 1)


class SelectorTests(unittest.TestCase):
    def test_pure_kernel_does_not_read_files_or_mutate_inputs(self):
        x1, x2, d1, d2, cal, pose = fixture()
        candidates = bank(x1, x2, cal, [pose]*4)
        before = s.base.json_safe({"x1": x1, "x2": x2, "d1": d1, "d2": d2,
                                   "calibration": cal, "candidates": candidates})
        with patch("builtins.open", side_effect=AssertionError("File reads forbidden")), \
             patch.object(Path, "open", side_effect=AssertionError("Path reads forbidden")):
            result = s.score_pair("pure", x1, x2, d1, d2, cal, candidates)
        after = s.base.json_safe({"x1": x1, "x2": x2, "d1": d1, "d2": d2,
                                  "calibration": cal, "candidates": candidates})
        self.assertEqual(before, after)
        self.assertEqual(result["key"], "pure")

    def test_four_folds_random_sizes_determinism_and_coordinate_ties(self):
        x1, *_ = fixture()
        a, b, w, meta, _ = s.partitions("key", x1)
        a2, b2, w2, *_ = s.partitions("key", x1)
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(b, b2)
        np.testing.assert_array_equal(w, w2)
        self.assertEqual(np.bincount(b, minlength=4).tolist(), meta["counts"])
        self.assertAlmostEqual(float(w.sum()), 1.)
        for k in range(4):
            self.assertAlmostEqual(float(w[a == k].sum()), .25)
        order = sorted(range(len(x1)), key=lambda i: (hashlib.sha256((s.RANDOM_SALT+"key\0"+str(i)).encode()).digest(), i))
        self.assertTrue(np.all(b[order[:meta["counts"][0]]] == 0))
        tied = np.repeat([[1., 2.]], 12, axis=0)
        _, random, weights, meta, _ = s.partitions("key", tied)
        self.assertIsNone(random)
        self.assertIsNone(weights)
        self.assertEqual(meta["counts"], [12, 0, 0, 0])

    def test_exact_case_all_selectors_json_and_fixed_ties(self):
        result = call(fixture())
        self.assertEqual(list(result["selectors"]), list(s.SELECTORS))
        self.assertEqual(result["arms"], list(s.ARMS))
        self.assertEqual(result["counters"], {"candidate_count": 4, "fit_calls": 40,
            "svd_calls": 40, "fit_completed_count": 40, "fit_unavailable_count": 0})
        for selector in result["selectors"].values():
            self.assertEqual(selector["selected_index"], 0)
            self.assertLess(selector["score"], 1e-20)
        for row in result["candidates"]:
            self.assertEqual(sum(len(v) for v in row["fits"].values()), 10)
            for fits in row["fits"].values():
                self.assertTrue(all(v["fit"]["status"] == "completed" for v in fits))
        json.dumps(result, allow_nan=False)

    def test_positive_affine_depth_and_translation_gauge_invariance(self):
        args = fixture()
        # Small fixed non-affine distortion avoids an all-zero score comparison.
        args[2][:] += .01*np.sin(np.arange(len(args[2])))
        baseline = call(args)
        x1, x2, d1, d2, cal, pose = args
        moved = copy.deepcopy(pose)
        moved["t"] *= 7.25
        changed = call((x1, x2, 2.7*d1-6., .13*d2+4., cal, moved))
        for name in s.SELECTORS:
            np.testing.assert_allclose(baseline["selectors"][name]["scores"],
                changed["selectors"][name]["scores"], atol=1e-10, rtol=1e-10)
            self.assertEqual(baseline["selectors"][name]["selected_index"], changed["selectors"][name]["selected_index"])

    def test_zero_spread_unrepresentable_and_empty_spatial_fallback(self):
        args = fixture()
        x1, x2, d1, d2, cal, pose = args
        for raw, reason in ((np.ones(len(d1)), "zero_depth_spread"),
                            (np.full(len(d1), 1e308), "nonrepresentable_depth_normalization")):
            result = call((x1, x2, raw, d2, cal, pose))
            self.assertEqual(result["normalization"]["depth1"]["reason"], reason)
            for row in result["candidates"]:
                self.assertEqual(set(row["depth_scores"].values()), {1.})
            json.dumps(result, allow_nan=False)
        collapsed = x1.copy()
        collapsed[:, 0] = collapsed[0, 0]
        result = call((collapsed, x2, d1, d2, cal, pose))
        self.assertEqual(result["spatial_partition"]["status"], "unavailable")
        for name in s.SELECTORS[2:]:
            self.assertEqual(result["selectors"][name]["selected_index"], result["selectors"]["pixel_msac"]["selected_index"])
            self.assertTrue(all(row["depth_scores"][name] == 1. for row in result["candidates"]))

    def test_disjoint_fold_fit_support_and_identical_validation_mass(self):
        args = fixture()
        x1, x2, d1, d2, cal, pose = args
        x2[:9, 0] += 100.
        result = call((x1, x2, d1, d2, cal, pose))
        g = s.base.prepare_geometry(x1, x2, cal)
        gcost, _ = s.pixel_costs(pose, g)
        valid = gcost < 1.
        spatial = np.asarray(result["spatial_partition"]["ids"])
        random = np.asarray(result["random_partition"]["ids"])
        for row in result["candidates"]:
            self.assertEqual(row["fits"]["global_affine_ls"][0]["fit"]["training_count"], int(valid.sum()))
            for name, ids in (("spatial_transfer", spatial), ("random_point_transfer", random)):
                self.assertAlmostEqual(sum(f["validation_weight_mass"] for f in row["fits"][name]), 1.)
                self.assertEqual(sum(f["validation"]["direction_denominator"] for f in row["fits"][name]), 2*len(x1))
                for k, f in enumerate(row["fits"][name]):
                    self.assertEqual(f["fit"]["training_count"], int(np.count_nonzero(valid & (ids != k))))

    def test_large_bounded_fixture_and_saturated_candidate(self):
        args = fixture(10000)
        bad = copy.deepcopy(args[-1])
        bad["t"] *= -1
        result = call(args, [args[-1], bad, args[-1], args[-1]])
        for selector in result["selectors"].values():
            self.assertEqual(selector["selected_index"], 0)
        self.assertEqual(result["candidates"][1]["fit_support_count"], 0)
        self.assertEqual(result["candidates"][1]["scores"]["spatial_transfer"], 2.)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
