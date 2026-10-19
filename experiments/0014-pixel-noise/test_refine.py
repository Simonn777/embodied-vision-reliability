"""Artificial-only checks of the known pixel-noise refinement controls."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("pixel_noise_refine", Path(__file__).with_name("refine.py"))
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


def artificial(seed=701, noise=0.2, focals=(620.0, 850.0, 1100.0, 730.0)):
    rng = np.random.default_rng(seed)
    X = rng.uniform([-2.0, -1.5, 4.0], [2.0, 1.5, 9.0], (120, 3))
    R = refine.so3_exp(np.array([0.025, -0.06, 0.015]))
    t = np.array([0.7, -0.1, 0.2])
    t /= np.linalg.norm(t)
    Y = X @ R.T + t
    K1 = np.array([[focals[0], 0.0, 320.0], [0.0, focals[1], 240.0], [0.0, 0.0, 1.0]])
    K2 = np.array([[focals[2], 0.0, 400.0], [0.0, focals[3], 300.0], [0.0, 0.0, 1.0]])
    x = X[:, :2] / X[:, 2:]
    y = Y[:, :2] / Y[:, 2:]
    p1 = x * np.diag(K1)[:2] + K1[:2, 2] + rng.normal(0.0, noise, x.shape)
    p2 = y * np.diag(K2)[:2] + K2[:2, 2] + rng.normal(0.0, noise, y.shape)
    R0 = refine.so3_exp(np.array([0.015, -0.009, 0.012])) @ R
    t0 = refine.retract_pose(R, t, np.array([0., 0., 0., 0.04, -0.03]))[1]
    return p1, p2, K1, K2, R0, t0


def normalized(case):
    p1, p2, K1, K2, R, t = case
    x = np.column_stack(((p1 - K1[:2, 2]) / np.diag(K1)[:2], np.ones(len(p1))))
    y = np.column_stack(((p2 - K2[:2, 2]) / np.diag(K2)[:2], np.ones(len(p2))))
    return x, y, R, t, np.diag(K1)[:2], np.diag(K2)[:2]


@pytest.mark.parametrize("mode", ["N", "P"])
def test_analytic_jacobian_matches_central_difference(mode):
    args = normalized(artificial())
    x, y, R, t, f1, f2 = args
    r, J = refine.residuals_and_jacobian(*args, mode=mode)
    numeric = np.zeros_like(J)
    eps = 1e-7
    for j in range(5):
        delta = np.zeros(5)
        delta[j] = eps
        Rp, tp = refine.retract_pose(R, t, delta)
        Rm, tm = refine.retract_pose(R, t, -delta)
        rp = refine.residuals_and_jacobian(x, y, Rp, tp, f1, f2, mode)[0]
        rm = refine.residuals_and_jacobian(x, y, Rm, tm, f1, f2, mode)[0]
        numeric[:, j] = (rp - rm) / (2.0 * eps)
    np.testing.assert_allclose(J, numeric, rtol=2e-6, atol=2e-6)
    expected_gradient = 2.0 * J.T @ (r / (0.25 + r * r))
    numeric_gradient = np.zeros(5)
    for j in range(5):
        delta = np.zeros(5)
        delta[j] = eps
        Rp, tp = refine.retract_pose(R, t, delta)
        Rm, tm = refine.retract_pose(R, t, -delta)
        cp = refine.cauchy_cost(refine.residuals_and_jacobian(x, y, Rp, tp, f1, f2, mode)[0])
        cm = refine.cauchy_cost(refine.residuals_and_jacobian(x, y, Rm, tm, f1, f2, mode)[0])
        numeric_gradient[j] = (cp - cm) / (2 * eps)
    np.testing.assert_allclose(expected_gradient, numeric_gradient, rtol=2e-6, atol=3e-5)


def test_equal_focals_residual_jacobian_and_trajectory_equivalence():
    case = artificial(focals=(800.0, 800.0, 800.0, 800.0))
    args = normalized(case)
    n = refine.residuals_and_jacobian(*args, mode="N")
    p = refine.residuals_and_jacobian(*args, mode="P")
    for left, right in zip(n, p):
        np.testing.assert_array_equal(left, right)
    nresult, presult = (refine.refine_pose(*case, mode=mode) for mode in ("N", "P"))
    for key in ("R", "t", "initial_cost", "final_cost", "iterations", "termination_reason"):
        assert nresult[key] == presult[key]


def test_pixel_residual_is_raw_pixel_fundamental_sampson():
    case = artificial()
    p1, p2, K1, K2, R, t = case
    E = refine.skew(t) @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    h1, h2 = np.column_stack((p1, np.ones(len(p1)))), np.column_stack((p2, np.ones(len(p2))))
    g1, g2 = h2 @ F, h1 @ F.T
    direct = np.sum(h2 * g2, axis=1) / np.sqrt(np.sum(g1[:, :2]**2 + g2[:, :2]**2, axis=1))
    actual = refine.residuals_and_jacobian(*normalized(case), mode="P")[0]
    np.testing.assert_allclose(actual, direct, rtol=3e-13, atol=3e-12)


@pytest.mark.parametrize("mode", ["N", "P"])
def test_artificial_optimization_is_finite_strict_monotone_and_deterministic(mode):
    case = artificial()
    result = refine.refine_pose(*case, mode=mode)
    assert result["iteration_count"] <= 100
    assert result["accepted_steps"] > 0
    assert result["final_cost"] < 0.25 * result["initial_cost"]
    assert result["converged"]
    cost = result["initial_cost"]
    for row in result["iterations"]:
        assert row["cost_before"] == cost
        assert row["cost_after"] <= cost
        if row["accepted"]:
            assert row["trial_cost"] < cost
            assert row["cost_after"] == row["trial_cost"]
            assert row["lambda_after"] == max(1e-10, row["lambda_before"] / 10)
        elif row["status"].startswith("rejected"):
            assert row["lambda_after"] == min(1e10, row["lambda_before"] * 10)
        cost = row["cost_after"]
    np.testing.assert_allclose(np.asarray(result["R"]).T @ result["R"], np.eye(3), atol=2e-14)
    assert np.linalg.det(result["R"]) > 0.0
    assert abs(np.linalg.norm(result["t"]) - 1.0) < 2e-15
    json.dumps(result, allow_nan=False)
    replay = refine.refine_pose(*case, mode=mode)
    assert replay["iterations"] == result["iterations"]
    assert replay["R"] == result["R"] and replay["t"] == result["t"]


def test_translation_scale_and_sign_invariance_of_cost():
    args = list(normalized(artificial()))
    r, J = refine.residuals_and_jacobian(*args, mode="P")
    args[3] = args[3] * 1e200
    rbig, Jbig = refine.residuals_and_jacobian(*args, mode="P")
    np.testing.assert_allclose(rbig, r, rtol=1e-14, atol=2e-14)
    np.testing.assert_allclose(Jbig, J, rtol=1e-13, atol=2e-12)
    args[3] = -args[3]
    rneg = refine.residuals_and_jacobian(*args, mode="P")[0]
    np.testing.assert_allclose(rneg, -r, rtol=1e-14, atol=2e-14)
    assert refine.cauchy_cost(rneg) == pytest.approx(refine.cauchy_cost(r), rel=1e-14)


def test_invalid_pose_intrinsics_and_collapsed_denominator_rejected():
    case = list(artificial())
    case[5] = np.zeros(3)
    with pytest.raises(ValueError, match="nonzero"):
        refine.refine_pose(*case)
    case = list(artificial())
    case[4] = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="proper"):
        refine.refine_pose(*case)
    case = list(artificial())
    case[2] = case[2].copy()
    case[2][0, 1] = 0.01
    with pytest.raises(ValueError, match="PINHOLE"):
        refine.refine_pose(*case)
    x = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="denominator"):
        refine.residuals_and_jacobian(x, x, np.eye(3), [0., 0., 1.], [500., 500.], [500., 500.])
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        refine.residuals_and_jacobian(x, x, np.eye(3), [1., 0., 0.], [500., 500.], [500., 500.])
