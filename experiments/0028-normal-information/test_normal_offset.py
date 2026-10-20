"""Small manufactured arithmetic tests; importing does not generate fixtures."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("normal_offset", Path(__file__).with_name("normal_offset.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def plane(shift=.4):
    h, w = 33, 37
    K = np.array([[200., 0., 18.], [0., 220., 16.], [0., 0., 1.]])
    y, x = np.mgrid[:h, :w]
    # The physical plane is 0.6 X - 0.3 Y + Z = 3.
    z = 3. / (1. + .6 * (x - 18.) / 200. - .3 * (y - 16.) / 220.)
    n = np.broadcast_to(np.array([-.6, .3, -1.]), (h, w, 3)).copy()
    return z + shift, n, np.ones((h, w), dtype=bool), K, z


@pytest.mark.parametrize("method", mod.METHODS)
def test_shifted_slanted_plane_recovery(method):
    d, n, mask, K, z = plane()
    prepared = mod.prepare(d, n, mask, K)
    result = mod.solve(prepared, method)
    assert result["status"] == "completed" and result["admissible"]
    target = -.4 / prepared["median_raw"]
    # Prescribed central differences approximate a rational plane depth:
    # their O((slope/focal)^2) truncation prevents exact algebraic recovery.
    assert abs(result["b"] - target) < 5e-5
    corrected = mod.corrected_depth(prepared, result)
    np.testing.assert_allclose(corrected, z / prepared["median_raw"], rtol=0., atol=5e-5)
    assert result["physical"]["positive_finite_count"] == d.size
    json.dumps(result, allow_nan=False)


def test_frontoparallel_angular_uninformative_but_tilted_ls_collapses():
    d, n, mask, K, _ = plane()
    d[:] = 2.
    n[:] = [.01, 0., -1.]  # A wrong slight tilt, constant measured depth.
    prepared = mod.prepare(d, n, mask, K)
    angular = mod.solve(prepared, "ANG")
    assert angular["status"] == "uninformative"
    assert angular["b"] == 0. and angular["admissible"]
    assert not angular["correction_applied"]
    assert len(angular["history"]) == 66
    assert "optimizer" not in angular
    for method in ("LS", "IRLS"):
        result = mod.solve(prepared, method)
        assert result["status"] == "unavailable" and not result["admissible"]
        assert abs(result["b"] + 1.) <= 8 * np.finfo(float).eps
        assert result["reason"] == "offset_out_of_domain_or_nonpositive"
        assert mod.corrected_depth(prepared, result) is None


def test_exact_frontoparallel_ls_has_zero_information():
    d, n, mask, K, _ = plane()
    d[:] = 2.
    n[:] = [0., 0., -1.]
    result = mod.solve(mod.prepare(d, n, mask, K), "LS")
    assert result["reason"] == "zero_ls_information"
    assert result["b"] is None


@pytest.mark.parametrize("method", mod.METHODS)
def test_independent_normal_sign_changes_leave_fit_unchanged(method):
    d, n, mask, K, _ = plane()
    original = mod.prepare(d, n, mask, K)
    n[::2, 1::2] *= -1
    flipped = mod.prepare(d, n, mask, K)
    a, b = mod.solve(original, method), mod.solve(flipped, method)
    assert a["status"] == b["status"]
    assert a["b"] == b["b"]
    assert a["angular_loss"] == b["angular_loss"]
    np.testing.assert_array_equal(original["samples"]["flat_indices"], flipped["samples"]["flat_indices"])


@pytest.mark.parametrize("method", mod.METHODS)
def test_metric_scale_equivariance_after_whole_image_normalization(method):
    d, n, mask, K, _ = plane()
    p = mod.prepare(d, n, mask, K)
    scaled = mod.prepare(d * 8., n, mask, K)
    assert scaled["median_raw"] == p["median_raw"] * 8.
    np.testing.assert_array_equal(p["normalized_depth"], scaled["normalized_depth"])
    a, b = mod.solve(p, method), mod.solve(scaled, method)
    assert a["b"] == b["b"] and a["status"] == b["status"]
    np.testing.assert_array_equal(mod.corrected_depth(p, a), mod.corrected_depth(scaled, b))


def test_fixed_stencil_hole_normal_and_discontinuity_counts():
    h = w = 13
    d = np.ones((h, w))
    n = np.broadcast_to([0., 0., -1.], (h, w, 3)).copy()
    mask = np.ones((h, w), dtype=bool)
    K = np.array([[30., 0., 6.], [0., 30., 6.], [0., 0., 1.]])
    # Nine centers (2,6,10)^2; exactly three separate stencil exclusions.
    mask[1, 1] = False
    n[5, 5] = np.nan
    d[9, 9] = 2.
    p = mod.prepare(d, n, mask, K)
    assert p["counts"]["grid_samples"] == 9
    assert p["counts"]["depth_stencil_valid"] == 8
    assert p["counts"]["normal_stencil_valid"] == 8
    assert p["counts"]["joint_stencil_valid"] == 7
    assert p["counts"]["discontinuity_rejected"] == 1
    assert p["counts"]["eligible_samples"] == 6
    assert {tuple(v) for v in p["samples"]["xy"]} == {
        (6, 2), (10, 2), (2, 6), (10, 6), (2, 10), (6, 10)}


def test_domain_uses_unsampled_original_valid_pixels_and_all_pixel_positivity():
    d, n, mask, K, _ = plane()
    d[0, 0] = 1e-7
    p = mod.prepare(d, n, mask, K)
    assert p["reason"] == "zero_not_in_physical_domain"
    for method in mod.METHODS:
        r = mod.solve(p, method)
        assert not r["admissible"] and mod.corrected_depth(p, r) is None


def test_out_of_domain_ls_not_clipped():
    d, n, mask, K, _ = plane(shift=0.)
    # Deliberately inconsistent normal polarity in x/y relative to the shape.
    n[:] = [.6, -.3, -1.]
    p = mod.prepare(d, n, mask, K)
    r = mod.solve(p, "LS")
    assert r["status"] == "unavailable"
    assert r["b"] < p["domain"][0]
    assert r["history"][0]["b"] == r["b"]
    assert mod.corrected_depth(p, r) is None


def test_angular_trace_is_finite_and_includes_zero_and_fixed_grid():
    d, n, mask, K, _ = plane()
    p = mod.prepare(d, n, mask, K)
    r = mod.solve(p)
    assert sum(v["phase"] == "grid" for v in r["history"]) == 65
    assert sum(v["phase"] == "extra_zero" for v in r["history"]) == 1
    assert r["optimizer"]["success"] and r["optimizer"]["nfev"] <= 64
    assert r["history"][r["selected_evaluation"]]["b"] == r["b"]
    assert r["angular_loss"] <= r["zero_loss"]
    json.dumps(r, allow_nan=False)


def test_wrong_shapes_and_true_arithmetic_overflow_are_not_silent():
    d, n, mask, K, _ = plane()
    with pytest.raises(ValueError):
        mod.prepare(d, n[:, :-1], mask, K)
    with pytest.raises(ValueError):
        mod.prepare(d, n, mask.astype(float), K)
    bad_K = K.copy()
    bad_K[2, 0] = .1
    with pytest.raises(ValueError):
        mod.prepare(d, n, mask, bad_K)
    tiny = np.full_like(d, 1e-300)
    tiny[0, 0] = 1e300
    with pytest.raises(FloatingPointError):
        mod.prepare(tiny, n, mask, K)


def test_empty_support_is_explicit_without_optimizer():
    d, n, mask, K, _ = plane()
    mask[:] = False
    p = mod.prepare(d, n, mask, K)
    assert p["reason"] == "no_positive_model_valid_depth"
    for method in mod.METHODS:
        r = mod.solve(p, method)
        assert r["status"] == "unavailable" and r["history"] == []
        json.dumps(r, allow_nan=False)
