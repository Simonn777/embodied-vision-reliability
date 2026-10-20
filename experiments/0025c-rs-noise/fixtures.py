"""Prospective 0025c manufactured inputs; generation only occurs via generate().

No images, caches, labels, model, native estimator, or network are accessed. The
only imported research source is the SHA-bound, published 0025 pure generator.
This file has been prepared without invoking it or generating any observations.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


SOURCE_RELATIVE = Path("experiments/0025-rs-identifiability/check_manufactured.py")
SOURCE_SHA256 = "0b058405fbc151b8765dafdb4e4a81f8c7e48f515da3d217b2df33b9a9565683"
SEED = 20261019
CASES = ("G", "N", "C", "E")
REALIZATIONS = 16
SCHEMA = "evr0025c.fixture.v1"


def _load_source():
    here = Path(__file__).resolve().parent
    # Support private preparation and the eventual repository experiment folder.
    possible = (here.parent / "embodied-vision-reliability" / SOURCE_RELATIVE,
                here.parent.parent / SOURCE_RELATIVE)
    matches = [p.resolve() for p in possible if p.is_file()]
    if len(set(matches)) != 1:
        raise RuntimeError("expected exactly one published 0025 generator path")
    path = matches[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("published 0025 generator SHA mismatch")
    spec = importlib.util.spec_from_file_location("_evr0025c_frozen0025", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen 0025 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _motion(old, case):
    q = deepcopy(old.truth("equal_row" if case == "E" else "generic"))
    if case == "N":
        q["R"] = old.rotation_and_left_jacobian(
            .1 * np.array([.20, -.13, .11], dtype=np.float64))[0]
    elif case == "C":
        q["xi2"] = q["xi1"].copy()
        q["R"], q["t"] = old.exp_se3(q["xi1"], .12)
    elif case not in ("G", "E"):
        raise ValueError("undeclared case")
    return q


def _clean(old, q):
    projections = []
    for R, t, xi, readout in (
        (np.eye(3), np.zeros(3), q["xi1"], q["readout1"]),
        (q["R"], q["t"], q["xi2"], q["readout2"]),
    ):
        projections.append([
            old.project_implicit(X, R, t, xi, readout) for X in old.POINTS
        ])
    pixels = np.array([[p["pixel"] for p in view] for view in projections])
    rays = np.array([[p["ray"] for p in view] for view in projections])
    times = np.array([[p["time"] for p in view] for view in projections])
    depths = np.array([[p["depth"] for p in view] for view in projections])
    row_residuals = np.array([
        [p["row_residual"] for p in view] for view in projections])
    a = np.array([1., q["a2"]], dtype=np.float64)
    b = np.array([q["b1"], q["b2"]], dtype=np.float64)
    raw_depths = (depths - b[:, None]) / a[:, None]
    uv = (pixels + .5) / np.array([old.WIDTH, old.HEIGHT])
    pattern_raw = np.sin(2. * np.pi * uv[:, :, 0]) + np.cos(2. * np.pi * uv[:, :, 1])
    pattern_centered = pattern_raw - np.mean(pattern_raw, axis=1, keepdims=True)
    pattern_rms = np.sqrt(np.mean(pattern_centered ** 2, axis=1))
    if not np.all(np.isfinite(pattern_rms)) or np.any(pattern_rms <= 0.):
        raise ValueError("undefined fixed spatial noise pattern")
    pattern = pattern_centered / pattern_rms[:, None]
    arrays = (pixels, rays, times, depths, row_residuals, raw_depths, pattern)
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("nonfinite clean manufactured input")
    return dict(points=old.POINTS.copy(), K=old.K.copy(), width=int(old.WIDTH),
                height=int(old.HEIGHT), readouts=np.array([q["readout1"], q["readout2"]]),
                pixels=pixels, rays=rays, times=times, axial_depths=depths,
                raw_depths=raw_depths, affine_scales=a, affine_shifts=b,
                implicit_row_residuals=row_residuals, normalized_pixels=uv,
                pattern_raw=pattern_raw, pattern_centered=pattern_centered,
                pattern_rms=pattern_rms, normalized_pattern=pattern)


def _truth(q, clean):
    return dict(R=q["R"].copy(), t=q["t"].copy(),
                w1=q["xi1"][:3].copy(), w2=q["xi2"][:3].copy(),
                v1=q["xi1"][3:].copy(), v2=q["xi2"][3:].copy(),
                b1=float(q["b1"]), a1=1., a2=float(q["a2"]), b2=float(q["b2"]),
                z1=clean["axial_depths"][0].copy(), z2=clean["axial_depths"][1].copy())


def _data(q, rays, times, depths, *, R=None, w1=None, w2=None):
    return dict(r1=rays[0].copy(), r2=rays[1].copy(),
                s1=times[0].copy(), s2=times[1].copy(),
                d1=depths[0].copy(), d2=depths[1].copy(),
                R=(q["R"] if R is None else R).copy(),
                w1=(q["xi1"][:3] if w1 is None else w1).copy(),
                w2=(q["xi2"][:3] if w2 is None else w2).copy())


def generate():
    """Return all 132 records; no solving, scoring, filtering, or resampling.

    Order is G,N,C,E; one clean record then P,S for realization 0..15 in
    each case. The single PCG64 stream draws pixel, c, epsilon, angular_axes
    in that exact order for every realization, regardless of P/S consumption.
    """
    old = _load_source()
    rng = np.random.Generator(np.random.PCG64(SEED))
    records = []
    for case in CASES:
        q = _motion(old, case)
        clean = _clean(old, q)
        truth = _truth(q, clean)
        common = dict(schema=SCHEMA, case=case, source_relative=SOURCE_RELATIVE.as_posix(),
                      source_sha256=SOURCE_SHA256, clean=clean, truth=truth)
        records.append(dict(deepcopy(common), regime="clean", realization=-1,
                            data=_data(q, clean["rays"], clean["times"], clean["raw_depths"]),
                            draws=None, observed_pixels=clean["pixels"].copy()))
        for realization in range(REALIZATIONS):
            before = deepcopy(rng.bit_generator.state)
            pixel = rng.standard_normal((2, 8, 2))
            coherent_c = rng.standard_normal(2)
            epsilon = rng.standard_normal((2, 8))
            axes = rng.standard_normal((3, 3))
            after = deepcopy(rng.bit_generator.state)
            axis_norms = np.linalg.norm(axes, axis=1)
            if not np.all(np.isfinite(axis_norms)) or np.any(axis_norms <= 0.):
                raise ValueError("undefined angular-input perturbation axis")
            unit_axes = axes / axis_norms[:, None]
            pixels = clean["pixels"] + .5 * pixel
            homogeneous = np.concatenate((pixels, np.ones((2, 8, 1))), axis=2)
            rays = np.linalg.solve(old.K, homogeneous.reshape(-1, 3).T).T.reshape(2, 8, 3)
            if not np.all(rays[:, :, 2] == 1.):
                raise ValueError("authored K did not produce unit-z rays")
            times = clean["readouts"][:, None] * (
                pixels[:, :, 1] - (old.HEIGHT - 1.) / 2.) / (old.HEIGHT - 1.)
            relative_error = .03 * (clean["normalized_pattern"] * coherent_c[:, None]
                                    + epsilon) / np.sqrt(2.)
            delta_depth = clean["axial_depths"] / clean["affine_scales"][:, None] * relative_error
            depths = clean["raw_depths"] + delta_depth
            delta_R_vector = np.deg2rad(.1) * unit_axes[0]
            delta_w = np.deg2rad(.5) * unit_axes[1:]
            delta_R = old.rotation_and_left_jacobian(delta_R_vector)[0]
            draws = dict(seed=SEED, rng_state_before=before, rng_state_after=after,
                         pixel_standard_normal=pixel, coherent_standard_normal=coherent_c,
                         epsilon_standard_normal=epsilon, angular_axes_standard_normal=axes,
                         angular_axes_unit=unit_axes, relative_depth_error=relative_error,
                         delta_raw_depth=delta_depth, delta_R_vector=delta_R_vector,
                         delta_angular_rates=delta_w)
            if not all(np.all(np.isfinite(a)) for a in (pixels, rays, times, depths, delta_R, delta_w)):
                raise ValueError("nonfinite noisy manufactured input")
            for regime in ("P", "S"):
                angular = {} if regime == "P" else dict(
                    R=delta_R @ q["R"], w1=q["xi1"][:3] + delta_w[0],
                    w2=q["xi2"][:3] + delta_w[1])
                records.append(dict(deepcopy(common), regime=regime, realization=realization,
                                    data=_data(q, rays, times, depths, **angular),
                                    draws=deepcopy(draws), observed_pixels=pixels.copy()))
    if len(records) != 132:
        raise RuntimeError("fixed cohort coverage violated")
    return records
