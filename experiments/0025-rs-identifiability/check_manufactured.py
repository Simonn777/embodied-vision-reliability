#!/usr/bin/env python3
"""One fixed, manufactured RS/depth identifiability qualification; not an experiment.

No random seeds, fitting, assets, network, model, or production imports. Six fixed
nonplanar fixtures are declared below before execution. Observations are generated
once by implicit row projection, then held FIXED for every parameter derivative.
The linear known-R/known-angular-velocity branch uses manufactured known inputs,
not an equal-information RGB claim. Rank witnesses are not generic proofs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


SCHEMA = "evr0025.manufactured-rank.v1"
POINTS = np.array([
    [-1.8, -1.2, 3.2], [1.5, -1.1, 4.5], [-1.3, 0.9, 2.8],
    [1.9, 1.2, 5.2], [0.2, -0.6, 2.3], [-0.4, 1.6, 4.1],
    [0.8, 0.4, 3.7], [-1.6, 0.2, 5.5],
], dtype=np.float64)
WIDTH, HEIGHT = 1280, 960
K = np.array([[620., 0., 639.5], [0., 600., 479.5], [0., 0., 1.]])
READOUT = 0.04  # Seconds from first pixel-center row to last pixel-center row.
STEPS = (0.002, 0.001, 0.0005, 0.00025)
RELATIVE_RANK_THRESHOLD = 1e-8
FD_SAFETY_FACTOR = 100.
SCALES = np.array([1.] * 6 + [1. / READOUT] * 12 + [1.] * 3)
NAMES = (["left_R_%s" % a for a in "xyz"] + ["t_%s" % a for a in "xyz"]
         + ["omega1_%s" % a for a in "xyz"] + ["v1_%s" % a for a in "xyz"]
         + ["omega2_%s" % a for a in "xyz"] + ["v2_%s" % a for a in "xyz"]
         + ["b1", "a2", "b2"])


def skew(x):
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def rotation_and_left_jacobian(w):
    """Exact SO(3) exponential and SE(3) translational left Jacobian."""
    theta2 = float(np.dot(w, w))
    W = skew(w)
    W2 = W @ W
    if theta2 < 1e-8:
        a = 1. - theta2 / 6. + theta2**2 / 120. - theta2**3 / 5040.
        b = .5 - theta2 / 24. + theta2**2 / 720. - theta2**3 / 40320.
        c = 1. / 6. - theta2 / 120. + theta2**2 / 5040. - theta2**3 / 362880.
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta) / theta
        b = (1. - math.cos(theta)) / theta2
        c = (theta - math.sin(theta)) / (theta2 * theta)
    return np.eye(3) + a * W + b * W2, np.eye(3) + b * W + c * W2


def exp_se3(xi, s=1.):
    Q, V = rotation_and_left_jacobian(s * xi[:3])
    return Q, V @ (s * xi[3:])


def transform(p, R, t):
    return R @ p + t


def truth(kind):
    R = rotation_and_left_jacobian(np.array([.20, -.13, .11]))[0]
    t = np.array([.42, -.16, .25])
    xi1 = np.array([2.1, -1.7, .9, .8, -.35, .42])
    xi2 = np.array([-1.4, .75, 1.8, -.55, .44, .2])
    readout = READOUT
    if kind != "generic":
        xi1 = np.zeros(6)
        xi2 = np.zeros(6)
    if kind == "zero_readout":
        readout = 0.
    if kind == "identity_rotation":
        R = np.eye(3)
    if kind == "vertical":
        R, t = np.eye(3), np.array([0., .30, 0.])
    if kind == "equal_row":
        R, t = np.eye(3), np.array([.30, 0., 0.])
    return dict(R=R, t=t, xi1=xi1, xi2=xi2, b1=.18, a2=1.17,
                b2=-.09, readout1=readout, readout2=readout)


def project_implicit(X, R, t, xi, readout):
    """Bisection of s - readout*(projected_y-center)/(H-1); no iteration tuning."""
    middle = transform(X, R, t)

    def evaluate(s):
        Q, u = exp_se3(xi, s)
        p = transform(middle, Q, u)
        if not np.all(np.isfinite(p)) or p[2] <= 0.:
            raise ValueError("manufactured point behind camera/nonfinite")
        ray = p / p[2]
        uv = (K @ ray)[:2]
        f = s - readout * (uv[1] - (HEIGHT - 1.) / 2.) / (HEIGHT - 1.)
        return float(f), ray, uv, float(p[2])

    if readout == 0.:
        s = 0.
    else:
        lo, hi = -readout / 2., readout / 2.
        flo, fhi = evaluate(lo)[0], evaluate(hi)[0]
        if not (flo < 0. < fhi):
            raise ValueError("fixed point has no bracketed visible row root")
        # No search for better points. A uniqueness/conditioning diagnostic only:
        # sampled positive derivatives are not claimed to prove global uniqueness.
        samples = np.linspace(lo, hi, 33)
        fs = np.array([evaluate(v)[0] for v in samples])
        if not np.all(np.diff(fs) > 0.):
            raise ValueError("fixed row equation not monotone on sampled bracket")
        for _ in range(80):
            mid = (lo + hi) / 2.
            if evaluate(mid)[0] < 0.:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2.
    f, ray, uv, z = evaluate(s)
    if not (0. <= uv[0] <= WIDTH - 1. and 0. <= uv[1] <= HEIGHT - 1.):
        raise ValueError("fixed point outside manufactured sensor")
    if abs(f) > 1e-13:
        raise ValueError("implicit row residual exceeds fixed tolerance")
    return dict(time=s, ray=ray, pixel=uv, depth=z, row_residual=f)


def observations(kind, n):
    q = truth(kind)
    a = [project_implicit(X, np.eye(3), np.zeros(3), q["xi1"], q["readout1"])
         for X in POINTS[:n]]
    b = [project_implicit(X, q["R"], q["t"], q["xi2"], q["readout2"])
         for X in POINTS[:n]]
    data = {"kind": kind, "count": n, "points": POINTS[:n].copy(), "truth": q,
            "r1": np.array([v["ray"] for v in a]), "r2": np.array([v["ray"] for v in b]),
            "s1": np.array([v["time"] for v in a]), "s2": np.array([v["time"] for v in b]),
            "d1": np.array([v["depth"] for v in a]) - q["b1"],
            "d2": (np.array([v["depth"] for v in b]) - q["b2"]) / q["a2"],
            "projection1": a, "projection2": b}
    for k in ("r1", "r2", "s1", "s2", "d1", "d2"):
        data[k].setflags(write=False)
    return data


def decode(data, x):
    q = data["truth"]
    return dict(R=rotation_and_left_jacobian(x[:3])[0] @ q["R"], t=q["t"] + x[3:6],
                xi1=q["xi1"] + x[6:12], xi2=q["xi2"] + x[12:18],
                b1=q["b1"] + x[18], a2=q["a2"] + x[19], b2=q["b2"] + x[20])


def residual(data, x):
    """Observed times, rays and predicted depth values never change here."""
    q = decode(data, x)
    if q["a2"] <= 0.:
        raise ValueError("nonpositive affine scale in manufactured perturbation")
    z1 = data["d1"] + q["b1"]
    z2 = q["a2"] * data["d2"] + q["b2"]
    if np.any(z1 <= 0.) or np.any(z2 <= 0.):
        raise ValueError("nonpositive depth in manufactured perturbation")
    rows = []
    for i in range(data["count"]):
        Q1, u1 = exp_se3(q["xi1"], data["s1"][i])
        Q2, u2 = exp_se3(q["xi2"], data["s2"][i])
        p1mid = Q1.T @ (z1[i] * data["r1"][i] - u1)
        predicted = transform(transform(p1mid, q["R"], q["t"]), Q2, u2)
        rows.append(z2[i] * data["r2"][i] - predicted)
    result = np.array(rows).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite manufactured residual")
    return result


def analytic_zero_twist(data):
    q = data["truth"]
    if np.any(q["xi1"]) or np.any(q["xi2"]):
        raise ValueError("zero-twist analytic formula used outside its scope")
    rows = []
    for i in range(data["count"]):
        r1, r2 = data["r1"][i], data["r2"][i]
        s1, s2 = data["s1"][i], data["s2"][i]
        p1 = (data["d1"][i] + q["b1"]) * r1
        p2 = q["R"] @ p1 + q["t"]
        rows.append(np.column_stack((skew(q["R"] @ p1), -np.eye(3),
                    -s1 * q["R"] @ skew(p1), s1 * q["R"], s2 * skew(p2),
                    -s2 * np.eye(3), -q["R"] @ r1, data["d2"][i] * r2, r2)))
    return np.vstack(rows)


def finite_jacobian(data, h):
    J = np.empty((3 * data["count"], 21))
    for j in range(21):
        delta = np.zeros(21)
        delta[j] = h * SCALES[j]
        J[:, j] = (residual(data, delta) - residual(data, -delta)) / (2. * h)
    return J  # Derivative in the declared dimensionless column chart, Jphysical*D.


def svd_record(J, threshold):
    s = np.linalg.svd(J, compute_uv=False)
    return dict(singular_values=s, rank=int(np.sum(s > threshold)), threshold=float(threshold),
                minimum=float(s[-1]), maximum=float(s[0]), shape=list(J.shape))


def full_check(data):
    at_truth = residual(data, np.zeros(21))
    if np.max(np.abs(at_truth)) > 1e-11:
        raise ValueError("manufactured truth does not satisfy row observations")
    central = [finite_jacobian(data, h) for h in STEPS]
    richardson = [(4. * central[i + 1] - central[i]) / 3. for i in range(len(STEPS) - 1)]
    J = richardson[-1]
    disagreements = [float(np.linalg.norm(richardson[i + 1] - richardson[i], 2))
                     for i in range(len(richardson) - 1)]
    smax = float(np.linalg.svd(J, compute_uv=False)[0])
    fd_floor = FD_SAFETY_FACTOR * max(disagreements + [np.finfo(float).eps * max(1., smax)])
    threshold = max(RELATIVE_RANK_THRESHOLD * smax, fd_floor)
    records = [svd_record(z, threshold) for z in richardson]
    out = dict(kind=data["kind"], count=data["count"], observations=data,
               truth_max_abs_residual=float(np.max(np.abs(at_truth))),
               jacobian_scaled=J, jacobian_physical=J / SCALES,
               central_steps=list(STEPS), richardson=records,
               richardson_spectral_disagreements=disagreements,
               finite_difference_floor=fd_floor,
               rank_record=svd_record(J, threshold),
               physical_spectrum=np.linalg.svd(J / SCALES, compute_uv=False),
               fixed_relative_threshold=RELATIVE_RANK_THRESHOLD,
               full_rank_witness=all(v["rank"] == 21 for v in records))
    if not np.any(data["truth"]["xi1"]) and not np.any(data["truth"]["xi2"]):
        A = analytic_zero_twist(data) * SCALES
        delta = float(np.linalg.norm(A - J, 2))
        tolerance = 1e-7 * max(1., float(np.linalg.norm(A, 2)))
        if delta > tolerance:
            raise ValueError("finite difference disagrees with independent zero-twist analytic J")
        out["analytic_check"] = dict(jacobian_scaled=A, spectral_difference=delta,
                                     tolerance=tolerance, rank_record=svd_record(A, threshold))
    if data["kind"] == "vertical":
        gamma = READOUT * K[1, 1] / (HEIGHT - 1.)
        null = np.zeros(21)
        null[10] = null[16] = 1. / gamma  # Physical v1_y / v2_y.
        null[18] = null[20] = 1.
        out["known_offset_null_physical"] = null
        out["known_offset_null_residual"] = analytic_zero_twist(data) @ null
        # Predeclared finite, positivity-preserving delta; no selection after output.
        out["finite_null_delta"] = .01
        out["finite_null_max_abs_residual"] = float(np.max(np.abs(residual(data, .01 * null))))
        if out["finite_null_max_abs_residual"] > 1e-10:
            raise ValueError("exact finite offset ambiguity counterexample failed")
    if data["kind"] == "equal_row":
        q = data["truth"]
        columns = []
        for k in range(6):
            c = np.eye(6)[k]
            ad = np.r_[q["R"] @ c[:3], np.cross(q["t"], q["R"] @ c[:3]) + q["R"] @ c[3:]]
            n = np.zeros(21)
            n[6:12], n[12:18] = c, ad
            columns.append(n)
        N = np.column_stack(columns)
        out["known_common_twist_nulls_physical"] = N
        out["known_common_twist_null_residual"] = analytic_zero_twist(data) @ N
        out["equal_row_time_difference_max"] = float(np.max(np.abs(data["s1"] - data["s2"])))
        if np.max(np.abs(out["known_common_twist_null_residual"])) > 1e-10:
            raise ValueError("analytic common-twist null family failed")
    return out


def known_rotation_linear(data, n):
    """Known middle R and both angular rates; NO normalization of dewarped rays."""
    q = data["truth"]
    rows, rhs, unnormalized = [], [], []
    for i in range(n):
        s1, s2 = data["s1"][i], data["s2"][i]
        Q1, V1 = rotation_and_left_jacobian(s1 * q["xi1"][:3])
        Q2, V2 = rotation_and_left_jacobian(s2 * q["xi2"][:3])
        rb1, rb2 = Q1.T @ data["r1"][i], Q2.T @ data["r2"][i]
        B1, B2 = Q1.T @ V1 * s1, Q2.T @ V2 * s2
        rows.append(np.column_stack((-np.eye(3), q["R"] @ B1, -B2,
                    -q["R"] @ rb1, data["d2"][i] * rb2, rb2)))
        rhs.append(q["R"] @ (data["d1"][i] * rb1))
        unnormalized.append([rb1, rb2])
    A, y = np.vstack(rows), np.array(rhs).reshape(-1)
    beta = np.r_[q["t"], q["xi1"][3:], q["xi2"][3:], q["b1"], q["a2"], q["b2"]]
    scale = np.array([1.] * 3 + [1. / READOUT] * 6 + [1.] * 3)
    As = A * scale
    error = float(np.max(np.abs(A @ beta - y)))
    if error > 1e-11:
        raise ValueError("known-R/angular linear system fails manufactured truth")
    singular = np.linalg.svd(As, compute_uv=False)
    threshold = RELATIVE_RANK_THRESHOLD * singular[0]
    record = dict(kind=data["kind"], count=n, known_middle_R=q["R"],
                  known_omega1=q["xi1"][:3], known_omega2=q["xi2"][:3],
                  unknown_order=["t(3)", "v1(3)", "v2(3)", "b1", "a2", "b2"],
                  matrix_physical=A, rhs=y, truth_beta=beta, column_scales=scale,
                  unnormalized_dewarped_rays=unnormalized, truth_max_abs_residual=error,
                  rank_record=svd_record(As, threshold),
                  physical_spectrum=np.linalg.svd(A, compute_uv=False),
                  full_rank_witness=int(np.sum(singular > threshold)) == 12)
    if data["kind"] == "vertical":
        gamma = READOUT * K[1, 1] / (HEIGHT - 1.)
        z = np.zeros(12)
        z[4] = z[7] = 1. / gamma
        z[9] = z[11] = 1.
        record["known_offset_null_physical"] = z
        record["known_offset_null_residual"] = A @ z
        if np.max(np.abs(A @ z)) > 1e-10:
            raise ValueError("linear offset-null counterexample failed")
    if data["kind"] == "identity_rotation":
        L = np.column_stack((np.ones(n), data["s1"][:n], data["s2"][:n]))
        U, singular_L, _ = np.linalg.svd(L, full_matrices=True)
        if singular_L[-1] <= 1e-10 * singular_L[0]:
            raise ValueError("fixed identity-R fixture has deficient row-time design")
        M = U[:, 3:]
        # Each column of M eliminates the three temporal translation bases.
        projected = np.einsum("ik,icj->kcj", M, A.reshape(n, 3, 12)).reshape(-1, 12)
        offset_vectors = projected[:, [9, 11]].reshape(n - 3, 3, 2)
        yz_error = float(np.max(np.abs(offset_vectors[:, 1:, :])))
        if yz_error > 1e-12:
            raise ValueError("identity-R offset columns not confined to x after elimination")
        remainder = projected[:, [9, 10, 11]]
        remainder_s = np.linalg.svd(remainder, compute_uv=False)
        remainder_rank = int(np.sum(remainder_s > RELATIVE_RANK_THRESHOLD * max(1., singular[0])))
        record["translation_elimination"] = dict(row_time_design=L, singular_values=singular_L,
            left_null_basis=M, projected_matrix=projected, offset_vectors=offset_vectors,
            offset_yz_max_abs=yz_error, remaining_singular_values=remainder_s,
            remaining_rank=remainder_rank,
            analytic_four_point_rank_upper_bound=11 if n == 4 else None)
        if n == 4 and (record["rank_record"]["rank"] > 11 or remainder_rank > 2):
            raise ValueError("numerical rank violates analytic four-point rank<=11 bound")
    return record


def safe(v):
    if isinstance(v, np.ndarray):
        return safe(v.tolist())
    if isinstance(v, np.generic):
        return safe(v.item())
    if isinstance(v, dict):
        return {k: safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("nonfinite JSON output")
    return v


def write(path, value):
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = dict(schema=SCHEMA, status="started", source_sha256=source,
                   python=sys.version, numpy=np.__version__, no_real_inputs=True,
                   parameter_count=21, parameter_order=NAMES, parameter_column_scales=SCALES,
                   finite_difference_steps=STEPS, fixed_rank_relative_threshold=RELATIVE_RANK_THRESHOLD,
                   finite_difference_floor_safety=FD_SAFETY_FACTOR,
                   fixed_points=POINTS, K=K, width=WIDTH, height=HEIGHT,
                   readout_seconds=READOUT, initial_generation_counts_as_no_fitting=True,
                   interpretation="manufactured local rank witnesses and exact counterexamples only")
    write(args.output_dir / "receipt.json", receipt)
    try:
        specifications = [("generic", 7), ("generic", 8), ("zero_motion", 8),
                          ("zero_readout", 8), ("vertical", 8), ("equal_row", 8)]
        full, generated = [], {}
        for kind, n in specifications:
            data = observations(kind, n)
            generated[(kind, n)] = data
            full.append(full_check(data))
        # Separately declared known-input branch. No optimization or source reuse
        # from full Jacobian: direct analytic affine-linear equations above.
        generated[("identity_rotation", 8)] = observations("identity_rotation", 8)
        linear = [known_rotation_linear(generated[(kind, 8)], n)
                  for kind in ("generic", "zero_motion", "vertical") for n in (4, 5)]
        linear.extend(known_rotation_linear(generated[("identity_rotation", 8)], n) for n in (4, 5))
        result = dict(schema=SCHEMA, full_unknown_checks=full, known_R_omega_linear_checks=linear,
                      additional_identity_rotation_fixture=generated[("identity_rotation", 8)],
                      interpretation="No generic identifiability theorem, real depth calibration, pose accuracy, or novelty follows.")
        path = args.output_dir / "mathematics.json"
        write(path, result)
        receipt.update(status="completed", result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       full_unknown_ranks=[dict(kind=x["kind"], count=x["count"],
                            rank=x["rank_record"]["rank"], witness=x["full_rank_witness"])
                            for x in full],
                       known_R_omega_ranks=[dict(kind=x["kind"], count=x["count"],
                            rank=x["rank_record"]["rank"], witness=x["full_rank_witness"])
                            for x in linear])
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        write(args.output_dir / "receipt.json", receipt)
    print(json.dumps(safe({k:receipt[k] for k in ("status", "wall_seconds", "full_unknown_ranks", "known_R_omega_ranks")})))


if __name__ == "__main__":
    main()
