"""Pure conditional RS linear algebra. No truth, fixture, I/O, or model access.

Column-L2 equilibration is undone before all physical checks. Velocity units are
length/second; no 1/readout column scaling is applied to the solves. A separate
1e-8 relative spectrum diagnostic never changes solver admission.
"""
import math

import numpy as np


WEAK_RTOL = 1e-8


def _array(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.all(np.isfinite(out)):
        raise ValueError("invalid shape or nonfinite " + name)
    return out


def _mm(a, b):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        out = np.einsum("ij,jk->ik", a, b, optimize=False)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("nonfinite matrix product")
    return out


def _mv(a, b):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        out = np.einsum("ij,j->i", a, b, optimize=False)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("nonfinite matrix-vector product")
    return out


def _norm(a):
    value = math.hypot(*np.asarray(a).reshape(-1).tolist())
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite norm")
    return value


def skew(v):
    x, y, z = _array(v, (3,), "rotation vector")
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_and_left_jacobian(w):
    """Exact SO(3) exponential and translational left Jacobian (stable at zero)."""
    w = _array(w, (3,), "rotation vector")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        theta2 = float(np.einsum("i,i->", w, w, optimize=False))
        W = skew(w)
        W2 = _mm(W, W)
        if theta2 < 1e-8:
            a = 1. - theta2 / 6. + theta2**2 / 120. - theta2**3 / 5040.
            b = .5 - theta2 / 24. + theta2**2 / 720. - theta2**3 / 40320.
            c = 1. / 6. - theta2 / 120. + theta2**2 / 5040. - theta2**3 / 362880.
        else:
            theta = math.sqrt(theta2)
            a = math.sin(theta) / theta
            b = (1. - math.cos(theta)) / theta2
            c = (theta - math.sin(theta)) / (theta2 * theta)
        Q = np.eye(3) + a * W + b * W2
        J = np.eye(3) + b * W + c * W2
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(J)):
        raise FloatingPointError("nonfinite SO3 arithmetic")
    return Q, J


def _geometry(data):
    r1 = np.asarray(data["r1"], dtype=np.float64)
    if r1.ndim != 2 or r1.shape[1] != 3 or len(r1) == 0:
        raise ValueError("r1 must have nonempty Nx3 shape")
    n = len(r1)
    r1 = _array(r1, (n, 3), "r1")
    r2 = _array(data["r2"], (n, 3), "r2")
    if not np.all(r1[:, 2] == 1.) or not np.all(r2[:, 2] == 1.):
        raise ValueError("input axial rays must have third component exactly one")
    s1 = _array(data["s1"], (n,), "s1")
    s2 = _array(data["s2"], (n,), "s2")
    R = _array(data["R"], (3, 3), "R")
    if (np.max(np.abs(_mm(R.T, R) - np.eye(3))) > 1e-8
            or abs(float(np.linalg.det(R)) - 1.) > 1e-8):
        raise ValueError("R must be a proper rotation")
    w1 = _array(data["w1"], (3,), "w1")
    w2 = _array(data["w2"], (3,), "w2")
    rb1, rb2, B1, B2 = (np.empty((n, 3)), np.empty((n, 3)),
                         np.empty((n, 3, 3)), np.empty((n, 3, 3)))
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for i in range(n):
            for r, s, w, rb, B in ((r1, s1, w1, rb1, B1), (r2, s2, w2, rb2, B2)):
                Q, J = rotation_and_left_jacobian(s[i] * w)
                rb[i] = _mv(Q.T, r[i])
                B[i] = _mm(Q.T, J) * s[i]
        Rrb1 = np.einsum("ij,nj->ni", R, rb1, optimize=False)
        RB1 = np.einsum("ij,njk->nik", R, B1, optimize=False)
    return dict(R=R, rbar1=rb1, rbar2=rb2, B1=B1, B2=B2,
                Rrbar1=Rrb1, RB1=RB1, pool_size=n)


def _indices(indices, n):
    if indices is None:
        return np.arange(n, dtype=np.int64)
    value = np.asarray(indices)
    if (value.ndim != 1 or value.size == 0 or value.dtype.kind not in "iu"
            or np.any(value < 0) or np.any(value >= n)
            or len(np.unique(value)) != len(value)):
        raise ValueError("indices must be distinct valid integers")
    return value.astype(np.int64, copy=True)


def build(data, indices=None, angular_only=False):
    """Build full12 or angular-only6 affine system; geometry covers whole pool."""
    g = _geometry(data)
    n = g["pool_size"]
    d1 = _array(data["d1"], (n,), "d1")
    d2 = _array(data["d2"], (n,), "d2")
    ids = _indices(indices, n)
    rows, rhs = [], []
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for i in ids:
            pieces = [-np.eye(3)]
            if not angular_only:
                pieces += [g["RB1"][i], -g["B2"][i]]
            pieces += [-g["Rrbar1"][i, :, None],
                       d2[i] * g["rbar2"][i, :, None], g["rbar2"][i, :, None]]
            rows.append(np.concatenate(pieces, axis=1))
            rhs.append(d1[i] * g["Rrbar1"][i])
        A, y = np.concatenate(rows, axis=0), np.concatenate(rhs)
    return dict(A=A, y=y, indices=ids, geometry=g)


def build_depth_free(data, indices=None):
    """Build exact homogeneous9 system. Does not inspect d1 or d2."""
    g = _geometry(data)
    ids = _indices(indices, g["pool_size"])
    rows = []
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for i in ids:
            n = np.cross(g["rbar2"][i], g["Rrbar1"][i])
            rows.append(np.concatenate([n, -_mv(g["RB1"][i].T, n),
                                        _mv(g["B2"][i].T, n)]))
    return dict(A=np.array(rows), y=None, indices=ids, geometry=g)


def _linear_svd(A, y, rank_rtol):
    """Direct, full-Vt SVD. y=None requests the homogeneous last right vector."""
    m, n = A.shape
    rtol = max(m, n) * np.finfo(np.float64).eps if rank_rtol is None else float(rank_rtol)
    if not math.isfinite(rtol) or not 0. < rtol < 1.:
        raise ValueError("rank_rtol must be finite and between zero and one")
    if not np.all(np.isfinite(A)) or (y is not None and not np.all(np.isfinite(y))):
        raise FloatingPointError("nonfinite linear system")
    norms = np.array([_norm(A[:, j]) for j in range(n)])
    if not np.all(np.isfinite(norms)):
        raise FloatingPointError("nonfinite column norms")
    scales = np.ones(n)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        np.divide(1., norms, out=scales, where=norms > 0.)
        eq = A * scales[None, :]
        U, s, Vt = np.linalg.svd(eq, full_matrices=True)
        if not all(np.all(np.isfinite(v)) for v in (U, s, Vt)):
            raise FloatingPointError("nonfinite SVD result")
        top = float(s[0]) if len(s) else 0.
        rank = int(np.count_nonzero(s > rtol * top))
        weak_rank = int(np.count_nonzero(s > WEAK_RTOL * top))
        if y is None:
            gamma = Vt[-1].copy()
        elif rank:
            uy = np.einsum("ij,i->j", U[:, :rank], y, optimize=False)
            gamma = np.einsum("ji,j->i", Vt[:rank], uy / s[:rank], optimize=False)
        else:
            gamma = np.zeros(n)
        solution = scales * gamma
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("nonfinite solution")
    condition = float(s[0]) / float(s[-1]) if len(s) and s[-1] > 0. else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return dict(solution=solution, equilibrated_solution=gamma,
                column_norms=norms, column_scales=scales, singular_values=s,
                Vt=Vt, rank=rank, rank_rtol=rtol, weak_rank=weak_rank,
                weak_rtol=WEAK_RTOL,
                condition=condition,
                full_matrices=True, right_vectors_shape=list(Vt.shape))


def _residuals(A, x, y):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        r = _mv(A, x)
        if y is not None:
            r = r - y
        normal = _mv(A.T, r)
    denominator = _norm(A) * _norm(x) + (0. if y is None else _norm(y))
    if not math.isfinite(denominator):
        raise FloatingPointError("nonfinite relative residual denominator")
    return dict(vector=r, norm=_norm(r), normal_equation_vector=normal,
                normal_equation_norm=_norm(normal),
                relative_norm=_norm(r) / denominator if denominator > 0. else None)


def _task_null(svd, t):
    tn = _norm(t)
    result = {"translation_norm": tn,
              "coordinates": "unit vectors in column-equilibrated coordinates; physical delta=column_scales*delta",
              "parameter_units": "t/depth: input-depth unit; velocity: input-depth unit/second"}
    for name, rank in (("numerical", svd["rank"]), ("weak", svd["weak_rank"])):
        N = svd["Vt"][rank:].T.copy()
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            physical = svd["column_scales"][:, None] * N
        item = {"nullity": int(N.shape[1]), "basis_scaled": N, "basis_physical": physical,
                "translation_direction_map": None, "translation_direction_frobenius": None}
        if tn > 0. and math.isfinite(tn):
            unit = t / tn
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                P = (np.eye(3) - np.outer(unit, unit)) / tn
            task = _mm(P, physical[:3])
            item.update(translation_direction_map=task, translation_direction_frobenius=_norm(task))
        result[name] = item
    return result


def _affine_positivity(data, parameters, indices=None):
    n = len(data["r1"])
    ids = _indices(indices, n)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        z1 = _array(data["d1"], (n,), "d1") + parameters["b1"]
        z2 = parameters["a2"] * _array(data["d2"], (n,), "d2") + parameters["b2"]
    finite = np.isfinite(z1) & np.isfinite(z2)
    positive = finite & (z1 > 0.) & (z2 > 0.)
    return dict(rule="positive_scale_and_all_pool_corrected_depths", pool_size=n,
                scale_positive=bool(parameters["a2"] > 0.), depth1=z1, depth2=z2,
                finite_count=int(np.count_nonzero(finite)),
                positive_count=int(np.count_nonzero(positive)),
                fitting_count=len(ids),
                fitting_finite_count=int(np.count_nonzero(finite[ids])),
                fitting_positive_count=int(np.count_nonzero(positive[ids])),
                valid=bool(parameters["a2"] > 0. and np.all(positive)))


def _triangulate_signs(g, x, rank_rtol):
    n = g["pool_size"]
    hypotheses = []
    for sign in (1, -1):
        t, v1, v2 = (sign * x[:3], sign * x[3:6], sign * x[6:9])
        points = []
        for i in range(n):
            M = np.column_stack([g["Rrbar1"][i], -g["rbar2"][i]])
            k = t - _mv(g["RB1"][i], v1) + _mv(g["B2"][i], v2)
            try:
                s = _linear_svd(M, -k, rank_rtol)
                z = s["solution"]
                good = bool(s["rank"] == 2 and np.all(np.isfinite(z)) and np.all(z > 0.))
                points.append(dict(status="valid" if s["rank"] == 2 else "rank_deficient",
                                   positive=good, depth1=float(z[0]), depth2=float(z[1]),
                                   rank=s["rank"], singular_values=s["singular_values"],
                                   residual_norm=_norm(_mv(M, z) + k)))
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                points.append(dict(status="numerical_error", positive=False, depth1=None,
                                   depth2=None, reason=type(exc).__name__))
        count = sum(int(p["positive"]) for p in points)
        all_valid = all(p["status"] == "valid" for p in points)
        hypotheses.append(dict(sign=sign, positive_count=count, points=points,
                               all_triangulations_valid=all_valid))
    required = n // 2 + 1
    all_valid = all(h["all_triangulations_valid"] for h in hypotheses)
    admitted = [h["sign"] for h in hypotheses if all_valid and h["positive_count"] >= required]
    return dict(rule="unique_strict_majority_all_pool_row_cheirality", pool_size=n,
                required_positive=required, hypotheses=hypotheses,
                all_triangulations_valid=all_valid,
                selected_sign=admitted[0] if len(admitted) == 1 else None,
                valid=len(admitted) == 1)


def _initial_result(built, arm):
    return dict(arm=arm, status="unavailable", reason=None, parameters=None,
                A=built["A"], y=built["y"], indices=built["indices"],
                pool_size=built["geometry"]["pool_size"],
                units={"velocity": "input-depth unit/second", "time": "second",
                       "physical_velocity_scaling_applied": False},
                cheirality=None, task_null=None, residuals=None)


def solve_affine(data, indices=None, *, angular_only=False, rank_rtol=None):
    """Full12 or angular-only6 affine initializer; all-pool physical checks."""
    built = build(data, indices, angular_only)
    result = _initial_result(built, "angular_only_affine" if angular_only else "affine_rs")
    try:
        svd = _linear_svd(built["A"], built["y"], rank_rtol)
        result.update(svd)
        x = svd["solution"]
        offset = 3 if angular_only else 9
        p = dict(t=x[:3].copy(), v1=np.zeros(3) if angular_only else x[3:6].copy(),
                 v2=np.zeros(3) if angular_only else x[6:9].copy(),
                 b1=float(x[offset]), a2=float(x[offset + 1]), b2=float(x[offset + 2]))
        result.update(parameters=p, cheirality=_affine_positivity(data, p, built["indices"]),
                      task_null=_task_null(svd, p["t"]),
                      residuals=_residuals(built["A"], x, built["y"]))
        if svd["rank"] != built["A"].shape[1]:
            result["reason"] = "rank_deficient"
        elif _norm(p["t"]) == 0.:
            result["reason"] = "undefined_translation"
        elif not result["cheirality"]["valid"]:
            result["reason"] = "nonpositive_affine_scale_or_pool_depth"
        else:
            result.update(status="valid", reason=None)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        result["reason"] = "numerical_error:" + type(exc).__name__
    return result


def solve_depth_free(data, indices=None, *, rank_rtol=None):
    """Homogeneous9 initializer; all-pool row triangulation chooses directed sign."""
    built = build_depth_free(data, indices)
    result = _initial_result(built, "depth_free_rs")
    try:
        svd = _linear_svd(built["A"], None, rank_rtol)
        result.update(svd)
        raw = svd["solution"].copy()
        tn = _norm(raw[:3])
        if tn == 0. or not math.isfinite(tn):
            result.update(reason="undefined_translation", task_null=_task_null(svd, raw[:3]))
            return result
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            x = raw / tn  # One common positive length-gauge factor for t and both velocities.
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("nonfinite homogeneous normalization")
        cheirality = _triangulate_signs(built["geometry"], x, rank_rtol)
        if cheirality["selected_sign"] is not None:
            x *= cheirality["selected_sign"]
        p = dict(t=x[:3].copy(), v1=x[3:6].copy(), v2=x[6:9].copy(),
                 b1=None, a2=None, b2=None)
        result.update(parameters=p, solution=x, unnormalized_solution=raw,
                      normalization="one common factor gives ||t||=1; sign from all-pool majority",
                      cheirality=cheirality, task_null=_task_null(svd, raw[:3]),
                      residuals=_residuals(built["A"], x, None))
        if svd["rank"] != 8:
            result["reason"] = "rank_not_eight"
        elif not cheirality["valid"]:
            result["reason"] = "unresolved_cheirality_sign"
        else:
            result.update(status="valid", reason=None)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        result["reason"] = "numerical_error:" + type(exc).__name__
    return result
