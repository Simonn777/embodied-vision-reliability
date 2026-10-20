"""Generic nonsingular-M axial-depth affine-tangent branch only.

No robust estimator, root ranking, fallback, refinement, input acquisition, or
ground truth. Machine-arithmetic checks below are diagnostics/qualification
rules, not formal error bounds or guarantees for noisy measurements.
"""
import math
import numpy as np


EPS = np.finfo(np.float64).eps
POLICY = {
    "matrix_rank": "sigma > max(rows,columns)*eps*sigma_max",
    "coefficient_zero": "128*eps*max(1,cond2(M))*3*max_abs(y_coefficients)^2",
    "root_imaginary": "128*eps*max(1,abs(real),abs(imaginary))",
    "gram_relative": 4096*EPS,
    "tangent_relative": 4096*EPS,
    "rotation_absolute": 512*EPS,
    "eps": EPS,
    "qualification_only_not_formal_error_bound": True,
}


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + " has invalid shape or nonfinite values")
    return result.copy()


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(name + " must be a finite real scalar")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != () or not np.isfinite(result):
        raise ValueError(name + " must be a finite real scalar")
    return float(result)


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite numerical intermediate")
    return value


def _mm(a, b):
    return _finite(np.einsum("ij,jk->ik", a, b, optimize=False))


def _mv(a, b):
    return _finite(np.einsum("ij,j->i", a, b, optimize=False))


def _gram(a, b=None):
    return _mm(a.T, a if b is None else b)


def _symvec(a):
    return np.array((a[0, 0], a[0, 1], a[1, 1]))


def _rank(a):
    singular = _finite(np.linalg.svd(a, compute_uv=False))
    threshold = max(a.shape)*EPS*float(singular[0])
    rank = int(np.sum(singular > threshold))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0. else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return {"rank": rank, "singular_values": singular,
            "rank_threshold": threshold, "condition2": condition}


def _norm(v):
    scale = float(np.max(np.abs(v)))
    if scale == 0.:
        return 0.
    out = scale*math.sqrt(float(np.sum((v/scale)**2)))
    if not math.isfinite(out):
        raise FloatingPointError("nonfinite vector norm")
    return out


def _relative_max(residual, *reference):
    numerator = float(np.max(np.abs(residual)))
    denominator = max(float(np.max(np.abs(v))) for v in reference)
    if denominator == 0.:
        return {"absolute": numerator, "scale": 0.,
                "relative": 0. if numerator == 0. else None}
    ratio = numerator/denominator
    return {"absolute": numerator, "scale": denominator,
            "relative": ratio if math.isfinite(ratio) else None}


def _frame(tangent):
    first = tangent[:, 0]/_norm(tangent[:, 0])
    second = tangent[:, 1]-first*float(np.dot(first, tangent[:, 1]))
    second /= _norm(second)
    return _finite(np.column_stack((first, second, np.cross(first, second))))


def _record(value):
    """Detach arrays and refuse nonfinite JSON, preserving explicit failure text."""
    if isinstance(value, np.ndarray):
        return _record(value.tolist())
    if isinstance(value, np.generic):
        return _record(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("nonfinite output observation")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_record(x) for x in value]
    if isinstance(value, dict):
        return {k: _record(v) for k, v in value.items()}
    raise TypeError("unsupported record value")


def solve_tangent(r1, r2, E1, E2, d1, d2, g1, g2, A):
    """Enumerate this branch's roots without truth-based selection.

    r=(x,y,1); E is its 3x2 derivative in the coordinate units of g and A.
    g1/g2 are length-two depth row-gradients in each image's own coordinates;
    A maps source coordinate increments to target increments. Raw depths may
    have either sign; recovered corrected depths and relative scale must be >0.
    Invalid input contracts raise ValueError. Degenerate/numerical branches
    return explicit status and no fallback. Root records include every root,
    even complex/nonphysical roots. Exact duplicate roots are not merged.
    """
    r1, r2 = _array(r1, (3,), "r1"), _array(r2, (3,), "r2")
    E1, E2 = _array(E1, (3, 2), "E1"), _array(E2, (3, 2), "E2")
    g1, g2 = _array(g1, (2,), "g1"), _array(g2, (2,), "g2")
    A = _array(A, (2, 2), "A")
    d1, d2 = _scalar(d1, "d1"), _scalar(d2, "d2")
    if r1[2] != 1. or r2[2] != 1. or np.any(E1[2] != 0.) or np.any(E2[2] != 0.):
        raise ValueError("axial model requires ray z exactly one and Jacobian z row zero")
    result = {"schema": "evr0027.tangent-generic.v1", "status": "started",
        "branch": "nonsingular_M_only", "fallback_used": False, "policy": dict(POLICY),
        "parameters": ["b", "a", "c"], "symmetric_vector_order": ["11", "12", "22"],
        "inputs": dict(r1=r1, r2=r2, E1=E1, E2=E2, d1=d1, d2=d2, g1=g1, g2=g2, A=A),
        "root_records": [], "candidates": [], "admissible_candidate_count": 0,
        "pose_usable_count": 0, "root_selection": None,
        "finite_record": True, "global_or_universal_minimal_solver_claimed": False}
    phase = "frames"
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            diagnostics = {k: _rank(v) for k, v in (("E1", E1), ("E2", E2), ("A", A))}
            result["input_rank_diagnostics"] = diagnostics
            if any(diagnostics[k]["rank"] < 2 for k in diagnostics):
                result.update(status="unavailable", reason="rank_deficient_input_differential")
                return _record(result)
            U0, U1 = _finite(d1*E1+np.outer(r1, g1)), E1.copy()
            V0 = _mm(_finite(d2*E2+np.outer(r2, g2)), A)
            V1 = _mm(E2, A)
            M = np.column_stack((_symvec(_gram(V0)),
                                 _symvec(_gram(V0, V1)+_gram(V1, V0)),
                                 _symvec(_gram(V1))))
            Q = np.column_stack((_symvec(_gram(U0)),
                                 _symvec(_gram(U0, U1)+_gram(U1, U0)),
                                 _symvec(_gram(U1))))
            _finite(M); _finite(Q)
            result["matrices"] = dict(U0=U0, U1=U1, V0=V0, V1=V1, M=M, Q=Q)
            phase = "M_rank"
            md = _rank(M)
            result["M_diagnostics"] = md
            if md["rank"] != 3:
                result.update(status="unavailable", reason="singular_M_generic_branch")
                return _record(result)
            if md["condition2"] is None:
                raise FloatingPointError("unrepresentable M condition")
            phase = "coefficient_solve"
            coefficients = _finite(np.linalg.solve(M, Q))
            result["y_coefficients_ascending"] = coefficients
            result["coefficient_solve_residual"] = _relative_max(_mm(M, coefficients)-Q, Q)
            positive = np.convolve(coefficients[0], coefficients[2])
            negative = np.convolve(coefficients[1], coefficients[1])
            polynomial = _finite(positive-negative)
            coefficient_scale = float(np.max(np.abs(coefficients)))
            # Conservative numerical-resolution heuristic, not a rigorous
            # rounding enclosure. Includes solve conditioning because an exact
            # zero y2 coefficient need not be stored as floating-point zero.
            zero_threshold = float(128*EPS*max(1., md["condition2"])*3*coefficient_scale**2)
            _finite(zero_threshold)
            resolved = np.flatnonzero(np.abs(polynomial) > zero_threshold)
            result["polynomial"] = {"coefficients_ascending": polynomial,
                "construction_positive": positive, "construction_negative": negative,
                "zero_threshold": zero_threshold,
                "degree": int(resolved[-1]) if len(resolved) else None}
            if not len(resolved):
                result.update(status="unavailable", reason="vanished_or_numerically_unresolved_polynomial")
                return _record(result)
            degree = int(resolved[-1])
            if degree == 0:
                result.update(status="unavailable", reason="resolved_nonzero_constant_polynomial")
                return _record(result)
            phase = "polynomial_roots"
            reduced = polynomial[:degree+1]
            roots = np.roots(reduced[::-1]/np.max(np.abs(reduced)))
            result["polynomial"]["root_count"] = len(roots)
            # np.roots order is retained. No quality/GT-based ordering or dedup.
            for index, root in enumerate(roots):
                row = {"root_index": index, "status": "rejected", "real": None, "imaginary": None}
                result["root_records"].append(row)
                try:
                    real, imaginary = float(root.real), float(root.imag)
                    if not math.isfinite(real) or not math.isfinite(imaginary):
                        row.update(reason="nonfinite_root", nonfinite_repr=str(root))
                        continue
                    row.update(real=real, imaginary=imaginary)
                    imaginary_threshold = 128*EPS*max(1., abs(real), abs(imaginary))
                    row["imaginary_threshold"] = imaginary_threshold
                    if abs(imaginary) > imaginary_threshold:
                        row["reason"] = "complex_root"
                        continue
                    b = real
                    powers = _finite(np.array((1., b, b*b)))
                    y = _mv(coefficients, powers)
                    row["y"] = y
                    if y[0] <= 0.:
                        row["reason"] = "nonpositive_a_squared"
                        continue
                    a = math.sqrt(float(y[0])); c = float(y[1]/a)
                    p, s = d1+b, a*d2+c
                    _finite([a, b, c, p, s])
                    row.update(parameters=[b, a, c], corrected_depth1=p, corrected_depth2=s,
                               rank_one_residual=float(y[0]*y[2]-y[1]*y[1]))
                    if p <= 0. or s <= 0.:
                        row["reason"] = "nonpositive_corrected_depth"
                        continue
                    U, V = _finite(U0+b*U1), _finite(a*V0+c*V1)
                    ud, vd = _rank(U), _rank(V)
                    row.update(U_rank=ud, V_rank=vd)
                    if ud["rank"] != 2 or vd["rank"] != 2:
                        row["reason"] = "rank_deficient_tangent_frame"
                        continue
                    gu, gv = _gram(U), _gram(V)
                    residual = _relative_max(gu-gv, gu, gv)
                    row["gram_residual"] = residual
                    if residual["relative"] is None or residual["relative"] > POLICY["gram_relative"]:
                        row["reason"] = "gram_residual_exceeds_arithmetic_tolerance"
                        continue
                    R = _mm(_frame(V), _frame(U).T)
                    t = _finite(s*r2-_mv(R, p*r1))
                    tangent_error = _relative_max(_mm(R, U)-V, U, V)
                    orthogonality = float(np.max(np.abs(_gram(R)-np.eye(3))))
                    determinant = float(np.linalg.det(R))
                    row.update(tangent_residual=tangent_error, rotation_orthogonality=orthogonality,
                               rotation_determinant=determinant)
                    if (tangent_error["relative"] is None or tangent_error["relative"] > POLICY["tangent_relative"]
                            or orthogonality > POLICY["rotation_absolute"]
                            or abs(determinant-1.) > POLICY["rotation_absolute"]):
                        row["reason"] = "rigid_recovery_residual_exceeds_arithmetic_tolerance"
                        continue
                    translation_norm = _norm(t)
                    usable = bool(np.any(t != 0.))  # no ad hoc near-zero cutoff
                    candidate = {"root_index": index, "b": b, "a": a, "c": c,
                        "R": R, "t": t, "translation_norm": translation_norm,
                        "translation_direction_defined": usable, "pose_usable": usable,
                        "corrected_depth1": p, "corrected_depth2": s,
                        "gram_residual": residual, "tangent_residual": tangent_error,
                        "rotation_orthogonality": orthogonality, "rotation_determinant": determinant}
                    result["candidates"].append(candidate)
                    row.update(status="admissible", reason=None, candidate_index=len(result["candidates"])-1)
                except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                    row.update(reason="numerical_root_validation_failure",
                               failure={"type": type(error).__name__, "message": str(error)})
            result["admissible_candidate_count"] = len(result["candidates"])
            result["pose_usable_count"] = sum(x["pose_usable"] for x in result["candidates"])
            result.update(status="completed" if result["candidates"] else "unavailable",
                          reason=None if result["candidates"] else "no_admissible_root")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        result.update(status="numerical_failure", reason="numerical_failure", failed_phase=phase,
                      failure={"type": type(error).__name__, "message": str(error)})
    return _record(result)
