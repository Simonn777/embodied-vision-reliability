"""Fixed vertex-weight arithmetic, not an estimator or an independence model.

X_i=(d1_i+b) r1_i; Y_i=(a*d2_i+c) r2_i. Rays are supplied unchanged.
Distance cost is sum_{i<j} w_i*w_j*(||X_i-X_j||^2-||Y_i-Y_j||^2)^2.
Point cost is min_{R in SO(3),t} sum_i w_i*||Y_i-R X_i-t||^2.
Construction is O(N); each objective uses fixed-size moments. No noise model,
covariance, global-optimality certificate or physical depth filter is implied.
Raw signed costs are retained, including negative floating cancellation.
"""

from dataclasses import dataclass
import numpy as np


PARAMETER_ORDER = ("b", "a", "c")
MONOMIAL_ORDER = ("1", "b", "b^2", "a^2", "a*c", "c^2")
FEATURE_ORDER = ("d1*r1", "r1", "d2*r2", "r2")


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError("nonfinite " + name)
    return value


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    result = np.asarray(value, dtype=np.float64)
    return _finite(result, name)


def _readonly(value):
    value = np.array(value, dtype=np.float64, copy=True)
    value.flags.writeable = False
    return value


def _basis():
    result = np.zeros((6, 12, 12), dtype=np.float64)
    eye = np.eye(3)
    result[0, :3, :3] = eye
    result[1, :3, 3:6] = eye
    result[1, 3:6, :3] = eye
    result[2, 3:6, 3:6] = eye
    result[3, 6:9, 6:9] = -eye
    result[4, 6:9, 9:12] = -eye
    result[4, 9:12, 6:9] = -eye
    result[5, 9:12, 9:12] = -eye
    return result


_A = _basis()
_A.flags.writeable = False


@dataclass(frozen=True)
class Moments:
    """C and mean use unnormalized vertex weights; H uses product edge weights."""

    H: np.ndarray
    C: np.ndarray
    mean: np.ndarray
    weight_sum: float
    input_count: int
    support_count: int


def build_moments(r1, r2, d1, d2, weights=None):
    r1, r2 = _array(r1, "r1"), _array(r2, "r2")
    d1, d2 = _array(d1, "d1"), _array(d2, "d2")
    if r1.ndim != 2 or r1.shape[1:] != (3,) or r2.shape != r1.shape:
        raise ValueError("rays must have equal Nx3 shapes")
    n = r1.shape[0]
    if d1.shape != (n,) or d2.shape != (n,):
        raise ValueError("depths must have shape N")
    w = np.ones(n, dtype=np.float64) if weights is None else _array(weights, "weights")
    if w.shape != (n,) or np.any(w < 0.0):
        raise ValueError("weights must have shape N and be nonnegative")
    support = w > 0.0
    if not np.any(support):
        raise ValueError("positive total weight required")
    # Ignore finite zero-weight vertices before products; their values cannot
    # affect either objective. Nonfinite input is rejected even at weight zero.
    x, y = r1[support], r2[support]
    u, v, w = d1[support], d2[support], w[support]
    if np.any(np.all(x == 0.0, axis=1)) or np.any(np.all(y == 0.0, axis=1)):
        raise ValueError("supported rays must be nonzero")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        total = float(np.sum(w))
        _finite(total, "weight sum")
        f = np.concatenate((u[:, None] * x, x, v[:, None] * y, y), axis=1)
        _finite(f, "features")
        mean = np.einsum("n,ni->i", w / total, f, optimize=False)
        fc = f - mean
        C = np.einsum("n,ni,nj->ij", w, fc, fc, optimize=False)
        q = np.einsum("ni,kij,nj->nk", fc, _A, fc, optimize=False)
        qsum = np.einsum("n,nk->k", w, q, optimize=False)
        Q = np.einsum("n,nk,nl->kl", w, q, q, optimize=False)
        AC = np.einsum("kij,jl->kil", _A, C, optimize=False)
        T = np.einsum("kij,lji->kl", AC, AC, optimize=False)
        H = total * Q + np.outer(qsum, qsum) + 2.0 * T
        # Only roundoff antisymmetry is removed, not eigenvalues or signed cost.
        H = 0.5 * (H + H.T)
        C = 0.5 * (C + C.T)
        for value, name in ((H, "H"), (C, "C"), (mean, "mean")):
            _finite(value, name)
    return Moments(_readonly(H), _readonly(C), _readonly(mean), total,
                   n, int(np.count_nonzero(support)))


def _parameters(b, a, c):
    value = _array((b, a, c), "parameters")
    if value.shape != (3,):
        raise ValueError("b, a, c must be scalar")
    return value


def distance_objective(moments, b, a, c):
    """Exact polynomial derivatives, parameter order (b,a,c); raw signed cost."""
    b, a, c = _parameters(b, a, c)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        z = np.array((1.0, b, b*b, a*a, a*c, c*c))
        J = np.array(((0., 0., 0.), (1., 0., 0.), (2*b, 0., 0.),
                      (0., 2*a, 0.), (0., c, a), (0., 0., 2*c)))
        hz = np.einsum("ij,j->i", moments.H, z, optimize=False)
        cost = float(np.einsum("i,i->", z, hz, optimize=False))
        gradient = 2.0 * np.einsum("ki,k->i", J, hz, optimize=False)
        hessian = 2.0 * np.einsum("ki,kl,lj->ij", J, moments.H, J, optimize=False)
        hessian[0, 0] += 4.0 * hz[2]
        hessian[1, 1] += 4.0 * hz[3]
        hessian[1, 2] += 2.0 * hz[4]
        hessian[2, 1] += 2.0 * hz[4]
        hessian[2, 2] += 4.0 * hz[5]
        for value, name in ((cost, "distance cost"), (gradient, "gradient"),
                            (hessian, "hessian")):
            _finite(value, name)
    return {"cost_raw": cost, "gradient": gradient, "hessian": hessian,
            "parameter_order": PARAMETER_ORDER}


def procrustes_objective(moments, b, a, c):
    """Profile point cost with a proper rotation mapping X to Y; no clipping."""
    b, a, c = _parameters(b, a, c)
    C = moments.C
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        Cxx = C[:3, :3] + b*(C[:3, 3:6] + C[3:6, :3]) + b*b*C[3:6, 3:6]
        Cyy = a*a*C[6:9, 6:9] + a*c*(C[6:9, 9:12] + C[9:12, 6:9]) + c*c*C[9:12, 9:12]
        Cxy = a*C[:3, 6:9] + c*C[:3, 9:12] + a*b*C[3:6, 6:9] + b*c*C[3:6, 9:12]
        for value, name in ((Cxx, "Cxx"), (Cyy, "Cyy"), (Cxy, "Cxy")):
            _finite(value, name)
        U, singular, Vt = np.linalg.svd(Cxy, full_matrices=True)
        # Cxy=sum w Xc Yc^T, hence R=V D U^T (not U D V^T).
        det_vut = float(np.linalg.det(np.einsum("ij,jk->ik", Vt.T, U.T, optimize=False)))
        correction = -1.0 if det_vut < 0.0 else 1.0
        signs = np.array((1.0, 1.0, correction))
        R = np.einsum("ij,j,kj->ik", Vt.T, signs, U, optimize=False)
        mux = moments.mean[:3] + b*moments.mean[3:6]
        muy = a*moments.mean[6:9] + c*moments.mean[9:12]
        t = muy - np.einsum("ij,j->i", R, mux, optimize=False)
        proper_trace = float(np.einsum("i,i->", signs, singular, optimize=False))
        trace_sum = float(np.trace(Cxx) + np.trace(Cyy))
        cost = trace_sum - 2.0*proper_trace
        unrestricted = trace_sum - 2.0*float(np.sum(singular))
        for value, name in ((cost, "point cost"), (unrestricted, "orthogonal cost"),
                            (R, "rotation"), (t, "translation"), (singular, "singular values")):
            _finite(value, name)
    return {"cost_raw": cost, "R": R, "t": t, "singular_values": singular,
            "proper_trace": proper_trace, "trace_sum": trace_sum,
            "determinant_correction": correction,
            "orthogonal_cost_raw": unrestricted}
