"""Known normal/point geometry controls; no novel pose construction is claimed.

Pure arithmetic: no files, models, native estimators, labels, or support selection.
R,t map camera 1 points into camera 2. Rays are axial (third coordinate one).
Scale-only geometry obeys a*Y = R*X+t. Normal pairs are oriented camera normals.
The caller supplies the SAME fixed fitting support to the intended arms and the
full match support separately to the pixel selector. No row is silently dropped.

Relative rank guards are numerical conventions, not calibrated uncertainty.
SciPy max_nfev=100 does not include every finite-difference residual evaluation;
both SciPy nfev and actual residual-call counts are returned.
"""
import time
import numpy as np


RANK_RTOL = 1e-10
NORMAL_SCALE = float(np.sin(np.deg2rad(5.0)))
PIXEL_SCALE = 2.0
MAX_NFEV = 100


class NumericalFailure(Exception):
    """Expected numeric/degenerate input or arithmetic, never a programming error."""

    def __init__(self, message, **diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _array(value, shape=None, name="array"):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    out = np.asarray(value, dtype=np.float64)
    if shape is not None and out.shape != shape:
        raise ValueError(name + " has wrong shape")
    if not np.isfinite(out).all():
        raise NumericalFailure(name + " is nonfinite")
    return out


def _unit(value):
    v = _array(value, (3,), "direction")
    scale = float(np.max(np.abs(v)))
    if scale == 0:
        raise NumericalFailure("zero direction")
    scaled = v / scale
    return scaled / np.sqrt(np.dot(scaled, scaled))


def _normals(value, count):
    normals = _array(value, (count, 3), "normals")
    return np.array([_unit(n) for n in normals])


def _rays(r1, r2, minimum):
    x = _array(r1, name="rays1")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("rays1 must be Nx3")
    y = _array(r2, x.shape, "rays2")
    if len(x) < minimum:
        raise NumericalFailure("insufficient fixed support")
    if not (np.allclose(x[:, 2], 1., atol=1e-12, rtol=0.)
            and np.allclose(y[:, 2], 1., atol=1e-12, rtol=0.)):
        raise ValueError("rays must be axial with third coordinate one")
    return x, y


def _matmul(a, b):
    return np.einsum("...ij,...jk->...ik", a, b, optimize=False)


def _rotate(R, points):
    return np.einsum("ij,nj->ni", R, points, optimize=False)


def _proper(R):
    R = _array(R, (3, 3), "rotation")
    if (not np.allclose(_matmul(R.T, R), np.eye(3), atol=1e-8, rtol=0.)
            or abs(float(np.linalg.det(R)) - 1.) > 1e-8):
        raise NumericalFailure("rotation is not proper")
    return R


def _svd(A, required=0):
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    _array(s, name="singular values")
    rank = int(np.count_nonzero(s > RANK_RTOL * s[0])) if len(s) and s[0] > 0 else 0
    if rank < required:
        raise NumericalFailure("rank_deficient", rank=rank, required_rank=required,
                               singular_values=s.tolist(), rank_rtol=RANK_RTOL)
    return U, s, Vt, rank


def _failed(reason, start, support_count=None, **details):
    return dict(status="unavailable", reason=str(reason),
                support_count=support_count, seconds=time.perf_counter()-start,
                numerical_diagnostics=getattr(reason, "diagnostics", {}),
                **details)


def _support(n):
    return dict(support_count=n, support_indices=list(range(n)))


def _triangulate(R, t, r1, r2):
    """Linear axial-depth diagnostics on every row, without deleting bad rows."""
    rotated = _rotate(R, r1)
    rows = []
    for x, y in zip(rotated, r2):
        A = np.column_stack((x, -y))
        U, s, Vt, rank = _svd(A)
        if rank < 2:
            rows.append(dict(status="rank_deficient", rank=rank,
                             singular_values=s.tolist(), depths=None,
                             positive=False, residual=None))
            continue
        depths = Vt.T @ ((U[:, :2].T @ (-t)) / s)
        residual = A @ depths + t
        _array(depths, (2,), "triangulated depths")
        _array(residual, (3,), "triangulation residual")
        rows.append(dict(status="completed", rank=rank,
                         singular_values=s.tolist(), depths=depths.tolist(),
                         positive=bool(np.all(depths > 0.)),
                         residual=residual.tolist()))
    return dict(points=rows, support_count=len(rows),
                finite_rank2_count=sum(x["status"] == "completed" for x in rows),
                positive_count=sum(x["positive"] for x in rows))


def translation_signs(R, t, r1, r2):
    """Both unit directions; max positive count, canonical sign breaks ties."""
    R = _proper(R)
    r1, r2 = _rays(r1, r2, 2)
    direction = _unit(t)
    # Canonical direction: its largest-magnitude component is positive.
    if direction[int(np.argmax(np.abs(direction)))] < 0.:
        direction = -direction
    hypotheses = []
    for sign in (1, -1):
        candidate = sign * direction
        hypotheses.append(dict(sign=sign, t=candidate.tolist(),
                               **_triangulate(R, candidate, r1, r2)))
    selected = 0 if hypotheses[0]["positive_count"] >= hypotheses[1]["positive_count"] else 1
    return dict(hypotheses=hypotheses, selected_index=selected,
                tie=bool(hypotheses[0]["positive_count"] == hypotheses[1]["positive_count"]),
                rule="max_positive_count_then_canonical_largest_component",
                t=hypotheses[selected]["t"])


def solve_scale_only(r1, r2, z1, z2):
    """Proper centered similarity, with target-depth scale a and fixed support."""
    start = time.perf_counter()
    n = None
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            r1, r2 = _rays(r1, r2, 3)
            n = len(r1)
            z1, z2 = _array(z1, (n,), "depth1"), _array(z2, (n,), "depth2")
            if np.any(z1 <= 0.) or np.any(z2 <= 0.):
                raise NumericalFailure("nonpositive scale-only depth")
            X, Y = z1[:, None]*r1, z2[:, None]*r2
            mx, my = X.mean(axis=0), Y.mean(axis=0)
            xc, yc = X-mx, Y-my
            _, sx, _, rank_x = _svd(xc, 2)
            _, sy, _, rank_y = _svd(yc, 2)
            C = np.einsum("ni,nj->ij", yc, xc, optimize=False)
            U, singular, Vt, rank_c = _svd(C, 2)
            sign = 1. if np.linalg.det(U @ Vt) >= 0. else -1.
            R = _proper(U @ np.diag([1., 1., sign]) @ Vt)
            denominator = float(np.einsum("ni,ni->", yc, yc, optimize=False))
            a = float((singular[0]+singular[1]+sign*singular[2])/denominator)
            if not np.isfinite(a) or a <= 0.:
                raise NumericalFailure("nonpositive or nonfinite relative scale")
            t = a*my - R @ mx
            _unit(t)  # Reject exactly zero direction, without a tiny-t threshold.
            residual = a*Y - _rotate(R, X) - t
            _array(residual, name="similarity residual")
            cost = float(np.einsum("ni,ni->", residual, residual, optimize=False))
            return dict(status="completed", R=R.tolist(), t=t.tolist(), a=a,
                        residual=residual.tolist(), sum_squared_residual=cost,
                        ranks=dict(source=rank_x, target=rank_y, cross=rank_c),
                        singular_values=dict(source=sx.tolist(), target=sy.tolist(),
                                             cross=singular.tolist()),
                        reflected_unconstrained=bool(sign < 0.),
                        cheirality=_triangulate(R, t, r1, r2),
                        **_support(n), seconds=time.perf_counter()-start)
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
        return _failed(exc, start, n)


def solve_normal_pose(r1, r2, n1, n2):
    """Established normal alignment plus epipolar translation, no depth fit."""
    start = time.perf_counter()
    n = None
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            r1, r2 = _rays(r1, r2, 2)
            n = len(r1)
            n1, n2 = _normals(n1, n), _normals(n2, n)
            _, s1, _, rank1 = _svd(n1, 2)
            _, s2, _, rank2 = _svd(n2, 2)
            C = np.einsum("ni,nj->ij", n2, n1, optimize=False)
            U, sn, Vt, rankn = _svd(C, 2)
            sign = 1. if np.linalg.det(U @ Vt) >= 0. else -1.
            R = _proper(U @ np.diag([1., 1., sign]) @ Vt)
            epipolar = np.cross(_rotate(R, r1), r2)
            _, se, Ve, ranke = _svd(epipolar, 2)
            signs = translation_signs(R, Ve[-1], r1, r2)
            t = np.asarray(signs["t"])
            residual = _rotate(R, n1)-n2
            return dict(status="completed", R=R.tolist(), t=t.tolist(),
                        normal_residual=residual.tolist(),
                        epipolar_residual=(epipolar @ t).tolist(),
                        singular_values=dict(normal1=s1.tolist(), normal2=s2.tolist(),
                                             cross=sn.tolist(), epipolar=se.tolist()),
                        ranks=dict(normal1=rank1, normal2=rank2,
                                   cross=rankn, epipolar=ranke),
                        cheirality=signs, **_support(n),
                        seconds=time.perf_counter()-start)
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
        return _failed(exc, start, n)


def _camera(K):
    K = _array(K, (3, 3), "K")
    if not np.array_equal(K[2], [0., 0., 1.]):
        raise ValueError("K must have pinhole bottom row (0,0,1)")
    return K, np.linalg.inv(K)


def _pixels(rays, K):
    projected = _rotate(K, rays)
    if np.any(projected[:, 2] == 0.):
        raise NumericalFailure("invalid pixel projection")
    return projected[:, :2]/projected[:, 2:]


def _sampson(R, t, xy1, xy2, K1, K2):
    R, t = _proper(R), _unit(t)
    _, iK1 = _camera(K1)
    _, iK2 = _camera(K2)
    x = _array(xy1, name="pixels1")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("pixels1 must be Nx2")
    y = _array(xy2, x.shape, "pixels2")
    tx, ty, tz = t
    cross = np.array([[0., -tz, ty], [tz, 0., -tx], [-ty, tx, 0.]])
    F = _matmul(_matmul(iK2.T, cross), _matmul(R, iK1))
    fscale = float(np.max(np.abs(F)))
    if not np.isfinite(fscale) or fscale == 0.:
        raise NumericalFailure("invalid fundamental matrix")
    F = F/fscale
    xh, yh = np.column_stack((x, np.ones(len(x)))), np.column_stack((y, np.ones(len(y))))
    Fx, Fty = _rotate(F, xh), _rotate(F.T, yh)
    numerator = np.einsum("ni,ni->n", yh, Fx, optimize=False)
    denominator = (np.einsum("ni,ni->n", Fx[:, :2], Fx[:, :2], optimize=False)
                   + np.einsum("ni,ni->n", Fty[:, :2], Fty[:, :2], optimize=False))
    good = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.)
    residual = np.zeros(len(x))
    residual[good] = numerator[good]/np.sqrt(denominator[good])
    return residual, good


def pixel_msac(R, t, xy1, xy2, K1, K2):
    """Full supplied match set; undefined residuals pay the full 2-pixel cap."""
    start = time.perf_counter()
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            residual, good = _sampson(R, t, xy1, xy2, K1, K2)
            if not len(residual):
                raise NumericalFailure("empty score support")
            # Clip before squaring, retaining a bounded score even for huge errors.
            costs = np.minimum(np.abs(residual)/PIXEL_SCALE, 1.)**2
            costs[~good] = 1.
            _, iK1 = _camera(K1)
            _, iK2 = _camera(K2)
            x = np.column_stack((xy1, np.ones(len(residual))))
            y = np.column_stack((xy2, np.ones(len(residual))))
            r1, r2 = _rotate(iK1, x), _rotate(iK2, y)
            r1, r2 = r1/r1[:, 2:], r2/r2[:, 2:]
            cheirality = _triangulate(_proper(R), _unit(t), r1, r2)
            return dict(status="completed", score=float(costs.sum()),
                        mean_score=float(costs.mean()), threshold_pixels=PIXEL_SCALE,
                        costs=costs.tolist(), residual_pixels=[
                            float(r) if ok else None for r, ok in zip(residual, good)],
                        invalid_count=int(np.count_nonzero(~good)),
                        cheirality=cheirality,
                        **_support(len(residual)), seconds=time.perf_counter()-start)
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
        return _failed(exc, start)


def select_by_pixel_msac(candidates, xy1, xy2, K1, K2):
    """Score all completed candidate poses; exact ties retain generation order."""
    start = time.perf_counter()
    attempts, selected, best = [], None, None
    for index, candidate in enumerate(candidates):
        if candidate["status"] != "completed":
            attempts.append(dict(index=index, status="unavailable", reason="candidate_unavailable"))
            continue
        score = pixel_msac(candidate["R"], candidate["t"], xy1, xy2, K1, K2)
        attempts.append(dict(index=index, **score))
        if score["status"] == "completed" and (best is None or score["score"] < best):
            selected, best = index, score["score"]
    return dict(status="completed" if selected is not None else "unavailable",
                selected_index=selected, attempts=attempts,
                tie_rule="first_generation_index", seconds=time.perf_counter()-start)


def _basis(t):
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(t)))] = 1.
    first = _unit(np.cross(t, axis))
    return np.column_stack((first, np.cross(t, first)))


def _chart(x, R0, t0, B):
    from scipy.spatial.transform import Rotation
    _array(x, (5,), "optimizer state")
    R = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
    v = x[3:]
    q = float(np.dot(v, v))/4.
    # Stereographic chart: both hemispheres, excludes only the exact antipode.
    t = ((1.-q)*t0+B @ v)/(1.+q)
    return R, _unit(t)


def refine_joint(r1, r2, n1, n2, K1, K2, starts):
    """Joint chord-normal + signed pixel-Sampson LS for each ordered start.

    Starts are dictionaries {R,t,label}. The two final translation signs and
    cheirality diagnostics are retained. No truth or score-based start pruning.
    An evaluation-budget iterate is usable when finite; converged stays false.
    """
    from scipy.optimize import least_squares
    begin = time.perf_counter()
    count = None
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            r1, r2 = _rays(r1, r2, 2)
            count = len(r1)
            n1, n2 = _normals(n1, count), _normals(n2, count)
            K1, _ = _camera(K1)
            K2, _ = _camera(K2)
            xy1, xy2 = _pixels(r1, K1), _pixels(r2, K2)
    except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
        return _failed(exc, begin, count, attempts=[
            dict(status="unavailable", index=i, label=s["label"], reason=str(exc),
                 residual_calls=0, seconds=0., support_count=count)
            for i, s in enumerate(starts)])
    attempts = []
    for index, initial in enumerate(starts):
        start = time.perf_counter()
        evaluations = 0
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                R0, t0 = _proper(initial["R"]), _unit(initial["t"])
                B = _basis(t0)

                def residual(x):
                    nonlocal evaluations
                    evaluations += 1
                    R, t = _chart(x, R0, t0, B)
                    sampson, valid = _sampson(R, t, xy1, xy2, K1, K2)
                    if not valid.all():
                        raise NumericalFailure("undefined fixed-support refinement residual")
                    result = np.concatenate(((_rotate(R, n1)-n2).ravel()/NORMAL_SCALE,
                                             sampson/PIXEL_SCALE))
                    return _array(result, name="joint residual")

                initial_residual = residual(np.zeros(5))
                result = least_squares(residual, np.zeros(5), method="trf", jac="2-point",
                                       max_nfev=MAX_NFEV, ftol=1e-8, xtol=1e-8,
                                       gtol=1e-8, x_scale=1., loss="linear")
                _array(result.x, (5,), "returned state")
                _array(result.fun, name="returned residual")
                R, t = _chart(result.x, R0, t0, B)
                signs = translation_signs(R, t, r1, r2)
                selected_t = np.asarray(signs["t"])
                final_sampson, valid = _sampson(R, selected_t, xy1, xy2, K1, K2)
                if not valid.all():
                    raise NumericalFailure("undefined final fixed-support residual")
                final_residual = np.concatenate(((_rotate(R, n1)-n2).ravel()/NORMAL_SCALE,
                                                 final_sampson/PIXEL_SCALE))
                # Enumerate the cheirality-selected sign first; the other sign
                # has identical epipolar/normal objective and MSAC score.
                order = [signs["selected_index"], 1-signs["selected_index"]]
                candidates = [dict(status="completed", R=R.tolist(),
                                   t=signs["hypotheses"][j]["t"],
                                   sign_hypothesis_index=j,
                                   positive_count=signs["hypotheses"][j]["positive_count"])
                              for j in order]
                attempts.append(dict(status="completed", index=index, label=initial["label"],
                                     R=R.tolist(), t=signs["t"], cheirality=signs,
                                     candidates=candidates,
                                     initial_sum_squared_residual=float(np.dot(initial_residual, initial_residual)),
                                     sum_squared_residual=float(np.dot(final_residual, final_residual)),
                                     residual=final_residual.tolist(), x=result.x.tolist(),
                                     converged=bool(result.success), optimizer_status=int(result.status),
                                     message=str(result.message), nfev=int(result.nfev),
                                     njev=int(result.njev) if result.njev is not None else None,
                                     residual_calls=evaluations, **_support(count),
                                     seconds=time.perf_counter()-start))
        except (NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
            attempts.append(_failed(exc, start, count, index=index, label=initial["label"],
                                    residual_calls=evaluations))
    return dict(status="completed", attempts=attempts, **_support(count),
                settings=dict(normal_chord_scale=NORMAL_SCALE, sampson_pixels=PIXEL_SCALE,
                              max_nfev=MAX_NFEV, translation_chart="stereographic_2d"),
                seconds=time.perf_counter()-begin)


def sample_normal(normal_map, mask, xy):
    """Bilinear normals at array coordinates; all four masked neighbors required.

    The four stored vectors and their interpolant must be finite and nonzero.
    Interpolation precedes normalization. Unsupported queries remain explicit.
    """
    start = time.perf_counter()
    values = np.asarray(normal_map)
    validity = np.asarray(mask)
    points = _array(xy, name="normal sample coordinates")
    if (np.iscomplexobj(values) or values.ndim != 3 or values.shape[2] != 3
            or validity.shape != values.shape[:2] or validity.dtype != bool
            or points.ndim != 2 or points.shape[1] != 2):
        raise ValueError("invalid normal map/mask/coordinates contract")
    height, width = values.shape[:2]
    result = []
    for x, y in points:
        if x < 0. or y < 0. or x >= width-1 or y >= height-1:
            result.append(dict(status="unavailable", reason="four_neighbor_bounds", normal=None))
            continue
        ix, iy = int(np.floor(x)), int(np.floor(y))
        block = np.asarray(values[iy:iy+2, ix:ix+2], dtype=np.float64).reshape(4, 3)
        if not validity[iy:iy+2, ix:ix+2].all():
            result.append(dict(status="unavailable", reason="model_mask_hole", normal=None))
            continue
        if not np.isfinite(block).all() or np.any(np.max(np.abs(block), axis=1) == 0.):
            result.append(dict(status="unavailable", reason="invalid_neighbor_normal", normal=None))
            continue
        dx, dy = x-ix, y-iy
        weights = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
        with np.errstate(over="ignore", invalid="ignore"):
            interpolated = np.einsum("i,ij->j", weights, block, optimize=False)
            norm = float(np.hypot(np.hypot(interpolated[0], interpolated[1]), interpolated[2]))
        if not np.isfinite(norm) or norm <= 1e-8:
            result.append(dict(status="unavailable", reason="interpolated_normal_collapse", normal=None))
            continue
        result.append(dict(status="completed", reason=None, normal=(interpolated/norm).tolist()))
    return dict(status="completed", samples=result, query_count=len(points),
                usable_count=sum(r["status"] == "completed" for r in result),
                seconds=time.perf_counter()-start)
