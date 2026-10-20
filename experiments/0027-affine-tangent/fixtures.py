"""Deferred manufactured observations for the reviewed 0027 plan, revision 2.

Nothing is generated at import. Truth and provenance are separate from the
strict observation dictionary consumed by study_core. No file or model access.
"""
import math
import time
import numpy as np


SEED = 20261019
CENTER = 4
K_VALUES = ((500., 0., 319.5), (0., 500., 239.5), (0., 0., 1.))
OBSERVATION_KEYS = frozenset(("K", "xy1", "xy2", "d1", "d2",
                              "center_index", "derivatives"))


def _mm(a, b):
    return np.einsum("ij,jk->ik", a, b, optimize=False)


def _finite(a):
    if not np.isfinite(a).all():
        raise FloatingPointError("nonfinite manufactured arithmetic")
    return a


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + " shape or finite contract")
    return a.copy()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("nonfinite JSON observation")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    raise TypeError("unsupported record")


def exp_so3(w):
    """Rodrigues formula, with its exact zero limit."""
    w = _array(w, (3,), "rotation vector")
    theta = float(np.linalg.norm(w))
    W = np.array(((0., -w[2], w[1]), (w[2], 0., -w[0]),
                  (-w[1], w[0], 0.)))
    if theta == 0.:
        return np.eye(3)
    return _finite(np.eye(3) + (math.sin(theta)/theta)*W
                   + ((1.-math.cos(theta))/(theta*theta))*_mm(W, W))


def _rays(xy, K):
    return np.column_stack(((xy[:, 0]-K[0, 2])/K[0, 0],
                            (xy[:, 1]-K[1, 2])/K[1, 1],
                            np.ones(len(xy))))


def _surface(case, delta):
    x, y = delta[:, 0], delta[:, 1]
    if case == "G":
        return 1.+x, np.array((1., 0.))
    if case == "F":
        return np.full(len(delta), 2.5), np.zeros(2)
    if case in ("M", "P"):
        return 2.5+.8*x-.6*y+.1*x*x+.15*x*y-.1*y*y, np.array((.8, -.6))
    raise ValueError("unknown fixed physical case")


def _physical(case, calibration):
    if calibration not in ("scale_only", "shifted"):
        raise ValueError("unknown fixed calibration")
    K = np.array(K_VALUES)
    offsets = np.array([(x, y) for y in (-16., 0., 16.)
                        for x in (-16., 0., 16.)])
    center = np.array((0., 0.)) if case == "G" else np.array((.12, -.08))
    if case == "G":
        R = np.array(((2., -2., 1.), (2., 1., -2.), (1., 2., 2.)))/3.
        t = np.array((1./6., 2./3., 1./3.))
    elif case in ("M", "P", "F"):
        R, t = exp_so3((.12, -.18, .08)), np.array((.25, -.05, .10))
    else:
        raise ValueError("unknown fixed physical case")
    delta = offsets/500.
    z1, dz1 = _surface(case, delta)
    r1 = np.column_stack((center+delta, np.ones(9)))
    X = z1[:, None]*r1
    if case == "P":
        rotated_center = np.einsum("ij,j->i", R, X[CENTER], optimize=False)
        t = np.array((-rotated_center[0], -rotated_center[1], .10))
    Y = _mm(X, R.T)+t
    z2 = Y[:, 2]
    if not np.all(z1 > 0.) or not np.all(z2 > 0.):
        raise ValueError("fixed physical fixture has nonpositive depth")
    xy1 = r1[:, :2]*500.+np.array((319.5, 239.5))
    xy2 = (Y[:, :2]/z2[:, None])*500.+np.array((319.5, 239.5))
    b, a, c = (0., 1.3, 0.) if calibration == "scale_only" else (.25, 1.3, -.20)
    d1, d2 = z1-b, (z2-c)/a
    E = np.array(((1./500., 0.), (0., 1./500.), (0., 0.)))
    U = z1[CENTER]*E+np.outer(r1[CENTER], dz1/500.)
    V = _mm(R, U)
    q = Y[CENTER]/z2[CENTER]
    projection = (500./z2[CENTER])*np.array(((1., 0., -q[0]),
                                           (0., 1., -q[1])))
    A = _mm(projection, V)
    g1, g2 = dz1/500., np.linalg.solve(A.T, V[2]/a)
    for value in (R, t, xy1, xy2, d1, d2, A, g1, g2):
        _finite(value)
    observations = dict(K=K, xy1=xy1, xy2=xy2, d1=d1, d2=d2,
                        center_index=CENTER)
    truth = dict(R=R, t=t, b=b, a=a, c=c, X=X, Y=Y,
                 axial_depth1=z1, axial_depth2=z2,
                 source_center_normalized=center,
                 exact_derivatives=dict(A=A, g1=g1, g2=g2))
    if case == "F":
        truth["calibration_family"] = [
            dict(k=k, b=k*(d1[CENTER]+b)-d1[CENTER],
                 a=k*a, c=k*c, R=R, t=k*t) for k in (.8, 1., 1.2)]
    return observations, truth


def fit_derivatives(xy1, xy2, d1, d2, center_index=CENTER):
    """Center-fixed OLS, including the zero center row in all nine equations."""
    start = time.perf_counter()
    xy1, xy2 = _array(xy1, (9, 2), "xy1"), _array(xy2, (9, 2), "xy2")
    d1, d2 = _array(d1, (9,), "d1"), _array(d2, (9,), "d2")
    if type(center_index) is not int or center_index != CENTER:
        raise ValueError("fixed center index must be four")
    result = dict(status="started", mode="center_fixed_ols", A=None, g1=None,
                  g2=None, fit=None)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            P, Q = xy1-xy1[CENTER], xy2-xy2[CENTER]
            y1, y2 = d1-d1[CENTER], d2-d2[CENTER]
            sv1, sv2 = np.linalg.svd(P, compute_uv=False), np.linalg.svd(Q, compute_uv=False)
            _finite(sv1); _finite(sv2)
            th1 = max(P.shape)*np.finfo(float).eps*sv1[0]
            th2 = max(Q.shape)*np.finfo(float).eps*sv2[0]
            rank1, rank2 = int(np.sum(sv1 > th1)), int(np.sum(sv2 > th2))
            result["fit"] = dict(source_differences=P, target_differences=Q,
                depth1_differences=y1, depth2_differences=y2,
                source_singular_values=sv1, target_singular_values=sv2,
                source_rank=rank1, target_rank=rank2,
                source_rank_threshold=th1, target_rank_threshold=th2)
            if rank1 != 2 or rank2 != 2:
                result.update(status="unavailable", reason="rank_deficient_neighborhood")
            else:
                rcond = max(P.shape)*np.finfo(float).eps
                B = _finite(np.linalg.lstsq(P, np.column_stack((Q, y1)), rcond=rcond)[0])
                g2 = _finite(np.linalg.lstsq(Q, y2, rcond=rcond)[0])
                A, g1 = B[:, :2].T, B[:, 2]
                result.update(status="completed", reason=None, A=A, g1=g1, g2=g2)
                result["fit"].update(
                    warp_residual=_finite(_mm(P, A.T)-Q),
                    depth1_residual=_finite(np.einsum("ij,j->i", P, g1, optimize=False)-y1),
                    depth2_residual=_finite(np.einsum("ij,j->i", Q, g2, optimize=False)-y2))
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        result.update(status="numerical_failure", reason="derivative_fit_arithmetic",
                      failure=dict(type=type(error).__name__, message=str(error)))
    result["elapsed_seconds"] = time.perf_counter()-start
    return json_safe(result)


def make_clean(case, calibration, derivative_mode):
    """Deferred fixed clean example; returns (observation, truth)."""
    if derivative_mode not in ("exact", "fitted"):
        raise ValueError("derivative mode")
    start = time.perf_counter()
    observation, truth = _physical(case, calibration)
    if derivative_mode == "exact":
        # Time the physical/analytic construction rather than report zero work.
        derivatives = dict(truth["exact_derivatives"], status="completed",
            mode="analytic", reason=None, fit=None,
            elapsed_seconds=time.perf_counter()-start)
    else:
        derivatives = fit_derivatives(observation["xy1"], observation["xy2"],
                                      observation["d1"], observation["d2"])
    observation["derivatives"] = derivatives
    return json_safe(observation), json_safe(truth)


def iter_cases():
    """Yield the complete fixed 138 records only when explicitly iterated."""
    row = 0
    for case in ("G", "M"):
        for calibration in ("scale_only", "shifted"):
            for mode in ("exact", "fitted"):
                observation, truth = make_clean(case, calibration, mode)
                yield dict(row_id=row, observation=observation, truth=truth,
                    provenance=dict(case=case, calibration=calibration,
                        regime="clean", derivative_mode=mode, realization=None,
                        noise_draws=None))
                row += 1
    rng = np.random.Generator(np.random.PCG64(SEED))
    for case in ("G", "M"):
        for calibration in ("scale_only", "shifted"):
            clean, truth = _physical(case, calibration)
            for realization in range(16):
                before = json_safe(rng.bit_generator.state)
                pixel_draws = rng.standard_normal((2, 9, 2))
                depth_draws = rng.standard_normal((2, 9))
                after = json_safe(rng.bit_generator.state)
                depth_units = np.array((truth["axial_depth1"][CENTER],
                                       truth["axial_depth2"][CENTER]/1.3))
                for regime, pixel_sigma, depth_sigma in (("L", .1, .001), ("H", .5, .01)):
                    xy1 = clean["xy1"]+pixel_sigma*pixel_draws[0]
                    xy2 = clean["xy2"]+pixel_sigma*pixel_draws[1]
                    d1 = clean["d1"]+depth_sigma*depth_units[0]*depth_draws[0]
                    d2 = clean["d2"]+depth_sigma*depth_units[1]*depth_draws[1]
                    derivatives = fit_derivatives(xy1, xy2, d1, d2)
                    observation = dict(K=clean["K"], xy1=xy1, xy2=xy2,
                        d1=d1, d2=d2, center_index=CENTER, derivatives=derivatives)
                    provenance = dict(case=case, calibration=calibration, regime=regime,
                        derivative_mode="fitted", realization=realization, seed=SEED,
                        noise_draws=dict(pixel=pixel_draws, depth=depth_draws),
                        rng_before=before, rng_after=after, pixel_sigma=pixel_sigma,
                        depth_sigmas=depth_sigma*depth_units,
                        exact_observation=clean)
                    yield dict(row_id=row, observation=json_safe(observation),
                               truth=json_safe(truth), provenance=json_safe(provenance))
                    row += 1
    for case in ("P", "F"):
        observation, truth = make_clean(case, "shifted", "exact")
        yield dict(row_id=row, observation=observation, truth=truth,
            provenance=dict(case=case, calibration="shifted", regime="clean",
                            derivative_mode="exact", realization=None, noise_draws=None))
        row += 1
    if row != 138:
        raise AssertionError("fixed complete cohort cardinality")
