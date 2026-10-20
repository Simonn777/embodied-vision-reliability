"""Local physical-calibration controls for fixed-weight moment objectives.

No estimator, robust reweighting, bounds, jitter, retries or global certificate.
The caller supplies the same support minima and Moments to both objectives.
Only optimize_calibration imports SciPy. Raw signed costs are not clipped.
"""
import math
import time
import numpy as np
import moments


OPTIONS = {"maxiter": 80, "maxfun": 400, "ftol": 1e-12, "gtol": 1e-7}


class ControlArithmeticError(FloatingPointError):
    """An explicitly invalid/overflowed numerical trial, not a repaired one."""


def _vector(value, name, nonfinite_is_arithmetic=False):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(name + " must be a vector of length three")
    if not np.isfinite(value).all():
        error_type = ControlArithmeticError if nonfinite_is_arithmetic else ValueError
        raise error_type(name + " must be finite")
    return value


def _minima(m1, m2):
    if np.iscomplexobj((m1, m2)):
        raise ValueError("support minima must be real")
    value = np.asarray((m1, m2), dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("support minima must be finite scalars")
    return float(value[0]), float(value[1])


def physical_to_unconstrained(seed, minimum_depth1, minimum_depth2):
    """Map physical (b,a,c) to logs of positive support minima and scale."""
    b, a, c = _vector(seed, "seed")
    m1, m2 = _minima(minimum_depth1, minimum_depth2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        positive = np.array((b + m1, a, c + a*m2))
        if not np.isfinite(positive).all() or np.any(positive <= 0.):
            raise ValueError("seed must have positive scale and corrected support depths")
        return np.log(positive)


def _physical_and_jacobian(q, m1, m2):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        eu, a, ew = np.exp(q)
        b, c = eu-m1, ew-a*m2
        physical = np.array((b, a, c))
        J = np.array(((eu, 0., 0.), (0., a, 0.), (0., -a*m2, ew)))
        # Check representable physical feasibility too: subtraction can erase
        # a very small positive margin, and exp may underflow to zero.
        minima = np.array((b+m1, a*m2+c))
        if (not np.isfinite(physical).all() or not np.isfinite(J).all()
                or not np.isfinite(minima).all() or a <= 0. or np.any(minima <= 0.)):
            raise ControlArithmeticError("nonfinite or nonpositive represented physical trial")
    return physical, J


def unconstrained_to_physical(q, minimum_depth1, minimum_depth2):
    q = _vector(q, "coordinates")
    m1, m2 = _minima(minimum_depth1, minimum_depth2)
    return _physical_and_jacobian(q, m1, m2)[0]


def _profile_gradient(C, b, a, c, R):
    """Envelope derivative for a selected optimal proper Procrustes rotation.

    At a nonunique/nonsmooth optimum, this is the selected active derivative;
    no uniqueness, Hessian or smoothness is asserted and no jitter is applied.
    """
    Cpr, Crp, Crr = C[:3, 3:6], C[3:6, :3], C[3:6, 3:6]
    Tpp, Tpr, Trp, Trr = C[6:9, 6:9], C[6:9, 9:12], C[9:12, 6:9], C[9:12, 9:12]
    Pp, Pr, Rp, Rr = C[:3, 6:9], C[:3, 9:12], C[3:6, 6:9], C[3:6, 9:12]
    cross = (a*Rp+c*Rr, Pp+b*Rp, Pr+b*Rr)
    trace_derivative = np.array((np.trace(Cpr+Crp+2*b*Crr),
                                 np.trace(2*a*Tpp+c*(Tpr+Trp)),
                                 np.trace(a*(Tpr+Trp)+2*c*Trr)))
    return trace_derivative - 2.*np.array([
        np.einsum("ij,ji->", R, value, optimize=False) for value in cross])


def evaluate_control(model, q, minimum_depth1, minimum_depth2, objective):
    """Return cost and analytic log-coordinate gradient for objective G or F."""
    if objective not in ("G", "F"):
        raise ValueError("objective must be G (distance) or F (proper Procrustes)")
    q = _vector(q, "coordinates", nonfinite_is_arithmetic=True)
    m1, m2 = _minima(minimum_depth1, minimum_depth2)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            physical, J = _physical_and_jacobian(q, m1, m2)
            b, a, c = physical
            if objective == "G":
                detail = moments.distance_objective(model, b, a, c)
                gradient_physical = detail["gradient"]
            else:
                detail = moments.procrustes_objective(model, b, a, c)
                gradient_physical = _profile_gradient(model.C, b, a, c, detail["R"])
            gradient = np.einsum("ji,j->i", J, gradient_physical, optimize=False)
            if (not math.isfinite(detail["cost_raw"]) or not np.isfinite(gradient).all()
                    or not np.isfinite(gradient_physical).all()):
                raise ControlArithmeticError("nonfinite objective or envelope gradient")
    except ValueError as error:
        # moments reports explicit nonfinite outputs as ValueError. Shape or
        # programming ValueErrors are deliberately not relabeled numerical.
        if str(error).startswith("nonfinite "):
            raise ControlArithmeticError(str(error)) from error
        raise
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise ControlArithmeticError(str(error)) from error
    return {"objective": objective, "cost_raw": float(detail["cost_raw"]),
            "gradient": gradient, "gradient_physical": gradient_physical,
            "coordinates": q.copy(), "parameters": physical, "detail": detail}


def optimize_calibration(model, seed, minimum_depth1, minimum_depth2, objective):
    """One fixed L-BFGS-B run; failed numerical trials terminate without retry.

    last_feasible is the last finite *evaluation*, not necessarily an accepted
    iterate or the best candidate. The returned final_evaluation is matched to
    SciPy's final x without an uncounted objective re-evaluation. Complete outer
    timing must additionally include import/moment construction if applicable.
    """
    if objective not in ("G", "F"):
        raise ValueError("objective must be G or F")
    m1, m2 = _minima(minimum_depth1, minimum_depth2)
    seed = _vector(seed, "seed")
    q0 = physical_to_unconstrained(seed, m1, m2)
    import scipy
    from scipy.optimize import minimize

    evaluations = 0
    feasible_evaluations = 0
    last_feasible = None
    cache = {}

    def function(q):
        nonlocal evaluations, feasible_evaluations, last_feasible
        evaluations += 1
        value = evaluate_control(model, q, m1, m2, objective)
        feasible_evaluations += 1
        last_feasible = value
        cache[value["coordinates"].tobytes()] = value
        return value["cost_raw"], value["gradient"]

    start = time.perf_counter()
    result = None
    failure = None
    try:
        result = minimize(function, q0, jac=True, method="L-BFGS-B",
                          bounds=None, options=dict(OPTIONS))
    except (ControlArithmeticError, FloatingPointError, OverflowError,
            np.linalg.LinAlgError) as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    elapsed = time.perf_counter()-start
    report = None
    final = None
    if result is not None:
        x, jac = np.asarray(result.x), np.asarray(result.jac)
        finite_x = x.shape == (3,) and bool(np.isfinite(x).all())
        finite_jac = jac.shape == (3,) and bool(np.isfinite(jac).all())
        finite_cost = math.isfinite(float(result.fun))
        report = {"success": bool(result.success), "status": int(result.status),
                  "message": str(result.message), "nit": int(result.nit),
                  "nfev": int(result.nfev), "njev": int(result.njev),
                  "x": x if finite_x else None, "x_finite": finite_x,
                  "jac": jac if finite_jac else None, "jac_finite": finite_jac,
                  "fun": float(result.fun) if finite_cost else None,
                  "fun_finite": finite_cost}
        if finite_x:
            final = cache.get(np.asarray(x, dtype=np.float64).tobytes())
        if not (finite_x and finite_jac and finite_cost):
            failure = {"type": "nonfinite_optimizer_result", "message": str(result.message)}
        elif final is None:
            failure = {"type": "unrecorded_final_evaluation", "message": str(result.message)}
        elif not result.success:
            failure = {"type": "optimizer_nonconvergence", "message": str(result.message)}
    return {"status": "completed" if failure is None else "failed", "failure": failure,
            "objective": objective, "seed": seed.copy(), "initial_coordinates": q0,
            "minimum_depth1": m1, "minimum_depth2": m2,
            "method": "L-BFGS-B", "options": dict(OPTIONS), "bounds": None,
            "scipy_version": scipy.__version__, "optimizer": report,
            "objective_evaluations": evaluations, "feasible_evaluations": feasible_evaluations,
            "last_feasible": last_feasible, "final_evaluation": final,
            "optimization_seconds": elapsed, "global_optimum_claimed": False,
            "gradient_semantics": "analytic polynomial G; selected proper-Procrustes envelope F"}
