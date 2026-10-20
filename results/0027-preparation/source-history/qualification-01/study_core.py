"""Truth-free initializer qualification over a strict nine-observation contract.

No generation, file access, model, labels, metric selection, or native import.
The caller injects the already-qualified native module. Unexpected API errors
propagate; numerical and unavailable branches remain explicit records.
"""
import math
import time
import warnings
import numpy as np
import controls
import tangent_solver


ARMS = ("T", "S", "V", "D8")
EPSILONS = (.25, 1., 4., 16.)
# Row-major nine-point grid. Center/axis first, then cyclic adjacent corners.
TRIANGLES = ((4, 3, 1), (4, 3, 7), (4, 5, 1), (4, 5, 7),
             (4, 0, 2), (4, 2, 8), (4, 8, 6), (4, 6, 0))
OBSERVATION_KEYS = frozenset(("K", "xy1", "xy2", "d1", "d2",
                              "center_index", "derivatives"))
DERIVATIVE_KEYS = frozenset(("status", "mode", "A", "g1", "g2", "fit",
                             "elapsed_seconds", "reason", "failure"))


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + " finite shape contract")
    return a.copy()


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("nonfinite study JSON")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    raise TypeError("unsupported study record")


def _mm(a, b):
    return np.einsum("ij,jk->ik", a, b, optimize=False)


def _rays(xy, K):
    return np.column_stack(((xy[:, 0]-K[0, 2])/K[0, 0],
                            (xy[:, 1]-K[1, 2])/K[1, 1], np.ones(len(xy))))


def _reject_label_keys(value):
    if isinstance(value, dict):
        if any(str(k).lower() in ("truth", "ground_truth", "gt") for k in value):
            raise ValueError("label key forbidden in observation")
        for item in value.values():
            _reject_label_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_label_keys(item)


def validate_observation(observation):
    if not isinstance(observation, dict) or set(observation) != OBSERVATION_KEYS:
        raise ValueError("unexpected/missing observation key; labels are forbidden")
    _reject_label_keys(observation)
    K = _array(observation["K"], (3, 3), "K")
    if not np.array_equal(K, np.array(((500., 0., 319.5), (0., 500., 239.5), (0., 0., 1.)))):
        raise ValueError("fixed pinhole calibration mismatch")
    if type(observation["center_index"]) is not int or observation["center_index"] != 4:
        raise ValueError("fixed center index mismatch")
    out = dict(K=K, center_index=4)
    for name, shape in (("xy1", (9, 2)), ("xy2", (9, 2)), ("d1", (9,)), ("d2", (9,))):
        out[name] = _array(observation[name], shape, name)
    derivative = observation["derivatives"]
    required = DERIVATIVE_KEYS-{"failure"}
    if (not isinstance(derivative, dict) or not required <= set(derivative)
            or not set(derivative) <= DERIVATIVE_KEYS):
        raise ValueError("derivative record contract")
    if derivative["status"] not in ("completed", "unavailable", "numerical_failure"):
        raise ValueError("incomplete derivative fit")
    if derivative["mode"] not in ("analytic", "center_fixed_ols"):
        raise ValueError("unknown derivative mode")
    seconds = derivative["elapsed_seconds"]
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0.:
        raise ValueError("invalid derivative timing")
    _safe(derivative)
    out["derivatives"] = derivative
    if derivative["status"] == "completed":
        out["A"] = _array(derivative["A"], (2, 2), "A")
        out["g1"] = _array(derivative["g1"], (2,), "g1")
        out["g2"] = _array(derivative["g2"], (2,), "g2")
    out["r1"], out["r2"] = _rays(out["xy1"], K), _rays(out["xy2"], K)
    return out


def _projection_terms(points, target_xy, K):
    """Invalid/behind-camera directions keep a full unit cost per point."""
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.)
    terms = np.ones(9)
    residuals = [None]*9
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in np.flatnonzero(valid):
            xy = points[i, :2]/points[i, 2]
            pixel = np.array((K[0, 0]*xy[0]+K[0, 2], K[1, 1]*xy[1]+K[1, 2]))
            delta = pixel-target_xy[i]
            square = float(np.sum(delta*delta))
            if not np.isfinite(pixel).all() or not math.isfinite(square):
                valid[i] = False
                continue
            residuals[i] = delta.tolist()
            terms[i] = min(square/4., 1.)
    return dict(terms=terms.tolist(), valid=valid.tolist(), residual_pixels=residuals,
                invalid_projection_count=int(np.sum(~valid)))


def validate_and_score(candidate, observation):
    """Only observations enter: all-nine physical checks and 18 capped terms."""
    row = dict(status="invalid", reason=None, pose_usable=False, score=None,
               R=candidate.get("R"), t=candidate.get("t"),
               a=candidate.get("a"), b=candidate.get("b"), c=candidate.get("c"))
    # Explicit native numerical-output records are observations, not malformed ABI.
    if any(row[key] is None for key in ("R", "t", "a", "b", "c")):
        row["reason"] = "missing_or_nonfinite_solver_geometry"
        return _safe(row)
    if candidate.get("status") in ("invalid", "numerical_failure"):
        row["reason"] = "solver_reported_invalid_candidate"
        return _safe(row)
    R, t = _array(row["R"], (3, 3), "candidate R"), _array(row["t"], (3,), "candidate t")
    values = _array([row["b"], row["a"], row["c"]], (3,), "candidate calibration")
    b, a, c = map(float, values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z1, z2 = observation["d1"]+b, a*observation["d2"]+c
        finite = np.isfinite(z1) & np.isfinite(z2)
        positive = finite & (z1 > 0.) & (z2 > 0.)
        row["all_nine_depths"] = dict(
            source=[float(v) if math.isfinite(float(v)) else None for v in z1],
            target=[float(v) if math.isfinite(float(v)) else None for v in z2],
            count=9, finite_count=int(np.sum(finite)), positive_count=int(np.sum(positive)))
        orth = float(np.max(np.abs(_mm(R.T, R)-np.eye(3))))
        det = float(np.linalg.det(R))
        if not math.isfinite(orth) or not math.isfinite(det):
            row["reason"] = "nonfinite_rotation_check"
            return _safe(row)
        row.update(rotation_orthogonality_max=orth, rotation_determinant=det,
                   translation_direction_defined=bool(np.any(t != 0.)))
        if orth > 1e-8 or abs(det-1.) > 1e-8:
            row["reason"] = "improper_or_nonorthogonal_rotation"
        elif not np.any(t != 0.):
            row["reason"] = "undefined_translation_direction"
        elif a <= 0. or not np.all(positive):
            row["reason"] = "nonpositive_all_nine_calibration"
        elif candidate.get("pose_usable") is False:
            # In particular, do not resurrect a native virtual-point branch
            # whose own fitting triangle has nonpositive corrected depths.
            row["reason"] = "solver_reported_nonusable_candidate"
        else:
            X, Y = z1[:, None]*observation["r1"], z2[:, None]*observation["r2"]
            forward = _projection_terms(_mm(X, R.T)+t, observation["xy2"], observation["K"])
            backward = _projection_terms(_mm(Y-t, R), observation["xy1"], observation["K"])
            score = float((sum(forward["terms"])+sum(backward["terms"]))/18.)
            row.update(status="completed", reason=None, pose_usable=True, score=score,
                       forward=forward, backward=backward)
    return _safe(row)


def _tangent_inputs(o):
    E = np.array(((1./500., 0.), (0., 1./500.), (0., 0.)))
    return dict(r1=o["r1"][4], r2=o["r2"][4], E1=E, E2=E,
                d1=o["d1"][4], d2=o["d2"][4], g1=o["g1"], g2=o["g2"], A=o["A"])


def _virtual_inputs(o, epsilon):
    steps = np.array(((0., 0.), (epsilon, 0.), (0., epsilon)))
    xy1 = o["xy1"][4]+steps
    target_steps = _mm(steps, o["A"].T)
    xy2 = o["xy2"][4]+target_steps
    d1 = o["d1"][4]+np.einsum("ij,j->i", steps, o["g1"], optimize=False)
    d2 = o["d2"][4]+np.einsum("ij,j->i", target_steps, o["g2"], optimize=False)
    for value in (xy1, xy2, d1, d2):
        if not np.isfinite(value).all():
            raise FloatingPointError("nonfinite virtual observation")
    return dict(x1=_rays(xy1, o["K"]), x2=_rays(xy2, o["K"]), d1=d1, d2=d2)


def process(observation, native, progress=None):
    start = time.perf_counter()
    o = validate_observation(observation)
    validated = time.perf_counter()
    result = dict(schema="evr0027.study-row.v1", status="started", active_attempt=None,
        arms={arm: dict(status="started", attempts=[], candidates=[], selected_index=None,
                        selected=None, timings=dict(solver_seconds=0., scoring_seconds=0., total_seconds=0.))
              for arm in ARMS},
        common_timing=dict(derivative_fit_seconds=float(o["derivatives"]["elapsed_seconds"]),
                           validation_seconds=validated-start),
        call_counts=dict(native_attempted=0, native_executed=0), warnings=[],
        timing_scope="process includes progress callbacks; arm solver time includes argument preparation and in-call progress")

    def publish():
        if progress is not None:
            progress(_safe(result))

    def attempt(arm, name, function, inputs, native_call=False, skip=None):
        bank = result["arms"][arm]
        attempt_index = len(bank["attempts"])
        row = dict(name=name, status="started", solver_called=False, native_called=False,
                   elapsed_seconds=0., scoring_seconds=0., result=None)
        bank["attempts"].append(row)
        result["active_attempt"] = dict(arm=arm, index=attempt_index, name=name)
        publish()
        if native_call:
            result["call_counts"]["native_attempted"] += 1
        if skip is not None:
            row.update(status="unavailable", reason=skip)
            publish()
            return
        begin = time.perf_counter()
        captured = []
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                arguments = inputs()  # including virtual-point construction
                row["inputs"] = _safe({k: v for k, v in arguments.items() if k != "native"})
                publish()
                row["solver_called"] = True
                if native_call:
                    row["native_called"] = True
                    result["call_counts"]["native_executed"] += 1
                # Durable invocation intent before control crosses the native
                # boundary; these flags do not assert a completed return.
                row["call_state"] = "invocation_intent"
                publish()
                produced = function(**arguments)
                row["call_state"] = "returned"
            row.update(status=produced["status"], result=produced)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            produced = dict(status="numerical_failure", reason="expected_numerical_failure",
                            failure=dict(type=type(error).__name__, message=str(error)), candidates=[])
            row.update(status=produced["status"], result=produced)
        except BaseException as error:
            row.update(status="technical_failure",
                       failure=dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter()-begin
            bank["timings"]["solver_seconds"] += row["elapsed_seconds"]
            result["warnings"].extend(dict(arm=arm, attempt_index=attempt_index,
                category=type(w.message).__name__, message=str(w.message)) for w in captured)
            publish()
        scoring_start = time.perf_counter()
        for raw in produced.get("candidates", []):
            candidate = validate_and_score(raw, o)
            candidate.update(generation_index=len(bank["candidates"]), attempt_index=attempt_index,
                             attempt_name=name, solver_candidate=raw)
            bank["candidates"].append(candidate)
        row["scoring_seconds"] = time.perf_counter()-scoring_start
        bank["timings"]["scoring_seconds"] += row["scoring_seconds"]
        publish()

    badfit = None if o["derivatives"]["status"] == "completed" else "common_derivative_fit_unavailable"
    attempt("T", "quartic", tangent_solver.solve_tangent, lambda: _tangent_inputs(o), skip=badfit)
    attempt("S", "svd", controls.solve_scale_only, lambda: _tangent_inputs(o), skip=badfit)
    attempt("S", "ordered_frame", controls.solve_scale_only_ordered,
            lambda: _tangent_inputs(o), skip=badfit)
    for epsilon in EPSILONS:
        attempt("V", "epsilon_"+str(epsilon), controls.solve_points_native,
                lambda epsilon=epsilon: dict(_virtual_inputs(o, epsilon), native=native),
                native_call=True, skip=badfit)
    for index, triangle in enumerate(TRIANGLES):
        ids = list(triangle)
        attempt("D8", "triangle_"+str(index), controls.solve_points_native,
                lambda ids=ids: dict(x1=o["r1"][ids], x2=o["r2"][ids],
                    d1=o["d1"][ids], d2=o["d2"][ids], native=native),
                native_call=True)
    for bank in result["arms"].values():
        admissible = [(c["score"], i) for i, c in enumerate(bank["candidates"]) if c["pose_usable"]]
        if admissible:
            _, index = min(admissible)
            bank.update(status="completed", selected_index=index, selected=bank["candidates"][index])
        else:
            bank["status"] = "unavailable"
        bank["timings"]["total_seconds"] = (bank["timings"]["solver_seconds"]
                                            + bank["timings"]["scoring_seconds"])
    duration = time.perf_counter()-start
    result["common_timing"].update(process_seconds=duration,
        total_with_derivatives_seconds=duration+result["common_timing"]["derivative_fit_seconds"])
    result.update(status="completed", active_attempt=None)
    publish()
    return _safe(result)
