"""P26 fixed-support orchestration; only injected arrays/records/backend.

No file, network, label access or data allocation. Numerical proposal failures
stay explicit; input/binding/programming/native-ABI errors propagate. Costs are
core-only, with the historical baseline runtime reported separately.
"""
import hashlib
import math
import time
import numpy as np
import moments
import lift_relaxation
import calibration_controls as controls


SLOTS = ("continued_tc", "G_neutral", "G_incumbent", "G_lift",
         "F_neutral", "F_incumbent", "F_lift")
TIE_ORDER = ("incumbent",) + SLOTS
UNIONS = {
    "incumbent": ("incumbent",),
    "continued_tc": ("incumbent", "continued_tc"),
    "G_local": ("incumbent", "continued_tc", "G_neutral", "G_incumbent"),
    "F_local": ("incumbent", "continued_tc", "F_neutral", "F_incumbent"),
    "G_lift": ("incumbent", "continued_tc", "G_lift"),
    "F_lift": ("incumbent", "continued_tc", "F_lift"),
    "F_all": ("incumbent", "continued_tc", "F_neutral", "F_incumbent", "F_lift"),
    "G_all": ("incumbent", "continued_tc", "G_neutral", "G_incumbent", "G_lift"),
}


class CandidateFailure(ArithmeticError):
    pass


class ProgressError(RuntimeError):
    """A persistence callback failure must never become a proposal failure."""


def unpack_support(record, count):
    if type(record["mask_length"]) is not int or record["mask_length"] != count:
        raise ValueError("baseline mask length mismatch")
    if type(record["kept_count"]) is not int or not 0 <= record["kept_count"] <= count:
        raise ValueError("invalid baseline kept count")
    encoded = record["native_inliers_packbits_big_hex"]
    if not isinstance(encoded, str) or len(encoded) != 2*((count+7)//8):
        raise ValueError("baseline packed mask byte length mismatch")
    packed = bytes.fromhex(encoded)
    if packed.hex() != encoded:
        raise ValueError("baseline mask is not canonical lowercase hex")
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
    if np.any(bits[count:]):
        raise ValueError("baseline mask has nonzero padding")
    mask = np.ascontiguousarray(bits[:count], dtype=np.bool_)
    digest = hashlib.sha256(mask.tobytes()).hexdigest()
    if digest != record["mask_sha256"] or int(mask.sum()) != record["kept_count"]:
        raise ValueError("baseline mask hash/count mismatch")
    return mask


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape:
        raise ValueError(name + " shape contract mismatch")
    if not np.isfinite(value).all():
        raise CandidateFailure("nonfinite " + name)
    return value


def _parameters(value, d1, d2):
    b, a, c = _array(value, (3,), "physical parameters")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if a <= 0.:
            raise CandidateFailure("nonpositive calibration scale")
        shift2 = c/a
        z1, z2, native_z2 = d1+b, a*d2+c, a*(d2+shift2)
        if (not np.isfinite(shift2) or not np.isfinite(z1).all()
                or not np.isfinite(z2).all() or not np.isfinite(native_z2).all()):
            raise CandidateFailure("nonfinite corrected supported depths")
        if np.any(z1 <= 0.) or np.any(z2 <= 0.) or np.any(native_z2 <= 0.):
            raise CandidateFailure("nonpositive corrected supported depths")
    return np.array((b, a, c)), float(shift2)


def _pose(estimate, base_common, native_input=False):
    R = _array(estimate["R"], (3, 3), "candidate R")
    t = _array(estimate["t"], (3,), "candidate t")
    if np.max(np.abs(t)) == 0.:
        raise CandidateFailure("zero candidate translation")
    tolerance = 1e-8 if native_input else 1e-6
    orthogonal = np.einsum("ij,kj->ik", R, R, optimize=False)
    if (np.max(np.abs(orthogonal-np.eye(3))) > tolerance
            or abs(np.linalg.det(R)-1.) > tolerance):
        raise CandidateFailure("candidate rotation is not proper orthogonal")
    base_common.validate_pose({"R": R, "t": t})
    return R, t


def _numerical_exception(error):
    if isinstance(error, (CandidateFailure, FloatingPointError, OverflowError,
                          np.linalg.LinAlgError)):
        return True
    # These are explicitly numerical errors from the frozen arithmetic modules.
    if isinstance(error, ValueError):
        text = str(error)
        return (text.startswith("nonfinite ") or text == "H is materially indefinite"
                or text == "strictly interior x and positive mu required"
                or text == "cone dual is not positive semidefinite numerically")
    # Status 1 is handled only at the explicitly validated native-call site.
    # Status 2 (unspecified C++ exception) and ABI/load faults always propagate.
    raw = getattr(error, "raw", None)
    return (type(error).__name__ == "RefinementError" and getattr(error, "status", None) == 3
            and isinstance(raw, dict) and raw.get("native_called") == 1)


def _failure(error):
    result = {"type": type(error).__name__, "message": str(error)}
    if hasattr(error, "status"):
        result["native_status"] = error.status
    if hasattr(error, "raw"):
        result["native_raw"] = error.raw
    return result


def _saveable(value):
    """Finite JSON observations; every nonfinite scalar gets an explicit path."""
    invalid = []
    def visit(item, path):
        if isinstance(item, np.ndarray):
            return visit(item.tolist(), path)
        if isinstance(item, np.generic):
            return visit(item.item(), path)
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                invalid.append({"path": path, "value": str(item)})
                return None
            return item
        if isinstance(item, (list, tuple)):
            return [visit(v, path+"/"+str(i)) for i, v in enumerate(item)]
        if isinstance(item, dict) and all(isinstance(k, str) for k in item):
            return {k: visit(v, path+"/"+k) for k, v in item.items()}
        raise TypeError("Unsupported observation type at " + path)
    result = visit(value, "")
    result["nonfinite_observations"] = invalid
    return result


def _compare_scores(actual, saved):
    if set(actual) != set(saved):
        raise ValueError("incumbent final score schema mismatch")
    for key, value in actual.items():
        old = saved[key]
        if isinstance(value, (float, np.floating)):
            if (not math.isfinite(value) or not isinstance(old, (int, float))
                    or not math.isfinite(old) or abs(value-old) > 1e-10):
                raise ValueError("incumbent full-array score mismatch: " + key)
        elif value != old:
            raise ValueError("incumbent full-array score count mismatch: " + key)


def process_pair(pair, baseline_record, native_refiner, base_common, pair_index, progress=None):
    started = time.perf_counter()
    if type(pair_index) is not int or not 0 <= pair_index < 80:
        raise ValueError("invalid pair index")
    if (baseline_record["pair_index"] != pair_index or baseline_record["status"] != "completed"
            or baseline_record["arm"] != "depth_affine"):
        raise ValueError("baseline identity/status mismatch")
    x1 = np.asarray(pair["x1"], dtype=np.float64)
    if x1.ndim != 2 or x1.shape[1] != 2 or len(x1) == 0:
        raise ValueError("invalid input x1 shape")
    n = len(x1)
    # Input errors are outside all candidate-error handlers and stop the cohort.
    for key, shape in (("x1", (n, 2)), ("x2", (n, 2)), ("depth1", (n,)), ("depth2", (n,))):
        value = np.asarray(pair[key], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid input array " + key)
    if base_common.json_safe(pair["calibration"]) != baseline_record["calibration"]:
        raise ValueError("baseline calibration mismatch")
    g = base_common.prepare_geometry(pair["x1"], pair["x2"], pair["calibration"])
    mask = unpack_support(baseline_record, n)
    count = int(mask.sum())
    d1, d2 = np.asarray(pair["depth1"], dtype=float), np.asarray(pair["depth2"], dtype=float)
    sd1, sd2 = d1[mask], d2[mask]
    minimum1 = float(np.min(sd1)) if count else None
    minimum2 = float(np.min(sd2)) if count else None
    geometry = baseline_record["geometry"]
    base_common.validate_pose(baseline_record["estimate"])
    base_common.validate_pose(geometry)
    for key in ("R", "t"):
        if not np.array_equal(np.asarray(geometry[key]), np.asarray(baseline_record["estimate"][key])):
            raise ValueError("baseline estimate/raw geometry mismatch")
    if not np.isfinite([geometry[k] for k in ("scale", "shift1", "shift2")]).all():
        raise ValueError("nonfinite baseline nuisance parameters")
    incumbent_scores = base_common.final_scores(baseline_record["estimate"], g)
    _compare_scores(incumbent_scores, baseline_record["final_scores"])
    incumbent = {"status": "completed", "selected_attempt": "incumbent",
        "estimate": baseline_record["estimate"], "geometry": geometry,
        "final_scores": incumbent_scores,
        "depth_diagnostics": base_common.depth_diagnostics(geometry, g, d1, d2, mask)}
    focal1 = float((g["params1"][0]+g["params1"][1])/2.)
    focal2 = float((g["params2"][0]+g["params2"][1])/2.)
    if not np.isfinite([focal1, focal2]).all() or min(focal1, focal2) <= 0.:
        raise ValueError("invalid effective focal lengths")
    preparations = {"common": {"status": "completed", "seconds": time.perf_counter()-started}}
    order = SLOTS[pair_index % 7:] + SLOTS[:pair_index % 7]
    attempts = {}
    persistence_seconds = 0.

    def emit(phase):
        nonlocal persistence_seconds
        if progress is None:
            return
        tick = time.perf_counter()
        snapshot = {"schema": "evr0026.pair-core.v1", "status": "started",
            "pair_index": pair_index, "phase": phase, "incumbent": incumbent,
            "support": {"count": count, "length": n, "mask_sha256": baseline_record["mask_sha256"]},
            "preparations": preparations, "execution_order": order, "attempts": attempts,
            "timings": {"elapsed_seconds": tick-started,
                        "progress_callback_seconds": persistence_seconds}}
        # Callback and serialization faults are technical faults. A detached
        # finite snapshot cannot mutate the live numerical objects.
        try:
            progress(_saveable(snapshot))
        except Exception as error:
            raise ProgressError("progress callback failed: " + str(error)) from error
        persistence_seconds += time.perf_counter()-tick

    model = None
    emit("before_moments")
    begin = time.perf_counter()
    if count < 4:
        preparations["moments"] = {"status": "unavailable", "reason": "support_count_below_four"}
    else:
        try:
            model = moments.build_moments(g["h1"], g["h2"], d1, d2, mask.astype(float))
            preparations["moments"] = {"status": "completed", "result": {
                "H": model.H, "C": model.C, "mean": model.mean,
                "weight_sum": model.weight_sum, "input_count": model.input_count,
                "support_count": model.support_count}}
        except Exception as error:
            if not _numerical_exception(error):
                raise
            preparations["moments"] = {"status": "failed", "failure": _failure(error)}
    preparations["moments"]["seconds"] = time.perf_counter()-begin
    emit("before_relaxation")
    begin = time.perf_counter()
    if model is None:
        preparations["relaxation"] = {"status": "unavailable", "reason": "moments_unavailable"}
    else:
        try:
            relaxation = lift_relaxation.solve_relaxation(model.H, minimum1, minimum2, max_newton=40)
            preparations["relaxation"] = {"status": "completed" if relaxation["status"] == "completed" else "failed",
                                          "result": relaxation}
        except Exception as error:
            if not _numerical_exception(error):
                raise
            preparations["relaxation"] = {"status": "failed", "failure": _failure(error)}
    preparations["relaxation"]["seconds"] = time.perf_counter()-begin
    emit("preparations_complete")
    for execution_index, name in enumerate(order):
        tick = time.perf_counter()
        attempt = {"name": name, "execution_index": execution_index, "status": "started",
            "phase": "seed", "optimizer": None, "recovery": None, "native": None,
            "estimate": None, "geometry": None, "final_scores": None, "depth_diagnostics": None,
            "timings": {k: 0. for k in ("seed_seconds", "optimizer_seconds", "recovery_seconds",
                                        "native_seconds", "score_seconds", "depth_diagnostics_seconds")}}
        attempts[name] = attempt
        emit("before_slot:"+name)
        try:
            mark = time.perf_counter()
            try:
                if count < 4:
                    raise CandidateFailure("support_count_below_four")
                if name == "continued_tc" or name.endswith("_incumbent"):
                    with np.errstate(over="raise", invalid="raise"):
                        seed = np.array((geometry["shift1"], geometry["scale"],
                                         np.float64(geometry["scale"])*geometry["shift2"]))
                elif name.endswith("_neutral"):
                    seed = np.array((0., 1., 0.))
                else:
                    prep = preparations["relaxation"]
                    if (prep["status"] != "completed"
                            or prep["result"]["physical"]["status"] != "feasible"):
                        raise CandidateFailure("completed_physical_lift_seed_unavailable")
                    seed = np.asarray(prep["result"]["physical"]["parameters"], dtype=float)
                attempt["seed_parameters"] = seed.copy()
                parameters, _ = _parameters(seed, sd1, sd2)
                if name != "continued_tc" and model is None:
                    raise CandidateFailure("moments_unavailable")
            finally:
                attempt["timings"]["seed_seconds"] = time.perf_counter()-mark
            if name == "continued_tc":
                initial_pose = geometry
            else:
                attempt["phase"] = "optimizer"
                emit("before_optimizer:"+name)
                mark = time.perf_counter()
                try:
                    optimization = controls.optimize_calibration(model, parameters, minimum1, minimum2, name[0])
                    attempt["optimizer"] = optimization
                    if optimization["status"] != "completed":
                        raise CandidateFailure("optimizer_not_completed")
                    final = optimization["final_evaluation"]
                    if final is None:
                        raise ValueError("completed optimizer missing final evaluation")
                    if optimization["optimizer"]["success"] is not True:
                        raise ValueError("completed optimizer has unsuccessful result")
                    if not math.isfinite(float(final["cost_raw"])):
                        raise CandidateFailure("nonfinite completed optimizer cost")
                    for key in ("gradient", "gradient_physical", "coordinates"):
                        _array(final[key], (3,), "completed optimizer "+key)
                    parameters, _ = _parameters(final["parameters"], sd1, sd2)
                finally:
                    attempt["timings"]["optimizer_seconds"] = time.perf_counter()-mark
                attempt["phase"] = "recovery"
                emit("before_recovery:"+name)
                mark = time.perf_counter()
                try:
                    recovery = moments.procrustes_objective(model, *parameters)
                    attempt["recovery"] = recovery
                    initial_pose = {"R": recovery["R"], "t": recovery["t"]}
                finally:
                    attempt["timings"]["recovery_seconds"] = time.perf_counter()-mark
            attempt["phase"] = "native"
            emit("before_native:"+name)
            mark = time.perf_counter()
            try:
                R, t = _pose(initial_pose, base_common, native_input=True)
                b, a, c = parameters
                attempt["native_initial"] = {"R": R, "t": t, "b": b, "a": a, "c": c}
                try:
                    native = native_refiner.refine(g["h1"][mask, :2], g["h2"][mask, :2],
                        sd1, sd2, R, t, a, b, c, focal1, focal2, max_iterations=100)
                except Exception as error:
                    # Only this site has already checked shapes, finite values,
                    # proper rotation, nonzero t and supported input depths.
                    if type(error).__name__ == "RefinementError" and getattr(error, "status", None) == 1:
                        attempt["native_input_rejection"] = _failure(error)
                        raise CandidateFailure("native_rejected_physical_input") from error
                    raise
                attempt["native"] = native
                if (native["schema"] != "evr0026.native-refinement.v1"
                        or native["output_checks"]["native_called"] is not True
                        or native["output_checks"]["completed"] is not True):
                    raise ValueError("native ABI/result schema mismatch")
            finally:
                attempt["timings"]["native_seconds"] = time.perf_counter()-mark
            attempt["phase"] = "score"
            mark = time.perf_counter()
            try:
                final_geometry = native["geometry"]
                R, t = _pose(final_geometry, base_common)
                nuisance = np.asarray([final_geometry[k] for k in ("scale", "shift1", "shift2")], dtype=float)
                if not np.isfinite(nuisance).all():
                    raise CandidateFailure("nonfinite native nuisance parameters")
                estimate = {"R": R, "t": t}
                attempt.update(estimate=estimate, geometry=final_geometry,
                               final_scores=base_common.final_scores(estimate, g))
            finally:
                attempt["timings"]["score_seconds"] = time.perf_counter()-mark
            attempt["phase"] = "depth_diagnostics"
            mark = time.perf_counter()
            try:
                # Native post-refinement depth violations are diagnostic only.
                attempt["depth_diagnostics"] = base_common.depth_diagnostics(final_geometry, g, d1, d2, mask)
            finally:
                attempt["timings"]["depth_diagnostics_seconds"] = time.perf_counter()-mark
            attempt.update(status="completed", phase="complete")
        except Exception as error:
            if not _numerical_exception(error):
                raise
            attempt.update(status="unavailable" if attempt["phase"] == "seed" or
                           "native_input_rejection" in attempt else "failed",
                           failed_phase=attempt["phase"], phase="finished_failure", failure=_failure(error))
        finally:
            attempt["timings"]["total_seconds"] = time.perf_counter()-tick
        emit("slot_complete:"+name)
    candidates = {"incumbent": incumbent, **attempts}
    selectors = {}
    for name, members in UNIONS.items():
        available = [k for k in members if candidates[k]["status"] == "completed"
                     and candidates[k]["final_scores"] is not None]
        winner = min(available, key=lambda k: (candidates[k]["final_scores"]["pixel_msac"], TIE_ORDER.index(k)))
        prerequisites = ["common"]
        if any(k.startswith(("G_", "F_")) for k in members):
            prerequisites.append("moments")
        if any(k.endswith("_lift") for k in members):
            prerequisites.append("relaxation")
        prep_seconds = sum(preparations[k]["seconds"] for k in prerequisites)
        attempt_seconds = sum(attempts[k]["timings"]["total_seconds"] for k in members if k != "incumbent")
        chosen = candidates[winner]
        selectors[name] = {"members": members, "available_members": available,
            "selected_attempt": winner, "score": chosen["final_scores"]["pixel_msac"],
            "final_scores": chosen["final_scores"], "estimate": chosen["estimate"],
            "geometry": chosen["geometry"], "prerequisites": prerequisites,
            "preparation_seconds": prep_seconds, "attempt_seconds": attempt_seconds,
            "charged_seconds": prep_seconds+attempt_seconds,
            "timing_scope": "core-only; excludes outer input/process and historical baseline"}
    record = {"schema": "evr0026.pair-core.v1", "status": "completed", "pair_index": pair_index,
        "support": {"count": count, "length": n, "mask_sha256": baseline_record["mask_sha256"],
            "native_inliers_packbits_big_hex": baseline_record["native_inliers_packbits_big_hex"],
            "minimum_depth1": minimum1, "minimum_depth2": minimum2,
            "weight_semantics": "exact fixed 0/1 saved pre-final epipolar mask"},
        "incumbent": incumbent, "preparations": preparations, "execution_order": order,
        "tie_order": TIE_ORDER, "attempts": attempts, "selectors": selectors,
        "historical_baseline_timings": baseline_record.get("timings"),
        "timings": {"core_processing_seconds": time.perf_counter()-started,
                    "progress_callback_seconds": persistence_seconds,
                    "native_attempt_count": sum(v["native"] is not None or
                        v.get("failure", {}).get("native_raw", {}).get("native_called") == 1 for v in attempts.values())},
        "primary_selector": "G_lift", "strongest_F_control": "F_all"}
    return _saveable(record)
