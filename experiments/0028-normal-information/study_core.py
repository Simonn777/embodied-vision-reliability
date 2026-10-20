"""Pure 0028 known-control pair orchestration on caller-supplied fixed arrays.

No files, data loading, truth, model inference, or backend loading. The caller
injects an admitted PoseLib module and the qualified 0026 refiner instance.
Every method uses one E-derived/original-eligibility support; scoring uses ALL
selected original matches. Finite snapshots precede external native calls.
"""
import copy
import json
import math
import sys
import time

import numpy as np

import pose_controls as pc


SIMILARITY_ARMS = ("A", "B", "C", "LS", "IRLS", "ANG", "FULL")
CORRECTIONS = ("LS", "IRLS", "ANG")
ARMS = ("E",) + SIMILARITY_ARMS + ("NORMAL", "NORMAL_REFINE", "AFFINE")
AFFINE_SEED_ORDER = ("E", "C", "LS", "IRLS", "ANG")
E_OPTIONS = {
    "max_error": 2.0, "tangent_sampson": False, "real_focal_check": False,
    "ransac": {"min_iterations": 1000, "max_iterations": 100000,
               "dyn_num_trials_mult": 3.0, "success_prob": .9999, "seed": 0,
               "progressive_sampling": False, "max_prosac_iterations": 100000,
               "score_initial_model": False},
    "bundle": {},
}


class ProgressError(RuntimeError):
    """Persistence failure is a technical failure, never a numeric fallback."""


def _serial(value):
    if isinstance(value, np.ndarray):
        return _serial(value.tolist())
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("record keys must be strings")
        return {k: _serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serial(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite unmarked record")
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise TypeError("unsupported record value")
    return value


def _observed(value):
    """Preserve invalid native diagnostics with explicit nonfinite locations."""
    nonfinite = []

    def visit(v, path):
        if isinstance(v, np.ndarray):
            return visit(v.tolist(), path)
        if isinstance(v, np.generic):
            return visit(v.item(), path)
        if isinstance(v, dict):
            return {k: visit(x, path+"/"+str(k)) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [visit(x, path+"/"+str(i)) for i, x in enumerate(v)]
        if isinstance(v, float) and not math.isfinite(v):
            nonfinite.append(dict(path=path, value=repr(v)))
            return None
        return _serial(v)

    return dict(values=visit(value, ""), nonfinite=nonfinite)


def _array(value, shape, name, finite=True):
    if np.iscomplexobj(value):
        raise ValueError(name+" must be real")
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or (finite and not np.isfinite(a).all()):
        raise ValueError(name+" shape/finite input contract")
    return a.copy()


def _boolean(value, n, name):
    a = np.asarray(value)
    if a.dtype != bool or a.shape != (n,):
        raise ValueError(name+" must be an N-vector of bools")
    return a.copy()


def _camera(pair, which):
    K = _array(pair["K"+which], (3, 3), "K"+which)
    width, height = pair["width"+which], pair["height"+which]
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or not np.array_equal(K[2], [0., 0., 1.])
            or K[0, 1] != 0. or K[1, 0] != 0.
            or K[0, 0] <= 0. or K[1, 1] <= 0.):
        raise ValueError("authored PINHOLE camera contract")
    camera = dict(model="PINHOLE", width=width, height=height,
                  params=[float(K[0, 0]), float(K[1, 1]),
                          float(K[0, 2]), float(K[1, 2])])
    return K, camera


def _inputs(pair):
    x = np.asarray(pair["xy1"])
    if x.ndim != 2 or x.shape[1] != 2 or not len(x):
        raise ValueError("xy1 must be nonempty Nx2")
    n = len(x)
    out = dict(xy1=_array(x, (n, 2), "xy1"),
               xy2=_array(pair["xy2"], (n, 2), "xy2"), n=n)
    for side in ("1", "2"):
        K, camera = _camera(pair, side)
        out["K"+side], out["camera"+side] = K, camera
        out["depths"+side] = {k: _array(pair["depths"+side][k], (n,), k+side, False)
                              for k in SIMILARITY_ARMS}
        out["original_valid"+side] = _boolean(pair["original_valid"+side], n, "original_valid"+side)
        out["normal"+side] = _array(pair["normal"+side], (n, 3), "normal"+side, False)
        out["normal_valid"+side] = _boolean(pair["normal_valid"+side], n, "normal_valid"+side)
        valid = out["original_valid"+side]
        for name in ("A", "B", "C"):
            d = out["depths"+side][name][valid]
            if not np.isfinite(d).all() or np.any(d <= 0.):
                raise ValueError("original eligibility contradicts "+name+side)
        normals = out["normal"+side][out["normal_valid"+side]]
        if not np.isfinite(normals).all() or np.any(np.max(np.abs(normals), axis=1) == 0.):
            raise ValueError("normal eligibility contradicts values")
        corrections = pair["correction"+side]
        for name in CORRECTIONS:
            c = corrections[name]
            if c["status"] not in ("completed", "uninformative", "unavailable"):
                raise ValueError("unknown correction status")
            if type(c["admissible"]) is not bool:
                raise ValueError("correction admissibility must be bool")
            if c["admissible"] and (not isinstance(c["b"], (int, float, np.number))
                                    or isinstance(c["b"], (bool, np.bool_))
                                    or not math.isfinite(float(c["b"]))):
                raise ValueError("admissible correction offset invalid")
        out["correction"+side] = corrections
        homogeneous = np.column_stack((out["xy"+side], np.ones(n)))
        rays = np.einsum("ij,nj->ni", np.linalg.inv(K), homogeneous, optimize=False)
        out["r"+side] = rays/rays[:, 2:]
    if type(pair["comparator_valid"]) is not bool:
        raise ValueError("comparator_valid must be bool")
    return out


def _pose(R, t):
    R, t = pc._proper(R), pc._array(t, (3,), "translation")
    pc._unit(t)
    return dict(R=R.tolist(), t=t.tolist())


def _refinement_error_type(refiner):
    error = getattr(refiner, "RefinementError", None)
    if error is None:
        module = sys.modules.get(type(refiner).__module__)
        error = getattr(module, "RefinementError", None)
    if not isinstance(error, type) or not issubclass(error, Exception):
        raise TypeError("refiner must expose its exact RefinementError type")
    return error


def _physical_seed(seed, d1, d2):
    _pose(seed["R"], seed["t"])
    a, b, c = (float(seed[k]) for k in ("a", "b", "c"))
    if not all(math.isfinite(v) for v in (a, b, c)) or a <= 0.:
        raise pc.NumericalFailure("invalid seed calibration")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        z1, z2 = d1+b, a*d2+c
        shift2 = c/a
        native_z2 = a*(d2+shift2)
    if (not np.isfinite(z1).all() or not np.isfinite(z2).all()
            or not np.isfinite(native_z2).all() or not math.isfinite(shift2)
            or np.any(z1 <= 0.) or np.any(z2 <= 0.) or np.any(native_z2 <= 0.)):
        raise pc.NumericalFailure("nonphysical supported seed depth")


def process_pair(pair, native_backend, native_refiner, emit=None):
    """Return finite JSON evidence; unexpected API/ABI/program errors propagate.

    Required pair fields: xy1/xy2, K1/K2, width1/height1/width2/height2;
    depths1/depths2 (A,B,C,LS,IRLS,ANG,FULL); original_valid1/2;
    normal1/2, normal_valid1/2; correction1/2 (LS,IRLS,ANG);
    comparator_valid. Optional pair_index/scene are metadata only.
    """
    begin = time.perf_counter()
    record = dict(schema="evr0028.pair.v1", status="started", phase="input_validation",
                  pair_index=pair.get("pair_index"), scene=pair.get("scene"),
                  arms={}, native_E=None, original_inlier_mask=None,
                  support_indices=None, support_count=None, similarity_attempts=[],
                  normal_attempt=None, normal_refinement=None,
                  affine=dict(seeds=[], attempts=[], candidates=[], selector=None),
                  comparator_valid=pair["comparator_valid"],
                  calls=dict(estimate_invocation_intents=0, estimate_returned=0,
                             refine_invocation_intents=0, refine_returned=0),
                  timing_seconds=dict(input_validation=0., support=0., selectors=0.))
    active = None

    def publish():
        record["elapsed_seconds"] = time.perf_counter()-begin
        snapshot = _serial(record)
        # Detached snapshot: later mutation never changes a persisted checkpoint.
        if emit is not None:
            try:
                emit(json.loads(json.dumps(snapshot, allow_nan=False)))
            except BaseException as exc:
                raise ProgressError("checkpoint callback failed") from exc
        return snapshot

    def fallback(name, reason, detail):
        E = record["arms"]["E"]
        record["arms"][name] = dict(
            status="fallback" if E["status"] == "completed" else "unavailable",
            R=E["R"], t=E["t"], detail=dict(reason=reason, attempt=detail))

    def score(candidate):
        stamp = time.perf_counter()
        value = pc.pixel_msac(candidate["R"], candidate["t"], data["xy1"], data["xy2"],
                              data["K1"], data["K2"])
        record["timing_seconds"]["selectors"] += time.perf_counter()-stamp
        return value

    def select(candidates):
        stamp = time.perf_counter()
        value = pc.select_by_pixel_msac(candidates, data["xy1"], data["xy2"],
                                        data["K1"], data["K2"])
        record["timing_seconds"]["selectors"] += time.perf_counter()-stamp
        return value

    try:
        publish()
        stamp = time.perf_counter()
        data = _inputs(pair)
        error_type = _refinement_error_type(native_refiner)
        record["timing_seconds"]["input_validation"] = time.perf_counter()-stamp
        record["selected_count"] = data["n"]
        record["cameras"] = dict(camera1=data["camera1"], camera2=data["camera2"])
        record["native_E"] = dict(status="started", call_state="invocation_intent",
                                  options=copy.deepcopy(E_OPTIONS), call_seconds=None)
        active = record["native_E"]
        record["phase"] = "native_E"
        record["calls"]["estimate_invocation_intents"] += 1
        publish()
        stamp = time.perf_counter()
        try:
            pose, info = native_backend.estimate_relative_pose(
                data["xy1"].copy(), data["xy2"].copy(), copy.deepcopy(data["camera1"]),
                copy.deepcopy(data["camera2"]), copy.deepcopy(E_OPTIONS), initial_pose=None)
        finally:
            active["call_seconds"] = time.perf_counter()-stamp
        record["calls"]["estimate_returned"] += 1
        active["call_state"] = "returned"
        if not isinstance(info, dict):
            raise ValueError("native E info must be a dict")
        mask = np.asarray(info["inliers"])
        if mask.shape != (data["n"],) or mask.dtype.kind not in "biu" or not np.isin(mask, [0, 1]).all():
            raise ValueError("native E returned malformed inlier mask")
        mask = mask.astype(bool)
        active["stats"] = _serial({k: v for k, v in info.items() if k != "inliers"})
        active["raw_pose"] = _observed(dict(R=pose.R, t=pose.t))
        record["original_inlier_mask"] = mask.tolist()
        try:
            Epose = _pose(pose.R, pose.t)
        except (pc.NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
            active.update(status="unavailable", reason=str(exc))
            record["arms"]["E"] = dict(status="unavailable", R=None, t=None,
                                      detail=dict(reason=str(exc)))
        else:
            Escore = score(Epose)
            active.update(status="completed", score=Escore)
            record["arms"]["E"] = dict(status="completed", **Epose,
                                      detail=dict(score=Escore, unmodified_native_pose=True))
        active = None
        record["phase"] = "fixed_support"
        stamp = time.perf_counter()
        support = (mask & data["original_valid1"] & data["original_valid2"]
                   & data["normal_valid1"] & data["normal_valid2"])
        ids = np.flatnonzero(support)
        record["support_indices"], record["support_count"] = ids.tolist(), len(ids)
        record["support_rule"] = "native_E_inliers_AND_original_ABC_and_normals_once"
        record["timing_seconds"]["support"] = time.perf_counter()-stamp
        publish()
        if record["arms"]["E"]["status"] != "completed" or len(ids) < 4:
            reason = "E_unavailable" if record["arms"]["E"]["status"] != "completed" else "support_below_four"
            for name in ARMS[1:]:
                fallback(name, reason, None)
            record.update(status="completed", phase="complete")
            return publish()

        r1, r2 = data["r1"][ids], data["r2"][ids]
        similarity = {}
        for name in SIMILARITY_ARMS:
            record["phase"] = "similarity_"+name
            skip = None
            if name in CORRECTIONS:
                cs = [data["correction"+s][name] for s in ("1", "2")]
                if any(c["status"] != "completed" or not c["admissible"] for c in cs):
                    skip = "correction_unavailable_or_uninformative"
            if name == "FULL" and not record["comparator_valid"]:
                skip = "full_map_comparator_invalid"
            if skip is not None:
                result = dict(status="unavailable", reason=skip, seconds=0.)
            else:
                result = pc.solve_scale_only(r1, r2, data["depths1"][name][ids],
                                              data["depths2"][name][ids])
            row = dict(arm=name, result=result)
            record["similarity_attempts"].append(row)
            if result["status"] != "completed":
                if name == "FULL":
                    record["comparator_valid"] = False
                fallback(name, "similarity_unavailable", result)
            else:
                similarity[name] = result
                scoring = score(result)
                record["arms"][name] = dict(status="completed", R=result["R"], t=result["t"],
                                            detail=dict(fit=result, score=scoring))
            publish()

        record["phase"] = "normal_pose"
        normals = (data["normal1"][ids], data["normal2"][ids])
        normal = pc.solve_normal_pose(r1, r2, *normals)
        record["normal_attempt"] = normal
        if normal["status"] == "completed":
            record["arms"]["NORMAL"] = dict(status="completed", R=normal["R"], t=normal["t"],
                                            detail=dict(fit=normal, score=score(normal)))
        else:
            fallback("NORMAL", "normal_pose_unavailable", normal)
        publish()

        record["phase"] = "normal_refinement"
        starts = []
        if normal["status"] == "completed":
            starts.append(dict(label="NORMAL", R=normal["R"], t=normal["t"]))
        starts.append(dict(label="E", R=Epose["R"], t=Epose["t"]))
        refined = pc.refine_joint(r1, r2, *normals, data["K1"], data["K2"], starts)
        bank = []
        for row in refined["attempts"]:
            if row["status"] == "completed":
                for candidate in row["candidates"]:
                    bank.append(dict(**candidate, start_label=row["label"]))
        selector = select(bank)
        record["normal_refinement"] = dict(fit=refined, candidates=bank, selector=selector)
        if selector["selected_index"] is None:
            fallback("NORMAL_REFINE", "all_normal_refinements_unavailable", refined)
        else:
            chosen = bank[selector["selected_index"]]
            record["arms"]["NORMAL_REFINE"] = dict(status="completed", R=chosen["R"], t=chosen["t"],
                                                    detail=dict(selected_index=selector["selected_index"]))
        publish()

        record["phase"] = "affine_seeds"
        seeds = [dict(label="E", R=Epose["R"], t=pc._unit(Epose["t"]).tolist(),
                      a=1., b=0., c=0.)]
        for label in AFFINE_SEED_ORDER[1:]:
            if label not in similarity:
                continue
            source = similarity[label]
            b1 = 0. if label == "C" else float(data["correction1"][label]["b"])
            b2 = 0. if label == "C" else float(data["correction2"][label]["b"])
            c = float(source["a"])*b2
            seeds.append(dict(label=label, R=source["R"], t=source["t"],
                              a=source["a"], b=b1, c=c if math.isfinite(c) else None,
                              parameters_available=math.isfinite(c),
                              parameter_failure=None if math.isfinite(c) else "derived_c_overflow"))
        record["affine"]["seeds"] = seeds
        bank = [dict(status="completed", label="E_incumbent", **Epose)]
        d1, d2 = data["depths1"]["C"][ids], data["depths2"]["C"][ids]
        focal1 = float((data["K1"][0, 0]+data["K1"][1, 1])/2.)
        focal2 = float((data["K2"][0, 0]+data["K2"][1, 1])/2.)
        for seed in seeds:
            row = dict(label=seed["label"], status="started", call_state="not_invoked",
                       seed=seed, call_seconds=0., native_result=None)
            record["affine"]["attempts"].append(row)
            if not seed.get("parameters_available", True):
                row.update(status="unavailable", reason=seed["parameter_failure"])
                publish()
                continue
            try:
                _physical_seed(seed, d1, d2)
            except (pc.NumericalFailure, FloatingPointError, np.linalg.LinAlgError) as exc:
                row.update(status="unavailable", reason="seed_validation: "+str(exc))
                publish()
                continue
            active = row
            record["phase"] = "affine_native_"+seed["label"]
            row["call_state"] = "invocation_intent"
            record["calls"]["refine_invocation_intents"] += 1
            publish()
            stamp = time.perf_counter()
            try:
                result = native_refiner.refine(
                    r1[:, :2].copy(), r2[:, :2].copy(), d1.copy(), d2.copy(),
                    np.asarray(seed["R"]), np.asarray(seed["t"]), seed["a"], seed["b"], seed["c"],
                    focal1, focal2, max_iterations=100)
            except error_type as exc:
                row["call_seconds"] = time.perf_counter()-stamp
                row["call_state"] = "raised"
                row["error"] = exc.as_record()
                if exc.status not in (1, 3):
                    raise
                row.update(status="unavailable", reason="native_numerical_failure")
            else:
                row["call_seconds"] = time.perf_counter()-stamp
                record["calls"]["refine_returned"] += 1
                row["call_state"] = "returned"
                row["native_result"] = _serial(result)
                if not result["output_checks"]["physical_admissible"]:
                    row.update(status="unavailable", reason="native_output_not_physical")
                else:
                    geometry = result["geometry"]
                    try:
                        _physical_seed(geometry, d1, d2)
                    except (pc.NumericalFailure, FloatingPointError, np.linalg.LinAlgError) as exc:
                        row.update(status="unavailable", reason="native_output_validation: "+str(exc))
                    else:
                        value = _pose(geometry["R"], geometry["t"])
                        row.update(status="completed", reason=None)
                        bank.append(dict(status="completed", label=seed["label"], **value))
            finally:
                if row["call_state"] == "invocation_intent":
                    row["call_seconds"] = time.perf_counter()-stamp
            active = None
            publish()
        record["affine"]["candidates"] = bank
        record["phase"] = "affine_selector"
        selector = select(bank)
        record["affine"]["selector"] = selector
        if selector["selected_index"] is None:
            fallback("AFFINE", "affine_selector_unavailable", selector)
        elif selector["selected_index"] == 0:
            fallback("AFFINE", "E_incumbent_selected", selector)
        else:
            chosen = bank[selector["selected_index"]]
            record["arms"]["AFFINE"] = dict(status="completed", R=chosen["R"], t=chosen["t"],
                                           detail=dict(selected_index=selector["selected_index"],
                                                       label=chosen["label"]))
        record.update(status="completed", phase="complete")
        return publish()
    except BaseException as exc:
        record.update(status="technical_failure", failed_phase=record["phase"],
                      error=dict(type=type(exc).__name__, message=str(exc)))
        if active is not None and active["status"] == "started":
            active.update(status="technical_failure", call_state="exception")
        if not isinstance(exc, ProgressError):
            publish()
        raise
