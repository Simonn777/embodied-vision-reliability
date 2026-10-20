"""Saved-only 0027 evaluation. No fixture, solver, native or model imports.

Truth bytes are hashed with the other artifacts, but truth JSON is not decoded
until the complete input/prediction/selection/source barrier has passed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time


ARMS = ("T", "S", "V", "D8")
ATTEMPTS = {"T": ("quartic",), "S": ("svd", "ordered_frame"),
            "V": ("epsilon_0.25", "epsilon_1.0", "epsilon_4.0", "epsilon_16.0"),
            "D8": tuple("triangle_"+str(i) for i in range(8))}
ROWS = 138
HERE = Path(__file__).resolve().parent
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class EvaluationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise EvaluationError("nonfinite JSON token: " + value)


def _parse_float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON number")
    return result


def decode_json(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=_reject_constant, parse_float=_parse_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError("invalid JSON") from error


def decode_jsonl(data):
    lines = data.splitlines()
    require(len(lines) == ROWS and all(line.strip() for line in lines),
            "JSONL must contain exactly 138 nonempty rows")
    rows = [decode_json(line) for line in lines]
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and type(row.get("row_id")) is int
                and row["row_id"] == index, "row_id/order mismatch")
    return rows


def decode_truth(data):
    """Explicit label-decode boundary, independently mocked in barrier tests."""
    return decode_jsonl(data)


def finite(value, name):
    require(type(value) in (int, float), name + " must be a real number")
    require(math.isfinite(value), name + " must be finite")
    return float(value)


def array(value, shape, name):
    if not shape:
        return finite(value, name)
    require(isinstance(value, list) and len(value) == shape[0],
            name + " shape mismatch")
    return [array(item, shape[1:], name) for item in value]


def proper_rotation(value):
    R = array(value, (3, 3), "R")
    orth = max(abs(sum(R[k][i]*R[k][j] for k in range(3))-(i == j))
               for i in range(3) for j in range(3))
    det = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])
           - R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
           + R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
    require(math.isfinite(orth) and math.isfinite(det)
            and orth <= 1e-8 and abs(det-1.) <= 1e-8,
            "rotation is not proper within 1e-8")
    return R


def unit(value):
    t = array(value, (3,), "t")
    scale = max(map(abs, t))
    require(scale > 0., "translation direction is undefined")
    scaled = [x/scale for x in t]
    length = math.sqrt(sum(x*x for x in scaled))
    return [x/length for x in scaled]


def pose_error(estimate, truth):
    """Independent atan2 metrics, with no sign folding or pose repair."""
    R, G = proper_rotation(estimate["R"]), proper_rotation(truth["R"])
    relative = [[sum(R[i][k]*G[j][k] for k in range(3))
                 for j in range(3)] for i in range(3)]
    skew = [(relative[2][1]-relative[1][2])/2,
            (relative[0][2]-relative[2][0])/2,
            (relative[1][0]-relative[0][1])/2]
    sine = math.hypot(*skew)
    cosine = (sum(relative[i][i] for i in range(3))-1.)/2
    rotation = math.degrees(math.atan2(sine, cosine))
    t, g = unit(estimate["t"]), unit(truth["t"])
    cross = [t[1]*g[2]-t[2]*g[1], t[2]*g[0]-t[0]*g[2],
             t[0]*g[1]-t[1]*g[0]]
    translation = math.degrees(math.atan2(
        math.hypot(*cross), sum(x*y for x, y in zip(t, g))))
    return {"rotation_error": rotation, "translation_error": translation,
            "task_error": max(rotation, translation)}


def validate_observation(observation):
    """Strict public observation contract; labels cannot hide in extra keys."""
    require(isinstance(observation, dict), "observation must be a dictionary")
    expected = {"K", "xy1", "xy2", "d1", "d2", "center_index", "derivatives"}
    require(set(observation) == expected, "observation keys mismatch")
    K = array(observation["K"], (3, 3), "K")
    require(K == [[500., 0., 319.5], [0., 500., 239.5], [0., 0., 1.]],
            "fixed calibration mismatch")
    for key in ("xy1", "xy2"):
        array(observation[key], (9, 2), key)
    for key in ("d1", "d2"):
        array(observation[key], (9,), key)
    require(type(observation["center_index"]) is int
            and observation["center_index"] == 4, "center index mismatch")
    derivatives = observation["derivatives"]
    required = {"status", "mode", "A", "g1", "g2", "fit", "elapsed_seconds", "reason"}
    require(isinstance(derivatives, dict) and required <= set(derivatives)
            and set(derivatives) <= required | {"failure"},
            "derivative observation keys mismatch")
    require(derivatives["status"] in ("completed", "unavailable", "numerical_failure"),
            "invalid derivative status")
    require(derivatives["mode"] in ("analytic", "center_fixed_ols"),
            "invalid derivative mode")
    require(derivatives["fit"] is None or isinstance(derivatives["fit"], dict),
            "invalid derivative fitting receipt")
    require(finite(derivatives["elapsed_seconds"], "derivative seconds") >= 0.,
            "negative derivative time")
    reason = derivatives["reason"]
    require(reason is None or isinstance(reason, str), "invalid derivative reason")
    if derivatives["status"] != "completed":
        require(isinstance(reason, str) and reason, "unavailable derivative lacks reason")
    for name, shape in (("A", (2, 2)), ("g1", (2,)), ("g2", (2,))):
        if derivatives[name] is not None or derivatives["status"] == "completed":
            array(derivatives[name], shape, name)
    if "failure" in derivatives:
        failure = derivatives["failure"]
        require(isinstance(failure, dict) and set(failure) == {"type", "message"}
                and all(isinstance(x, str) for x in failure.values()), "invalid failure receipt")
    def no_labels(value):
        if isinstance(value, dict):
            require(not ({"truth", "ground_truth", "gt"} & {k.lower() for k in value}),
                    "truth field is forbidden in observations")
            for item in value.values():
                no_labels(item)
        elif isinstance(value, list):
            for item in value:
                no_labels(item)
    no_labels(observation)


def validate_arm(arm, observation):
    require(isinstance(arm, dict), "arm must be a dictionary")
    require({"candidates", "selected_index", "selected"} <= set(arm),
            "missing arm fields")
    candidates = arm["candidates"]
    require(isinstance(candidates, list), "candidates must be a list")
    eligible = []
    for index, candidate in enumerate(candidates):
        require(isinstance(candidate, dict), "candidate must be a dictionary")
        require({"R", "t", "a", "b", "c", "pose_usable", "score",
                 "generation_index", "status", "reason"} <= set(candidate),
                "missing normalized candidate fields")
        generation = candidate["generation_index"]
        require(type(generation) is int and generation == index,
                "candidate generation index must equal list index")
        require(type(candidate["pose_usable"]) is bool, "invalid usable flag")
        require(isinstance(candidate["status"], str) and candidate["status"],
                "missing candidate status")
        require(candidate["reason"] is None or isinstance(candidate["reason"], str),
                "invalid candidate reason")
        if not candidate["pose_usable"]:
            require(candidate["score"] is None and isinstance(candidate["reason"], str)
                    and candidate["reason"], "unusable candidate needs null score and reason")
            continue
        proper_rotation(candidate["R"])
        unit(candidate["t"])
        a = finite(candidate["a"], "a")
        b = finite(candidate["b"], "b")
        c = finite(candidate["c"], "c")
        require(a > 0., "eligible candidate has nonpositive scale")
        corrected = [d+b for d in observation["d1"]]
        corrected += [a*d+c for d in observation["d2"]]
        require(all(math.isfinite(d) and d > 0. for d in corrected),
                "eligible candidate violates full-nine positive depths")
        score = finite(candidate["score"], "score")
        require(0. <= score <= 1., "eligible score outside capped range")
        eligible.append(index)
    expected = min(eligible, key=lambda i: (
        candidates[i]["score"], candidates[i]["generation_index"])) if eligible else None
    selected = arm["selected_index"]
    require(selected is None or type(selected) is int, "invalid selected_index")
    require(selected == expected, "selection differs from score/generation minimum")
    require(arm["selected"] == (candidates[expected] if expected is not None else None),
            "selected candidate copy mismatch")
    require(arm.get("status") == ("completed" if expected is not None else "unavailable"),
            "arm completion status contradicts selection")
    return eligible


def validate_completion(result, observation):
    require(result.get("active_attempt") is None, "active attempt remains")
    require(isinstance(result["warnings"], list), "warnings must be retained as a list")
    executed = 0
    for name in ARMS:
        bank = result["arms"][name]
        attempts = bank.get("attempts")
        require(isinstance(attempts, list) and len(attempts) == len(ATTEMPTS[name]),
                "incomplete attempt coverage")
        for attempt, expected_name in zip(attempts, ATTEMPTS[name]):
            require(isinstance(attempt, dict) and attempt.get("name") == expected_name,
                    "attempt order/name mismatch")
            require(attempt.get("status") in ("completed", "unavailable", "numerical_failure"),
                    "nonterminal or technical-failure attempt in completed prediction")
            require(type(attempt.get("solver_called")) is bool
                    and type(attempt.get("native_called")) is bool, "invalid call flags")
            require(not attempt["native_called"] or
                    (name in ("V", "D8") and attempt["solver_called"]),
                    "native flag contradicts arm/solver flag")
            if name == "D8":
                require(attempt["native_called"], "D8 raw-point attempt was skipped")
            if name in ("T", "S", "V") and observation["derivatives"]["status"] != "completed":
                require(not attempt["solver_called"], "unavailable derivative entered solver")
            for key in ("elapsed_seconds", "scoring_seconds"):
                require(finite(attempt.get(key), key) >= 0., "negative attempt time")
            executed += int(attempt["native_called"])
        timing = bank.get("timings")
        require(isinstance(timing, dict), "missing arm timings")
        times = {k: finite(timing.get(k), k) for k in
                 ("solver_seconds", "scoring_seconds", "total_seconds")}
        require(all(v >= 0. for v in times.values()), "negative arm time")
        require(math.isclose(times["total_seconds"], times["solver_seconds"]+times["scoring_seconds"],
                             rel_tol=1e-10, abs_tol=1e-12), "arm time sum mismatch")
    calls = result["call_counts"]
    require(isinstance(calls, dict) and set(calls) == {"native_attempted", "native_executed"}
            and type(calls["native_attempted"]) is int and calls["native_attempted"] == 12
            and type(calls["native_executed"]) is int and calls["native_executed"] == executed,
            "native call-count mismatch")
    timing = result["common_timing"]
    require(isinstance(timing, dict), "missing common timing")
    values = {k: finite(timing.get(k), k) for k in
              ("derivative_fit_seconds", "validation_seconds", "process_seconds",
               "total_with_derivatives_seconds")}
    require(all(v >= 0. for v in values.values()), "negative common time")
    require(values["derivative_fit_seconds"] == observation["derivatives"]["elapsed_seconds"]
            and values["validation_seconds"] <= values["process_seconds"]
            and math.isclose(values["total_with_derivatives_seconds"],
                values["process_seconds"]+values["derivative_fit_seconds"],
                rel_tol=1e-10, abs_tol=1e-12), "common timing mismatch")


def _source_barrier(seal, seal_path, source_root):
    require(isinstance(seal, dict)
            and seal.get("schema") == "evr0027.seal.v1"
            and seal.get("status") == "completed"
            and type(seal.get("row_count")) is int and seal["row_count"] == ROWS,
            "incomplete or invalid seal")
    commit = seal.get("expected_commit")
    require(isinstance(commit, str) and HEX40.fullmatch(commit),
            "invalid expected commit")
    hashes = seal.get("source_sha256")
    require(isinstance(hashes, dict) and "evaluate.py" in hashes,
            "missing evaluation source binding")
    require(set(hashes) == {p.name for p in source_root.glob("*.py")} | {"protocol.md"},
            "source map does not cover the full scientific directory")
    snapshots = {}
    for name, digest in hashes.items():
        require(isinstance(name, str) and Path(name).name == name
                and name not in ("", ".", "..") and "\\" not in name,
                "unsafe source filename")
        require(isinstance(digest, str) and HEX64.fullmatch(digest),
                "invalid source SHA256")
        path = source_root/name
        require(not path.is_symlink() and path.is_file(), "invalid source file")
        snapshots[path] = digest
        require(sha256(path.read_bytes()) == digest, "source hash mismatch: " + name)
    job_path = seal_path.parent/"job.json"
    digest = seal.get("job_sha256")
    require(isinstance(digest, str) and HEX64.fullmatch(digest), "missing job hash")
    job_bytes = job_path.read_bytes()
    require(sha256(job_bytes) == digest, "job hash mismatch")
    job = decode_json(job_bytes)
    require(isinstance(job, dict) and job.get("expected_commit") == commit
            and job.get("source_sha256") == hashes, "job/source/commit mismatch")
    snapshots[job_path] = digest
    return snapshots


def _provenance_key(p):
    require(isinstance(p, dict), "provenance must be a dictionary")
    names = ("case", "calibration", "derivative_mode", "regime", "realization")
    require(set(names) <= set(p), "incomplete provenance")
    require(p["case"] in ("G", "M", "P", "F")
            and p["calibration"] in ("scale_only", "shifted")
            and p["derivative_mode"] in ("exact", "fitted")
            and p["regime"] in ("clean", "L", "H"), "unknown provenance category")
    require(p["realization"] is None or type(p["realization"]) is int,
            "invalid realization")
    return tuple(p[key] for key in names)


def expected_provenance():
    expected = set()
    for case in ("G", "M"):
        for calibration in ("scale_only", "shifted"):
            for mode in ("exact", "fitted"):
                expected.add((case, calibration, mode, "clean", None))
            for regime in ("L", "H"):
                for realization in range(16):
                    expected.add((case, calibration, "fitted", regime, realization))
    for case in ("P", "F"):
        expected.add((case, "shifted", "exact", "clean", None))
    return expected


def summary(errors):
    require(errors and all(math.isfinite(x) and 0. <= x <= 180. for x in errors),
            "invalid metric error cohort")
    return {"count": len(errors), "useful_count": sum(x <= 5. for x in errors),
            "useful_fraction": sum(x <= 5. for x in errors)/len(errors),
            "step_cdf_auc5": 100.*math.fsum(max(0., 1.-x/5.) for x in errors)/len(errors),
            "mean_error": math.fsum(errors)/len(errors)}


def summarize_rows(rows):
    def metrics(subset):
        return {arm: summary([r["arms"][arm]["task_error"] for r in subset])
                for arm in ARMS}
    noisy = [r for r in rows if r["provenance"]["regime"] in ("L", "H")]
    groups = {}
    for case in ("G", "M"):
        for calibration in ("scale_only", "shifted"):
            for regime in ("L", "H"):
                subset = [r for r in noisy if (
                    r["provenance"]["case"], r["provenance"]["calibration"],
                    r["provenance"]["regime"]) == (case, calibration, regime)]
                require(len(subset) == 16, "noise group coverage mismatch")
                groups["/".join((case, calibration, regime))] = metrics(subset)
    exact = [r for r in rows if r["provenance"]["case"] in ("G", "M")
             and r["provenance"]["regime"] == "clean"
             and r["provenance"]["derivative_mode"] == "exact"]
    clause1 = {"rows": [{"row_id": r["row_id"], "error": r["arms"]["T"]["task_error"]}
                        for r in exact],
               "pass": len(exact) == 4 and all(
                   r["arms"]["T"]["task_error"] <= 1e-4 for r in exact)}
    absolute, scale_only, point_noninfer, offset_gain, point_gain = [], [], [], [], []
    for case in ("G", "M"):
        for regime in ("L", "H"):
            group = groups[case+"/shifted/"+regime]
            count = group["T"]["useful_count"]
            threshold = 12 if regime == "L" else 8
            absolute.append({"case": case, "regime": regime, "count": count,
                             "threshold": threshold, "pass": count >= threshold})
            zero = groups[case+"/scale_only/"+regime]
            delta = zero["T"]["useful_count"]-zero["S"]["useful_count"]
            scale_only.append({"case": case, "regime": regime, "delta": delta,
                               "pass": delta >= -2})
        group = groups[case+"/shifted/H"]
        t = group["T"]["useful_count"]
        offset_gain.append({"case": case, "delta": t-group["S"]["useful_count"],
                            "pass": t-group["S"]["useful_count"] >= 2})
        for arm in ("V", "D8"):
            delta = t-group[arm]["useful_count"]
            point_noninfer.append({"case": case, "control": arm, "delta": delta,
                                   "pass": delta >= -2})
        delta = t-max(group["V"]["useful_count"], group["D8"]["useful_count"])
        point_gain.append({"case": case, "delta": delta, "pass": delta >= 2})
    clauses = {
        "1_clean_exact": clause1,
        "2_shifted_yield": {"checks": absolute, "pass": all(x["pass"] for x in absolute)},
        "3_scale_only_and_offset": {"scale_only": scale_only, "offset_gain": offset_gain,
            "pass": all(x["pass"] for x in scale_only) and any(x["pass"] for x in offset_gain)},
        "4_point_controls": {"noninferiority": point_noninfer, "improvement": point_gain,
            "pass": all(x["pass"] for x in point_noninfer) and any(x["pass"] for x in point_gain)}}
    return {"all_rows": metrics(rows), "noise_rows": metrics(noisy), "groups": groups,
            "gates": {"clauses": clauses, "pass": all(x["pass"] for x in clauses.values())}}


def evaluate_saved(prediction_path, input_path, truth_path, seal_path, *, _source_root=None):
    source_root = HERE if _source_root is None else Path(_source_root).resolve()
    paths = {"predictions": Path(prediction_path), "inputs": Path(input_path),
             "truth": Path(truth_path), "seal": Path(seal_path)}
    require(len({p.resolve() for p in paths.values()}) == 4, "artifact paths must differ")
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: sha256(blob) for name, blob in blobs.items()}
    seal = decode_json(blobs["seal"])
    snapshots = _source_barrier(seal, paths["seal"], source_root)
    for name in ("predictions", "inputs", "truth"):
        require(seal.get(name+"_sha256") == hashes[name], name + " file hash mismatch")
    # No truth decoder has been reached. Validate the complete predictive record.
    inputs, predictions = decode_jsonl(blobs["inputs"]), decode_jsonl(blobs["predictions"])
    eligible = []
    for index, (inp, pred) in enumerate(zip(inputs, predictions)):
        require(set(inp) == {"row_id", "observation"}, "input record keys mismatch")
        require(set(pred) == {"row_id", "input_sha256", "result"},
                "prediction record keys mismatch")
        validate_observation(inp["observation"])
        require(pred["input_sha256"] == sha256(canonical(inp["observation"])),
                "canonical observation hash mismatch")
        result = pred["result"]
        require(isinstance(result, dict)
                and result.get("schema") == "evr0027.study-row.v1"
                and result.get("status") == "completed", "incomplete prediction")
        require({"arms", "common_timing", "call_counts", "warnings"} <= set(result),
                "prediction result fields missing")
        require(isinstance(result["arms"], dict) and set(result["arms"]) == set(ARMS),
                "arm coverage mismatch")
        eligible.append({arm: validate_arm(result["arms"][arm], inp["observation"])
                         for arm in ARMS})
        validate_completion(result, inp["observation"])
    truths = decode_truth(blobs["truth"])
    provenance = [_provenance_key(row.get("provenance")) for row in truths]
    require(len(set(provenance)) == ROWS and set(provenance) == expected_provenance(),
            "fixed provenance cohort mismatch")
    rows = []
    for index, (pred, true) in enumerate(zip(predictions, truths)):
        require(set(true) == {"row_id", "truth", "provenance"}, "truth record keys mismatch")
        truth_keys = {"R", "t", "a", "b", "c", "X", "Y", "axial_depth1",
                      "axial_depth2", "source_center_normalized", "exact_derivatives",
                      "calibration_family"}
        require(isinstance(true["truth"], dict) and {"R", "t"} <= set(true["truth"])
                and set(true["truth"]) <= truth_keys,
                "truth pose fields mismatch")
        proper_rotation(true["truth"]["R"])
        unit(true["truth"]["t"])
        require(inputs[index]["observation"]["derivatives"]["mode"] ==
                ("analytic" if true["provenance"]["derivative_mode"] == "exact"
                 else "center_fixed_ols"), "input/provenance derivative-mode mismatch")
        arms = {}
        for arm in ARMS:
            record = pred["result"]["arms"][arm]
            candidates = record["candidates"]
            scored = [(i, pose_error(candidates[i], true["truth"]))
                      for i in eligible[index][arm]]
            selected_index = record["selected_index"]
            selected_error = (pose_error(record["selected"], true["truth"])
                              if selected_index is not None else
                              dict(rotation_error=180., translation_error=180., task_error=180.))
            oracle = min(scored, key=lambda item: (
                item[1]["task_error"], candidates[item[0]]["generation_index"])) if scored else None
            arms[arm] = dict(selected_error, selected_index=selected_index,
                unavailable=selected_index is None, useful=selected_error["task_error"] <= 5.,
                selected_score=record["selected"]["score"] if selected_index is not None else None,
                candidate_count=len(candidates), eligible_count=len(scored),
                oracle={"label": "oracle_upper_bound_never_used_for_gates",
                        "candidate_index": oracle[0] if oracle else None,
                        "task_error": oracle[1]["task_error"] if oracle else 180.})
        rows.append({"row_id": index, "provenance": true["provenance"], "arms": arms})
    for name, path in paths.items():
        require(sha256(path.read_bytes()) == hashes[name], "artifact changed during evaluation")
    for path, digest in snapshots.items():
        require(sha256(path.read_bytes()) == digest, "source/job changed during evaluation")
    return {"schema": "evr0027.evaluation.v1", "status": "completed",
            "row_count": ROWS, "noise_row_count": 128,
            "bindings": dict(hashes, job=seal["job_sha256"],
                             expected_commit=seal["expected_commit"],
                             source_sha256=seal["source_sha256"]),
            "metric": {"task": "max(rotation,directed_translation),degrees",
                       "unavailable_error": 180., "useful_threshold_degrees": 5.,
                       "auc": "100*mean(max(0,1-error/5)); empirical step-CDF, not DMS trapezoid",
                       "selection": "in-sample minimum score, then generation_index",
                       "oracle_used_for_gates": False},
            "rows": rows, **summarize_rows(rows),
            "research_admission": False,
            "interpretation": (
                "Necessary manufactured sensitivity screen only. The pinned native V/D8 root-compaction "
                "defect limits the control: threshold passage does not admit efficacy or F-all advancement "
                "without a trustworthy corrected/independent native control. Full-nine F-all remains required.")}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("predictions", "inputs", "truth", "seal", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "evr0027.evaluation.v1", "status": "started"})+"\n")
        handle.flush()
    try:
        report = evaluate_saved(args.predictions, args.inputs, args.truth, args.seal)
        report["wall_seconds"] = time.monotonic()-started
        args.output.write_bytes(canonical(report)+b"\n")
        return 0
    except BaseException as error:
        failure = {"schema": "evr0027.evaluation.v1", "status": "failed",
                   "error": {"type": type(error).__name__, "message": str(error)},
                   "wall_seconds": time.monotonic()-started}
        args.output.write_bytes(canonical(failure)+b"\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
