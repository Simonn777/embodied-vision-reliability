"""Independent saved-output arithmetic; no estimator, generator or evaluator imports.

Score comparisons are absolute 1e-10, angles 1e-6 degrees, AUC 1e-5 percentage
points. Exact tie ordering is reconstructed from saved scores after every
eligible score is independently checked; sub-tolerance arithmetic is not used
to manufacture a different exact tie. No root generation/completeness claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("T", "S", "V", "D8")
ROWS = 138
SCORE_TOL, ANGLE_TOL, AUC_TOL = 1e-10, 1e-6, 1e-5
NATIVE_SHA = "3228d488ca11f966378b6b7657ee6ebc5716da474c7ae88fccff7ab433a6981d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode(data):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def bad(value):
        raise ValueError("nonfinite JSON token " + value)
    def real(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON float")
        return result
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad, parse_float=real)


def rows(data):
    lines = data.splitlines()
    require(len(lines) == ROWS and all(x.strip() for x in lines), "incomplete row coverage")
    out = [decode(line) for line in lines]
    require(all(type(r.get("row_id")) is int and r["row_id"] == i
                for i, r in enumerate(out)), "row order mismatch")
    return out


def decode_truth(data):
    return rows(data)


def array(value, shape):
    require(not np.iscomplexobj(value), "complex numerical input")
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), "nonfinite or malformed array")
    return a


def rotation(value):
    R = array(value, (3, 3))
    gram = np.einsum("ki,kj->ij", R, R, optimize=False)
    require(np.max(np.abs(gram-np.eye(3))) <= 1e-8
            and abs(float(np.linalg.det(R))-1.) <= 1e-8, "improper saved rotation")
    return R


def unit(value):
    x = array(value, (3,))
    scale = float(np.max(np.abs(x)))
    require(scale > 0., "undefined translation direction")
    x = x/scale
    return x/math.sqrt(math.fsum(float(v)*float(v) for v in x))


def pose_errors(candidate, truth):
    R, G = rotation(candidate["R"]), rotation(truth["R"])
    Q = np.einsum("ik,jk->ij", R, G, optimize=False)
    skew = np.array([Q[2, 1]-Q[1, 2], Q[0, 2]-Q[2, 0], Q[1, 0]-Q[0, 1]])/2.
    er = math.degrees(math.atan2(math.hypot(*skew), (float(np.trace(Q))-1.)/2.))
    u, v = unit(candidate["t"]), unit(truth["t"])
    et = math.degrees(math.atan2(math.hypot(*np.cross(u, v)), math.fsum(u*v)))
    return dict(rotation_error=er, translation_error=et, task_error=max(er, et))


def pixel_score(candidate, observation):
    """Separate pointwise bidirectional projection, including full invalid mass."""
    K = array(observation["K"], (3, 3))
    require(np.array_equal(K, [[500., 0., 319.5], [0., 500., 239.5], [0., 0., 1.]]),
            "calibration mismatch")
    p, q = array(observation["xy1"], (9, 2)), array(observation["xy2"], (9, 2))
    d1, d2 = array(observation["d1"], (9,)), array(observation["d2"], (9,))
    b, a, c = array([candidate["b"], candidate["a"], candidate["c"]], (3,))
    R, t = rotation(candidate["R"]), array(candidate["t"], (3,))
    unit(t)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z1, z2 = d1+b, a*d2+c
    require(a > 0. and np.isfinite(z1).all() and np.isfinite(z2).all()
            and np.all(z1 > 0.) and np.all(z2 > 0.), "all-nine positivity failed")
    ray1 = np.column_stack(((p-K[:2, 2])/np.diag(K)[:2], np.ones(9)))
    ray2 = np.column_stack(((q-K[:2, 2])/np.diag(K)[:2], np.ones(9)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        X, Y = z1[:, None]*ray1, z2[:, None]*ray2
        forward = np.einsum("ij,nj->ni", R, X, optimize=False)+t
        backward = np.einsum("ji,nj->ni", R, Y-t, optimize=False)
    terms = []
    for points, target in ((forward, q), (backward, p)):
        for point, xy in zip(points, target):
            cost = 1.
            if np.isfinite(point).all() and point[2] > 0.:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    projection = point[:2]/point[2]*np.diag(K)[:2]+K[:2, 2]
                    delta = projection-xy
                if np.isfinite(delta).all() and np.max(np.abs(delta)) < 2.:
                    cost = min(math.fsum(float(v)*float(v) for v in delta)/4., 1.)
            terms.append(cost)
    return math.fsum(terms)/18., terms


def stats(errors):
    a = np.asarray(errors, dtype=float)
    require(len(a) > 0 and np.isfinite(a).all() and np.all((a >= 0.) & (a <= 180.)),
            "invalid full error denominator")
    return dict(count=len(a), useful_count=int(np.count_nonzero(a <= 5.)),
                useful_fraction=float(np.mean(a <= 5.)),
                step_cdf_auc5=float(100.*np.mean(np.maximum(0., 1.-a/5.))),
                mean_error=float(np.mean(a)))


def aggregate(records):
    """Independent group counts and Boolean clauses, not evaluator code reuse."""
    noisy = [r for r in records if r["provenance"]["regime"] != "clean"]
    def metric(rs):
        return {a: stats([r["arms"][a]["task_error"] for r in rs]) for a in ARMS}
    groups = {}
    for case in ("G", "M"):
        for calibration in ("scale_only", "shifted"):
            for regime in ("L", "H"):
                subset = [r for r in noisy if (r["provenance"]["case"],
                    r["provenance"]["calibration"], r["provenance"]["regime"])
                    == (case, calibration, regime)]
                require(len(subset) == 16, "noise group incomplete")
                groups[f"{case}/{calibration}/{regime}"] = metric(subset)
    exact = [r for r in records if r["provenance"]["regime"] == "clean"
             and r["provenance"]["case"] in ("G", "M")
             and r["provenance"]["derivative_mode"] == "exact"]
    clean = len(exact) == 4 and max(r["arms"]["T"]["task_error"] for r in exact) <= 1e-4
    absolute, noninfer, offset, point_ok, point_gain = [], [], [], [], []
    for case in ("G", "M"):
        for regime, threshold in (("L", 12), ("H", 8)):
            g = groups[f"{case}/shifted/{regime}"]
            absolute.append(g["T"]["useful_count"] >= threshold)
            z = groups[f"{case}/scale_only/{regime}"]
            noninfer.append(z["T"]["useful_count"] >= z["S"]["useful_count"]-2)
        h = groups[f"{case}/shifted/H"]
        n = h["T"]["useful_count"]
        offset.append(n >= h["S"]["useful_count"]+2)
        point_ok += [n >= h[a]["useful_count"]-2 for a in ("V", "D8")]
        point_gain.append(n >= max(h[a]["useful_count"] for a in ("V", "D8"))+2)
    clauses = dict(zip(("1_clean_exact", "2_shifted_yield",
                       "3_scale_only_and_offset", "4_point_controls"),
                      (clean, all(absolute), all(noninfer) and any(offset),
                       all(point_ok) and any(point_gain))))
    return dict(all_rows=metric(records), noise_rows=metric(noisy), groups=groups,
                clause_pass=clauses, gate_pass=all(clauses.values()))


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.max_difference = {}
    def equal(self, label, actual, expected):
        self.count += 1
        if actual != expected:
            self.failures.append(dict(check=label, actual=actual, expected=expected))
    def near(self, label, actual, expected, tolerance, category):
        self.count += 1
        require(type(actual) in (int, float) and type(expected) in (int, float)
                and math.isfinite(actual) and math.isfinite(expected), "invalid comparison number")
        difference = abs(actual-expected)
        self.max_difference[category] = max(self.max_difference.get(category, 0.), difference)
        if difference > tolerance:
            self.failures.append(dict(check=label, actual=actual, expected=expected,
                                      difference=difference, tolerance=tolerance))


def verify_receipt(path, seal_hash, commit, snapshots):
    path = Path(path)
    blob = path.read_bytes()
    snapshots[path] = digest(blob)
    r = decode(blob)
    require(r.get("schema") == "evr0026.operation.v1" and r.get("status") == "completed"
            and type(r.get("exit_code")) is int and r["exit_code"] == 0
            and r.get("termination") is None and r.get("remaining_descendant_pids") == []
            and r.get("cleanup_errors") == [], "prediction outer is not clean")
    require(r.get("expected_commit") == commit and r.get("output_sha256") == seal_hash,
            "outer commit/seal binding mismatch")
    for name in ("spec_sha256", "supervisor_sha256", "monitor_sha256"):
        require(isinstance(r.get(name), str) and re.fullmatch("[0-9a-f]{64}", r[name]),
                "missing outer helper/spec digest")
    require(0. < r["hard_timeout_seconds"] <= 30.
            and 0. <= r["wall_seconds"] <= r["hard_timeout_seconds"]
            and r["group_rss_limit_kib"] == 8388608
            and 0 <= r["peak_group_rss_kib"] <= 8388608, "outer resource bound violated")
    require(set(r["thread_environment"].values()) == {"4"}, "outer thread bound mismatch")
    command = r.get("command")
    require(isinstance(command, list) and "predict" in command
            and "--expected-commit" in command
            and command[command.index("--expected-commit")+1] == commit
            and any(str(v).endswith("/pipeline.py") or v == "pipeline.py" for v in command),
            "outer command is not the sealed predictor")
    require(set(r["log_sha256"]) == {"stdout.log", "stderr.log"}, "outer logs incomplete")
    for name, expected in r["log_sha256"].items():
        p = path.parent/name
        data = p.read_bytes()
        require(digest(data) == expected, "outer log mismatch")
        snapshots[p] = expected
    return r


def audit(paths, *, source_root=HERE):
    started = time.monotonic()
    checks, snapshots = Checks(), {}
    report = dict(schema="evr0027.independent-audit.v1", status="started", passed=False,
                  truth_decoded=False, tolerance=dict(score_absolute=SCORE_TOL,
                    angle_degrees=ANGLE_TOL, auc_percentage_points=AUC_TOL),
                  limitations=["No solver/fit/model rerun or root completeness verification.",
                    "Original pinned native compaction limitation remains unresolved.",
                    "Outer spec/helper digests retained, not independently rehashed; dispatcher owns their bytes.",
                    "No independent generation-outer verification; prediction is checked before truth decoding.",
                    "Independent implementation; NumPy numerical backend is shared."])
    try:
        require(len({Path(p).resolve() for p in paths.values()}) == len(paths),
                "audit input paths must differ")
        blobs = {}
        for name, p in paths.items():
            p = Path(p)
            require(p.is_file() and not p.is_symlink(), "unsafe/missing audit input")
            blobs[name] = p.read_bytes()
            snapshots[p] = digest(blobs[name])
        seal = decode(blobs["seal"])
        require(seal.get("schema") == "evr0027.seal.v1" and seal.get("status") == "completed"
                and seal.get("row_count") == ROWS, "prediction seal incomplete")
        commit, sources = seal["expected_commit"], seal["source_sha256"]
        require(re.fullmatch("[0-9a-f]{40}", commit), "invalid source commit")
        source_root = Path(source_root)
        names = {p.name for p in source_root.glob("*.py")} | {"protocol.md"}
        require(set(sources) == names and {"audit_saved.py", "evaluate.py", "pipeline.py",
            "fixtures.py", "study_core.py", "controls.py", "tangent_solver.py"} <= names,
            "incomplete current source map")
        for name, expected in sources.items():
            p = source_root/name
            require(p.name == name and p.is_file() and not p.is_symlink()
                    and digest(p.read_bytes()) == expected, "current frozen source mismatch")
            snapshots[p] = expected
        for name in ("inputs", "predictions", "truth"):
            require(seal[name+"_sha256"] == digest(blobs[name]), "sealed artifact mismatch: "+name)
        jobpath = Path(paths["seal"]).parent/"job.json"
        data = jobpath.read_bytes()
        require(digest(data) == seal["job_sha256"], "prediction job mismatch")
        snapshots[jobpath] = digest(data)
        job = decode(data)
        require(job.get("schema") == "evr0027.prediction-job.v1"
                and job["source_sha256"] == sources and job["expected_commit"] == commit
                and job["native_sha256"] == NATIVE_SHA, "job scientific bindings mismatch")
        for name in ("inputs", "truth"):
            require(job[name+"_sha256"] == seal[name+"_sha256"], "job input binding mismatch")
        verify_receipt(paths["prediction_receipt"], digest(blobs["seal"]), commit, snapshots)
        inputs, predictions = rows(blobs["inputs"]), rows(blobs["predictions"])
        candidates_checked = 0
        for i, (inp, pred) in enumerate(zip(inputs, predictions)):
            require(set(inp) == {"row_id", "observation"}
                    and set(pred) == {"row_id", "input_sha256", "result"}
                    and pred["input_sha256"] == digest(canonical(inp["observation"])),
                    "input/row binding mismatch")
            result, observation = pred["result"], inp["observation"]
            require(result.get("schema") == "evr0027.study-row.v1"
                    and result.get("status") == "completed" and set(result["arms"]) == set(ARMS),
                    "prediction/arm incomplete")
            require(set(observation) == {"K", "xy1", "xy2", "d1", "d2", "center_index",
                                       "derivatives"} and observation["center_index"] == 4,
                    "observation contract mismatch")
            attempted, executed = 0, 0
            for arm in ARMS:
                bank = result["arms"][arm]
                names = {"T": ["quartic"], "S": ["svd", "ordered_frame"],
                         "V": ["epsilon_"+str(e) for e in (.25, 1., 4., 16.)],
                         "D8": ["triangle_"+str(k) for k in range(8)]}[arm]
                require([r["name"] for r in bank["attempts"]] == names,
                        "method attempt coverage/order mismatch")
                for attempt in bank["attempts"]:
                    require(attempt["status"] in ("completed", "unavailable", "numerical_failure")
                            and type(attempt["native_called"]) is bool
                            and type(attempt["solver_called"]) is bool, "unfinished/malformed attempt")
                    if arm in ("V", "D8"):
                        attempted += 1
                        executed += int(attempt["native_called"])
                        require(attempt["native_called"] == attempt["solver_called"],
                                "native/solver invocation flags disagree")
                    else:
                        require(not attempt["native_called"], "false native call count")
                    if arm == "D8":
                        require(attempt["native_called"] and attempt["solver_called"],
                                "physical-point control was skipped")
                    elif observation["derivatives"]["status"] != "completed":
                        require(not attempt["solver_called"] and attempt["status"] == "unavailable",
                                "derivative-dependent method ran without derivatives")
                for field, component in (("solver_seconds", "elapsed_seconds"),
                                         ("scoring_seconds", "scoring_seconds")):
                    total = math.fsum(r[component] for r in bank["attempts"])
                    require(total >= 0. and math.isfinite(total), "invalid saved timing")
                    checks.near(f"{i}/{arm}/{field}", bank["timings"][field], total, 1e-9, "time")
                checks.near(f"{i}/{arm}/total_seconds", bank["timings"]["total_seconds"],
                    bank["timings"]["solver_seconds"]+bank["timings"]["scoring_seconds"], 1e-9, "time")
                eligible = []
                reconstructed = {}
                for k, candidate in enumerate(bank["candidates"]):
                    require(candidate["generation_index"] == k
                            and type(candidate["pose_usable"]) is bool, "candidate order/flag mismatch")
                    if candidate["pose_usable"]:
                        require(candidate["status"] == "completed", "eligible candidate not complete")
                        score, terms = pixel_score(candidate, observation)
                        checks.near(f"{i}/{arm}/{k}/score", candidate["score"], score, SCORE_TOL, "score")
                        reconstructed[k] = score
                        eligible.append(k)
                        candidates_checked += 1
                        for direction, part in (("forward", terms[:9]), ("backward", terms[9:])):
                            for j, value in enumerate(part):
                                checks.near(f"{i}/{arm}/{k}/{direction}/{j}",
                                    candidate[direction]["terms"][j], value, SCORE_TOL, "score")
                choice = min(eligible, key=lambda k: (bank["candidates"][k]["score"], k)) if eligible else None
                checks.equal(f"{i}/{arm}/selected", bank["selected_index"], choice)
                checks.equal(f"{i}/{arm}/selected_copy", bank["selected"],
                             bank["candidates"][choice] if choice is not None else None)
                if choice is not None:
                    require(reconstructed[choice] <= min(reconstructed.values())+2*SCORE_TOL,
                            "selected score not independently minimal within arithmetic tolerance")
            checks.equal(f"{i}/native_attempted", result["call_counts"]["native_attempted"], attempted)
            checks.equal(f"{i}/native_executed", result["call_counts"]["native_executed"], executed)
        report["eligible_candidates_checked"] = candidates_checked
        require(not checks.failures, "prediction score/selection audit failed before truth decoding")
        # All predictive bindings and numerical scores are checked before labels.
        truths = decode_truth(blobs["truth"])
        report["truth_decoded"] = True
        evaluation = decode(blobs["evaluation"])
        require(evaluation.get("schema") == "evr0027.evaluation.v1"
                and evaluation.get("status") == "completed"
                and evaluation.get("row_count") == ROWS, "evaluation incomplete")
        for name in ("inputs", "predictions", "truth", "seal"):
            require(evaluation["bindings"][name] == digest(blobs[name]), "evaluation binding mismatch")
        require(evaluation["bindings"]["job"] == seal["job_sha256"]
                and evaluation["bindings"]["source_sha256"] == sources
                and evaluation["bindings"]["expected_commit"] == commit, "evaluation source binding mismatch")
        require(len(evaluation["rows"]) == ROWS, "evaluation rows incomplete")
        metrics = []
        identities = set()
        for i, (pred, true, saved) in enumerate(zip(predictions, truths, evaluation["rows"])):
            require(saved["row_id"] == i and saved["provenance"] == true["provenance"],
                    "evaluation provenance mismatch")
            p = true["provenance"]
            identity = tuple(p[k] for k in ("case", "calibration", "derivative_mode", "regime", "realization"))
            require(identity not in identities, "duplicate cohort identity")
            identities.add(identity)
            arms = {}
            for arm in ARMS:
                bank = pred["result"]["arms"][arm]
                candidate_errors = {k: pose_errors(c, true["truth"])
                    for k, c in enumerate(bank["candidates"]) if c["pose_usable"]}
                errs = candidate_errors[bank["selected_index"]] if bank["selected"] is not None else {
                    "rotation_error": 180., "translation_error": 180., "task_error": 180.}
                for name, value in errs.items():
                    checks.near(f"{i}/{arm}/{name}", saved["arms"][arm][name], value, ANGLE_TOL, "angle")
                checks.equal(f"{i}/{arm}/selected_index", saved["arms"][arm]["selected_index"], bank["selected_index"])
                checks.equal(f"{i}/{arm}/unavailable", saved["arms"][arm]["unavailable"], bank["selected"] is None)
                checks.equal(f"{i}/{arm}/useful", saved["arms"][arm]["useful"], errs["task_error"] <= 5.)
                checks.equal(f"{i}/{arm}/candidate_count", saved["arms"][arm]["candidate_count"], len(bank["candidates"]))
                checks.equal(f"{i}/{arm}/eligible_count", saved["arms"][arm]["eligible_count"], len(candidate_errors))
                oracle = saved["arms"][arm]["oracle"]
                require(oracle["label"] == "oracle_upper_bound_never_used_for_gates", "oracle label mismatch")
                best = min(candidate_errors, key=lambda k: (candidate_errors[k]["task_error"], k)) if candidate_errors else None
                if best is None:
                    checks.equal(f"{i}/{arm}/oracle_index", oracle["candidate_index"], None)
                    checks.near(f"{i}/{arm}/oracle_error", oracle["task_error"], 180., ANGLE_TOL, "angle")
                else:
                    chosen = oracle["candidate_index"]
                    require(type(chosen) is int and chosen in candidate_errors, "oracle index not eligible")
                    checks.near(f"{i}/{arm}/oracle_error", oracle["task_error"],
                                candidate_errors[chosen]["task_error"], ANGLE_TOL, "angle")
                    # Equivalent clean solutions can exchange order at rounding
                    # scale. Preserve both indices; require numerical optimality
                    # under the predeclared angular tolerance, not false bitwise identity.
                    require(candidate_errors[chosen]["task_error"]
                            <= candidate_errors[best]["task_error"]+ANGLE_TOL,
                            "saved oracle is not independently optimal")
                arms[arm] = dict(errs, candidate_pose_errors=[
                    dict(candidate_index=k, **value) for k, value in candidate_errors.items()],
                    independent_oracle_index=best, saved_oracle_index=oracle["candidate_index"])
            metrics.append(dict(row_id=i, provenance=p, arms=arms))
        expected = {(c, a, m, "clean", None) for c in ("G", "M")
                    for a in ("scale_only", "shifted") for m in ("exact", "fitted")}
        expected |= {(c, a, "fitted", r, n) for c in ("G", "M")
                     for a in ("scale_only", "shifted") for r in ("L", "H") for n in range(16)}
        expected |= {(c, "shifted", "exact", "clean", None) for c in ("P", "F")}
        require(identities == expected, "fixed cohort identity mismatch")
        summaries = aggregate(metrics)
        def compare_stats(label, actual, saved):
            for field, value in actual.items():
                if field in ("count", "useful_count"):
                    checks.equal(label+"/"+field, saved[field], value)
                else:
                    tol = AUC_TOL if field == "step_cdf_auc5" else ANGLE_TOL if field == "mean_error" else 1e-12
                    checks.near(label+"/"+field, saved[field], value, tol,
                                "auc_pp" if field == "step_cdf_auc5" else "summary")
        for section in ("all_rows", "noise_rows"):
            for arm in ARMS:
                compare_stats(section+"/"+arm, summaries[section][arm], evaluation[section][arm])
        for key, values in summaries["groups"].items():
            for arm in ARMS:
                compare_stats(key+"/"+arm, values[arm], evaluation["groups"][key][arm])
        for name, passed in summaries["clause_pass"].items():
            checks.equal(name, evaluation["gates"]["clauses"][name]["pass"], passed)
        checks.equal("overall_gate", evaluation["gates"]["pass"], summaries["gate_pass"])
        report.update(row_count=ROWS, arm_rows=ROWS*len(ARMS), summaries=summaries,
                      recomputed_rows=metrics, scientific_gate_pass=summaries["gate_pass"])
        for path, expected_hash in snapshots.items():
            require(digest(path.read_bytes()) == expected_hash, "bound file changed during audit")
        report.update(status="completed", passed=not checks.failures,
                      bindings={name: digest(blob) for name, blob in blobs.items()},
                      expected_commit=commit, source_sha256=sources)
    except BaseException as error:
        report.update(status="failed", passed=False, failure=dict(type=type(error).__name__, message=str(error)))
    finally:
        report.update(check_count=checks.count, failures=checks.failures,
                      max_difference=checks.max_difference, wall_seconds=time.monotonic()-started)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("inputs", "predictions", "truth", "seal", "evaluation", "prediction-receipt", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    # Reserve once; this path is never overwritten, even after a failed audit.
    with args.output.open("x", encoding="utf-8") as stream:
        paths = {name: getattr(args, name) for name in (
            "inputs", "predictions", "truth", "seal", "evaluation", "prediction_receipt")}
        report = audit(paths)
        stream.write(json.dumps(report, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
