"""Independent stdlib saved-scalar review; never imports the producer."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

D = Path(__file__).resolve().parents[1]
R = D.parent / "embodied-vision-reliability"
COHORT = [("4691_cut_guava", 86), ("817_cut_onion", 90), ("9675_fold_clothes", 201), ("1934_tear_film", 69)]
ARMS = ["B", "B+M1", "B+M2", "B+M1+M2"]
METHODS = ARMS + ["O_B", "O"]
THREADS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
PIN = "7fe9592298d1cced864eaa6f75faf973e98c10ea3ce8c91100a678cc6cd60fe3"
checks, errors, inputs, results, maximum = 0, [], {}, {}, 0.0


def check(ok, message):
    global checks
    checks += 1
    if not ok:
        errors.append(message)


def near(a, b, label):
    global maximum
    difference = abs(a - float(b))
    maximum = max(maximum, difference)
    check(math.isfinite(difference) and difference <= 1e-12, label)


def raw(path):
    p = Path(path).resolve()
    if p.suffix not in (".json", ".py", ".md", ".log") or p.stat().st_size > 32 * 1024**2:
        raise ValueError("bounded text only")
    data = p.read_bytes()
    data.decode("utf-8")
    inputs[str(p)] = hashlib.sha256(data).hexdigest()
    return data


def load(path):
    return json.loads(raw(path), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def average(values):
    values = list(values)
    return sum((Fraction(v) for v in values), Fraction()) / len(values)


began = time.monotonic()
try:
    check(all(os.environ.get(k) == "4" for k in THREADS), "four-thread envelope")
    spec, operation = load(D / "calculation-spec-01.json"), load(D / "calculation-outer-spec-01.json")
    receipt, report = load(D / "calculation-01/receipt.json"), load(D / "calculation-01/report.json")
    outer = load(D / "calculation-supervision-01/receipt.json")
    original = load(spec["input"]["path"])
    check(sha(spec["input"]["path"]) == spec["input"]["sha256"] == PIN, "exact original D35 input")
    check(receipt["status"] == "completed" and receipt["report_sha256"] == sha(D / "calculation-01/report.json")
          and receipt["spec_sha256"] == sha(D / "calculation-spec-01.json")
          and receipt["input"] == spec["input"] and receipt["input_source_verified_before_after"] is True,
          "complete inner/spec/report/input chain")
    check(all(receipt[k] == 0 for k in ("media_reads", "model_calls", "metric_pixel_recomputations")), "scalar-only receipt")
    source = R / "experiments/0036-candidate-verification/saved_selection.py"
    check(sha(source) == spec["source_sha256"] == receipt["source_sha256"], "current producer source")
    frozen = subprocess.check_output(["git", "show", operation["commit"] + ":" + source.relative_to(R).as_posix()],
                                     cwd=R, timeout=10)
    check(hashlib.sha256(frozen).hexdigest() == spec["source_sha256"], "source matches fixed Git commit")
    check(outer["status"] == "completed" and outer["exit_code"] == 0 and outer["termination"] is None
          and not outer["remaining_descendant_pids"] and not outer["cleanup_errors"]
          and not outer["post_exit_snapshot"]["rows"] and not outer["cleanup_snapshots"][-1]["rows"],
          "clean original outer")
    check(outer["output_sha256"] == sha(D / "calculation-01/receipt.json")
          and outer["spec_sha256"] == sha(D / "calculation-outer-spec-01.json")
          and outer["expected_commit"] == operation["commit"], "outer source/spec/output bindings")
    check(outer["thread_environment"] == {k: "4" for k in THREADS}
          and 0 <= outer["wall_seconds"] < outer["hard_timeout_seconds"] == 60
          and 0 <= outer["peak_group_rss_kib"] <= outer["group_rss_limit_kib"] == 8388608, "original resource bounds")
    check(outer["supervisor_sha256"] == sha(R / "experiments/0029-memory-information/supervise_exit_checked.py")
          and outer["monitor_sha256"] == sha(R / "experiments/0025c-rs-noise/supervise.py"), "supervisor sources")
    command = [v.replace(str(sys.executable), "$PYTHON").replace(str(R), "$REPOSITORY")
               .replace(str(R.parent), "$PRIVATE") for v in operation["command"]]
    check(command == outer["command"], "exact original command")
    for name, digest in outer["log_sha256"].items():
        check(sha(D / "calculation-supervision-01" / name) == digest, "original log " + name)
    check(report["tie_order"] == ARMS, "fixed tie order")
    for policy in ("primary", "void_ignore_sensitivity"):
        saved, sequences = report[policy], original[policy]["per_sequence"]
        check([(s["sequence"], s["frame_count"]) for s in sequences] == COHORT, policy + " cohort")
        check(len(saved["raw_sequences"]) == 4, policy + " all clips")
        fixed, oracle, constants, tie_count = [], [], {"J": [], "Jtr": []}, 0
        for s, out, (name, n) in zip(sequences, saved["raw_sequences"], COHORT):
            positions = {"J": list(range(1, n - 1)), "Jtr": list(range(1 + 3 * (n - 2) // 4, n - 1))}
            values = {arm: s["methods"][arm]["per_object"][0]["frame_J"] for arm in METHODS}
            check(s["object_ids"] == [1] and out["sequence"] == name and out["object_id"] == 1
                  and out["positions"] == positions and out["frame_J"] == values, policy + name + " raw preservation")
            for arm, row in values.items():
                check(len(row) == n and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                      for v in row), policy + name + arm + " finite all frames")
            winners, best = [], []
            for i in range(n):
                high = max(values[arm][i] for arm in ARMS)
                ties = [arm for arm in ARMS if values[arm][i] == high]
                winners.append(ties[0]); best.append(high); tie_count += len(ties) > 1
                check(out["best_arm_by_frame"][i] == ties[0] and out["best_J_by_frame"][i] == high
                      and high <= values["O"][i], policy + name + f" frame {i}: maximum/tie/ceiling")
            fm = {arm: {m: average(values[arm][i] for i in use) for m, use in positions.items()} for arm in METHODS}
            om = {m: average(best[i] for i in use) for m, use in positions.items()}
            fixed.append(fm); oracle.append(om)
            for metric in positions:
                chosen = max(ARMS, key=lambda arm: fm[arm][metric])
                constants[metric].append(fm[chosen][metric])
                check(out["constant_arm_by_metric"][metric] == chosen, policy + name + metric + " constant choice")
                near(out["per_frame_oracle"][metric], om[metric], policy + name + metric + " oracle mean")
                for arm in METHODS:
                    near(out["fixed_scores"][arm][metric], fm[arm][metric], policy + name + arm + metric)
        pooled = {arm: {m: average(s[arm][m] for s in fixed) for m in ("J", "Jtr")} for arm in METHODS}
        means = {m: average(s[m] for s in oracle) for m in ("J", "Jtr")}
        for arm in METHODS:
            for m in ("J", "Jtr"):
                near(saved["fixed_methods"][arm][m], pooled[arm][m], policy + arm + m + " fixed pooled")
        gains, clauses = {}, {}
        for m in means:
            choice = max(ARMS, key=lambda arm: pooled[arm][m])
            gains[m] = means[m] - pooled[choice][m]
            check(saved["strongest_global_fixed"][m]["arm"] == choice, policy + m + " global choice")
            near(saved["strongest_global_fixed"][m]["value"], pooled[choice][m], policy + m + " global value")
            near(saved["per_frame_oracle"]["values"][m], means[m], policy + m + " equal-clip oracle")
            near(saved["per_frame_oracle"]["gain_over_strongest_global_fixed"][m], gains[m], policy + m + " gain")
            near(saved["per_frame_oracle"]["gap_to_pixel_O"][m], pooled["O"][m] - means[m], policy + m + " remaining gap")
            near(saved["per_clip_constant_oracle"][m]["value"], average(constants[m]), policy + m + " constant bound")
            clauses[m + "_gain_at_least_3pp"] = gains[m] >= Fraction(3, 100)
        check(saved["necessary_clauses"] == clauses and saved["whole_arm_support_sufficient"] == all(clauses.values()),
              policy + " exact two clauses")
        check(saved["counts"] == dict(frames=446, interior=438, tail=110, objects=4)
              and sum(n - 2 for _, n in COHORT) == 438
              and sum((n - 2) - 3 * (n - 2) // 4 for _, n in COHORT) == 110, policy + " 446/438/110")
        results[policy] = dict(oracle={m: float(v) for m, v in means.items()}, gains={m: float(v) for m, v in gains.items()},
                               clauses=clauses, passed=all(clauses.values()), tied_frames=tie_count)
    check(report["primary_whole_arm_support_sufficient"] == receipt["primary_whole_arm_support_sufficient"]
          == results["primary"]["passed"], "primary decision chain")
    for path, digest in list(inputs.items()):
        check(sha(path) == digest, "text immutable during review: " + path)
except BaseException as exc:
    errors.append(type(exc).__name__ + ": " + str(exc))
    traceback.print_exc()
result = dict(schema="evr0036.independent-scalar-review.v1", passed=not errors, checks=checks, errors=errors,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_text_sha256=inputs,
              max_absolute_difference=maximum, comparison_tolerance=1e-12, recomputed=results,
              wall_seconds=time.monotonic() - began,
              scope="Exact Fraction arithmetic on saved scalar scores; no original producer import/rerun or media/array/model/network access.",
              conclusion="Whole-arm action-class sufficiency fails; the earlier D35 support opportunity remains unchanged.")
with Path(__file__).with_name("review.json").open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
print(json.dumps({k: result[k] for k in ("passed", "checks", "errors", "max_absolute_difference", "wall_seconds")}))
raise SystemExit(0 if not errors else 1)
