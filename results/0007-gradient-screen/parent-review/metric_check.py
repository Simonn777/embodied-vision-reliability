#!/usr/bin/env python3
"""Read-only public-artifact arithmetic audit; no raw-gradient/bootstrap replay.

Usage: python3 this.py --results PATH --report NEW_JSON
Uses only the Python standard library, never production scientific code.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

METHODS = ["independent", "random", "mean", "samepatch"]
CORE_METHODS = ["independent", "random_antithetic", "mean_antithetic", "same_patch"]
REGIMES = ["A_augmentation", "B_total"]
CASES = ["d%d-colored-%d" % (d, s) for d in range(1, 5) for s in (0, 1)]
SCORE_PIN = "ff4542f059ae02cce36aeacfda2d5346075e524d7863e8666b5346756c8306a9"
CHECKER_PIN = "2e6ab7912ee1e56836da9cd61b059598d09125eb87d9693c1be8a79bd17fde6e"
SOURCE_COMMIT = "67bccdd57a2ec4a8204cf42a159c374caed0be2b"
FREEZE_COMMIT = "dd7c84c9afb774e2d411cf815aa17f953409d6e6"
RTOL, ATOL = 1e-10, 1e-12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    def __init__(self, root):
        self.root = root
        self.hashes = {}
        self.count = 0
        self.numbers = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.failure_count = 0
        self.failures = []

    def load(self, rel):
        path = self.root / rel
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing/symlink public evidence: " + rel)
        data = path.read_bytes()
        self.hashes[rel] = hashlib.sha256(data).hexdigest()
        def reject(value):
            raise ValueError("nonfinite JSON: " + value)
        return json.loads(data, parse_constant=reject)

    def require(self, condition, label):
        self.count += 1
        if not condition:
            self.failure_count += 1
            if len(self.failures) < 30:
                self.failures.append(label)

    def exact(self, got, expected, label):
        self.require(type(got) is type(expected) and got == expected, label)

    def close(self, got, expected, label):
        self.numbers += 1
        good = type(got) in (int, float) and math.isfinite(got) and math.isfinite(expected)
        if good:
            delta = abs(got - expected)
            self.max_abs = max(self.max_abs, delta)
            self.max_rel = max(self.max_rel, delta / max(abs(expected), ATOL))
            good = delta <= ATOL + RTOL * abs(expected)
        self.require(good, label)


def gm(values):
    return math.exp(math.fsum(math.log(x) for x in values) / len(values))


def run(root):
    a = Audit(root)
    score = a.load("study/scores.json")
    a.exact(a.hashes["study/scores.json"], SCORE_PIN, "pinned score hash")
    a.exact(score["status"], "analyzed", "score completed")
    a.exact(score["shape"], [8, 2, 4, 64, 5313], "shape")
    a.exact(score["methods"], METHODS, "method order")
    a.exact(score["regimes"], REGIMES, "regime order")
    a.exact(score["production_contract"], True, "production contract")
    a.exact(score["bindings"]["source_commit"], SOURCE_COMMIT, "source commit")
    a.exact(score["bindings"]["freeze_commit"], FREEZE_COMMIT, "freeze commit")

    summaries = {}
    for row in score["summaries"]:
        key = (row["case_index"], row["regime"], row["method"])
        a.require(key not in summaries, "unique summary row")
        summaries[key] = row
        a.require(math.isfinite(row["trace_covariance"]) and row["trace_covariance"] > 0,
                  "positive finite published variance")
        a.require(math.isfinite(row["mean_gradient_norm"]) and row["mean_gradient_norm"] >= 0,
                  "finite published mean norm")
    expected_summary_keys = {(c, r, m) for c in range(8) for r in REGIMES for m in METHODS}
    a.require(set(summaries) == expected_summary_keys and len(score["summaries"]) == 64, "complete 64-summary grid")

    times, total_records = {}, 0
    for c, name in enumerate(CASES):
        rel = "cases/" + name + "/receipt.json"
        receipt = a.load(rel)
        a.exact(score["bindings"]["cases"][c], {"case": name, "receipt_sha256": a.hashes[rel]}, "score-to-receipt binding " + name)
        for k, expected in (("case_index", c), ("status", "completed"), ("completed_measurements", 512),
                            ("expected_measurements", 512), ("optimizer_steps", 0), ("weight_unchanged", True),
                            ("methods", CORE_METHODS), ("regimes", ["A_fixed", "B_redraw"])):
            a.exact(receipt[k], expected, "receipt." + name + "." + k)
        a.exact(receipt["weight_before_sha256"], receipt["weight_after_sha256"], "weight hash preserved " + name)
        seen, partitions = set(), {(r, m): [] for r in range(2) for m in range(4)}
        a.exact(len(receipt["records"]), 512, "512 case records " + name)
        for rec in receipt["records"]:
            key = (rec["regime"], rec["method_index"], rec["replicate"])
            a.require(key not in seen, "unique measurement " + name)
            seen.add(key)
            a.require(key[0] in range(2) and key[1] in range(4) and key[2] in range(64), "measurement coordinate")
            a.exact(rec["method"], CORE_METHODS[key[1]], "measurement method mapping")
            a.exact(rec["status"], "completed", "measurement completed")
            a.exact(rec["forward_calls"], 1, "one forward per measurement")
            a.exact(rec["backward_calls"], 1, "one backward per measurement")
            a.exact(rec["batch_occurrences"], 512, "512 occurrences")
            t = rec["wall_seconds"]
            a.require(type(t) in (int, float) and math.isfinite(t) and t > 0, "positive step timing")
            partitions[(key[0], key[1])].append(t)
        a.require(seen == set(itertools.product(range(2), range(4), range(64))), "complete measurement coordinates")
        total_records += len(seen)
        for (r, m), values in partitions.items():
            summed = math.fsum(values)
            times[(c, REGIMES[r], METHODS[m])] = summed
            summary = summaries[(c, REGIMES[r], METHODS[m])]
            a.close(summary["step_time_sum_seconds"], summed, "summary time from all 64 records")
            a.close(summary["step_time_mean_seconds"], summed / 64, "summary mean time")
        a.close(receipt["estimator_wall_seconds"], math.fsum(t for values in partitions.values() for t in values), "receipt all estimator wall")
    a.exact(total_records, 4096, "4096 total measurements")

    integrity = score["integrity"]
    expected_integrity = {(c, REGIMES[r], (METHODS[i], METHODS[j]))
                          for c in range(8) for r in range(2)
                          for i, j in itertools.combinations(range(3 if r == 0 else 4), 2)}
    integrity_keys, shared_means = set(), {}
    observed_checks, failures, smallest_margin, largest_fraction = 0, 0, math.inf, 0.
    for row in integrity["records"]:
        key = (row["case_index"], row["regime"], tuple(row["pair"]))
        a.require(key not in integrity_keys, "unique projected pair")
        integrity_keys.add(key)
        for field in ("projected_means_left", "projected_means_right", "paired_mean_difference",
                      "paired_standard_error", "tolerance", "passed"):
            a.exact(len(row[field]), 8, "eight projections per pair")
        for k in range(8):
            left, right, se = row["projected_means_left"][k], row["projected_means_right"][k], row["paired_standard_error"][k]
            a.require(all(math.isfinite(x) for x in (left, right, se)) and se >= 0, "finite public projection summaries")
            difference = left - right
            limit = 5 * se + 1e-7 * max(1., abs(left), abs(right))
            a.close(row["paired_mean_difference"][k], difference, "paired mean = difference of means")
            a.close(row["tolerance"][k], limit, "5SE plus fixed floor")
            passed = abs(difference) <= limit
            a.exact(row["passed"][k], passed, "projected mean decision")
            failures += not passed
            observed_checks += 1
            smallest_margin = min(smallest_margin, limit - abs(difference))
            largest_fraction = max(largest_fraction, abs(difference) / limit)
            for method, value in zip(row["pair"], (left, right)):
                mean_key = (key[0], key[1], method, k)
                if mean_key in shared_means:
                    a.close(value, shared_means[mean_key], "shared projected mean across pairs")
                else:
                    shared_means[mean_key] = value
    a.require(integrity_keys == expected_integrity and len(integrity["records"]) == 72, "72 complete projected pair records")
    a.exact(observed_checks, 576, "576 checks, not pair-record count")
    a.exact(integrity["checks"], observed_checks, "reported projection check count")
    a.exact(integrity["failed_checks"], failures, "reported mean failures")
    a.exact(integrity["all_passed"], failures == 0, "integrity aggregate")

    # Confidence bounds are not regenerated locally. Check their frozen schema,
    # gate application and binding to the saved independent CLOUD replay.
    a.exact(score["bootstrap"]["draws"], 2000, "bootstrap count")
    a.exact(score["bootstrap"]["seed_formula"], "2000000 + case_index", "bootstrap seeds")
    a.close(score["bootstrap"]["upper_quantile"], 0.9833333333333333, "bootstrap quantile")
    replay = a.load("outer-control/independent-replay.stdout.log")
    execution = a.load("outer-control/independent-replay.execution.json")
    a.exact(replay["status"], "PASS", "saved cloud replay PASS")
    a.exact(replay["scores_sha256"], a.hashes["study/scores.json"], "cloud replay binds exact scores")
    a.exact(replay["checker_sha256"], CHECKER_PIN, "cloud checker expected source")
    a.exact(digest(root / "outer-control/independent_checker.py"), CHECKER_PIN, "archived checker source bytes")
    a.exact(execution["actual_exit"], 0, "cloud replay actual exit")
    a.exact(execution["raw_wait_status"], 0, "cloud replay raw wait status")
    a.exact(execution["stdout_sha256"], a.hashes["outer-control/independent-replay.stdout.log"], "cloud replay stdout binding")
    a.exact(replay["projection_checks"], 576, "cloud replay mean count")
    a.exact(replay["bootstrap_draws"], 2000, "cloud replay bootstrap count")
    a.exact(replay["decision_recomputed"], score["decision"], "cloud replay agrees decision")
    for c, name in enumerate(CASES):
        a.exact(replay["input_file_hashes"][c]["receipt_sha256"], a.hashes["cases/" + name + "/receipt.json"], "cloud receipt binding " + name)

    comparisons = {}
    for row in score["comparisons"]:
        key = (row["regime"], row["control"])
        a.require(key not in comparisons, "unique comparison")
        a.exact(row["numerator"], "mean", "ratio numerator")
        r, control = key
        ratios = [summaries[(c, r, "mean")]["trace_covariance"] / summaries[(c, r, control)]["trace_covariance"] for c in range(8)]
        adjusted = [ratios[c] * times[(c, r, "mean")] / times[(c, r, control)] for c in range(8)]
        a.exact(len(row["per_case_variance_ratio"]), 8, "eight equal-weight case ratios")
        for got, expected in zip(row["per_case_variance_ratio"], ratios):
            a.close(got, expected, "case variance ratio")
        a.close(row["variance_ratio_geomean"], gm(ratios), "equal-case geometric variance ratio")
        a.close(row["variance_times_step_time_sum_ratio_geomean"], gm(adjusted), "equal-case geometric V*time ratio")
        a.require(math.isfinite(row["bootstrap_upper"]) and row["bootstrap_upper"] > 0, "finite positive bound")
        comparisons[key] = {"variance_ratio": gm(ratios), "variance_times_time_ratio": gm(adjusted),
                            "published_cloud_replayed_upper": row["bootstrap_upper"], "per_case_ratios": ratios}
    a.require(set(comparisons) == set(itertools.product(REGIMES, ("independent", "random", "samepatch"))), "all six comparisons")
    wins = [c for c in range(8) if all(summaries[(c, "B_total", "mean")]["trace_covariance"] < summaries[(c, "B_total", m)]["trace_covariance"] for m in ("independent", "random", "samepatch"))]
    total_independent = math.fsum(times[(c, "B_total", "independent")] for c in range(8))
    total_mean = math.fsum(times[(c, "B_total", "mean")] for c in range(8))
    cost_ratio = total_mean / total_independent
    a_ratio = comparisons[("A_augmentation", "random")]["variance_ratio"]
    a.close(score["B_total_mean_vs_independent_step_time_ratio"], cost_ratio, "total B cost from 1024 individual records")
    a.close(score["A_mean_vs_random_variance_geomean"], a_ratio, "A mechanism ratio")
    a.exact(score["B_best_control_win_count"], len(wins), "strict best-control win count")
    a.exact(score["B_best_control_win_cases"], wins, "strict best-control winning cases")
    a.exact(score["B_required_case_wins"], 6, "six required wins")
    b = [comparisons[("B_total", m)] for m in ("independent", "random", "samepatch")]
    gates = {"finite_positive_inputs_and_denominators": True,
             "projected_mean_integrity": failures == 0,
             "A_mean_vs_random_variance_geomean_below_1": a_ratio < 1,
             "B_variance_geomean_at_most_090_each_control": all(row["variance_ratio"] <= 0.90 for row in b),
             "B_bootstrap_upper_below_1_each_control": all(row["published_cloud_replayed_upper"] < 1 for row in b),
             "B_best_control_case_wins": len(wins) >= 6,
             "B_total_step_time_vs_independent_at_most_110": cost_ratio <= 1.10,
             "B_variance_times_time_geomean_at_most_090_each_control": all(row["variance_times_time_ratio"] <= 0.90 for row in b)}
    gates["all_passed"] = all(gates.values())
    a.exact(score["gates"], gates, "complete gate dictionary")
    decision = "pass_local_mechanism_screen_only" if gates["all_passed"] else "stop_exact_mean_pairing_proposal"
    a.exact(score["decision"], decision, "decision recomputed")
    for rel, expected_hash in a.hashes.items():
        a.exact(digest(root / rel), expected_hash, "public artifact unchanged: " + rel)
    return {"schema": "0007-parent-public-metric-check-v1", "status": "PASS" if a.failure_count == 0 else "FAIL",
            "checks": a.count, "numeric_checks": a.numbers, "failure_count": a.failure_count,
            "failures": a.failures, "max_absolute_difference": a.max_abs, "max_relative_difference": a.max_rel,
            "rtol": RTOL, "atol": ATOL, "source_commit": SOURCE_COMMIT, "freeze_commit": FREEZE_COMMIT,
            "public_input_hashes": a.hashes, "point_comparisons": [{"regime": r, "control": m, **v} for (r, m), v in comparisons.items()],
            "timing_records_checked": total_records, "projection_pair_records": 72,
            "projection_checks": observed_checks, "projection_failures": failures,
            "minimum_projection_threshold_margin": smallest_margin,
            "maximum_absolute_difference_over_projection_threshold": largest_fraction,
            "B_independent_step_seconds": total_independent, "B_mean_step_seconds": total_mean,
            "B_total_cost_ratio": cost_ratio, "B_cost_ratio_margin_to_110": 1.10 - cost_ratio,
            "B_mean_additional_seconds_allowed_at_gate": 1.10 * total_independent - total_mean,
            "B_strict_best_control_wins": wins, "gates": gates, "decision": decision,
            "cloud_replay_record": {k: replay[k] for k in ("status", "checks", "numeric_checks", "max_absolute_difference", "scores_sha256", "checker_sha256")},
            "scope": {"fresh_raw_gradient_replay": False, "fresh_bootstrap_replay": False,
                      "fresh_mean_standard_error_recomputation": False, "public_arithmetic_recomputed": True,
                      "bootstrap_and_raw_gradient_support": "Bound to the saved independent cloud replay; not regenerated locally.",
                      "claim_limit": "Local fixed-checkpoint mechanism screen only. Timing is one observed run, not a robust speedup guarantee; no optimizer or denoising-quality conclusion."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = run(args.results)
    report.update(checker_sha256=digest(Path(__file__)), elapsed_seconds=time.monotonic() - started,
                  python_version=platform.python_version())
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.report.open("x") as f:
        f.write(encoded)
    print(json.dumps({k: report[k] for k in ("status", "checks", "numeric_checks", "max_absolute_difference", "failure_count", "B_total_cost_ratio", "B_cost_ratio_margin_to_110", "B_strict_best_control_wins", "decision", "checker_sha256")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
