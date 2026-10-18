#!/usr/bin/env python3
"""Offline arithmetic replay of saved 0007 gradients. No production imports.

Run under an external 120-second hard limit and four-thread environment.
Reads only scores.json and fixed cases/{name}/{gradients.npy,walls.npy,receipt.json}.
It performs no model calls and emits compact JSON, never raw gradients.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"
import numpy as np


NAMES = ["d%d-colored-%d" % (d, seed) for d in range(1, 5) for seed in (0, 1)]
METHODS = ["independent", "random", "mean", "samepatch"]
REGIMES = ["A_augmentation", "B_total"]
QUANTILE = 0.9833333333333333
RTOL, ATOL = 1e-10, 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({"shape": list(a.shape), "dtype": a.dtype.str},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    h = hashlib.sha256(header + b"\0")
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def load_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("invalid or oversized JSON evidence file: " + path.name)
    def bad_constant(_):
        raise ValueError("nonfinite JSON constant")
    def unique_pairs(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out
    with path.open() as f:
        return json.load(f, parse_constant=bad_constant, object_pairs_hook=unique_pairs)


class Comparison:
    def __init__(self):
        self.checks = 0
        self.numeric_checks = 0
        self.max_absolute_difference = 0.0
        self.max_relative_difference = 0.0
        self.failures = []

    def check(self, path, got, expected):
        self.checks += 1
        failure = None
        if isinstance(expected, dict):
            if not isinstance(got, dict) or set(got) != set(expected):
                failure = "object keys differ"
            else:
                for k in expected:
                    self.check(path + "." + k, got[k], expected[k])
        elif isinstance(expected, list):
            if not isinstance(got, list) or len(got) != len(expected):
                failure = "list length/type differs"
            else:
                for i, item in enumerate(expected):
                    self.check(path + "[%d]" % i, got[i], item)
        elif isinstance(expected, bool):
            if type(got) is not bool or got != expected:
                failure = "boolean differs"
        elif isinstance(expected, int):
            if type(got) is not int or got != expected:
                failure = "integer differs"
        elif isinstance(expected, float):
            self.numeric_checks += 1
            if type(got) not in (int, float) or not math.isfinite(got) or not math.isfinite(expected):
                failure = "nonfinite/non-numeric scalar"
            else:
                delta = abs(got - expected)
                self.max_absolute_difference = max(self.max_absolute_difference, delta)
                self.max_relative_difference = max(self.max_relative_difference,
                                                   delta / max(abs(expected), ATOL))
                if delta > ATOL + RTOL * abs(expected):
                    failure = "numeric tolerance exceeded"
        elif got != expected or type(got) is not type(expected):
            failure = "value/type differs"
        if failure and len(self.failures) < 40:
            self.failures.append({"field": path, "reason": failure})


def resampling_indices(case, replicates, draws):
    # Freeze specifies NumPy default_rng integer draws, including dtype default.
    return np.random.default_rng(2000000 + case).integers(
        0, replicates, size=(draws, replicates))


def distance_bootstrap(x, selection):
    """Independent identity: sum of resampled pair distances / [R(R-1)].

    Squared distances use direct row subtraction, not the production Gram
    variance/correction formula. No bootstrap gradient tensor is allocated.
    """
    r = x.shape[0]
    ii, jj = np.triu_indices(r, k=1)
    distances = np.empty(len(ii), dtype=np.float64)
    offset = 0
    for i in range(r - 1):
        delta = x[i + 1:] - x[i]
        length = len(delta)
        distances[offset:offset + length] = np.sum(np.square(delta), axis=1)
        offset += length
    counts = np.array([np.bincount(row, minlength=r) for row in selection], dtype=np.float64)
    result = np.empty(len(selection), dtype=np.float64)
    for first in range(0, len(selection), 128):
        w = counts[first:first + 128]
        pair_weights = w[:, ii] * w[:, jj]
        result[first:first + len(w)] = np.sum(pair_weights * distances[None, :], axis=1) / (r * (r - 1))
    return result


def explicit_upper(samples):
    values = sorted(float(x) for x in samples)
    position = (len(values) - 1) * QUANTILE
    low = math.floor(position)
    fraction = position - low
    if fraction == 0:
        return values[low]
    return values[low] * (1 - fraction) + values[low + 1] * fraction


def geometric(values):
    return math.exp(math.fsum(math.log(float(v)) for v in values) / len(values))


def replay_arrays(case_arrays, wall_arrays, *, draws=2000):
    """Small fixtures may call this; the file entry point enforces full shape."""
    cases = len(case_arrays)
    r, parameters = case_arrays[0].shape[-2:]
    bits = np.random.default_rng(709999).integers(0, 2, size=(8, parameters), dtype=np.int8)
    directions = np.where(bits == 0, -1.0, 1.0) / math.sqrt(parameters)
    variance = np.empty((cases, 2, 4))
    totals = np.empty((cases, 2, 4))
    boot = np.empty((cases, 2, 4, draws))
    summaries, checks = [], []
    for c, (raw, timing) in enumerate(zip(case_arrays, wall_arrays)):
        selection = resampling_indices(c, r, draws)
        projections = np.empty((2, 4, r, 8))
        for a in range(2):
            for m in range(4):
                x = np.asarray(raw[a, m], dtype=np.float64)
                variance[c, a, m] = np.var(x, axis=0, dtype=np.float64, ddof=1).sum(dtype=np.float64)
                means = x.mean(axis=0, dtype=np.float64)
                norm = math.sqrt(float(np.square(means).sum(dtype=np.float64)))
                totals[c, a, m] = math.fsum(float(v) for v in timing[a, m])
                for k in range(8):
                    projections[a, m, :, k] = np.sum(x * directions[k], axis=1, dtype=np.float64)
                boot[c, a, m] = distance_bootstrap(x, selection)
                summaries.append({"case_index": c, "regime": REGIMES[a], "method": METHODS[m],
                                  "trace_covariance": float(variance[c, a, m]),
                                  "mean_gradient_norm": norm,
                                  "step_time_sum_seconds": float(totals[c, a, m]),
                                  "step_time_mean_seconds": float(totals[c, a, m] / r)})
        for a in range(2):
            for i in range(3 if a == 0 else 4):
                for j in range(i + 1, 3 if a == 0 else 4):
                    record = {"case_index": c, "regime": REGIMES[a], "pair": [METHODS[i], METHODS[j]],
                              "projected_means_left": [], "projected_means_right": [],
                              "paired_mean_difference": [], "paired_standard_error": [],
                              "tolerance": [], "passed": []}
                    for k in range(8):
                        left, right = projections[a, i, :, k], projections[a, j, :, k]
                        mean_left = math.fsum(left) / r
                        mean_right = math.fsum(right) / r
                        delta = left - right
                        average_delta = math.fsum(delta) / r
                        se = math.sqrt(math.fsum((float(z) - average_delta)**2 for z in delta) / (r * (r - 1)))
                        limit = 5 * se + 1e-7 * max(1, abs(mean_left), abs(mean_right))
                        for key, val in (("projected_means_left", mean_left), ("projected_means_right", mean_right),
                                         ("paired_mean_difference", average_delta), ("paired_standard_error", se),
                                         ("tolerance", limit), ("passed", abs(average_delta) <= limit)):
                            record[key].append(val)
                    checks.append(record)
    # Strictly positive finite variances are required for this complete replay.
    if any(not np.isfinite(v).all() or (v <= 0).any() for v in (variance, totals, boot)):
        raise ValueError("degenerate/nonfinite variance or timing; full numeric replay cannot pass")
    integrity = {"all_passed": all(all(row["passed"]) for row in checks),
                 "checks": len(checks) * 8,
                 "failed_checks": sum(sum(not p for p in row["passed"]) for row in checks),
                 "records": checks,
                 "interpretation": "Finite Monte Carlo diagnostic; failure is an unresolved integrity warning, not proof of bias. A samepatch is deliberately excluded."}
    comparisons = []
    for a in range(2):
        for control in (0, 1, 3):
            per_case = [float(variance[c, a, 2] / variance[c, a, control]) for c in range(cases)]
            # Same bootstrap b spans the eight independently resampled cases.
            ratios = np.exp(np.log(boot[:, a, 2] / boot[:, a, control]).mean(axis=0))
            cost_ratios = [per_case[c] * float(totals[c, a, 2] / totals[c, a, control]) for c in range(cases)]
            comparisons.append({"regime": REGIMES[a], "numerator": "mean", "control": METHODS[control],
                                "variance_ratio_geomean": geometric(per_case),
                                "bootstrap_upper": explicit_upper(ratios),
                                "variance_times_step_time_sum_ratio_geomean": geometric(cost_ratios),
                                "per_case_variance_ratio": per_case})
    won = [c for c in range(cases) if all(variance[c, 1, 2] < variance[c, 1, m] for m in (0, 1, 3))]
    elapsed_ratio = math.fsum(totals[:, 1, 2]) / math.fsum(totals[:, 1, 0])
    required_wins = 6 if cases == 8 else math.ceil(cases * 0.75)
    b = comparisons[3:]
    a_ratio = comparisons[1]["variance_ratio_geomean"]
    gates = {
        "finite_positive_inputs_and_denominators": True,
        "projected_mean_integrity": integrity["all_passed"],
        "A_mean_vs_random_variance_geomean_below_1": a_ratio < 1,
        "B_variance_geomean_at_most_090_each_control": all(row["variance_ratio_geomean"] <= 0.90 for row in b),
        "B_bootstrap_upper_below_1_each_control": all(row["bootstrap_upper"] < 1 for row in b),
        "B_best_control_case_wins": len(won) >= required_wins,
        "B_total_step_time_vs_independent_at_most_110": elapsed_ratio <= 1.10,
        "B_variance_times_time_geomean_at_most_090_each_control": all(row["variance_times_step_time_sum_ratio_geomean"] <= 0.90 for row in b),
    }
    gates["all_passed"] = all(gates.values())
    return {"schema_version": 1, "status": "analyzed", "shape": [cases, 2, 4, r, parameters],
            "methods": METHODS, "regimes": REGIMES,
            "scope": "Monte Carlo uncertainty on fixed saved development cases only; no independent-scene or population coverage claim.",
            "bootstrap": {"draws": draws, "seed_formula": "2000000 + case_index",
                          "paired_across": "all methods and both regimes within each case",
                          "case_aggregation": "equal arithmetic mean of log variance ratios",
                          "upper_quantile": QUANTILE,
                          "quantile_rule": "sort; h=(n-1)*q; interpolate floor(h),ceil(h)",
                          "interpretation": "Approximate paired empirical bootstrap Monte Carlo uncertainty, not an exact nominal confidence guarantee. Three prespecified comparisons use the stated one-sided quantile."},
            "projection": {"count": 8, "seed": 709999,
                           "rule": "numpy.default_rng(seed).integers(0,2,(8,P),dtype=int8); (2*bits-1)/sqrt(P) in float64"},
            "integrity": integrity, "summaries": summaries, "comparisons": comparisons,
            "gates": gates, "decision": "pass_local_mechanism_screen_only" if gates["all_passed"] else "stop_exact_mean_pairing_proposal",
            "B_best_control_win_count": len(won), "B_best_control_win_cases": won,
            "B_required_case_wins": required_wins,
            "B_total_mean_vs_independent_step_time_ratio": elapsed_ratio,
            "A_mean_vs_random_variance_geomean": a_ratio, "production_contract": cases == 8 and r == 64 and parameters == 5313}


def check_directory(root, expected_scores_sha256=None):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("output root must be a real directory")
    score_path = root / "scores.json"
    score_sha = sha(score_path)
    if expected_scores_sha256 and score_sha != expected_scores_sha256:
        raise ValueError("scores SHA256 differs from caller pin")
    score = load_json(score_path)
    comparison = Comparison()
    binding = score.get("bindings", {})
    if set(binding) != {"cases", "freeze_commit", "source_commit", "environment"}:
        raise ValueError("unexpected score binding schema")
    if len(binding["cases"]) != 8:
        raise ValueError("all eight score case bindings required")
    gradients, timings, files = [], [], []
    for c, name in enumerate(NAMES):
        folder = root / "cases" / name
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("fixed case directory missing/symlink")
        receipt_path = folder / "receipt.json"
        receipt_sha = sha(receipt_path)
        receipt = load_json(receipt_path)
        comparison.check("bindings.cases[%d]" % c, binding["cases"][c],
                         {"case": name, "receipt_sha256": receipt_sha})
        expected_fields = {"schema": "0007-gradient-core-v1", "case_index": c,
                           "status": "completed", "failure": None, "regimes": ["A_fixed", "B_redraw"],
                           "methods": ["independent", "random_antithetic", "mean_antithetic", "same_patch"],
                           "replicates": 64, "batch_size": 512, "parameter_count": 5313,
                           "architecture": [49, 64, 32, 1], "optimizer_steps": 0,
                           "completed_measurements": 512, "expected_measurements": 512,
                           "gradient_array_shape": [2, 4, 64, 5313], "weight_unchanged": True,
                           "clean_reference_accessed": False, "device": "cpu", "numeric_threads": 4,
                           "source_commit": binding["source_commit"], "freeze_commit": binding["freeze_commit"]}
        for field, expected in expected_fields.items():
            comparison.check("receipt.%s.%s" % (name, field), receipt.get(field), expected)
        comparison.check("receipt.%s.weight_preservation" % name,
                         receipt.get("weight_after_sha256"), receipt.get("weight_before_sha256"))
        if receipt.get("case_identity", {}).get("name") != name or receipt.get("case_identity", {}).get("index") != c:
            raise ValueError("case identity/order mismatch")
        arrays = []
        file_hashes = {"case": name, "receipt_sha256": receipt_sha}
        for filename, shape, dtype in (("gradients.npy", (2, 4, 64, 5313), "float32"),
                                       ("walls.npy", (2, 4, 64), "float64")):
            path = folder / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing/symlink saved array")
            digest = sha(path)
            comparison.check("receipt.%s.files.%s" % (name, filename), receipt.get("files", {}).get(filename), digest)
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
                raise ValueError("saved gradient/timing shape, dtype, or finite check failed")
            if filename == "walls.npy" and (value <= 0).any():
                raise ValueError("nonpositive saved timing")
            arrays.append(value)
            file_hashes[filename + "_sha256"] = digest
        gradients.append(arrays[0]); timings.append(arrays[1])
        comparison.check("receipt.%s.gradient_array_sha256" % name,
                         receipt.get("gradient_array_sha256"), array_sha(arrays[0]))
        comparison.check("receipt.%s.walls_sha256" % name, receipt.get("walls_sha256"), array_sha(arrays[1]))
        files.append(file_hashes)
    if comparison.failures:
        raise ValueError("evidence binding checks failed: " + str(comparison.failures[:3]))
    expected = replay_arrays(gradients, timings)
    actual = {key: value for key, value in score.items() if key != "bindings"}
    comparison.check("scores", actual, expected)
    # Detect accidental concurrent mutation; no scientific rerun is attempted.
    comparison.check("scores_sha256_after", sha(score_path), score_sha)
    for row in files:
        folder = root / "cases" / row["case"]
        for filename in ("gradients.npy", "walls.npy", "receipt.json"):
            key = "receipt_sha256" if filename == "receipt.json" else filename + "_sha256"
            comparison.check("unchanged.%s.%s" % (row["case"], filename), sha(folder / filename), row[key])
    return {"schema": "0007-independent-saved-gradient-replay-v1",
            "status": "PASS" if not comparison.failures else "FAIL",
            "checks": comparison.checks, "numeric_checks": comparison.numeric_checks,
            "max_absolute_difference": comparison.max_absolute_difference,
            "max_relative_difference": comparison.max_relative_difference,
            "rtol": RTOL, "atol": ATOL, "failures": comparison.failures,
            "scores_sha256": score_sha, "input_file_hashes": files,
            "source_commit": binding["source_commit"], "freeze_commit": binding["freeze_commit"],
            "bootstrap_draws": 2000, "projection_checks": 576,
            "variance_estimates": 64, "decision_recomputed": expected["decision"],
            "no_model_calls": True, "raw_gradients_exported": False,
            "scope": "Arithmetic replay and internal receipt/file binding only. Source/freeze authenticity, original model execution and absence of hidden input access require the separate execution audit. Monte Carlo intervals concern fixed cases, not population coverage."}


def self_test():
    count = 0
    x = np.array([[1., 2., 3.], [3., 4., 1.], [9., 1., -1.], [2., 5., 0.], [4., -1., 8.]])
    indices = resampling_indices(0, len(x), 23)
    from_pairs = distance_bootstrap(x, indices)
    explicit = np.array([np.var(x[row], axis=0, ddof=1).sum() for row in indices])
    np.testing.assert_allclose(from_pairs, explicit, rtol=1e-13, atol=1e-13); count += 1
    translated = distance_bootstrap(x + 1000, indices)
    np.testing.assert_allclose(translated, from_pairs, rtol=1e-13, atol=1e-13); count += 1
    rng = np.random.default_rng(19)
    gradients, walls = [], []
    for _ in range(2):
        base = rng.normal(size=(2, 1, 32, 3)).astype(np.float32)
        case = np.repeat(base, 4, axis=1)
        case[:, 2] *= 0.5
        gradients.append(case); walls.append(np.ones((2, 4, 32)))
    out = replay_arrays(gradients, walls, draws=64)
    assert out["gates"]["all_passed"]; count += 1
    assert out["integrity"]["checks"] == 144; count += 1
    for row in out["comparisons"]:
        for key in ("variance_ratio_geomean", "bootstrap_upper", "variance_times_step_time_sum_ratio_geomean"):
            assert abs(row[key] - 0.25) < 1e-12
    count += 1
    altered = Comparison(); altered.check("scalar", 1.001, 1.0)
    assert altered.failures; count += 1
    exact = Comparison(); exact.check("whole", out, out)
    assert not exact.failures; count += 1
    gradients[0][0, 3] += 100
    shifted_a = replay_arrays(gradients, walls, draws=16)
    assert shifted_a["integrity"]["all_passed"]; count += 1
    gradients[0][1, 3] += 100
    shifted_b = replay_arrays(gradients, walls, draws=16)
    assert not shifted_b["integrity"]["all_passed"]; count += 1
    assert math.isclose(explicit_upper([0., 10.]), 9.833333333333332); count += 1
    json.dumps(out, allow_nan=False); count += 1
    return {"status": "PASS", "self_test_assertion_groups": count,
            "scope": "small mathematical fixtures only; no scientific data or production implementation import"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-scores-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        if args.self_test:
            if args.output_root is not None:
                raise ValueError("self-test never accepts an evidence directory")
            result = self_test()
        else:
            if args.output_root is None:
                raise ValueError("--output-root required")
            result = check_directory(args.output_root, args.expected_scores_sha256)
        result.update(checker_sha256=sha(__file__), elapsed_seconds=time.monotonic() - started,
                      python_version=sys.version.split()[0], numpy_version=np.__version__)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "PASS" else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
                          "elapsed_seconds": time.monotonic() - started}, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
