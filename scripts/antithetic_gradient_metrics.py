"""Independent NumPy-only 0007 Monte Carlo scorer; no model or image access.

Production input is strictly float32 gradients [8,2,4,64,5313] and positive
floating-point seconds [8,2,4,64]. Public output contains summaries only.
Private helpers accept small mathematical fixtures, not alternate studies.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

METHODS = ("independent", "random", "mean", "samepatch")
REGIMES = ("A_augmentation", "B_total")
CONTROLS = (0, 1, 3)
SHAPE = (8, 2, 4, 64, 5313)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED_BASE = 2000000
PROJECTION_SEED = 709999
PROJECTIONS = 8
UPPER_QUANTILE = 0.9833333333333333


def _linear_quantile(values, q):
    """Explicit sorted linear interpolation at (n-1)*q; no hidden default."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or not 0 <= q <= 1:
        raise ValueError("quantile requires a finite nonempty vector and q in [0,1]")
    x = np.sort(x)
    h = (len(x) - 1) * float(q)
    lower, upper = int(np.floor(h)), int(np.ceil(h))
    return float(x[lower] + (h - lower) * (x[upper] - x[lower]))


def _rademacher(dimension):
    rng = np.random.default_rng(PROJECTION_SEED)
    bits = rng.integers(0, 2, size=(PROJECTIONS, dimension), dtype=np.int8)
    return (2.0 * bits.astype(np.float64) - 1.0) / np.sqrt(float(dimension))


def _centered_gram(gradients):
    x = np.asarray(gradients, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("finite [replicate,parameter] gradients required")
    mean = x.mean(axis=0, dtype=np.float64)
    centered = x - mean
    gram = np.einsum("ri,si->rs", centered, centered, optimize=False)
    variance = float(np.trace(gram) / (x.shape[0] - 1))
    return gram, variance, float(np.linalg.norm(mean))


def _resample_counts(replicates, draws, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, replicates, size=(draws, replicates))
    counts = np.zeros((draws, replicates), dtype=np.float64)
    np.add.at(counts, (np.arange(draws)[:, None], indices), 1.0)
    return counts


def _bootstrap_variance(gram, counts):
    """Unbiased sample trace covariance of resampled rows via a centered Gram.

    Each row of counts describes R draws with replacement. The centering used
    to build gram need not equal a resample mean: the correction below supplies
    the resample-specific mean exactly (up to float64 arithmetic).
    """
    k = np.asarray(gram, dtype=np.float64)
    w = np.asarray(counts, dtype=np.float64)
    r = k.shape[0]
    if k.shape != (r, r) or r < 2 or w.ndim != 2 or w.shape[1] != r:
        raise ValueError("invalid Gram/count shapes")
    if not np.isfinite(k).all() or not np.isfinite(w).all():
        raise ValueError("nonfinite Gram/counts")
    if (w < 0).any() or not np.equal(w, np.floor(w)).all() or not (w.sum(1) == r).all():
        raise ValueError("counts must encode exactly R draws per row")
    first = np.einsum("bi,i->b", w, np.diag(k), optimize=False)
    quadratic = np.einsum("bi,ij,bj->b", w, k, w, optimize=False)
    return (first - quadratic / r) / (r - 1)


def _mean_integrity(projected):
    # projected: [case,regime,method,replicate,projection]
    c_count, _, _, r, _ = projected.shape
    records = []
    for c in range(c_count):
        for regime in range(2):
            included = range(3) if regime == 0 else range(4)
            for i, j in itertools.combinations(included, 2):
                left, right = projected[c, regime, i], projected[c, regime, j]
                means_left, means_right = left.mean(0), right.mean(0)
                differences = left - right
                mean_difference = differences.mean(0)
                stderr = differences.std(0, ddof=1) / np.sqrt(float(r))
                tolerance = 5.0 * stderr + 1e-7 * np.maximum(
                    1.0, np.maximum(np.abs(means_left), np.abs(means_right))
                )
                passed = np.abs(mean_difference) <= tolerance
                records.append({
                    "case_index": c, "regime": REGIMES[regime],
                    "pair": [METHODS[i], METHODS[j]],
                    "projected_means_left": means_left.tolist(),
                    "projected_means_right": means_right.tolist(),
                    "paired_mean_difference": mean_difference.tolist(),
                    "paired_standard_error": stderr.tolist(),
                    "tolerance": tolerance.tolist(), "passed": passed.tolist(),
                })
    passed = all(all(x["passed"]) for x in records)
    return {"all_passed": passed, "checks": len(records) * PROJECTIONS,
            "failed_checks": sum(sum(not v for v in x["passed"]) for x in records),
            "records": records,
            "interpretation": "Finite Monte Carlo diagnostic; failure is an unresolved integrity warning, not proof of bias. A samepatch is deliberately excluded."}


def _geomean_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise ValueError("nonfinite ratio inputs")
    if (numerator <= 0).any() or (denominator <= 0).any():
        raise ValueError("variance ratios require strictly positive values")
    return float(np.exp(np.mean(np.log(numerator) - np.log(denominator))))


def _analyze(gradients, timings, *, bootstrap_draws=BOOTSTRAP_DRAWS,
             min_case_wins=None):
    """Private variable-size mathematical-fixture entry; use public API for data."""
    g, t = np.asarray(gradients), np.asarray(timings)
    if g.dtype != np.dtype("float32") or g.ndim != 5 or g.shape[1:3] != (2, 4):
        raise ValueError("gradients must be float32 [case,2,4,replicate,parameter]")
    cases, _, _, r, p = g.shape
    if cases < 1 or r < 2 or p < 1 or t.shape != g.shape[:-1]:
        raise ValueError("invalid sizes or timing shape")
    if t.dtype.kind != "f" or not np.isfinite(g).all() or not np.isfinite(t).all():
        raise ValueError("all gradients and floating-point timings must be finite")
    if (t <= 0).any():
        raise ValueError("every step timing must be strictly positive")
    if not isinstance(bootstrap_draws, int) or bootstrap_draws < 2:
        raise ValueError("at least two bootstrap draws required")
    if min_case_wins is None:
        min_case_wins = int(np.ceil(0.75 * cases))
    if not 1 <= min_case_wins <= cases:
        raise ValueError("invalid minimum case wins")

    directions = _rademacher(p)
    projected = np.empty((cases, 2, 4, r, PROJECTIONS), dtype=np.float64)
    variances = np.empty((cases, 2, 4), dtype=np.float64)
    norms = np.empty_like(variances)
    boot = np.empty((cases, 2, 4, bootstrap_draws), dtype=np.float64)
    # Process only one [R,P] array at a time in float64. Bootstrap work uses
    # [R,R] Gram matrices, never a [draw,R,P] gradient allocation.
    for c in range(cases):
        counts = _resample_counts(r, bootstrap_draws, BOOTSTRAP_SEED_BASE + c)
        for regime in range(2):
            for method in range(4):
                x = g[c, regime, method].astype(np.float64)
                gram, variances[c, regime, method], norms[c, regime, method] = _centered_gram(x)
                projected[c, regime, method] = np.einsum("ri,ki->rk", x, directions, optimize=False)
                boot[c, regime, method] = _bootstrap_variance(gram, counts)

    integrity = _mean_integrity(projected)
    step_sums = t.astype(np.float64).sum(axis=3)
    summaries = [
        {"case_index": c, "regime": REGIMES[a], "method": METHODS[m],
         "trace_covariance": float(variances[c, a, m]),
         "mean_gradient_norm": float(norms[c, a, m]),
         "step_time_sum_seconds": float(step_sums[c, a, m]),
         "step_time_mean_seconds": float(step_sums[c, a, m] / r)}
        for c in range(cases) for a in range(2) for m in range(4)
    ]
    result = {
        "schema_version": 1, "status": "analyzed", "shape": list(g.shape),
        "methods": list(METHODS), "regimes": list(REGIMES),
        "scope": "Monte Carlo uncertainty on fixed saved development cases only; no independent-scene or population coverage claim.",
        "bootstrap": {"draws": bootstrap_draws,
                      "seed_formula": "2000000 + case_index",
                      "paired_across": "all methods and both regimes within each case",
                      "case_aggregation": "equal arithmetic mean of log variance ratios",
                      "upper_quantile": UPPER_QUANTILE,
                      "quantile_rule": "sort; h=(n-1)*q; interpolate floor(h),ceil(h)",
                      "interpretation": "Approximate paired empirical bootstrap Monte Carlo uncertainty, not an exact nominal confidence guarantee. Three prespecified comparisons use the stated one-sided quantile."},
        "projection": {"count": PROJECTIONS, "seed": PROJECTION_SEED,
                       "rule": "numpy.default_rng(seed).integers(0,2,(8,P),dtype=int8); (2*bits-1)/sqrt(P) in float64"},
        "integrity": integrity, "summaries": summaries,
    }
    # Degenerate/nonfinite resamples are not deleted, clipped, or redrawn.
    invalid = []
    for name, values in (("trace_covariances", variances), ("bootstrap_trace_covariances", boot),
                         ("timing_sums", step_sums), ("gradient_mean_norms", norms)):
        if not np.isfinite(values).all():
            invalid.append(name + ": nonfinite")
        if name != "gradient_mean_norms" and (values <= 0).any():
            invalid.append(name + ": nonpositive; ratios undefined")
    if invalid:
        result.update(status="invalid", errors=invalid, decision="stop_invalid", gates={"all_passed": False})
        # JSON does not admit NaN/Inf; overflowing summaries are unusable.
        if not all(np.isfinite(a).all() for a in (variances, step_sums, norms)):
            result["summaries"] = []
        return result

    comparisons = []
    for regime in range(2):
        for control in CONTROLS:
            observed = _geomean_ratio(variances[:, regime, 2], variances[:, regime, control])
            draws = np.exp(np.mean(np.log(boot[:, regime, 2]) - np.log(boot[:, regime, control]), axis=0))
            # Equivalent to V*time ratios, without overflowing intermediate products.
            cost_log_ratio = (np.log(variances[:, regime, 2])
                              - np.log(variances[:, regime, control])
                              + np.log(step_sums[:, regime, 2])
                              - np.log(step_sums[:, regime, control]))
            cost_adjusted = float(np.exp(np.mean(cost_log_ratio)))
            comparisons.append({"regime": REGIMES[regime], "numerator": "mean",
                                "control": METHODS[control],
                                "variance_ratio_geomean": observed,
                                "bootstrap_upper": _linear_quantile(draws, UPPER_QUANTILE),
                                "variance_times_step_time_sum_ratio_geomean": cost_adjusted,
                                "per_case_variance_ratio": (variances[:, regime, 2] / variances[:, regime, control]).tolist()})
    b_results = comparisons[3:]
    best = np.min(variances[:, 1, :][:, CONTROLS], axis=1)
    win_mask = variances[:, 1, 2] < best
    time_ratio = float(step_sums[:, 1, 2].sum() / step_sums[:, 1, 0].sum())
    a_mechanism = comparisons[1]["variance_ratio_geomean"]
    gates = {
        "finite_positive_inputs_and_denominators": True,
        "projected_mean_integrity": integrity["all_passed"],
        "A_mean_vs_random_variance_geomean_below_1": a_mechanism < 1.0,
        "B_variance_geomean_at_most_090_each_control": all(x["variance_ratio_geomean"] <= 0.90 for x in b_results),
        "B_bootstrap_upper_below_1_each_control": all(x["bootstrap_upper"] < 1.0 for x in b_results),
        "B_best_control_case_wins": int(win_mask.sum()) >= min_case_wins,
        "B_total_step_time_vs_independent_at_most_110": time_ratio <= 1.10,
        "B_variance_times_time_geomean_at_most_090_each_control": all(x["variance_times_step_time_sum_ratio_geomean"] <= 0.90 for x in b_results),
    }
    gates["all_passed"] = all(gates.values())
    result.update(comparisons=comparisons, gates=gates,
                  decision="pass_local_mechanism_screen_only" if gates["all_passed"] else "stop_exact_mean_pairing_proposal",
                  B_best_control_win_count=int(win_mask.sum()),
                  B_best_control_win_cases=np.flatnonzero(win_mask).tolist(),
                  B_required_case_wins=min_case_wins,
                  B_total_mean_vs_independent_step_time_ratio=time_ratio,
                  A_mean_vs_random_variance_geomean=a_mechanism)
    return result


def evaluate_gradients(gradients, timings):
    """Only production scoring API. No clean labels, images or weights accepted."""
    if np.shape(gradients) != SHAPE or np.shape(timings) != SHAPE[:-1]:
        raise ValueError("production contract requires gradients [8,2,4,64,5313] and timings [8,2,4,64]")
    result = _analyze(gradients, timings, bootstrap_draws=BOOTSTRAP_DRAWS, min_case_wins=6)
    result["production_contract"] = True
    # Fail closed if any unexpected arithmetic path produced an unpublishable
    # NaN/Inf; callers never receive a purportedly valid non-JSON summary.
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradients", type=Path, required=True, help="Private float32 .npy")
    parser.add_argument("--timings", type=Path, required=True, help="Private positive float .npy")
    parser.add_argument("--output", type=Path, required=True, help="New public summary JSON; refuses overwrite")
    args = parser.parse_args()
    # NPY mmap avoids ZIP decompression and never enables object deserialization.
    g = np.load(args.gradients, allow_pickle=False, mmap_mode="r")
    t = np.load(args.timings, allow_pickle=False, mmap_mode="r")
    result = evaluate_gradients(g, t)
    result["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.output.open("x") as output:
        output.write(encoded)
    return 0 if result["status"] == "analyzed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
