"""Thresholds and sequence-clustered evaluation for a frozen predictor."""

import math

import numpy as np

FAILURE_TOLERANCE = 0.025
METHODS = ("static", "temporal", "random", "accept_all")


def ranked_score(row, method):
    if method == "random":
        return (0., row["tie_key"])
    if method == "oracle":
        return (row["error_norm"], row["tie_key"])
    return (row[f"score_{method}"], row["tie_key"])


def calibrate(rows, method, target_coverage=0.8):
    if not rows or not 0 < target_coverage <= 1:
        raise ValueError("Calibration needs nonempty rows and coverage in (0,1]")
    if any(row["partition"] != "cal" or row["condition"] != "clean" for row in rows):
        raise ValueError("Thresholds can only use clean calibration rows")
    if method not in ("static", "temporal", "random"):
        raise ValueError("Only deployable scored methods are calibrated")
    valid = [row for row in rows if row["detected"]]
    if not valid:
        return {"score": None, "tie_key": None, "calibration_total": len(rows), "calibration_accepted": 0,
                "target_reachable": False, "realized_calibration_coverage": 0.}
    rank = min(len(valid), math.ceil(target_coverage * len(rows))) - 1
    selected = sorted(ranked_score(row, method) for row in valid)[rank]
    return {"score": selected[0], "tie_key": selected[1], "calibration_total": len(rows), "calibration_accepted": rank + 1,
            "target_reachable": len(valid) >= math.ceil(target_coverage * len(rows)),
            "realized_calibration_coverage": (rank + 1) / len(rows)}


def accepted(row, method, threshold=None):
    if not row["detected"]:
        return False
    if method == "accept_all":
        return True
    if threshold is None or threshold["score"] is None:
        return False
    return ranked_score(row, method) <= (threshold["score"], threshold["tie_key"])


def risk_coverage_curve(rows, method):
    """Descriptive AURC ends at detectable coverage; missing rows remain total."""
    if not rows:
        return {"coverage": [], "risk": [], "aurc_reachable": None, "max_coverage": 0.}
    valid = sorted((row for row in rows if row["detected"]), key=lambda row: ranked_score(row, method))
    if not valid:
        return {"coverage": [], "risk": [], "aurc_reachable": None, "max_coverage": 0.}
    failure = np.asarray([row["error_norm"] > FAILURE_TOLERANCE for row in valid])
    risk = np.cumsum(failure) / np.arange(1, len(valid) + 1)
    coverage = np.arange(1, len(valid) + 1) / len(rows)
    return {"coverage": coverage.tolist(), "risk": risk.tolist(),
            "aurc_reachable": float(risk.sum() / len(rows)), "max_coverage": float(coverage[-1])}


def summarize(rows, method, threshold=None):
    if not rows:
        raise ValueError("Evaluation requires rows")
    valid = [row for row in rows if row["detected"]]
    selected = [row for row in rows if accepted(row, method, threshold)]
    errors = np.array([row["error_norm"] for row in valid], dtype=float)
    selected_errors = np.array([row["error_norm"] for row in selected], dtype=float)
    total = len(rows)
    return {
        "frames": total, "sequences": len({row["sequence_id"] for row in rows}),
        "detected": len(valid), "missing": total - len(valid), "missing_rate": (total - len(valid)) / total,
        "valid_mean_error_norm": float(errors.mean()) if len(errors) else None,
        "valid_p90_error_norm": float(np.quantile(errors, .9)) if len(errors) else None,
        "all_frame_failure_rate": (total - len(valid) + int(np.sum(errors > FAILURE_TOLERANCE))) / total,
        "accepted": len(selected), "coverage": len(selected) / total,
        "selected_mean_error_norm": float(selected_errors.mean()) if len(selected_errors) else None,
        "selected_failure_rate": float(np.mean(selected_errors > FAILURE_TOLERANCE)) if len(selected_errors) else None,
    }


def sequence_totals(rows, thresholds):
    """Export additive counts for independently checking pooled estimates."""
    grouped = {}
    for row in rows:
        key = (row["partition"], row["sequence_id"], row["condition"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (partition, sequence_id, condition), group in sorted(grouped.items()):
        valid = [row for row in group if row["detected"]]
        for method in METHODS:
            selected = [row for row in group if accepted(row, method, thresholds.get(method))]
            output.append({
                "partition": partition, "sequence_id": sequence_id, "condition": condition,
                "method": method, "frames": len(group), "detected": len(valid),
                "missing": len(group) - len(valid),
                "valid_failure_count": sum(row["error_norm"] > FAILURE_TOLERANCE for row in valid),
                "valid_error_sum_norm": sum(row["error_norm"] for row in valid),
                "accepted": len(selected),
                "accepted_failure_count": sum(row["error_norm"] > FAILURE_TOLERANCE for row in selected),
                "accepted_error_sum_norm": sum(row["error_norm"] for row in selected),
            })
    return output


def cluster_resamples(sequence_ids, repeats=1000, seed=20261017):
    ids = np.asarray(sorted(sequence_ids), dtype=np.int64)
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids) or repeats < 1:
        raise ValueError("Bootstrap requires unique sequences and positive repeats")
    return np.random.default_rng(seed).choice(ids, size=(repeats, len(ids)), replace=True)


def _interval(values):
    valid = np.asarray(values)[np.isfinite(values)]
    required = math.ceil(.95 * len(values))
    return {"low": float(np.quantile(valid, .025)) if len(valid) >= required else None,
            "high": float(np.quantile(valid, .975)) if len(valid) >= required else None,
            "defined_resamples": len(valid), "required_defined_resamples": required}


def paired_bootstrap(rows, thresholds, repeats=1000, seed=20261017):
    """One shared draw of whole base sequences across all conditions/methods."""
    if not rows or any(row["partition"] != "test" for row in rows):
        raise ValueError("Bootstrap requires test rows")
    sequence_ids = sorted({row["sequence_id"] for row in rows})
    draws = cluster_resamples(sequence_ids, repeats, seed)
    id_to_index = {value: index for index, value in enumerate(sequence_ids)}
    indices = np.array([[id_to_index[value] for value in sample] for sample in draws])
    result = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        if {row["sequence_id"] for row in subset} != set(sequence_ids):
            raise ValueError("Every condition must contain the same base sequences")
        result[condition] = {}
        risk_samples = {}
        coverage_samples = {}
        for method in METHODS:
            totals = np.zeros((len(sequence_ids), 4), dtype=float)
            for row in subset:
                slot = id_to_index[row["sequence_id"]]
                totals[slot, 0] += 1
                if accepted(row, method, thresholds.get(method)):
                    totals[slot, 1] += 1
                    totals[slot, 2] += row["error_norm"] > FAILURE_TOLERANCE
                    totals[slot, 3] += row["error_norm"]
            aggregate = totals[indices].sum(axis=1)
            coverage = aggregate[:, 1] / aggregate[:, 0]
            risks = np.divide(aggregate[:, 2], aggregate[:, 1], out=np.full(repeats, np.nan), where=aggregate[:, 1] > 0)
            errors = np.divide(aggregate[:, 3], aggregate[:, 1], out=np.full(repeats, np.nan), where=aggregate[:, 1] > 0)
            risk_samples[method] = risks
            coverage_samples[method] = coverage
            result[condition][method] = {"coverage": _interval(coverage), "selected_failure_rate": _interval(risks),
                                         "selected_mean_error_norm": _interval(errors)}
        result[condition]["temporal_minus_static_selected_failure"] = _interval(risk_samples["temporal"] - risk_samples["static"])
        result[condition]["temporal_minus_static_coverage"] = _interval(coverage_samples["temporal"] - coverage_samples["static"])
    return {"unit": "base sequence", "repeats": repeats, "seed": seed,
            "sequence_count": len(sequence_ids), "paired_across_conditions_and_methods": True, "intervals": result}
