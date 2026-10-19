#!/usr/bin/env python3
"""Evaluate sealed search candidates; selection is fixed before any GT access."""

import argparse
from collections import Counter
import importlib.util
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PAIR_COUNT = 64
BACKENDS = ("old_norm", "current_norm", "current_ts")
SEEDS = (0, 1, 2, 3)
BUDGETS = (0.1, 0.25, 0.5, 1.0, 2.0)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = _load_module("evr0016_metrics_0013", HERE.parent / "0013-pose-headroom" / "evaluate_sealed.py")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonnegative(value, label):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "invalid finite nonnegative " + label)
    return float(value)


def select_candidate(candidates, budget=None):
    """Saved pixel MSAC only; an exact budget boundary is eligible; ties use seed."""
    if budget is not None:
        _nonnegative(budget, "budget")
    eligible = [c for c in candidates if c["status"] == "completed"
                and (budget is None or c["charged_end_seconds"] <= budget)]
    return min(eligible, key=lambda c: (c["final_scores"]["pixel_msac"], c["seed"])) if eligible else None


def prepare_records(records, selected_pair_keys):
    """Validate all candidates and freeze every deployable choice before HDF5."""
    keys = selected_pair_keys
    _require(isinstance(keys, list) and len(keys) == PAIR_COUNT
             and len(set(keys)) == PAIR_COUNT
             and all(isinstance(k, str) and k and "/" not in k for k in keys),
             "expected 64 unique manifest pair keys")
    _require(isinstance(records, list) and len(records) == PAIR_COUNT * len(BACKENDS),
             "expected all 192 sealed prediction records")
    # Refuse nonfinite values anywhere, including error receipts.
    metrics.json_safe(records)
    seen = set()
    prepared = []
    for record in records:
        index, backend = record.get("pair_index"), record.get("backend")
        _require(type(index) is int and 0 <= index < PAIR_COUNT and backend in BACKENDS,
                 "invalid prediction index/backend")
        _require(record.get("key") == keys[index] and record.get("status") == "completed",
                 "prediction key/status mismatch")
        identity = (backend, index)
        _require(identity not in seen, "duplicate backend/pair record")
        seen.add(identity)
        candidates = record.get("candidates")
        _require(isinstance(candidates, list) and len(candidates) == len(SEEDS),
                 "expected four terminal candidates")
        _require([c.get("seed") for c in candidates] == list(SEEDS)
                 and all(type(c.get("seed")) is int for c in candidates),
                 "candidates must have fixed ordered seeds 0,1,2,3")
        input_seconds = _nonnegative(record.get("input_seconds"), "input_seconds")
        charged_total = _nonnegative(record.get("charged_total_seconds"), "charged_total_seconds")
        cumulative = input_seconds
        last_end = 0.0
        for candidate in candidates:
            _require(candidate.get("status") in ("completed", "estimator_error"),
                     "candidate has nonterminal status")
            for field in ("call_seconds", "score_seconds", "charged_end_seconds"):
                _nonnegative(candidate.get(field), field)
            end = candidate["charged_end_seconds"]
            _require(end >= last_end, "charged completion times are not monotone")
            cumulative += candidate["call_seconds"] + candidate["score_seconds"]
            _require(math.isclose(end, cumulative, rel_tol=1e-12, abs_tol=1e-9),
                     "charged completion time omits or duplicates work")
            last_end = end
            if candidate["status"] == "completed":
                estimate = candidate.get("estimate", {})
                metrics.valid_rotation(estimate.get("R"))
                metrics.unit_translation(estimate.get("t"))
                scores = candidate.get("final_scores", {})
                _require(isinstance(scores, dict), "missing final score object")
                _nonnegative(scores.get("pixel_msac"), "pixel_msac")
            else:
                _require(not candidate.get("estimate"), "failed candidate contains an estimate")
                _require(candidate["score_seconds"] == 0.0,
                         "estimator_error must have zero score_seconds")
        _require(math.isclose(charged_total, last_end, rel_tol=1e-12, abs_tol=1e-9),
                 "charged_total_seconds differs from final charged completion")
        selections = {"seed0": candidates[0] if candidates[0]["status"] == "completed" else None}
        for budget in BUDGETS:
            selections["pixel_msac_budget_" + str(budget)] = select_candidate(candidates, budget)
        selections["pixel_msac_full4"] = select_candidate(candidates)
        prepared.append({"record": record, "selections": selections})
    _require(len(seen) == PAIR_COUNT * len(BACKENDS), "incomplete backend/pair Cartesian product")
    return sorted(prepared, key=lambda row: (BACKENDS.index(row["record"]["backend"]),
                                           row["record"]["pair_index"]))


def error_with_folded_translation(estimate, gt):
    error, metadata = metrics.pose_errors(estimate["R"], estimate["t"], gt["R"], gt["t"])
    folded = min(error["translation_degrees"], 180.0 - error["translation_degrees"])
    error["translation_signfolded_degrees"] = folded
    error["max_signfolded_degrees"] = max(error["rotation_degrees"], folded)
    return error, metadata


def candidate_dispersion(candidates):
    """Continuous prediction-only differences; no tuned stable/wrong classifier."""
    successful = [c for c in candidates if c["status"] == "completed"]
    rows = []
    for i, a in enumerate(successful):
        Ra = metrics.valid_rotation(a["estimate"]["R"])
        ta, _ = metrics.unit_translation(a["estimate"]["t"])
        for b in successful[i + 1:]:
            Rb = metrics.valid_rotation(b["estimate"]["R"])
            tb, _ = metrics.unit_translation(b["estimate"]["t"])
            r = math.degrees(math.acos(float(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))
            t = math.degrees(math.acos(float(np.clip(np.dot(ta, tb), -1, 1))))
            rows.append({"seed_a": a["seed"], "seed_b": b["seed"],
                         "rotation_degrees": r, "translation_directed_degrees": t})
    return {"successful_candidate_count": len(successful), "pairwise": rows,
            "max_rotation_degrees": max((r["rotation_degrees"] for r in rows), default=None),
            "max_translation_directed_degrees": max((r["translation_directed_degrees"] for r in rows), default=None)}


def _selection_result(candidate, errors, reason):
    if candidate is None:
        return {"status": "unavailable", "reason": reason, "seed": None,
                "error": None, "penalized_max_degrees": 180.0,
                "penalized_max_signfolded_degrees": 180.0}
    error = errors[candidate["seed"]]
    return {"status": "completed", "reason": None, "seed": candidate["seed"],
            "pixel_msac": candidate["final_scores"]["pixel_msac"],
            "charged_end_seconds": candidate["charged_end_seconds"], "error": error,
            "penalized_max_degrees": error["max_degrees"],
            "penalized_max_signfolded_degrees": error["max_signfolded_degrees"]}


def _summary(rows):
    _require(len(rows) == PAIR_COUNT, "summary denominator must be exactly 64")
    result = {"pair_count": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
              "selected_seed_counts": dict(Counter(str(r["seed"]) for r in rows if r["seed"] is not None))}
    for label, field in (("directed_primary", "penalized_max_degrees"),
                         ("signfolded_secondary", "penalized_max_signfolded_degrees")):
        values = [r[field] for r in rows]
        result[label] = {"auc_fraction": dict(zip(("5", "10", "20"), metrics.dms_auc(values))),
                         "errors_degrees": values,
                         "count_strictly_below": {str(t): sum(e < t for e in values) for t in (5, 10, 20)}}
    return result


def evaluate_records(prepared, ground_truth):
    """GT is used for errors and explicitly nondeployable oracle diagnostics only."""
    rows = []
    for item in prepared:
        record = item["record"]
        gt = ground_truth[record["key"]]
        errors, gt_metadata = {}, None
        candidates_output = []
        for candidate in record["candidates"]:
            if candidate["status"] == "completed":
                errors[candidate["seed"]], metadata = error_with_folded_translation(candidate["estimate"], gt)
                if gt_metadata is None:
                    gt_metadata = {k: v for k, v in metadata.items() if k != "prediction_translation_norm"}
            candidates_output.append({**candidate, "error": errors.get(candidate["seed"])})
        if gt_metadata is None:
            # Validate and disclose GT even when every estimator call failed.
            Rgt, projection = metrics.project_gt_rotation(gt["R"])
            tgt, norm = metrics.unit_translation(gt["t"])
            gt_metadata = {"R_original": gt["R"], "t_original": gt["t"],
                           "R_projected": Rgt, "t_unit": tgt,
                           "translation_original_norm": norm, **projection}
        choices = {name: _selection_result(c, errors, "no completed candidate eligible for this selection")
                   for name, c in item["selections"].items()}
        successful = [c for c in record["candidates"] if c["status"] == "completed"]
        oracle = min(successful, key=lambda c: (errors[c["seed"]]["max_degrees"], c["seed"])) if successful else None
        choices["oracle_best4_nondeployable"] = _selection_result(oracle, errors, "all four candidates failed")
        spreads = [e["max_degrees"] for e in errors.values()]
        rows.append({"pair_index": record["pair_index"], "key": record["key"], "backend": record["backend"],
                     "candidates": candidates_output, "selections": choices, "ground_truth": gt_metadata,
                     "calibration": record.get("calibration"),
                     "input_seconds": record["input_seconds"],
                     "charged_total_seconds": record["charged_total_seconds"],
                     "prediction_only_dispersion": candidate_dispersion(record["candidates"]),
                     "candidate_error_spread": {"min_directed_max_degrees": min(spreads) if spreads else None,
                                                "max_directed_max_degrees": max(spreads) if spreads else None,
                                                "range_directed_max_degrees": max(spreads) - min(spreads) if spreads else None}})
    summaries = {}
    for backend in BACKENDS:
        subset = [r for r in rows if r["backend"] == backend]
        summaries[backend] = {name: _summary([r["selections"][name] for r in subset])
                              for name in subset[0]["selections"]}
    return metrics.json_safe({"rows": rows, "summaries": summaries})


def evaluate(data_path, manifest_path, seal_path, output_path, common_module=None):
    """Seal -> all score/pose validation and choices -> opaque hash -> selected GT."""
    common = common_module or _load_module("evr0016_common_evaluation", HERE / "common.py")
    data_path, output_path = Path(data_path), Path(output_path)
    _require(not output_path.exists(), "output already exists")
    manifest, seal, records = common.verify_seal(manifest_path, seal_path)
    prepared = prepare_records(records, manifest["selected_pair_keys"])
    source_hashes = common.source_hashes()
    manifest_hash, seal_hash = common.sha256(manifest_path), common.sha256(seal_path)
    _require(data_path.stat().st_size == manifest["dataset_bytes"], "dataset size mismatch")
    _require(common.sha256(data_path) == manifest["dataset_sha256"], "dataset hash mismatch")
    import h5py
    ground_truth = {}
    with h5py.File(data_path, "r") as handle:
        for key in manifest["selected_pair_keys"]:
            group = handle[key]
            ground_truth[key] = {"R": group["R"][:], "t": group["t"][:]}
    results = evaluate_records(prepared, ground_truth)
    _require(common.source_hashes() == source_hashes, "source changed during evaluation")
    _require(common.sha256(manifest_path) == manifest_hash and common.sha256(seal_path) == seal_hash,
             "manifest/seal changed during evaluation")
    common.verify_seal(manifest_path, seal_path)
    output = {"schema": "evr0016.search-evaluation.v1", "status": "completed",
              "manifest_sha256": manifest_hash, "seal_sha256": seal_hash,
              "source_sha256": source_hashes, "dataset_sha256": manifest["dataset_sha256"],
              "dataset_bytes": manifest["dataset_bytes"], "pair_count": PAIR_COUNT,
              "backend_count": len(BACKENDS), "budgets_seconds": list(BUDGETS),
              "selection": "minimum saved independent pixel_msac, then seed; frozen before GT access",
              "budget_semantics": "offline prefix replay: charged_end_seconds <= budget; full4 uses all terminal candidates",
              "failure_penalty": "Unavailable selections use 180 degrees in the full 64-pair denominator, including signfolded summaries; this resource/coverage penalty is not measured pose error.",
              "oracle_warning": "oracle_best4_nondeployable uses GT to choose among four candidates. It is secondary diagnostic only; candidate multiplicity creates optimistic selection, not a deployable improvement.",
              "dispersion_warning": "Prediction dispersion and candidate-error spread are descriptive. Low dispersion with error does not establish input bias or its cause; no stable/wrong threshold is fitted.",
              **results}
    common.write_new(output_path, common.json_safe(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "seal", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate(args.data, args.manifest, args.seal, args.output)


if __name__ == "__main__":
    main()
