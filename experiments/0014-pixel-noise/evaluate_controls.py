"""Score sealed known controls against the frozen, already-read 0013 labels."""
import argparse
import json
from pathlib import Path

import run_controls as runner

BASELINE_EVALUATION_SHA = "f8fe46392810b78d485a8dd90ae1849eae6491eaba620d2c9454b6c0f15cb5a4"
MODES = ("original", "native", "N", "P")
metrics = runner.baseline_seal


def interpretation_gate(record):
    costs, gradients = record["objectives"], record["objective_gradients_infinity"]
    reasons = []
    if costs["N"]["N"] > costs["native"]["N"] + 1e-6 * max(1., costs["native"]["N"]):
        reasons.append("N_is_not_objective_competitive_with_native")
    details = {}
    for mode in ("N", "P"):
        result = record["kernel_results"][mode]
        relative_gradient = gradients[mode][mode] / max(1., costs[mode][mode])
        if costs[mode][mode] > costs["original"][mode]:
            reasons.append(mode + "_objective_increased")
        if result["termination_reason"] == "maximum_iterations" or result["iteration_count"] >= 100:
            reasons.append(mode + "_exhausted_budget")
        stationarity = record["stationarity"][mode][mode]
        if (stationarity.get("valid") is not True
                or stationarity.get("relative_predicted_decrease") is None
                or not 0 <= stationarity["relative_predicted_decrease"] <= 1e-8):
            reasons.append(mode + "_GN_stationarity_failed")
        details[mode] = {"normalized_gradient_infinity": relative_gradient,
                         "termination_reason": result["termination_reason"],
                         "iteration_count": result["iteration_count"],
                         "accepted_steps": result["accepted_steps"],
                         "local_GN_stationarity": stationarity}
    return {"passed": not reasons, "reasons": reasons, "details": details}


def verify_predictions(manifest_path, original_seal_path, seal_path):
    manifest, baseline, originals, mh, bh = metrics.verify_seal(manifest_path, original_seal_path)
    seal, sh = metrics.load_hashed_json(seal_path)
    if (seal.get("schema") != "evr0014.seal.v1" or seal.get("complete") is not True
            or seal.get("eligible_for_evaluation") is not True
            or seal.get("manifest_sha256") != mh or seal.get("baseline_seal_sha256") != bh
            or seal.get("dataset_sha256_declared") != manifest["dataset_sha256"]
            or seal.get("source_sha256") != runner.bindings()):
        raise ValueError("continuation seal is incomplete or incorrectly bound")
    if len(seal.get("rows", [])) != 16:
        raise ValueError("all16 continuation rows are required")
    records = []
    seen = set()
    for i, (row, (old_row, old_record)) in enumerate(zip(seal["rows"], originals)):
        relative = row.get("path")
        if (row.get("pair_index") != i or row.get("key") != old_row["key"]
                or row.get("status") != "completed" or not isinstance(relative, str)
                or Path(relative).name != relative or relative in seen):
            raise ValueError("continuation row identity/path differs")
        seen.add(relative)
        record, digest = metrics.load_hashed_json(Path(seal_path).parent / relative)
        if (digest != row.get("sha256") or record.get("pair_index") != i
                or record.get("key") != old_row["key"] or record.get("status") != "completed"
                or record.get("schema") != "evr0014.prediction.v1"
                or record.get("baseline_prediction_sha256") != old_row["sha256"]
                or record.get("baseline_seal_sha256") != bh
                or record.get("source_sha256") != seal["source_sha256"]["experiments/0014-pixel-noise/predict_pair.py"]
                or record.get("kernel_sha256") != seal["source_sha256"]["experiments/0014-pixel-noise/refine.py"]):
            raise ValueError("continuation prediction binding differs")
        if record.get("library") != old_record["library"] or record.get("calibration") != old_record["calibration"]:
            raise ValueError("continuation library/calibration differs")
        for mode in ("native", "N", "P"):
            metrics.valid_rotation(record["estimates"][mode]["R"])
            metrics.unit_translation(record["estimates"][mode]["t"])
        interpretation_gate(record)  # Required finite fields before label-file access.
        records.append((old_record, record))
    return seal, sh, bh, records


def evaluate(manifest, original_seal, seal_path, baseline_evaluation, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("evaluation output already exists")
    seal, sh, bh, records = verify_predictions(manifest, original_seal, seal_path)
    # Labels are read only after the entire continuation cohort is verified.
    truth, truth_hash = metrics.load_hashed_json(baseline_evaluation)
    if (truth_hash != BASELINE_EVALUATION_SHA or truth.get("seal_sha256") != bh
            or len(truth.get("rows", [])) != 16):
        raise ValueError("saved baseline labels are not the frozen0013 artifact")
    errors = {mode: [] for mode in MODES}
    rows = []
    for i, ((original, record), old_evaluation) in enumerate(zip(records, truth["rows"])):
        if old_evaluation["key"] != record["key"] or old_evaluation["pair_index"] != i:
            raise ValueError("saved label identity differs")
        gt = old_evaluation["ground_truth"]
        estimates = {"original": original["estimate"], **record["estimates"]}
        row_errors = {}
        for mode in MODES:
            pose = estimates[mode]
            values, _ = metrics.pose_errors(pose["R"], pose["t"], gt["R_original"], gt["t_original"])
            row_errors[mode] = values
            errors[mode].append(values["max_degrees"])
        disagreement, _ = metrics.pose_errors(estimates["N"]["R"], estimates["N"]["t"],
                                                estimates["native"]["R"], estimates["native"]["t"])
        rows.append({"pair_index": i, "key": record["key"], "errors": row_errors,
                     "N_native_pose_disagreement_degrees": disagreement,
                     "interpretation_gate": interpretation_gate(record),
                     "objectives": record["objectives"],
                     "objective_gradients_infinity": record["objective_gradients_infinity"],
                     "timing_seconds": record["timing_seconds"]})
    auc = {mode: metrics._auc_summary(errors[mode]) for mode in MODES}
    passed = sum(row["interpretation_gate"]["passed"] for row in rows)
    result = {"schema": "evr0014.evaluation.v1", "source_sha256": seal["source_sha256"],
              "seal_sha256": sh, "baseline_evaluation_sha256": truth_hash,
              "coverage": {"selected": 16, "completed_all_branches": 16},
              "all_pair_interpretation_gate": {"passed": passed == 16, "passed_pairs": passed,
                  "required_pairs": 16, "failure_means": "convergence confounding; no rows removed"},
              "auc": auc,
              "P_minus_control_auc_percentage_points": {
                  mode: {k: auc["P"]["auc_percent"][k] - auc[mode]["auc_percent"][k]
                         for k in ("5", "10", "20")} for mode in ("original", "native", "N")},
              "rows": rows,
              "scope": "Known pixel-Sampson control on reused one-scene development; no novelty or population inference"}
    runner.infrastructure.write_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline-seal", "seal", "baseline-evaluation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.baseline_seal, args.seal, args.baseline_evaluation, args.output)
