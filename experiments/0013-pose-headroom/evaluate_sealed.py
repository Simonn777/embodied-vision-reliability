#!/usr/bin/env python3
"""Evaluate a complete sealed 0013 prediction cohort; never run estimation.

Metric convention: jastermark/DMS, commit
7d5b228de9b6fab460f43e466fb0ed8565ffe137, evaluation/metrics.py and dataloader.py.
Only the selected pair groups' R and t datasets are read, after seal validation.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PAIR_COUNT = 16
THRESHOLDS = (5.0, 10.0, 20.0)
ROTATION_ATOL = 1e-6
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(__file__).resolve().parent
SOURCE_NAMES = {"predict_pair.py", "evaluate_sealed.py", "run_baseline.py"}


def json_safe(value):
    """Recursively convert NumPy values, rejecting nonfinite or unsupported data."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: json_safe(item) for key, item in value.items()}
    raise ValueError("unsupported JSON type: " + type(value).__name__)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def load_hashed_json(path):
    """Hash and decode the same bytes, rather than reopening between the two."""
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=_unique_object)
    return json_safe(value), hashlib.sha256(raw).hexdigest()


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + " is not a numeric array") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(label + " has invalid shape or nonfinite entries")
    return result


def valid_rotation(value):
    """Validate a predicted SO(3) matrix without silently repairing it."""
    R = _array(value, (3, 3), "predicted rotation")
    with np.errstate(over="ignore", invalid="ignore"):
        gram = R @ R.T
        determinant = float(np.linalg.det(R))
    if (not math.isfinite(determinant)
            or not np.allclose(gram, np.eye(3), rtol=0, atol=ROTATION_ATOL)
            or abs(determinant - 1.0) > ROTATION_ATOL):
        raise ValueError("predicted rotation is not a valid proper rotation")
    return R


def project_gt_rotation(value):
    """Apply exactly U @ Vt; reject a reflection/rank defect, never flip a sign."""
    raw = _array(value, (3, 3), "ground-truth rotation")
    try:
        U, singular_values, Vt = np.linalg.svd(raw)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ground-truth rotation SVD failed") from exc
    projected = U @ Vt
    determinant = float(np.linalg.det(projected))
    if (not np.isfinite(singular_values).all() or singular_values[-1] <= 0
            or not math.isfinite(determinant) or determinant <= 0
            or not np.allclose(projected @ projected.T, np.eye(3), rtol=0, atol=1e-10)):
        raise ValueError("ground-truth rotation projection is invalid/improper")
    change = math.hypot(*(projected - raw).ravel().tolist())
    if not math.isfinite(change):
        raise ValueError("ground-truth rotation projection change is nonfinite")
    return projected, {
        "projection": "U @ Vt, no determinant-sign correction",
        "projection_frobenius_change": change,
        "singular_values": singular_values,
        "projected_determinant": determinant,
    }


def unit_translation(value):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape not in ((3,), (3, 1), (1, 3)) or not np.isfinite(vector).all():
        raise ValueError("translation has invalid shape or nonfinite entries")
    vector = vector.reshape(3)
    norm = math.hypot(*vector.tolist())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("translation norm must be finite and positive")
    return vector / norm, norm


def pose_errors(predicted_R, predicted_t, ground_truth_R, ground_truth_t):
    """Degrees, directed translation, with the documented GT preprocessing."""
    R_pred = valid_rotation(predicted_R)
    t_pred, pred_norm = unit_translation(predicted_t)
    R_gt, projection = project_gt_rotation(ground_truth_R)
    t_gt, gt_norm = unit_translation(ground_truth_t)
    rotation = math.degrees(math.acos(float(np.clip(
        (np.trace(R_pred @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0))))
    translation = math.degrees(math.acos(float(np.clip(np.dot(t_pred, t_gt), -1.0, 1.0))))
    return {
        "rotation_degrees": rotation,
        "translation_degrees": translation,
        "max_degrees": max(rotation, translation),
    }, {
        "R_original": ground_truth_R, "t_original": ground_truth_t,
        "R_projected": R_gt, "t_unit": t_gt,
        "translation_original_norm": gt_norm,
        "prediction_translation_norm": pred_norm,
        **projection,
    }


def dms_auc(errors, thresholds=THRESHOLDS):
    """DMS's left-search/trapezoid fraction convention, including threshold ties."""
    values = np.asarray(errors, dtype=np.float64)
    if (values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
            or np.any(values < 0)):
        raise ValueError("AUC requires a nonempty finite nonnegative error vector")
    sorted_values = np.concatenate(([0.0], np.sort(values)))
    recall = np.arange(len(values) + 1, dtype=np.float64) / len(values)
    result = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("AUC thresholds must be finite and positive")
        stop = int(np.searchsorted(sorted_values, threshold, side="left"))
        x = np.concatenate((sorted_values[:stop], [threshold]))
        y = np.concatenate((recall[:stop], [recall[stop - 1]]))
        # Explicit trapezoids support both NumPy 1.x and 2.x.
        area = np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5)
        result.append(float(area / threshold))
    return result


def _auc_summary(errors):
    if not errors:
        return {"count": 0, "auc_fraction": None, "auc_percent": None,
                "count_strictly_below_threshold": None}
    fractions = dms_auc(errors)
    keys = [str(int(t)) for t in THRESHOLDS]
    return {
        "count": len(errors),
        "auc_fraction": dict(zip(keys, fractions)),
        "auc_percent": dict(zip(keys, [100.0 * value for value in fractions])),
        "count_strictly_below_threshold": {
            key: sum(error < threshold for error in errors)
            for key, threshold in zip(keys, THRESHOLDS)
        },
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_seal(manifest_path, seal_path):
    """Verify the entire cohort before a caller is permitted to open HDF5."""
    manifest, manifest_hash = load_hashed_json(manifest_path)
    seal, seal_hash = load_hashed_json(seal_path)
    _require(isinstance(manifest, dict) and isinstance(seal, dict),
             "manifest and seal must be JSON objects")
    keys = manifest.get("selected_pair_keys", [])
    _require(isinstance(keys, list)
             and all(isinstance(key, str) and key and "/" not in key for key in keys),
             "invalid selected pair key")
    _require(len(keys) == PAIR_COUNT and len(set(keys)) == PAIR_COUNT,
             "manifest must select exactly 16 distinct pair keys")
    _require(seal.get("schema") == "evr0013.seal.v1", "unexpected seal schema")
    _require(seal.get("complete") is True, "prediction seal is incomplete")
    _require(seal.get("eligible_for_evaluation") is True, "seal is not evaluation eligible")
    _require(seal.get("manifest_sha256") == manifest_hash, "manifest hash mismatch")
    declared = manifest.get("dataset_sha256")
    _require(isinstance(declared, str) and len(declared) == 64
             and all(c in "0123456789abcdef" for c in declared),
             "invalid manifest dataset hash")
    _require(seal.get("dataset_sha256_declared") == declared,
             "seal dataset declaration mismatch")
    rows = seal.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == PAIR_COUNT,
             "seal must contain all 16 prediction rows")
    source_hashes = seal.get("source_sha256", {})
    _require(isinstance(source_hashes, dict) and set(source_hashes) == SOURCE_NAMES,
             "seal must bind the exact three 0013 source files")
    for relative, expected_hash in source_hashes.items():
        path = SOURCE_DIR / relative
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash,
                 "source hash mismatch: " + relative)
    protocol_path = REPO_ROOT / "protocols" / "0013-baseline-pose-headroom.md"
    _require(hashlib.sha256(protocol_path.read_bytes()).hexdigest() == seal.get("protocol_sha256"),
             "protocol hash mismatch")
    records = []
    seen_paths = set()
    for index, (key, row) in enumerate(zip(keys, rows)):
        _require(isinstance(row, dict) and type(row.get("pair_index")) is int
                 and row.get("pair_index") == index and row.get("key") == key,
                 "seal row order/key/index mismatch")
        relative = row.get("path")
        _require(isinstance(relative, str) and Path(relative).name == relative,
                 "prediction path must be a filename beside the seal")
        path = (Path(seal_path).resolve().parent / relative).resolve()
        _require(path not in seen_paths, "duplicate prediction path")
        seen_paths.add(path)
        prediction, prediction_hash = load_hashed_json(path)
        _require(prediction_hash == row.get("sha256"), "prediction hash mismatch")
        _require(isinstance(prediction, dict)
                 and prediction.get("schema") == "evr0013.prediction.v1",
                 "unexpected prediction schema")
        _require(type(prediction.get("pair_index")) is int
                 and prediction.get("pair_index") == index and prediction.get("key") == key,
                 "prediction index/key mismatch")
        _require(prediction.get("manifest_sha256") == manifest_hash,
                 "prediction manifest hash mismatch")
        _require(prediction.get("dataset_sha256_declared") == manifest.get("dataset_sha256"),
                 "prediction dataset declaration mismatch")
        _require(prediction.get("source_sha256") == source_hashes["predict_pair.py"],
                 "prediction worker source hash mismatch")
        status = row.get("status")
        _require(status in ("completed", "estimator_error", "timeout"), "nonterminal seal status")
        _require(prediction.get("status") == status, "prediction/seal status mismatch")
        if status == "completed":
            estimate = prediction.get("estimate", {})
            valid_rotation(estimate.get("R"))
            unit_translation(estimate.get("t"))
        records.append((row, prediction))
    return manifest, seal, records, manifest_hash, seal_hash


def evaluate(data_path, manifest_path, seal_path, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("evaluation output already exists: " + str(output_path))
    manifest, seal, records, manifest_hash, seal_hash = verify_seal(manifest_path, seal_path)
    _require(Path(data_path).stat().st_size == manifest.get("dataset_bytes"),
             "evaluation dataset size differs from sealed input")
    digest = hashlib.sha256()
    with Path(data_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    _require(digest.hexdigest() == manifest["dataset_sha256"],
             "evaluation dataset hash differs from sealed input")
    import h5py  # No HDF5 handle is opened before every seal/prediction check above.
    rows, successful_errors, penalized_errors = [], [], []
    with h5py.File(data_path, "r") as data:
        for row, prediction in records:
            pair = data[row["key"]]
            R_raw, t_raw = pair["R"][()], pair["t"][()]
            # Even failed predictions require valid GT; never invent missing labels.
            R_gt, projection = project_gt_rotation(R_raw)
            t_gt, t_norm = unit_translation(t_raw)
            gt = {"R_original": R_raw, "t_original": t_raw,
                  "R_projected": R_gt, "t_unit": t_gt,
                  "translation_original_norm": t_norm, **projection}
            errors = None
            penalty = 180.0
            if row["status"] == "completed":
                estimate = prediction["estimate"]
                errors, gt = pose_errors(estimate["R"], estimate["t"], R_raw, t_raw)
                penalty = errors["max_degrees"]
                successful_errors.append(penalty)
            penalized_errors.append(penalty)
            rows.append({"pair_index": row["pair_index"], "key": row["key"],
                         "status": row["status"], "seal_row": row,
                         "prediction": prediction, "ground_truth": gt,
                         "errors": errors, "all_pair_penalized_error_degrees": penalty})
    result = json_safe({
        "schema": "evr0013.evaluation.v1", "manifest_sha256": manifest_hash,
        "seal_sha256": seal_hash, "source_sha256": seal["source_sha256"],
        "protocol_sha256": seal["protocol_sha256"],
        "dataset_sha256_declared": manifest["dataset_sha256"],
        "ground_truth_reads": "Only R/t for the 16 sealed selected keys",
        "metric_convention": "DMS directed translation; U @ Vt GT rotation; trapezoidal AUC fractions",
        "prediction_rotation_validation_atol": ROTATION_ATOL,
        "coverage": {"selected": PAIR_COUNT, "attempted": PAIR_COUNT,
                     "successful": len(successful_errors),
                     "successful_fraction": len(successful_errors) / PAIR_COUNT,
                     "status_counts": dict(Counter(row["status"] for row in rows))},
        "primary_all_pair_coverage_penalized": {
            "failure_penalty_degrees": 180.0,
            "interpretation": "Nonscientific resource/missing-estimate penalty; not an observed pose error",
            **_auc_summary(penalized_errors)},
        "secondary_successful_only": _auc_summary(successful_errors),
        "uncertainty": "Descriptive one-scene development cohort; no population confidence interval",
        "rows": rows,
    })
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "seal", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.manifest, args.seal, args.output)


if __name__ == "__main__":
    main()
