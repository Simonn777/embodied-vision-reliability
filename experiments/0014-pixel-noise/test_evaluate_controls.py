"""Artificial output-chain and metric-gate tests; no dataset or native call."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import evaluate_controls as ev


def put(path, obj):
    raw = json.dumps(obj).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    source = {"experiments/0014-pixel-noise/predict_pair.py": "worker",
              "experiments/0014-pixel-noise/refine.py": "kernel"}
    monkeypatch.setattr(ev.runner, "bindings", lambda: source)
    originals, truth_rows, seal_rows = [], [], []
    for i in range(16):
        key = "synthetic-%d" % i
        pose = {"R": np.eye(3).tolist(), "t": [1., 0., 0.]}
        old = {"estimate": pose, "library": {}, "calibration": {}}
        old_row = {"key": key, "sha256": "original%d" % i}
        originals.append((old_row, old))
        record = {"schema": "evr0014.prediction.v1", "status": "completed",
                  "pair_index": i, "key": key, "library": {}, "calibration": {},
                  "source_sha256": "worker", "kernel_sha256": "kernel",
                  "baseline_seal_sha256": "baseline", "baseline_prediction_sha256": old_row["sha256"],
                  "estimates": {mode: copy.deepcopy(pose) for mode in ("native", "N", "P")},
                  "objectives": {mode: {"N": 10., "P": 10.} for mode in ev.MODES},
                  "objective_gradients_infinity": {mode: {"N": 1e-8, "P": 1e-8} for mode in ev.MODES},
                  "stationarity": {mode: {loss: {"valid": True, "relative_predicted_decrease": 1e-12}
                      for loss in ("N", "P")} for mode in ev.MODES},
                  "kernel_results": {mode: {"termination_reason": "gradient_tolerance",
                      "iteration_count": 2, "accepted_steps": 1} for mode in ("N", "P")},
                  "timing_seconds": {"native": .01, "N": .02, "P": .02}}
        path = tmp_path / ("pair-%02d.json" % i)
        digest = put(path, record)
        seal_rows.append({"pair_index": i, "key": key, "path": path.name,
                          "sha256": digest, "status": "completed"})
        truth_rows.append({"pair_index": i, "key": key, "ground_truth": {
            "R_original": pose["R"], "t_original": pose["t"]}})
    monkeypatch.setattr(ev.metrics, "verify_seal", lambda *a: (
        {"dataset_sha256": "data"}, {}, originals, "manifest", "baseline"))
    seal = {"schema": "evr0014.seal.v1", "complete": True, "eligible_for_evaluation": True,
            "source_sha256": source, "manifest_sha256": "manifest",
            "baseline_seal_sha256": "baseline", "dataset_sha256_declared": "data", "rows": seal_rows}
    seal_path = tmp_path / "seal.json"
    put(seal_path, seal)
    truth_path = tmp_path / "saved-labels.json"
    digest = put(truth_path, {"seal_sha256": "baseline", "rows": truth_rows})
    monkeypatch.setattr(ev, "BASELINE_EVALUATION_SHA", digest)
    return {"root": tmp_path, "seal": seal, "seal_path": seal_path,
            "truth_path": truth_path, "output": tmp_path / "result.json"}


def run(case):
    return ev.evaluate("synthetic-manifest", "synthetic-baseline", case["seal_path"],
                       case["truth_path"], case["output"])


def test_complete_all_pair_metrics_and_no_improvement_claim(case):
    result = run(case)
    assert result["coverage"]["completed_all_branches"] == 16
    assert result["all_pair_interpretation_gate"]["passed"]
    for mode in ev.MODES:
        assert result["auc"][mode]["auc_percent"] == {"5": 100., "10": 100., "20": 100.}
    assert len(result["rows"]) == 16
    with pytest.raises(FileExistsError):
        run(case)


@pytest.mark.parametrize("defect", ["incomplete", "record_hash", "source", "pose"])
def test_entire_cohort_checked_before_saved_labels(case, monkeypatch, defect):
    seal = case["seal"]
    if defect == "incomplete":
        seal["complete"] = False
    else:
        path = case["root"] / seal["rows"][-1]["path"]
        record = json.loads(path.read_text())
        if defect == "record_hash":
            record["key"] = "wrong"
            put(path, record)
        else:
            if defect == "source": record["kernel_sha256"] = "wrong"
            if defect == "pose": record["estimates"]["P"]["t"] = [0, 0, 0]
            seal["rows"][-1]["sha256"] = put(path, record)
    put(case["seal_path"], seal)
    original = ev.metrics.load_hashed_json
    def guarded(path):
        if Path(path) == case["truth_path"]:
            pytest.fail("labels accessed before complete prediction validation")
        return original(path)
    monkeypatch.setattr(ev.metrics, "load_hashed_json", guarded)
    with pytest.raises(ValueError):
        run(case)


def test_wrong_labels_rejected(case):
    case["truth_path"].write_text('{"rows":[]}')
    with pytest.raises(ValueError, match="saved baseline labels"):
        run(case)


def test_failed_interpretation_keeps_all_rows_and_metrics(case):
    row = case["seal"]["rows"][3]
    path = case["root"] / row["path"]
    record = json.loads(path.read_text())
    record["objectives"]["N"]["N"] = 11.
    record["objective_gradients_infinity"]["P"]["P"] = 1.
    record["stationarity"]["P"]["P"]["relative_predicted_decrease"] = 1e-2
    record["kernel_results"]["P"]["termination_reason"] = "maximum_iterations"
    record["kernel_results"]["P"]["iteration_count"] = 100
    row["sha256"] = put(path, record)
    put(case["seal_path"], case["seal"])
    result = run(case)
    assert not result["all_pair_interpretation_gate"]["passed"]
    assert result["all_pair_interpretation_gate"]["passed_pairs"] == 15
    assert all(result["auc"][mode]["count"] == 16 for mode in ev.MODES)
    assert len(result["rows"][3]["interpretation_gate"]["reasons"]) == 4
