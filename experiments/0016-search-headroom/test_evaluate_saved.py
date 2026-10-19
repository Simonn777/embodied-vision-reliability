"""Artificial-only tests; no scientific dataset or estimator is used."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("evaluate_saved_tested", HERE / "evaluate_saved.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def fixture():
    keys = ["artificial-%02d" % i for i in range(64)]
    records = []
    for backend in ev.BACKENDS:
        for i, key in enumerate(keys):
            candidates = []
            for seed in ev.SEEDS:
                candidates.append({"seed": seed, "status": "completed",
                    "estimate": {"R": np.eye(3).tolist(), "t": [1., 0., 0.]},
                    "call_seconds": [0.07, 0.13, 0.23, 0.48][seed], "score_seconds": 0.02,
                    "charged_end_seconds": [0.1, 0.25, 0.5, 1.][seed],
                    "final_scores": {"pixel_msac": [5., 3., 3., 2.][seed]}})
            records.append({"pair_index": i, "key": key, "backend": backend,
                            "status": "completed", "candidates": candidates,
                            "input_seconds": 0.01, "charged_total_seconds": 1.0,
                            "calibration": {"synthetic": True}})
    gt = {key: {"R": np.eye(3), "t": np.array([1., 0., 0.])} for key in keys}
    return keys, records, gt


def failed(candidate):
    candidate["status"] = "estimator_error"
    candidate.pop("estimate", None)
    candidate.pop("final_scores", None)
    candidate["error"] = {"type": "SyntheticFailure", "message": "artificial"}
    # Keep this artificial call's total charge unchanged when simulating failure.
    candidate["call_seconds"] += candidate["score_seconds"]
    candidate["score_seconds"] = 0.0


def test_selector_budget_ties_and_error_coverage():
    keys, records, _ = fixture()
    candidates = records[0]["candidates"]
    assert ev.select_candidate(candidates, 0.1)["seed"] == 0
    assert ev.select_candidate(candidates, float(np.nextafter(0.1, 0.0))) is None
    assert ev.select_candidate(candidates, 0.5)["seed"] == 1
    assert ev.select_candidate(candidates)["seed"] == 3
    failed(candidates[3])
    assert ev.select_candidate(candidates)["seed"] == 1
    prepared = ev.prepare_records(records, keys)
    assert prepared[0]["selections"]["pixel_msac_full4"]["seed"] == 1


def test_directed_translation_and_signfold_secondary():
    estimate = {"R": np.eye(3), "t": [-1., 0., 0.]}
    error, _ = ev.error_with_folded_translation(estimate, {"R": np.eye(3), "t": [2., 0., 0.]})
    assert error["translation_degrees"] == 180.
    assert error["max_degrees"] == 180.
    assert error["translation_signfolded_degrees"] == 0.
    assert error["max_signfolded_degrees"] == 0.
    estimate["R"] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    assert ev.error_with_folded_translation(estimate, {"R": np.eye(3), "t": [1., 0., 0.]})[0]["max_signfolded_degrees"] == 90.


def test_all64_retained_and_gt_cannot_change_primary_selection():
    keys, records, gt = fixture()
    for record in records:
        if record["pair_index"] == 0:
            for candidate in record["candidates"]:
                failed(candidate)
        if record["pair_index"] == 1:
            # The low-score choice is wrong; GT only changes the named oracle.
            record["candidates"][3]["estimate"]["t"] = [-1., 0., 0.]
    prepared = ev.prepare_records(records, keys)
    result = ev.evaluate_records(prepared, gt)
    assert len(result["rows"]) == 192
    for backend in ev.BACKENDS:
        primary = result["summaries"][backend]["pixel_msac_full4"]
        assert primary["pair_count"] == 64
        assert primary["status_counts"] == {"unavailable": 1, "completed": 63}
        assert primary["directed_primary"]["auc_fraction"]["5"] == 62 / 64
        assert primary["signfolded_secondary"]["auc_fraction"]["5"] == 63 / 64
        oracle = result["summaries"][backend]["oracle_best4_nondeployable"]
        assert oracle["directed_primary"]["auc_fraction"]["5"] == 63 / 64
    row = result["rows"][1]
    assert row["selections"]["pixel_msac_full4"]["seed"] == 3
    assert row["selections"]["oracle_best4_nondeployable"]["seed"] == 0
    assert row["candidate_error_spread"]["range_directed_max_degrees"] == 180.
    assert row["prediction_only_dispersion"]["max_translation_directed_degrees"] == 180.


@pytest.mark.parametrize("mutation", ["score_nan", "bad_pose", "bad_time", "incomplete", "duplicate", "started"])
def test_bad_prediction_rejected_before_gt(mutation):
    keys, records, _ = fixture()
    if mutation == "score_nan": records[0]["candidates"][0]["final_scores"]["pixel_msac"] = float("nan")
    elif mutation == "bad_pose": records[0]["candidates"][0]["estimate"]["t"] = [0., 0., 0.]
    elif mutation == "bad_time": records[0]["candidates"][1]["charged_end_seconds"] = 0.01
    elif mutation == "incomplete": records.pop()
    elif mutation == "duplicate": records[-1] = copy.deepcopy(records[0])
    else: records[0]["candidates"][0]["status"] = "started"
    with pytest.raises(ValueError): ev.prepare_records(records, keys)


@pytest.mark.parametrize("mutation", ["missing_charge", "wrong_total", "missing_input", "failed_score"])
def test_inconsistent_budget_charge_rejected(mutation):
    keys, records, _ = fixture()
    record = records[0]
    if mutation == "missing_charge":
        # Still strictly monotone: merely checking order cannot detect undercharge.
        for candidate in record["candidates"]:
            candidate["charged_end_seconds"] -= 0.005
        record["charged_total_seconds"] -= 0.005
    elif mutation == "wrong_total":
        record["charged_total_seconds"] += 0.01
    elif mutation == "missing_input":
        record.pop("input_seconds")
    else:
        candidate = record["candidates"][0]
        failed(candidate)
        candidate["call_seconds"] -= 0.01
        candidate["score_seconds"] = 0.01
    with pytest.raises(ValueError):
        ev.prepare_records(records, keys)


def test_charge_tolerance_accepts_timer_roundoff():
    keys, records, _ = fixture()
    for candidate in records[0]["candidates"]:
        candidate["charged_end_seconds"] += 1e-11
    records[0]["charged_total_seconds"] += 2e-11
    assert len(ev.prepare_records(records, keys)) == 192


def setup_io(tmp_path, monkeypatch):
    keys, records, gt = fixture()
    data = tmp_path / "opaque-not-an-hdf5.bin"
    data.write_bytes(b"artificial opaque bytes; no numeric data")
    manifest_path, seal_path = tmp_path / "manifest.json", tmp_path / "seal.json"
    manifest_path.write_text("{}")
    seal_path.write_text("{}")
    output_path = tmp_path / "evaluation.json"
    hash_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = {"selected_pair_keys": keys, "dataset_bytes": data.stat().st_size,
                "dataset_sha256": hash_file(data)}
    events = []

    def verify(*args):
        events.append("seal_verified")
        return manifest, {"status": "completed"}, records

    class Dataset:
        def __init__(self, key, field): self.key, self.field = key, field
        def __getitem__(self, item):
            assert item == slice(None)
            events.append((self.key, self.field))
            return gt[self.key][self.field].copy()

    class Group:
        def __init__(self, key): self.key = key
        def __getitem__(self, field):
            assert field in ("R", "t"), "non-GT dataset read forbidden"
            return Dataset(self.key, field)

    class File:
        def __init__(self, path, mode):
            assert Path(path) == data and mode == "r"
            assert events and events[0] == "seal_verified"
            events.append("hdf5_open")
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key):
            assert key in keys, "unselected pair forbidden"
            return Group(key)

    def write_new(path, value):
        with Path(path).open("x") as handle:
            json.dump(value, handle, allow_nan=False)

    common = SimpleNamespace(verify_seal=verify, source_hashes=lambda: {"source": "synthetic-hash"},
        sha256=hash_file, json_safe=ev.metrics.json_safe, write_new=write_new)
    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=File))
    return common, data, manifest_path, seal_path, output_path, events, manifest, records


def test_sealed_full_evaluation_reads_exactly_selected_truth(tmp_path, monkeypatch):
    common, data, m, s, out, events, manifest, records = setup_io(tmp_path, monkeypatch)
    result = ev.evaluate(data, m, s, out, common)
    assert result["pair_count"] == 64 and len(result["rows"]) == 192
    assert len([e for e in events if isinstance(e, tuple)]) == 128
    assert events.count("hdf5_open") == 1
    assert events.count("seal_verified") == 2
    saved = json.loads(out.read_text())
    assert saved["summaries"]["old_norm"]["pixel_msac_full4"]["directed_primary"]["auc_fraction"]["5"] == 1.
    assert "nondeployable" in saved["oracle_warning"] or "nondeployable" in saved["rows"][0]["selections"]


@pytest.mark.parametrize("failure", ["tampered_seal", "size", "hash", "bad_candidate", "existing_output"])
def test_pre_gt_barriers(tmp_path, monkeypatch, failure):
    common, data, m, s, out, events, manifest, records = setup_io(tmp_path, monkeypatch)
    if failure == "tampered_seal":
        def reject(*args): raise ValueError("Prediction hash mismatch")
        common.verify_seal = reject
    elif failure == "size": manifest["dataset_bytes"] += 1
    elif failure == "hash": manifest["dataset_sha256"] = "0" * 64
    elif failure == "bad_candidate": records[0]["candidates"][0]["final_scores"]["pixel_msac"] = float("inf")
    else: out.write_text("preserve")
    with pytest.raises(ValueError): ev.evaluate(data, m, s, out, common)
    assert "hdf5_open" not in events
    if failure == "existing_output": assert out.read_text() == "preserve"
    else: assert not out.exists()
