"""Artificial-only metric, seal-order, access-boundary and serialization checks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "evaluate_sealed", Path(__file__).with_name("evaluate_sealed.py"))
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def write_json(path, value):
    path.write_text(json.dumps(ev.json_safe(value), indent=2, allow_nan=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_artificial_dataset(bundle):
    """Build consistent synthetic evidence before testing an evaluation defect."""
    manifest, seal = bundle["manifest"], bundle["seal"]
    manifest["dataset_sha256"] = hashlib.sha256(bundle["data_path"].read_bytes()).hexdigest()
    manifest["dataset_bytes"] = bundle["data_path"].stat().st_size
    mh = write_json(bundle["manifest_path"], manifest)
    seal.update(manifest_sha256=mh, dataset_sha256_declared=manifest["dataset_sha256"])
    for row in seal["rows"]:
        path = bundle["root"] / row["path"]
        record = json.loads(path.read_text())
        record.update(manifest_sha256=mh, dataset_sha256_declared=manifest["dataset_sha256"])
        row["sha256"] = write_json(path, record)
    write_json(bundle["seal_path"], seal)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source_dir = tmp_path / "experiments" / "0013-pose-headroom"
    source_dir.mkdir(parents=True)
    source_hashes = {}
    for name in ev.SOURCE_NAMES:
        content = (Path(ev.__file__).read_bytes() if name == "evaluate_sealed.py"
                   else b"# Synthetic source binding fixture.\n")
        (source_dir / name).write_bytes(content)
        source_hashes[name] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(ev, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ev, "SOURCE_DIR", source_dir)
    protocol = tmp_path / "protocols" / "0013-baseline-pose-headroom.md"
    protocol.parent.mkdir()
    protocol.write_text("Synthetic evaluation protocol; no actual data.\n")
    keys = ["synthetic-0015-pair-%02d" % index for index in range(16)]
    manifest = {"selected_pair_keys": keys, "dataset_sha256": "a" * 64}
    manifest_path = tmp_path / "manifest.json"
    manifest_hash = write_json(manifest_path, manifest)
    seal = {"schema": "evr0013.seal.v1", "complete": True,
            "eligible_for_evaluation": True, "manifest_sha256": manifest_hash,
            "dataset_sha256_declared": manifest["dataset_sha256"],
            "source_sha256": source_hashes,
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(), "rows": []}
    for index, key in enumerate(keys):
        status = "completed" if index < 14 else ("timeout" if index == 14 else "estimator_error")
        record = {"schema": "evr0013.prediction.v1", "pair_index": index, "key": key,
                  "status": status, "manifest_sha256": manifest_hash,
                  "dataset_sha256_declared": manifest["dataset_sha256"],
                  "source_sha256": source_hashes["predict_pair.py"],
                  "settings": {"seed": 0}}
        if status == "completed":
            record["estimate"] = {"R": np.eye(3), "t": [1.0, 0.0, 0.0]}
        else:
            record["process"] = {"returncode": -15 if status == "timeout" else 1}
        path = tmp_path / ("pair-%02d.json" % index)
        digest = write_json(path, record)
        seal["rows"].append({"pair_index": index, "key": key, "path": path.name,
                             "sha256": digest, "status": status})
    seal_path = tmp_path / "seal.json"
    write_json(seal_path, seal)
    data_path = tmp_path / "synthetic-only.h5"
    with h5py.File(data_path, "w") as data:
        for key in keys + ["unselected-scene-0022"]:
            group = data.create_group(key)
            group["R"] = np.eye(3)
            group["t"] = np.array([[2.0], [0.0], [0.0]])
            for forbidden in ("x1", "x2", "K1", "K2", "confidence", "matching_score"):
                group[forbidden] = np.array([777.0])
    result = {"root": tmp_path, "source_dir": source_dir, "protocol": protocol,
            "keys": keys, "manifest": manifest, "manifest_path": manifest_path,
            "seal": seal, "seal_path": seal_path, "data_path": data_path,
            "output_path": tmp_path / "evaluation.json"}
    bind_artificial_dataset(result)
    return result


def run_bundle(bundle):
    return ev.evaluate(bundle["data_path"], bundle["manifest_path"],
                       bundle["seal_path"], bundle["output_path"])


def test_directed_translation_and_rotation_angles():
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    errors, details = ev.pose_errors(quarter_turn, [-4, 0, 0], np.eye(3), [2, 0, 0])
    assert errors == {"rotation_degrees": 90.0, "translation_degrees": 180.0,
                      "max_degrees": 180.0}
    assert details["translation_original_norm"] == 2.0
    assert details["prediction_translation_norm"] == 4.0


def test_gt_projection_is_dms_uvt_without_reflection_repair():
    raw = np.diag([1.001, 0.999, 1.0])
    projected, details = ev.project_gt_rotation(raw)
    np.testing.assert_array_equal(projected, np.eye(3))
    assert details["projection_frobenius_change"] == pytest.approx(np.sqrt(2) * .001)
    for invalid in (np.diag([-1., 1., 1.]), np.zeros((3, 3)), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError):
            ev.project_gt_rotation(invalid)
    with pytest.raises(ValueError):
        ev.valid_rotation(raw)
    with pytest.raises(ValueError):
        ev.unit_translation([0, 0, 0])
    unit, norm = ev.unit_translation([1e-300, 0, 0])
    np.testing.assert_array_equal(unit, [1, 0, 0])
    assert norm == 1e-300


def test_dms_auc_known_values_and_exact_threshold_ties():
    assert ev.dms_auc([0, 5, 10, 20, 180]) == pytest.approx([.2, .35, .5])
    assert ev.dms_auc([10]) == pytest.approx([0, 0, .75])
    assert ev.dms_auc([0, 0]) == [1, 1, 1]
    assert ev.dms_auc([180, 180]) == [0, 0, 0]
    for bad in ([], [float("nan")], [-1], [float("inf")]):
        with pytest.raises(ValueError):
            ev.dms_auc(bad)


def test_complete_seal_then_only_selected_gt_reads_and_no_writes(bundle, monkeypatch):
    allowed = {"/" + key + "/" + field for key in bundle["keys"] for field in ("R", "t")}
    reads, opens = [], []
    original_get = h5py.Dataset.__getitem__
    original_file = h5py.File
    digest_before = hashlib.sha256(bundle["data_path"].read_bytes()).hexdigest()

    def checked_read(dataset, selection):
        assert dataset.name in allowed, "forbidden dataset read: " + dataset.name
        reads.append(dataset.name)
        return original_get(dataset, selection)

    def readonly_open(path, mode, *args, **kwargs):
        assert mode == "r"
        opens.append((path, mode))
        return original_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_read)
    monkeypatch.setattr(h5py, "File", readonly_open)
    result = run_bundle(bundle)
    assert len(opens) == 1
    assert len(reads) == 32 and set(reads) == allowed
    assert hashlib.sha256(bundle["data_path"].read_bytes()).hexdigest() == digest_before
    assert result["coverage"]["successful"] == 14
    assert result["coverage"]["successful_fraction"] == 14 / 16
    assert result["primary_all_pair_coverage_penalized"]["auc_fraction"] == {
        "5": .875, "10": .875, "20": .875}
    assert result["secondary_successful_only"]["auc_fraction"] == {"5": 1., "10": 1., "20": 1.}
    assert result["rows"][14]["errors"] is None
    assert result["rows"][14]["all_pair_penalized_error_degrees"] == 180
    assert result["rows"][0]["ground_truth"]["translation_original_norm"] == 2
    assert result["rows"][0]["prediction"]["estimate"]["R"] == np.eye(3).tolist()
    assert json.loads(bundle["output_path"].read_text()) == result


@pytest.mark.parametrize("defect", [
    "incomplete", "ineligible", "missing_row", "wrong_key", "status", "late_prediction_hash",
    "manifest_hash", "source_hash", "protocol_hash", "dataset_declaration", "invalid_pose",
    "prediction_source_hash",
])
def test_bad_seal_blocks_hdf5_open_for_entire_cohort(bundle, monkeypatch, defect):
    seal = bundle["seal"]
    if defect == "incomplete":
        seal["complete"] = False
    elif defect == "ineligible":
        seal["eligible_for_evaluation"] = False
    elif defect == "missing_row":
        seal["rows"].pop()
    elif defect == "wrong_key":
        seal["rows"][-1]["key"] = "unselected-scene-0022"
    elif defect == "status":
        seal["rows"][-1]["status"] = "started"
    elif defect == "late_prediction_hash":
        (bundle["root"] / "pair-15.json").write_text("{}")
    elif defect == "manifest_hash":
        bundle["manifest_path"].write_text(bundle["manifest_path"].read_text() + " ")
    elif defect == "source_hash":
        (bundle["source_dir"] / "predict_pair.py").write_text("changed\n")
    elif defect == "protocol_hash":
        bundle["protocol"].write_text("changed\n")
    elif defect == "dataset_declaration":
        seal["dataset_sha256_declared"] = "b" * 64
    elif defect == "invalid_pose":
        path = bundle["root"] / "pair-13.json"
        record = json.loads(path.read_text())
        record["estimate"]["t"] = [0, 0, 0]
        seal["rows"][13]["sha256"] = write_json(path, record)
    elif defect == "prediction_source_hash":
        path = bundle["root"] / "pair-15.json"
        record = json.loads(path.read_text())
        record["source_sha256"] = "b" * 64
        seal["rows"][15]["sha256"] = write_json(path, record)
    write_json(bundle["seal_path"], seal)

    def forbidden_open(*args, **kwargs):
        pytest.fail("HDF5 opened before all prediction/seal checks passed")

    monkeypatch.setattr(h5py, "File", forbidden_open)
    with pytest.raises(ValueError):
        run_bundle(bundle)
    assert not bundle["output_path"].exists()


def test_existing_output_blocks_truth_access(bundle, monkeypatch):
    bundle["output_path"].write_text("preserve me")
    monkeypatch.setattr(h5py, "File", lambda *a, **kw: pytest.fail("unexpected truth access"))
    with pytest.raises(FileExistsError):
        run_bundle(bundle)
    assert bundle["output_path"].read_text() == "preserve me"


def test_all_failed_predictions_keep_all_pair_denominator(bundle):
    for row in bundle["seal"]["rows"]:
        path = bundle["root"] / row["path"]
        record = json.loads(path.read_text())
        record["status"] = row["status"] = "estimator_error"
        record.pop("estimate", None)
        row["sha256"] = write_json(path, record)
    write_json(bundle["seal_path"], bundle["seal"])
    result = run_bundle(bundle)
    assert result["coverage"]["successful"] == 0
    assert result["primary_all_pair_coverage_penalized"]["count"] == 16
    assert result["primary_all_pair_coverage_penalized"]["auc_fraction"] == {"5": 0., "10": 0., "20": 0.}
    assert result["secondary_successful_only"]["auc_fraction"] is None
    assert len(result["rows"]) == 16


@pytest.mark.parametrize("field,value", [("t", [0, 0, 0]), ("R", np.diag([-1, 1, 1]))])
def test_invalid_gt_refuses_metrics_including_failed_predictions(bundle, field, value):
    with h5py.File(bundle["data_path"], "r+") as data:
        group = data[bundle["keys"][-1]]  # This prediction failed, but its GT must still be valid.
        del group[field]
        group[field] = value
    bind_artificial_dataset(bundle)
    with pytest.raises(ValueError):
        run_bundle(bundle)
    assert not bundle["output_path"].exists()


def test_different_data_file_rejected_before_truth_open(bundle, monkeypatch):
    # Same size, changed opaque content, while all seal/record hashes remain valid.
    raw = bytearray(bundle["data_path"].read_bytes())
    raw[-1] ^= 1
    bundle["data_path"].write_bytes(raw)
    monkeypatch.setattr(h5py, "File", lambda *a, **kw: pytest.fail("wrong data opened"))
    with pytest.raises(ValueError, match="dataset hash"):
        run_bundle(bundle)


def test_recursive_numpy_conversion_and_nonfinite_json_rejection(tmp_path):
    value = {"a": [np.int64(4), {"b": np.array([.2, .4])}], "bool": np.bool_(True)}
    assert ev.json_safe(value) == {"a": [4, {"b": [.2, .4]}], "bool": True}
    with pytest.raises(ValueError):
        ev.json_safe({"nested": [np.array([float("inf")])]})
    path = tmp_path / "bad.json"
    for raw in ('{"nested":[NaN]}', '{"x":1,"x":2}', '{"x":1e999}'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            ev.load_hashed_json(path)
