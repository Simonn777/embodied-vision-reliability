"""Artificial supervisor tests; no dataset, worker, or estimator is executed."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


spec = importlib.util.spec_from_file_location(
    "pixel_noise_run_controls", Path(__file__).with_name("run_controls.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

PREFIX = "experiments/0014-pixel-noise/"
SOURCES = {PREFIX + name: hashlib.sha256(name.encode()).hexdigest()
           for name in runner.NAMES}
SOURCES.update({"experiments/0013-pose-headroom/" + name: "e" * 64
                for name in ("predict_pair.py", "run_baseline.py", "evaluate_sealed.py")})
SOURCES["protocols/0014-pixel-noise-controls.md"] = "f" * 64
LIBRARY = {"version": "2.0.5", "binary_sha256": "c" * 64,
           "test_runtime": "artificial, no library imported"}
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
CALIBRATION = {
    "camera1": {"model": "PINHOLE", "params": [600., 800., 320., 240.]},
    "camera2": {"model": "PINHOLE", "params": [900., 700., 400., 300.]},
    "K1": [[600., 0., 320.], [0., 800., 240.], [0., 0., 1.]],
    "K2": [[900., 0., 400.], [0., 700., 300.], [0., 0., 1.]],
}


def receipt(exit_code=0, termination=None):
    return {"exit_code": exit_code, "termination": termination,
            "wall_seconds": 0.01, "sampled_peak_rss_kib": 512,
            "watchdog_error": None, "rss_sampling": "artificial"}


class ArtificialOperation:
    def __init__(self, root):
        self.root = root
        # These three inputs deliberately do not exist. All accesses are mocked.
        self.data = root / "forbidden-dataset.h5"
        self.manifest = root / "manifest.json"
        self.baseline_seal = root / "original-seal.json"
        self.output = root / "predictions"
        self.manifest_hash, self.baseline_hash, self.data_hash = "a" * 64, "b" * 64, "d" * 64
        self.keys = ["artificial-pair-%02d" % i for i in range(16)]
        self.metadata = {"dataset_bytes": 1024, "dataset_sha256": self.data_hash,
                         "selected_pair_keys": list(self.keys)}
        self.records = []
        for index, key in enumerate(self.keys):
            row = {"pair_index": index, "key": key,
                   "sha256": hashlib.sha256(key.encode()).hexdigest(), "status": "completed"}
            prediction = {"calibration": copy.deepcopy(CALIBRATION),
                          "estimate": {"R": copy.deepcopy(IDENTITY), "t": [1., 0., 0.]},
                          "returned_ransac_stats": {"inliers": [i % 3 != 0 for i in range(32)]}}
            self.records.append((row, prediction))
        self.calls, self.hash_calls, self.read_calls = [], [], []
        self.fail_index, self.mutation_index, self.mutate = None, 0, None
        self.failure_receipt = receipt(1)
        self.changed_source = False
        self.verify_error = None

    def fake_record(self, index):
        row, old_prediction = self.records[index]
        mask = old_prediction["returned_ransac_stats"]["inliers"]
        return {
            "schema": "evr0014.prediction.v1", "status": "completed",
            "pair_index": index, "key": row["key"],
            "source_sha256": SOURCES[PREFIX + "predict_pair.py"],
            "kernel_sha256": SOURCES[PREFIX + "refine.py"],
            "baseline_seal_sha256": self.baseline_hash,
            "baseline_prediction_sha256": row["sha256"],
            "library": copy.deepcopy(LIBRARY),
            "calibration": copy.deepcopy(old_prediction["calibration"]),
            "mask_sha256": hashlib.sha256(bytes(mask)).hexdigest(), "kept_count": sum(mask),
            "estimates": {mode: {"R": copy.deepcopy(IDENTITY), "t": [1., 0., 0.]}
                          for mode in ("native", "N", "P")},
        }

    def fake_child(self, command, log_path, remaining):
        index = int(command[command.index("--pair-index") + 1])
        output = Path(command[command.index("--output") + 1])
        assert command[0] == sys.executable
        assert Path(command[1]) == runner.HERE / "predict_pair.py"
        assert Path(command[command.index("--data") + 1]) == self.data
        assert Path(command[command.index("--manifest") + 1]) == self.manifest
        assert Path(command[command.index("--baseline-seal") + 1]) == self.baseline_seal
        assert 0 < remaining <= 1800
        assert output == self.output / ("pair-%02d.json" % index)
        self.calls.append(index)
        value = self.fake_record(index)
        if index == self.mutation_index and self.mutate is not None:
            self.mutate(value)
        runner.infrastructure.write_new(output, value)
        Path(log_path).write_text("artificial child receipt only\n")
        return self.failure_receipt if index == self.fail_index else receipt()

    def fake_sha(self, path):
        path = Path(path)
        self.hash_calls.append(path)
        expected = {self.data: self.data_hash, self.manifest: self.manifest_hash,
                    self.baseline_seal: self.baseline_hash}
        assert path in expected, "unexpected input hash/read: " + str(path)
        return expected[path]

    def execute(self):
        original_stat, original_open = Path.stat, Path.open

        def guarded_stat(path, *args, **kwargs):
            if path == self.data:
                return SimpleNamespace(st_size=self.metadata["dataset_bytes"])
            return original_stat(path, *args, **kwargs)

        def guarded_open(path, mode="r", *args, **kwargs):
            if "r" in mode or "+" in mode:
                assert (path.parent == self.output
                        and re.fullmatch(r"pair-\d{2}\.json", path.name)), (
                            "forbidden input/GT/evaluation read: " + str(path))
                self.read_calls.append(path.name)
            return original_open(path, mode, *args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("GT, evaluation, HDF5, or actual child access is forbidden")

        changed = {**SOURCES, PREFIX + "refine.py": "9" * 64}
        verify_result = (copy.deepcopy(self.metadata), {"complete": True},
                         copy.deepcopy(self.records), self.manifest_hash, self.baseline_hash)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(Path, "stat", guarded_stat))
            stack.enter_context(patch.object(Path, "open", guarded_open))
            binding_mock = stack.enter_context(patch.object(
                runner, "bindings", side_effect=[dict(SOURCES), changed]
                if self.changed_source else None, return_value=dict(SOURCES)))
            verify_mock = stack.enter_context(patch.object(
                runner.baseline_seal, "verify_seal", return_value=verify_result,
                side_effect=self.verify_error))
            stack.enter_context(patch.object(runner, "sha", side_effect=self.fake_sha))
            library_mock = stack.enter_context(patch.object(
                runner.infrastructure, "expected_library", return_value=copy.deepcopy(LIBRARY)))
            stack.enter_context(patch.object(runner.infrastructure, "run_child", side_effect=self.fake_child))
            stack.enter_context(patch.object(runner.infrastructure.subprocess, "Popen", side_effect=forbidden))
            stack.enter_context(patch.object(runner.baseline_seal, "evaluate", side_effect=forbidden))
            stack.enter_context(patch.object(runner.baseline_seal, "project_gt_rotation", side_effect=forbidden))
            stack.enter_context(patch.object(runner.baseline_seal, "pose_errors", side_effect=forbidden))
            stack.enter_context(patch.dict(sys.modules, {"h5py": SimpleNamespace(File=forbidden)}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = runner.run(self.data, self.manifest, self.baseline_seal, self.output)
            self.binding_calls = binding_mock.call_count
            self.verify_calls = verify_mock.call_args_list
            self.library_calls = library_mock.call_count
        return code

    def seal(self):
        return json.loads((self.output / "seal.json").read_text())


def test_complete16_seal_binds_outputs_without_any_truth_or_evaluation_read(tmp_path):
    operation = ArtificialOperation(tmp_path)
    assert operation.execute() == 0
    seal = operation.seal()
    assert seal["schema"] == "evr0014.seal.v1"
    assert seal["complete"] and seal["eligible_for_evaluation"]
    assert seal["error"] is None
    assert operation.calls == list(range(16))
    assert [row["key"] for row in seal["rows"]] == operation.keys
    assert seal["source_sha256"] == SOURCES
    assert seal["manifest_sha256"] == operation.manifest_hash
    assert seal["baseline_seal_sha256"] == operation.baseline_hash
    assert seal["dataset_sha256_declared"] == operation.data_hash
    assert operation.read_calls == ["pair-%02d.json" % i for i in range(16)]
    assert operation.hash_calls == [operation.data, operation.manifest, operation.baseline_seal]
    assert operation.binding_calls == 2 and operation.library_calls == 1
    assert len(operation.verify_calls) == 1
    assert operation.verify_calls[0].args == (operation.manifest, operation.baseline_seal)
    for index, row in enumerate(seal["rows"]):
        assert row["pair_index"] == index and row["status"] == "completed"
        raw = (operation.output / row["path"]).read_bytes()
        assert row["sha256"] == hashlib.sha256(raw).hexdigest()
        assert (operation.output / ("pair-%02d.process.json" % index)).is_file()
    integrity = json.loads((operation.output / "integrity.json").read_text())
    assert integrity["numeric_arrays_decoded"] is False
    assert not operation.data.exists() and not operation.manifest.exists()
    assert not operation.baseline_seal.exists()


def test_existing_output_directory_refuses_before_inputs_or_children(tmp_path):
    operation = ArtificialOperation(tmp_path)
    operation.output.mkdir()
    sentinel = operation.output / "preserve.txt"
    sentinel.write_text("existing evidence\n")
    with patch.object(runner, "bindings", side_effect=AssertionError("source access")), \
         patch.object(runner.baseline_seal, "verify_seal", side_effect=AssertionError("baseline access")), \
         patch.object(runner.infrastructure, "run_child", side_effect=AssertionError("child access")):
        with pytest.raises(FileExistsError):
            runner.run(operation.data, operation.manifest, operation.baseline_seal, operation.output)
    assert sentinel.read_text() == "existing evidence\n"
    assert [p.name for p in operation.output.iterdir()] == ["preserve.txt"]


@pytest.mark.parametrize("failure", [receipt(1), receipt(-9, "pair_timeout"),
                                    receipt(0, "rss_limit"), receipt(-11)])
def test_critical_child_failure_stops_and_retains_receipt(tmp_path, failure):
    operation = ArtificialOperation(tmp_path)
    operation.fail_index, operation.failure_receipt = 4, failure
    assert operation.execute() == 1
    seal = operation.seal()
    assert not seal["complete"] and not seal["eligible_for_evaluation"]
    assert operation.calls == [0, 1, 2, 3, 4]
    assert len(seal["rows"]) == 4
    assert seal["error"]["type"] == "RuntimeError"
    assert (operation.output / "pair-04.json").is_file()
    assert (operation.output / "pair-04.process.json").is_file()
    assert not (operation.output / "pair-05.json").exists()
    assert operation.read_calls == ["pair-%02d.json" % i for i in range(4)]


@pytest.mark.parametrize("field", ["source_sha256", "kernel_sha256", "baseline_seal_sha256",
                                  "baseline_prediction_sha256", "key", "pair_index", "status", "schema"])
def test_record_source_baseline_identity_and_status_mismatch_blocks_seal(tmp_path, field):
    operation = ArtificialOperation(tmp_path)
    operation.mutate = lambda record: record.update({field: "changed"})
    assert operation.execute() == 1
    seal = operation.seal()
    assert operation.calls == [0] and seal["rows"] == []
    assert not seal["eligible_for_evaluation"]
    assert field in seal["error"]["message"]


@pytest.mark.parametrize("kind", ["library", "calibration"])
def test_library_or_calibration_mismatch_blocks_seal(tmp_path, kind):
    operation = ArtificialOperation(tmp_path)
    def mutate(record):
        if kind == "library":
            record["library"]["binary_sha256"] = "0" * 64
        else:
            record["calibration"]["K1"][0][0] += 1.0
    operation.mutate = mutate
    assert operation.execute() == 1
    seal = operation.seal()
    assert operation.calls == [0] and seal["rows"] == []
    assert not seal["eligible_for_evaluation"]
    assert kind in seal["error"]["message"]


@pytest.mark.parametrize("field", ["mask_sha256", "kept_count"])
def test_original_mask_digest_and_count_mismatch_blocks_seal(tmp_path, field):
    operation = ArtificialOperation(tmp_path)
    operation.mutate = lambda record: record.update({field: "changed"})
    assert operation.execute() == 1
    seal = operation.seal()
    assert operation.calls == [0] and seal["rows"] == []
    assert not seal["eligible_for_evaluation"]
    assert "inlier mask" in seal["error"]["message"]


@pytest.mark.parametrize("mode", ["native", "N", "P"])
@pytest.mark.parametrize("field", ["R", "t"])
def test_invalid_pose_in_any_control_blocks_seal(tmp_path, mode, field):
    operation = ArtificialOperation(tmp_path)
    def mutate(record):
        record["estimates"][mode][field] = (
            [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] if field == "R" else [0., 0., 0.])
    operation.mutate = mutate
    assert operation.execute() == 1
    seal = operation.seal()
    assert operation.calls == [0] and seal["rows"] == []
    assert not seal["eligible_for_evaluation"]
    assert seal["error"]["type"] == "ValueError"


def test_postrun_source_change_preserves16_rows_but_makes_seal_ineligible(tmp_path):
    operation = ArtificialOperation(tmp_path)
    operation.changed_source = True
    assert operation.execute() == 1
    seal = operation.seal()
    assert operation.calls == list(range(16)) and len(seal["rows"]) == 16
    assert not seal["complete"] and not seal["eligible_for_evaluation"]
    assert "frozen input changed" in seal["error"]["message"]
    assert seal["source_sha256"] == SOURCES


def test_original_seal_refusal_prevents_data_hash_library_and_all_children(tmp_path):
    operation = ArtificialOperation(tmp_path)
    operation.verify_error = ValueError("original seal does not bind source")
    assert operation.execute() == 1
    seal = operation.seal()
    assert not seal["eligible_for_evaluation"] and seal["rows"] == []
    assert operation.calls == [] and operation.hash_calls == [] and operation.read_calls == []
    assert operation.library_calls == 0
    assert not (operation.output / "integrity.json").exists()
