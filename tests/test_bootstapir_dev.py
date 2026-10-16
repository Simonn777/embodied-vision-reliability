import io
import pickle
import numpy as np
import pytest
from evr.bootstapir_dev import (NumpyOnlyUnpickler, inspect_pickle,
                               pixel_query, validate_run, compare, validate_example)


def test_restricted_numpy_round_trip():
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    payload = pickle.dumps(x, protocol=4)
    assert inspect_pickle(io.BytesIO(payload))["opcodes"]["STACK_GLOBAL"] > 0
    np.testing.assert_array_equal(NumpyOnlyUnpickler(io.BytesIO(payload)).load(), x)


def test_forbidden_global_never_executes(tmp_path):
    marker = tmp_path / "executed"
    class Attack:
        def __reduce__(self):
            return eval, (f"open({str(marker)!r}, 'w').write('bad')",)
    with pytest.raises(pickle.UnpicklingError, match="Forbidden global"):
        NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(Attack()))).load()
    assert not marker.exists()


def test_arrival_coordinate_order_uses_local_zero():
    np.testing.assert_array_equal(pixel_query([0.25, 0.75], 17, 100, 200), [0, 75, 50])


@pytest.mark.parametrize("chunk", [0, -1, None, True, 1.5])
def test_chunk_gate(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        validate_run(np.zeros((4, 256, 256, 3)), [], chunk)


def test_arrival_gate_and_empty_query_set():
    f = np.zeros((4, 256, 256, 3), np.uint8)
    validate_run(f, [], 1)
    with pytest.raises(ValueError, match="Arrival"):
        validate_run(f, [{"arrival": 4, "xy": [0.5, 0.5]}], 1)


def test_compare_checks_masks_and_tolerance():
    a = np.array([np.nan, 2.0])
    assert compare(a, a)["passed"]
    assert not compare(a, np.array([0., 2.]))["passed"]
    assert not compare(a, a + 0.001)["passed"]


def test_decoder_schema_rejects_wrong_visibility_shape():
    with pytest.raises(ValueError, match="visibility"):
        validate_example({"video": np.zeros((4, 256, 256, 3), np.uint8),
                          "points": np.zeros((2, 4, 2), np.float32),
                          "occluded": np.zeros((1, 4), bool)})


def test_upstream_digest_gate(tmp_path):
    from evr.bootstapir_dev import UPSTREAM_HASHES, verify_upstream
    for path in UPSTREAM_HASHES:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("modified")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_upstream(tmp_path)


@pytest.mark.parametrize("arrival", [0.5, True, np.bool_(False), "0"])
def test_nonintegral_arrival_rejected(arrival):
    with pytest.raises(ValueError, match="integer"):
        validate_run(np.zeros((4, 256, 256, 3), np.uint8),
                     [{"arrival": arrival, "xy": [0.5, 0.5]}], 1)


def test_input_dtype_and_coordinate_gates():
    with pytest.raises(ValueError, match="256px"):
        validate_run(np.zeros((4, 256, 256, 3), np.float32), [], 1)
    for xy in ([np.nan, .5], [-.1, .5], [.5, 1.1]):
        with pytest.raises(ValueError, match="Invalid query"):
            pixel_query(xy, 0, 256, 256)


def test_gate_failure_is_not_completed():
    from evr.bootstapir_dev import validation_status
    assert validation_status({"prefix": {"passed": False}}) == "validation_failed"
    assert validation_status({"prefix": {"passed": True}}) == "completed"


def test_tampered_development_manifest_rejected(tmp_path):
    import json
    from evr.bootstapir_dev import verify_development
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"count": 50, "development_indices": [1]}))
    with pytest.raises(ValueError, match="index zero"):
        verify_development(tmp_path / "unused.npz", manifest)


def test_data_digest_cannot_be_replaced_by_manifest_claim(tmp_path):
    import json
    from evr.bootstapir_dev import verify_development
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"count": 50, "development_indices": [0],
                                   "video_sha256": "forged"}))
    arrays = tmp_path / "arrays.npz"
    np.savez(arrays, video=np.zeros((2, 256, 256, 3), np.uint8),
             points=np.zeros((1, 2, 2), np.float32), occluded=np.zeros((1, 2), bool))
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_development(arrays, manifest)


def test_actual_import_path_gate_rejects_shadow(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import evr.bootstapir_dev as dev
    monkeypatch.setattr(dev, "verify_upstream", lambda source: None)
    monkeypatch.setattr(dev.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "shadow.py")))
    with pytest.raises(ValueError, match="Unexpected model module path"):
        dev.verify_module_paths(tmp_path)


def test_stateful_adapter_arrival_and_zero_active(monkeypatch):
    """Exercise adapter state lifecycle without installing torch in normal CI."""
    import contextlib
    import sys
    from types import SimpleNamespace
    from evr.bootstapir_dev import stream
    class Tensor(np.ndarray):
        def float(self): return self.astype(np.float32).view(Tensor)
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    def tensor(x): return np.asarray(x).view(Tensor)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=tensor, inference_mode=contextlib.nullcontext))
    class Model:
        def __init__(self): self.states = []; self.frames = 0; self.calls = 0
        def get_feature_grids(self, frame, is_training):
            self.frames += 1
            return frame
        def get_query_features(self, frame, is_training, query_points, feature_grids):
            assert np.all(query_points[..., 0] == 0)
            return SimpleNamespace(resolutions=[(256, 256)] * 2, xy=query_points[0, :, [2, 1]].T.copy())
        def construct_initial_causal_state(self, n, levels):
            assert levels == 1
            state = {"step": 0, "n": n}
            self.states.append(state)
            return state
        def estimate_trajectories(self, size, **kwargs):
            state = kwargs["causal_context"]
            assert state is not None and kwargs["query_chunk_size"] > 0
            state["step"] += 1; self.calls += 1
            feats = kwargs["query_features"]
            coords = tensor((feats.xy + state["step"])[None, :, None])
            logits = tensor(np.full((1, state["n"], 1), state["step"]))
            return {"tracks": [coords], "occlusion": [logits], "expected_dist": [logits], "causal_context": state}
    images = np.zeros((6, 256, 256, 3), np.uint8)
    model = Model()
    out, _ = stream(model, images, [{"arrival": 2, "xy": [.2, .3]},
                                   {"arrival": 4, "xy": [.6, .7]}])
    assert model.frames == 4 and model.calls == 6
    assert [s["step"] for s in model.states] == [4, 2]
    assert np.isnan(out[:2]).all() and np.isnan(out[:4, 1]).all()
    assert np.isfinite(out[2:, 0]).all() and np.isfinite(out[4:, 1]).all()
    independent, _ = stream(Model(), images[4:], [{"arrival": 0, "xy": [.6, .7]}])
    np.testing.assert_array_equal(out[4:, 1], independent[:, 0])
    empty = Model()
    result, _ = stream(empty, images, [])
    assert result.shape == (6, 0, 4) and empty.calls == 0 and empty.frames == 0
