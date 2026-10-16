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
    f = np.zeros((4, 256, 256, 3))
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
