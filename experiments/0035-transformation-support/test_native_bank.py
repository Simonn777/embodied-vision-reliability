"""Manufactured contract/arithmetic tests; prepared, not executed by author."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


worker = load("test_native_bank_worker", HERE / "native_bank.py")


class Contracts(unittest.TestCase):
    def test_only_complete_rgb_and_initial_are_admitted_opaquely(self):
        # These are deliberately not actual JPEG/PNG payloads. This helper
        # verifies metadata/hash identity without decoding any scientific data.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def desc(name, frame, content):
                path = root / name
                path.write_bytes(content)
                return dict(path=str(path), frame=frame, bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest())

            sequence = dict(sequence="manufactured", role="TRAIN",
                rgb=[desc("0.jpg", 0, b"fake-jpeg-0"), desc("1.jpg", 1, b"fake-jpeg-1")],
                initial=desc("0.png", 0, b"fake-initial-png"))
            worker.validate_sequence(sequence)
            Path(sequence["rgb"][1]["path"]).write_bytes(b"fake-jpeg-X")
            with self.assertRaisesRegex(ValueError, "input bytes changed"):
                worker.validate_sequence(sequence)

    def test_later_annotation_list_rejected_before_any_input_read(self):
        sequence = dict(sequence="manufactured", role="TRAIN", rgb=[], initial={},
                        labels=[{"frame": 1, "path": "/must/not/read.png"}])
        with mock.patch.object(worker, "bound_file", side_effect=AssertionError("input read")):
            with self.assertRaisesRegex(ValueError, "sequence metadata keys"):
                worker.validate_sequence(sequence)

    def test_bounded_writer_allows_zip_header_seek_but_not_extent_overrun(self):
        raw = io.BytesIO()
        stream = worker.BoundedOutput(raw, 8)
        self.assertEqual(stream.write(b"12345678"), 8)
        stream.seek(2)
        self.assertEqual(stream.write(b"AB"), 2)
        self.assertEqual(raw.getvalue(), b"12AB5678")
        stream.seek(8)
        with self.assertRaisesRegex(ValueError, "NPZ output byte cap"):
            stream.write(b"X")
        self.assertEqual(len(raw.getvalue()), 8)

    def test_record_is_finite_json_and_never_qualifies_science_itself(self):
        record = worker.initialize_record(Path("/manufactured-only"), "1" * 64, {},
                                          max_output_bytes=10000)
        decoded = json.loads(json.dumps(record, allow_nan=False))
        self.assertFalse(decoded["scientific_eligible"])
        self.assertEqual(decoded["bytes_written"], 0)
        self.assertEqual(decoded["max_output_bytes"], 10000)
        self.assertEqual(decoded["calls"]["image_encoder_intent"], 0)
        with self.assertRaisesRegex(ValueError, "positive output cap"):
            worker.initialize_record("/manufactured", "1" * 64, {}, max_output_bytes=True)

    def test_streamed_state_hash_is_exact_for_bf16_strided_scalar_and_empty(self):
        import torch
        fixture = load("test_native_bank_old_hash", HERE.parent /
            "embodied-vision-reliability/experiments/0029-memory-information/native_fixture.py")
        state = dict(
            matrix=torch.arange(96, dtype=torch.float32).reshape(8, 12)[:, ::2],
            bf16=torch.arange(16, dtype=torch.float32).to(torch.bfloat16),
            scalar=torch.tensor(2., dtype=torch.float32),
            empty=torch.empty((0, 3), dtype=torch.float32),
            nested={2: (None, True, ["native", 4.5, torch.device("cpu")])},
        )
        self.assertEqual(worker.streamed_state_hash(state, torch), fixture.state_hash(state, torch))
        state["matrix"][0, 0] = 7
        self.assertEqual(worker.streamed_state_hash(state, torch), fixture.state_hash(state, torch))


if __name__ == "__main__":
    unittest.main()
