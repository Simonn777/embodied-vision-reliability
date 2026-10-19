"""Handmade frames and fake states only; no model, cached video or labels."""

import importlib.util
from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_oblique_predictions.py"
SPEC = importlib.util.spec_from_file_location("oblique_runner_test", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PROTOCOL_SHA = "a" * 64
VIEW_NAMES = ["original", "reference", "oblique", "axis_control"]


def inputs():
    video = np.empty((16, 256, 256, 3), dtype=np.uint8)
    for index in range(16):
        video[index] = [index, 101, 199]
    return {"video": video,
            "query_xy": np.array([[30 + 10 * i, 80 + 5 * i] for i in range(8)], dtype=np.float32),
            "point_ids": np.arange(8, dtype=np.int64) * 2,
            "frame_ids": np.arange(0, 256, 16, dtype=np.int64)}


def hand_mapped_query(original, view_index):
    angle, scale = [(0, 1), (0, .7), (45, .7), (90, .7)][view_index]
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    xy = original.astype(np.float64) - 128
    mapped = np.stack([scale * (cosine * xy[:, 0] - sine * xy[:, 1]) + 128,
                       scale * (sine * xy[:, 0] + cosine * xy[:, 1]) + 128], axis=-1)
    tyx = np.zeros((1, 8, 3), dtype=np.float32)
    tyx[0, :, 1:] = mapped[:, ::-1].astype(np.float32)
    return tyx


class FakeBackend:
    def __init__(self, original_query, *, invalid=None, raise_on=None):
        self.original_query = original_query.copy()
        self.invalid = invalid
        self.raise_on = raise_on
        self.calls = []
        self.closed = False
        self.metadata = {"implementation": "fake software test; no scientific inference"}

    def forward(self, rgb, query_tyx, state):
        index = len(self.calls)
        view_index, observation = divmod(index, 16)
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise AssertionError("A backend must see only the current rendered RGB frame")
        np.testing.assert_array_equal(rgb[128, 128], [observation, 101, 199])
        expected = hand_mapped_query(self.original_query, view_index)
        if observation == 0:
            if state is not None:
                raise AssertionError("A new view must begin with an independent state")
            np.testing.assert_array_equal(query_tyx, expected)
        else:
            if query_tyx is not None or state.step != observation or state.view != view_index:
                raise AssertionError("Only the previous state from this view is allowed")
        self.calls.append((view_index, observation))
        if index == self.raise_on:
            raise RuntimeError("Deliberate terminal fake failure")
        xy = expected[0][:, [2, 1]].copy() + np.float32(observation / 16)
        arrays = {
            "coordinate_logits": np.tile(np.arange(512, dtype=np.float32), (8, 1)) + index,
            "positions_xy": xy,
            "visibility_logits": np.arange(8, dtype=np.float32) - 4,
        }
        if self.invalid == "nan":
            arrays["coordinate_logits"][0, 0] = np.nan
        elif self.invalid == "shape":
            arrays["positions_xy"] = np.zeros((8, 3), dtype=np.float32)
        elif self.invalid == "dtype":
            arrays["visibility_logits"] = arrays["visibility_logits"].astype(np.float64)
        elif self.invalid == "extra":
            arrays["unexpected"] = np.zeros(1, dtype=np.float32)
        next_step = observation + (2 if self.invalid == "step" else 1)
        return arrays, SimpleNamespace(step=next_step, view=view_index)

    def close(self):
        self.closed = True


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.values = inputs()
        self.input_path = self.root / "prepared.npz"
        self.output = self.root / "attempt"
        self.write_inputs()

    def write_inputs(self):
        with self.input_path.open("wb") as stream:
            np.savez(stream, **self.values)
        self.input_sha = runner.sha(self.input_path)

    def execute(self, backend=None, **overrides):
        backend = backend or FakeBackend(self.values["query_xy"])
        options = dict(prepared_input=self.input_path, input_sha256=self.input_sha,
                       group_id="g-036", output=self.output, protocol_sha256=PROTOCOL_SHA,
                       backend_factory=lambda: backend)
        options.update(overrides)
        return runner.run_predictions(**options)

    def test_four_independent_causal_views_exact64_and_native_output_identity(self):
        backend = FakeBackend(self.values["query_xy"])
        receipt = self.execute(backend)
        self.assertEqual(backend.calls, [(v, t) for v in range(4) for t in range(16)])
        self.assertTrue(backend.closed)
        self.assertEqual(receipt["status"], "completed_predictions")
        self.assertEqual(receipt["forwards_attempted"], 64)
        self.assertEqual(receipt["completed_predictions"], 64)
        self.assertEqual(receipt["view_order"], VIEW_NAMES)
        self.assertEqual(receipt["point_ids"], self.values["point_ids"].tolist())
        self.assertEqual(receipt["input_sha256"], self.input_sha)
        self.assertEqual(len(list(self.output.glob("*.npz"))), 64)
        for index, frame in enumerate(receipt["frames"]):
            view, observation = divmod(index, 16)
            self.assertEqual(frame["view"], VIEW_NAMES[view])
            self.assertEqual(frame["observation_index"], observation)
            self.assertEqual(frame["frame_index"], 16 * observation)
            self.assertEqual(frame["state_step"], observation + 1)
            path = self.output / frame["output_file"]
            self.assertEqual(runner.sha(path), frame["output_sha256"])
            with np.load(path, allow_pickle=False) as raw:
                self.assertEqual(set(raw.files), set(runner.OUTPUT_SCHEMA))
                for name, shape in runner.OUTPUT_SCHEMA.items():
                    self.assertEqual(raw[name].dtype, np.float32)
                    self.assertEqual(raw[name].shape, shape)
                np.testing.assert_array_equal(raw["coordinate_logits"][0], np.arange(512) + index)
                expected_xy = hand_mapped_query(self.values["query_xy"], view)[0][:, [2, 1]]
                np.testing.assert_array_equal(raw["positions_xy"], expected_xy + np.float32(observation / 16))
                np.testing.assert_array_equal(raw["visibility_logits"], np.arange(8) - 4)
        complete = json.loads((self.output / "completed.json").read_text())
        self.assertEqual(complete["receipt_sha256"], runner.sha(self.output / "receipt.json"))
        self.assertGreater(receipt["peak_rss_bytes"], 0)
        self.assertTrue((self.output / "started.json").exists())

    def test_existing_directory_is_never_overwritten_even_if_empty(self):
        self.output.mkdir()
        calls = []
        with self.assertRaises(FileExistsError):
            self.execute(backend_factory=lambda: calls.append("forbidden"))
        self.assertEqual(calls, [])
        self.assertEqual(list(self.output.iterdir()), [])

    def test_partial_failure_is_preserved_and_cannot_resume(self):
        backend = FakeBackend(self.values["query_xy"], raise_on=2)
        with self.assertRaisesRegex(RuntimeError, "terminal fake"):
            self.execute(backend)
        receipt = json.loads((self.output / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["forwards_attempted"], 3)
        self.assertEqual(receipt["completed_predictions"], 2)
        self.assertEqual(len(list(self.output.glob("*.npz"))), 2)
        self.assertFalse((self.output / "completed.json").exists())
        before = {p.name: runner.sha(p) for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.execute(backend)
        self.assertEqual(before, {p.name: runner.sha(p) for p in self.output.iterdir()})
        self.assertEqual(len(backend.calls), 3)
        self.assertTrue(backend.closed)

    def test_invalid_model_outputs_fail_before_saving_and_never_retry(self):
        for invalid in ("nan", "shape", "dtype", "extra", "step"):
            with self.subTest(invalid=invalid):
                backend = FakeBackend(self.values["query_xy"], invalid=invalid)
                output = self.root / invalid
                with self.assertRaises(ValueError):
                    self.execute(backend, output=output)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(receipt["forwards_attempted"], 1)
                self.assertEqual(receipt["completed_predictions"], 0)
                self.assertEqual(list(output.glob("*.npz")), [])
                self.assertEqual(len(backend.calls), 1)
                self.assertTrue(backend.closed)

    def test_label_key_is_rejected_before_backend_construction(self):
        self.values["evaluation_labels"] = np.zeros((8, 16, 2), dtype=np.float32)
        self.write_inputs()
        calls = []
        with self.assertRaisesRegex(ValueError, "label-free"):
            self.execute(backend_factory=lambda: calls.append("forbidden"))
        self.assertEqual(calls, [])
        self.assertEqual(json.loads((self.output / "receipt.json").read_text())["forwards_attempted"], 0)

    def test_input_shape_precision_frame_ids_queries_and_point_ids_are_bound(self):
        mutations = {
            "shape": lambda a: a.update(video=a["video"][:15]),
            "precision": lambda a: a.update(query_xy=a["query_xy"].astype(np.float64)),
            "frame_ids": lambda a: a.update(frame_ids=np.arange(16, dtype=np.int64)),
            "nan": lambda a: a["query_xy"].__setitem__((0, 0), np.nan),
            "range": lambda a: a["query_xy"].__setitem__((0, 1), 257),
            "point_ids": lambda a: a["point_ids"].__setitem__(1, 0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self.values = inputs()
                mutate(self.values)
                self.write_inputs()
                calls = []
                with self.assertRaises(ValueError):
                    self.execute(output=self.root / name,
                                 backend_factory=lambda: calls.append("forbidden"))
                self.assertEqual(calls, [])

    def test_wrong_hash_and_unallocated_group_fail_without_backend(self):
        for name, options in (("hash", {"input_sha256": "b" * 64}),
                              ("group", {"group_id": "g-033"}),
                              ("protocol", {"protocol_sha256": "not-a-digest"})):
            with self.subTest(name=name):
                calls = []
                with self.assertRaises(ValueError):
                    self.execute(output=self.root / name,
                                 backend_factory=lambda: calls.append("forbidden"), **options)
                self.assertEqual(calls, [])

    def test_compressed_and_duplicate_members_are_not_accepted(self):
        with self.input_path.open("wb") as stream:
            np.savez_compressed(stream, **self.values)
        with self.assertRaisesRegex(ValueError, "ZIP_STORED"):
            runner.read_prepared_input(self.input_path, runner.sha(self.input_path))
        self.write_inputs()
        import warnings
        import zipfile
        with zipfile.ZipFile(self.input_path, "a") as archive:
            original = archive.read("frame_ids.npy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("frame_ids.npy", original)
        with self.assertRaisesRegex(ValueError, "four label-free"):
            runner.read_prepared_input(self.input_path, runner.sha(self.input_path))

    def test_import_does_not_load_numpy_torch_geometry_or_run_any_backend(self):
        program = (
            "import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('isolated_runner',{str(PATH)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
            "assert '_oblique_prediction_geometry' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-S", "-c", program],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class FrozenIdentityTests(unittest.TestCase):
    def test_pins_agree_with_reviewed_0005_configuration(self):
        path = PATH.parents[1] / "configs" / "0005-native-capability.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["expected_packages"], runner.PACKAGES)
        source_assets = [a for a in config["preparation"]["assets"] if a["kind"] == "source"]
        got = {a["relative_path"].removeprefix("source/").removesuffix(".py").replace("/", "."):
               a["sha256"] for a in source_assets}
        self.assertEqual(got, runner.SOURCES)
        weight = next(a for a in config["preparation"]["assets"] if a["kind"] == "checkpoint")
        self.assertEqual((weight["sha256"], weight["bytes"]), (runner.WEIGHT_SHA, runner.WEIGHT_BYTES))

    def test_receipt_identity_and_success_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            payload = {"status": "completed_native_capability", "source_sha256": runner.SOURCES,
                       "checkpoint_sha256": runner.WEIGHT_SHA, "source_revision": runner.SOURCE_REV,
                       "checkpoint_revision": runner.HF_REV, "packages": runner.PACKAGES}
            path.write_text(json.dumps(payload))
            runner.verify_asset_receipt(path, runner.sha(path))
            payload["status"] = "failed"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "does not establish"):
                runner.verify_asset_receipt(path, runner.sha(path))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                runner.verify_asset_receipt(path, "c" * 64)


class NativeBoundaryTests(unittest.TestCase):
    """Exercise wrapper/raw-output conversion with fake tensors, never Torch."""

    @staticmethod
    def tensor(array):
        return SimpleNamespace(shape=array.shape, device=SimpleNamespace(type="cpu"),
                               dtype=np.dtype("float32"),
                               detach=lambda: SimpleNamespace(numpy=lambda: array))

    def backend(self, *, alter_state_query=False, disagree_positions=False):
        backend = runner.NativeBackend.__new__(runner.NativeBackend)
        backend.np = np
        backend.torch = SimpleNamespace(float32=np.dtype("float32"), inference_mode=nullcontext)
        backend._captured = []
        backend._initial_query = None
        self.seen = []
        layer = SimpleNamespace(
            rg_lru_state=self.tensor(np.broadcast_to(np.float32(0), (1032, 768))),
            conv1d_state=self.tensor(np.broadcast_to(np.float32(0), (1032, 3, 768))))

        def track_frame(bgr, *, query_points_xy, state, autocast):
            self.seen.append((bgr.copy(), query_points_xy, state, autocast))
            if state is None:
                tyx = np.zeros((1, 8, 3), dtype=np.float32)
                tyx[0, :, 1:] = query_points_xy[:, ::-1]
                step = 1
            else:
                tyx = state.query_points.detach().numpy().copy()
                step = state.step + 1
            if alter_state_query:
                tyx[0, 0, 1] += 1
            next_state = SimpleNamespace(step=step, hidden_state=[layer] * 12,
                                         query_points=self.tensor(tyx))
            tracks = tyx[:, None, :, 1:].copy() + np.float32(step / 8)
            logits = np.arange(4096, dtype=np.float32).reshape(1, 1, 8, 512)
            visibility = (np.arange(8, dtype=np.float32) - 4).reshape(1, 1, 8, 1)
            backend._captured.append(((tracks, logits, visibility), next_state))
            xy = tracks[0, 0, :, ::-1].copy()
            if disagree_positions:
                xy[0, 0] += 1
            return xy, visibility[0, 0, :, 0] > 0, next_state

        backend.model = SimpleNamespace(track_frame=track_frame)
        return backend

    def test_native_wrapper_bgr_xy_and_raw_yx_are_exact(self):
        backend = self.backend()
        rgb = np.full((256, 256, 3), [11, 22, 33], dtype=np.uint8)
        query = hand_mapped_query(inputs()["query_xy"], 2)
        arrays, state = backend.forward(rgb, query, None)
        np.testing.assert_array_equal(self.seen[0][0][100, 100], [33, 22, 11])
        np.testing.assert_array_equal(self.seen[0][1], query[0][:, [2, 1]])
        np.testing.assert_array_equal(arrays["positions_xy"], query[0][:, [2, 1]] + np.float32(1 / 8))
        np.testing.assert_array_equal(arrays["coordinate_logits"], np.arange(4096).reshape(8, 512))
        np.testing.assert_array_equal(arrays["visibility_logits"], np.arange(8) - 4)
        self.assertFalse(self.seen[0][3])
        _, next_state = backend.forward(rgb, None, state)
        self.assertIsNone(self.seen[1][1])
        self.assertIs(self.seen[1][2], state)
        self.assertEqual(next_state.step, 2)
        self.assertEqual(len(self.seen), 2)

    def test_native_state_query_and_wrapper_disagreement_fail_closed(self):
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        query = hand_mapped_query(inputs()["query_xy"], 0)
        for options, message in (({"alter_state_query": True}, "query changed"),
                                 ({"disagree_positions": True}, "output disagree")):
            with self.subTest(options=options):
                backend = self.backend(**options)
                with self.assertRaisesRegex(ValueError, message):
                    backend.forward(rgb, query, None)


if __name__ == "__main__":
    unittest.main()
