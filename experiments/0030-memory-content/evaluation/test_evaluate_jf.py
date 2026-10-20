"""Manufactured arrays/files only. Written, not executed during source preparation."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import evaluate_jf as e


class EvaluationTests(unittest.TestCase):
    def test_empty_and_interior_object(self):
        zero = np.zeros((20, 30), bool)
        square = zero.copy()
        square[5:10, 8:14] = True
        self.assertEqual(e.jf(zero, zero), (1., 1.))
        self.assertEqual(e.jf(zero, square), (0., 0.))
        self.assertEqual(e.jf(square, square), (1., 1.))
        # Official boundary convention: full-frame foreground has no internal boundary.
        self.assertEqual(e.jf(zero, ~zero), (0., 1.))

    def test_diagonal_scaled_boundary_tolerance(self):
        self.assertEqual(e.boundary_radius((480, 640)), 7)
        self.assertEqual(e.boundary_radius((480, 854)), 8)
        self.assertEqual(e.boundary_radius((480, 911)), 9)
        a = np.zeros((480, 640), bool)
        a[:, :100] = True
        b, c = np.zeros_like(a), np.zeros_like(a)
        b[:, :107], c[:, :108] = True, True
        self.assertEqual(e.jf(a, b)[1], 1.)
        self.assertEqual(e.jf(a, c)[1], 0.)
        np.testing.assert_array_equal(e.boundary_map(a).sum(axis=0),
                                      np.eye(1, 640, 99, dtype=int)[0] * 480)

    @staticmethod
    def descriptor(path):
        raw = path.read_bytes()
        return {"path": str(path), "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}

    def make_spec(self, folder):
        labels = np.zeros((480, 640), np.uint8)
        labels[10:30, 20:40] = 1
        labels[50:65, 80:95] = 2
        labels[100:105, 100:105] = 255
        spec = {"sequence": "manufactured", "parent_id": "parent",
                "shape_hw": [480, 640], "object_mode": "target", "target_id": 1,
                "frame_ids": [f"{i:05d}" for i in range(5)],
                "predictions": [], "annotations": []}
        for i in range(5):
            gt = folder / f"{i:05d}.png"
            Image.fromarray(labels).save(gt)
            pred = folder / f"{i:05d}.npz"
            mask = labels == 1 if i < 3 else np.zeros_like(labels, bool)
            np.savez(pred, masks=mask[None], object_ids=np.array([1], np.int64))
            spec["predictions"].append(self.descriptor(pred))
            spec["annotations"].append(self.descriptor(gt))
        return spec

    def test_raw_endpoints_preserved_and_single_object_not_union(self):
        with tempfile.TemporaryDirectory() as temp:
            result = e.evaluate_sequence(self.make_spec(Path(temp)))
        rows = result["objects"][0]["frames"]
        self.assertEqual([r["included"] for r in rows], [False, True, True, True, False])
        self.assertEqual([r["J"] for r in rows], [1., 1., 1., 0., 0.])
        self.assertEqual(result["mean"]["JF"], 2 / 3)
        self.assertEqual(result["initial_object_ids"], [1, 2])
        self.assertFalse(result["official_benchmark_result"])

    def test_255_is_background_in_official_semisupervised_path(self):
        labels = np.zeros((20, 30), np.uint8)
        labels[5:10, 5:10] = 255
        self.assertEqual(e.jf(labels == 255, labels == 1)[0], 0.)

    def test_all_objects_and_no_silent_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.make_spec(Path(temp))
            spec.update(object_mode="all", object_ids=[1])
            with self.assertRaisesRegex(ValueError, "omitted initial"):
                e.evaluate_sequence(spec)
            pred = Path(temp) / "overlap.npz"
            m = np.zeros((2, 480, 640), bool)
            m[:, 5:10, 5:10] = True
            np.savez(pred, masks=m, object_ids=np.array([1, 2]))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                e._prediction(pred, [1, 2], (480, 640))

    def test_all_hashes_checked_before_label_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.make_spec(Path(temp))
            Path(spec["predictions"][-1]["path"]).write_bytes(b"corrupt")
            with patch.object(e, "_labels", side_effect=AssertionError("GT decoded")) as decoder:
                with self.assertRaisesRegex(ValueError, "size mismatch"):
                    e.evaluate_sequence(spec)
                decoder.assert_not_called()

    def test_equal_parent_weight_is_not_frame_or_sequence_weight(self):
        def row(name, parent, score):
            mean = dict(J=score, F=score, JF=score)
            return {"sequence": name, "parent_id": parent, "scope": "target",
                    "endpoint_policy": "exclude_first_and_last",
                    "void_policy": "official_background", "reference_commit": e.REFERENCE,
                    "mean": mean, "objects": [{"mean": mean}]}
        report = e.aggregate([row("a", "p", 0.), row("b", "p", 0.),
                              row("c", "q", 1.)])
        self.assertEqual(report["mean"]["JF"], .5)
        self.assertEqual(report["equal_object_mean"]["JF"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
