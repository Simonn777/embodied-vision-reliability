"""Artificial arrays/ZIPs only; never open the real research archive or labels."""
import importlib.util
import io
import json
from pathlib import Path
import pickle
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("oblique_preparation", ROOT / "scripts/prepare_oblique_development.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
TOY_SHAPE = (3, 2, 4, 10)
TOY_FRAMES = (0, 2)


def example():
    t, h, w, n = TOY_SHAPE
    video = np.arange(t * h * w * 3, dtype=np.uint8).reshape(t, h, w, 3)
    points = np.zeros((n, t, 2), dtype=np.float64)
    for i in range(n):
        for frame in range(t):
            points[i, frame] = [i / 20 + frame / 10, i / 30 + frame / 20]
    occ = np.zeros((n, t), dtype=np.bool_)
    occ[[0, 3], 0] = True
    occ[2, 2] = True
    return {"video": video, "points": points, "occluded": occ}


def extract(ex, identity=None, **kwargs):
    return prep.extract_example(ex, prep.array_records(ex) if identity is None else identity,
                                shape=TOY_SHAPE, frame_ids=TOY_FRAMES, **kwargs)


class ExtractionTests(unittest.TestCase):
    def test_fixed_production_contract(self):
        self.assertEqual(prep.GROUPS, (("g-036", 36), ("g-002", 2)))
        self.assertEqual(prep.SHAPE, (250, 256, 256, 30))
        self.assertEqual(prep.FRAME_IDS, tuple(range(0, 250, 16)))
        self.assertEqual(len(prep.FRAME_IDS), 16)

    def test_first_visible_ids_and_axis_order(self):
        ex = example()
        pred, gt, identities = extract(ex)
        np.testing.assert_array_equal(pred["point_ids"], [1, 2, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(pred["frame_ids"], [0, 2])
        self.assertEqual(pred["video"].shape, (2, 2, 4, 3))
        self.assertEqual(gt["points"].shape, (2, 8, 2))
        for row, frame in enumerate([0, 2]):
            for col, point in enumerate(pred["point_ids"]):
                np.testing.assert_array_equal(gt["points"][row, col], ex["points"][point, frame] * [4, 2])
                self.assertEqual(gt["occluded"][row, col], ex["occluded"][point, frame])
        np.testing.assert_array_equal(pred["query_xy"], (ex["points"][pred["point_ids"], 0] * [4, 2]).astype(np.float32))
        self.assertEqual(pred["point_ids"].dtype, np.int64)
        self.assertEqual(pred["frame_ids"].dtype, np.int64)
        self.assertEqual(pred["query_xy"].dtype, np.float32)
        self.assertEqual(gt["points"].dtype, np.float64)
        self.assertEqual(gt["occluded"].dtype, np.bool_)
        self.assertEqual(identities, prep.array_records(ex))

    def test_predictor_has_no_future_truth_and_inputs_unchanged(self):
        ex = example()
        before = prep.array_records(ex)
        pred1, _, _ = extract(ex)
        ex["points"][:, 1:] += 15
        ex["occluded"][:, 1:] = ~ex["occluded"][:, 1:]
        pred2, gt2, _ = extract(ex)
        self.assertEqual(set(pred2), {"video", "query_xy", "point_ids", "frame_ids"})
        for key in pred1:
            np.testing.assert_array_equal(pred1[key], pred2[key])
        self.assertGreater(gt2["points"][1].max(), 1)
        self.assertEqual(before["video"], prep.array_records(ex)["video"])
        self.assertFalse(np.shares_memory(pred2["video"], ex["video"]))

    def test_original_identity_mismatch_stops(self):
        ex = example()
        identity = prep.array_records(ex)
        ex["points"][1, 1, 0] += .01
        with self.assertRaisesRegex(ValueError, "identity"):
            extract(ex, identity)

    def test_insufficient_queries_no_replacement(self):
        ex = example()
        ex["occluded"][1, 0] = True
        with self.assertRaisesRegex(ValueError, "insufficient"):
            extract(ex)

    def test_query_bounds_but_future_outside_is_preserved(self):
        ex = example()
        ex["points"][1, 2] = [-2, 3]
        _, gt, _ = extract(ex)
        np.testing.assert_array_equal(gt["points"][1, 0], [-8, 6])
        ex["points"][1, 0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "query outside"):
            extract(ex)

    def test_array_contract_and_finite_values(self):
        for key, value in (("video", example()["video"].astype(np.float64)),
                           ("points", example()["points"].astype(np.float32)),
                           ("occluded", example()["occluded"].astype(np.uint8)),
                           ("points", example()["points"].transpose(1, 0, 2))):
            ex = example()
            ex[key] = value
            with self.subTest(key=key, dtype=value.dtype, shape=value.shape), self.assertRaises(ValueError):
                extract(ex)
        ex = example()
        ex["points"][0, 2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            extract(ex)


class ArchiveTests(unittest.TestCase):
    def in_memory_zip(self, names, modes=None):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            for i, name in enumerate(names):
                info = zipfile.ZipInfo(name)
                if modes:
                    info.external_attr = modes[i] << 16
                z.writestr(info, b"payload")
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_zip_path_symlink_and_multiple_pickle_rejected(self):
        for names, modes in ((["../x.pkl"], None), (["/x.pkl"], None),
                             (["a\\x.pkl"], None), (["C:x.pkl"], None),
                             (["x.pkl"], [stat.S_IFLNK | 0o777]),
                             (["x.pkl", "y.pkl"], None), (["x.txt"], None)):
            with self.subTest(names=names), self.in_memory_zip(names, modes) as z:
                with self.assertRaises(ValueError):
                    prep.pickle_member(z)

    def test_zip_size_before_decode(self):
        with self.in_memory_zip(["safe/x.pkl"]) as z:
            z.infolist()[0].file_size = prep.MAX_UNCOMPRESSED + 1
            with self.assertRaisesRegex(ValueError, "4 GiB"):
                prep.pickle_member(z)

    def test_numpy_only_unpickler_rejects_global_and_persistent(self):
        _, reader, inspect, _ = prep.dependencies()
        for payload in (b"cbuiltins\neval\n.", b"Panything\n."):
            with self.assertRaises(pickle.UnpicklingError):
                reader(io.BytesIO(payload)).load()
        with self.assertRaises(pickle.UnpicklingError):
            inspect(io.BytesIO(b"\x80\x02\x82\x01."))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / "fake.zip"
        self.manifest = self.root / "fake-manifest.json"
        self.output = self.root / "new-output"
        self.data = [example() for _ in range(50)]
        self.write_archive()
        self.meta = {"archive": {"bytes": self.archive.stat().st_size,
                                  "sha256": prep.file_digest(self.archive)},
                     "groups": [{"group_id": group, "members": [index], "partition": "reserve"}
                                for group, index in prep.GROUPS],
                     "allocation": [{"archive_index": index, "group_id": group, "partition": "reserve"}
                                    for group, index in prep.GROUPS],
                     "identities": [{"archive_index": i, "arrays": prep.array_records(ex)}
                                    for i, ex in enumerate(self.data)]}
        self.manifest.write_text(json.dumps(self.meta))

    def write_archive(self, extra=b""):
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("toy.pkl", pickle.dumps(self.data, protocol=4) + extra)

    def run_preparation(self, **kwargs):
        original_extract = prep.extract_example
        def tiny(ex, identity):
            return original_extract(ex, identity, shape=TOY_SHAPE, frame_ids=TOY_FRAMES)
        with mock.patch.multiple(prep, ARCHIVE_BYTES=self.archive.stat().st_size,
                                 ARCHIVE_SHA256=prep.file_digest(self.archive),
                                 MANIFEST_BYTES=self.manifest.stat().st_size,
                                 MANIFEST_SHA256=prep.file_digest(self.manifest)), \
                mock.patch.object(prep, "extract_example", side_effect=tiny):
            return prep.prepare(self.archive, self.manifest, self.output, **kwargs)

    def test_complete_contract_hashes_and_originals_preserved(self):
        before = [prep.file_digest(p) for p in (self.archive, self.manifest)]
        receipt = self.run_preparation()
        self.assertEqual(receipt["status"], "completed_preparation")
        self.assertEqual(set(receipt["files"]), {g + suffix for g, _ in prep.GROUPS
                                               for suffix in (".predictor.npz", ".ground_truth.npz")})
        self.assertEqual([r["group_id"] for r in receipt["groups"]], ["g-036", "g-002"])
        self.assertEqual(len(list(self.output.iterdir())), 6)
        for record in receipt["groups"]:
            for kind in ("predictor", "ground_truth"):
                artifact = record[kind]
                path = self.output / artifact["name"]
                self.assertEqual(artifact["sha256"], prep.file_digest(path))
                with np.load(path, allow_pickle=False) as arrays:
                    self.assertEqual(prep.array_records(dict(arrays)), artifact["arrays"])
        self.assertEqual(before, [prep.file_digest(p) for p in (self.archive, self.manifest)])
        self.assertTrue(receipt["pickle_member"]["crc_checked_to_member_eof"])

    def test_existing_directory_and_attempt_marker_stop_retry(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_preparation()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_wrong_identity_failure_preserved_and_no_outputs(self):
        self.meta["identities"][36]["arrays"]["points"]["sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_preparation()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(set(before), {"started.json", "receipt.json"})
        self.assertEqual(json.loads(before["receipt.json"])["status"], "failed")
        with self.assertRaises(FileExistsError):
            self.run_preparation()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_reserve_order_or_partition_mismatch(self):
        self.meta["groups"].reverse()
        self.manifest.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "reserve"):
            self.run_preparation()

    def test_trailing_pickle_data_rejected(self):
        self.write_archive(extra=b"unexpected")
        self.meta["archive"] = {"bytes": self.archive.stat().st_size,
                                "sha256": prep.file_digest(self.archive)}
        self.manifest.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "trailing"):
            self.run_preparation()

    def test_wrong_list_length_stops_without_export(self):
        self.data.pop()
        self.write_archive()
        self.meta["archive"] = {"bytes": self.archive.stat().st_size,
                                "sha256": prep.file_digest(self.archive)}
        self.manifest.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "50-example"):
            self.run_preparation()
        self.assertEqual({p.name for p in self.output.iterdir()}, {"started.json", "receipt.json"})

    def test_partial_preserved_after_write_failure(self):
        def broken(stream, **arrays):
            stream.write(b"incomplete")
            raise OSError("artificial output failure")
        with mock.patch.object(np, "savez", side_effect=broken):
            with self.assertRaises(OSError):
                self.run_preparation()
        self.assertEqual((self.output / "g-036.predictor.npz.partial").read_bytes(), b"incomplete")
        self.assertFalse((self.output / "g-036.predictor.npz").exists())
        self.assertEqual(json.loads((self.output / "receipt.json").read_text())["status"], "failed")

    def test_pins_and_symlinks_fail_before_unpickling(self):
        size = self.archive.stat().st_size
        for expected_size, digest in ((size + 1, prep.file_digest(self.archive)), (size, "0" * 64)):
            with self.assertRaises(ValueError):
                prep.pinned_open(self.archive, expected_size, digest)
        link = self.root / "link.zip"
        link.symlink_to(self.archive)
        with self.assertRaises(OSError):
            prep.pinned_open(link, size, prep.file_digest(self.archive))


if __name__ == "__main__":
    unittest.main()
