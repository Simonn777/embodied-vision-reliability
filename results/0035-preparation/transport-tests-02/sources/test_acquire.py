"""Manufactured byte/HTTP tests only. No image, model, network or real archive."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zlib

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


acq = module("_test_vost_acquire", HERE / "acquire.py")
REPO = next((p for p in HERE.parents if (p / acq.HELPER).is_file()),
            HERE.parent / "embodied-vision-reliability")
helper = module("_test_vost_range", REPO / acq.HELPER)


def fixture(*, offset=0, name="VOST/Annotations/fake/frame00000.png", payload=b"manufactured opaque bytes",
            size_mode="zero", zip64_values=None, extra=b"", crc_value=None, packed_override=None):
    compressor = zlib.compressobj(wbits=-15)
    packed = compressor.compress(payload) + compressor.flush()
    if packed_override is not None:
        packed = packed_override
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    if size_mode == "zero":
        csize = usize = 0
    elif size_mode == "correct":
        csize, usize = len(packed), len(payload)
    elif size_mode == "zip64":
        csize = usize = 0xFFFFFFFF
        values = zip64_values if zip64_values is not None else (len(payload), len(packed))
        extra = struct.pack("<HH2Q", 1, 16, *values) + extra
    else:
        raise ValueError(size_mode)
    raw = struct.pack("<4s5H3I2H", b"PK\x03\x04", 45, 8, 8, 7, 9,
                      0 if crc_value is None else crc_value, csize, usize, len(name), len(extra))
    data = raw + name.encode() + extra + packed
    descriptor = struct.pack("<4I", 0x08074B50, crc, len(packed), len(payload))
    entry = dict(filename=name, local_header_offset=offset, compressed_bytes=len(packed),
                 uncompressed_bytes=len(payload), crc32_hex=f"{crc:08x}", version_needed=45,
                 flags=8, compression_method=8, mod_time=7, mod_date=9,
                 next_local_header_or_central_offset=offset + len(data) + len(descriptor),
                 phase="initial", sequence="fake", frame_index=0, stem=Path(name).stem)
    return data + descriptor, entry, payload


class ByteReader:
    def __init__(self, body, entry):
        self.body, self.base, self.calls = body, entry["local_header_offset"], []
        self.index = types.SimpleNamespace(directory_offset=self.base + len(body) + 100)

    def read(self, start, end):
        self.calls.append((start, end))
        if not self.base <= start <= end < self.base + len(self.body):
            raise AssertionError("fixture range")
        return self.body[start - self.base:end - self.base + 1]


class Response:
    status = 206

    def __init__(self, body, start, end, index):
        self.body, self.cursor = body[start:end + 1], 0
        self.headers = {"ETag": index.etag, "Content-Length": str(end - start + 1),
                        "Content-Range": f"bytes {start}-{end}/{index.archive_bytes}"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, count):
        value = self.body[self.cursor:self.cursor + count]
        self.cursor += len(value)
        return value


class AcquisitionTests(unittest.TestCase):
    def test_flag8_zero_correct_crc_and_64bit_offset(self):
        for mode in ("zero", "correct"):
            body, entry, payload = fixture(offset=(1 << 35) + 17, size_mode=mode)
            reader = ByteReader(body, entry)
            result, details = acq.extract_bounded(helper, reader, entry)
            self.assertEqual(result, payload)
            self.assertEqual(len(reader.calls), 3)
            self.assertFalse(details["data_descriptor_read"])
            self.assertEqual(reader.calls[-1][1] + 1, entry["next_local_header_or_central_offset"] - 16)

    def test_zip64_correct_and_zero_placeholders(self):
        for values in (None, (0, 0)):
            body, entry, payload = fixture(size_mode="zip64", zip64_values=values)
            result, _ = acq.extract_bounded(helper, ByteReader(body, entry), entry)
            self.assertEqual(result, payload)
        body, entry, _ = fixture(size_mode="zip64", zip64_values=(999, 0))
        with self.assertRaisesRegex(ValueError, "contradicts"):
            acq.extract_bounded(helper, ByteReader(body, entry), entry)

    def test_identity_and_conflicting_sizes_rejected_before_payload(self):
        body, entry, payload = fixture()
        changes = ((4, "<H", 20), (6, "<H", 0), (8, "<H", 0), (10, "<H", 8),
                   (14, "<I", 123), (18, "<I", 1), (22, "<I", 999), (26, "<H", 1))
        for offset, fmt, value in changes:
            changed = bytearray(body)
            struct.pack_into(fmt, changed, offset, value)
            reader = ByteReader(changed, entry)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                acq.extract_bounded(helper, reader, entry)
            self.assertEqual(len(reader.calls), 1)
        changed = bytearray(body)
        changed[30] ^= 1
        reader = ByteReader(changed, entry)
        with self.assertRaisesRegex(ValueError, "filename"):
            acq.extract_bounded(helper, reader, entry)
        self.assertEqual(len(reader.calls), 2)

    def test_extra_bound_duplicate_missing_and_truncated_zip64(self):
        cases = (
            dict(extra=b"x" * 4097),
            dict(size_mode="zip64", extra=struct.pack("<HH", 1, 0)),
            dict(extra=b"\x02\x00\xff"),
        )
        for kwargs in cases:
            body, entry, _ = fixture(**kwargs)
            reader = ByteReader(body, entry)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                acq.extract_bounded(helper, reader, entry)
            self.assertLessEqual(len(reader.calls), 2)
        body, entry, _ = fixture()
        changed = bytearray(body)
        struct.pack_into("<I", changed, 18, 0xFFFFFFFF)
        with self.assertRaisesRegex(ValueError, "ZIP64"):
            acq.extract_bounded(helper, ByteReader(changed, entry), entry)

    def test_physical_overlap_and_decompression_failures(self):
        body, entry, payload = fixture()
        overlapping = dict(entry, next_local_header_or_central_offset=entry["local_header_offset"] + 30)
        reader = ByteReader(body, overlapping)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            acq.extract_bounded(helper, reader, overlapping)
        self.assertEqual(len(reader.calls), 1)
        wrong_crc = dict(entry, crc32_hex="00000001")
        with self.assertRaisesRegex(ValueError, "CRC"):
            acq.extract_bounded(helper, ByteReader(body, wrong_crc), wrong_crc)
        for packed in (b"\x03", b"\x03\x00trailing"):
            bad, row, _ = fixture(packed_override=packed)
            with self.subTest(packed=packed), self.assertRaises((ValueError, zlib.error)):
                acq.extract_bounded(helper, ByteReader(bad, row), row)

    def test_success_receipt_and_source_postcheck(self):
        body, entry, _ = fixture()
        index = helper.Index("https://invalid.example/test.zip", '"fake"', len(body) + 100,
                             len(body) + 1, {entry["filename"]: entry}, {"manifest": "fake"})
        opener = lambda request, timeout: Response(body, *map(int, request.get_header("Range")[6:].split("-")), index)
        postcheck = Mock()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(acq.PHASE_COUNTS, initial=1):
            out = Path(tmp) / "output"
            result = acq.acquire_phase(helper, {"archive": {}}, index, "initial", out,
                                      commit="a" * 40, bindings={}, postcheck=postcheck, opener=opener)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["members"]), 1)
            self.assertEqual(len(result["requests"]), 3)
            self.assertEqual(result["requested_body_upper_bound"], result["response_body_bytes"] + 3)
            self.assertEqual(json.loads((out / "receipt.json").read_text())["status"], "completed")
            postcheck.assert_called_once()

    def test_interruption_preserves_completed_member_and_failed_request(self):
        first, e1, _ = fixture()
        second, e2, _ = fixture(offset=len(first), name="VOST/Annotations/other/frame00000.png")
        body = first + second
        index = helper.Index("https://invalid.example/test.zip", '"fake"', len(body) + 100,
                             len(body) + 1, {e1["filename"]: e1, e2["filename"]: e2}, {"manifest": "fake"})
        calls = []
        def opener(request, timeout):
            calls.append(request)
            if len(calls) == 4:
                raise KeyboardInterrupt("manufactured interruption")
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            return Response(body, start, end, index)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(acq.PHASE_COUNTS, initial=2):
            out = Path(tmp) / "output"
            with self.assertRaises(KeyboardInterrupt):
                acq.acquire_phase(helper, {"archive": {}}, index, "initial", out,
                                  commit="a" * 40, bindings={}, postcheck=Mock(), opener=opener)
            saved = json.loads((out / "receipt.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error_type"], "KeyboardInterrupt")
            self.assertEqual(len(saved["members"]), 1)
            self.assertEqual(len(saved["requests"]), 4)
            self.assertTrue((out / saved["members"][0]["saved_relative"]).is_file())

    def test_metric_default_closed_before_http_and_rgb_needs_initial(self):
        index = helper.Index("https://invalid.example/test.zip", '"fake"', 100, 99, {}, {"manifest": "fake"})
        opener = Mock(side_effect=AssertionError("must not call transport"))
        with tempfile.TemporaryDirectory() as tmp:
            for phase, initial, message in (("metric", {"members": 4}, "metric phase closed"),
                                            ("rgb", None, "initial masks")):
                with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, message):
                    acq.acquire_phase(helper, {"archive": {}}, index, phase, Path(tmp) / phase,
                                      commit="a" * 40, bindings={}, postcheck=Mock(),
                                      initial_admission=initial, opener=opener)
                self.assertEqual(json.loads((Path(tmp) / phase / "receipt.json").read_text())["status"], "failed")
            opener.assert_not_called()

    def test_wrong_source_blocks_helper_import_and_proposal_is_not_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(repo_root=tmp, metadata_root=tmp, output_dir=str(Path(tmp) / "external"))
            args.output_dir = str(Path(tmp).parent / ("not-created-" + Path(tmp).name))
            args.expected_commit, args.manifest_relative, args.admission_relative = "a" * 40, "manifest.json", "admission.json"
            with patch.object(acq, "source_bindings", side_effect=ValueError("source differs")), \
                 patch.object(acq, "load_module") as loader:
                with self.assertRaisesRegex(ValueError, "source differs"):
                    acq.run(args)
                loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "admission"):
            acq.validate_admission({"allocated": False, "acquisition_authorized": False})

    def test_inspection_gate_binds_initial_source_outer_and_resource_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            initial_dir = root / "initial"
            initial_dir.mkdir()
            initial_path = initial_dir / "receipt.json"
            initial_path.write_text('{"manufactured":true}')
            source = root / "ADMISSION.md"
            source.write_text("manufactured frozen admission")
            path, outer = root / "inspection.json", root / "outer.json"
            outer.write_text('{"manufactured":true}')
            record = dict(schema="evr0035.initial-inspection.v1", status="completed", eligible=True,
                          initial_only=True, noninitial_labels_read=0, model_calls=0,
                          prediction_output_cap_bytes=100000,
                          fingerprints={str(initial_path): acq.sha(initial_path.read_bytes())},
                          source_sha256={str(source): acq.sha(source.read_bytes())},
                          checks={k: True for k in ("native_object_frames_at_most_2000",
                              "planning_memory_at_most_7GiB", "prediction_output_at_most_20GiB",
                              "free_disk_covers_twice_output_plus_5GiB")})
            def bound_json(desc, fingerprints):
                raw = Path(desc["path"]).read_bytes()
                self.assertEqual(acq.sha(raw), desc["sha256"])
                return json.loads(raw)
            admission = types.SimpleNamespace(bound_json=bound_json, clean_outer=Mock())
            initial = {"receipt_sha256": acq.sha(initial_path.read_bytes()), "members": 4}
            bindings = {acq.ADMISSION: acq.sha(source.read_bytes())}
            path.write_text(json.dumps(record))
            result = acq.verify_inspection(path, outer, initial_dir, initial, root, bindings, admission)
            self.assertEqual(result["inspection_sha256"], acq.sha(path.read_bytes()))
            self.assertEqual(result["inspection_outer_sha256"], acq.sha(outer.read_bytes()))
            self.assertEqual(result["prediction_output_cap_bytes"], 100000)
            admission.clean_outer.assert_called_once()
            record["checks"]["planning_memory_at_most_7GiB"] = False
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "eligible"):
                acq.verify_inspection(path, outer, initial_dir, initial, root, bindings, admission)
            record["checks"]["planning_memory_at_most_7GiB"] = True
            record["fingerprints"][str(initial_path)] = "0" * 64
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "different initial"):
                acq.verify_inspection(path, outer, initial_dir, initial, root, bindings, admission)
            record["fingerprints"][str(initial_path)] = initial["receipt_sha256"]
            path.write_text(json.dumps(record))
            source.write_text("mutated")
            with self.assertRaisesRegex(ValueError, "source changed"):
                acq.verify_inspection(path, outer, initial_dir, initial, root, bindings, admission)


if __name__ == "__main__":
    unittest.main()
