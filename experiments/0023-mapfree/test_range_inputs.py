"""Artificial ZIP/range tests only: no network, datasets, image decode or GT."""
from http.client import IncompleteRead
from email.message import Message
import importlib.util
import io
import json
from pathlib import Path
import signal
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

spec = importlib.util.spec_from_file_location("evr0023_range_inputs", Path(__file__).with_name("range_inputs.py"))
acq = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = acq
spec.loader.exec_module(acq)
IMAGE = "val/s00001/seq0/frame_00000.jpg"
K = "val/s00001/intrinsics.txt"
GT = "val/s00001/poses.txt"
ETAG = '"artificial-zip-identity"'


class Response:
    def __init__(self, body, start, end, archive_bytes):
        self.body, self.read_calls = io.BytesIO(body), 0
        self.status = 206
        self.headers = {"ETag": ETAG, "Content-Range": "bytes %d-%d/%d" % (start, end, archive_bytes),
                        "Content-Length": str(end - start + 1)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.body.close()

    def read(self, n):
        self.read_calls += 1
        return self.body.read(n)


class FakeHTTP:
    def __init__(self, blob, mutate=None):
        self.blob, self.mutate = blob, mutate
        self.calls, self.responses = [], []

    def __call__(self, request, timeout):
        assert request.full_url == acq.URL
        assert request.get_header("If-match") == ETAG
        assert request.get_header("Accept-encoding") == "identity"
        assert 0 < timeout <= 30
        start, end = map(int, request.get_header("Range")[6:].split("-"))
        self.calls.append((start, end))
        response = Response(self.blob[start:end + 1], start, end, len(self.blob))
        if self.mutate:
            self.mutate(response)
        self.responses.append(response)
        return response


class RangeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {k: self.root / k for k in ("members", "directory", "receipt")}
        self.output, self.outreceipt = self.root / "output", self.root / "acquisition.json"
        # No image semantics or scientific K/pose values: arbitrary bounded bytes.
        self.contents = {IMAGE: b"artificial JPEG member" * 31,
                         K: b"artificial intrinsics bytes\n", GT: b"forbidden pose placeholder"}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in self.contents.items():
                z.writestr(name, data)
        self.blob = buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(self.blob)) as z:
            self.offset = z.start_dir
            self.entries = [{"filename": x.filename, "flags": x.flag_bits,
                             "compression_method": x.compress_type, "crc32_hex": "%08x" % x.CRC,
                             "compressed_bytes": x.compress_size, "uncompressed_bytes": x.file_size,
                             "local_header_offset": x.header_offset} for x in z.infolist()]
        self.directory = self.blob[self.offset:-22]
        self.write_index()
        self.limits = acq.Limits(100000, 100000, 10000, 12, 60, 30)
        self.network_guard = mock.patch.object(acq.urllib.request, "build_opener",
                                               side_effect=AssertionError("network prohibited in artificial tests"))
        self.network_guard.start()

    def tearDown(self):
        self.network_guard.stop()
        self.temp.cleanup()

    def write_index(self):
        m = dict(url=acq.URL, etag=ETAG, archive_bytes=len(self.blob), entries=self.entries)
        self.paths["members"].write_text(json.dumps(m))
        self.paths["directory"].write_bytes(self.directory)
        end = self.offset + len(self.directory) - 1
        receipt = dict(schema="evr0018.resource-directory.v1", status="completed", url=acq.URL,
                       payload_members_read=0, head=dict(status=200, etag=ETAG, archive_bytes=len(self.blob)),
                       members_sha256=acq.sha(self.paths["members"].read_bytes()),
                       central_directory=dict(offset=self.offset, bytes=len(self.directory), entries=len(self.entries)),
                       requests=[dict(kind="central directory only", status=206, etag=ETAG,
                                      requested_range=[self.offset, end], body_bytes=len(self.directory),
                                      content_range="bytes %d-%d/%d" % (self.offset, end, len(self.blob)),
                                      sha256=acq.sha(self.directory))])
        self.paths["receipt"].write_text(json.dumps(receipt))
        self.hashes = {k: acq.sha(p.read_bytes()) for k, p in self.paths.items()}

    def index(self):
        return acq.load_index(self.paths["members"], self.paths["directory"], self.paths["receipt"], self.hashes)

    def run_acquire(self, http=None, names=(IMAGE, K), limits=None):
        return acq.acquire(self.index(), names, self.output, self.outreceipt,
                           limits or self.limits, http or FakeHTTP(self.blob))

    def test_two_inputs_exact_ranges_crc_sha_and_no_gt_payload(self):
        http = FakeHTTP(self.blob)
        result = self.run_acquire(http)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["index_sha256"], self.hashes)
        self.assertEqual(len(http.calls), 6)
        self.assertEqual(result["response_body_bytes"], sum(b - a + 1 for a, b in http.calls))
        self.assertEqual(result["requested_body_upper_bound"], result["response_body_bytes"] + 6)
        self.assertTrue(result["response_body_bytes_exact"])
        self.assertFalse((self.output / GT).exists())
        for member in result["members"]:
            name = member["filename"]
            self.assertEqual((self.output / name).read_bytes(), self.contents[name])
            self.assertEqual(member["sha256"], acq.sha(self.contents[name]))
        forbidden_start = self.entries[2]["local_header_offset"]
        self.assertTrue(all(b < forbidden_start for a, b in http.calls))

    def test_gt_unsafe_and_unlisted_members_fail_before_http(self):
        for names in ((GT,), ("val/s00001/poses_device.txt",), ("../intrinsics.txt",),
                      ("val/s00001/seq1/frame_12345.jpg",), (K, K)):
            with self.subTest(names=names):
                http = FakeHTTP(self.blob)
                with self.assertRaises(ValueError):
                    self.run_acquire(http, names)
                self.assertEqual(http.calls, [])
                self.assertFalse(self.output.exists())

    def test_hash_and_inventory_disagreement_are_local_barriers(self):
        self.paths["members"].write_bytes(self.paths["members"].read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.index()
        self.entries[0]["uncompressed_bytes"] += 1
        self.write_index()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.index()

    def test_zip64_offset_only_and_malformed_extra(self):
        raw = bytearray(self.directory[:46 + len(IMAGE)])
        struct.pack_into("<I", raw, 42, 0xffffffff)
        struct.pack_into("<H", raw, 30, 12)
        offset = 2**32 + 123
        raw.extend(struct.pack("<HHQ", 1, 8, offset))
        parsed = acq.parse_directory(bytes(raw), offset + 10000, offset + 9000)
        self.assertEqual(parsed[IMAGE]["local_header_offset"], offset)
        with self.assertRaisesRegex(ValueError, "truncated"):
            acq.extra_fields(b"\x01\x00\x08\x00abc")

    def test_symlink_encoding_and_duplicate_central_entries_rejected(self):
        raw = bytearray(self.directory)
        struct.pack_into("<I", raw, 38, 0o120777 << 16)
        with self.assertRaisesRegex(ValueError, "non-regular"):
            acq.parse_directory(bytes(raw), len(self.blob), self.offset)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            acq.parse_directory(self.directory * 2, len(self.blob), self.offset)

    def test_changed_etag_status_range_length_encoding_refuses_body(self):
        changes = [lambda r: r.headers.update(ETag='"different"'),
                   lambda r: setattr(r, "status", 200),
                   lambda r: r.headers.update({"Content-Range": "bytes 0-29/1"}),
                   lambda r: r.headers.update({"Content-Length": "29"}),
                   lambda r: r.headers.update({"Content-Encoding": "gzip"})]
        for change in changes:
            with self.subTest(change=change):
                http = FakeHTTP(self.blob, change)
                receipt = dict(response_body_bytes=0, response_body_bytes_exact=True,
                               requested_body_upper_bound=0, requests=[])
                reader = acq.RangeReader(self.index(), self.limits, receipt, lambda: None, http)
                with self.assertRaisesRegex(ValueError, "header mismatch"):
                    reader.read(0, 29)
                self.assertEqual(http.responses[0].read_calls, 0)
                self.assertEqual(receipt["response_body_bytes"], 0)
                self.assertEqual(len(http.calls), 1)

    def test_header_and_filename_tamper_stops_before_payload(self):
        for where, expected_calls in ((14, 1), (30, 2)):
            with self.subTest(where=where):
                blob = bytearray(self.blob)
                blob[where] ^= 1
                http = FakeHTTP(bytes(blob))
                reader = acq.RangeReader(self.index(), self.limits,
                                        dict(response_body_bytes=0, response_body_bytes_exact=True,
                                             requested_body_upper_bound=0, requests=[]), lambda: None, http)
                with self.assertRaises(ValueError):
                    acq.extract_member(reader, self.index().entries[IMAGE])
                self.assertEqual(len(http.calls), expected_calls)

    def test_crc_and_uncompressed_length_checked_after_inflate(self):
        for crc_change, size_change, reason in ((1, 0, "CRC32"), (0, -1, "size")):
            with self.subTest(reason=reason):
                index = self.index()
                entry = dict(index.entries[IMAGE])
                blob = bytearray(self.blob)
                entry["crc32_hex"] = "%08x" % (int(entry["crc32_hex"], 16) ^ crc_change)
                entry["uncompressed_bytes"] += size_change
                struct.pack_into("<I", blob, 14, int(entry["crc32_hex"], 16))
                struct.pack_into("<I", blob, 22, entry["uncompressed_bytes"])
                reader = acq.RangeReader(index, self.limits,
                                        dict(response_body_bytes=0, response_body_bytes_exact=True,
                                             requested_body_upper_bound=0, requests=[]), lambda: None, FakeHTTP(bytes(blob)))
                with self.assertRaisesRegex(ValueError, reason):
                    acq.extract_member(reader, entry)

    def test_caps_and_existing_paths_refuse_without_http(self):
        http = FakeHTTP(self.blob)
        with self.assertRaisesRegex(ValueError, "caps"):
            self.run_acquire(http, limits=acq.Limits(1, 100000, 10000, 12, 60, 30))
        self.assertEqual(http.calls, [])
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_acquire(http)
        self.assertEqual(http.calls, [])

    def test_incomplete_read_preserves_known_partial_bytes_and_alarm(self):
        def incomplete(response):
            response.read = mock.Mock(side_effect=IncompleteRead(b"abc", 27))
        previous = signal.getsignal(signal.SIGALRM)
        http = FakeHTTP(self.blob, incomplete)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_acquire(http)
        receipt = json.loads(self.outreceipt.read_text())
        self.assertEqual(receipt["status"], "error")
        self.assertEqual(receipt["response_body_bytes"], 3)
        self.assertTrue(receipt["response_body_bytes_exact"])
        self.assertEqual(receipt["requests"][0]["body_sha256"], acq.sha(b"abc"))
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_interruption_has_durable_failed_receipt_without_retry(self):
        def interrupt(response):
            response.read = mock.Mock(side_effect=KeyboardInterrupt("artificial"))
        http = FakeHTTP(self.blob, interrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.run_acquire(http)
        receipt = json.loads(self.outreceipt.read_text())
        self.assertEqual(receipt["status"], "error")
        self.assertFalse(receipt["response_body_bytes_exact"])
        self.assertEqual(receipt["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(len(http.calls), 1)

    def test_wall_timeout_before_http_and_short_response(self):
        http = FakeHTTP(self.blob)
        receipt = dict(response_body_bytes=0, response_body_bytes_exact=True,
                       requested_body_upper_bound=0, requests=[])
        reader = acq.RangeReader(self.index(), self.limits, receipt, lambda: None, http)
        reader.started -= 61
        with self.assertRaises(TimeoutError):
            reader.read(0, 29)
        self.assertEqual(http.calls, [])
        def shorten(r):
            r.body = io.BytesIO(r.body.getvalue()[:-1])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            self.run_acquire(FakeHTTP(self.blob, shorten))
        self.assertEqual(json.loads(self.outreceipt.read_text())["response_body_bytes"], 29)

    def test_duplicate_http_identity_headers_refused_before_body(self):
        for name in ("ETag", "Content-Range", "Content-Length", "Content-Encoding"):
            with self.subTest(name=name):
                def duplicate(response):
                    headers = Message()
                    for key, value in response.headers.items():
                        headers[key] = value
                    if name == "Content-Encoding":
                        headers[name] = "identity"
                    headers[name] = headers[name]
                    response.headers = headers
                http = FakeHTTP(self.blob, duplicate)
                receipt = dict(response_body_bytes=0, response_body_bytes_exact=True,
                               requested_body_upper_bound=0, requests=[])
                reader = acq.RangeReader(self.index(), self.limits, receipt, lambda: None, http)
                with self.assertRaisesRegex(ValueError, "duplicate HTTP"):
                    reader.read(0, 29)
                self.assertEqual(http.responses[0].read_calls, 0)

    def test_slow_final_member_persistence_cannot_mark_completed(self):
        clock = [0.0]
        real_dump = json.dump
        def slow_dump(obj, stream, **kwargs):
            real_dump(obj, stream, **kwargs)
            if obj.get("members") and obj["status"] == "started":
                clock[0] = 61.0
        with mock.patch.object(acq.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(acq.json, "dump", side_effect=slow_dump):
            with self.assertRaisesRegex(TimeoutError, "after output persistence"):
                self.run_acquire(names=(IMAGE,))
        receipt = json.loads(self.outreceipt.read_text())
        self.assertEqual(receipt["status"], "error")
        self.assertEqual(receipt["wall_seconds"], 61.0)
        self.assertEqual(len(receipt["members"]), 1)


if __name__ == "__main__":
    unittest.main()
