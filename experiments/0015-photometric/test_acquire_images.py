"""Artificial ZIP/HTTP tests; no network, HDF5, or real images."""
import importlib.util
import http.client
import io
import json
from pathlib import Path
import struct
import signal
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

spec = importlib.util.spec_from_file_location(
    "evr0015_acquire_tested", Path(__file__).with_name("acquire_images.py"))
acq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acq)


def jpeg(index=0, width=64, height=48):
    sof = struct.pack(">BHHB", 8, height, width, 3) + bytes(
        [1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0])
    return (b"\xff\xd8\xff\xc0" + struct.pack(">H", len(sof) + 2)
            + sof + b"\xff\xda\x00\x02" + bytes([index]) * 80 + b"\xff\xd9")


class Response:
    def __init__(self, body, start, end, status=206, bad_range=False):
        self.body = body
        self.status = status
        self.read_calls = 0
        self.headers = {
            "Content-Range": "bytes %d-%d/%d" % (
                start, end, acq.ARCHIVE_BYTES + int(bad_range)),
            "Content-Length": str(end - start + 1)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, maximum):
        self.read_calls += 1
        return self.body[:maximum]


class FakeHTTP:
    def __init__(self, blob, status=206, bad_range=False, truncate=False):
        self.blob = blob
        self.status = status
        self.bad_range = bad_range
        self.truncate = truncate
        self.calls = []
        self.responses = []

    def __call__(self, request, timeout):
        assert request.full_url == acq.URL
        assert timeout == 30
        assert request.get_header("Accept-encoding") == "identity"
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        self.calls.append((start, end))
        body = self.blob[start:end + 1]
        if self.truncate:
            body = body[:-1]
        response = Response(body, start, end, self.status, self.bad_range)
        self.responses.append(response)
        return response


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.archive = self.root / "archive.json"
        self.output = self.root / "private-images"
        self.receipt = self.root / "public-receipt.json"
        keys = ["0015-img%d.jpg_0015-img%d.jpg" % (i, 16 + i % 7)
                for i in range(16)]
        self.m = {"schema": "evr0012.development.v1", "development_scene": "0015",
                  "selected_pair_keys": keys}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for i in range(23):
                archive.writestr("images/0015/img%d.jpg" % i, jpeg(i))
        self.blob = buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(self.blob)) as archive:
            entries = [{"filename": x.filename, "local_header_offset": x.header_offset,
                        "compression_method": x.compress_type, "flags": x.flag_bits,
                        "compressed_bytes": x.compress_size,
                        "uncompressed_bytes": x.file_size, "crc32_hex": "%08x" % x.CRC}
                       for x in archive.infolist()]
        self.a = {"schema": "evr0015.archive-metadata.v1", "url": acq.URL,
                  "archive_bytes": acq.ARCHIVE_BYTES, "selected_entries": entries}
        self.write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self):
        self.manifest.write_text(json.dumps(self.m))
        self.archive.write_text(json.dumps(self.a))

    def run_acquire(self, http):
        return acq.acquire(
            self.manifest, self.archive, self.output, self.receipt, http,
            expected_manifest_sha256=acq.sha(self.manifest.read_bytes()),
            expected_archive_sha256=acq.sha(self.archive.read_bytes()))

    def test_complete_all23_exact_ranges_hashes_and_public_receipt(self):
        http = FakeHTTP(self.blob)
        result = self.run_acquire(http)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["images"]), 23)
        self.assertEqual(len(http.calls), 69)
        self.assertEqual(result["network_body_bytes"], sum(b - a + 1 for a, b in http.calls))
        self.assertEqual(len(list(self.output.rglob("*.jpg"))), 23)
        self.assertNotIn(str(self.root), self.receipt.read_text())
        for item in result["images"]:
            data = (self.output / item["filename"]).read_bytes()
            self.assertEqual(acq.sha(data), item["sha256"])
            self.assertEqual((item["width"], item["height"]), (64, 48))
            self.assertFalse(item["pixel_decoding_performed"])
        self.assertEqual(result["manifest_sha256"], acq.sha(self.manifest.read_bytes()))

    def test_existing_output_or_receipt_refuses_without_network(self):
        for which in ("directory", "receipt"):
            with self.subTest(which=which):
                if which == "directory":
                    self.output.mkdir()
                else:
                    self.receipt.write_text("preserve")
                http = FakeHTTP(self.blob)
                with self.assertRaises(FileExistsError):
                    self.run_acquire(http)
                self.assertEqual(http.calls, [])
                if which == "directory":
                    self.output.rmdir()
                else:
                    self.assertEqual(self.receipt.read_text(), "preserve")

    def test_unselected_scene_or_entry_rejected_before_network(self):
        self.a["selected_entries"][0]["filename"] = "images/0022/other.jpg"
        self.write_inputs()
        http = FakeHTTP(self.blob)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            self.run_acquire(http)
        self.assertFalse(self.output.exists())
        self.assertEqual(http.calls, [])
        self.assertEqual(json.loads(self.receipt.read_text())["status"], "error")

    def test_frozen_input_hash_rejection_before_network(self):
        original_hash = acq.sha(self.manifest.read_bytes())
        self.manifest.write_text(self.manifest.read_text() + "\n")
        http = FakeHTTP(self.blob)
        with self.assertRaisesRegex(ValueError, "frozen manifest hashes"):
            acq.acquire(self.manifest, self.archive, self.output, self.receipt, http,
                        expected_manifest_sha256=original_hash,
                        expected_archive_sha256=acq.sha(self.archive.read_bytes()))
        self.assertEqual(http.calls, [])
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads(self.receipt.read_text())["status"], "error")

    def test_bad_status_and_content_range_bodies_never_read(self):
        for status, bad_range in [(200, False), (206, True)]:
            with self.subTest(status=status, bad_range=bad_range):
                http = FakeHTTP(self.blob, status=status, bad_range=bad_range)
                with self.assertRaisesRegex(ValueError, "Content-Range"):
                    self.run_acquire(http)
                self.assertEqual(len(http.calls), 1)
                self.assertEqual(http.responses[0].read_calls, 0)
                self.assertEqual(json.loads(self.receipt.read_text())["network_body_bytes"], 0)
                self.receipt.unlink()
                (self.output / "images" / "0015").rmdir()
                (self.output / "images").rmdir()
                self.output.rmdir()

    def test_wrong_local_filename_stops_before_compressed_payload(self):
        first = sorted(self.a["selected_entries"], key=lambda x: x["filename"])[0]
        blob = bytearray(self.blob)
        blob[first["local_header_offset"] + 30] = ord("X")
        http = FakeHTTP(bytes(blob))
        with self.assertRaisesRegex(ValueError, "filename mismatch"):
            self.run_acquire(http)
        self.assertEqual(len(http.calls), 2)
        self.assertEqual(list(self.output.rglob("*.jpg")), [])

    def test_crc_mismatch_and_output_expansion_bound(self):
        first = sorted(self.a["selected_entries"], key=lambda x: x["filename"])[0]
        first["crc32_hex"] = "%08x" % (int(first["crc32_hex"], 16) ^ 1)
        blob = bytearray(self.blob)
        struct.pack_into("<I", blob, first["local_header_offset"] + 14,
                         int(first["crc32_hex"], 16))
        self.write_inputs()
        http = FakeHTTP(bytes(blob))
        with self.assertRaisesRegex(ValueError, "CRC32 mismatch"):
            self.run_acquire(http)
        self.assertEqual(len(http.calls), 3)
        self.assertEqual(list(self.output.rglob("*.jpg")), [])
        self.receipt.unlink()
        (self.output / "images" / "0015").rmdir()
        (self.output / "images").rmdir()
        self.output.rmdir()
        first["uncompressed_bytes"] = 32
        struct.pack_into("<I", blob, first["local_header_offset"] + 22, 32)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "expansion"):
            self.run_acquire(FakeHTTP(bytes(blob)))
        self.assertEqual(list(self.output.rglob("*.jpg")), [])

    def test_budget_preflight_timeout_and_no_retries(self):
        http = FakeHTTP(self.blob)
        with mock.patch.object(acq, "MAX_NETWORK_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "budget"):
                self.run_acquire(http)
        self.assertEqual(http.calls, [])
        self.receipt.unlink()
        timed = mock.Mock(side_effect=TimeoutError("artificial request timeout"))
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            self.run_acquire(timed)
        self.assertEqual(timed.call_count, 1)
        record = json.loads(self.receipt.read_text())
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"]["type"], "TimeoutError")
        self.assertEqual(record["images"], [])
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_incomplete_body_preserves_observed_bytes_without_retry(self):
        response = Response(b"", 0, 29)
        response.read = mock.Mock(
            side_effect=http.client.IncompleteRead(b"abc", 27))
        opener = mock.Mock(return_value=response)
        with self.assertRaisesRegex(ValueError, "incomplete HTTP"):
            self.run_acquire(opener)
        self.assertEqual(opener.call_count, 1)
        result = json.loads(self.receipt.read_text())
        self.assertEqual(result["network_body_bytes"], 3)
        self.assertEqual(result["requested_body_upper_bound"], 31)
        self.assertTrue(result["network_body_bytes_are_exact"])

    def test_interrupts_preserve_failed_receipt_and_restore_deadline(self):
        previous = signal.getsignal(signal.SIGALRM)
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception):
                response = Response(b"", 0, 29)
                response.read = mock.Mock(side_effect=exception("artificial interruption"))
                opener = mock.Mock(return_value=response)
                with self.assertRaises(exception):
                    self.run_acquire(opener)
                record = json.loads(self.receipt.read_text())
                self.assertEqual(record["status"], "error")
                self.assertFalse(record["network_body_bytes_are_exact"])
                self.assertEqual(record["error"]["type"], exception.__name__)
                self.assertEqual(opener.call_count, 1)
                self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
                self.receipt.unlink()
                shutil.rmtree(self.output)

    def test_short_response_and_invalid_jpeg_dimensions(self):
        http = FakeHTTP(self.blob, truncate=True)
        with self.assertRaisesRegex(ValueError, "incorrect length"):
            self.run_acquire(http)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(json.loads(self.receipt.read_text())["network_body_bytes"], 29)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            acq.jpeg_header(jpeg(width=16000, height=16000))


if __name__ == "__main__":
    unittest.main()
