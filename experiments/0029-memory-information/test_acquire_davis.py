"""Synthetic transport/source tests; no real HTTP, media, model, or GT decode."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


a = load("evr0029_acquisition_under_test", HERE / "acquire_davis.py")
HELPER_PATH = HERE.parent / "embodied-vision-reliability" / a.HELPER
if not HELPER_PATH.is_file():  # Same test after parent copies it into the repository.
    HELPER_PATH = HERE.parents[1] / a.HELPER
h = load("evr0029_transport_under_test", HELPER_PATH)


class Response:
    def __init__(self, transport, start, end):
        self.transport = transport
        self.status = transport.status
        self.headers = {
            "ETag": a.ETAG, "Content-Range": "bytes %d-%d/%d" % (start, end, a.ARCHIVE_BYTES),
            "Content-Length": str(end-start+1), "Content-Encoding": "identity"}
        self.headers.update(transport.header_overrides)
        self.data = io.BytesIO(transport.archive[start:end+1])
        self.fail_read = transport.interrupt_request == len(transport.requests)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, count):
        self.transport.read_calls += 1
        if self.fail_read:
            raise KeyboardInterrupt("manufactured interrupted read")
        return self.data.read(count)


class Transport:
    def __init__(self, archive):
        self.archive = archive
        self.requests = []
        self.read_calls = 0
        self.status = 206
        self.header_overrides = {}
        self.interrupt_request = None

    def __call__(self, request, timeout):
        assert request.full_url == a.URL
        assert request.get_header("If-match") == a.ETAG
        assert request.get_header("Accept-encoding") == "identity"
        assert 0 < timeout <= 30
        start, end = map(int, request.get_header("Range")[6:].split("-"))
        self.requests.append((start, end))
        return Response(self, start, end)


def manufactured(*, bad_crc=False, huge_extra=False):
    archive = bytearray()
    entries = {}
    expected = {}
    for index in range(6):
        name = "DAVIS/Annotations/480p/artificial-%d/00000.png" % index
        data = bytes([65+index]) * 64  # Inert bytes, not any image format.
        stream = zlib.compressobj(wbits=-15)
        packed = stream.compress(data) + stream.flush()
        if index == 0 and bad_crc:
            stream = zlib.compressobj(wbits=-15)
            other = stream.compress(b"z"*64) + stream.flush()
            assert len(other) == len(packed)
            packed = other
        extra = b"x"*4097 if index == 0 and huge_extra else b""
        offset = len(archive)
        crc = zlib.crc32(data) & 0xffffffff
        header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, 8, 0, 0,
                             crc, len(packed), len(data), len(name), len(extra))
        archive.extend(header + name.encode() + extra + packed)
        entries[name] = {
            "filename": name, "version_needed": 20, "flags": 0,
            "compression_method": 8, "mod_time": 0, "mod_date": 0,
            "crc32_hex": "%08x" % crc, "compressed_bytes": len(packed),
            "uncompressed_bytes": len(data), "local_header_offset": offset,
            "filename_bytes": len(name), "stage": "initial_supervision",
            "output_relative": "initial_supervision/artificial-%d/00000.png" % index}
        expected[name] = data
    directory_offset = len(archive)
    archive.extend(b"synthetic directory marker")
    idx = h.Index(a.URL, a.ETAG, a.ARCHIVE_BYTES, directory_offset,
                  entries, {"manifest": "1"*64, "central_directory": "2"*64})
    plan = {"archive": {"url": a.URL, "archive_bytes": a.ARCHIVE_BYTES, "etag": a.ETAG}}
    return plan, idx, Transport(bytes(archive)), expected


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.output = self.base / "output"

    def receipt(self):
        return json.loads((self.output / "receipt.json").read_text())

    def run_fake(self, data=None, **kwargs):
        plan, index, transport, expected = data or manufactured()
        result = a.acquire_phase(h, plan, index, "initial_supervision",
                                 self.output, opener=transport, **kwargs)
        return result, transport, expected

    def test_exact_ranges_crc_and_separate_phase_files(self):
        result, transport, expected = self.run_fake()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["media_decoded"])
        self.assertEqual(len(result["members"]), 6)
        self.assertEqual(len(transport.requests), 18)
        for row in result["members"]:
            self.assertEqual((self.output / row["saved_relative"]).read_bytes(),
                             expected[row["filename"]])
            self.assertEqual(row["sha256"], hashlib.sha256(expected[row["filename"]]).hexdigest())
            self.assertTrue(row["saved_relative"].startswith("members/initial_supervision/"))
        self.assertEqual(result["response_body_bytes"],
                         sum(end-start+1 for start, end in transport.requests))
        self.assertLessEqual(result["all_phases_reserved_bytes"], 60000000)

    def test_ignored_range_or_wrong_identity_reads_no_body(self):
        for number, patch in enumerate(({"status": 200},
                {"headers": {"ETag": '"other"'}},
                {"headers": {"Content-Range": "bytes 0-29/9"}})):
            with self.subTest(patch=patch):
                self.output = self.base / ("bad-%d" % number)
                plan, index, transport, _ = manufactured()
                transport.status = patch.get("status", 206)
                transport.header_overrides = patch.get("headers", {})
                with self.assertRaisesRegex(ValueError, "HTTP identity"):
                    a.acquire_phase(h, plan, index, "initial_supervision",
                                    self.output, opener=transport)
                self.assertEqual(transport.read_calls, 0)
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(self.receipt()["status"], "failed")

    def test_oversized_local_extra_rejected_before_reading_names_or_payload(self):
        plan, index, transport, _ = manufactured(huge_extra=True)
        with self.assertRaisesRegex(ValueError, "extra field"):
            a.acquire_phase(h, plan, index, "initial_supervision", self.output,
                            opener=transport)
        self.assertEqual(transport.requests, [(0, 29)])
        self.assertEqual(self.receipt()["response_body_bytes"], 30)
        self.assertEqual(self.receipt()["members"], [])

    def test_bad_crc_retains_failure_without_output_member(self):
        plan, index, transport, _ = manufactured(bad_crc=True)
        with self.assertRaisesRegex(ValueError, "CRC32"):
            a.acquire_phase(h, plan, index, "initial_supervision", self.output,
                            opener=transport)
        self.assertEqual(len(transport.requests), 3)
        self.assertEqual(self.receipt()["members"], [])
        self.assertEqual(self.receipt()["status"], "failed")

    def test_interrupted_second_member_preserves_first_and_uncertain_read_count(self):
        plan, index, transport, _ = manufactured()
        transport.interrupt_request = 5
        with self.assertRaises(KeyboardInterrupt):
            a.acquire_phase(h, plan, index, "initial_supervision", self.output,
                            opener=transport)
        saved = self.receipt()
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error_type"], "KeyboardInterrupt")
        self.assertFalse(saved["response_body_bytes_exact"])
        self.assertEqual(len(saved["members"]), 1)
        self.assertTrue((self.output / saved["members"][0]["saved_relative"]).is_file())

    def test_gt_phase_closed_even_with_self_reported_completed_proof(self):
        plan, index, transport, _ = manufactured()
        proof = self.base / "untrusted-proof.json"
        proof.write_text('{"status":"completed","eligible":true,"videos":6}')
        with self.assertRaisesRegex(ValueError, "evaluator_gt is closed"):
            a.acquire_phase(h, plan, index, "evaluator_gt", self.output,
                            opener=transport, proof_path=proof)
        self.assertEqual(transport.requests, [])
        self.assertEqual(self.receipt()["status"], "failed")

    def test_existing_output_and_dangling_symlink_refused(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("preserve")
        plan, index, transport, _ = manufactured()
        with self.assertRaisesRegex(ValueError, "fresh"):
            a.acquire_phase(h, plan, index, "initial_supervision", self.output,
                            opener=transport)
        self.assertEqual(marker.read_text(), "preserve")
        link = self.base / "dangling"
        link.symlink_to(self.base / "absent-target")
        with self.assertRaisesRegex(ValueError, "fresh"):
            a.acquire_phase(h, plan, index, "initial_supervision", link,
                            opener=transport)
        self.assertEqual(transport.requests, [])
        self.assertFalse((self.base / "absent-target").exists())

    def test_postcheck_failure_cannot_publish_completed_receipt(self):
        def changed():
            raise ValueError("manufactured source changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_fake(source_check=changed)
        self.assertEqual(self.receipt()["status"], "failed")
        self.assertEqual(len(self.receipt()["members"]), 6)

    def test_budget_preflight_prevents_first_http(self):
        plan, index, transport, _ = manufactured()
        with mock.patch.dict(a.PHASE_CAPS, initial_supervision=30):
            with self.assertRaisesRegex(ValueError, "cap exceeded before HTTP"):
                a.acquire_phase(h, plan, index, "initial_supervision", self.output,
                                opener=transport)
        self.assertEqual(transport.requests, [])

    def test_changed_frozen_source_rejected_without_transport(self):
        root = self.base / "repository"
        snapshots = {}
        for name in (a.SELF, a.HELPER, a.MANIFEST, a.PROTOCOL):
            data = (HERE / "acquire_davis.py").read_bytes() if name == a.SELF else name.encode()
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            snapshots[name] = data
        def git(argv, cwd, timeout):
            self.assertEqual(argv[:2], ["git", "show"])
            self.assertEqual(Path(cwd), root)
            self.assertEqual(timeout, 15)
            return snapshots[argv[2].split(":", 1)[1]]
        with mock.patch.object(a.subprocess, "check_output", side_effect=git):
            bound = a.source_bindings(root, "a"*40)
            self.assertEqual(len(bound), 4)
            (root / a.PROTOCOL).write_text("changed")
            with self.assertRaisesRegex(ValueError, "differs from freeze"):
                a.source_bindings(root, "a"*40)


if __name__ == "__main__":
    unittest.main()
