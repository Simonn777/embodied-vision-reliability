"""Manufactured bytes and mocked HTTP only. No real archive/member access."""
import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transport0030", HERE / "acquire_masks.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PROJECT = HERE.parents[2]
REPO = PROJECT / "work/embodied-vision-reliability"


class Response(io.BytesIO):
    status = 206

    def __init__(self, body, headers, incomplete=False):
        super().__init__(body)
        self.headers = headers
        self.incomplete = incomplete

    def read(self, size=-1):
        if self.incomplete:
            self.incomplete = False
            raise http.client.IncompleteRead(super().read(2), 99)
        return super().read(size)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.name = "DAVIS/Annotations/480p/artificial/00001.png"
        self.data = b"manufactured opaque bytes; not an image, mask or scientific archive"
        compressor = zlib.compressobj(wbits=-15)
        packed = compressor.compress(self.data) + compressor.flush()
        crc = zlib.crc32(self.data) & 0xffffffff
        header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, 8, 0, 0,
                             crc, len(packed), len(self.data), len(self.name), 0)
        self.blob = header + self.name.encode() + packed
        self.row = dict(filename=self.name, sequence="artificial", frame_stem="00001",
            role="TRAIN", reused=False, local_header_offset=0, version_needed=20, flags=0,
            compression_method=8, mod_time=0, mod_date=0, crc32_hex=f"{crc:08x}",
            compressed_bytes=len(packed), uncompressed_bytes=len(self.data),
            filename_bytes=len(self.name))
        inventory = self.root / "inventory.json"
        inventory.write_text(json.dumps([self.row]))
        self.config = dict(root=str(self.root), body_cap=6000000, request_cap=1992,
            max_starts=2, sequences=["artificial"], url="https://invalid.local/fake.zip",
            etag='"fake"', archive_bytes=len(self.blob)+100, directory_offset=len(self.blob),
            inventory=str(inventory),
            range_helper=str(REPO / "experiments/0023-mapfree/range_inputs.py"),
            extractor=str(REPO / "experiments/0029-memory-information/acquire_davis.py"))
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def opener(self, request, timeout, *, fault=None):
        interval = request.get_header("Range").split("=")[1]
        start, end = map(int, interval.split("-"))
        self.calls.append([start, end])
        headers = {"ETag": '"changed"' if fault == "etag" else '"fake"',
            "Content-Range": f"bytes {start}-{end}/{len(self.blob)+100}",
            "Content-Length": str(end-start+1), "Content-Encoding": "identity"}
        return Response(self.blob[start:end+1], headers,
                        incomplete=(fault == "partial" and len(self.calls) == 3))

    def run_fake(self, name, fault=None):
        opener = lambda request, timeout: self.opener(request, timeout, fault=fault)
        with patch.object(module, "admit", return_value=(self.config, [self.row], {})):
            return module.run("fake.json", "config", "source", "artificial",
                              self.root / "attempts" / name, opener=opener)

    def test_three_ranges_opaque_output_and_append_only_events(self):
        record = self.run_fake("one")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["response_body_bytes"], len(self.blob))
        self.assertEqual(record["requested_body_upper_bound"], len(self.blob)+3)
        self.assertEqual(Path(record["members"][0]["path"]).read_bytes(), self.data)
        events = [json.loads(s) for s in (self.root/"attempts/one/events.jsonl").read_text().splitlines()]
        self.assertGreater(len(events), 3)
        self.assertEqual(events[-1]["status"], "completed")
        with self.assertRaisesRegex(ValueError, "no pending"):
            self.run_fake("duplicate")
        self.assertEqual(len(self.calls), 3)

    def test_changed_etag_stops_before_body_without_retry(self):
        with self.assertRaisesRegex(ValueError, "identity/header"):
            self.run_fake("one", "etag")
        record = module.read(self.root / "attempts/one/receipt.json")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(record["response_body_bytes"], 0)
        self.assertFalse(record["retryable"])
        with self.assertRaisesRegex(ValueError, "non-transient"):
            self.run_fake("two")
        self.assertEqual(len(self.calls), 1)

    def test_partial_bytes_preserved_then_explicit_fresh_continuation(self):
        with self.assertRaisesRegex(ValueError, "incomplete range"):
            self.run_fake("one", "partial")
        record = module.read(self.root / "attempts/one/receipt.json")
        self.assertEqual(record["response_body_bytes"], 30 + len(self.name) + 2)
        self.assertTrue(record["retryable"])
        self.assertEqual(len(self.calls), 3)
        result = self.run_fake("two")
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(result["previous_charged_body_bytes"], len(self.blob)+3)
        self.assertTrue((self.root / "attempts/one/receipt.json").exists())

    def test_lifetime_cap_prevents_http(self):
        self.config["body_cap"] = 1
        with self.assertRaisesRegex(ValueError, "reservation exhausted"):
            self.run_fake("one")
        self.assertEqual(self.calls, [])

    def test_opaque_hash_mutation_rejected(self):
        path = self.root / "local.bin"
        path.write_bytes(b"declared")
        sha = hashlib.sha256(b"declared").hexdigest()
        module.checked_file(path, sha, 8)
        path.write_bytes(b"mutated!")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            module.checked_file(path, sha, 8)


if __name__ == "__main__":
    unittest.main()
