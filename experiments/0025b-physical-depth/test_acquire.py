"""Artificial byte/HTTP tests only; real urllib network entry points denied."""
import contextlib
import email.message
import gzip
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import types
import unittest
from unittest import mock


SOURCE = Path(__file__).with_name("acquire.py")
spec = importlib.util.spec_from_file_location("icl_artificial_acquire", SOURCE)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def raw_tar(members=None, format=tarfile.USTAR_FORMAT):
    members = members or [("./", None), ("./rgb/", None),
                          ("./rgb/0.png", b"deliberately not a PNG"),
                          ("./groundtruth.txt", b"\xff\x00not numerical pose data")]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=format) as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            if payload is None:
                info.type = tarfile.DIRTYPE
            elif isinstance(payload, tuple):
                info.type, info.linkname = payload
            else:
                info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload) if isinstance(payload, bytes) else None)
    return buffer.getvalue()


def packed(members=None, format=tarfile.USTAR_FORMAT):
    return gzip.compress(raw_tar(members, format), mtime=0)


class FakeResponse:
    def __init__(self, asset, body, *, status=200, url=None, length=None,
                 etag=None, duplicate=None, incomplete=False):
        self.status = status
        self.url = url or asset["url"]
        self.body = io.BytesIO(body)
        self.incomplete = incomplete
        self.headers = email.message.Message()
        self.headers["Content-Length"] = str(asset["bytes"] if length is None else length)
        self.headers["ETag"] = asset["etag"] if etag is None else etag
        self.headers["Content-Encoding"] = "x-gzip"
        if duplicate:
            self.headers[duplicate] = self.headers[duplicate]

    def geturl(self):
        return self.url

    def read(self, size):
        if self.incomplete:
            self.incomplete = False
            raise http.client.IncompleteRead(self.body.read(min(size, 17)), 20)
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.body.close()


class FakeOpener:
    def __init__(self, responses, callback=None):
        self.responses = list(responses)
        self.calls = []
        self.callback = callback

    def open(self, request, timeout):
        self.calls.append((request.full_url, dict(request.header_items()), timeout))
        if self.callback:
            self.callback(len(self.calls))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="artificial-icl-")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        for target in ("urllib.request.OpenerDirector.open", "urllib.request.urlopen"):
            patcher = mock.patch(target, side_effect=AssertionError("REAL NETWORK FORBIDDEN"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def index(self, body, label="input"):
        path = self.root / (label + ".tar.gz")
        output = self.root / (label + ".index.json")
        path.write_bytes(body)
        with mock.patch.object(tarfile.TarFile, "extractfile", side_effect=AssertionError("extract forbidden")), \
             mock.patch.object(tarfile.TarFile, "extractall", side_effect=AssertionError("extract forbidden")):
            result = a.index_archive(path, output, time.monotonic())
        return result, json.loads(output.read_text())

    def catalog(self):
        bodies = [packed(), packed([("./", None), ("depth/1.png", b"opaque artificial depth")])]
        assets = tuple({**asset, "bytes": len(body)} for asset, body in zip(a.ASSETS, bodies))
        return assets, bodies

    def args(self, name="destination"):
        return types.SimpleNamespace(destination=str(self.root / name), expected_commit="a" * 40,
                                     repo_root=str(self.root), source_relative="source.py",
                                     protocol_relative="protocol.md")

    @contextlib.contextmanager
    def setup_run(self, assets, opener, *, cap=None, verify=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(a, "ASSETS", assets))
            stack.enter_context(mock.patch.object(a, "BODY_CAP", sum(x["bytes"] for x in assets) if cap is None else cap))
            stack.enter_context(mock.patch.dict(os.environ, {name: "4" for name in a.THREAD_ENV}))
            stack.enter_context(mock.patch.object(a.urllib.request, "build_opener", return_value=opener))
            if verify is None:
                stack.enter_context(mock.patch.object(a, "verify_frozen", return_value={"source.py": "fake-fixture-only"}))
            yield

    def test_ustar_and_old_gnu_short_names_skip_opaque_pose(self):
        for index, fmt in enumerate((tarfile.USTAR_FORMAT, tarfile.GNU_FORMAT)):
            result, record = self.index(packed(format=fmt), "format" + str(index))
            self.assertEqual([x["name"] for x in record["members"]], [".", "rgb", "rgb/0.png", "groundtruth.txt"])
            self.assertFalse(record["member_values_decoded"])
            self.assertEqual(record["members"][-1]["size"], len(b"\xff\x00not numerical pose data"))
            self.assertEqual(result["archive_sha256_tofu"], record["archive_sha256_tofu"])
            self.assertEqual(sorted(x.suffix for x in self.root.iterdir()).count(".png"), 0)

    def test_corrupt_crc_truncated_and_missing_end_are_rejected(self):
        body = bytearray(packed())
        body[-8] ^= 1
        fixtures = [bytes(body), packed()[:-3], gzip.compress(raw_tar()[:3584], mtime=0),
                    gzip.compress(raw_tar() + b"hidden nonzero member", mtime=0)]
        for i, value in enumerate(fixtures):
            with self.subTest(i=i), self.assertRaises((ValueError, EOFError, gzip.BadGzipFile, tarfile.TarError)):
                self.index(value, "bad" + str(i))
            self.assertFalse((self.root / ("bad" + str(i) + ".index.json")).exists())

    def test_unsafe_links_duplicates_and_extensions_fail_closed(self):
        cases = [([("../escape", b"x")], tarfile.USTAR_FORMAT),
                 ([("/absolute", b"x")], tarfile.USTAR_FORMAT),
                 ([("C:\\escape", b"x")], tarfile.USTAR_FORMAT),
                 ([("link", (tarfile.SYMTYPE, "target"))], tarfile.USTAR_FORMAT),
                 ([("x", b"a"), ("./x", b"b")], tarfile.USTAR_FORMAT),
                 ([("x" * 140, b"x")], tarfile.PAX_FORMAT),
                 ([("x" * 140, b"x")], tarfile.GNU_FORMAT)]
        for i, (members, fmt) in enumerate(cases):
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.index(packed(members, fmt), "unsafe" + str(i))

    def test_structural_and_resource_caps(self):
        for i, (name, value) in enumerate((("MAX_MEMBERS", 1), ("MAX_MEMBER_BYTES", 2),
                                          ("MAX_EXPANDED_BYTES", 512))):
            with mock.patch.object(a, name, value), self.assertRaises(ValueError):
                self.index(packed(), "cap" + str(i))
        with mock.patch.object(a.time, "monotonic", return_value=1800):
            with self.assertRaises(TimeoutError):
                a.limits(0)
        unit = 1 if a.sys.platform == "darwin" else 1024
        usage = types.SimpleNamespace(ru_maxrss=a.RSS_BYTES // unit + 1)
        with mock.patch.object(a.resource, "getrusage", return_value=usage):
            with self.assertRaises(MemoryError):
                a.limits(time.monotonic())

    def test_success_exact_two_requests_persistent_hashes_and_no_overwrite(self):
        assets, bodies = self.catalog()
        opener = FakeOpener([FakeResponse(x, b) for x, b in zip(assets, bodies)])
        args = self.args()
        with self.setup_run(assets, opener):
            result = a.run(args)
            with self.assertRaises(FileExistsError):
                a.run(args)
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(result["body_bytes"], sum(map(len, bodies)))
        saved = json.loads((Path(args.destination) / "receipt.json").read_text())
        self.assertEqual(saved["status"], "completed")
        for row, asset, body, call in zip(saved["assets"], assets, bodies, opener.calls):
            self.assertEqual(call[0], asset["url"])
            self.assertEqual({k.lower(): v for k, v in call[1].items()}["if-match"], asset["etag"])
            self.assertEqual(call[2], 30)
            self.assertEqual(row["sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual(row["index"]["archive_sha256_tofu"], row["sha256"])
            self.assertEqual((Path(args.destination) / asset["name"]).read_bytes(), body)

    def test_changed_headers_redirect_and_transport_failure_do_not_retry(self):
        assets, bodies = self.catalog()
        bad = [FakeResponse(assets[0], bodies[0], etag='"changed"'),
               FakeResponse(assets[0], bodies[0], length=len(bodies[0]) + 1),
               FakeResponse(assets[0], bodies[0], duplicate="ETag"),
               FakeResponse(assets[0], bodies[0], status=302),
               FakeResponse(assets[0], bodies[0], url="https://example.invalid/mirror"),
               TimeoutError("fake request timeout")]
        for i, response in enumerate(bad):
            args = self.args("bad-response" + str(i))
            opener = FakeOpener([response])
            with self.setup_run(assets, opener), self.assertRaises((ValueError, TimeoutError)):
                a.run(args)
            receipt = json.loads((Path(args.destination) / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "error")
            self.assertEqual(receipt["body_bytes"], 0)
            self.assertEqual(len(opener.calls), 1)
        with self.assertRaises(ValueError):
            a.NoRedirect().redirect_request(None, None, 302, "fake", {}, "https://example.invalid")

    def test_short_and_exception_partial_bodies_are_counted(self):
        assets, bodies = self.catalog()
        for i, incomplete in enumerate((False, True)):
            short = bodies[0][:17]
            opener = FakeOpener([FakeResponse(assets[0], short, incomplete=incomplete)])
            args = self.args("partial" + str(i))
            with self.setup_run(assets, opener), self.assertRaises((EOFError, http.client.IncompleteRead)):
                a.run(args)
            receipt = json.loads((Path(args.destination) / "receipt.json").read_text())
            self.assertEqual(receipt["body_bytes"], 17)
            self.assertEqual(receipt["assets"][0]["received_sha256"], hashlib.sha256(short).hexdigest())
            self.assertEqual((Path(args.destination) / (assets[0]["name"] + ".part")).read_bytes(), short)
            self.assertEqual(len(opener.calls), 1)

    def test_total_cap_includes_completed_and_failed_second_body(self):
        assets, bodies = self.catalog()
        opener = FakeOpener([FakeResponse(x, b) for x, b in zip(assets, bodies)])
        cap = len(bodies[0]) + 19
        args = self.args()
        with self.setup_run(assets, opener, cap=cap), self.assertRaises(ValueError):
            a.run(args)
        receipt = json.loads((Path(args.destination) / "receipt.json").read_text())
        self.assertEqual(receipt["body_bytes"], cap)
        self.assertEqual([r["status"] for r in receipt["assets"]], ["completed", "error"])
        self.assertEqual(receipt["assets"][1]["saved_partial_bytes"], 19)
        self.assertEqual(len(opener.calls), 2)

    def test_post_source_mutation_and_late_write_timeout_cannot_complete(self):
        assets, bodies = self.catalog()
        opener = FakeOpener([FakeResponse(x, b) for x, b in zip(assets, bodies)])
        args = self.args("mutated")
        with self.setup_run(assets, opener, verify=True), \
             mock.patch.object(a, "verify_frozen", side_effect=[{"source": "before"}, {"source": "after"}]), \
             self.assertRaises(ValueError):
            a.run(args)
        self.assertEqual(json.loads((Path(args.destination) / "receipt.json").read_text())["status"], "error")
        clock = [0.0]
        original_save = a.save

        def slow_final_save(path, value):
            original_save(path, value)
            if value.get("schema") == "evr0025b.acquisition.v1" and value.get("status") == "completed":
                clock[0] = 1801.0

        opener = FakeOpener([FakeResponse(x, b) for x, b in zip(assets, bodies)])
        args = self.args("late")
        with self.setup_run(assets, opener), mock.patch.object(a.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(a, "save", side_effect=slow_final_save), self.assertRaises(TimeoutError):
            a.run(args)
        self.assertEqual(json.loads((Path(args.destination) / "receipt.json").read_text())["status"], "error")

    def test_real_local_git_byte_guard_and_preflight_prevents_requests(self):
        repo = self.root / "fake-git"
        repo.mkdir()
        source = repo / "source.py"
        source.write_bytes(SOURCE.read_bytes())
        protocol = repo / "protocol.md"
        protocol.write_text("artificial protocol only\n")

        def git(*args):
            return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout

        git("init")
        git("add", "source.py", "protocol.md")
        git("-c", "user.name=Artificial Test", "-c", "user.email=test@example.invalid",
            "commit", "-m", "Artificial frozen bytes")
        args = self.args()
        args.repo_root = str(repo)
        args.expected_commit = git("rev-parse", "HEAD").decode().strip()
        with mock.patch.object(a, "__file__", str(source)):
            self.assertEqual(len(a.verify_frozen(args)), 2)
            original = source.read_bytes()
            source.write_bytes(original + b"\n# changed\n")
            with self.assertRaises(ValueError):
                a.verify_frozen(args)
            source.write_bytes(original)
            protocol.write_text("mutated protocol\n")
            assets, _ = self.catalog()
            opener = FakeOpener([])
            with self.setup_run(assets, opener, verify=True), self.assertRaises(ValueError):
                a.run(args)
            self.assertEqual(opener.calls, [])
        receipt = json.loads((Path(args.destination) / "receipt.json").read_text())
        self.assertEqual(receipt["body_bytes"], 0)
        self.assertEqual(receipt["status"], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
