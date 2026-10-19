"""Acquire only the 23 frozen development JPEGs; no decoding or estimation."""
import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath
import re
import signal
import struct
import time
import urllib.request
import zlib

SCHEMA = "evr0015.image-acquisition.v1"
URL = "https://vision.maths.lth.se/viktor/posebench/relative/megadepth1500-images.zip"
ARCHIVE_BYTES = 623957998
MAX_NETWORK_BYTES = 24 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_JPEG_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 32000000
REQUEST_TIMEOUT = 30
EXPECTED_MANIFEST_SHA256 = "94008d2da4a28ca482b394453c60bbcee40335b687989b222d554a9f7427f74c"
EXPECTED_ARCHIVE_MANIFEST_SHA256 = "a5bd7b9723d4c657d21d549684cf1cc56a5c8da69215c1283da8d6cef379f4d7"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("archive integer must be nonnegative")
    return value


def selected_members(manifest, archive):
    """Join names only; never open the coordinate/ground-truth HDF5."""
    keys = manifest.get("selected_pair_keys", [])
    if (manifest.get("schema") != "evr0012.development.v1"
            or manifest.get("development_scene") != "0015"
            or len(keys) != 16 or len(set(keys)) != 16):
        raise ValueError("expected the fixed 16-pair development manifest")
    names = set()
    for key in keys:
        match = re.fullmatch(r"0015-([^/\\]+\.jpg)_0015-([^/\\]+\.jpg)", key)
        if match is None:
            raise ValueError("pair key is not in the selected development scene")
        for name in match.groups():
            if PurePosixPath(name).name != name or name in (".", ".."):
                raise ValueError("unsafe image name")
            names.add("images/0015/" + name)
    if len(names) != 23:
        raise ValueError("expected exactly 23 unique selected images")
    if (archive.get("schema") != "evr0015.archive-metadata.v1"
            or archive.get("url") != URL
            or archive.get("archive_bytes") != ARCHIVE_BYTES):
        raise ValueError("archive source identity mismatch")
    entries = archive.get("selected_entries", [])
    if len(entries) != 23 or {x["filename"] for x in entries} != names:
        raise ValueError("archive selected entries must equal the manifest allowlist")
    for item in entries:
        if item["flags"] != 0 or item["compression_method"] != 8:
            raise ValueError("only expected unencrypted raw-deflate entries are allowed")
        offset = integer(item["local_header_offset"])
        packed = integer(item["compressed_bytes"])
        unpacked = integer(item["uncompressed_bytes"])
        if (not 0 < unpacked <= MAX_JPEG_BYTES or packed == 0
                or offset + 30 + packed >= ARCHIVE_BYTES):
            raise ValueError("entry sizes or offset exceed the allowed bounds")
        if re.fullmatch(r"[0-9a-f]{8}", item["crc32_hex"]) is None:
            raise ValueError("invalid CRC32 metadata")
    if sum(x["compressed_bytes"] for x in entries) >= MAX_NETWORK_BYTES:
        raise ValueError("compressed payload exceeds network budget")
    if sum(x["uncompressed_bytes"] for x in entries) > MAX_OUTPUT_BYTES:
        raise ValueError("uncompressed JPEG total exceeds output budget")
    return sorted(entries, key=lambda x: x["filename"])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused; source must remain the frozen URL")


@contextmanager
def request_deadline():
    """Enforce a whole-request wall deadline in this single-thread CLI."""
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("existing process alarm conflicts with request deadline")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("request exceeded 30-second wall deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, REQUEST_TIMEOUT)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class RangeReader:
    def __init__(self, receipt, opener=None):
        self.receipt = receipt
        self.opener = opener or urllib.request.build_opener(NoRedirect()).open

    def read(self, start, end):
        length = end - start + 1
        if not 0 <= start <= end < ARCHIVE_BYTES:
            raise ValueError("range outside frozen archive")
        if self.receipt["network_body_bytes"] + length + 1 > MAX_NETWORK_BYTES:
            raise ValueError("network body budget exceeded before request")
        item = {"start": start, "end": end, "expected_body_bytes": length,
                "body_bytes": 0, "status": None}
        self.receipt["requests"].append(item)
        self.receipt["requested_body_upper_bound"] += length + 1
        request = urllib.request.Request(
            URL, headers={"Range": "bytes=%d-%d" % (start, end),
                          "Accept-Encoding": "identity"})
        started = time.monotonic()
        try:
            with request_deadline(), self.opener(request, timeout=REQUEST_TIMEOUT) as response:
                item["status"] = response.status
                item["content_range"] = response.headers.get("Content-Range")
                wanted = "bytes %d-%d/%d" % (start, end, ARCHIVE_BYTES)
                if response.status != 206 or item["content_range"] != wanted:
                    raise ValueError("HTTP status or Content-Range mismatch; body not read")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("encoded range response refused")
                if response.headers.get("Content-Length") != str(length):
                    raise ValueError("Content-Length mismatch; body not read")
                try:
                    body = response.read(length + 1)
                except http.client.IncompleteRead as exc:
                    item["body_bytes"] = len(exc.partial)
                    self.receipt["network_body_bytes"] += len(exc.partial)
                    raise ValueError("incomplete HTTP body") from exc
                except BaseException:
                    self.receipt["network_body_bytes_are_exact"] = False
                    raise
                item["body_bytes"] = len(body)
                self.receipt["network_body_bytes"] += len(body)
                if len(body) != length:
                    raise ValueError("range body has incorrect length")
                return body
        finally:
            item["elapsed_seconds"] = time.monotonic() - started


def jpeg_header(data):
    """Read SOF dimensions only, with no pixel decompression or EXIF transform."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("payload is not a JPEG")
    p = 2
    sof = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9,
           0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while p < len(data):
        if data[p] != 255:
            raise ValueError("malformed JPEG marker")
        while p < len(data) and data[p] == 255:
            p += 1
        if p == len(data):
            break
        marker = data[p]
        p += 1
        if marker in (0xd9, 0xda):
            break
        if marker == 0x01 or 0xd0 <= marker <= 0xd8:
            continue
        if p + 2 > len(data):
            break
        length = int.from_bytes(data[p:p + 2], "big")
        if length < 2 or p + length > len(data):
            raise ValueError("JPEG segment length invalid")
        if marker in sof:
            if length < 8:
                raise ValueError("JPEG SOF too short")
            precision, height, width, channels = struct.unpack(">BHHB", data[p + 2:p + 8])
            if (precision != 8 or not 0 < width <= 16384 or not 0 < height <= 16384
                    or width * height > MAX_IMAGE_PIXELS or channels not in (1, 3, 4)
                    or length != 8 + 3 * channels):
                raise ValueError("JPEG dimensions, precision or channels outside bounds")
            return {"width": width, "height": height, "channels": channels,
                    "precision_bits": precision, "pixel_decoding_performed": False}
        p += length
    raise ValueError("JPEG has no supported dimension header")


def extract_member(reader, entry):
    offset = entry["local_header_offset"]
    header = reader.read(offset, offset + 29)
    fields = struct.unpack("<4s5H3I2H", header)
    signature, version, flags, method, _, _, crc, packed, unpacked, nlen, xlen = fields
    if (signature != b"PK\x03\x04" or flags != 0 or flags != entry["flags"]
            or method != 8 or packed != entry["compressed_bytes"]
            or unpacked != entry["uncompressed_bytes"]
            or crc != int(entry["crc32_hex"], 16) or version > 20
            or nlen != len(entry["filename"].encode("utf-8"))):
        raise ValueError("local ZIP header disagrees with frozen metadata")
    names = reader.read(offset + 30, offset + 30 + nlen + xlen - 1)
    if names[:nlen] != entry["filename"].encode("utf-8"):
        raise ValueError("local ZIP filename mismatch; payload not read")
    start = offset + 30 + nlen + xlen
    compressed = reader.read(start, start + packed - 1)
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(compressed, unpacked + 1)
    if (len(data) != unpacked or not decoder.eof or decoder.unconsumed_tail
            or decoder.unused_data):
        raise ValueError("deflate expansion length or stream termination mismatch")
    if zlib.crc32(data) & 0xffffffff != crc:
        raise ValueError("uncompressed JPEG CRC32 mismatch")
    return data, jpeg_header(data)


def acquire(manifest_path, archive_path, output_dir, receipt_path, opener=None, *,
            expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
            expected_archive_sha256=EXPECTED_ARCHIVE_MANIFEST_SHA256):
    manifest_path, archive_path = Path(manifest_path), Path(archive_path)
    output_dir, receipt_path = Path(output_dir), Path(receipt_path)
    if output_dir.exists() or receipt_path.exists():
        raise FileExistsError("fresh output directory and receipt required; no resume")
    receipt = {"schema": SCHEMA, "status": "started", "stage": "inputs",
               "url": URL, "archive_bytes": ARCHIVE_BYTES,
               "source_sha256": sha(Path(__file__).read_bytes()),
               "network_body_limit": MAX_NETWORK_BYTES, "network_body_bytes": 0,
               "network_body_bytes_are_exact": True, "requested_body_upper_bound": 0,
               "output_bytes_limit": MAX_OUTPUT_BYTES, "output_bytes": 0,
               "request_timeout_seconds": REQUEST_TIMEOUT, "hidden_retries": False,
               "request_wall_deadline_seconds": REQUEST_TIMEOUT,
               "expected_manifest_sha256": expected_manifest_sha256,
               "expected_archive_manifest_sha256": expected_archive_sha256,
               "requests": [], "images": []}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)

    def save():
        temporary = receipt_path.with_name(receipt_path.name + ".tmp")
        with temporary.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(receipt_path)

    try:
        mb, ab = manifest_path.read_bytes(), archive_path.read_bytes()
        receipt["manifest_sha256"], receipt["archive_manifest_sha256"] = sha(mb), sha(ab)
        if sha(mb) != expected_manifest_sha256 or sha(ab) != expected_archive_sha256:
            raise ValueError("input bytes do not match the frozen manifest hashes")
        entries = selected_members(json.loads(mb), json.loads(ab))
        receipt["selected_images"] = len(entries)
        output_dir.mkdir()
        target_root = output_dir / "images" / "0015"
        target_root.mkdir(parents=True)
        reader = RangeReader(receipt, opener)
        for entry in entries:
            receipt["stage"] = "acquire:" + entry["filename"]
            save()
            data, dimensions = extract_member(reader, entry)
            target = output_dir / entry["filename"]
            with target.open("xb") as stream:
                stream.write(data)
            receipt["output_bytes"] += len(data)
            receipt["images"].append({**entry, **dimensions, "sha256": sha(data)})
            save()
        receipt["status"], receipt["stage"] = "completed", "completed"
        save()
        return receipt
    except BaseException as exc:
        receipt["status"] = "error"
        if not isinstance(exc, Exception):
            receipt["network_body_bytes_are_exact"] = False
        message = str(exc)
        for path, label in [(output_dir, "<private-output-dir>"),
                            (receipt_path, "<receipt>"), (manifest_path, "<manifest>"),
                            (archive_path, "<archive-manifest>")]:
            message = message.replace(str(path), label)
        receipt["error"] = {"type": type(exc).__name__, "message": message}
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    try:
        result = acquire(args.manifest, args.archive_manifest, args.output_dir, args.receipt)
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "error", "type": type(exc).__name__}))
        return 1
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "images": len(result["images"]),
                      "network_body_bytes": result["network_body_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
