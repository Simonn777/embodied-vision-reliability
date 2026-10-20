"""Range-only Map-free image/K acquisition primitives; no allocation or CLI.

This is a source preparation helper. A future frozen caller must supply an
explicit member list, all three cached-index hashes, and resource limits.
Pose/ground-truth members are deliberately outside this frontend API.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import struct
import time
import urllib.error
import urllib.request
import zlib

URL = "https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/dataset/val.zip"
SCHEMA = "evr0023.frontend-range-acquisition.v1"
CORE_FIELDS = ("filename", "flags", "compression_method", "crc32_hex",
               "compressed_bytes", "uncompressed_bytes", "local_header_offset")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def extra_fields(data):
    out, pos = {}, 0
    while pos < len(data):
        if pos + 4 > len(data):
            raise ValueError("truncated ZIP extra field")
        tag, size = struct.unpack_from("<HH", data, pos)
        pos += 4
        if tag in out or pos + size > len(data):
            raise ValueError("duplicate or truncated ZIP extra field")
        out[tag] = data[pos:pos + size]
        pos += size
    return out


def safe_name(name):
    if (not isinstance(name, str) or not name.isascii() or "\\" in name
            or "\x00" in name or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe ZIP member name")
    return name


def frontend_kind(name):
    safe_name(name)
    if re.fullmatch(r"val/s[0-9]{5}/seq[01]/frame_[0-9]{5}\.jpg", name):
        return "image"
    if re.fullmatch(r"val/s[0-9]{5}/intrinsics\.txt", name):
        return "intrinsics"
    raise ValueError("frontend allowlist excludes poses/GT and other member types")


def parse_directory(data, archive_bytes, directory_offset):
    """Parse cached central-directory bytes, including ZIP64 *offsets*.

    The observed source uses small deflated members and ZIP64 archive offsets.
    Other methods, data descriptors, ZIP64 member sizes, encryption, symlinks,
    multiple disks and non-ASCII names fail closed instead of guessing.
    """
    entries, pos = {}, 0
    while pos < len(data):
        if pos + 46 > len(data):
            raise ValueError("truncated central-directory header")
        f = struct.unpack_from("<4s6H3I5H2I", data, pos)
        (sig, made, version, flags, method, mtime, mdate, crc, packed, unpacked,
         nlen, xlen, clen, disk, _, external, offset) = f
        end = pos + 46 + nlen + xlen + clen
        if sig != b"PK\x01\x02" or end > len(data) or not nlen:
            raise ValueError("invalid central-directory record")
        if (version != 20 or flags != 0 or method != 8 or disk != 0
                or packed in (0, 0xffffffff) or unpacked in (0, 0xffffffff)):
            raise ValueError("unsupported ZIP member encoding")
        try:
            name = safe_name(data[pos + 46:pos + 46 + nlen].decode("ascii"))
        except UnicodeDecodeError as exc:
            raise ValueError("non-ASCII ZIP member name") from exc
        if name in entries:
            raise ValueError("duplicate ZIP member name")
        if made >> 8 == 3 and stat.S_IFMT(external >> 16) not in (0, stat.S_IFREG):
            raise ValueError("non-regular ZIP member")
        extra = extra_fields(data[pos + 46 + nlen:pos + 46 + nlen + xlen])
        if offset == 0xffffffff:
            if len(extra.get(1, b"")) != 8:
                raise ValueError("ZIP64 offset field missing or ambiguous")
            offset = struct.unpack("<Q", extra[1])[0]
        elif 1 in extra:
            raise ValueError("unneeded ZIP64 field")
        if not 0 <= offset < directory_offset < archive_bytes:
            raise ValueError("local ZIP offset outside archive data")
        if offset + 30 + nlen + packed > directory_offset:
            raise ValueError("member minimum extent overlaps central directory")
        entries[name] = dict(filename=name, flags=flags, compression_method=method,
                             crc32_hex="%08x" % crc, compressed_bytes=packed,
                             uncompressed_bytes=unpacked, local_header_offset=offset,
                             version_needed=version, mod_time=mtime, mod_date=mdate)
        pos = end
    return entries


@dataclass(frozen=True)
class Index:
    url: str
    etag: str
    archive_bytes: int
    directory_offset: int
    entries: dict
    fingerprints: dict


def load_index(members_path, directory_path, receipt_path, expected_sha256):
    """Read metadata only. Expected hashes must come from a future frozen caller."""
    paths = dict(members=Path(members_path), directory=Path(directory_path),
                 receipt=Path(receipt_path))
    if set(expected_sha256) != set(paths):
        raise ValueError("all three index fingerprints are required")
    blobs = {}
    for kind, path in paths.items():
        # Technical cap, above the known 3.95 MB directory/roughly 12 MB inventory.
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024**2:
            raise ValueError("index must be a bounded regular local file")
        blobs[kind] = path.read_bytes()
        if sha(blobs[kind]) != expected_sha256[kind]:
            raise ValueError("cached index fingerprint mismatch: " + kind)
    members, receipt = json.loads(blobs["members"]), json.loads(blobs["receipt"])
    head, central = receipt["head"], receipt["central_directory"]
    size = integer(members["archive_bytes"], "archive_bytes", 1)
    offset = integer(central["offset"], "directory offset", 1)
    etag = members["etag"]
    if (not isinstance(etag, str) or re.fullmatch(r'"[^"\r\n]+"', etag) is None
            or members["url"] != URL or receipt["url"] != URL
            or receipt["schema"] != "evr0018.resource-directory.v1"
            or receipt["status"] != "completed" or receipt["payload_members_read"] != 0
            or head["status"] != 200 or head["etag"] != etag
            or head["archive_bytes"] != size or receipt["members_sha256"] != sha(blobs["members"])
            or central["bytes"] != len(blobs["directory"])
            or offset + len(blobs["directory"]) > size):
        raise ValueError("cached source identity/receipt mismatch")
    records = [r for r in receipt["requests"] if r["kind"] == "central directory only"]
    wanted_range = [offset, offset + len(blobs["directory"]) - 1]
    if (len(records) != 1 or records[0]["status"] != 206
            or records[0]["etag"] != etag or records[0]["requested_range"] != wanted_range
            or records[0]["content_range"] != "bytes %d-%d/%d" % (*wanted_range, size)
            or records[0]["body_bytes"] != len(blobs["directory"])
            or records[0]["sha256"] != sha(blobs["directory"])):
        raise ValueError("central-directory receipt mismatch")
    entries = parse_directory(blobs["directory"], size, offset)
    listed = members["entries"]
    if (len(entries) != central["entries"] or len(listed) != len(entries)
            or len({v["filename"] for v in listed}) != len(listed)):
        raise ValueError("cached inventory count mismatch")
    for row in listed:
        if row["filename"] not in entries or any(row[k] != entries[row["filename"]][k]
                                                    for k in CORE_FIELDS):
            raise ValueError("inventory disagrees with parsed central directory")
    return Index(URL, etag, size, offset, entries, dict(expected_sha256))


@dataclass(frozen=True)
class Limits:
    body_bytes: int
    output_bytes: int
    member_bytes: int
    requests: int
    wall_seconds: float
    request_seconds: float

    def validate(self):
        for name in ("body_bytes", "output_bytes", "member_bytes", "requests"):
            integer(getattr(self, name), name, 1)
        for name in ("wall_seconds", "request_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid time limit: " + name)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise ValueError("redirect refused; no full ZIP or alternate source fallback")


def single_header(headers, name, default=None):
    """HTTPMessage permits duplicates; identity-bearing headers must be unique."""
    if hasattr(headers, "get_all"):
        values = headers.get_all(name, [])
        if len(values) > 1:
            raise ValueError("duplicate HTTP identity header: " + name)
        return values[0] if values else default
    return headers.get(name, default)  # Plain dicts in artificial transports.


@contextmanager
def deadline(seconds):
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("existing process alarm conflicts with request deadline")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*_):
        raise TimeoutError("range request wall deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class RangeReader:
    def __init__(self, index, limits, receipt, save, opener=None):
        self.index, self.limits, self.receipt, self.save = index, limits, receipt, save
        self.opener = opener or urllib.request.build_opener(NoRedirect()).open
        self.started = time.monotonic()

    def read(self, start, end):
        index, limits, receipt = self.index, self.limits, self.receipt
        integer(start, "range start")
        integer(end, "range end")
        length = end - start + 1
        remaining = limits.wall_seconds - (time.monotonic() - self.started)
        if not 0 <= start <= end < index.directory_offset:
            raise ValueError("range outside local member-data region")
        if remaining <= 0:
            raise TimeoutError("acquisition wall limit exceeded")
        if (receipt["response_body_bytes"] + length + 1 > limits.body_bytes
                or len(receipt["requests"]) >= limits.requests):
            raise ValueError("response-body/request cap exceeded before HTTP")
        item = dict(start=start, end=end, expected_bytes=length, body_bytes=0,
                    status=None, body_sha256=None)
        receipt["requests"].append(item)
        receipt["requested_body_upper_bound"] += length + 1
        self.save()
        request = urllib.request.Request(index.url, headers={
            "Range": "bytes=%d-%d" % (start, end), "If-Match": index.etag,
            "Accept-Encoding": "identity"})
        began, body = time.monotonic(), bytearray()
        duration = min(limits.request_seconds, remaining)
        try:
            with deadline(duration), self.opener(request, timeout=duration) as response:
                item.update(status=response.status, etag=single_header(response.headers, "ETag"),
                            content_range=single_header(response.headers, "Content-Range"),
                            content_length=single_header(response.headers, "Content-Length"))
                if (response.status != 206 or item["etag"] != index.etag
                        or item["content_range"] != "bytes %d-%d/%d" % (start, end, index.archive_bytes)
                        or item["content_length"] != str(length)
                        or single_header(response.headers, "Content-Encoding", "identity") != "identity"):
                    raise ValueError("range HTTP identity/header mismatch; body not read")
                while len(body) <= length:
                    try:
                        chunk = response.read(min(65536, length + 1 - len(body)))
                    except http.client.IncompleteRead as exc:
                        chunk = exc.partial
                        body.extend(chunk)
                        item["body_bytes"] += len(chunk)
                        receipt["response_body_bytes"] += len(chunk)
                        raise ValueError("incomplete range response") from exc
                    except BaseException:
                        receipt["response_body_bytes_exact"] = False
                        raise
                    if not chunk:
                        break
                    body.extend(chunk)
                    item["body_bytes"] += len(chunk)
                    receipt["response_body_bytes"] += len(chunk)
                if len(body) != length:
                    raise ValueError("range response length mismatch")
                return bytes(body)
        except urllib.error.HTTPError as exc:
            item["status"] = exc.code
            exc.close()  # Error body is not consumed, and there is no retry.
            raise
        finally:
            item["elapsed_seconds"] = time.monotonic() - began
            item["body_sha256"] = sha(body)
            self.save()


def extract_member(reader, entry):
    offset, packed, unpacked = (entry[k] for k in
                                ("local_header_offset", "compressed_bytes", "uncompressed_bytes"))
    raw = reader.read(offset, offset + 29)
    (sig, version, flags, method, mtime, mdate, crc, csize, usize,
     nlen, xlen) = struct.unpack("<4s5H3I2H", raw)
    expected = (b"PK\x03\x04", entry["version_needed"], entry["flags"],
                entry["compression_method"], entry["mod_time"], entry["mod_date"],
                int(entry["crc32_hex"], 16), packed, unpacked)
    if ((sig, version, flags, method, mtime, mdate, crc, csize, usize) != expected
            or nlen != len(entry["filename"].encode("ascii"))):
        raise ValueError("local ZIP header disagrees with cached central metadata")
    start = offset + 30 + nlen + xlen
    if start + packed > reader.index.directory_offset:
        raise ValueError("local member extent overlaps central directory")
    names = reader.read(offset + 30, start - 1)
    if names[:nlen] != entry["filename"].encode("ascii"):
        raise ValueError("local ZIP filename mismatch")
    extra_fields(names[nlen:])
    compressed = reader.read(start, start + packed - 1)
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(compressed, unpacked + 1)
    if (len(data) != unpacked or not decoder.eof or decoder.unconsumed_tail
            or decoder.unused_data):
        raise ValueError("deflate size or stream termination mismatch")
    if zlib.crc32(data) & 0xffffffff != crc:
        raise ValueError("uncompressed member CRC32 mismatch")
    return data, dict(local_header_sha256=sha(raw + names), compressed_sha256=sha(compressed),
                      sha256=sha(data), local_extra_bytes=xlen, data_offset=start)


def acquire(index, names, output_dir, receipt_path, limits, opener=None):
    """Acquire explicit images/K only. Never chooses scenes or reads pose members.

    SHA256 is a locally observed member checksum, not an author-published member
    digest. ETag/index/header/CRC checks bind extraction to the cached ZIP identity.
    Files are binary preserved; this function does not decode images or parse K.
    """
    limits.validate()
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("explicit unique frontend member names required")
    entries = []
    for name in names:
        frontend_kind(name)
        if name not in index.entries:
            raise ValueError("selected member absent from verified index")
        entries.append(index.entries[name])
    if (any(e["uncompressed_bytes"] > limits.member_bytes for e in entries)
            or sum(e["uncompressed_bytes"] for e in entries) > limits.output_bytes
            or sum(e["compressed_bytes"] + 30 + len(e["filename"]) + 3 for e in entries) > limits.body_bytes
            or 3 * len(entries) > limits.requests):
        raise ValueError("selected members exceed acquisition caps")
    output_dir, receipt_path = Path(output_dir), Path(receipt_path)
    if output_dir.exists() or output_dir.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("fresh output directory and receipt required; no overwrite/resume")
    receipt = dict(schema=SCHEMA, status="started", phase="prepare", url=index.url,
                   etag=index.etag, archive_bytes=index.archive_bytes,
                   index_sha256=index.fingerprints, source_sha256=sha(Path(__file__).read_bytes()),
                   selected_names=names, limits=vars(limits), requests=[], members=[],
                   response_body_bytes=0, response_body_bytes_exact=True,
                   requested_body_upper_bound=0, output_bytes=0, pose_gt_members_allowed=False,
                   hidden_retries=False, full_zip_fallback=False,
                   checksum_semantics="member SHA256 observed locally; not publisher-supplied")
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)

    def save():
        temp = receipt_path.with_name(receipt_path.name + ".tmp")
        with temp.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temp.replace(receipt_path)

    began = time.monotonic()
    try:
        output_dir.mkdir()
        reader = RangeReader(index, limits, receipt, save, opener)
        for entry in entries:
            receipt["phase"] = "member:" + entry["filename"]
            save()
            data, metadata = extract_member(reader, entry)
            if time.monotonic() - began > limits.wall_seconds:
                raise TimeoutError("acquisition wall limit exceeded before output")
            target = output_dir / entry["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            # An existing/symlinked parent in this newly owned tree is an error.
            parent = target.parent
            while parent != output_dir.parent:
                if parent.is_symlink():
                    raise ValueError("symlinked output path refused")
                parent = parent.parent
            with target.open("xb") as stream:
                stream.write(data)
            receipt["members"].append({**entry, **metadata, "kind": frontend_kind(entry["filename"])})
            receipt["output_bytes"] += len(data)
            save()
        if time.monotonic() - began > limits.wall_seconds:
            raise TimeoutError("acquisition wall limit exceeded after output persistence")
        receipt.update(status="completed", phase="completed")
        return receipt
    except BaseException as exc:
        message = str(exc).replace(str(output_dir), "<output>").replace(str(receipt_path), "<receipt>")
        receipt.update(status="error", error=dict(type=type(exc).__name__, message=message))
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - began
        save()
