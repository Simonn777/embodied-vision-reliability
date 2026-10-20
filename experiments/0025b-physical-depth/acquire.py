#!/usr/bin/env python3
"""Prepared only: frozen, two-URL acquisition and structural TAR indexing.

No image/depth/pose decoder or member extraction is present. Requires an outer
process-group supervisor (1800 s, 8 GiB descendants RSS, four thread limits).
"""
import argparse
import datetime
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request


ASSETS = (
    {"name": "living_room_traj2_frei_png.tar.gz",
     "url": "https://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz",
     "bytes": 429825445, "etag": '"199e9da5-5111591ffd040"'},
    {"name": "traj2_frei_png.tar.gz",
     "url": "https://www.doc.ic.ac.uk/~ahanda/traj2_frei_png.tar.gz",
     "bytes": 301828475, "etag": '"11fd897b-51115bf98c980"'},
)
BODY_CAP = 731653920
WALL_SECONDS = 1800
RSS_BYTES = 8 * 1024 ** 3
REQUEST_TIMEOUT = 30
CHUNK = 1024 * 1024
# Structural safety caps, not claims about uninspected archive contents.
MAX_MEMBERS = 100000
MAX_MEMBER_BYTES = 512 * 1024 ** 2
MAX_EXPANDED_BYTES = 16 * 1024 ** 3
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_relative(value):
    if not value or "\\" in value or "\x00" in value or value.startswith("/"):
        raise ValueError("unsafe relative path")
    if any(part == ".." for part in value.split("/")) or ":" in value:
        raise ValueError("unsafe relative path component")
    return str(PurePosixPath(value))


def verify_frozen(args):
    repo = Path(args.repo_root).resolve(strict=True)
    commit = args.expected_commit
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("expected-commit must be a full lowercase commit SHA")
    resolved = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", commit + "^{commit}"],
        check=True, capture_output=True, timeout=15).stdout.decode().strip()
    if resolved != commit:
        raise ValueError("commit resolution mismatch")
    result = {}
    for relative, current in ((args.source_relative, Path(__file__).resolve()),
                              (args.protocol_relative, repo / args.protocol_relative)):
        relative = safe_relative(relative)
        frozen = subprocess.run(
            ["git", "-C", str(repo), "show", commit + ":" + relative],
            check=True, capture_output=True, timeout=15).stdout
        if len(frozen) > 2 * 1024 ** 2:
            raise ValueError("unexpected source/protocol size")
        current_bytes = current.read_bytes()
        if current_bytes != frozen:
            raise ValueError("source/protocol differs from expected commit: " + relative)
        result[relative] = hashlib.sha256(frozen).hexdigest()
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError("redirect forbidden for fixed official URL")


def one_header(headers, name, required=True):
    values = headers.get_all(name) or []
    if len(values) > 1 or (required and len(values) != 1):
        raise ValueError("missing/duplicate " + name)
    return values[0].strip() if values else None


def validate_response(response, asset):
    if response.status != 200 or response.geturl() != asset["url"]:
        raise ValueError("unexpected response status/URL")
    length = one_header(response.headers, "Content-Length")
    etag = one_header(response.headers, "ETag")
    encoding = one_header(response.headers, "Content-Encoding", required=False)
    transfer = one_header(response.headers, "Transfer-Encoding", required=False)
    if length != str(asset["bytes"]) or etag != asset["etag"]:
        raise ValueError("publisher size/ETag changed; no retry")
    if transfer is not None or encoding not in (None, "identity", "x-gzip", "gzip"):
        raise ValueError("unexpected transfer/content encoding")
    return {"content_length": int(length), "etag": etag,
            "content_encoding": encoding, "status": response.status,
            "final_url": response.geturl()}


def limits(start):
    if time.monotonic() - start >= WALL_SECONDS:
        raise TimeoutError("1800-second operation limit")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    if peak_bytes > RSS_BYTES:
        raise MemoryError("self sampled/OS-reported peak exceeds 8 GiB")


class MeteredGzipReader:
    """Count all decompressed TAR bytes, including skipped member contents."""
    def __init__(self, compressed, start):
        self.stream = gzip.GzipFile(fileobj=compressed, mode="rb")
        self.start = start
        self.count = 0

    def read(self, size=-1):
        limits(self.start)
        if size < 0:
            size = CHUNK
        remaining = MAX_EXPANDED_BYTES - self.count
        if remaining <= 0:
            raise ValueError("expanded TAR byte cap exhausted")
        block = self.stream.read(min(size, remaining))
        self.count += len(block)
        return block


def index_archive(path, output, start):
    entries = []
    names = set()
    content_bytes = 0
    with path.open("rb") as compressed:
        reader = MeteredGzipReader(compressed, start)
        # Parse exactly one bounded 512-byte header at a time. Do not allow
        # tarfile's PAX/GNU-extension handlers to allocate attacker-sized buffers.
        # Unsupported extensions are a technical failure, never auto-repaired.
        while True:
            limits(start)
            header = reader.read(512)
            if len(header) != 512:
                raise EOFError("TAR missing complete header/end marker")
            if header == bytes(512):
                if reader.read(512) != bytes(512):
                    raise ValueError("TAR requires two zero end blocks")
                break
            if len(entries) >= MAX_MEMBERS:
                raise ValueError("member count cap exceeded")
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            name = safe_relative(member.name)
            if name in names:
                raise ValueError("duplicate canonical archive path")
            names.add(name)
            if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
                raise ValueError("links/devices/sparse/extension entries forbidden")
            if member.linkname:
                raise ValueError("nonempty link target forbidden")
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise ValueError("unexpected member size")
            if member.isdir() and member.size != 0:
                raise ValueError("directory with nonzero content size")
            content_bytes += member.size
            if content_bytes > MAX_EXPANDED_BYTES:
                raise ValueError("member content byte cap exceeded")
            entries.append({"name": name, "size": member.size,
                            "type": "file" if member.isfile() else "directory"})
            remaining = ((member.size + 511) // 512) * 512
            while remaining:
                block = reader.read(min(CHUNK, remaining))
                if not block:
                    raise EOFError("truncated TAR member/padding")
                remaining -= len(block)
        # Drain through gzip EOF to trigger CRC/trailer validation. Require only
        # TAR zero padding beyond the end marker; no second hidden archive.
        while True:
            tail = reader.read(CHUNK)
            if not tail:
                break
            if any(tail):
                raise ValueError("nonzero bytes beyond TAR end")
        expanded = reader.count
        reader.stream.close()
    record = {"schema": "evr0025b.archive-index.v1", "archive": path.name,
              "archive_sha256_tofu": digest(path), "members": entries,
              "member_count": len(entries), "member_content_bytes": content_bytes,
              "expanded_bytes": expanded, "member_values_decoded": False,
              "pixel_dimensions": "not inspected; later stage must verify 640x480"}
    save(output, record)
    limits(start)
    return {"filename": output.name, "sha256": digest(output),
            "bytes": output.stat().st_size, "member_count": len(entries),
            "expanded_bytes": expanded,
            "archive_sha256_tofu": record["archive_sha256_tofu"]}


def download(asset, destination, receipt, save_receipt, opener, start):
    row = {**asset, "status": "started", "body_bytes": 0,
           "sha256_kind": "trust-on-first-use; not a publisher digest"}
    receipt["assets"].append(row)
    save_receipt()
    partial = destination / (asset["name"] + ".part")
    h = hashlib.sha256()
    request = urllib.request.Request(
        asset["url"], headers={"Accept-Encoding": "identity", "If-Match": asset["etag"]})
    try:
        # urllib does not transparently decode Content-Encoding; raw gzip bytes
        # are stored exactly as received. No redirect/retry/range fallback.
        with opener.open(request, timeout=REQUEST_TIMEOUT) as response:
            row["observed_response"] = {"status": response.status,
                                        "final_url": response.geturl(),
                                        "headers": list(response.headers.items())}
            row["response"] = validate_response(response, asset)
            with partial.open("xb") as handle:
                while row["body_bytes"] < asset["bytes"]:
                    limits(start)
                    remaining = min(asset["bytes"] - row["body_bytes"],
                                    BODY_CAP - receipt["body_bytes"])
                    if remaining <= 0:
                        raise ValueError("total response-body cap exhausted")
                    try:
                        block = response.read(min(CHUNK, remaining))
                    except http.client.IncompleteRead as exc:
                        # A partial body exposed with an exception is still
                        # charged and preserved; it never enables another GET.
                        block = exc.partial
                        row["body_bytes"] += len(block)
                        receipt["body_bytes"] += len(block)
                        h.update(block)
                        handle.write(block)
                        raise
                    if not block:
                        raise EOFError("short response body")
                    row["body_bytes"] += len(block)
                    receipt["body_bytes"] += len(block)
                    h.update(block)
                    handle.write(block)
                    if row["body_bytes"] % (4 * CHUNK) == 0:
                        save_receipt()
                handle.flush()
                os.fsync(handle.fileno())
        limits(start)
        row["sha256"] = h.hexdigest()
        if partial.stat().st_size != asset["bytes"]:
            raise ValueError("saved file length mismatch")
        final = destination / asset["name"]
        if final.exists() or final.is_symlink():
            raise FileExistsError("refuse archive overwrite")
        partial.rename(final)
        row["status"] = "downloaded"
        save_receipt()
        row["index"] = index_archive(final, destination / (asset["name"] + ".index.json"), start)
        if row["index"]["archive_sha256_tofu"] != row["sha256"]:
            raise ValueError("stored archive changed after download")
        row["status"] = "completed"
        save_receipt()
    except BaseException:
        row["status"] = "error"
        row["received_sha256"] = h.hexdigest()
        row["saved_partial_bytes"] = partial.stat().st_size if partial.exists() else None
        raise


def run(args):
    destination = Path(args.destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination must be fresh; never resume/retry")
    destination.mkdir()
    start = time.monotonic()
    receipt_path = destination / "receipt.json"
    receipt = {"schema": "evr0025b.acquisition.v1", "status": "started",
               "started_utc": utc(), "expected_commit": args.expected_commit,
               "body_bytes": 0, "assets": [], "payload_parsing": "TAR structural index only",
               "ground_truth_transport": "whole archive may contain pose bytes; not decoded",
               "limits": {"wall_seconds": WALL_SECONDS, "body_bytes": BODY_CAP,
                          "request_timeout_seconds": REQUEST_TIMEOUT,
                          "outer_descendants_rss_bytes": RSS_BYTES,
                          "maximum_payload_GETs": 2, "automatic_retries": 0}}

    def persist():
        receipt["wall_seconds"] = time.monotonic() - start
        save(receipt_path, receipt)

    def alarm(_signum, _frame):
        raise TimeoutError("1800-second SIGALRM deadline")

    prior_handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        persist()
        receipt["source_sha256"] = verify_frozen(args)
        for name in THREAD_ENV:
            if os.environ.get(name) != "4":
                raise ValueError("outer supervisor must set " + name + "=4")
        receipt["thread_environment"] = {name: os.environ[name] for name in THREAD_ENV}
        opener = urllib.request.build_opener(NoRedirect())
        for asset in ASSETS:
            download(asset, destination, receipt, persist, opener, start)
        if receipt["body_bytes"] != BODY_CAP:
            raise ValueError("completed byte count mismatch")
        # Recheck executable/protocol bytes after all IO, without new payload.
        if verify_frozen(args) != receipt["source_sha256"]:
            raise ValueError("source binding changed")
        limits(start)
        receipt["status"] = "completed"
        persist()
        limits(start)  # Late fsync/save must not turn an over-time run into success.
    except BaseException as exc:
        receipt["status"] = "error"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        persist()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--protocol-relative", required=True)
    parser.add_argument("--source-relative", default="experiments/0025b-physical-depth/acquire.py")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "body_bytes": result["body_bytes"],
                      "wall_seconds": result["wall_seconds"]}, allow_nan=False))


if __name__ == "__main__":
    main()
