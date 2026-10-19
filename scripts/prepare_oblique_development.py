"""Cache-only 0011 preparation; the restricted pickle materializes all 50 videos.

Only the two newly allocated reserve groups are inspected/exported. This module
does not run a model or score predictions. Use a credential-free Linux process
and the external 300-second supervisor; the pickle allowlist is not a DoS sandbox.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import time
import zipfile

ARCHIVE_BYTES = 187291581
ARCHIVE_SHA256 = "ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc"
MANIFEST_BYTES = 70658
MANIFEST_SHA256 = "b5415fd3eab77806ec71a974e2d875734f052ed3e44087a6e5b0a6d543cbbba3"
GROUPS = (("g-036", 36), ("g-002", 2))
FRAME_IDS = tuple(range(0, 250, 16))
SHAPE = (250, 256, 256, 30)  # time, height, width, points
MAX_UNCOMPRESSED = 4 * 1024**3
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def dependencies():
    import numpy as np
    from evr.bootstapir_dev import NumpyOnlyUnpickler, inspect_pickle
    from evr.pilot_identity import digest_array
    return np, NumpyOnlyUnpickler, inspect_pickle, digest_array


def array_records(arrays):
    _, _, _, digest_array = dependencies()
    return {k: {"shape": list(a.shape), "dtype": str(a.dtype),
                "sha256": digest_array(a)} for k, a in arrays.items()}


def extract_example(example, expected_identity, *, shape=SHAPE,
                    frame_ids=FRAME_IDS, query_count=8):
    """Pure extraction. Smaller dimensions/identities are for artificial tests only.

    Predictor keys: video THWC uint8, query_xy Q2 float32 in original pixels,
    point_ids Q int64, frame_ids T int64. Ground truth: points TQ2 float64 in
    original pixels, occluded TQ bool and the same point_ids/frame_ids. Future
    annotations never select queries.
    """
    np, _, _, _ = dependencies()
    t, h, w, n = shape
    if (type(example) is not dict or set(example) != {"video", "points", "occluded"}
            or any(type(a) is not np.ndarray for a in example.values())):
        raise ValueError("exact numeric example schema required")
    expected = {"video": ((t, h, w, 3), np.dtype("uint8")),
                "points": ((n, t, 2), np.dtype("float64")),
                "occluded": ((n, t), np.dtype("bool"))}
    for key, (dims, dtype) in expected.items():
        if example[key].shape != dims or example[key].dtype != dtype:
            raise ValueError("array shape/dtype mismatch: " + key)
    points, occluded = example["points"], example["occluded"]
    if not np.isfinite(points).all():
        raise ValueError("nonfinite points")
    identities = array_records(example)
    if identities != expected_identity:
        raise ValueError("original array identity mismatch")
    if (type(query_count) is not int or query_count < 1 or
            any(type(x) is not int for x in frame_ids) or
            not frame_ids or frame_ids[0] != 0 or
            list(frame_ids) != sorted(set(frame_ids)) or frame_ids[-1] >= t):
        raise ValueError("invalid extraction indices")
    ids = np.flatnonzero(~occluded[:, 0])[:query_count].astype(np.int64)
    if len(ids) != query_count:
        raise ValueError("insufficient frame-zero-visible queries; no replacements")
    queries = points[ids, 0]
    if np.any(queries < 0) or np.any(queries > 1):
        raise ValueError("frame-zero query outside normalized image")
    frames = np.asarray(frame_ids, dtype=np.int64)
    scale = np.array([w, h], dtype=np.float64)
    predictor = {"video": example["video"][frames].copy(),
                 "query_xy": (queries * scale).astype(np.float32),
                 "point_ids": ids, "frame_ids": frames}
    truth = {"points": (points[ids][:, frames].transpose(1, 0, 2) * scale),
             "occluded": occluded[ids][:, frames].T.copy(),
             "point_ids": ids.copy(), "frame_ids": frames.copy()}
    return predictor, truth, identities


def stable_stat(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def pinned_open(path, size, digest):
    """Return the same verified regular-file descriptor that will be consumed."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    stream = os.fdopen(fd, "rb")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError("input regular-file/size gate")
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
        if h.hexdigest() != digest or stable_stat(before) != stable_stat(os.fstat(fd)):
            raise ValueError("input digest/stability gate")
        stream.seek(0)
        return stream, stable_stat(before)
    except BaseException:
        stream.close()
        raise


def pickle_member(archive):
    members = archive.infolist()
    names = [m.filename for m in members]
    if len(names) != len(set(names)):
        raise ValueError("duplicate ZIP member names")
    for m in members:
        name = m.filename
        parts = PurePosixPath(name).parts
        mode = (m.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if (not name or "\\" in name or ":" in name or "\x00" in name or
                name.startswith("/") or ".." in parts or
                kind not in (0, stat.S_IFREG, stat.S_IFDIR) or m.flag_bits & 1):
            raise ValueError("unsafe ZIP member")
    if sum(m.file_size for m in members) > MAX_UNCOMPRESSED:
        raise ValueError("uncompressed archive exceeds 4 GiB")
    choices = [m for m in members if not m.is_dir() and m.filename.endswith(".pkl")]
    if len(choices) != 1:
        raise ValueError("exactly one pickle member required")
    return choices[0]


def validate_manifest(manifest):
    if (manifest.get("archive", {}).get("bytes") != ARCHIVE_BYTES or
            manifest["archive"].get("sha256") != ARCHIVE_SHA256):
        raise ValueError("manifest archive mismatch")
    reserve = [g for g in manifest["groups"] if g["partition"] == "reserve"][:2]
    if [(g["group_id"], g["members"]) for g in reserve] != [(g, [i]) for g, i in GROUPS]:
        raise ValueError("first two reserve groups differ")
    identities = {item["archive_index"]: item for item in manifest["identities"]}
    if len(manifest["identities"]) != 50 or set(identities) != set(range(50)):
        raise ValueError("50 original identities required")
    for group, index in GROUPS:
        rows = [r for r in manifest["allocation"] if r["archive_index"] == index]
        if len(rows) != 1 or rows[0]["partition"] != "reserve" or rows[0]["group_id"] != group:
            raise ValueError("reserve allocation mismatch")
    return identities


def write_json_exclusive(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz(path, arrays):
    np, _, _, _ = dependencies()
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)  # Atomic no-overwrite publication, after stream close.
    partial.unlink()
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": file_digest(path), "arrays": array_records(arrays)}


def prepare(archive_path, manifest_path, output, *, resource_record=None):
    """One attempt; failures retain the reserved directory, marker and partials."""
    output = Path(output)
    output.mkdir(exist_ok=False)  # Parent must already exist; existing empty dirs refuse.
    start = time.monotonic()
    marker = {"schema_version": 1, "operation": "0011-oblique-development-preparation",
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json_exclusive(output / "started.json", marker)
    receipt = dict(marker, status="failed", resources=resource_record,
                   archive={"bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256},
                   original_manifest={"bytes": MANIFEST_BYTES, "sha256": MANIFEST_SHA256},
                   original_manifest_modified=False, new_network_bytes=0, groups=[], files={},
                   materialization="restricted pickle loads all 50 examples; only 36 and 2 inspected/exported",
                   allocation_scope="new development; original reserve allocation remains immutable",
                   scene_independence="unknown", full_raw_publication_allowed=False)
    try:
        np, unpickler, inspect_pickle, _ = dependencies()
        receipt["numpy_version"] = np.__version__
        stream, before = pinned_open(manifest_path, MANIFEST_BYTES, MANIFEST_SHA256)
        with stream:
            manifest = json.load(stream)
            if stable_stat(os.fstat(stream.fileno())) != before:
                raise ValueError("manifest changed while reading")
        identities = validate_manifest(manifest)
        stream, before = pinned_open(archive_path, ARCHIVE_BYTES, ARCHIVE_SHA256)
        with stream, zipfile.ZipFile(stream) as archive:
            member = pickle_member(archive)
            with archive.open(member) as payload:
                receipt["pickle_opcode_audit"] = inspect_pickle(payload)
            with archive.open(member) as payload:
                data = unpickler(payload).load()
                if payload.read(1):
                    raise ValueError("trailing bytes after pickle STOP")
            if type(data) is not list or len(data) != 50:
                raise ValueError("50-example list required")
            prepared = []
            for group, index in GROUPS:
                prediction, truth, arrays = extract_example(data[index], identities[index]["arrays"])
                prepared.append((group, index, prediction, truth, arrays))
            if stable_stat(os.fstat(stream.fileno())) != before:
                raise ValueError("archive changed while decoding")
        receipt["pickle_member"] = {"name": member.filename, "bytes": member.file_size,
                                     "crc32": member.CRC, "crc_checked_to_member_eof": True}
        for group, index, prediction, truth, arrays in prepared:
            record = {"group_id": group, "archive_index": index, "input_arrays": arrays,
                      "point_ids": prediction["point_ids"].tolist(),
                      "frame_ids": prediction["frame_ids"].tolist(),
                      "query_selection": "first 8 visible at frame 0, ascending; no future labels"}
            receipt["groups"].append(record)
            record["predictor"] = save_npz(output / (group + ".predictor.npz"), prediction)
            receipt["files"][record["predictor"]["name"]] = {
                k: record["predictor"][k] for k in ("bytes", "sha256")}
            record["ground_truth"] = save_npz(output / (group + ".ground_truth.npz"), truth)
            receipt["files"][record["ground_truth"]["name"]] = {
                k: record["ground_truth"][k] for k in ("bytes", "sha256")}
        receipt["status"] = "completed_preparation"
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - start
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["marker_sha256"] = file_digest(output / "started.json")
        write_json_exclusive(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    import resource
    if not hasattr(os, "sched_getaffinity") or any(os.environ.get(k) != "4" for k in THREAD_ENV):
        raise RuntimeError("Linux CPU affinity and all five thread environment variables=4 required")
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if not cpus:
        raise RuntimeError("no CPU available")
    os.sched_setaffinity(0, cpus)
    old_hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    limit = min(8 * 1024**3, old_hard) if old_hard != resource.RLIM_INFINITY else 8 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    def timeout(signum, frame):
        raise TimeoutError("300-second preparation deadline")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        prepare(args.archive, args.manifest, args.output,
                resource_record={"affinity": sorted(os.sched_getaffinity(0)),
                                 "limit_as_bytes": limit, "thread_environment": {k: os.environ[k] for k in THREAD_ENV},
                                 "alarm_seconds": 300, "external_hard_supervisor_required": True})
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
