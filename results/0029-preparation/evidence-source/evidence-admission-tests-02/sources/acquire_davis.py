"""Draft phased DAVIS range acquisition. No work or network occurs on import.

The evaluator-GT phase deliberately remains closed until the parent supplies a
frozen verifier for all six prediction seals and clean supervision receipts.
This module never decodes JPEGs, PNGs, masks, or model files.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
import time
import zlib

SELF = "experiments/0029-memory-information/acquire_davis.py"
HELPER = "experiments/0023-mapfree/range_inputs.py"
MANIFEST = "configs/0029-davis-development.json"
PROTOCOL = "protocols/0029-signed-memory-diagnostic.md"
URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
ARCHIVE_BYTES = 832766765
ETAG = '"31a3032d-54c6d0af8b4dc"'
PHASE_CAPS = {"predictive_rgb": 57000000, "initial_supervision": 100000,
              "evaluator_gt": 500000}
PHASE_COUNTS = {"predictive_rgb": 386, "initial_supervision": 6,
                "evaluator_gt": 48}
COHORT = (("dogs-scale", 83), ("upside-down", 65), ("dog-agility", 25),
          ("cat-girl", 89), ("planes-water", 38), ("dog-gooses", 86))
MAX_EXTRA = 4096
MAX_MEMBER = 8 * 1024**2
TOTAL_RESERVATION = 60000000
SCHEMA = "evr0029.davis-acquisition.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounded_bytes(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size <= limit, "missing, symlink or oversized input")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "input grew beyond limit")
    return data


def read_json(path):
    def bad(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(bounded_bytes(path, 16 * 1024**2), parse_constant=bad)


def relative(name):
    require(isinstance(name, str) and name.isascii() and "\\" not in name
            and "\x00" not in name, "unsafe relative name")
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name
            and all(x not in ("", ".", "..") for x in path.parts),
            "unsafe relative name")
    return path


def source_bindings(root, commit):
    require(re.fullmatch(r"[a-f0-9]{40}", commit) is not None, "exact commit required")
    result = {}
    for name in (SELF, HELPER, MANIFEST, PROTOCOL):
        expected = subprocess.check_output(
            ["git", "show", commit + ":" + name], cwd=root, timeout=15)
        current = bounded_bytes(root / name, 16 * 1024**2)
        require(current == expected, "source differs from freeze: " + name)
        if name == SELF:
            require(bounded_bytes(Path(__file__), 1024**2) == current,
                    "executing acquisition source differs from frozen source")
        result[name] = sha(current)
    return result


def load_helper(path):
    name = "_evr0029_frozen_range_inputs"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result  # dataclass resolves its defining module.
    spec.loader.exec_module(result)
    return result


def load_plan(manifest_path, metadata_root, helper):
    plan = read_json(manifest_path)
    require(plan.get("schema") == "evr0029.davis-development-acquisition-plan.v1"
            and plan.get("metadata_only") is True
            and plan.get("media_member_bodies_read") == 0, "wrong metadata manifest")
    archive = plan["archive"]
    require(archive["url"] == URL and archive["archive_bytes"] == ARCHIVE_BYTES
            and archive["etag"] == ETAG, "wrong official archive identity")
    metadata = {}
    for name, expected in plan["source_bindings"].items():
        path = metadata_root / relative(name)
        data = bounded_bytes(path, 16 * 1024**2)
        require(len(data) == expected["bytes"] and sha(data) == expected["sha256"],
                "cached metadata changed: " + name)
        metadata[name] = data
    central_name = "work/0029-screening/video-cohort-metadata/davis-central.bin"
    require(central_name in metadata, "missing bound central directory")
    central = metadata[central_name]
    require(len(central) == archive["central_directory_bytes"]
            and sha(central) == archive["central_directory_sha256"],
            "central directory identity mismatch")
    offset = archive["central_directory_offset"]
    require(type(offset) is int and 0 < offset < offset + len(central) < ARCHIVE_BYTES,
            "central directory extent")
    parsed, position = {}, 0
    while position < len(central):
        require(position + 46 <= len(central), "truncated directory")
        values = struct.unpack_from("<4s6H3I5H2I", central, position)
        (sig, made, version, flags, method, mt, md, crc, packed, unpacked,
         nlen, xlen, clen, disk, _, external, local) = values
        end = position + 46 + nlen + xlen + clen
        require(sig == b"PK\x01\x02" and nlen and end <= len(central)
                and disk == 0, "invalid central record")
        name = central[position + 46:position + 46 + nlen].decode("ascii")
        require(name not in parsed, "duplicate central filename")
        parsed[name] = dict(filename=name, version_needed=version, flags=flags,
            compression_method=method, mod_time=mt, mod_date=md,
            crc32_hex="%08x" % crc, compressed_bytes=packed,
            uncompressed_bytes=unpacked, local_header_offset=local,
            central_record_sha256=sha(central[position:end]),
            central_offset_relative=position, filename_bytes=nlen,
            regular=(made >> 8 != 3 or stat.S_IFMT(external >> 16)
                     in (0, stat.S_IFREG)))
        position = end
    rows = plan["members"]
    require(isinstance(rows, list) and len(rows) == 440
            and len({r["filename"] for r in rows}) == 440, "wrong member inventory")
    require([(c["sequence"], c["frame_count"]) for c in plan["cohorts"]] == list(COHORT),
            "cohort changed")
    expected = {}
    for video_index, ((sequence, count), cohort) in enumerate(zip(COHORT, plan["cohorts"])):
        stems = sorted(PurePosixPath(n).stem for n in parsed
                       if n.startswith("DAVIS/JPEGImages/480p/" + sequence + "/")
                       and n.endswith(".jpg"))
        mask_stems = sorted(PurePosixPath(n).stem for n in parsed
                            if n.startswith("DAVIS/Annotations/480p/" + sequence + "/")
                            and n.endswith(".png"))
        queries = [4 + (2*j*(count-5)+7)//14 for j in range(8)]
        require(len(stems) == count and stems == mask_stems
                and stems == cohort["frame_stems"]
                and cohort["query_frame_indices"] == queries
                and cohort["query_frame_stems"] == [stems[i] for i in queries],
                "paired stems or fixed query schedule changed")
        for phase, indices in (("predictive_rgb", range(count)),
                               ("initial_supervision", [0]), ("evaluator_gt", queries)):
            kind, extension = ("JPEGImages", ".jpg") if phase == "predictive_rgb" else ("Annotations", ".png")
            for frame in indices:
                name = "DAVIS/%s/480p/%s/%s%s" % (kind, sequence, stems[frame], extension)
                expected[name] = (phase, video_index, sequence, frame, stems[frame],
                                  "%s/%s/%s%s" % (phase, sequence, stems[frame], extension))
    entries = {}
    for row in rows:
        name = row["filename"]
        relative(name)
        require(name in expected and name in parsed, "unselected or missing member")
        require(tuple(row[k] for k in ("stage", "video_index", "sequence", "frame_index",
                                      "frame_stem", "output_relative")) == expected[name],
                "member access phase or identity changed")
        record = parsed[name]
        require(record["regular"] and record["version_needed"] == 20
                and record["flags"] == 0 and record["compression_method"] == 8
                and 0 < record["compressed_bytes"] < MAX_MEMBER
                and 0 < record["uncompressed_bytes"] <= MAX_MEMBER,
                "unsupported selected ZIP member")
        require(all(row[k] == record[k] for k in record if k != "regular"),
                "manifest disagrees with cached central bytes")
        require(0 <= record["local_header_offset"]
                < record["local_header_offset"] + 30 + record["filename_bytes"]
                + record["compressed_bytes"] < offset, "invalid member extent")
        entries[name] = {**row}
    require(set(entries) == set(expected), "incomplete selected members")
    for phase, count in PHASE_COUNTS.items():
        chosen = [e for e in entries.values() if e["stage"] == phase]
        worst = sum(e["compressed_bytes"] + 30 + e["filename_bytes"] + MAX_EXTRA + 3
                    for e in chosen)
        require(len(chosen) == count and worst <= PHASE_CAPS[phase], "phase cap exceeded")
    require(sum(PHASE_CAPS.values()) <= TOTAL_RESERVATION, "total phase reservation exceeded")
    fingerprints = {"manifest": sha(bounded_bytes(manifest_path, 16 * 1024**2)),
                    "central_directory": sha(central)}
    index = helper.Index(URL, ETAG, ARCHIVE_BYTES, offset, entries, fingerprints)
    return plan, index


def require_prediction_admission(plan, manifest_sha256, proof_path):
    """No production GT authorization exists in this draft."""
    raise ValueError("evaluator_gt is closed: frozen all-six-prediction/clean-outer verifier required")


def extract_bounded(helper, reader, entry):
    """0023 extract_member logic with local-extra preflight before reading it."""
    offset = entry["local_header_offset"]
    packed, unpacked = entry["compressed_bytes"], entry["uncompressed_bytes"]
    raw = reader.read(offset, offset + 29)
    fields = struct.unpack("<4s5H3I2H", raw)
    sig, version, flags, method, mt, md, crc, csize, usize, nlen, xlen = fields
    expected = (b"PK\x03\x04", entry["version_needed"], entry["flags"],
                entry["compression_method"], entry["mod_time"], entry["mod_date"],
                int(entry["crc32_hex"], 16), packed, unpacked)
    require(fields[:9] == expected and nlen == len(entry["filename"].encode("ascii")),
            "local header differs from central metadata")
    require(xlen <= MAX_EXTRA, "local ZIP extra field exceeds frozen cap")
    start = offset + 30 + nlen + xlen
    require(start + packed <= reader.index.directory_offset, "member overlaps central directory")
    names = reader.read(offset + 30, start - 1)
    require(names[:nlen] == entry["filename"].encode("ascii"), "local filename changed")
    helper.extra_fields(names[nlen:])
    compressed = reader.read(start, start + packed - 1)
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(compressed, unpacked + 1)
    require(len(data) == unpacked and decoder.eof and not decoder.unconsumed_tail
            and not decoder.unused_data, "deflate size/termination mismatch")
    require(zlib.crc32(data) & 0xffffffff == crc, "member CRC32 mismatch")
    return data, {"sha256": sha(data), "compressed_sha256": sha(compressed),
                  "local_header_sha256": sha(raw + names),
                  "data_offset": start, "local_extra_bytes": xlen}


def acquire_phase(helper, plan, index, phase, output, *, opener=None,
                  expected_commit=None, source_sha256=None, source_check=None, proof_path=None,
                  gt_verifier=None):
    """Transport unit; production calls this only after source/metadata binding."""
    require(phase in PHASE_CAPS, "unknown phase")
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "fresh phase output required")
    output = output.resolve()
    output.mkdir(parents=True)
    started = time.monotonic()
    record = {"schema": SCHEMA, "status": "started", "phase": phase,
              "expected_commit": expected_commit,
              "archive": plan["archive"], "source_sha256": source_sha256 or {},
              "manifest_sha256": index.fingerprints["manifest"],
              "index_sha256": index.fingerprints, "members": [], "requests": [],
              "response_body_bytes": 0, "response_body_bytes_exact": True,
              "requested_body_upper_bound": 0, "output_bytes": 0,
              "phase_body_cap_bytes": PHASE_CAPS[phase],
              "all_phases_reserved_bytes": sum(PHASE_CAPS.values()),
              "total_reservation_bytes": TOTAL_RESERVATION,
              "media_decoded": False, "hidden_retries": False,
              "full_zip_fallback": False, "gt_admission": None,
              "checksum_semantics": "member SHA256 observed locally; not publisher-supplied"}

    def save():
        record["wall_seconds"] = time.monotonic() - started
        pending = output / "receipt.pending"
        with pending.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        pending.replace(output / "receipt.json")

    save()
    try:
        if phase == "evaluator_gt":
            verifier = gt_verifier or require_prediction_admission
            record["gt_admission"] = verifier(plan, index.fingerprints["manifest"], proof_path)
            require(isinstance(record["gt_admission"], dict)
                    and record["gt_admission"].get("eligible") is True,
                    "missing prediction admission")
            save()
        selected = [entry for entry in index.entries.values() if entry["stage"] == phase]
        require(selected and len(selected) == PHASE_COUNTS[phase], "phase member count")
        limits = helper.Limits(PHASE_CAPS[phase], 64 * 1024**2, MAX_MEMBER,
                              3 * len(selected), 1800, 30)
        limits.validate()
        record["limits"] = vars(limits)
        record["selected_names"] = [e["filename"] for e in selected]
        save()
        reader = helper.RangeReader(index, limits, record, save, opener)
        for entry in selected:
            record["current_member"] = entry["filename"]
            save()
            data, metadata = extract_bounded(helper, reader, entry)
            require(len(data) <= limits.member_bytes
                    and record["output_bytes"] + len(data) <= limits.output_bytes,
                    "output-byte cap")
            target = output / "members" / relative(entry["output_relative"])
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent
            while parent != output:
                require(not parent.is_symlink(), "symlinked output parent")
                parent = parent.parent
            with target.open("xb") as stream:
                stream.write(data)
            record["members"].append({**entry, **metadata, "bytes": len(data),
                "saved_relative": target.relative_to(output).as_posix()})
            record["output_bytes"] += len(data)
            require(time.monotonic() - started <= 1800, "phase wall limit")
            save()
        if source_check is not None:
            source_check()
        if phase == "evaluator_gt":
            require(verifier(plan, index.fingerprints["manifest"], proof_path)
                    == record["gt_admission"], "prediction admission changed")
        require(time.monotonic() - started <= 1800, "phase wall limit")
        record.update(status="completed", current_member=None)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__,
                      error=str(error).replace(str(output), "$OUTPUT"))
        raise
    finally:
        save()
    return record


def run(args):
    root = Path(args.repo_root).resolve()
    metadata_root = Path(args.metadata_root).resolve()
    raw_output = Path(args.output_dir).absolute()
    require(not raw_output.exists() and not raw_output.is_symlink(), "fresh phase output required")
    output = raw_output.resolve()
    require(output != root and root not in output.parents, "media output must be outside repository")
    bindings = source_bindings(root, args.expected_commit)
    helper = load_helper(root / HELPER)
    plan, index = load_plan(root / MANIFEST, metadata_root, helper)

    def postcheck():
        require(all(sha(bounded_bytes(root / name, 16 * 1024**2)) == digest
                    for name, digest in bindings.items()), "frozen source changed")
        _, current = load_plan(root / MANIFEST, metadata_root, helper)
        require(current.fingerprints == index.fingerprints, "cached inputs changed")

    result = acquire_phase(helper, plan, index, args.phase, output,
        expected_commit=args.expected_commit, source_sha256=bindings, source_check=postcheck,
        proof_path=getattr(args, "prediction_proof", None))
    print(json.dumps({k: result[k] for k in
                     ("status", "phase", "response_body_bytes", "wall_seconds")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "metadata-root", "phase", "output-dir"):
        parser.add_argument("--" + name, required=True,
                            **({"choices": tuple(PHASE_CAPS)} if name == "phase" else {}))
    parser.add_argument("--prediction-proof")
    run(parser.parse_args())
