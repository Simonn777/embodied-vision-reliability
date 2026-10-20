#!/usr/bin/env python3
"""Build an UNALLOCATED VOST proposal from cached metadata; no transport/decoding."""
import hashlib
import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
META = WORK / "0029-screening/video-cohort-metadata"
SALT = b"evr0029:video-cohort:2026-10-20\0vost-dev\0"
EXPECTED = (("4691_cut_guava", 86), ("817_cut_onion", 90),
            ("9675_fold_clothes", 201), ("1934_tear_film", 69))
PINS = {
    "0029-screening/video-cohort-plan.json": "f4962c6c8b3ca0806ee4bb751bc6bb6ad74de97771aed26ebd33106b078fc5d1",
    "0029-screening/video-cohort-metadata/vost-members.json": "585a8e7e2719149f9de89af86c1594ac322c264b3e3c6a43c8cee5f1f0920b10",
    "0029-screening/video-cohort-metadata/vost-train-txt.txt": "3f30c775779a30cc239cffbd9b254e083553734862462358e9c356ba94587060",
    "0029-screening/video-cohort-metadata/proposed-selection-rule.txt": "7de45f7e4aa2d23024f14aa7727e4301a808cff5ea9a3083743c03d955941796",
    "0029-screening/video-cohort-metadata/vost-central.bin": "7685955fd97adaade763a9a8f667ab8589b8fd63bac596eb207697ac68ed6d34",
    "0029-screening/video-cohort-metadata/vost-eocd.bin": "af99a3715c787a348a641001d14595eab805693d52e480c3ad35ab59a90d65cd",
    "0029-screening/video-cohort-metadata/vost-head.json": "26e9debf371d8c10a1e3c492128465778e14ce3af48ece06f772cbe14d1cd028",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def central_entry(raw, entry):
    """Verify a selected descriptor, resolving ZIP64 sentinel fields in order."""
    offset = entry["central_offset_relative"]
    f = struct.unpack_from("<4s6H3I5H2I", raw, offset)
    sig, made, needed, flags, method, _, _, crc, comp, size, n, e, c, disk, _, _, local = f
    require(sig == b"PK\x01\x02" and offset + 46 + n + e + c <= len(raw), "central bounds")
    name = raw[offset + 46:offset + 46 + n].decode("utf-8" if flags & 0x800 else "cp437")
    extra = raw[offset + 46 + n:offset + 46 + n + e]
    cursor, zip64 = 0, None
    while cursor < len(extra):
        require(cursor + 4 <= len(extra), "extra header truncated")
        tag, length = struct.unpack_from("<HH", extra, cursor)
        require(cursor + 4 + length <= len(extra), "extra payload truncated")
        if tag == 1:
            require(zip64 is None, "duplicate ZIP64 extra")
            zip64 = extra[cursor + 4:cursor + 4 + length]
        cursor += 4 + length
    values, cursor = [size, comp, local, disk], 0
    used_zip64 = []
    for i, label in enumerate(("uncompressed_bytes", "compressed_bytes", "local_header_offset", "disk")):
        if values[i] == (0xFFFF if i == 3 else 0xFFFFFFFF):
            fmt, width = ("<I", 4) if i == 3 else ("<Q", 8)
            require(zip64 is not None and cursor + width <= len(zip64), "missing ZIP64 value")
            values[i] = struct.unpack_from(fmt, zip64, cursor)[0]
            cursor += width
            used_zip64.append(label)
    size, comp, local, disk = values
    actual = dict(filename=name, local_header_offset=local, compressed_bytes=comp,
                  uncompressed_bytes=size, compression=method, crc32_hex=f"{crc:08x}",
                  flags=flags, central_offset_relative=offset)
    require(actual == entry and disk == 0, "central/metadata mismatch or multidisk")
    return dict(central_version_made_by=made, central_version_needed=needed,
                central_extra_bytes=e, zip64_fields=used_zip64)


def build():
    sources, blobs = {}, {}
    for relative, digest in PINS.items():
        data = (WORK / relative).read_bytes()
        require(sha(data) == digest, f"source hash mismatch: {relative}")
        sources[relative] = {"sha256": digest, "bytes": len(data)}
        blobs[Path(relative).name] = data
    inventory = json.loads(blobs["vost-members.json"])
    plan = json.loads(blobs["video-cohort-plan.json"])["datasets"]["vost"]
    head = json.loads(blobs["vost-head.json"])["headers"]
    end, central = blobs["vost-eocd.bin"], blobs["vost-central.bin"]
    require(len(end) == 98, "unexpected ZIP64 end-record size")
    z = struct.unpack_from("<4sQ2H2I4Q", end)
    locator = struct.unpack_from("<4sIQI", end, 56)
    eocd = struct.unpack_from("<4s4H2IH", end, 76)
    require(z[:2] == (b"PK\x06\x06", 44) and z[4:6] == (0, 0), "ZIP64 EOCD invalid")
    count, central_size, central_start = z[7], z[8], z[9]
    require(z[6] == count == len(inventory["members"]) == 153136, "entry count mismatch")
    require(central_size == len(central) == 21042989, "central size mismatch")
    require(locator == (b"PK\x06\x07", 0, central_start + central_size, 1), "ZIP64 locator")
    require(eocd == (b"PK\x05\x06", 0, 0, 65535, 65535, central_size, 0xFFFFFFFF, 0), "EOCD")
    require(central_start + central_size + len(end) == inventory["archive_bytes"] == 54012104924,
            "archive span mismatch")
    require(inventory["etag"] == head["ETag"] == plan["archive_etag"], "ETag mismatch")
    require(int(head["Content-Length"]) == plan["archive_bytes"] == inventory["archive_bytes"],
            "archive byte mismatch")
    require(inventory["central_sha256"] == sha(central) == plan["central_sha256"], "central hash")
    require(plan["member_metadata_sha256"] == sha(blobs["vost-members.json"]), "plan inventory hash")
    train = blobs["vost-train-txt.txt"].decode().splitlines()
    require(len(train) == len(set(train)) == 572 and all(x.strip() == x and x for x in train), "TRAIN list")
    declared = {s["sequence"]: s for s in plan["sequences"] if s["split"] == "train"}
    require(set(train) == set(declared), "plan/TRAIN membership mismatch")
    ranked = sorted(train, key=lambda name: (sha(SALT + name.encode()), name))
    require(tuple(ranked[:4]) == tuple(name for name, _ in EXPECTED), "rank reproduction")
    entries = inventory["members"]
    by_name = {e["filename"]: e for e in entries}
    offsets = sorted(e["local_header_offset"] for e in entries)
    require(len(by_name) == len(set(offsets)) == len(entries), "duplicate member/offset")
    following = dict(zip(offsets, offsets[1:] + [central_start]))
    clips, members = [], []
    for rank, (name, expected_count) in enumerate(EXPECTED, 1):
        rgb = {Path(k).stem: v for k, v in by_name.items()
               if k.startswith(f"VOST/JPEGImages/{name}/") and k.endswith(".jpg")}
        masks = {Path(k).stem: v for k, v in by_name.items()
                 if k.startswith(f"VOST/Annotations/{name}/") and k.endswith(".png")}
        stems = sorted(rgb)
        require(set(rgb) == set(masks) and len(stems) == expected_count, "complete paired clip")
        require(stems == declared[name]["stems"], "plan frame stems differ")
        clips.append(dict(sequence=name, split="train", rank=rank,
                          rank_sha256=sha(SALT + name.encode()), frames=len(stems), frame_stems=stems,
                          first_frame_stem=stems[0], original_recording_id=None,
                          first_frame_object_ids=None, object_policy="all supplied IDs except 0 and 255"))
        for i, stem in enumerate(stems):
            for entry, phase in ((rgb[stem], "rgb"), (masks[stem], "initial" if i == 0 else "metric")):
                central_fields = central_entry(central, entry)
                require(entry["flags"] == 8 and entry["compression"] == 8, "unexpected selected flags/method")
                header = 30 + len(entry["filename"].encode("utf-8")) + 4096
                require(entry["local_header_offset"] + 30 + entry["compressed_bytes"]
                        <= following[entry["local_header_offset"]], "member overlap")
                members.append(dict(entry, **central_fields, sequence=name, frame_index=i, stem=stem,
                                    phase=phase, payload_sha256=None,
                                    local_header_bytes_cap=header,
                                    response_body_bytes_cap=header + entry["compressed_bytes"],
                                    next_local_header_or_central_offset=following[entry["local_header_offset"]]))
    def totals(rows):
        return {"members": len(rows),
                "compressed_bytes": sum(r["compressed_bytes"] for r in rows),
                "uncompressed_bytes": sum(r["uncompressed_bytes"] for r in rows),
                "local_header_bytes_cap": sum(r["local_header_bytes_cap"] for r in rows),
                "response_body_bytes_cap": sum(r["response_body_bytes_cap"] for r in rows)}
    total = totals(members)
    require(len(members) == 892 and sum(c["frames"] for c in clips) == 446, "coverage")
    require(total["compressed_bytes"] == 168229493 and total["response_body_bytes_cap"] == 171951497,
            "byte-bound reproduction")
    return dict(schema="evr0035.proposed-vost-manifest.v1", status="proposed_unallocated",
                allocated=False, acquisition_authorized=False, scientific_execution_authorized=False,
                builder={"path": "0035-transformation-support/build_manifest.py", "sha256": sha(Path(__file__).read_bytes())},
                metadata_sources=sources, archive={"url": inventory["url"], "bytes": inventory["archive_bytes"],
                    "etag": inventory["etag"], "sha256": None, "zip64": True, "central_start": central_start,
                    "central_bytes": central_size, "central_sha256": sha(central), "entries": count,
                    "end_record_version_needed": z[3], "identity_refreshed": False},
                selection={"status": "new prospective extension, not historical allocation",
                    "rank_input_utf8": SALT.decode() + "{clip}", "sort": ["sha256", "clip"],
                    "population": "all 572 official TRAIN names", "prefix_count": 4,
                    "independent_parent_claim": False, "semantic_or_outcome_filter": False},
                clips=clips, members=members, totals=total,
                phases={p: totals([r for r in members if r["phase"] == p]) for p in ("rgb", "initial", "metric")},
                transport_requirements={"status": "unqualified; this is not a downloader configuration",
                    "local_extra_field_cap": 4096, "zip64_offsets_and_sizes": True,
                    "selected_flags": 8, "selected_compression": 8,
                    "selected_local_version": "unread; cannot assume version 20",
                    "central_size_exceeds_16MiB": True,
                    "checks": ["exact archive ETag/length, 206 and Content-Range before body acceptance",
                               "64-bit ZIP64 central/EOCD parsing and bounded local extra fields",
                               "local name/method/flags and physical bounds; central sizes/CRC for flag-8 descriptor entries",
                               "bounded DEFLATE output, exact uncompressed length and CRC; no whole archive fallback"],
                    "error_body_budget": None, "retry_budget": None, "request_cap": None,
                    "new_metadata_refresh_budget": None, "allocation_bytes": None,
                    "unknown_limits_block_acquisition": True,
                    "bound_excludes": ["HTTP/TLS overhead", "errors", "retries", "new metadata requests"]},
                phase_policy={"rgb": "all 446 frames", "initial": "first mask of each clip; all supplied target IDs",
                    "metric": "442 remaining masks; future scientific admission/seal barrier not implemented",
                    "initial_metric_reuse": "reuse the four initial masks if official metric needs them; no duplicate acquisition"},
                scientific_limits=["official VOST J/Jtr, 255 ignored; no official JF claim",
                                   "unknown within-TRAIN original recording groups and first-mask object counts",
                                   "no metric/overlap/oracle rule or experiment admission supplied here"])


def main():
    output, report = HERE / "proposed-manifest.json", HERE / "manifest-build-report.json"
    require(not output.exists() and not report.exists(), "fresh own outputs required")
    manifest = build()
    data = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    output.write_bytes(data)
    result = dict(status="metadata_validation_passed", allocated=False, acquisition_authorized=False,
                  scientific_payload_reads=0, network_calls=0, manifest_sha256=sha(data), manifest_bytes=len(data),
                  builder_sha256=manifest["builder"]["sha256"], clips=[c["sequence"] for c in manifest["clips"]],
                  frames=446, totals=manifest["totals"], phases=manifest["phases"])
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
