"""Prospective VOST phased range acquisition; no I/O on import, no admission implied.

Uses immutable 0023 HTTP primitives and preserved metadata central parser.
Metric transport is closed unless a separately frozen caller supplies a verifier.
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

SELF = "experiments/0035-transformation-support/acquire.py"
PARSER = "experiments/0035-transformation-support/build_manifest.py"
BUILDER_V2 = "experiments/0035-transformation-support/build_manifest_v2.py"
ADMIT = "experiments/0035-transformation-support/admit.py"
ADMISSION = "experiments/0035-transformation-support/ADMISSION.md"
HELPER = "experiments/0023-mapfree/range_inputs.py"
PINS = {
    PARSER: "d42448b5382df8fe62c1703c80046701416bbd6462ecc743c917ee787552e925",
    BUILDER_V2: "081555b78e5b5c0149536b02e2317274cbec535421f829263b5566a9268f5445",
    HELPER: "519a9eb08fab5a8ad8d9a9257927281093abac3354fd14cadd8e7599fa69ea50",
}
MANIFEST_SHA = "141371225d59c0b1ef483ee3b7565aace037033df072eabb2839e8d4f19bfb07"
PHASE_CAPS = {"initial": 50000, "rgb": 170000000, "metric": 5000000}
PHASE_COUNTS = {"initial": 4, "rgb": 446, "metric": 442}
RESERVATION, MAX_EXTRA, MAX_MEMBER, MAX_OUTPUT = 180000000, 4096, 8 * 1024**2, 200 * 1024**2
SCHEMA = "evr0035.vost-acquisition.v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    require(isinstance(name, str) and name.isascii() and "\\" not in name and "\0" not in name,
            "unsafe relative name")
    p = PurePosixPath(name)
    require(not p.is_absolute() and str(p) == name and all(v not in ("", ".", "..") for v in name.split("/")),
            "unsafe relative name")
    return p


def read_bytes(path, limit=64 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "bounded regular file required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "input grew beyond limit")
    return raw


def read_json(path):
    return json.loads(read_bytes(path), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def source_bindings(root, commit, manifest_relative, admission_relative):
    require(re.fullmatch("[a-f0-9]{40}", commit or "") is not None, "exact commit required")
    names = (SELF, HELPER, PARSER, BUILDER_V2, ADMIT, ADMISSION, manifest_relative, admission_relative)
    require(len(set(names)) == len(names), "duplicate source path")
    bindings = {}
    for name in names:
        relative(name)
        expected = subprocess.check_output(["git", "show", commit + ":" + name], cwd=root, timeout=15)
        current = read_bytes(root / name)
        require(current == expected, "source differs from freeze: " + name)
        if name == SELF:
            require(current == read_bytes(Path(__file__)), "executing source differs")
        if name in PINS:
            require(sha(current) == PINS[name], "immutable helper/parser changed")
        bindings[name] = sha(current)
    require(bindings[manifest_relative] == MANIFEST_SHA, "wrong v2 manifest")
    return bindings


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_admission(admission):
    """A proposed manifest does not authorize transport; require a later frozen declaration."""
    expected = dict(schema="evr0035.acquisition-admission.v1", acquisition_authorized=True,
                    manifest_sha256=MANIFEST_SHA, reservation_bytes=RESERVATION,
                    phase_body_caps=PHASE_CAPS, phase_order=["initial", "rgb", "metric"],
                    retries=0, request_seconds=30, wall_seconds=1800, member_bytes=MAX_MEMBER,
                    output_bytes=MAX_OUTPUT, local_extra_cap=MAX_EXTRA)
    require(all(admission.get(k) == v and type(admission.get(k)) is type(v) for k, v in expected.items()),
            "explicit exact frozen acquisition admission required")
    require(isinstance(admission.get("allocation_reference"), str) and admission["allocation_reference"].strip(),
            "allocation reference required")


def load_plan(manifest_path, metadata_root, helper, parser):
    raw = read_bytes(manifest_path, 2 * 1024**2)
    require(sha(raw) == MANIFEST_SHA, "manifest bytes changed")
    plan = json.loads(raw)
    require(plan["schema"] == "evr0035.proposed-vost-manifest.v2" and plan["allocated"] is False,
            "wrong proposal")
    require(set(plan["metadata_sources"]) == set(parser.PINS), "metadata binding set")
    blobs = {}
    for name, desc in plan["metadata_sources"].items():
        data = read_bytes(metadata_root / relative(name), desc["bytes"])
        require(len(data) == desc["bytes"] and sha(data) == desc["sha256"] == parser.PINS[name],
                "cached metadata mismatch: " + name)
        blobs[Path(name).name] = data
    central = blobs["vost-central.bin"]
    archive = plan["archive"]
    inventory = json.loads(blobs["vost-members.json"])
    require(len(central) == archive["central_bytes"] == 21042989
            and sha(central) == archive["central_sha256"]
            and archive["central_start"] == 53991061837 and archive["bytes"] == 54012104924
            and archive["url"] == inventory["url"] and archive["etag"] == inventory["etag"],
            "archive identity mismatch")
    entries = {e["filename"]: e for e in inventory["members"]}
    offsets = sorted(e["local_header_offset"] for e in inventory["members"])
    require(len(entries) == len(set(offsets)) == 153136, "duplicate index member/offset")
    following = dict(zip(offsets, offsets[1:] + [archive["central_start"]]))
    train = blobs["vost-train-txt.txt"].decode().splitlines()
    ranked = sorted(train, key=lambda n: (sha(parser.SALT + n.encode()), n))
    require(tuple(ranked[:4]) == tuple(n for n, _ in parser.EXPECTED), "selection mismatch")
    require([(c["sequence"], c["frames"]) for c in plan["clips"]] == list(parser.EXPECTED), "clip count")
    wanted = {}
    for clip in plan["clips"]:
        name, stems = clip["sequence"], clip["frame_stems"]
        actual = sorted(PurePosixPath(n).stem for n in entries
                        if n.startswith("VOST/JPEGImages/" + name + "/") and n.endswith(".jpg"))
        masks = sorted(PurePosixPath(n).stem for n in entries
                       if n.startswith("VOST/Annotations/" + name + "/") and n.endswith(".png"))
        require(stems == actual == masks and len(stems) == clip["frames"], "complete paired stems")
        for frame, stem in enumerate(stems):
            wanted[f"VOST/JPEGImages/{name}/{stem}.jpg"] = ("rgb", name, frame, stem)
            wanted[f"VOST/Annotations/{name}/{stem}.png"] = ("initial" if frame == 0 else "metric", name, frame, stem)
    selected = {}
    for row in plan["members"]:
        name = row["filename"]
        relative(name)
        require(name not in selected and name in wanted and name in entries, "duplicate/unselected member")
        require(tuple(row[k] for k in ("phase", "sequence", "frame_index", "stem")) == wanted[name], "phase changed")
        entry = entries[name]
        require(all(row[k] == v for k, v in entry.items()), "inventory/manifest differs")
        require(all(row[k] == v for k, v in parser.central_entry(central, entry).items()), "central fields differ")
        f = struct.unpack_from("<4s6H3I5H2I", central, entry["central_offset_relative"])
        require(f[2:5] == (45, 8, 8) and (f[1] >> 8 != 3 or stat.S_IFMT(f[15] >> 16) in (0, stat.S_IFREG)),
                "unsupported central member type")
        require(0 < entry["compressed_bytes"] <= MAX_MEMBER and 0 < entry["uncompressed_bytes"] <= MAX_MEMBER,
                "member cap")
        require(row["next_local_header_or_central_offset"] == following[entry["local_header_offset"]],
                "physical neighbor differs")
        selected[name] = dict(row, version_needed=45, compression_method=8, mod_time=f[5], mod_date=f[6])
    require(set(selected) == set(wanted) and len(selected) == 892, "incomplete selected inventory")
    for phase, count in PHASE_COUNTS.items():
        rows = [e for e in selected.values() if e["phase"] == phase]
        require(len(rows) == count and sum(e["response_body_bytes_cap"] + 3 for e in rows) <= PHASE_CAPS[phase]
                and sum(e["uncompressed_bytes"] for e in rows) <= MAX_OUTPUT, "phase limits")
    require(sum(PHASE_CAPS.values()) <= RESERVATION, "phase reservation sum")
    return plan, helper.Index(archive["url"], archive["etag"], archive["bytes"], archive["central_start"],
                             selected, {"manifest": sha(raw), "central": sha(central)})


def extract_bounded(helper, reader, entry):
    offset, packed, unpacked = (entry[k] for k in ("local_header_offset", "compressed_bytes", "uncompressed_bytes"))
    require(0 < packed <= MAX_MEMBER and 0 < unpacked <= MAX_MEMBER, "member cap")
    raw = reader.read(offset, offset + 29)
    f = struct.unpack("<4s5H3I2H", raw)
    sig, version, flags, method, mt, md, crc, csize, usize, nlen, xlen = f
    name = entry["filename"].encode("ascii")
    require((sig, version, flags, method, mt, md) ==
            (b"PK\x03\x04", entry["version_needed"], 8, 8, entry["mod_time"], entry["mod_date"]),
            "local identity mismatch")
    require(crc in (0, int(entry["crc32_hex"], 16)) and nlen == len(name) and xlen <= MAX_EXTRA,
            "local CRC/name/extra mismatch")
    require(csize in (0, packed, 0xFFFFFFFF) and usize in (0, unpacked, 0xFFFFFFFF), "local sizes contradict central")
    start = offset + 30 + nlen + xlen
    require(start + packed <= min(reader.index.directory_offset, entry["next_local_header_or_central_offset"]),
            "member overlaps next header or central")
    names = reader.read(offset + 30, start - 1)
    require(names[:nlen] == name, "local filename differs")
    extras = helper.extra_fields(names[nlen:])
    sentinels = [(value, expected) for value, expected in ((usize, unpacked), (csize, packed))
                 if value == 0xFFFFFFFF]
    if sentinels:
        require(len(extras.get(1, b"")) == 8 * len(sentinels), "missing/ambiguous ZIP64 size extra")
        for i, (_, expected) in enumerate(sentinels):
            require(struct.unpack_from("<Q", extras[1], 8 * i)[0] in (0, expected),
                    "ZIP64 size contradicts central")
    else:
        require(1 not in extras, "unneeded local ZIP64 size extra")
    compressed = reader.read(start, start + packed - 1)
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(compressed, unpacked + 1)
    require(len(data) == unpacked and decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
            "deflate size/termination mismatch")
    require(zlib.crc32(data) & 0xFFFFFFFF == int(entry["crc32_hex"], 16), "central CRC mismatch")
    return data, {"sha256": sha(data), "compressed_sha256": sha(compressed),
                  "local_header_sha256": sha(raw + names), "data_offset": start, "local_extra_bytes": xlen,
                  "data_descriptor_read": False, "data_descriptor_integrity_claimed": False}


def closed_metric(*args):
    raise ValueError("metric phase closed: explicit frozen all-four clean prediction proof required")


def verify_initial(directory, index, commit, bindings):
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "completed"
            and receipt.get("phase") == "initial" and receipt.get("expected_commit") == commit
            and receipt.get("source_sha256") == bindings and receipt.get("index_sha256") == index.fingerprints,
            "initial phase not completely bound")
    expected = {e["filename"]: e for e in index.entries.values() if e["phase"] == "initial"}
    rows = receipt["members"]
    require(len(rows) == len(expected) == 4 and {r["filename"] for r in rows} == set(expected), "initial coverage")
    for row in rows:
        require(all(row[k] == v for k, v in expected[row["filename"]].items()), "initial member metadata changed")
        data = read_bytes(directory / relative(row["saved_relative"]), MAX_MEMBER)
        require(len(data) == row["bytes"] == row["uncompressed_bytes"] and sha(data) == row["sha256"],
                "initial bytes changed")
    return {"receipt_sha256": sha(read_bytes(directory / "receipt.json")), "members": 4}


def verify_inspection(path, outer_path, initial_dir, initial, root, bindings, admission):
    path, outer_path = Path(path).absolute(), Path(outer_path).absolute()
    fingerprints = {}
    desc = {"path": str(path), "sha256": sha(read_bytes(path))}
    inspected = admission.bound_json(desc, fingerprints)
    outer_desc = {"path": str(outer_path), "sha256": sha(read_bytes(outer_path))}
    admission.clean_outer(outer_desc, desc["sha256"], fingerprints)
    checks = ("native_object_frames_at_most_2000", "planning_memory_at_most_7GiB",
              "prediction_output_at_most_20GiB", "free_disk_covers_twice_output_plus_5GiB")
    require(inspected.get("schema") == "evr0035.initial-inspection.v1"
            and inspected.get("status") == "completed" and inspected.get("eligible") is True
            and inspected.get("initial_only") is True and inspected.get("noninitial_labels_read") == 0
            and inspected.get("model_calls") == 0
            and set(inspected.get("checks", {})) == set(checks)
            and all(inspected["checks"][k] is True for k in checks), "eligible complete initial inspection required")
    initial_path = (Path(initial_dir) / "receipt.json").resolve()
    require(any(Path(p).resolve() == initial_path and digest == initial["receipt_sha256"]
                for p, digest in inspected["fingerprints"].items()), "inspection used different initial receipt")
    require(inspected["source_sha256"], "inspection source closure missing")
    admission_hashes = []
    for name, digest in inspected["source_sha256"].items():
        source = Path(name) if Path(name).is_absolute() else root / relative(name)
        require(sha(read_bytes(source)) == digest, "inspection source changed")
        if source.name == "ADMISSION.md":
            admission_hashes.append(digest)
    require(admission_hashes == [bindings[ADMISSION]], "inspection ADMISSION differs from Git binding")
    for name, digest in inspected["fingerprints"].items():
        require(sha(read_bytes(name)) == digest, "inspection input changed")
    cap = inspected["prediction_output_cap_bytes"]
    require(type(cap) is int and 0 < cap <= 20 * 1024**3, "inspection prediction cap")
    return dict(initial, inspection_sha256=desc["sha256"], inspection_outer_sha256=outer_desc["sha256"],
                prediction_output_cap_bytes=cap)


def acquire_phase(helper, plan, index, phase, output, *, commit, bindings, postcheck,
                  initial_admission=None, proof_path=None, gt_verifier=None, opener=None):
    require(phase in PHASE_CAPS, "unknown phase")
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "fresh phase output required")
    output = output.resolve()
    output.mkdir(parents=True)
    began = time.monotonic()
    record = dict(schema=SCHEMA, status="started", phase=phase, expected_commit=commit, source_sha256=bindings,
                  index_sha256=index.fingerprints, archive=plan["archive"], initial_admission=initial_admission,
                  members=[], requests=[], response_body_bytes=0, response_body_bytes_exact=True,
                  requested_body_upper_bound=0, output_bytes=0, retry_count=0, hidden_retries=False,
                  full_archive_fallback=False, media_decoded=False, gt_admission=None,
                  phase_body_cap_bytes=PHASE_CAPS[phase], prospective_reservation_bytes=RESERVATION)
    def save():
        record["wall_seconds"] = time.monotonic() - began
        temporary = output / "receipt.pending"
        with temporary.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(output / "receipt.json")
    save()
    try:
        if phase != "initial":
            require(initial_admission is not None, "all four initial masks required first")
        if phase == "metric":
            verifier = gt_verifier or closed_metric
            record["gt_admission"] = verifier(plan, index.fingerprints["manifest"], proof_path)
            require(isinstance(record["gt_admission"], dict) and record["gt_admission"].get("eligible") is True,
                    "all-four prediction proof rejected")
            save()
        rows = [e for e in index.entries.values() if e["phase"] == phase]
        require(len(rows) == PHASE_COUNTS[phase], "phase coverage")
        limits = helper.Limits(PHASE_CAPS[phase], MAX_OUTPUT, MAX_MEMBER, 3 * len(rows), 1800, 30)
        limits.validate()
        record["limits"] = vars(limits)
        reader = helper.RangeReader(index, limits, record, save, opener)
        for entry in rows:
            record["current_member"] = entry["filename"]
            save()
            data, details = extract_bounded(helper, reader, entry)
            require(record["output_bytes"] + len(data) <= MAX_OUTPUT
                    and time.monotonic() - began <= 1800, "output/time cap")
            target = output / "members" / relative(entry["filename"])
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target
            while parent != output:
                require(not parent.is_symlink(), "symlink output")
                parent = parent.parent
            with target.open("xb") as stream:
                stream.write(data)
            record["members"].append(dict(entry, **details, bytes=len(data),
                                           saved_relative=target.relative_to(output).as_posix()))
            record["output_bytes"] += len(data)
            save()
        postcheck()
        if phase == "metric":
            require(verifier(plan, index.fingerprints["manifest"], proof_path) == record["gt_admission"],
                    "prediction proof changed")
        require(record["requested_body_upper_bound"] <= PHASE_CAPS[phase]
                and time.monotonic() - began <= 1800, "final byte/time cap")
        record.update(status="completed", current_member=None)
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
    return record


def run(args, *, gt_verifier=None):
    root, metadata = Path(args.repo_root).resolve(), Path(args.metadata_root).resolve()
    output = Path(args.output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    output = output.resolve()
    require(output != root and root not in output.parents, "payload output outside repository required")
    bindings = source_bindings(root, args.expected_commit, args.manifest_relative, args.admission_relative)
    validate_admission(read_json(root / relative(args.admission_relative)))
    helper = load_module("_evr0035_range", root / HELPER)
    parser = load_module("_evr0035_central", root / PARSER)
    plan, index = load_plan(root / args.manifest_relative, metadata, helper, parser)
    admission = load_module("_evr0035_admission", root / ADMIT) if args.phase != "initial" else None
    if args.phase == "metric" and gt_verifier is None:
        require(args.prediction_proof is not None, "prediction-proof required")
        def gt_verifier(_plan, manifest_digest, proof_path):
            before = sha(read_bytes(proof_path))
            context = admission.verify_prediction_proof(proof_path, manifest_digest, check_outputs=True)
            require(isinstance(context, dict) and context.get("manifest_sha256") == manifest_digest
                    and before == sha(read_bytes(proof_path)), "prediction proof changed")
            proof = context["proof"]
            return dict(eligible=context["eligible"], manifest_sha256=manifest_digest,
                        prediction_proof_sha256=before, prediction_receipt_sha256=proof["predict_receipt"]["sha256"],
                        prediction_seal_sha256=proof["predict_seal"]["sha256"],
                        output_count=len(context["prediction_receipt"]["outputs"]),
                        object_frames=context["object_frames"], outputs_verified=context["outputs_verified"],
                        binding_sha256=sha(json.dumps(context["fingerprints"], sort_keys=True).encode()))
    initial = None
    if args.phase != "initial":
        require(args.initial_dir is not None and args.initial_inspection is not None
                and args.initial_inspection_outer is not None, "initial receipt/inspection/outer required")
        initial = verify_inspection(
            args.initial_inspection, args.initial_inspection_outer, args.initial_dir,
            verify_initial(args.initial_dir, index, args.expected_commit, bindings), root, bindings, admission)
    def postcheck():
        require(all(sha(read_bytes(root / name)) == digest for name, digest in bindings.items()), "source changed")
        _, current = load_plan(root / args.manifest_relative, metadata, helper, parser)
        require(current.fingerprints == index.fingerprints, "metadata changed")
        if initial is not None:
            refreshed = verify_inspection(
                args.initial_inspection, args.initial_inspection_outer, args.initial_dir,
                verify_initial(args.initial_dir, index, args.expected_commit, bindings), root, bindings, admission)
            require(refreshed == initial, "initial receipt/inspection changed")
    return acquire_phase(helper, plan, index, args.phase, output, commit=args.expected_commit, bindings=bindings,
                         postcheck=postcheck, initial_admission=initial,
                         proof_path=args.prediction_proof, gt_verifier=gt_verifier)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "metadata-root", "manifest-relative", "admission-relative", "output-dir"):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--phase", choices=tuple(PHASE_CAPS), required=True)
    cli.add_argument("--initial-dir")
    cli.add_argument("--initial-inspection")
    cli.add_argument("--initial-inspection-outer")
    cli.add_argument("--prediction-proof")
    args = cli.parse_args()
    outcome = run(args)
    print(json.dumps({k: outcome[k] for k in ("status", "phase", "response_body_bytes", "wall_seconds")}))
