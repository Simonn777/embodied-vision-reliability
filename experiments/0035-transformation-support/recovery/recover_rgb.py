"""Complete RGB materialization from verified failed-attempt files plus missing ranges.

No I/O on import. No media decoding. Old failures remain failures. The compatible
acquisition schema is explicitly extended; its new clean status describes this
whole-population verification/copy/fetch operation, not either parent operation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

BASE = "experiments/0035-transformation-support/"
SELF = BASE + "recovery/recover_rgb.py"
DECLARATION = BASE + "recovery/RECOVERY_ADMISSION_02.md"
ALLOCATION = BASE + "recovery/recovery-allocation-02.json"
OLD_COMMIT = "784fc1a22d068dd7ce155129ac60d09d723f3f4d"
ACQUIRE_SHA = "a69768de68ac61585df7ceb077c48830a72c11dc88685472b2c0afd8f6b5341f"
PARENT_SHAS = (
    "6cc84b6c40a1d0bc4255478e13516b66b6dc314c07a33fc165268f472c4d5a9d",
    "3968bcc6cb1c34786b0a1d9dae2659ebb26dd18d21a88ec5bbba127f7342ef5a",
)
MANIFEST_SHA = "141371225d59c0b1ef483ee3b7565aace037033df072eabb2839e8d4f19bfb07"
EXPECTED = dict(schema="evr0035.rgb-recovery-allocation.v1", authorized=True,
    manifest_sha256=MANIFEST_SHA, reservation_bytes=20000000, body_bytes=20000000,
    output_bytes=209715200, member_bytes=8388608, local_extra_cap=4096,
    requests=114, request_seconds=30, wall_seconds=1800, retries=0,
    expected_total=446, expected_reused=408, expected_fetched=38,
    expected_parent_members=[408, 308], expected_overlap=308,
    prior_balance=933590729, remaining_balance=913590729, refunds_taken=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit=64 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "bounded regular input required")
    raw = path.read_bytes()
    require(len(raw) <= limit, "input grew beyond bound")
    return raw


def load(path):
    return json.loads(read(path), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def bound(desc, fingerprints):
    path = Path(desc["path"])
    require(path.is_absolute(), "absolute descriptor required")
    raw = read(path)
    require(sha(raw) == desc["sha256"], "bound input changed: " + str(path))
    fingerprints[str(path)] = desc["sha256"]
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def owned(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and all(p not in (".", "..") for p in name.split("/")),
            "safe relative path required")
    result = Path(root) / path
    require(result.resolve().is_relative_to(Path(root).resolve()), "path escaped root")
    current = result
    while current != Path(root):
        require(not current.is_symlink(), "symlink in artifact path")
        current = current.parent
    return result


def raw_member(root, row):
    raw = read(owned(root, row["saved_relative"]), EXPECTED["member_bytes"])
    require(len(raw) == row["bytes"] == row["uncompressed_bytes"]
            and sha(raw) == row["sha256"]
            and zlib.crc32(raw) & 0xffffffff == int(row["crc32_hex"], 16),
            "member SHA/CRC/size changed")
    return raw


def check_allocation(value):
    require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in EXPECTED.items()),
            "exact prospective recovery allocation required")


def source_binding(spec):
    root = Path(spec["repo_root"]).resolve()
    require(spec["expected_commit"] == OLD_COMMIT
            and re.fullmatch("[a-f0-9]{40}", spec["recovery_source_commit"]) is not None,
            "exact old and new source commits required")
    # Verify the original executable before importing any of its helpers.
    raw = read(root / (BASE + "acquire.py"))
    require(sha(raw) == ACQUIRE_SHA, "unchanged acquisition implementation required")
    acq = module("_rgb_recovery_original_acquire", root / (BASE + "acquire.py"))
    original = acq.source_bindings(root, OLD_COMMIT, BASE + "proposed-manifest-v2.json",
                                   BASE + "acquisition-admission.json")
    sources = dict(original)
    for name in (SELF, DECLARATION, ALLOCATION):
        raw = read(owned(root, name))
        frozen = subprocess.check_output(["git", "show", spec["recovery_source_commit"] + ":" + name],
                                         cwd=root, timeout=15)
        require(raw == frozen, "new recovery source differs from freeze")
        sources[name] = sha(raw)
    require(read(Path(__file__)) == read(root / SELF), "executing recovery source differs")
    return root, acq, original, sources


def check_parent(desc, index, original_sources, spec, admit, fingerprints):
    """Accept only the byte-bound failed original producer, never mark it clean."""
    record = bound(desc["receipt"], fingerprints)
    outer = bound(desc["outer"], fingerprints)
    operation = bound(desc["operation"], fingerprints)
    require(record["schema"] == "evr0035.vost-acquisition.v1" and record["status"] == "failed"
            and record["phase"] == "rgb" and record["expected_commit"] == OLD_COMMIT
            and record["source_sha256"] == original_sources
            and record["index_sha256"] == index.fingerprints
            and record["media_decoded"] is False and record["gt_admission"] is None
            and record["retry_count"] == 0 and record["hidden_retries"] is False
            and record["full_archive_fallback"] is False and record["error_type"] == "URLError",
            "original failed RGB producer required")
    require(outer["schema"] == "evr0029.exit-checked-operation.v1" and outer["status"] == "failed"
            and outer["exit_code"] == 1 and outer["termination"] is None
            and not outer["remaining_descendant_pids"] and not outer["cleanup_errors"]
            and outer["supervisor_sha256"] == admit.SUPERVISOR_SHA256
            and outer["monitor_sha256"] == admit.MONITOR_SHA256
            and outer["thread_environment"] == {k: "4" for k in admit.THREAD_KEYS}
            and outer["hard_timeout_seconds"] == 1800 and outer["group_rss_limit_kib"] == 8388608
            and outer["expected_commit"] == OLD_COMMIT
            and math.isfinite(outer["wall_seconds"]) and 0 <= outer["wall_seconds"] < 1800
            and type(outer["peak_group_rss_kib"]) is int and 0 <= outer["peak_group_rss_kib"] <= 8388608
            and outer["output_sha256"] == desc["receipt"]["sha256"]
            and outer["spec_sha256"] == desc["operation"]["sha256"]
            and set(outer["log_sha256"]) == {"stdout.log", "stderr.log"},
            "failed outer provenance/resource envelope")
    directory = Path(desc["outer"]["path"]).parent
    for name, digest in outer["log_sha256"].items():
        p = owned(directory, name)
        require(sha(read(p)) == digest, "failed outer log changed")
        fingerprints[str(p)] = digest
    command = operation["command"]
    expected_args = ["-B", "-S", "-W", "error", str(Path(spec["repo_root"]) / (BASE + "acquire.py")),
        "--repo-root", spec["repo_root"], "--expected-commit", OLD_COMMIT,
        "--metadata-root", spec["metadata_root"], "--manifest-relative", BASE + "proposed-manifest-v2.json",
        "--admission-relative", BASE + "acquisition-admission.json",
        "--output-dir", str(Path(desc["receipt"]["path"]).parent), "--phase", "rgb",
        "--initial-dir", spec["initial_dir"], "--initial-inspection", desc["initial_inspection"]["path"],
        "--initial-inspection-outer", desc["initial_inspection_outer"]["path"]]
    require(command[1:] == expected_args and operation["repo"] == spec["repo_root"]
            and operation["commit"] == OLD_COMMIT and operation["timeout"] == 1800
            and operation["result_file"] == desc["receipt"]["path"]
            and operation["operation_output"] == str(directory),
            "parent operation invocation differs")
    normalized = [s.replace(command[0], "$PYTHON").replace(spec["repo_root"], "$REPOSITORY")
                  .replace(str(Path(spec["repo_root"]).parent), "$PRIVATE") for s in command]
    require(normalized == outer["command"], "parent observed command differs")
    inspection = bound(desc["initial_inspection"], fingerprints)
    admit.clean_outer(desc["initial_inspection_outer"], desc["initial_inspection"]["sha256"], fingerprints)
    require(record["initial_admission"]["receipt_sha256"] == spec["initial_receipt"]["sha256"]
            and record["initial_admission"]["inspection_sha256"] == desc["initial_inspection"]["sha256"]
            and record["initial_admission"]["inspection_outer_sha256"] == desc["initial_inspection_outer"]["sha256"]
            and inspection["status"] == "completed" and inspection["eligible"] is True,
            "parent initial provenance")
    entries = [e for e in index.entries.values() if e["phase"] == "rgb"]
    rows = record["members"]
    require(0 < len(rows) < len(entries) and [r["filename"] for r in rows]
            == [r["filename"] for r in entries[:len(rows)]]
            and record["current_member"] == entries[len(rows)]["filename"], "parent exact completed prefix")
    for row, expected in zip(rows, entries):
        require(all(row[k] == v for k, v in expected.items()), "parent member metadata differs")
        raw_member(Path(desc["receipt"]["path"]).parent, row)
    requests = record["requests"]
    require(len(requests) <= 3 * len(entries)
            and sum(r["body_bytes"] for r in requests) == record["response_body_bytes"]
            and sum(r["expected_bytes"] + 1 for r in requests) == record["requested_body_upper_bound"]
            and record["response_body_bytes_exact"] is True, "parent request accounting")
    for i, row in enumerate(rows):
        triplet = requests[3*i:3*i+3]
        start, data = row["local_header_offset"], row["data_offset"]
        ranges = [(start, start+29), (start+30, data-1), (data, data+row["compressed_bytes"]-1)]
        require(len(triplet) == 3, "parent complete member ranges")
        for request, (first, last) in zip(triplet, ranges):
            require(request["start"] == first and request["end"] == last
                    and request["expected_bytes"] == request["body_bytes"] == last-first+1
                    and request["status"] == 206 and request["etag"] == index.etag
                    and request["content_range"] == f"bytes {first}-{last}/{index.archive_bytes}"
                    and request["content_length"] == str(last-first+1), "parent verified range provenance")
        require(triplet[2]["body_sha256"] == row["compressed_sha256"], "parent compressed SHA binding")
    return record


def select_reuse(parents):
    """Earliest declared parent wins only after every overlap agrees exactly."""
    chosen, overlap = {}, 0
    for parent_index, (desc, record) in enumerate(parents):
        for row in record["members"]:
            name = row["filename"]
            if name in chosen:
                previous = chosen[name][2]
                require((row["bytes"], row["sha256"], row["crc32_hex"]) ==
                        (previous["bytes"], previous["sha256"], previous["crc32_hex"]),
                        "conflicting completed parent copies")
                overlap += 1
            else:
                chosen[name] = (parent_index, desc, row)
    return chosen, overlap


def materialize(acq, helper, plan, index, chosen, output, record, save, began, *, opener=None):
    entries = [e for e in index.entries.values() if e["phase"] == "rgb"]
    missing = [e for e in entries if e["filename"] not in chosen]
    require(len(entries) == EXPECTED["expected_total"] and len(chosen) == EXPECTED["expected_reused"]
            and len(missing) == EXPECTED["expected_fetched"]
            and sum(e["response_body_bytes_cap"] + 3 for e in missing) <= EXPECTED["body_bytes"],
            "exact recovery coverage/budget")
    limits = helper.Limits(EXPECTED["body_bytes"], EXPECTED["output_bytes"], EXPECTED["member_bytes"],
                           EXPECTED["requests"], 1800, 30)
    limits.validate()
    record["limits"] = vars(limits)
    reader = helper.RangeReader(index, limits, record, save, opener)
    reader.started = began  # Source/parent validation also belongs to this whole-operation deadline.
    for entry in entries:
        name = entry["filename"]
        require(time.monotonic() - began < 1800, "whole recovery wall cap")
        record["current_member"] = name
        save()
        if name in chosen:
            parent_index, desc, old = chosen[name]
            data = raw_member(Path(desc["receipt"]["path"]).parent, old)
            details = {k: old[k] for k in ("sha256", "compressed_sha256", "local_header_sha256",
                "data_offset", "local_extra_bytes", "local_version_needed",
                "data_descriptor_read", "data_descriptor_integrity_claimed")}
            origin = dict(kind="verified_reuse", parent_index=parent_index,
                          parent_receipt_sha256=desc["receipt"]["sha256"],
                          parent_saved_relative=old["saved_relative"], original_status="failed")
        else:
            start = len(record["requests"])
            data, details = acq.extract_bounded(helper, reader, entry)
            origin = dict(kind="new_range_fetch", request_indices=list(range(start, len(record["requests"]))))
        require(record["output_bytes"] + len(data) <= EXPECTED["output_bytes"], "complete output cap")
        target = owned(output, "members/" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        row = dict(entry, **details, bytes=len(data), saved_relative=target.relative_to(output).as_posix(),
                   recovery_origin=origin)
        raw_member(output, row)
        record["members"].append(row)
        record["output_bytes"] += len(data)
        record["reused_members" if name in chosen else "fetched_members"] += 1
        save()


def verify_recovery_receipt(record, index, output):
    """Additional extension check before proof construction; no outer bypass."""
    expected = [e for e in index.entries.values() if e["phase"] == "rgb"]
    require(record["schema"] == "evr0035.vost-acquisition.v1"
            and record["transport_mode"] == "verified_complete_recovery" and record["phase"] == "rgb"
            and record["status"] == "completed" and record["reused_members"] == 408
            and record["fetched_members"] == 38 and len(record["members"]) == 446
            and record["parent_members_verified"] == [408, 308] and record["overlap_verified"] == 308
            and record["retry_count"] == 0 and record["media_decoded"] is False
            and record["parent_rechecks_complete"] is True, "complete recovery extension required")
    require([r["filename"] for r in record["members"]] == [e["filename"] for e in expected],
            "complete ordered recovered inventory")
    fetched = reused = 0
    indices = []
    for row, entry in zip(record["members"], expected):
        require(all(row[k] == v for k, v in entry.items()), "recovered manifest differs")
        raw_member(output, row)
        origin = row["recovery_origin"]
        if origin["kind"] == "verified_reuse":
            reused += 1
            i = origin["parent_index"]
            require(i in (0, 1) and origin["original_status"] == "failed"
                    and origin["parent_receipt_sha256"] == record["parent_failures"][i]["receipt"]["sha256"],
                    "reuse origin differs")
        else:
            require(origin["kind"] == "new_range_fetch" and len(origin["request_indices"]) == 3,
                    "new fetch origin")
            fetched += 1
            indices.extend(origin["request_indices"])
    require((reused, fetched) == (408, 38) and indices == list(range(len(record["requests"])))
            and len(indices) == 114 and record["response_body_bytes_exact"] is True
            and sum(r["body_bytes"] for r in record["requests"]) == record["response_body_bytes"]
            and sum(r["expected_bytes"]+1 for r in record["requests"]) == record["requested_body_upper_bound"]
            and record["requested_body_upper_bound"] <= EXPECTED["body_bytes"]
            and sum(r["bytes"] for r in record["members"]) == record["output_bytes"]
            <= EXPECTED["output_bytes"], "complete recovery accounting")


def run(path, *, opener=None):
    began = time.monotonic()
    path = Path(path).resolve()
    spec_raw = read(path)
    spec = json.loads(spec_raw)
    require(spec["schema"] == "evr0035.rgb-recovery-spec.v1", "recovery spec schema")
    output = Path(spec["output"]).absolute()
    root = Path(spec["repo_root"]).resolve()
    require(not output.exists() and not output.is_symlink()
            and not output.resolve().is_relative_to(root), "fresh private recovery output")
    output.mkdir(parents=True)
    fingerprints = {str(path): sha(spec_raw)}
    record = dict(schema="evr0035.vost-acquisition.v1", status="started", phase="rgb",
        transport_mode="verified_complete_recovery", expected_commit=OLD_COMMIT,
        recovery_source_commit=spec["recovery_source_commit"], spec_sha256=sha(spec_raw),
        parent_failures=spec["parents"], parent_rechecks_complete=False,
        members=[], requests=[], response_body_bytes=0, response_body_bytes_exact=True,
        requested_body_upper_bound=0, output_bytes=0, reused_members=0, fetched_members=0,
        retry_count=0, hidden_retries=False, full_archive_fallback=False, media_decoded=False, gt_admission=None,
        phase_body_cap_bytes=EXPECTED["body_bytes"], prospective_reservation_bytes=EXPECTED["reservation_bytes"])

    def save():
        record["wall_seconds"] = time.monotonic() - began
        temporary = output / "receipt.pending"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "receipt.json")

    save()
    try:
        root, acq, original, sources = source_binding(spec)
        record["source_sha256"] = sources
        check_allocation(load(root / ALLOCATION))
        helper = module("_recovery_range", root / acq.HELPER)
        parser = module("_recovery_central", root / acq.PARSER)
        admit = module("_recovery_admit", root / acq.ADMIT)
        plan, index = acq.load_plan(root / (BASE + "proposed-manifest-v2.json"),
                                    Path(spec["metadata_root"]), helper, parser)
        record.update(index_sha256=index.fingerprints, archive=plan["archive"])
        require(spec["initial_receipt"]["path"] == str(Path(spec["initial_dir"]) / "receipt.json"),
                "initial directory/receipt identity")
        bound(spec["initial_receipt"], fingerprints)
        bound(spec["initial_inspection"], fingerprints)
        bound(spec["initial_inspection_outer"], fingerprints)
        def initial():
            return acq.verify_inspection(spec["initial_inspection"]["path"], spec["initial_inspection_outer"]["path"],
                spec["initial_dir"], acq.verify_initial(spec["initial_dir"], index, OLD_COMMIT, original),
                root, original, admit)
        record["initial_admission"] = initial()
        require([p["receipt"]["sha256"] for p in spec["parents"]] == list(PARENT_SHAS),
                "exact two failed parents required")
        parents = [(p, check_parent(p, index, original, spec, admit, fingerprints)) for p in spec["parents"]]
        require([len(r["members"]) for _, r in parents] == EXPECTED["expected_parent_members"],
                "parent coverage changed")
        chosen, overlap = select_reuse(parents)
        require(overlap == EXPECTED["expected_overlap"], "parent overlap changed")
        record.update(parent_members_verified=[len(r["members"]) for _, r in parents], overlap_verified=overlap,
            parent_costs=[dict(receipt_sha256=p["receipt"]["sha256"], status=r["status"],
                response_body_bytes=r["response_body_bytes"], requested_body_upper_bound=r["requested_body_upper_bound"],
                wall_seconds=r["wall_seconds"]) for p, r in parents])
        save()
        materialize(acq, helper, plan, index, chosen, output, record, save, began, opener=opener)
        # Every one of the 716 completed parent files is checked again, including duplicate copies.
        for desc, before in parents:
            require(check_parent(desc, index, original, spec, admit, fingerprints) == before, "parent changed")
        require(initial() == record["initial_admission"], "initial admission changed")
        require(all(sha(read(p)) == digest for p, digest in fingerprints.items()), "bound provenance changed")
        require(all(sha(read(root / p)) == digest for p, digest in sources.items()), "source changed")
        _, after = acq.load_plan(root / (BASE + "proposed-manifest-v2.json"),
                                  Path(spec["metadata_root"]), helper, parser)
        require(after.fingerprints == index.fingerprints, "metadata changed")
        record.update(parent_rechecks_complete=True, status="completed", current_member=None,
                      provenance_sha256=fingerprints)
        verify_recovery_receipt(record, index, output)
        require(time.monotonic() - began < 1800, "whole recovery wall cap")
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    return record


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--spec", required=True)
    result = run(cli.parse_args().spec)
    print(json.dumps({k: result[k] for k in ("status", "reused_members", "fetched_members",
                                           "response_body_bytes", "wall_seconds")}), flush=True)
