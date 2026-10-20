"""Private 0030 development-label transport. Standard library; never decode PNGs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.error


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 16*1024**2,
            "missing/unsafe/oversized JSON")
    return json.loads(p.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError("nonfinite JSON: " + x)))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_file(path, sha, size=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/unsafe input: " + str(path))
    require(size is None or path.stat().st_size == size, "input size changed")
    require(digest(path) == sha, "input hash changed: " + str(path))
    return path


def admit(config_path, config_sha, source_sha):
    checked_file(__file__, source_sha)
    checked_file(config_path, config_sha)
    config = read(config_path)
    require(config["schema"] == "evr0030.development-mask-transport.v1"
            and config["body_cap"] == 6000000 and config["request_cap"] == 1992
            and config["max_starts"] == 2, "wrong fixed resource contract")
    for row in config["bindings"]:
        checked_file(row["path"], row["sha256"], row["bytes"])
    inventory = read(config["inventory"])
    require(len(inventory) == 386 and len({r["filename"] for r in inventory}) == 386,
            "full 386 inventory required")
    require(sum(r["reused"] for r in inventory) == 54, "reuse count")
    by_name = {r["filename"]: r for r in inventory}
    for row in inventory:
        name = row["filename"]
        path = PurePosixPath(name)
        require(str(path) == name and not path.is_absolute()
                and all(x not in ("", ".", "..") for x in path.parts)
                and name == "DAVIS/Annotations/480p/%s/%s.png" %
                    (row["sequence"], row["frame_stem"]), "annotation allowlist")
        require(row["role"] in ("TRAIN", "DEV"), "role")
    reused = {}
    for chain in config["reuse_chains"]:
        receipt = read(chain["receipt"])
        outer = read(chain["outer"])
        require(receipt["status"] == "completed"
                and outer["status"] == "completed" and outer["exit_code"] == 0
                and outer["termination"] is None and not outer["cleanup_errors"]
                and not outer["remaining_descendant_pids"], "reuse chain failed")
        operation_path = chain.get("operation", chain["receipt"])
        require(outer["output_sha256"] == digest(operation_path), "reuse outer output")
        if "operation" in chain:
            operation = read(operation_path)
            require(operation["status"] == "completed" and operation["eligible"] is True
                    and operation["acquisition_receipt_sha256"] == digest(chain["receipt"]),
                    "reuse operation linkage")
        for filename, sha in outer["log_sha256"].items():
            checked_file(Path(chain["outer"]).parent / filename, sha)
        for member in receipt["members"]:
            name = member["filename"]
            require(name in by_name and by_name[name]["reused"] and name not in reused,
                    "unexpected/duplicate reused member")
            expected = by_name[name]
            require(all(member[k] == expected[k] for k in
                        ("crc32_hex", "compressed_bytes", "uncompressed_bytes",
                         "local_header_offset")), "reuse metadata mismatch")
            path = Path(chain["receipt"]).parent / member["saved_relative"]
            checked_file(path, member["sha256"], member["bytes"])
            reused[name] = dict(path=str(path.resolve()), sha256=member["sha256"],
                                bytes=member["bytes"], origin_receipt=chain["receipt"])
    require(set(reused) == {r["filename"] for r in inventory if r["reused"]},
            "incomplete reuse inventory")
    return config, inventory, reused


def previous_attempts(root, config_sha, source_sha):
    """Completed members may be reused from a failed transport, never its success status."""
    records, starts, complete, charge, requests, actual = [], {}, {}, 0, 0, 0
    for directory in sorted((root / "attempts").glob("*")):
        require(directory.is_dir() and not directory.is_symlink(), "unsafe attempt directory")
        path = directory / "receipt.json"
        record = read(path)
        require(record["config_sha256"] == config_sha and record["source_sha256"] == source_sha,
                "attempt source/config changed")
        require(record["status"] in ("completed", "failed"), "unfinished attempt needs review")
        records.append(dict(path=str(path), sha256=digest(path), status=record["status"]))
        charge += record["requested_body_upper_bound"]
        requests += len(record["requests"])
        actual += record["response_body_bytes"]
        for name in record["started_members"]:
            starts[name] = starts.get(name, 0) + 1
        for member in record["members"]:
            name = member["filename"]
            require(name not in complete, "duplicate committed download")
            checked_file(member["path"], member["sha256"], member["bytes"])
            complete[name] = {**member, "origin_receipt": str(path)}
    return records, starts, complete, charge, requests, actual


def retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429, 500, 502, 503, 504)
    return isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError)) or (
        isinstance(error, ValueError) and str(error) == "incomplete range response")


def run(config_path, config_sha, source_sha, sequence, output, *, opener=None):
    config, inventory, reused = admit(config_path, config_sha, source_sha)
    root = Path(config["root"])
    require(Path(output).parent == root / "attempts" and not Path(output).exists(),
            "fresh private attempt required")
    lock = root / "transport.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        old, starts, complete, charge, requests, _ = previous_attempts(root, config_sha, source_sha)
        require(sequence in config["sequences"], "unknown sequence")
        selected = [r for r in inventory if r["sequence"] == sequence
                    and not r["reused"] and r["filename"] not in complete]
        require(selected, "no pending members; no duplicate acquisition")
        for row in selected:
            require(starts.get(row["filename"], 0) < config["max_starts"], "member attempts exhausted")
        # Resuming is a new explicit invocation, never an internal transport retry.
        for prior in old:
            rec = read(prior["path"])
            if rec["status"] == "failed" and rec["sequence"] == sequence:
                require(rec.get("retryable") is True, "non-transient prior failure requires review")
        budget = sum(r["compressed_bytes"] + 30 + r["filename_bytes"] + 4096 + 3
                     for r in selected)
        require(charge + budget <= config["body_cap"]
                and requests + 3*len(selected) <= config["request_cap"],
                "lifetime transfer reservation exhausted")
        helper = load("_evr0030_range", config["range_helper"])
        extractor = load("_evr0030_extractor", config["extractor"])
        index = helper.Index(config["url"], config["etag"], config["archive_bytes"],
                             config["directory_offset"],
                             {r["filename"]: r for r in selected},
                             {"inventory": digest(config["inventory"]), "config": config_sha})
        output = Path(output)
        output.mkdir(parents=True)
        start = time.monotonic()
        record = dict(schema="evr0030.mask-acquisition.v1", status="started",
            sequence=sequence, config_sha256=config_sha, source_sha256=source_sha,
            prior_attempts=old, previous_charged_body_bytes=charge,
            previous_requests=requests, selected_names=[r["filename"] for r in selected],
            started_members=[], members=[], requests=[], response_body_bytes=0,
            response_body_bytes_exact=True, requested_body_upper_bound=0, output_bytes=0,
            retryable=False, media_decoded=False, hidden_retry=False, full_zip_fallback=False)
        event_number, previous_event = 0, None
        last_request, last_member = None, None

        def save():
            nonlocal event_number, previous_event, last_request, last_member
            record["wall_seconds"] = time.monotonic() - start
            raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
            request = record["requests"][-1] if record["requests"] else None
            member = record["members"][-1] if record["members"] else None
            event = dict(event=event_number, previous_event_sha256=previous_event,
                receipt_sha256=hashlib.sha256(raw).hexdigest(),
                status=record["status"], current_member=record.get("current_member"),
                response_body_bytes=record["response_body_bytes"],
                requested_body_upper_bound=record["requested_body_upper_bound"],
                request_count=len(record["requests"]), wall_seconds=record["wall_seconds"])
            if request != last_request:
                event["request"] = request
                last_request = json.loads(json.dumps(request))
            if member != last_member:
                event["committed_member"] = member
                last_member = json.loads(json.dumps(member))
            event_raw = json.dumps(event, sort_keys=True, allow_nan=False).encode()
            with (output / "events.jsonl").open("ab") as stream:
                stream.write(event_raw + b"\n"); stream.flush(); os.fsync(stream.fileno())
            previous_event = hashlib.sha256(event_raw).hexdigest()
            event_number += 1
            pending = output / "receipt.pending"
            with pending.open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            pending.replace(output / "receipt.json")

        save()
        try:
            limits = helper.Limits(budget, sum(r["uncompressed_bytes"] for r in selected),
                                   8*1024**2, 3*len(selected), 1790, 30)
            limits.validate()
            record["limits"] = vars(limits)
            reader = helper.RangeReader(index, limits, record, save, opener)
            for row in selected:
                record["current_member"] = row["filename"]
                record["started_members"].append(row["filename"]); save()
                data, metadata = extractor.extract_bounded(helper, reader, row)
                require(len(data) == row["uncompressed_bytes"]
                        and record["output_bytes"] + len(data) <= limits.output_bytes,
                        "exact output byte budget")
                destination = output / (row["frame_stem"] + ".png")
                with destination.open("xb") as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                record["members"].append(dict(filename=row["filename"], sequence=sequence,
                    frame=int(row["frame_stem"]), role=row["role"], path=str(destination),
                    bytes=len(data), **metadata))
                record["output_bytes"] += len(data)
                require(time.monotonic() - start < 1790, "transport wall limit")
                save()
            admit(config_path, config_sha, source_sha)
            require(record["requested_body_upper_bound"] <= budget
                    and charge + record["requested_body_upper_bound"] <= config["body_cap"],
                    "final transfer budget")
            record.update(status="completed", current_member=None)
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error),
                          retryable=retryable(error))
            raise
        finally:
            save()
        return record
    finally:
        lock.unlink()


def finalize(config_path, config_sha, source_sha, output):
    config, inventory, reused = admit(config_path, config_sha, source_sha)
    old, starts, downloaded, charge, requests, actual = previous_attempts(
        Path(config["root"]), config_sha, source_sha)
    require(not (Path(config["root"]) / "transport.lock").exists(), "transport active")
    expected_new = {r["filename"] for r in inventory if not r["reused"]}
    require(set(downloaded) == expected_new and all(n <= 2 for n in starts.values())
            and charge <= config["body_cap"] and requests <= config["request_cap"],
            "incomplete or over-budget development inventory")
    supervision = []
    for attempt in old:
        name = Path(attempt["path"]).parent.name
        outer_path = Path(config["root"]) / "supervision" / name / "receipt.json"
        outer = read(outer_path)
        require(outer["output_sha256"] == attempt["sha256"], "attempt outer hash mismatch")
        if attempt["status"] == "completed":
            require(outer["status"] == "completed" and outer["exit_code"] == 0
                    and outer["termination"] is None and not outer["cleanup_errors"]
                    and not outer["remaining_descendant_pids"], "completed attempt outer failed")
        for filename, sha in outer["log_sha256"].items():
            checked_file(outer_path.parent / filename, sha)
        supervision.append(dict(path=str(outer_path), sha256=digest(outer_path),
            status=outer["status"], wall_seconds=outer["wall_seconds"],
            peak_group_rss_kib=outer["peak_group_rss_kib"]))
    mapping = []
    for row in inventory:
        source = reused[row["filename"]] if row["reused"] else downloaded[row["filename"]]
        mapping.append(dict(sequence=row["sequence"], frame=int(row["frame_stem"]),
            filename=row["filename"], role=row["role"], reused=row["reused"],
            **{k: source[k] for k in ("path", "sha256", "bytes", "origin_receipt")}))
    record = dict(schema="evr0030.complete-development-masks.v1", status="completed",
        config_sha256=config_sha, source_sha256=source_sha, mappings=mapping,
        query_count=None, member_count=386, reused_count=54, downloaded_count=332,
        total_png_bytes=sum(m["bytes"] for m in mapping), prior_attempts=old,
        supervision=supervision,
        new_actual_response_body_bytes=actual, new_conservative_charge_bytes=charge,
        new_requests=requests, reservation_bytes=config["body_cap"],
        remaining_planning_balance=config["remaining_planning_balance"],
        separate_metric_source_debit_bytes=27585, previous_holds_released=False,
        media_decoded=False, final_test_access=False)
    with Path(output).open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "config-sha256", "source-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    common = (args.config, args.config_sha256, args.source_sha256)
    result = (finalize(*common, args.output) if args.finalize else
              run(*common, args.sequence, args.output))
    print(json.dumps({k: result[k] for k in ("schema", "status")}), flush=True)
