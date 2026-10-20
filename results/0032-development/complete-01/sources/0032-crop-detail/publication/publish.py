"""Explicit-list 0032 text export; no model, array, metric or scientific imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


SUPERVISOR = "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8"
MONITOR = "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
TEXT_SUFFIXES = {".json", ".py", ".md", ".txt", ".log"}
SENSITIVE = re.compile(
    r"/Use" r"rs/|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|"
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})|"
    r"(?i:\bBearer\s+[A-Za-z0-9._~+/-]{20,})|"
    r"""(?i:["'](?:password|api_key|access_token|client_secret)["']\s*[:=]\s*["'][^"'\s]{8,}["'])"""
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def run(spec_path):
    here = Path(__file__).resolve()
    directory = here.parents[1]
    workspace = directory.parents[1]
    spec_path = spec_path.resolve()
    spec_raw = spec_path.read_bytes()
    spec = decode(spec_raw)
    output = Path(spec["output"]).resolve()
    require(output.parent == directory and not output.exists(), "fresh private sibling output required")
    raw_files, destinations, duplicates = {}, {}, []
    for item in spec["files"]:
        path = Path(item["path"]).resolve()
        path.relative_to(workspace / "work")
        require(path.suffix in TEXT_SUFFIXES and "/installation/sources/" not in str(path),
                "only explicit local text; upstream source is referenced, not copied")
        relative = Path(item["published_path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.suffix in TEXT_SUFFIXES and str(relative) != "manifest.json", "unsafe public path")
        raw = path.read_bytes()
        raw.decode("utf-8")
        require(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "explicit input binding changed")
        if path in raw_files:
            require(raw_files[path] == raw, "duplicate origin changed")
            duplicates.append(dict(path=str(path), omitted_duplicate_destination=relative.as_posix()))
            continue
        require(relative.as_posix() not in destinations.values(), "duplicate destination")
        raw_files[path], destinations[path] = raw, relative.as_posix()

    def raw(path):
        path = Path(path).resolve()
        require(path in raw_files, "guarded artifact absent from explicit file list: " + str(path))
        return raw_files[path]

    def load(path):
        return decode(raw(path))

    completed = {}
    require(len(spec["operations"]) == 2, "both prediction and evaluation operations required")
    for operation in spec["operations"]:
        kind = operation["kind"]
        require(kind in ("prediction", "evaluation") and kind not in completed, "unique operation kinds")
        receipt, outer, worker, outer_spec = (load(operation[k]) for k in
            ("receipt", "outer", "worker_spec", "operation_spec"))
        require(receipt["status"] == outer["status"] == "completed" and outer["exit_code"] == 0
                and outer["termination"] is None and outer["remaining_descendant_pids"] == []
                and outer["cleanup_errors"] == [] and outer["signal_attempts"] == []
                and outer["triggering_snapshot"] is None and outer["post_exit_snapshot"]["rows"] == []
                and len(outer["cleanup_snapshots"]) == 1
                and outer["cleanup_snapshots"][0]["label"] == "final"
                and outer["cleanup_snapshots"][0]["rows"] == [], "unclean operation")
        require(outer["schema"] == "evr0029.exit-checked-operation.v1"
                and outer["supervisor_sha256"] == SUPERVISOR and outer["monitor_sha256"] == MONITOR
                and outer["expected_commit"] == outer_spec["commit"]
                and outer["spec_sha256"] == sha(raw(operation["operation_spec"]))
                and outer["output_sha256"] == sha(raw(operation["receipt"]))
                and receipt["spec_sha256"] == sha(raw(operation["worker_spec"]))
                and receipt["source_sha256"] == worker["source_sha256"], "operation source/hash binding")
        require(Path(outer_spec["result_file"]).resolve() == Path(operation["receipt"]).resolve()
                and Path(outer_spec["operation_output"]).resolve() == Path(operation["outer"]).resolve().parent
                and Path(worker["output"]).resolve() == Path(operation["receipt"]).resolve().parent
                and Path(outer_spec["command"][-1]).resolve() == Path(operation["worker_spec"]).resolve(),
                "operation path binding")
        normalized = [s.replace(outer_spec["repo"], "$REPOSITORY")
                      .replace(str(workspace / "work"), "$PRIVATE") for s in outer_spec["command"]]
        require(outer["command"] == normalized, "outer command mismatch")
        require(0 < outer["hard_timeout_seconds"] == outer_spec["timeout"] <= 1800
                and outer["group_rss_limit_kib"] == 8388608
                and math.isfinite(outer["wall_seconds"]) and 0 <= outer["wall_seconds"] <= outer_spec["timeout"]
                and 0 <= outer["peak_group_rss_kib"] <= 8388608
                and outer["thread_environment"] == {k: "4" for k in THREADS}
                and all(outer_spec["environment"][k] == "4" for k in THREADS), "operation resource limits")
        require(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "both original logs required")
        for name, digest in outer["log_sha256"].items():
            require(sha(raw(Path(operation["outer"]).parent / name)) == digest, "operation log changed")
        for path, digest in worker["source_sha256"].items():
            if Path(path).resolve() in raw_files:
                require(sha(raw(path)) == digest, "published source differs from executed source")
        completed[kind] = (operation, receipt, outer)

    prediction, evaluation = completed["prediction"][1], completed["evaluation"][1]
    seal, report = load(spec["prediction_seal"]), load(spec["evaluation_report"])
    require(prediction["schema"] == "evr0032.crop-detail.v1" and prediction["completed_frames"] == 241
            and prediction["native_parity_checks"] == 241 and len(prediction["rows"]) == 6
            and len(prediction["outputs"]) == 241 and prediction["current_label_decodes"] == 0
            and prediction["offline_attempts"] == []
            and prediction["model_initial_hash"] == prediction["model_final_hash"]
            and seal["schema"] == "evr0032.crop-detail-seal.v1" and seal["complete"] is True
            and seal["frame_count"] == 241
            and seal["receipt_sha256"] == sha(raw(completed["prediction"][0]["receipt"]))
            and seal["source_sha256"] == prediction["source_sha256"]
            and seal["spec_sha256"] == prediction["spec_sha256"]
            and seal["outputs"] == prediction["outputs"] == [p for r in prediction["rows"] for p in r["predictions"]]
            and seal["diagnostics"] == prediction["diagnostics"], "complete prediction seal required")
    require(evaluation["schema"] == "evr0032.crop-detail-evaluation.v1"
            and evaluation["model_calls"] == 0 and evaluation["predictions_sealed_before_gt"] is True
            and evaluation["prediction_seal_sha256"] == sha(raw(spec["prediction_seal"]))
            and evaluation["evaluation_sha256"] == sha(raw(spec["evaluation_report"]))
            and evaluation["bindings"]["refinement_receipt"]["sha256"] == seal["receipt_sha256"]
            and evaluation["train_events"] == report["train"]["event_count"] == 96
            and evaluation["train_unique_frames"] == report["train"]["unique_future_frames"] == 87
            and evaluation["dev_raw_frames"] == report["dev"]["raw_frames"] == 154
            and evaluation["dev_interior_frames"] == report["dev"]["included_frames"] == 150
            and evaluation["continuation_passed"] == report["continuation"]["passed"], "complete metric report required")

    def sanitize(text):
        text = text.replace(str(workspace), "${WORKSPACE}")
        require("\0" not in text and not SENSITIVE.search(text), "private path or credential-like text")
        return text

    prepared, entries = {}, []
    for path, original in raw_files.items():
        published = sanitize(original.decode("utf-8")).encode("utf-8")
        if path.suffix == ".json":
            # No reserialization: exact original formatting, keys, rows and values
            # survive except declared literal substitutions inside string values.
            decode(original)
            decode(published)
        relative = destinations[path]
        prepared[relative] = published
        entries.append(dict(path=relative, sha256=sha(published), bytes=len(published),
            origin=dict(path=sanitize(str(path)), sha256=sha(original), bytes=len(original)),
            transformation="byte-identical" if original == published else "literal workspace prefix -> ${WORKSPACE} only"))
    source_raw = here.read_bytes()
    manifest = dict(schema="evr0032.text-publication.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        entries=entries, publication_spec=decode(sanitize(spec_raw.decode("utf-8"))),
        publication_spec_sha256=sha(spec_raw), publication_spec_bytes=len(spec_raw),
        publisher=dict(path=sanitize(str(here)), sha256=sha(source_raw), bytes=len(source_raw)),
        deduplicated_inputs=decode(sanitize(json.dumps(duplicates))), all_json_fields_rows_preserved=True,
        no_scientific_recalculation=True, no_binary_payload_reads=True,
        upstream_source_scope="Hashes/pin references retained in executed specs; upstream sources not copied.",
        validation_scope="Completed receipts, logs, seal and report bindings; no independent metric or payload audit.")
    manifest_raw = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    sanitize(manifest_raw.decode())
    require(spec_path.read_bytes() == spec_raw and here.read_bytes() == source_raw
            and all(p.read_bytes() == b for p, b in raw_files.items()), "input changed during export")
    output.mkdir()
    for relative, data in {**prepared, "manifest.json": manifest_raw}.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        require(path.read_bytes() == data, "published-byte verification failed")
    print(json.dumps(dict(files=len(prepared)+1, bytes=sum(map(len, prepared.values()))+len(manifest_raw),
                          manifest_sha256=sha(manifest_raw), output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    run(parser.parse_args().spec)
