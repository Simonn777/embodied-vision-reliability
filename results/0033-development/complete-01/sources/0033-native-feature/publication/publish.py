"""Explicit-list 0033 text export; no model, array, metric or scientific imports."""
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
TEXT_SUFFIXES = {".json", ".jsonl", ".py", ".md", ".txt", ".log"}
ARMS = ("visual", "logit_only")
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


def decode_lines(raw):
    lines = raw.decode("utf-8").splitlines()
    require(lines and all(line.strip() for line in lines), "empty JSONL row")
    return [decode(line) for line in lines]


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
    require(len(spec["operations"]) == 3, "fit, prediction and evaluation operations required")
    for operation in spec["operations"]:
        kind = operation["kind"]
        require(kind in ("fit", "prediction", "evaluation") and kind not in completed, "unique operation kinds")
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

    fit, prediction, evaluation = (completed[k][1] for k in ("fit", "prediction", "evaluation"))
    for child, parent in ((prediction, "fit"), (evaluation, "fit"), (evaluation, "prediction")):
        for suffix, key in (("receipt", "receipt"), ("outer", "outer"),
                            ("operation", "operation_spec"), ("worker_spec", "worker_spec")):
            binding, original = child["bindings"][parent + "_" + suffix], completed[parent][0][key]
            require(Path(binding["path"]).resolve() == Path(original).resolve()
                    and binding["sha256"] == sha(raw(original)) and binding["bytes"] == len(raw(original)),
                    "cross-operation metadata binding")
    fit_root = Path(completed["fit"][0]["receipt"]).resolve().parent
    require(fit["schema"] == "evr0033.native-feature-fit.v1" and fit["stage"] == "complete"
            and fit["completed_train_frames"] == len(fit["train_frames"]) == 228
            and fit["completed_arms"] == 2 and set(fit["arms"]) == set(ARMS)
            and fit["attempted_frames"] == 5472 and fit["native_parity_checks"] == 228
            and fit["optimizer_steps"] == 24 * fit["active_frames"]
            and fit["active_frames"] + fit["no_op_frames"] == 228
            and fit["offline_attempts"] == [], "complete fixed two-arm fit required")

    def fit_text(key):
        descriptor = fit[key]
        relative = Path(descriptor["path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and descriptor in fit["outputs"], "fit text descriptor")
        data = raw(fit_root / relative)
        require(sha(data) == descriptor["sha256"] and len(data) == descriptor["bytes"], "fit text changed")
        return data

    normalization, orders = decode(fit_text("normalization")), decode(fit_text("orders"))
    training_log = decode_lines(fit_text("training_log"))
    require(normalization["source_frames"] == 228 and normalization["population_pixels"] == 228 * 65536
            and all(len(normalization[k]) == 32 for k in ("mean", "std", "raw_std"))
            and orders["seed"] == 0 and len(orders["orders"]) == 12
            and all(sorted(order) == list(range(228)) for order in orders["orders"])
            and len(orders["identities"]) == 228 and len(training_log) == 5472, "complete fit text")
    terminal = {}
    for arm in ARMS:
        result = fit["arms"][arm]
        terminal[arm] = descriptor = result["terminal_checkpoint"]
        require(result["status"] == "completed" and result["epochs"] == 12
                and result["attempted_frames"] == 2736 and result["optimizer_steps"] == 12 * fit["active_frames"]
                and result["zero_features"] == (arm == "logit_only")
                and result["initial_state_sha256"] == fit["initial_state_sha256"]
                and descriptor in fit["outputs"] and descriptor["kind"] == "terminal-head"
                and re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
                and descriptor["bytes"] > 0, "complete terminal checkpoint metadata")
    expected_log = [(arm, epoch, position, index)
                    for arm in ARMS for epoch, order in enumerate(orders["orders"], 1)
                    for position, index in enumerate(order)]
    require([(r["arm"], r["epoch"], r["position"], r["index"]) for r in training_log] == expected_log
            and all({k: r[k] for k in ("sequence", "frame")} == orders["identities"][r["index"]]
                    for r in training_log), "all 5472 original training-log identities required")

    seal, report = load(spec["prediction_seal"]), load(spec["evaluation_report"])
    flattened = [row["predictions"][arm][i] for row in prediction["rows"]
                 for i in range(len(row["frame_records"])) for arm in ARMS]
    require(prediction["schema"] == "evr0033.native-feature-prediction.v1"
            and prediction["arms"] == list(ARMS) and prediction["completed_frames"] == 241
            and prediction["native_parity_checks"] == 241 and len(prediction["rows"]) == 6
            and prediction["completed_arm_predictions"] == prediction["zero_head_parity_checks"] == 482
            and sum(len(r["frame_records"]) for r in prediction["rows"]) == 241
            and all(r["status"] == "completed" and all(len(r["predictions"][a]) == len(r["frame_records"])
                    for a in ARMS) for r in prediction["rows"])
            and len(prediction["outputs"]) == 482 and prediction["current_label_decodes"] == 0
            and prediction["offline_attempts"] == []
            and prediction["native_calls"] == prediction["image_encoder_calls"] == 0
            and prediction["native_weights_loaded"] is False
            and prediction["native_weights_unchanged_by_reference"] is True
            and prediction["learned_weights_unchanged"] is True
            and prediction["terminal_checkpoints"] == terminal and prediction["normalization"] == fit["normalization"]
            and prediction["bindings"]["fit_receipt"]["sha256"] == sha(raw(completed["fit"][0]["receipt"]))
            and seal["schema"] == "evr0033.native-feature-seal.v1" and seal["complete"] is True
            and seal["frame_count"] == 241 and seal["arm_prediction_count"] == 482
            and seal["arms"] == list(ARMS)
            and seal["receipt_sha256"] == sha(raw(completed["prediction"][0]["receipt"]))
            and seal["source_sha256"] == prediction["source_sha256"]
            and seal["spec_sha256"] == prediction["spec_sha256"]
            and seal["outputs"] == prediction["outputs"] == flattened, "complete prediction seal required")
    require(evaluation["schema"] == "evr0033.native-feature-evaluation.v1"
            and report["schema"] == "evr0033.native-feature-metrics.v1"
            and report["official_benchmark_result"] is False
            and evaluation["model_calls"] == evaluation["native_calls"] == evaluation["image_encoder_calls"] == 0
            and evaluation["predictions_sealed_before_gt"] is True
            and evaluation["verified_unique_frames"] == evaluation["label_decodes"] == 241
            and evaluation["verified_arm_predictions"] == 482
            and evaluation["prediction_seal_sha256"] == sha(raw(spec["prediction_seal"]))
            and evaluation["evaluation_sha256"] == sha(raw(spec["evaluation_report"]))
            and evaluation["bindings"]["prediction_receipt"]["sha256"] == seal["receipt_sha256"]
            and evaluation["bindings"]["fit_receipt"] == prediction["bindings"]["fit_receipt"]
            and evaluation["train_events"] == report["train"]["event_count"] == len(report["train"]["rows"]) == 96
            and evaluation["train_unique_frames"] == report["train"]["unique_future_frames"] == 87
            and evaluation["dev_raw_frames"] == report["dev"]["raw_frames"] == len(report["dev"]["rows"]) == 154
            and evaluation["dev_interior_frames"] == report["dev"]["included_frames"] == 150
            and len(report["train"]["parents"]) == 4 and len(report["dev"]["parents"]) == 2
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
        elif path.suffix == ".jsonl":
            require(len(decode_lines(original)) == len(decode_lines(published)), "JSONL row count changed")
        relative = destinations[path]
        prepared[relative] = published
        entries.append(dict(path=relative, sha256=sha(published), bytes=len(published),
            origin=dict(path=sanitize(str(path)), sha256=sha(original), bytes=len(original)),
            transformation="byte-identical" if original == published else "literal workspace prefix -> ${WORKSPACE} only"))
    source_raw = here.read_bytes()
    manifest = dict(schema="evr0033.text-publication.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        entries=entries, publication_spec=decode(sanitize(spec_raw.decode("utf-8"))),
        publication_spec_sha256=sha(spec_raw), publication_spec_bytes=len(spec_raw),
        publisher=dict(path=sanitize(str(here)), sha256=sha(source_raw), bytes=len(source_raw)),
        deduplicated_inputs=decode(sanitize(json.dumps(duplicates))), all_json_fields_rows_preserved=True,
        all_jsonl_rows_preserved=True, training_log_rows=len(training_log),
        counts=dict(train_fit_frames=228, train_fit_attempts=5472, arm_predictions=482, unique_prediction_frames=241,
                    train_metric_events=96, train_unique_future_frames=87, dev_raw_frames=154, dev_interior_frames=150),
        terminal_checkpoints=decode(sanitize(json.dumps(terminal))),
        operation_costs={k: {name: value[2][name] for name in ("wall_seconds", "peak_group_rss_kib")}
                         for k, value in completed.items()},
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
