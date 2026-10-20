"""Explicit-list 0034 text export; no model, array, metric or scientific imports."""
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
STAGES = ("prepare", "fit_predict", "evaluate")
PARENTS = (("dog-agility", 25), ("planes-water", 38), ("dogs-scale", 83), ("dog-gooses", 86))
ARMS = ("native", "logit", "absolute", "prototype", "nn", "ordinary_relations", "bilateral",
        "signed_no_gram", "signed_gram", "displaced_gram", "rgb256")
OLD_KEYS = {"cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec", "mapping", "data_index"}
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
    require(len(spec["operations"]) == 3, "prepare, fit_predict and evaluate operations required")
    for operation in spec["operations"]:
        kind = operation["kind"]
        require(kind in STAGES and kind not in completed, "unique operation kinds")
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
                and math.isfinite(outer["peak_group_rss_kib"]) and 0 <= outer["peak_group_rss_kib"] <= 8388608
                and outer["thread_environment"] == {k: "4" for k in THREADS}
                and all(outer_spec["environment"][k] == "4" for k in THREADS), "operation resource limits")
        require(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "both original logs required")
        for name, digest in outer["log_sha256"].items():
            require(sha(raw(Path(operation["outer"]).parent / name)) == digest, "operation log changed")
        require(receipt["schema"] == "evr0034.relative-rgb-operation.v1"
                and receipt["stage"] == worker["stage"] == kind
                and receipt["bindings"] == worker["bindings"]
                and len(worker["source_sha256"]) == 15
                and receipt["offline_attempts"] == []
                and receipt["native_calls"] == receipt["image_encoder_calls"] == 0
                and receipt["native_model_loaded"] is False and receipt["history_modified"] is False
                and receipt["dev_payload_access"] is False and receipt["final_test_access"] is False,
                "fixed stage/source/data scope")
        for path, digest in worker["source_sha256"].items():
            require(sha(raw(path)) == digest, "all 15 executed sources must be included unchanged")
        seal = load(operation["seal"])
        require(Path(operation["seal"]).resolve() == Path(operation["receipt"]).resolve().parent / "seal.json"
                and seal["schema"] == "evr0034.stage-seal.v1" and seal["complete"] is True
                and seal["stage"] == kind and seal["receipt_sha256"] == sha(raw(operation["receipt"]))
                and seal["source_sha256"] == receipt["source_sha256"]
                and seal["outputs"] == receipt["outputs"], "complete stage seal binding")
        completed[kind] = (operation, receipt, outer)

    prepared, fitted, evaluated = (completed[k][1] for k in STAGES)
    require(prepared["source_sha256"] == fitted["source_sha256"] == evaluated["source_sha256"],
            "all three stages must use identical scientific source bytes")
    for child, keys in ((prepared, OLD_KEYS),
                        (fitted, OLD_KEYS | {"prepared_" + s for s in ("receipt", "outer", "operation", "worker_spec")}),
                        (evaluated, OLD_KEYS | {p + "_" + s for p in ("prepared", "fitpred")
                                              for s in ("receipt", "outer", "operation", "worker_spec")})):
        require(set(child["bindings"]) == keys
                and all(child["bindings"][k] == prepared["bindings"][k] for k in OLD_KEYS),
                "exact inherited input bindings")
    for child, parent, prefix in ((fitted, "prepare", "prepared"),
                                  (evaluated, "prepare", "prepared"),
                                  (evaluated, "fit_predict", "fitpred")):
        for suffix, key in (("receipt", "receipt"), ("outer", "outer"),
                            ("operation", "operation_spec"), ("worker_spec", "worker_spec")):
            binding, original = child["bindings"][prefix + "_" + suffix], completed[parent][0][key]
            require(Path(binding["path"]).resolve() == Path(original).resolve()
                    and binding["sha256"] == sha(raw(original)) and binding["bytes"] == len(raw(original)),
                    "cross-operation metadata binding")

    inventories = {}
    for stage, (_, receipt, _) in completed.items():
        inventory = {}
        root = Path(completed[stage][0]["receipt"]).resolve().parent
        for descriptor in receipt["outputs"]:
            relative = Path(descriptor["path"])
            require(not relative.is_absolute() and ".." not in relative.parts
                    and descriptor["path"] not in inventory
                    and re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
                    and type(descriptor["bytes"]) is int and descriptor["bytes"] > 0, "output descriptor")
            require(relative.suffix in (".json", ".npz"), "unexpected stage output type")
            inventory[descriptor["path"]] = descriptor
            if relative.suffix == ".json":
                content = raw(root / relative)
                require(sha(content) == descriptor["sha256"] and len(content) == descriptor["bytes"],
                        "every produced JSON must be included byte-bound, without omissions")
            # Binary outputs are deliberately never opened, stat'ed, or rehashed here.
        inventories[stage] = inventory

    def text_output(stage, descriptor):
        require(inventories[stage].get(descriptor["path"]) == descriptor, "text output inventory mismatch")
        return load(Path(completed[stage][0]["receipt"]).resolve().parent / descriptor["path"])

    identities = [(name, frame) for name, n in PARENTS for frame in range(n)]
    def row_identities(rows):
        return [(row["sequence"], row["frame"]) for row in rows]

    require(prepared["frame_count"] == prepared["native_parities"] == len(prepared["frames"]) == 232
            and prepared["raw_frames"] == 232 and prepared["noninitial_frames"] == 228
            and prepared["learned_metric_frames"] == 224 and prepared["label_decodes"] == 0
            and prepared["all_features_gt_free"] is True
            and row_identities(prepared["frames"]) == identities
            and prepared["outputs"] == [row["feature"] for row in prepared["frames"]]
            and all(d["kind"] == "causal-features" and Path(d["path"]).suffix == ".npz"
                    for d in prepared["outputs"]), "complete 232-frame preparation")
    require(fitted["completed_fits"] == len(fitted["fits"]) == 40 and len(fitted["folds"]) == 4
            and fitted["prediction_frames"] == len(fitted["prediction_rows"]) == 232
            and fitted["arm_predictions"] == 2552 and fitted["arms"] == list(ARMS)
            and fitted["raw_frames"] == 232 and fitted["noninitial_frames"] == 228
            and fitted["primary_interior_frames"] == 224
            and fitted["labels_used_only_from_other_parents_per_fold"] is True
            and fitted["held_parent_scoring_performed"] is False
            and fitted["terminal_models_sealed_before_prediction"] is True
            and fitted["terminal_models_unchanged"] is True
            and row_identities(fitted["frames"]) == [(n, f) for n, count in PARENTS for f in range(1, count)]
            and row_identities(fitted["prediction_rows"]) == identities, "complete fixed cross-fit coverage")
    expected_fits = [(parent, arm) for parent, _ in PARENTS for arm in ARMS[1:]]
    require([(r["held_parent"], r["arm"]) for r in fitted["fits"]] == expected_fits
            and [r["held_parent"] for r in fitted["folds"]] == [n for n, _ in PARENTS],
            "all 40 named attempts and four folds retained")
    terminal = text_output("fit_predict", fitted["terminal_models"])
    require(set(terminal) == {n for n, _ in PARENTS}
            and fitted["terminal_models"]["kind"] == "all-fold-terminal-models", "terminal aggregate")
    expected_outputs, failed = [], []
    for fold in fitted["folds"]:
        name = fold["held_parent"]
        expected = [(n, f) for n, count in PARENTS if n != name for f in range(1, count)]
        require(fold["training_parents"] == [n for n, _ in PARENTS if n != name]
                and row_identities(fold["training_identities"]) == expected
                and fold["training_frames"] == len(expected)
                and terminal[name]["normalization"] == fold["normalization"]
                and all(len(fold["normalization"][k]) == 32 for k in ("mean", "std"))
                and set(terminal[name]["arms"]) == set(ARMS[1:]), "fold identities/normalization metadata")
    for fit in fitted["fits"]:
        name, arm = fit["held_parent"], fit["arm"]
        require(fit["status"] in ("completed", "failed")
                and math.isfinite(fit["wall_seconds"]) and fit["wall_seconds"] >= 0,
                "fit attempt status/cost")
        for key, kind in (("model", "terminal-readout"), ("trace", "complete-optimizer-closure-trace")):
            descriptor = fit[key]
            require(descriptor["kind"] == kind and descriptor["held_parent"] == name
                    and descriptor["arm"] == arm and Path(descriptor["path"]).suffix == ".json",
                    "all individual terminal models and closure traces required")
            expected_outputs.append(descriptor)
        model = text_output("fit_predict", fit["model"])
        text_output("fit_predict", fit["trace"])  # Preserve every key/closure, including failure traces.
        require(model == terminal[name]["arms"][arm], "individual/aggregate terminal mismatch")
        if fit["status"] == "failed":
            failed.append((name, arm))
    require([(r["held_parent"], r["arm"]) for r in fitted["fit_failures"]] == failed,
            "all failed fit attempts must be retained")
    expected_outputs.append(fitted["terminal_models"])
    for prior, row in zip(prepared["frames"], fitted["prediction_rows"]):
        require(all(row[k] == prior[k] for k in ("sequence", "parent_id", "frame", "frames", "shape", "target_id",
                "query_count", "sample_count", "native_presence", "noop_reason", "label", "native", "feature"))
                and set(row["detail"]) == set(ARMS[1:]), "prediction/preparation identity")
        descriptor = row["prediction"]
        require(descriptor["kind"] == "all-arm-predictions" and Path(descriptor["path"]).suffix == ".npz"
                and (descriptor["sequence"], descriptor["frame"]) == (row["sequence"], row["frame"]),
                "prediction descriptor identity")
        expected_outputs.append(descriptor)
    require(fitted["outputs"] == expected_outputs and len(expected_outputs) == 313,
            "80 individual fit JSONs, terminal aggregate and 232 prediction inventories required")

    report = load(spec["evaluation_report"])
    require(Path(spec["evaluation_report"]).resolve() ==
            Path(completed["evaluate"][0]["receipt"]).resolve().parent / evaluated["evaluation"]["path"]
            and evaluated["outputs"] == [evaluated["evaluation"]]
            and evaluated["evaluation"]["kind"] == "complete-cross-fit-metrics"
            and text_output("evaluate", evaluated["evaluation"]) == report
            and evaluated["predictions_sealed_before_metric_labels"] is True
            and evaluated["verified_frames"] == evaluated["frame_count"] == evaluated["native_parities"] == 232
            and evaluated["label_decodes"] == 232 and evaluated["verified_arm_predictions"] == 2552
            and evaluated["fit_failures"] == report["fit_failures"] == fitted["fit_failures"],
            "complete sealed metric output and preserved fit failures")
    require(report["schema"] == "evr0034.relative-rgb-metrics.v1"
            and report["official_benchmark_result"] is False
            and report["goal_achieved"] is False and report["manuscript_claim"] is False
            and report["arm_order"] == list(ARMS) and row_identities(report["rows"]) == identities
            and report["coverage"] == dict(complete=True, raw_frames=232, noninitial_frames=228,
                primary_frames=224, parents=4, arms=11, raw_arm_rows=2552)
            and report["primary"]["frame_count"] == 224 and len(report["primary"]["parents"]) == 4
            and report["secondary"]["event_count"] == len(report["secondary"]["events"]) == 96
            and report["secondary"]["unique_frames"] == 87 and report["secondary"]["final_frame_events"] == 4
            and len(report["secondary"]["parents"]) == 4 and report["native_secondary_parity"] is True
            and evaluated["continuation_passed"] == report["decision"]["passed"]
            and type(report["decision"]["passed"]) is bool,
            "complete 232-row / 224-primary / 96-event saved metric coverage")
    require(all(set(row["arms"]) == set(ARMS) for row in report["rows"])
            and (not failed or report["decision"]["passed"] is False), "failure-inclusive metric report")
    # No pose/mask metrics, optimization, or scientific gate arithmetic is recomputed.

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
    manifest = dict(schema="evr0034.text-publication.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        entries=entries, publication_spec=decode(sanitize(spec_raw.decode("utf-8"))),
        publication_spec_sha256=sha(spec_raw), publication_spec_bytes=len(spec_raw),
        publisher=dict(path=sanitize(str(here)), sha256=sha(source_raw), bytes=len(source_raw)),
        deduplicated_inputs=decode(sanitize(json.dumps(duplicates))), all_json_fields_rows_preserved=True,
        all_jsonl_rows_preserved=True,
        counts=dict(prepared_frames=232, fit_attempts=40, failed_fits=len(failed),
                    individual_terminal_and_trace_jsons=80, terminal_aggregate_jsons=1,
                    prediction_archives=232, arm_masks=2552, metric_raw_frames=232,
                    primary_interiors=224, secondary_events=96, secondary_unique_frames=87,
                    secondary_final_frame_events=4),
        all_fit_failures_and_optimizer_traces_preserved=True,
        scientific_gate_passed=report["decision"]["passed"],
        operation_costs={k: {name: value[2][name] for name in ("wall_seconds", "peak_group_rss_kib")}
                         for k, value in completed.items()},
        no_scientific_recalculation=True, no_binary_payload_reads=True,
        upstream_source_scope="Hashes/pin references retained in executed specs; upstream sources not copied.",
        validation_scope="Three completed stage receipts/specs/logs/seals, all text outputs and full descriptor inventories; no independent metric or binary payload audit.")
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
