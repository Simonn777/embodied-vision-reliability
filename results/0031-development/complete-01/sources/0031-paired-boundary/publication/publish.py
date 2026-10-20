"""Publish completed 0031 text evidence only; no scientific imports or payload reads."""
import argparse
from collections import Counter
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
ARMS = ("paired", "unpaired_max", "deranged", "native")
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("nonfinite JSON token: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def run(output):
    root = Path(__file__).resolve().parents[1]
    work, task = root.parent, root.parent.parent
    repo = work / "embodied-vision-reliability"
    output = output.resolve()
    require(output == root / "publication-complete-01" and not output.exists(), "fresh fixed output required")
    inputs, files, costs, source_destinations = {}, {}, {}, {}

    def read(path):
        path = Path(path).resolve()
        require(path.suffix in (".json", ".log", ".txt", ".py", ".md"), "text inputs only")
        path.relative_to(work)
        raw = path.read_bytes()
        raw.decode("utf-8")
        require(path not in inputs or inputs[path] == sha(raw), "input changed during publication")
        inputs[path] = sha(raw)
        return raw

    def load(path):
        return parse(read(path))

    def copy(path, relative):
        path = Path(path).resolve()
        require(relative not in files or files[relative][1] == path, "duplicate destination")
        files[relative] = (read(path), path)

    def source(path, expected=None):
        path = Path(path).resolve()
        raw = read(path)
        require(expected is None or sha(raw) == expected, "source hash mismatch: " + str(path))
        destination = "sources/" + path.relative_to(work).as_posix()
        copy(path, destination)
        source_destinations[str(path)] = destination

    def operation(label, directory, number, worker, completed=True):
        sp, op = directory / f"spec-{number}.json", directory / f"outer-spec-{number}.json"
        rp, xp = directory / f"run-{number}/receipt.json", directory / f"supervision-{number}/receipt.json"
        spec, outer_spec, receipt, outer = load(sp), load(op), load(rp), load(xp)
        require(Path(spec["output"]) == rp.parent and Path(outer_spec["result_file"]) == rp
                and Path(outer_spec["operation_output"]) == xp.parent, "operation paths disagree")
        require(outer_spec["command"][-2:] == [str(worker), str(sp)], "unexpected worker command")
        normalized = [s.replace(str(repo), "$REPOSITORY").replace(str(work), "$PRIVATE")
                      for s in outer_spec["command"]]
        require(outer["command"] == normalized, "outer command binding")
        require(outer["schema"] == "evr0029.exit-checked-operation.v1"
                and outer["supervisor_sha256"] == SUPERVISOR and outer["monitor_sha256"] == MONITOR,
                "unexpected supervisor/monitor")
        require(outer["spec_sha256"] == inputs[op] and outer["output_sha256"] == inputs[rp]
                and outer["expected_commit"] == outer_spec["commit"], "outer hash/commit binding")
        require(receipt["spec_sha256"] == inputs[sp] and receipt["source_sha256"] == spec["source_sha256"],
                "inner source/spec binding")
        require(0 < outer["hard_timeout_seconds"] == outer_spec["timeout"] <= 1800
                and outer["group_rss_limit_kib"] == 8388608
                and math.isfinite(outer["wall_seconds"]) and 0 <= outer["wall_seconds"] <= outer_spec["timeout"]
                and 0 <= outer["peak_group_rss_kib"] <= 8388608, "resource bound")
        require(outer["thread_environment"] == {k: "4" for k in THREADS}
                and all(outer_spec["environment"][k] == "4" for k in THREADS), "thread bound")
        if completed:
            require(receipt["status"] == outer["status"] == "completed" and outer["exit_code"] == 0
                    and outer["termination"] is None and outer["remaining_descendant_pids"] == []
                    and outer["cleanup_errors"] == [] and outer["signal_attempts"] == []
                    and outer["triggering_snapshot"] is None and outer["post_exit_snapshot"]["rows"] == []
                    and len(outer["cleanup_snapshots"]) == 1
                    and outer["cleanup_snapshots"][0]["label"] == "final"
                    and outer["cleanup_snapshots"][0]["rows"] == [], "completed clean operation required")
        else:
            require(receipt["status"] == outer["status"] == "failed" and outer["exit_code"] != 0,
                    "original failed attempt must remain failed")
        for path, destination in ((sp, "worker-spec.json"), (op, "outer-spec.json"),
                                  (rp, "run/receipt.json"), (xp, "supervision/receipt.json")):
            copy(path, label + "/" + destination)
        require(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "both original logs required")
        for name, expected in outer["log_sha256"].items():
            path = xp.parent / name
            copy(path, label + "/supervision/" + name)
            require(inputs[path] == expected, "outer log hash")
        for path, expected in spec["source_sha256"].items():
            source(path, expected)
        # These six entries are metadata only. Never follow their array/media descriptors.
        for entry in spec.get("bindings", {}).values():
            raw = read(entry["path"])
            require(sha(raw) == entry["sha256"] and len(raw) == entry["bytes"], "metadata reference binding")
        costs[label] = dict(status=outer["status"], exit_code=outer["exit_code"],
                            outer_wall_seconds=outer["wall_seconds"],
                            inner_wall_seconds=receipt["wall_seconds"],
                            peak_group_rss_kib=outer["peak_group_rss_kib"])
        return spec, receipt

    def declared_outputs(label, directory, receipt):
        indexed = {}
        for descriptor in receipt["outputs"]:
            name = descriptor["path"]
            require(Path(name).name == name and name.endswith(".json") and name not in indexed,
                    "unique flat JSON output required")
            path = directory / name
            raw = read(path)
            require(sha(raw) == descriptor["sha256"] and len(raw) == descriptor["bytes"], "output binding")
            copy(path, label + "/run/" + name)
            indexed[name] = descriptor
        require({p.name for p in directory.iterdir() if p.is_file()} == {"receipt.json", *indexed},
                "unaccounted run output")
        return indexed

    source(repo / "experiments/0029-memory-information/supervise_exit_checked.py", SUPERVISOR)
    source(repo / "experiments/0025c-rs-noise/supervise.py", MONITOR)
    cs, cr = operation("calibration", root / "calibration", "01", root / "calibration/calibrate.py")
    s1, r1 = operation("screen01-failed", root, "01", root / "screen.py", completed=False)
    ts, tr = operation("arithmetic-qualification", root / "technical-tests", "01",
                       root / "technical-tests/qualify_arithmetic.py")
    s2, r2 = operation("screen02", root, "02", root / "screen_sum.py")
    require(s1["contract"] == s2["contract"] == r1["contract"] == r2["contract"]
            and s1["bindings"] == s2["bindings"] == cs["bindings"]
            and all(s2["source_sha256"][p] == h for p, h in s1["source_sha256"].items()),
            "technical continuation changed original scientific contract/source/inputs")
    require(r1["rows"] == [] and r1["stage"] == "ranking" and r1["error_type"] == "RuntimeWarning",
            "unexpected original failure scope")
    declared_outputs("screen01-failed", root / "run-01", r1)
    outputs = declared_outputs("screen02", root / "run-02", r2)
    require(len(outputs) == 198 and len(r2["rows"]) == r2["completed_events"] == 96
            and r2["completed_parents"] == 4 and r2["stage"] == "complete" and r2["pending"] is None
            and r2["model_calls"] == 0 and r2["final_test_access"] is False and r2["offline_attempts"] == [],
            "screen incomplete or incompatible")
    rank_names, diagnostic_names, identities = [], [], set()
    identity_keys = ("sequence", "parent_id", "anchor", "horizon", "frame", "target_id")
    for row in r2["rows"]:
        identity = tuple(row[k] for k in identity_keys)
        require(identity not in identities and row["status"] == "completed"
                and row["horizon"] in (1, 3) and row["frame"] == row["anchor"] + row["horizon"],
                "invalid/duplicate completed event")
        identities.add(identity)
        for key, names in (("ranking", rank_names), ("diagnostic", diagnostic_names)):
            name = row[key]["path"]; names.append(name)
            require(outputs[name] == row[key], "row/output descriptor mismatch")
            obj = load(root / "run-02" / name)
            require(tuple(obj[k] for k in identity_keys) == identity, "row/file identity mismatch")
            if key == "diagnostic":
                require(obj["summary"] == row["summary"], "diagnostic/summary mismatch")
    require(len(set(rank_names)) == len(set(diagnostic_names)) == 96
            and Counter(r["sequence"] for r in r2["rows"]) == Counter({p: 24 for p in s2["contract"]["train"]}),
            "complete four-parent event coverage")
    seal = load(root / "run-02/ranking-seal.json")
    require(seal["count"] == 96 and seal["spec_sha256"] == r2["spec_sha256"]
            and seal["source_sha256"] == r2["source_sha256"]
            and seal["outputs"] == r2["outputs"][:100]
            and {d["path"] for d in seal["outputs"]} == set(rank_names) |
                {p + "-reference.json" for p in s2["contract"]["train"]}, "pre-label ranking seal binding")
    report = load(root / "run-02/analysis.json")
    require(report["rows"] == r2["rows"] and report["contract"] == r2["contract"], "analysis receipt mismatch")
    agg = report["aggregation"]; delta = agg["equal_parent_event_delta_from_native"]
    decision = load(root / "decision-02.json")
    clauses = (all(delta["paired"]["joint_loss"] < delta[a]["joint_loss"] for a in ARMS[1:]),
               sum(all(p["event_mean_delta"]["paired"]["joint_loss"] < p["event_mean_delta"][a]["joint_loss"]
                       for a in ("unpaired_max", "deranged")) for p in agg["parents"]),
               delta["paired"]["correct"] >= 0)
    require(decision["analysis_sha256"] == inputs[root / "run-02/analysis.json"]
            and decision["admission_sha256"] == inputs[root / "ADMISSION.md"]
            and (decision["mean_joint_loss_below_three_controls"], decision["parents_below_both_matching_controls"],
                 decision["paired_correct_crossing_no_lower_than_native"]) == clauses
            and not (clauses[0] and clauses[1] >= 3 and clauses[2])
            and decision["action"] == "close_fixed_initial_pair_lookup"
            and decision["final_evidence_gate_met"] is False, "saved closure/criterion mismatch")
    copy(root / "decision-02.json", "decision.json")

    selection_path = root / "calibration/run-01/train-selection.json"
    evaluation_path = root / "calibration/run-01/evaluation.json"
    selection, evaluation = load(selection_path), load(evaluation_path)
    require(cr["train_selection_sha256"] == inputs[selection_path]
            and cr["evaluation_sha256"] == inputs[evaluation_path]
            and cr["selected_threshold"] == selection["selected_threshold"] == evaluation["selected_threshold"]
            and cr["train_complete_before_dev_decode"] is True and cr["completed_train_events"] == 96
            and len(selection["rows"]) == 96 and len(evaluation["rows"]) == 154
            and cr["completed_dev_interior_frames"] == evaluation["included_frames"] == 150
            and cr["tensor_loads"] == len(cr["zero_threshold_parity"]) == 241
            and all(x["differing_pixels"] == 0 for x in cr["zero_threshold_parity"]), "calibration completeness/parity")
    for path in (selection_path, evaluation_path):
        copy(path, "calibration/run/" + path.name)
    require(tr["completed_cases"] == len(tr["cases"]) == 100 and tr["stable_adapter_passed"] is True
            and tr["source_post_sha256"] == ts["source_sha256"]
            and tr["geometry_tests"]["count"] == 7 and tr["geometry_tests"]["failures"] == 0
            and tr["geometry_tests"]["errors"] == 0, "arithmetic qualification incomplete")
    log = root / "technical-tests/run-01/adapter-geometry-tests.txt"
    copy(log, "arithmetic-qualification/run/" + log.name)
    require(inputs[log] == tr["geometry_tests"]["log_sha256"], "geometry log hash")
    for filename, expected in ts["source_sha256"].items():
        snapshot = root / "technical-tests/sources-01" / Path(filename).name
        require(sha(read(snapshot)) == expected, "manufactured source snapshot changed")
    mt = load(root / "manufactured-tests-01.json")
    require(mt["exit_code"] == 0, "first manufactured tests unsuccessful")
    for name, expected in mt["source_sha256"].items():
        source(root / name, expected)
    for suffix in ("json", "stdout.log", "stderr.log"):
        path = root / ("manufactured-tests-01." + suffix)
        copy(path, "manufactured-tests/" + path.name)
    for path in (root / "paired-reference-prior.md", root / "initial-mask-adaptation-prior.md",
                 Path(__file__).resolve(), Path(__file__).with_name("USAGE.md")):
        if path.is_file():
            source(path)
    ci_root = work / "0030-memory-content/active-write-diagnostic"
    for name, commit, run_id in (
        ("result-ci-540943f.json", "540943f2d78dd815cb0db1cd411f2011d4c52620", 39416982031),
        ("main-ci-bc220ac.json", "bc220ace2cdf2b138cb968ebddb967a6fadf20ca", 39417542844)):
        ci = load(ci_root / name)
        require(ci["headSha"] == commit and ci["databaseId"] == run_id
                and ci["status"] == "completed" and ci["conclusion"] == "success", "prior CI metadata")
        copy(ci_root / name, "prior-publication/" + name)

    metrics = agg["pooled_crossing_means"]
    table = "\n".join(f"| {a} | {metrics[a]['joint_loss']:.10f} | {metrics[a]['correct']:.10f} |"
                      for a in ARMS)
    parent_table = "\n".join(f"| {p['parent_id']} | {p['events']} | {p['empty_events']} | "
        f"{p['event_mean_delta']['paired']['joint_loss']:+.10f} | "
        f"{p['event_mean_delta']['paired']['correct']:+.10f} |" for p in agg["parents"])
    cost_table = "\n".join(f"| {name} | {c['status']} | {c['outer_wall_seconds']:.12f} | "
                          f"{c['peak_group_rss_kib']} |" for name, c in costs.items())
    summary = dict(scope=report["scope"], decision=decision, aggregation=agg,
        calibration=dict(selected_threshold=cr["selected_threshold"], native=evaluation["native"],
                         selected=evaluation["selected"], delta=evaluation["delta"], exact_parity_frames=241),
        operations=costs, original_manufactured_test_wall_seconds=mt["wall_seconds"],
        outer_wall_sum_seconds=math.fsum(c["outer_wall_seconds"] for c in costs.values()),
        arithmetic=dict(cases=100, geometry_tests=7, matmul_attempts=tr["matmul_attempts"],
                        matmul_successes=tr["matmul_successes"], warnings=len(tr["warnings"]),
                        max_explicit_fsum_error=tr["max_explicit_fsum_error"]))
    files["summary.json"] = ((json.dumps(summary, allow_nan=False) + "\n").encode(), None)
    readme = f"""# 0031 completed development screen — fixed lookup closed

The fixed initial-pair lookup is closed. The saved prospective continuation clauses
are {clauses[0]}, {clauses[1]}/4 parents (required at least 3), and {clauses[2]}.
No repair head is admitted by this result. No novelty, whole-mask gain, independent
replication or final two-source +3 J/F evidence is established.

All 96 TRAIN anchor/horizon events, 87 unique future frames and
{agg['pooled_crossings']} current crossing queries remain, including fixed fallbacks.
The 96 ranking files were sealed before current TRAIN labels were decoded; all
96 diagnostic files and every candidate field are retained. Four final-frame
events are included. These correlated TRAIN events are not an official benchmark.

| Arm | Pooled joint loss (lower better) | Correctly oriented crossing fraction |
|---|---:|---:|
{table}

The decision uses equal-parent/event deltas, not a redefined pooled gate.

| Parent | Events | Empty events | Paired joint-loss delta from native | Paired correct-fraction delta |
|---|---:|---:|---:|---:|
{parent_table}

The separate TRAIN-selected global output threshold is {cr['selected_threshold']}.
All 241 saved native-zero-mask parity checks passed. Fixed DEV J/F/JF deltas
are {json.dumps(evaluation['delta'])} in fraction units, with all 150 interior
frames (154 raw) retained. This is a calibration control, not a contribution.

Attempt01 remains failed: {r1['error_type']}: {r1['error']}; zero completed ranking
events, before current-label diagnostics. The fresh attempt02 uses explicit
float64 multiply/sum reduction after manufactured qualification; its scientific
contract and original source are unchanged. Exact historical cause is not proven.
The qualification recorded {len(tr['warnings'])}/{tr['matmul_attempts']} warning-producing
matmul attempts and {tr['matmul_successes']} successful matmul calls; a stored zero
max error for zero successful calls is not an accuracy result. The explicit
adapter passed 100 synthetic cases and all seven geometry tests.

| Operation | Status | Complete outer seconds | Peak sampled group RSS KiB |
|---|---|---:|---:|
{cost_table}

Outer sum {summary['outer_wall_sum_seconds']:.12f}s includes the failed screen.
The separate original manufactured unit command took {mt['wall_seconds']:.12f}s.
Inner times overlap outer times and are not added again. Prior 0030 cache/model
construction costs are referenced by the bound metadata, not remeasured or free.

This export only reads text. No arrays, media, model, metrics or scientific
pipeline were rerun. JSON is compactly reserialized with every key/value retained
then literal task/home prefixes aliased; original raw and public byte hashes
identify the distinct representations. Source snapshots are deduplicated by
source identity; source origins, tests, failure logs, specs and receipts remain.
Embedded output hashes still address original bytes, as mapped by the manifest.
The two PR67 CI records are prior publication context, not CI for these private
development workers; their worker sources are bound by operation specifications.
Original prospective scope/admission notes are preserved unchanged.
"""
    files["README.md"] = (readme.encode(), None)
    aliases = ((str(task), "$TASK"), (str(Path.home()), "$USER_HOME"))
    def alias(value):
        if isinstance(value, str):
            for old, new in aliases:
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [alias(v) for v in value]
        if isinstance(value, dict):
            return {alias(k): alias(v) for k, v in value.items()}
        return value
    entries, prepared = [], {}
    for name, (raw, origin) in files.items():
        if name.endswith(".json"):
            original_value = parse(raw)
            text = json.dumps(original_value, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
            public = alias(text).encode()
            require(parse(public) == alias(original_value), "JSON values/keys changed during transformation")
            transform = "JSON compact reserialization preserving all values/keys; then ordered literal prefix aliases"
        else:
            public = alias(raw.decode()).encode()
            transform = "byte-identical" if raw == public else "ordered literal prefix aliases only"
        require(not re.search(rb"/Users/[A-Za-z0-9_.-]+/|/(?:private/)?var/folders/[A-Za-z0-9]+/", public),
                "unaliased private path")
        prepared[name] = public
        entry = dict(path=name, sha256=sha(public), bytes=len(public), transformation=transform)
        if origin:
            entry["origin"] = dict(path=alias(str(origin)), sha256=sha(raw), bytes=len(raw))
        else:
            entry["derivation"] = "Saved scalar aggregation, decision, calibration and operation costs; no scientific recalculation."
        entries.append(entry)
    for path, expected in inputs.items():
        require(sha(path.read_bytes()) == expected, "input changed before export")
    output.mkdir(exist_ok=False)
    for name, raw in prepared.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = dict(schema="evr0031.text-publication.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        entries=entries, inputs_sha256={alias(str(p)): h for p, h in inputs.items()},
        source_destinations=alias(source_destinations),
        substitutions=["literal task root -> $TASK", "then literal current user home -> $USER_HOME"],
        ranking_files=96, diagnostic_files=96, all_json_keys_values_preserved=True,
        scientific_payload_reads=False, original_failed_attempt_preserved=True)
    manifest_raw = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    (output / "manifest.json").write_bytes(manifest_raw)
    for path, expected in inputs.items():
        require(sha(path.read_bytes()) == expected, "input changed after export")
    print(json.dumps(dict(files=len(prepared)+1, bytes=sum(map(len, prepared.values()))+len(manifest_raw),
                          manifest_sha256=sha(manifest_raw), ranking_files=96, diagnostic_files=96)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
