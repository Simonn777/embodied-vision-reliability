"""Publish the completed all-active triple from saved text only; no polling or science."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

BUDGET = 1168.509588667017
SUMMARY_SHA = "ccd09ae5786d25b564e7093c36c62060b1ad8a9e1949434e5034dcd4573c2111"
SUPERVISOR_SHA = "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8"
MONITOR_SHA = "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(root, output):
    root, output = root.resolve(), output.resolve()
    native, task = root.parent, root.parent.parents[1]
    require(output == native / "publication-active-write-01" and not output.exists(),
            "fresh fixed private publication directory required")
    inputs, files, operations, source_map = {}, {}, [], {}

    def read(path):
        path = Path(path)
        require(path.suffix in (".json", ".py", ".md", ".log"), "text-only input")
        raw = path.read_bytes()
        found = digest(raw)
        require(path not in inputs or inputs[path] == found, "input changed between reads")
        inputs[path] = found
        return raw

    def load(path):
        return json.loads(read(path), parse_constant=lambda v: (_ for _ in ()).throw(
            ValueError("nonfinite JSON: " + v)))

    def copy(path, relative):
        raw = read(path)
        require(relative not in files or files[relative][:2] == (raw, Path(path)),
                "conflicting export destination")
        files[relative] = (raw, Path(path), None)

    def generated(relative, value, derivation):
        require(relative not in files, "duplicate generated destination")
        files[relative] = ((json.dumps(value, indent=2, allow_nan=False) + "\n").encode(),
                           None, derivation)

    def descriptor(entry, expected):
        require(Path(entry["path"]) == expected and entry["sha256"] == digest(read(expected))
                and entry["bytes"] == len(read(expected)), "saved descriptor mismatch")

    helper = native / "analysis/summarize_receipts.py"
    require(digest(read(helper)) == SUMMARY_SHA, "reporting helper changed")
    loader = importlib.util.spec_from_file_location("active_saved_summary", helper)
    summary = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(summary)  # Pinned stdlib reporting functions; no model imports.
    state_path = root / "continuation-01.json"
    state = load(state_path)
    require(state["status"] == "completed" and [r["name"] for r in state["operations"]] ==
            ["rollout-01", "evaluate-01"] and
            all(r["status"] == "completed" for r in state["operations"]),
            "continuation incomplete; no publication output created")
    require(digest(read(root / "finish.py")) == state["source_sha256"] and
            digest(read(native / "campaign/continue_campaign.py")) == state["helper_sha256"],
            "continuation sources changed")

    def operation(name, spec_name, outer_name, script):
        spec_path, outer_spec_path = root / spec_name, root / outer_name
        spec, outer_spec = load(spec_path), load(outer_spec_path)
        receipt_path = root / name / "receipt.json"
        supervision = root / ("fit-supervision-01" if name == "fit-01" else name + "-supervision")
        outer_path = supervision / "receipt.json"
        receipt, outer = load(receipt_path), load(outer_path)
        require(Path(spec["output"]) == receipt_path.parent and
                Path(outer_spec["operation_output"]) == supervision and
                Path(outer_spec["result_file"]) == receipt_path, "fixed operation destinations")
        require(receipt["status"] == "completed" and
                not summary.clean_outer(outer, inputs[receipt_path], inputs[outer_spec_path]),
                name + " incomplete/unclean")
        require(outer["schema"] == "evr0029.exit-checked-operation.v1" and
                outer["supervisor_sha256"] == SUPERVISOR_SHA and outer["monitor_sha256"] == MONITOR_SHA and
                outer["expected_commit"] == outer_spec["commit"] and
                outer["hard_timeout_seconds"] == outer_spec["timeout"] == 1800 and
                outer["group_rss_limit_kib"] == 8388608 and
                math.isfinite(outer["wall_seconds"]) and 0 <= outer["wall_seconds"] <= 1800 and
                math.isfinite(outer["peak_group_rss_kib"]) and
                0 <= outer["peak_group_rss_kib"] <= 8388608, "outer source/resource envelope")
        keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        require(outer["thread_environment"] == {k: "4" for k in keys} and
                all(outer_spec["environment"][k] == "4" for k in keys), "four-thread contract")
        python = str(Path(spec["native_root"]) / "installation/venv/bin/python")
        command = [python] + ([] if name == "evaluate-01" else ["-S"])
        command += ["-B", "-W", "error", str(script), str(spec_path)]
        repo = Path(spec["repo"])
        displayed = [v.replace(str(repo), "$REPOSITORY").replace(str(repo.parent), "$PRIVATE")
                     for v in command]
        require(outer_spec["command"] == command and outer["command"] == displayed and
                receipt["spec_sha256"] == inputs[spec_path] and
                receipt["source_sha256"] == spec["source_sha256"], "command/spec/source binding")
        for source, expected in spec["source_sha256"].items():
            require(source not in source_map or source_map[source] == expected, "source drift across stages")
            source_map[source] = expected
        if name == "fit-01":
            descriptor(state["fit"], receipt_path); descriptor(state["fit_outer"], outer_path)
        else:
            saved = next(r for r in state["operations"] if r["name"] == name)
            descriptor(saved["worker"], spec_path); descriptor(saved["operation"], outer_spec_path)
            descriptor(saved["result"], receipt_path)
            require(saved["wall_seconds"] == outer["wall_seconds"] and
                    saved["peak_group_rss_kib"] == outer["peak_group_rss_kib"], "continuation costs")
        stem = "operations/" + name + "/"
        for path, relative in ((spec_path, "worker-spec.json"), (outer_spec_path, "outer-spec.json"),
                               (receipt_path, "receipt.json"), (outer_path, "supervision/receipt.json")):
            copy(path, stem + relative)
        require(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "complete logs")
        for filename, expected in outer["log_sha256"].items():
            path = supervision / filename
            copy(path, stem + "supervision/" + filename)
            require(inputs[path] == expected, "raw log changed")
        operations.append(dict(name=name, inner_sha256=inputs[receipt_path],
            outer_sha256=inputs[outer_path], outer_wall_seconds=outer["wall_seconds"],
            inner_wall_seconds=receipt["wall_seconds"], peak_group_rss_kib=outer["peak_group_rss_kib"]))
        return spec, receipt

    fit_spec, fit = operation("fit-01", "fit-spec-01.json", "fit-outer-spec-01.json", root / "train_active.py")
    require(fit_spec["rho"] == .15 and fit_spec["seed"] == 0 and
            fit_spec["mode"] == "direct" and fit_spec["direct_comparison"] == "cost_matched" and
            fit_spec["complete_cost_budget_seconds"] == fit["complete_cost_budget_seconds"] == BUDGET and
            fit["arm"] == "direct_all_active" and fit["write_scope"] == "all_active_noninitial_spatial" and
            operations[0]["outer_wall_seconds"] <= BUDGET, "admitted complete fit budget/scope")
    require(fit["rows"] and [r["step"] for r in fit["rows"]] == list(range(1, len(fit["rows"]) + 1)) and
            fit["checkpoints"][-1]["step"] == len(fit["rows"]) and
            all(math.isfinite(r["loss"]) for r in fit["rows"]), "complete fit rows/final step")
    checkpoint = fit["checkpoints"][-1]
    require(Path(checkpoint["path"]).name == checkpoint["path"] and checkpoint["bytes"] > 0 and
            all(re.fullmatch(r"[0-9a-f]{64}", c["sha256"]) for c in fit["checkpoints"]),
            "checkpoint descriptors")
    rspec, rollout = operation("rollout-01", "rollout-01-spec.json", "rollout-01-outer-spec.json",
                               native / "campaign/rollout.py")
    require(rspec["checkpoint"] == dict(checkpoint, path=str(root / "fit-01" / checkpoint["path"])) and
            rollout["arm"] == rspec["arm"] == "direct_all_active" and
            rspec["dev_sequences"] == ["cat-girl", "upside-down"] and rollout["frames"] == 154 and
            [r["sequence"] for r in rollout["rows"]] == rspec["dev_sequences"] and
            [r["frames"] for r in rollout["rows"]] == [89, 65] and
            rollout["outputs"] == [p for r in rollout["rows"] for p in r["predictions"]] and
            len(rollout["outputs"]) == 154 and all(
                [p["frame"] for p in r["predictions"]] == list(range(r["frames"])) for r in rollout["rows"]),
            "final-checkpoint/complete DEV rollout")
    for record in (fit, rollout):
        require(not record["offline_attempts"] and record["model_initial_hash"] == record["model_final_hash"],
                "frozen model/offline contract")
    espec, evaluation = operation("evaluate-01", "evaluate-01-spec.json", "evaluate-01-outer-spec.json",
                                  native / "evaluation/evaluate_rollout.py")
    for key, name in (("rollout_receipt", "rollout-01/receipt.json"),
                      ("rollout_outer", "rollout-01-supervision/receipt.json"),
                      ("rollout_operation", "rollout-01-outer-spec.json"),
                      ("rollout_worker_spec", "rollout-01-spec.json")):
        descriptor(espec["bindings"][key], root / name)
    report_path = root / "evaluate-01/evaluation.json"
    report = load(report_path)
    require(evaluation["arm"] == "direct_all_active" and evaluation["raw_frames"] == 154 and
            evaluation["included_frames"] == 150 and evaluation["model_calls"] == 0 and
            evaluation["final_test_access"] is False and evaluation["bindings"] == espec["bindings"] and
            evaluation["evaluation_sha256"] == inputs[report_path] and
            state["score"] == dict(mean=report["mean"], parents=report["parents"]), "saved evaluation binding")
    copy(report_path, "operations/evaluate-01/evaluation.json")
    references = {}
    require(set(fit_spec["development_comparison_references"]) == {"native", "gt_memory"}, "fixed references")
    for arm, entry in fit_spec["development_comparison_references"].items():
        path = Path(entry["path"]); references[arm] = load(path)
        require(inputs[path] == entry["sha256"], "development reference changed")
    score = summary.accuracy(report)
    reference_scores = {arm: summary.accuracy(value) for arm, value in references.items()}
    passes_mean = report["mean"]["JF"] > references["gt_memory"]["mean"]["JF"]
    native_parents = {p["parent_id"]: p["mean"]["JF"] for p in references["native"]["parents"]}
    passes_parents = all(p["mean"]["JF"] > native_parents[p["parent_id"]] for p in report["parents"])
    decision = dict(exceeds_gt_control_mean=passes_mean, both_parents_exceed_native=passes_parents,
        action="replicate_at_seed1" if passes_mean and passes_parents else "close_exact_channel_affine_family",
        final_evidence_gate_met=False)
    require(state["development_decision"] == decision, "fixed decision differs from bound reference files")
    prior = native / "publication-direct-complete-01"
    prior_manifest, prior_summary = load(prior / "manifest.json"), load(prior / "all-arms-summary.json")
    require(inputs[prior / "manifest.json"] == "3732238f1c5434ce2a5e85f044c7a5ec25b4d7ff39ae7f304e6df9027bfb3e0c" and
            next(r for r in prior_manifest["entries"] if r["path"] == "all-arms-summary.json")["sha256"] ==
            inputs[prior / "all-arms-summary.json"] and prior_summary["training_budget"]["total_seconds"] == BUDGET,
            "prior bundle/budget changed")
    comparisons = dict(prior_summary["comparisons"])
    require(set(comparisons) == {"native", "multistep", "one_step", "gt_memory",
                                "direct-cost_matched", "direct-update_matched"} and all(
                comparisons[a]["saved_accuracy"] == reference_scores[a] for a in references), "reference metric consistency")
    comparisons["direct_all_active"] = dict(saved_accuracy=score,
        versus_native=summary.differences(score, reference_scores["native"]))
    generated("summary.json", dict(schema="evr0030.active-write-publication-summary.v1",
        comparisons=comparisons, development_decision=decision, rho=.15, seed=0,
        actual_operations=operations, new_outer_seconds=math.fsum(r["outer_wall_seconds"] for r in operations),
        complete_training_budget_seconds=BUDGET, final_checkpoint=checkpoint, completed_updates=len(fit["rows"]),
        versus_gt_control=summary.differences(score, reference_scores["gt_memory"]),
        training_diagnostics_scope="All fit losses/checkpoint hashes retained in complete receipt; not accuracy.",
        binary_contents_rehashed=False), "Saved scalar comparisons and fixed decision replay; no J/F recomputation.")
    generated("earlier-evidence-reference.json", dict(bundle=str(prior),
        manifest_sha256=inputs[prior / "manifest.json"], summary_sha256=inputs[prior / "all-arms-summary.json"],
        decision_references=fit_spec["development_comparison_references"],
        initial_twelve_operations_outer_seconds=prior_summary["initial_twelve_operations_outer_seconds"],
        prior_seven_operations_outer_seconds=prior_summary["new_seven_operations_outer_seconds"],
        historical_cache_and_evaluation_costs=prior_summary["historical_cache_and_evaluation_costs"],
        failed_evaluator_attempt_reference="Initial bundle baseline-bundle-reference.json, transitively bound by prior bundle"),
        "Prior publication and bound reference evaluation identities; original operations/failed attempt not duplicated.")
    for source, expected in source_map.items():
        path = Path(source); copy(path, "sources/" + path.relative_to(native).as_posix())
        require(inputs[path] == expected, "executed source changed")
    for path, relative in ((state_path, "continuation-completed.json"), (root / "finish.py", "sources/active-write-diagnostic/finish.py"),
                           (native / "campaign/continue_campaign.py", "sources/campaign/continue_campaign.py"),
                           (helper, "reporting/summarize_receipts.py"), (Path(__file__).resolve(), "reporting/publish.py")):
        copy(path, relative)
    table = "\n".join(f"| {arm} | {v['saved_accuracy']['mean_points']['J']:.7f} | "
        f"{v['saved_accuracy']['mean_points']['F']:.7f} | {v['saved_accuracy']['mean_points']['JF']:.7f} | "
        f"{v['versus_native']['mean_delta_pp']['JF']:+.7f} | "
        f"{v['versus_native']['parent_delta_pp']['cat-girl']['JF']:+.7f} | "
        f"{v['versus_native']['parent_delta_pp']['upside-down']['JF']:+.7f} |" for arm, v in comparisons.items())
    text = f"""# All-active spatial-write direct development result

All three operations and their continuation completed clean before export. Only
the final completed checkpoint was rolled out. rho=.15, seed=0; all six parents
remain permanently development. No additional checkpoint selection or metric run.

| Arm | J % | F % | JF % | ΔJF pp vs native | cat-girl ΔJF pp | upside-down ΔJF pp |
|---|---:|---:|---:|---:|---:|---:|
{table}

Saved fixed decision: **{decision['action']}**. Exceeds the bound GT-control mean:
{passes_mean}; both parents exceed native: {passes_parents}. Exact reference-file
values, rather than a rounded admission literal, determine this decision.
This is a development continuation rule, not a paper claim or independent-seed
replication. The two-independent-held-out-source +3 J&F-point gate remains unmet.

The fit completed {len(fit['rows'])} updates. Its complete outer cost is
{operations[0]['outer_wall_seconds']:.9f}s against {BUDGET:.9f}s (teacher plus
multistep regression). Rollout/evaluation outer costs are
{operations[1]['outer_wall_seconds']:.9f}s/{operations[2]['outer_wall_seconds']:.9f}s.
All inner timers overlap these outer costs; do not add them again.
All per-step losses, active write sets and checkpoint hashes are retained in
the fit receipt. They are TRAIN diagnostics, not segmentation accuracy.

This variant differentiates through all consumed noninitial spatial writes,
with complete fixed baseline histories and unchanged pointers. It is not
recurrent BPTT. More causal write contexts receiving gradients are part of the
intervention; a change is not uniquely attributable to one mechanism.
The causal rollout is scored on 150 interior frames of 154 raw frames, one target
per parent, equal-parent DEV J/F. It is not official all-object held-out DAVIS.

Earlier six-arm results, all earlier costs and the original failed evaluator
attempt are preserved by the reference bundle; they are not copied or discarded.
New source/scope/decision documents are separately byte-bound, not claimed to
have appeared in the older fit integration commit's CI.

Only complete UTF-8 receipts, specs, logs, source and saved metric reports are
published. No PT/NPZ/model/media/GT payload is opened or copied. Checkpoint identity
is checked from bound descriptors; binary rehashes are inherited from the clean
dispatcher/rollout/evaluator, not independently repeated by this publisher.
Manifest original hashes address raw bytes; public hashes address the exact
ordered literal task-root→$TASK and home→$USER_HOME substitutions. No copied
JSON field is deleted. Scalar subtraction/decision replay is the only arithmetic.
"""
    files["README.md"] = (text.encode(), None, "Rendered saved metrics, costs, fixed decision and declared limitations.")
    aliases = [(str(task), "$TASK"), (str(Path.home()), "$USER_HOME")]

    def alias(value):
        for old, new in aliases:
            value = value.replace(old, new)
        return value

    for path, expected in inputs.items():
        require(digest(path.read_bytes()) == expected, "input changed before publication")
    prepared, entries = {}, []
    for relative, (raw, origin, derivation) in files.items():
        data = alias(raw.decode("utf-8")).encode()
        require(not re.search(rb"/Users/[A-Za-z0-9_.-]+/|/(?:private/)?var/folders/[A-Za-z0-9]+/", data),
                "private path remains")
        prepared[relative] = data
        row = dict(path=relative, bytes=len(data), sha256=digest(data))
        if origin:
            row.update(origin=dict(path=alias(str(origin)), bytes=len(raw), sha256=digest(raw)),
                       transformation="byte-identical" if raw == data else "ordered literal prefix replacement only")
        else:
            row["derivation"] = derivation
        entries.append(row)
    manifest = dict(schema="evr0030.active-write-publication.v1",
        created_utc=datetime.now(timezone.utc).isoformat(), entries=entries, completed_operations=3,
        inputs_sha256={alias(str(p)): h for p, h in inputs.items()},
        substitutions=["literal task root → $TASK", "then literal current user home → $USER_HOME"],
        original_json_fields_preserved=True, no_scientific_payloads_opened=True)
    output.mkdir(exist_ok=False)
    for relative, data in prepared.items():
        path = output / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for path, expected in inputs.items():
        require(digest(path.read_bytes()) == expected, "input changed during publication")
    print(json.dumps(dict(output=str(output), files=len(prepared) + 1,
        bytes=sum(map(len, prepared.values())) + (output / "manifest.json").stat().st_size,
        manifest_sha256=digest((output / "manifest.json").read_bytes()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
