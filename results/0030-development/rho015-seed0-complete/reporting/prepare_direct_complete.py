"""Publish seven completed saved-JSON operations, never run science.

Refuses to create the fresh output until both direct fit/rollout/evaluation
triples, TRAIN fixed-fit diagnostic and continuation state are complete.
Prior twelve-operation evidence is referenced, not copied. Stdlib only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import summarize_receipts as summary


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(root, output, initial):
    root, output, initial = root.resolve(), output.resolve(), initial.resolve()
    task, campaign, analysis = root.parents[1], root/"campaign", root/"analysis"
    inputs, files, operation_rows = {}, {}, []

    def remember(path, raw):
        digest = sha(raw)
        require(path not in inputs or inputs[path] == digest, "input changed between reads")
        inputs[path] = digest
        return digest

    def load(path):
        raw = path.read_bytes()
        value, digest = json.loads(raw), remember(path, raw)
        return value, raw, digest

    def copy(path, relative):
        require(path.suffix in (".json", ".log", ".py", ".md"), "text only")
        raw = path.read_bytes()
        remember(path, raw)
        files[relative] = (raw, path, None)

    def generated(relative, value, derivation):
        files[relative] = ((json.dumps(value, indent=2, allow_nan=False)+"\n").encode(),
                           None, derivation)

    state_path = campaign/"continuation-02.json"
    state, _, state_sha = load(state_path)
    require(state["status"] == "completed" and
            all(r["status"] == "completed" for r in state["operations"]),
            "continuation is incomplete; no publication directory created")
    state_rows = {r["name"]: r for r in state["operations"]}

    def operation(name, spec_path, outer_spec_path, in_continuation=True):
        spec, _, spec_sha = load(spec_path)
        outer_spec, _, outer_spec_sha = load(outer_spec_path)
        receipt_path = Path(spec["output"])/"receipt.json"
        receipt, _, receipt_sha = load(receipt_path)
        outer_path = Path(outer_spec["operation_output"])/"receipt.json"
        outer, _, outer_sha = load(outer_path)
        require(receipt["status"] == "completed", name + " inner incomplete")
        require(not summary.clean_outer(outer, receipt_sha, outer_spec_sha),
                name + " outer incomplete/unclean or hash mismatch")
        require(outer["schema"] == "evr0029.exit-checked-operation.v1" and
                outer["supervisor_sha256"] == "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8"
                and outer["monitor_sha256"] == "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"
                and math.isfinite(outer["wall_seconds"]) and
                0 <= outer["wall_seconds"] <= outer_spec["timeout"] <= 1800
                and 0 <= outer["peak_group_rss_kib"] <= 8388608, name + " outer envelope")
        require(receipt["spec_sha256"] == spec_sha and
                receipt["source_sha256"] == spec["source_sha256"] and
                Path(outer_spec["result_file"]) == receipt_path and
                Path(outer_spec["command"][-1]) == spec_path, name + " spec binding")
        if in_continuation:
            s = state_rows[name]
            require((s["spec_sha256"], s["outer_spec_sha256"], s["result_sha256"]) ==
                    (spec_sha, outer_spec_sha, receipt_sha), name + " continuation binding")
        stem = "operations/"+name
        for path, suffix in ((receipt_path, "receipt.json"), (spec_path, "worker-spec.json"),
                             (outer_spec_path, "outer-spec.json"),
                             (outer_path, "supervision/receipt.json")):
            copy(path, stem+"/"+suffix)
        require(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "log inventory")
        for filename, expected in outer["log_sha256"].items():
            path = outer_path.parent/filename
            require(sha(path.read_bytes()) == expected, "log changed")
            copy(path, stem+"/supervision/"+filename)
        row = dict(name=name, status="completed", inner_sha256=receipt_sha,
                   outer_sha256=outer_sha, source_sha256=receipt["source_sha256"],
                   inner_wall_seconds=receipt.get("wall_seconds"),
                   outer_wall_seconds=outer["wall_seconds"],
                   peak_group_rss_kib=outer["peak_group_rss_kib"])
        operation_rows.append(row)
        return spec, receipt, row

    prior_manifest, _, prior_manifest_sha = load(initial/"manifest.json")
    prior, prior_raw, prior_sha = load(initial/"receipts-summary.json")
    historical, historical_raw, historical_sha = load(initial/"baseline-bundle-reference.json")
    old_entries = {r["path"]: r for r in prior_manifest["entries"]}
    require(old_entries["receipts-summary.json"]["sha256"] == sha(prior_raw) and
            old_entries["baseline-bundle-reference.json"]["sha256"] == sha(historical_raw),
            "prior reference changed")
    old = {Path(r["id"]).parent.name: r for r in prior["attempts"]}
    require(len(old) == 12 and all(r["clean_completed"] for r in old.values()),
            "complete initial twelve-operation reference required")
    teacher_seconds = old["teacher-rho015-01"]["outer_wall_seconds"]
    regression_seconds = old["fit-multistep-rho015-seed0-01"]["outer_wall_seconds"]
    budget = teacher_seconds + regression_seconds
    require(state["direct_complete_cost_budget_seconds"] == budget, "budget reference mismatch")
    scores = {arm: old[f"evaluate-{arm}-rho015-seed0-01"]["accuracy"]
              for arm in ("native", "multistep", "one_step", "gt_memory")}
    direct_costs, direct_fits = {}, {}
    for mode in ("cost_matched", "update_matched"):
        arm = "direct-"+mode
        fit_name = "fit-"+arm+"-rho015-seed0-01"
        spec, fit, cost = operation(fit_name, campaign/(fit_name+"-spec.json"),
                                    campaign/(fit_name+"-outer-spec.json"))
        require(spec["mode"] == "direct" and spec["direct_comparison"] == mode and
                spec["rho"] == .15 and spec["seed"] == 0 and
                spec["complete_cost_budget_seconds"] == budget, "direct scope")
        steps = fit["checkpoints"][-1]["step"]
        require(steps > 0 and (mode != "update_matched" or steps == 288), "direct updates incomplete")
        within_budget = cost["outer_wall_seconds"] <= budget
        require(mode != "cost_matched" or within_budget, "cost-matched complete outer exceeds budget")
        cost.update(completed_updates=steps, training_budget_seconds=budget,
                    within_training_budget=within_budget)
        direct_costs[arm] = cost
        direct_fits[mode] = fit
        name = "rollout-"+arm+"-rho015-seed0-01"
        rspec, rollout, _ = operation(name, campaign/(name+"-spec.json"),
                                      campaign/(name+"-outer-spec.json"))
        require(rspec["arm"] == rollout["arm"] == arm and rollout["frames"] == 154 and
                rspec["checkpoint"]["sha256"] == fit["checkpoints"][-1]["sha256"],
                "direct rollout/checkpoint incomplete")
        name = "evaluate-"+arm+"-rho015-seed0-01"
        _, evaluation, _ = operation(name, campaign/(name+"-spec.json"),
                                     campaign/(name+"-outer-spec.json"))
        require(evaluation["arm"] == arm and evaluation["raw_frames"] == 154 and
                evaluation["included_frames"] == 150, "complete direct evaluation required")
        path = campaign/name/"evaluation.json"
        report, _, report_sha = load(path)
        require(report_sha == evaluation["evaluation_sha256"], "evaluation hash")
        scores[arm] = summary.accuracy(report)
        copy(path, "operations/"+name+"/evaluation.json")

    cost_fit, update_fit = direct_fits["cost_matched"], direct_fits["update_matched"]
    shorter = min(direct_costs["direct-cost_matched"]["completed_updates"],
                  direct_costs["direct-update_matched"]["completed_updates"])
    for fit in (cost_fit, update_fit):
        require([r["step"] for r in fit["rows"]] == list(range(1, len(fit["rows"])+1))
                and len(fit["rows"]) == fit["checkpoints"][-1]["step"], "direct loss row coverage")
    loss_mismatches = [a["step"] for a, b in zip(cost_fit["rows"][:shorter], update_fit["rows"][:shorter])
                       if a["loss"] != b["loss"]]
    identity_mismatches = [
        a["step"] for a, b in zip(cost_fit["rows"][:shorter], update_fit["rows"][:shorter])
        if any(a[k] != b[k] for k in ("epoch", "sequence", "anchor"))]
    checkpoint_pairs = [
        dict(step=a["step"], cost_checkpoint=a["path"], update_checkpoint=b["path"],
             cost_sha256=a["sha256"], update_sha256=b["sha256"],
             exactly_equal=a["sha256"] == b["sha256"])
        for a in cost_fit["checkpoints"] for b in update_fit["checkpoints"]
        if a["step"] == b["step"]]
    prefix = dict(
        scope="Deterministic same-seed prefix consistency, NOT independent-seed replication; saved receipt fields only",
        compared_loss_steps=shorter, loss_prefix_exactly_equal=not loss_mismatches,
        loss_mismatch_steps=loss_mismatches, identity_mismatch_steps=identity_mismatches,
        common_checkpoint_pairs=checkpoint_pairs,
        all_common_checkpoint_hashes_equal=all(r["exactly_equal"] for r in checkpoint_pairs)
            if checkpoint_pairs else None,
        different_step_final_checkpoints_compared=False)

    fixed_spec, fixed, _ = operation("fixed-fit-01", analysis/"fixed-fit-spec-01.json",
                                     analysis/"fixed-fit-outer-spec-01.json", False)
    require(fixed["schema"] == "evr0030.fixed-checkpoint-fit.v1" and
            fixed["train_only"] and not fixed["dev_access"] and not fixed["final_test_access"] and
            fixed["sam2_calls"] == fixed["fitting_updates"] == 0 and
            fixed["head_forward_calls"] == 144 and fixed["target_count"] == 48 and
            [a["arm"] for a in fixed["arms"]] == ["multistep", "one_step", "gt_memory"] and
            all(len(a["rows"]) == 48 for a in fixed["arms"]), "fixed TRAIN-fit incomplete")
    copy(analysis/"fixed_checkpoint_error.py", "reporting/fixed_checkpoint_error.py")
    require(inputs[analysis/"fixed_checkpoint_error.py"] ==
            fixed_spec["source_sha256"][str(analysis/"fixed_checkpoint_error.py")], "fixed-fit source changed")
    copy(state_path, "continuation-completed.json")
    copy(campaign/"integration-provenance-01.md", "docs/integration-provenance-01.md")
    copy(Path(summary.__file__).resolve(), "reporting/summarize_receipts.py")
    copy(Path(__file__).resolve(), "reporting/prepare_direct_complete.py")
    comparisons = {arm: dict(saved_accuracy=scores[arm],
                             versus_native=summary.differences(scores[arm], scores["native"]))
                   for arm in scores}
    generated("all-arms-summary.json", dict(
        schema="evr0030.completed-direct-comparison.v1", rho=.15, seed=0,
        comparisons=comparisons, operations=operation_rows,
        same_seed_prefix_consistency=prefix,
        training_budget=dict(teacher_seconds=teacher_seconds,
                             multistep_regression_seconds=regression_seconds,
                             total_seconds=budget, direct=direct_costs),
        fixed_train_fit={a["arm"]: dict(equal_parent=a["equal_parent"], parents=a["parents"])
                         for a in fixed["arms"]},
        new_seven_operations_outer_seconds=math.fsum(r["outer_wall_seconds"] for r in operation_rows),
        initial_twelve_operations_outer_seconds=prior["executed_outer_wall_seconds"],
        historical_cache_and_evaluation_costs=historical,
        limits=["Saved scalar subtraction only; no mask/GT metric recomputation.",
                "TRAIN fit and optimization diagnostics are not segmentation accuracy.",
                "Training budget is teacher plus multistep regression, not a claim of identical end-to-end rollout costs.",
                "All outer costs are separate actual executions; inner durations must not be added.",
                "Initial cache/failure costs are referenced separately, not silently discarded."]),
        "Saved metrics from prior summary plus two completed evaluations; scalar differences only. Fixed-fit summaries copied from completed diagnostic receipt.")
    generated("initial-bundle-reference.json", dict(
        bundle=str(initial), manifest_sha256=prior_manifest_sha, summary_sha256=prior_sha,
        baseline_failure_reference_sha256=historical_sha,
        referenced_operations=list(old), original_failed_evaluator_preserved=True),
        "Reference to existing initial twelve-operation bundle and its baseline-bundle reference; no original operation files duplicated.")
    table = "\n".join(
        f"| {a} | {v['mean_points']['J']:.7f} | {v['mean_points']['F']:.7f} | "
        f"{v['mean_points']['JF']:.7f} | "
        f"{comparisons[a]['versus_native']['mean_delta_pp']['JF']:+.7f} | "
        f"{comparisons[a]['versus_native']['parent_delta_pp']['cat-girl']['JF']:+.7f} | "
        f"{comparisons[a]['versus_native']['parent_delta_pp']['upside-down']['JF']:+.7f} |"
        for a, v in scores.items())
    cost_table = "\n".join(f"| {r['name']} | {r['outer_wall_seconds']:.6f} |"
                           for r in operation_rows)
    text = f"""# Completed first-configuration direct controls and TRAIN-fit diagnostic

rho=0.15 / seed=0; all six parent recordings remain permanently development.
Both direct fit/rollout/evaluation triples, the continuation state and the
TRAIN-only fixed-fit diagnostic completed clean before this directory was created.

| Arm | J (%) | F (%) | J&F (%) | ΔJF pp vs native | cat-girl ΔJF pp | upside-down ΔJF pp |
|---|---:|---:|---:|---:|---:|---:|
{table}

Metrics are saved single-target DEV values, with 150 interior frames of 154
raw frames, equal-parent aggregation. No metrics were recomputed during export.
The table is not an official all-object benchmark or held-out paper result.
The unchanged final two-independent-source +3 J&F gate remains outstanding.

The measured teacher-plus-multistep-regression training budget is
{teacher_seconds:.9f} + {regression_seconds:.9f} = {budget:.9f} seconds.
Cost-matched direct used {direct_costs['direct-cost_matched']['completed_updates']} complete updates
and {direct_costs['direct-cost_matched']['outer_wall_seconds']:.9f} seconds
(within budget). Update-matched direct completed 288 updates and used
{direct_costs['direct-update_matched']['outer_wall_seconds']:.9f} seconds.
These comparisons do not omit rollout/evaluation costs:

| New complete operation | Actual outer wall seconds |
|---|---:|
{cost_table}

Same-seed prefix check from saved receipt fields: the first {shorter} losses
are exactly equal: {prefix['loss_prefix_exactly_equal']}; sample-identity mismatch
count: {len(identity_mismatches)}. Common checkpoint steps are
{sorted({r['step'] for r in checkpoint_pairs})}; all corresponding saved SHA-256
hashes equal: {prefix['all_common_checkpoint_hashes_equal']}. This is deterministic
prefix consistency, **not independent-seed replication**. Checkpoint tensors
were not loaded; final checkpoints at different step counts are not compared.
All mismatch indices and comparison pairs, if any, remain in the summary.

Fixed-fit reevaluates the three final heads against all 48 TRAIN targets:
144 head forwards, no SAM2 call, no fitting update or DEV label access. Its
saved E/E0 and direction summaries are training diagnostics, not accuracy,
and are retained in full rather than substituted for rollout results.

The initial twelve operations and original failed baseline evaluation are
hash-bound references in `initial-bundle-reference.json`; they are not recopied.
Shared cache and historical failure costs remain in `all-arms-summary.json`.
Inner/anchor timers overlap with outer time and must not be added.

Only text receipts/specs/logs/source maps and reporting sources are exported.
No PT/NPZ, model, RGB, mask or GT payload is copied. Original SHA fields continue
to refer to original bytes; public hashes record the literal path redactions.
No completed receipt field is deleted. This package asserts no novelty.
"""
    files["README.md"] = (text.encode(), None, "Rendered saved comparisons, complete-operation costs and scope.")
    # All seven and all referenced state/source bindings are checked before mutation.
    for path, expected in inputs.items():
        require(sha(path.read_bytes()) == expected, "input changed before export")
    output.mkdir(exist_ok=False)
    aliases = [(str(task), "$TASK"), (str(Path.home()), "$USER_HOME")]
    def alias(text):
        for old, new in aliases:
            text = text.replace(old, new)
        return text
    entries = []
    for relative, (raw, origin, derivation) in files.items():
        data = alias(raw.decode()).encode()
        require(not re.search(rb"/Users/[A-Za-z0-9_.-]+/|/(?:private/)?var/folders/[A-Za-z0-9]+/", data),
                "private path remains")
        path = output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        row = dict(path=relative, bytes=len(data), sha256=sha(data))
        if origin:
            row.update(origin=dict(path=alias(str(origin)), bytes=len(raw), sha256=sha(raw)),
                       transformation="ordered literal prefix replacement only" if data != raw else "byte-identical")
        else:
            row["derivation"] = derivation
        entries.append(row)
    for path, expected in inputs.items():
        require(sha(path.read_bytes()) == expected, "input changed during export")
    manifest = dict(schema="evr0030.direct-complete-publication.v1",
        created_utc=datetime.now(timezone.utc).isoformat(), entries=entries,
        input_json_and_text_sha256={alias(str(p)): v for p, v in inputs.items()},
        substitutions=["literal resolved --root parent.parent → $TASK", "then literal current user home → $USER_HOME"],
        original_json_fields_preserved=True, completed_operations=7,
        continuation_sha256=state_sha, no_arrays_models_or_gt_loaded=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    payloads = [p for p in output.rglob("*") if p.is_file()]
    print(json.dumps(dict(output=str(output), files=len(payloads),
                         bytes=sum(p.stat().st_size for p in payloads),
                         manifest_sha256=sha((output/"manifest.json").read_bytes()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial-bundle", type=Path)
    args = parser.parse_args()
    run(args.root, args.output or args.root/"publication-direct-complete-01",
        args.initial_bundle or args.root/"publication-first-config-03")
