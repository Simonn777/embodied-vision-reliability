"""Summarize saved 0030 JSON only; no model/array loads or metric recomputation.

Scans direct campaign/evaluation specs and one-level receipt directories.
All attempts remain separate; never selects a best checkpoint or replaces failures.
Completed accuracy requires matching completed inner/outer receipts and saved-report
hash. This is a compact reporting tool, not a replacement provenance audit.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = ("J", "F", "JF")
DEV = {"cat-girl", "upside-down"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    if Path(path).suffix != ".json":
        raise ValueError("JSON-only input")
    data = Path(path).read_bytes()
    return json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError("nonfinite JSON constant: " + x))), digest(data)


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return dict(count=0, mean=None, median=None, min=None, max=None)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite summary values")
    return dict(count=len(values), mean=math.fsum(values)/len(values),
                median=statistics.median(values), min=min(values), max=max(values))


def teacher_summary(rows):
    for r in rows:
        if len(r["losses"]) != 6:
            raise ValueError("teacher row does not contain all six recorded states")
    return dict(
        interpretation="TRAIN fixed-context surrogate diagnostics, not accuracy; best-state loss improvement is selected on TRAIN",
        anchors_recorded=len(rows), anchors_expected=48,
        parents=dict(Counter(r["sequence"] for r in rows)),
        best_steps=dict(Counter(str(r["best_step"]) for r in rows)),
        one_steps=dict(Counter(str(r["one_step"]) for r in rows)),
        baseline_loss=stats(r["baseline_loss"] for r in rows),
        selected_loss=stats(r["selected_loss"] for r in rows),
        baseline_minus_selected=stats(r["baseline_loss"]-r["selected_loss"] for r in rows),
        one_step_minus_multistep=stats(
            r["losses"][r["one_step"]]-r["selected_loss"] for r in rows),
        changed_bf16_entries=stats(r["changed_bf16_entries"] for r in rows),
        zero_bf16_change_anchors=sum(r["changed_bf16_entries"] == 0 for r in rows),
        post_bf16_rms=stats(r["post_bf16_rms"] for r in rows),
        anchor_wall_seconds_sum=math.fsum(r["wall_seconds"] for r in rows),
        cost_note="Anchor wall sum excludes context loading before each row timer; use complete outer wall for cost.")


def clean_outer(outer, inner_hash, operation_hash):
    reasons = []
    for ok, text in (
        (outer.get("status") == "completed", "outer not completed"),
        (outer.get("exit_code") == 0, "nonzero/missing exit"),
        ("termination" in outer and outer["termination"] is None, "termination missing/set"),
        (outer.get("remaining_descendant_pids") == [], "descendant state"),
        (outer.get("cleanup_errors") == [], "cleanup state"),
        (outer.get("output_sha256") == inner_hash, "inner receipt hash mismatch"),
        (outer.get("spec_sha256") == operation_hash, "operation spec hash mismatch"),
    ):
        if not ok:
            reasons.append(text)
    if outer.get("schema") == "evr0029.exit-checked-operation.v1":
        snapshots = outer.get("cleanup_snapshots", [])
        if (outer.get("signal_attempts") != [] or outer.get("triggering_snapshot") is not None
                or (outer.get("post_exit_snapshot") or {}).get("rows") != []
                or len(snapshots) != 1 or snapshots[0].get("label") != "final"
                or snapshots[0].get("rows") != []):
            reasons.append("fresh post-exit/final telemetry not clean")
    elif outer.get("schema") != "evr0026.operation.v1":
        reasons.append("unknown outer schema")
    return reasons


def accuracy(report):
    if {p["parent_id"] for p in report["parents"]} != DEV or len(report["parents"]) != 2:
        raise ValueError("incomplete or duplicate DEV parents")
    for sequence in report["sequences"]:
        if (sequence["endpoint_policy"] != "exclude_first_and_last"
                or sequence["void_policy"] != "official_background"):
            raise ValueError("incompatible metric policy")
    result = dict(mean_points={k: 100*report["mean"][k] for k in METRICS},
                  parents={p["parent_id"]: {k: 100*p["mean"][k] for k in METRICS}
                           for p in report["parents"]},
                  scope="Saved single-target exposed-DEV J/F/JF; not final held-out evidence")
    if not all(math.isfinite(v) and 0 <= v <= 100 for d in
               [result["mean_points"], *result["parents"].values()] for v in d.values()):
        raise ValueError("invalid saved metric")
    return result


def differences(left, right):
    """Subtract already evaluated scalar means; never recompute J/F from masks."""
    return dict(mean_delta_pp={k: left["mean_points"][k]-right["mean_points"][k] for k in METRICS},
                parent_delta_pp={p: {k: left["parents"][p][k]-right["parents"][p][k]
                                     for k in METRICS} for p in sorted(DEV)})


def summarize(root):
    root = Path(root).resolve()
    directories = [root/"campaign", root/"evaluation"]
    errors, fingerprints, specs, operations = [], {}, {}, defaultdict(list)
    # Deliberately never traverse model/cache tensor folders or publication copies.
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            try:
                obj, sha = read(path)
                if isinstance(obj, dict) and "output" in obj and "source_sha256" in obj:
                    specs[str(Path(obj["output"]).resolve()/"receipt.json")] = (path, obj, sha)
                if isinstance(obj, dict) and all(k in obj for k in
                                                ("result_file", "operation_output", "command")):
                    operations[str(Path(obj["result_file"]).resolve())].append((path, obj, sha))
            except (OSError, ValueError, TypeError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
    attempts = []
    receipt_paths = {p.resolve() for d in directories for p in d.glob("*/receipt.json")}
    # Include not-yet-started specs so missing stages do not silently disappear.
    receipt_paths.update(Path(p) for p in specs)
    for path in sorted(receipt_paths):
        if str(path) not in specs:
            continue  # Outer/test receipts are referenced below, not counted again.
        spath, spec, ssha = specs[str(path)]
        row = dict(id=str(path.relative_to(root)), worker_spec=str(spath), spec_sha256=ssha,
                   arm=spec.get("arm"), rho=spec.get("rho"), seed=spec.get("seed"),
                   mode=spec.get("mode"), comparison=spec.get("direct_comparison"),
                   status="missing", clean_completed=False, issues=[])
        fingerprints[str(spath)] = ssha
        try:
            receipt, rsha = read(path)
            fingerprints[str(path)] = rsha
            row.update(status=receipt.get("status", "unknown"), schema=receipt.get("schema"),
                       error_type=receipt.get("error_type"), error=receipt.get("error"),
                       inner_wall_seconds=receipt.get("wall_seconds"),
                       source_sha256=receipt.get("source_sha256"),
                       rows_recorded=len(receipt.get("rows", [])))
            schema = receipt.get("schema", "")
            stage = ("cache" if "cache" in schema else "rollout" if "own-rollout" in schema
                     else "evaluation" if "evaluation" in schema else spec.get("mode", "unknown"))
            row["stage"] = stage
            if receipt.get("spec_sha256") != ssha:
                row["issues"].append("worker spec hash mismatch")
            choices = operations[str(path)]
            if len(choices) != 1:
                row["issues"].append("missing/ambiguous outer operation")
            else:
                opath, operation, osha = choices[0]
                outer_path = Path(operation["operation_output"])/"receipt.json"
                fingerprints[str(opath)] = osha
                if outer_path.exists():
                    outer, outer_hash = read(outer_path)
                    fingerprints[str(outer_path)] = outer_hash
                    row.update(outer_receipt=str(outer_path), outer_status=outer.get("status"),
                               outer_wall_seconds=outer.get("wall_seconds"),
                               peak_group_rss_kib=outer.get("peak_group_rss_kib"),
                               termination=outer.get("termination"),
                               outer_schema=outer.get("schema"),
                               supervisor_sha256=outer.get("supervisor_sha256"))
                    row["issues"].extend(clean_outer(outer, rsha, osha))
                    if row["outer_schema"] == "evr0026.operation.v1":
                        row["supervision_limit"] = "Legacy semantics; no fresh post-exit snapshot proof"
                else:
                    row["issues"].append("outer receipt missing")
            row["clean_completed"] = row["status"] == "completed" and not row["issues"]
            rows = receipt.get("rows", [])
            if stage == "teacher":
                row["training_diagnostics"] = teacher_summary(rows)
                row["future_forward_calls"] = receipt.get("future_forward_calls")
                row["zero_parity_checks"] = receipt.get("zero_parity_checks")
            elif stage in ("regress", "direct"):
                row["training_diagnostics"] = dict(
                    interpretation="Training objective only, not DEV accuracy",
                    completed_steps=max([r.get("steps", r.get("step", 0)) for r in rows], default=0),
                    epoch_mean_losses=[r["mean_loss"] for r in rows if "mean_loss" in r],
                    direct_losses=stats(r["loss"] for r in rows if "loss" in r),
                    checkpoint_steps=[c["step"] for c in receipt.get("checkpoints", [])])
                if spec.get("complete_cost_budget_seconds") is not None:
                    budget = spec["complete_cost_budget_seconds"]
                    row["cost_budget_seconds"] = budget
                    wall = row.get("outer_wall_seconds")
                    row["within_complete_cost_budget"] = row["clean_completed"] and wall is not None and wall <= budget
            elif stage == "rollout":
                writes = [w for r in rows for w in r.get("writes", [])]
                row["rollout_diagnostics"] = dict(
                    frames=receipt.get("frames"), changed_writes=receipt.get("changed_writes"),
                    post_bf16_rms=stats(w["post_bf16_rms"] for w in writes),
                    changed_entries=stats(w["changed_bf16_entries"] for w in writes))
            elif stage == "evaluation" and row["clean_completed"]:
                report, esha = read(path.parent/"evaluation.json")
                if esha != receipt.get("evaluation_sha256"):
                    raise ValueError("saved evaluation hash mismatch")
                fingerprints[str(path.parent/"evaluation.json")] = esha
                row["accuracy"] = accuracy(report)
                row["arm"] = receipt.get("arm", "native")
                # Retain provenance pointers without loading their model/array payloads.
                row["rollout_receipt"] = receipt.get("bindings", {}).get("rollout_receipt")
            row["dependency_paths"] = {k: spec[k] for k in
                ("targets", "cache", "checkpoint") if k in spec}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["issues"].append(str(exc))
            row["clean_completed"] = False
            row.pop("accuracy", None)
        attempts.append(row)
    valid = [r for r in attempts if "accuracy" in r]
    paired = [dict(left=a["id"], left_arm=a["arm"], right=b["id"], right_arm=b["arm"],
                   **differences(a["accuracy"], b["accuracy"]))
              for i, a in enumerate(valid) for b in valid[:i]]
    costs = {}
    for stage in sorted({r.get("stage", "unknown") for r in attempts}):
        rows = [r for r in attempts if r.get("stage", "unknown") == stage]
        measured = [r for r in rows if isinstance(r.get("outer_wall_seconds"), (int, float))]
        costs[stage] = dict(
            all_measured_attempt_wall_seconds=math.fsum(r["outer_wall_seconds"] for r in measured),
            nonclean_measured_attempt_wall_seconds=math.fsum(r["outer_wall_seconds"] for r in measured
                                                            if not r["clean_completed"]),
            measured_attempts=len(measured), unmeasured_attempts=len(rows)-len(measured))
    return dict(schema="evr0030.saved-json-summary.v1",
                created_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                script_sha256=digest(Path(__file__).read_bytes()), inputs_sha256=fingerprints,
                attempts=attempts, paired_dev_differences=paired, costs_by_stage=costs,
                status_counts=dict(Counter(r["status"] for r in attempts)), discovery_errors=errors,
                limitations=[
                    "JSON snapshot only; source/array/model/GT payloads and metrics are not re-executed.",
                    "Started receipts can advance after this snapshot; missing outer cost is unknown, not zero.",
                    "Costs sum each actual outer attempt once; inner/anchor timers are nested and are not added.",
                    "Shared cache cost is listed once as executed; this is not a per-arm complete-pipeline cost claim.",
                    "All attempts and checkpoints are retained; no best-on-DEV selection or incomplete-result accuracy.",
                    "Teacher best-state improvement is TRAIN-selected and cannot establish causal student benefit."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), attempts=len(result["attempts"]),
                          paired=len(result["paired_dev_differences"]),
                          status_counts=result["status_counts"])))


if __name__ == "__main__":
    main()
