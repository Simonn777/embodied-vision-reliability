"""Stage completed 0030 receipt evidence; stdlib JSON/text only, no model/media."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import summarize_receipts as summary


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(root, destination):
    root, destination = root.resolve(), destination.resolve()
    task = root.parents[1]
    campaign = root/"campaign"
    substitutions = [(str(task), "$TASK"), (str(Path.home()), "$USER_HOME")]
    destination.mkdir(exist_ok=False)
    entries, origins = [], {}

    def alias(text):
        for old, new in substitutions:
            text = text.replace(old, new)
        return text

    def put(relative, raw, origin=None, derivation=None):
        text = raw.decode("utf-8")
        data = alias(text).encode("utf-8")
        if re.search(r"/Users/[A-Za-z0-9_.-]+/|/(?:private/)?var/folders/[A-Za-z0-9]+/", data.decode()):
            raise ValueError("unaliased private path in " + relative)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        row = dict(path=relative, bytes=len(data), sha256=sha(data),
                   transformation="ordered literal prefix substitution only" if data != raw else "byte-identical")
        if origin is not None:
            row["origin"] = dict(path=alias(str(origin)), bytes=len(raw), sha256=sha(raw))
            origins[origin] = sha(raw)
        else:
            row["derivation"] = derivation
        entries.append(row)

    def copy(path, relative):
        if path.suffix not in (".json", ".md", ".py", ".log"):
            raise ValueError("nontext file not allowed")
        put(relative, path.read_bytes(), path)

    def generated(relative, obj, derivation):
        put(relative, (json.dumps(obj, indent=2, allow_nan=False)+"\n").encode(), derivation=derivation)

    snapshot = summary.summarize(root)
    required = [
        "teacher-rho015-01", "fit-multistep-rho015-seed0-01",
        "fit-one_step-rho015-seed0-01", "fit-gt_memory-rho015-seed0-01",
        "rollout-native-rho015-seed0-01", "evaluate-native-rho015-seed0-01",
    ]
    attempts = {Path(r["id"]).parent.name: r for r in snapshot["attempts"]}
    for name in required:
        if not attempts.get(name, {}).get("clean_completed"):
            raise ValueError("required operation not complete: " + name)
    selected, learned_arms = required.copy(), []
    for arm in ("multistep", "one_step", "gt_memory"):
        optional = [f"rollout-{arm}-rho015-seed0-01", f"evaluate-{arm}-rho015-seed0-01"]
        if all(attempts.get(n, {}).get("clean_completed") for n in optional):
            selected.extend(optional)
            learned_arms.append(arm)
    source_maps = {}
    operation_map = {}
    for path in campaign.glob("*.json"):
        obj, _ = summary.read(path)
        if isinstance(obj, dict) and "operation_output" in obj and "result_file" in obj:
            operation_map[str(Path(obj["result_file"]).resolve())] = (path, obj)
    for name in selected:
        row = attempts[name]
        receipt_path = campaign/name/"receipt.json"
        receipt, receipt_sha = summary.read(receipt_path)
        if receipt["status"] != "completed":
            raise ValueError("receipt changed to incomplete")
        worker_path = Path(row["worker_spec"])
        worker, worker_sha = summary.read(worker_path)
        operation_path, operation = operation_map[str(receipt_path)]
        outer_path = Path(row["outer_receipt"])
        outer, _ = summary.read(outer_path)
        problems = summary.clean_outer(outer, receipt_sha, sha(operation_path.read_bytes()))
        if problems or receipt["spec_sha256"] != worker_sha:
            raise ValueError("completion binding: " + str(problems))
        source_maps[name] = receipt["source_sha256"]
        copy(receipt_path, f"operations/{name}/receipt.json")
        copy(worker_path, f"operations/{name}/worker-spec.json")
        copy(operation_path, f"operations/{name}/outer-spec.json")
        copy(outer_path, f"operations/{name}/supervision/receipt.json")
        for log, expected in outer["log_sha256"].items():
            path = outer_path.parent/log
            if sha(path.read_bytes()) != expected:
                raise ValueError("log digest mismatch")
            copy(path, f"operations/{name}/supervision/{log}")
        if row["stage"] == "evaluation":
            path = receipt_path.parent/"evaluation.json"
            if sha(path.read_bytes()) != receipt["evaluation_sha256"]:
                raise ValueError("evaluation digest mismatch")
            copy(path, f"operations/{name}/evaluation.json")
    copied_summary = [attempts[n] for n in selected]
    valid = [r for r in copied_summary if "accuracy" in r]
    generated("receipts-summary.json", dict(
        schema="evr0030.first-configuration-receipts-summary.v1",
        snapshot_utc=snapshot["created_utc"], attempts=copied_summary,
        paired_dev_differences=[
            dict(left=a["id"], right=b["id"], **summary.differences(a["accuracy"], b["accuracy"]))
            for i, a in enumerate(valid) for b in valid[:i]],
        executed_outer_wall_seconds=math_sum(r["outer_wall_seconds"] for r in copied_summary),
        pending_arms=(
            "Direct training/rollout controls are outside this completed snapshot."
            if len(learned_arms) == 3 else
            "Other learned rollouts/direct controls are outside this completed snapshot."),
        limitations=snapshot["limitations"]),
        "Selected complete attempts from summarize_receipts.summarize; all scalar statistics and differences use saved JSON only.")
    generated("source-bindings.json", source_maps,
              "Exact source_sha256 maps copied structurally from each included original inner receipt; no model payload hashed/read.")
    baseline = root/"publication-baseline"
    baseline_summary, bsha = summary.read(baseline/"summary.json")
    bmanifest = sha((baseline/"bindings.json").read_bytes())
    generated("baseline-bundle-reference.json", dict(
        bundle="$TASK/work/0030-memory-content/publication-baseline",
        original_summary_sha256=bsha, original_bindings_sha256=bmanifest,
        attempts=baseline_summary["attempts"], cache_cost=baseline_summary["cache_cost"],
        note="Existing bundle retains both original evaluation attempts, including failure 01 and all its cost. Not recopied here."),
        "Constructed reference: attempts and cache_cost copied from publication-baseline/summary.json; SHA fields bind that file and bindings.json.")
    origins[baseline/"summary.json"] = bsha
    origins[baseline/"bindings.json"] = bmanifest
    for path, relative in [
        (campaign/"integration-provenance-01.md", "docs/integration-provenance-01.md"),
        (root/"future-memory-write-prior-update.md", "docs/future-memory-write-prior-update.md"),
        (root/"optimization-prior-note.md", "docs/optimization-prior-note.md"),
        (Path(summary.__file__), "reporting/summarize_receipts.py"),
        (Path(__file__), "reporting/prepare_first_config.py"),
    ]:
        copy(path, relative)
    teacher = attempts[required[0]]["training_diagnostics"]
    native, _ = summary.read(campaign/"rollout-native-rho015-seed0-01/receipt.json")
    metric = attempts["evaluate-native-rho015-seed0-01"]["accuracy"]["mean_points"]
    table = "\n".join(f"| {n} | {attempts[n]['outer_wall_seconds']:.6f} |" for n in selected)
    learned = "No learned-arm DEV accuracy is included. Teacher target fitting does not establish segmentation improvement."
    if learned_arms:
        base = attempts["evaluate-native-rho015-seed0-01"]["accuracy"]
        descriptions = []
        for arm in learned_arms:
            actual = attempts[f"evaluate-{arm}-rho015-seed0-01"]["accuracy"]
            delta = summary.differences(actual, base)
            means = actual["mean_points"]
            descriptions.append(
                f"The completed {arm} head's own rollout has saved J/F/J&F "
                f"{means['J']:.7f}/{means['F']:.7f}/{means['JF']:.7f}%, "
                f"a J&F difference of {delta['mean_delta_pp']['JF']:+.7f} percentage points versus native. "
                f"Parent J&F differences are cat-girl {delta['parent_delta_pp']['cat-girl']['JF']:+.7f} "
                f"and upside-down {delta['parent_delta_pp']['upside-down']['JF']:+.7f} points.")
        if "gt_memory" in learned_arms:
            conclusion = (
                "\n\nMultistep and one-step show no gain. The GT-memory head has a modest positive "
                "DEV result; it does not demonstrate a benefit from multistep future optimization "
                "and does not meet the paper gate. Direct controls are outside this snapshot; "
                "this does not close the whole development campaign.")
        elif len(learned_arms) == 2:
            conclusion = (
                "\n\nBoth completed learned arms show no gain. GT-memory rollout and direct controls "
                "are outside this snapshot; this does not close the whole development campaign.")
        else:
            conclusion = (
                "\n\nThis first learned result shows no gain. Other learned rollouts and direct controls "
                "are outside this snapshot; this does not close the whole development campaign.")
        learned = "\n\n".join(descriptions) + conclusion
    readme = f"""# First 0030 development configuration: completed evidence snapshot

rho=0.15, seed=0. All recordings are permanently exposed development data.
The teacher completed all 48 TRAIN anchors. Its mean fixed-context surrogate
loss was {teacher['baseline_loss']['mean']:.9f} before correction and
{teacher['selected_loss']['mean']:.9f} at its selected state. Selection used TRAIN
loss; this is an optimization diagnostic, not accuracy or generalization.
All 48 selected writes changed after BF16 storage. The three KD heads completed
288 regression updates each; their objectives are target-fitting diagnostics.

The native causal rollout completed {native['frames']} DEV frames with
{native['zero_parity_checks']} exact noninitial native parity checks and no learned writes.
Saved equal-parent J/F/J&F are {metric['J']:.7f}/{metric['F']:.7f}/{metric['JF']:.7f}%.
This covers initial target ID 1 only, excluding first/final frames from the
means: 150 included of 154 raw frames. It is not an official all-object result.
{learned}

| Complete operation | Actual outer wall seconds |
|---|---:|
{table}

These are actual executed operation costs. Inner/anchor timers overlap and must
not be added. Shared baseline cache cost and the original failed evaluation are
referenced in `baseline-bundle-reference.json`; failures are not removed from
development accounting. Complete same-information and cost-matched controls,
further development, and the unchanged two-source held-out +3 J&F gate remain
outstanding.

Each included operation has its original inner receipt, worker/outer specs,
outer receipt and raw logs, with only the literal path-prefix substitutions
declared in `manifest.json`. Original SHA fields are retained unchanged and
refer to original bytes; a redacted public file has its own manifest SHA.
No JSON fields are removed. Source maps are recorded per operation; the
integration note explains the inherited historical expected_commit field.

No PT/NPZ, checkpoint, model, RGB, mask or annotation file is included.
No started receipt is copied. The saved-summary tool does not recalculate
metrics or rerun models. This snapshot claims no novelty or method efficacy.
"""
    put("README.md", readme.encode(), derivation="Rendered completed receipt statistics/costs and fixed development-scope description.")
    for path, expected in origins.items():
        if sha(path.read_bytes()) != expected:
            raise ValueError("source changed during export: " + str(path))
    manifest = dict(schema="evr0030.first-configuration-publication.v1",
        created_utc=datetime.now(timezone.utc).isoformat(), entries=entries,
        literal_substitutions=[
            dict(source_prefix="task workspace root = resolved --root parent.parent", replacement="$TASK"),
            dict(source_prefix="current user's absolute home directory", replacement="$USER_HOME")],
        substitution_order="task root first, then home; literal UTF-8 replacement, no JSON reserialization for copied files",
        source_revalidation="Every copied origin plus referenced baseline summary/bindings rehashed unchanged after staging",
        copied_operation_names=selected, no_models_arrays_or_gt_decoded=True)
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    print(json.dumps(dict(output=str(destination), files=len(files),
                         bytes=sum(p.stat().st_size for p in files),
                         manifest_sha256=sha((destination/"manifest.json").read_bytes()),
                         included_operations=selected)))


def math_sum(values):
    import math
    return math.fsum(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
