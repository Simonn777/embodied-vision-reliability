"""Export complete saved replay text only; never opens arrays, labels or models."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(root, output):
    root, output = root.resolve(), output.resolve()
    native_root, task = root.parent, root.parent.parents[1]
    helper = native_root / "analysis/summarize_receipts.py"
    loader = importlib.util.spec_from_file_location("full_gt_saved_summary", helper)
    summary = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(summary)
    inputs, files = {}, {}

    def read(path):
        require(path.suffix in (".json", ".py", ".md", ".log"), "text-only publication")
        raw = path.read_bytes()
        found = digest(raw)
        require(path not in inputs or inputs[path] == found, "input changed between reads")
        inputs[path] = found
        return raw

    def load(path):
        return json.loads(read(path))

    def copy(path, relative):
        require(relative not in files, "duplicate export destination")
        files[relative] = (read(path), path, None)

    spec_path, outer_spec_path = root / "spec-01.json", root / "outer-spec-01.json"
    spec, operation = load(spec_path), load(outer_spec_path)
    require(output.is_relative_to(native_root) and not output.is_relative_to(root)
            and not output.exists(), "fresh private publication root outside the source/run directory")
    inner_path = Path(spec["output"]) / "receipt.json"
    outer_path = Path(operation["operation_output"]) / "receipt.json"
    inner, outer = load(inner_path), load(outer_path)
    require(inner["status"] == "completed" and
            not summary.clean_outer(outer, inputs[inner_path], inputs[outer_spec_path]),
            "complete clean replay required")
    require(outer["schema"] == "evr0029.exit-checked-operation.v1" and
            outer["supervisor_sha256"] ==
            "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8" and
            outer["monitor_sha256"] ==
            "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd" and
            math.isfinite(outer["wall_seconds"]) and 0 <= outer["wall_seconds"] <= 1800 and
            0 <= outer["peak_group_rss_kib"] <= 8388608, "outer envelope/source")
    require(inner["spec_sha256"] == inputs[spec_path] and
            inner["source_sha256"] == spec["source_sha256"] and
            Path(operation["result_file"]) == inner_path and
            operation["command"][-2:] == [str(root / "replay.py"), str(spec_path)],
            "worker/spec binding")
    require(inner["completed_anchors"] == 48 and inner["completed_parents"] == 4 and
            inner["future_forward_calls"] == inner["track_step_calls"] == 96 and
            inner["image_encoder_calls"] == inner["memory_encoder_calls"] == inner["optimization_steps"] == 0 and
            inner["projection_applied"] is False and
            inner["schema"] == "evr0030.development.full-gt-write.v1" and
            inner["gradients"] is False and inner["deployment_result"] is False and
            inner["final_test_access"] is False and not inner["offline_attempts"] and
            inner["model_initial_hash"] == inner["model_final_hash"] and
            len(inner["history_hashes"]) == 4 and
            all(r["before"] == r["after"] for r in inner["history_hashes"]), "complete frozen replay")
    report_path = inner_path.parent / "evaluation.json"
    report = load(report_path)
    saved_outputs = {r["path"]: r for r in inner["outputs"]}
    require(len(inner["outputs"]) == len(saved_outputs) == 97 and
            saved_outputs["evaluation.json"]["sha256"] == inputs[report_path] and
            saved_outputs["evaluation.json"]["bytes"] == report_path.stat().st_size,
            "saved evaluation binding")
    require(report["rows"] == inner["rows"] and len(report["rows"]) == 96 and
            [r["parent_id"] for r in report["parents"]] == spec["train_sequences"] and
            len(report["parents"]) == 4 and report["official_benchmark_result"] is False,
            "complete saved metric population")
    keys = [(r["sequence"], r["anchor"], r["horizon"]) for r in report["rows"]]
    require(len(set(keys)) == 96 and sum(r["final_frame"] for r in report["rows"]) == 4,
            "duplicate/missing anchor futures")
    for sequence in spec["train_sequences"]:
        rows = [r for r in report["rows"] if r["sequence"] == sequence]
        require(len(rows) == 24 and len({r["anchor"] for r in rows}) == 12 and
                all({r["horizon"] for r in rows if r["anchor"] == anchor} == {1, 3}
                    for anchor in {r["anchor"] for r in rows}), "full parent coverage")
    require(report["unique_future_frames"] ==
            len({(r["sequence"], r["frame"]) for r in report["rows"]}), "unique frame count")
    for path, relative in ((spec_path, "worker-spec.json"), (outer_spec_path, "outer-spec.json"),
                           (inner_path, "run/receipt.json"), (report_path, "run/evaluation.json"),
                           (outer_path, "supervision/receipt.json"),
                           (root / "ADMISSION.md", "docs/ADMISSION.md")):
        copy(path, relative)
    require(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "complete log inventory")
    for filename, expected in outer["log_sha256"].items():
        path = outer_path.parent / filename
        copy(path, "supervision/" + filename)
        require(inputs[path] == expected, "changed raw log")
    for filename, expected in spec["source_sha256"].items():
        path = Path(filename)
        relative = ("docs/SCOPE.md" if path == root / "SCOPE.md" else
                    "sources/" + path.relative_to(native_root).as_posix())
        copy(path, relative)
        require(inputs[path] == expected, "executed source changed")
    copy(native_root / "local-storage-prior.md", "docs/local-storage-prior.md")
    copy(Path(__file__).resolve(), "reporting/publish.py")
    copy(helper, "reporting/summarize_receipts.py")
    diagnostics = inner["anchor_diagnostics"]
    require(diagnostics == report["anchor_diagnostics"] and len(diagnostics) == 48 and
            {(r["sequence"], r["anchor"]) for r in diagnostics} ==
            {(r["sequence"], r["anchor"]) for r in report["rows"]}, "all anchor diagnostics")
    ratios = [r["full_delta_to_family_radius_ratio"] for r in diagnostics if r["radius_ratio_defined"]]
    require(all(math.isfinite(x) and x >= 0 for x in ratios) and
            all(r["full_delta_to_family_radius_ratio"] is None for r in diagnostics
                if not r["radius_ratio_defined"]), "saved ratio domain")
    ratio_mean = math.fsum(ratios)/len(ratios) if ratios else None
    ratio_max = max(ratios) if ratios else None
    presence_agreements = sum(r["gt_native_presence_agreement"] for r in diagnostics)
    table = "\n".join(
        f"| {p['parent_id']} | {100*p['baseline']['J']:.7f} | {100*p['baseline']['F']:.7f} | "
        f"{100*p['full_gt_write']['J']:.7f} | {100*p['full_gt_write']['F']:.7f} | "
        f"{100*p['delta']['JF']:+.7f} |" for p in report["parents"])
    text = f"""# Full GT-mask/presence memory write: TRAIN diagnostic

All 48 TRAIN anchors and both +1/+3 future contexts completed: 96 native future
forwards, {report['unique_future_frames']} distinct (parent, frame) pairs, and no
image or memory encoder calls, optimization or new data acquisition. The four final-frame
events remain included and flagged. Repeated frames are separate anchor-context
events, not independent observations. All original rows and diagnostics are retained.

Equal-parent baseline J/F/J&F is
{100*report['baseline']['J']:.8f}/{100*report['baseline']['F']:.8f}/{100*report['baseline']['JF']:.8f}%.
Cached full GT-mask/presence write replacements give
{100*report['full_gt_write']['J']:.8f}/{100*report['full_gt_write']['F']:.8f}/{100*report['full_gt_write']['JF']:.8f}%.
Paired changes are J {100*report['delta']['J']:+.8f},
F {100*report['delta']['F']:+.8f}, J&F {100*report['delta']['JF']:+.8f} percentage points.

| TRAIN parent | Baseline J (%) | Baseline F (%) | Full GT write J (%) | Full GT write F (%) | ΔJF pp |
|---|---:|---:|---:|---:|---:|
{table}

Each intervention replaces one anchor's spatial memory with the cached native
GT-mask/presence encoded BF16 tensor. Other writes, pointers, positions and native
outputs are fixed. No family projection or magnitude bound is applied. The native
hard-mask encoding path and GT-presence embedding remain explicit confounds;
this is not a pure spatial-locality test.

Saved presence signs agree for {presence_agreements}/48 anchors; that does not
make the GT presence logits or hard-mask encoding identical to the baseline.
The full replacement norm divided by the original rho=.15 pre-BF16 family radius
is defined for {len(ratios)}/48 anchors: mean {ratio_mean}, maximum {ratio_max}.
These are scalar summaries of stored diagnostics, not new tensor measurements.
Norms compare stored BF16 values lifted to float32 with the reference family
bound, not an unobserved prequantization GT tensor.

This is privileged TRAIN target-object, fixed-context horizon-set J/F. It uses
native selected post-presence masks and unchanged pixel metric formulas, but
includes the specified horizon endpoints rather than the official sequence
interior-only aggregate. It is not official all-object DAVIS, causal rollout,
DEV/held-out efficacy, an attainable causal-head result, or an upper bound on
possible improvement. Every parent is retained; no new gate is applied. The
observed change is small relative to the unchanged +3 J&F-point requirement on
each of two independent held-out sources. A small result here does not prove
that every alternative memory intervention must fail.

The unchanged source-only prior note in `docs/local-storage-prior.md` is included
with its original read-scope limitations; the exporter adds no literature claim.

Actual complete outer wall: {outer['wall_seconds']:.9f} seconds; peak group RSS:
{outer['peak_group_rss_kib']} KiB. Inner wall: {inner['wall_seconds']:.9f} seconds.
Inner and row timers overlap the outer cost and must not be added to it.
The original model state and cached histories/positions/pointers were unchanged.
The admission identifies the integration commit; the new diagnostic source is
separately bound by exact hashes, not asserted to have been covered by older CI.

Only UTF-8 source, scope, admission, specs, complete receipts/evaluation and raw
logs are exported. No PT/NPZ/PNG/JPEG, masks, labels, tensors, checkpoint or model
is opened by this publisher or included. Output NPZ hashes remain in receipts;
this export checks their recorded inventory, not their binary contents.

`manifest.json` records original and public byte lengths and SHA256 hashes.
The only transformations are ordered literal path-prefix substitutions:
task root to `$TASK`, then user home to `$USER_HOME`. Embedded original SHA fields
remain unchanged and refer to original bytes, not their sanitized public copies.
No unfavorable field or row is removed. README values are rendered from saved
aggregates only; no metric or scientific computation is rerun.
"""
    files["README.md"] = (text.encode(), None, "Rendered saved aggregates, parent table, costs, scope and scalar anchor-diagnostic summaries.")
    aliases = [(str(task), "$TASK"), (str(Path.home()), "$USER_HOME")]

    def alias(value):
        for old, new in aliases:
            value = value.replace(old, new)
        return value

    for path, expected in inputs.items():
        require(digest(path.read_bytes()) == expected, "input changed before publication")
    prepared, entries = {}, []
    for relative, (raw, origin, derivation) in files.items():
        data = alias(raw.decode("utf-8")).encode("utf-8")
        require(not re.search(rb"/Users/[A-Za-z0-9_.-]+/|/(?:private/)?var/folders/[A-Za-z0-9]+/", data),
                "private path remains")
        prepared[relative] = data
        row = dict(path=relative, bytes=len(data), sha256=digest(data))
        if origin:
            row.update(origin=dict(path=alias(str(origin)), bytes=len(raw), sha256=digest(raw)),
                       transformation="byte-identical" if data == raw
                       else "ordered literal prefix replacement only")
        else:
            row["derivation"] = derivation
        entries.append(row)
    manifest = dict(schema="evr0030.full-gt-write-publication.v1",
        created_utc=datetime.now(timezone.utc).isoformat(), entries=entries,
        input_json_and_text_sha256={alias(str(p)): v for p, v in inputs.items()},
        substitutions=["literal task root → $TASK", "then literal current user home → $USER_HOME"],
        original_json_fields_preserved=True, completed_operations=1,
        no_arrays_models_or_gt_loaded=True, output_npz_content_rehashed=False)
    output.mkdir(exist_ok=False)
    for relative, data in prepared.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for path, expected in inputs.items():
        require(digest(path.read_bytes()) == expected, "input changed during publication")
    print(json.dumps(dict(output=str(output), files=len(prepared)+1,
                         bytes=sum(len(v) for v in prepared.values()) + (output/"manifest.json").stat().st_size,
                         manifest_sha256=digest((output/"manifest.json").read_bytes()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
