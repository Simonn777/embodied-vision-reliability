"""Copy completed output-error text evidence; no array/GT reads or recomputation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def run(output):
    root = Path(__file__).resolve().parent
    native, task = root.parent, root.parent.parents[1]
    output = output.resolve()
    if output != native / "publication-output-error-01" or output.exists():
        raise ValueError("fresh fixed private output required")
    inputs, files = {}, {}
    sha = lambda raw: hashlib.sha256(raw).hexdigest()

    def read(path):
        if path.suffix not in (".json", ".log", ".md", ".py"):
            raise ValueError("text only")
        raw = path.read_bytes()
        if path in inputs and inputs[path] != sha(raw):
            raise ValueError("input changed")
        inputs[path] = sha(raw)
        return raw

    def load(path):
        return json.loads(read(path))

    def copy(path, relative):
        if relative in files:
            raise ValueError("duplicate destination")
        files[relative] = (read(path), path)

    helper = native / "analysis/summarize_receipts.py"
    assert sha(read(helper)) == "ccd09ae5786d25b564e7093c36c62060b1ad8a9e1949434e5034dcd4573c2111"
    loader = importlib.util.spec_from_file_location("saved_error_reporting", helper)
    summary = importlib.util.module_from_spec(loader); loader.loader.exec_module(summary)
    spec_path, operation_path = root / "spec-01.json", root / "outer-spec-01.json"
    spec, operation = load(spec_path), load(operation_path)
    receipt_path, outer_path = root / "run-01/receipt.json", root / "supervision-01/receipt.json"
    receipt, outer = load(receipt_path), load(outer_path)
    assert receipt["status"] == "completed" and not summary.clean_outer(
        outer, inputs[receipt_path], inputs[operation_path])
    assert outer["schema"] == "evr0029.exit-checked-operation.v1" and outer["supervisor_sha256"] == \
        "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8"
    assert outer["monitor_sha256"] == "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"
    assert 0 <= outer["wall_seconds"] <= 1800 and 0 <= outer["peak_group_rss_kib"] <= 8388608
    assert receipt["spec_sha256"] == inputs[spec_path] and receipt["source_sha256"] == spec["source_sha256"]
    assert Path(operation["result_file"]) == receipt_path and Path(spec["output"]) == receipt_path.parent
    assert operation["command"][-2:] == [str(root / "analyze.py"), str(spec_path)]
    assert receipt["model_calls"] == 0 and receipt["final_test_access"] is False
    assert receipt["included_frames"] == 150 and receipt["raw_frames_bound"] == 154
    report_path = root / "run-01/analysis.json"; report = load(report_path)
    assert receipt["analysis_sha256"] == inputs[report_path] and report["rows"] == receipt["rows"]
    assert len(report["rows"]) == 150 and report["exact_saved_native_metric_consistency"] is True
    assert receipt["exact_saved_native_metric_consistency"] is True and report["scientific_gain_claim"] is False
    assert [p["parent_id"] for p in report["parents"]] == ["cat-girl", "upside-down"]
    assert [p["frames"] for p in report["parents"]] == [87, 63]
    for path, relative in ((spec_path, "worker-spec.json"), (operation_path, "outer-spec.json"),
                           (receipt_path, "run/receipt.json"), (report_path, "run/analysis.json"),
                           (outer_path, "supervision/receipt.json"), (root / "ADMISSION.md", "docs/ADMISSION.md")):
        copy(path, relative)
    assert set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}
    for name, expected in outer["log_sha256"].items():
        path = outer_path.parent / name; copy(path, "supervision/" + name); assert inputs[path] == expected
    for filename, expected in spec["source_sha256"].items():
        path = Path(filename); copy(path, "sources/" + path.relative_to(native).as_posix())
        assert inputs[path] == expected
    # Bind prior metadata only; never follow the PNG/NPZ entries stored within it.
    for entry in spec["bindings"].values():
        path = Path(entry["path"]); raw = read(path)
        assert sha(raw) == entry["sha256"] and len(raw) == entry["bytes"]
    copy(helper, "reporting/summarize_receipts.py"); copy(Path(__file__).resolve(), "reporting/publish.py")
    rows = "\n".join(f"| {p['parent_id']} | {p['frames']} | "
        f"{100*p['error_location_pixel_sums']['within_r']['error_mass_fraction']:.10f}% | "
        f"{p['flag_frame_counts']['complete_target_miss']} | {p['flag_frame_counts']['nonempty_disjoint']} |"
        for p in report["parents"])
    readme = f"""# Native DEV output-error location

Completed saved-output analysis of all 150 interior frames (154 raw bound), with
exact agreement to the existing native J/F/JF. No model calls or final data.

| Parent | Interior frames | Error pixels within r of GT boundary | Complete target misses | Nonempty disjoint masks |
|---|---:|---:|---:|---:|
{rows}

Here r=ceil(.008*hypot(H,W)), the unchanged official boundary tolerance. These
pixel-sum fractions are descriptive, weighted by error-pixel counts; they are
not equal-frame accuracy, independent samples, or attainable repair gains.
GT boundary location is evaluator information, not an available causal signal.
No complete misses/disjoint masks in this set does not prove identity correctness.
The separately prepared TRAIN paired-boundary screen is not evaluated here.

Outer wall {outer['wall_seconds']:.12f}s; peak sampled group RSS
{outer['peak_group_rss_kib']} KiB. Inner wall {receipt['wall_seconds']:.12f}s
overlaps the outer and is not added again. This analysis does not reopen the
closed channel-affine family or establish novelty, efficacy or the paper gate.

The complete rows, source/spec/admission and clean outer/logs are retained.
Source scope retains its original prospective wording. This publisher copies
text only: no masks, images, arrays or model; no metric rerun. All original JSON
fields survive ordered literal task-root→$TASK, home→$USER_HOME substitutions.
Original hashes address raw bytes and public hashes address transformed bytes.
Prior metadata is rehashed by reference; referenced payloads are not opened.
"""
    files["README.md"] = (readme.encode(), None)
    aliases = [(str(task), "$TASK"), (str(Path.home()), "$USER_HOME")]
    def alias(text):
        for old, new in aliases:
            text = text.replace(old, new)
        return text
    entries, prepared = [], {}
    for relative, (raw, origin) in files.items():
        public = alias(raw.decode()).encode()
        assert not re.search(rb"/Users/[A-Za-z0-9_.-]+/|/(?:private/)?var/folders/[A-Za-z0-9]+/", public)
        prepared[relative] = public
        row = dict(path=relative, sha256=sha(public), bytes=len(public))
        if origin:
            row.update(origin=dict(path=alias(str(origin)), sha256=sha(raw), bytes=len(raw)),
                       transformation="byte-identical" if raw == public else "ordered literal prefix replacement only")
        else:
            row["derivation"] = "Rendered saved parent error fractions/counts, costs and limitations; no metric recomputation."
        entries.append(row)
    for path, expected in inputs.items():
        assert sha(path.read_bytes()) == expected
    output.mkdir(exist_ok=False)
    for relative, raw in prepared.items():
        path = output / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    manifest = dict(schema="evr0030.output-error-publication.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        entries=entries, inputs_sha256={alias(str(p)): h for p, h in inputs.items()},
        substitutions=["literal task root → $TASK", "then literal current user home → $USER_HOME"],
        original_json_fields_preserved=True, publication_payload_reads=False, completed_operations=1)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for path, expected in inputs.items():
        assert sha(path.read_bytes()) == expected
    print(json.dumps(dict(files=len(prepared)+1,
        bytes=sum(map(len, prepared.values()))+(output/"manifest.json").stat().st_size,
        manifest_sha256=sha((output/"manifest.json").read_bytes()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
