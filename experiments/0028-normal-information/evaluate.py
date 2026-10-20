"""Admit complete sealed predictions, then reuse the fixed historical truth.

No models, scientific arrays, native estimators, or new truth acquisition.
Historical truth remains development data. All 24 queries are retained.
"""
import argparse
import math
from pathlib import Path
import sys
import time

import common as c
import metrics


GROUND_TRUTH_SHA256 = "092aea6e4612d8b1d9d76e891a56db042fe9a5c9179f3e01107dc43a818919b0"
IDENTITY = ("scene", "query", "anchor", "target")


def _seconds(value):
    c.require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1800,
              "invalid stage wall seconds")
    return float(value)


def _file(root, name):
    c.require(isinstance(name, str) and name and Path(name).name == name
              and name not in (".", ".."), "unsafe sealed filename")
    path = root / name
    c.require(path.is_file() and not path.is_symlink(), "missing or linked sealed file")
    return path


def _inventory(root, seal):
    c.require(root.is_dir() and not root.is_symlink(), "invalid sealed directory")
    c.require(isinstance(seal["files"], dict) and "receipt.json" in seal["files"]
              and "seal.json" not in seal["files"], "invalid seal inventory")
    c.require({p.name for p in root.iterdir()} == set(seal["files"]) | {"seal.json"},
              "sealed directory inventory changed")
    for name, digest in seal["files"].items():
        c.require(c.sha(_file(root, name)) == digest, "sealed file changed: " + name)
    c.require(seal["receipt_sha256"] == seal["files"]["receipt.json"],
              "receipt seal binding differs")


def _header(record, schema, job_hash, commit, sources):
    c.require(record["schema"] == schema and record["status"] == "completed"
              and record["expected_commit"] == commit
              and record["source_sha256"] == sources
              and record["job_sha256"] == job_hash
              and record["ground_truth_access"] is False,
              "incomplete or mismatched prediction-stage record")


def _array_meta(root, meta, name):
    """Opaque integrity only: never import NumPy or decode an array."""
    path = _file(root, name)
    c.require(meta["filename"] == name and type(meta["bytes"]) is int
              and meta["bytes"] >= 0 and path.stat().st_size == meta["bytes"]
              and c.sha(path) == meta["sha256"] and isinstance(meta["arrays"], dict),
              "array metadata/integrity mismatch")


def admit_predictions(job, job_path, commit, sources):
    """Complete all provenance, population and hash checks before any GT byte."""
    job_hash = c.sha(job_path)
    c.require(job["expected_commit"] == commit
              and job["ground_truth_evaluation_sha256"] == GROUND_TRUTH_SHA256,
              "job freeze or fixed historical truth declaration differs")
    root, raw_root = Path(job["predictions"]), Path(job["raw_frontend"])
    prediction_outer = c.verify_outer(
        job["predictions_outer"], root / "seal.json", commit, "predict.py", job_path)
    seal, record = c.read(_file(root, "seal.json")), c.read(_file(root, "receipt.json"))
    _header(seal, "evr0028.prediction-seal.v1", job_hash, commit, sources)
    _header(record, "evr0028.prediction.v1", job_hash, commit, sources)
    _inventory(root, seal)
    c.require(seal["query_count"] == 24 and seal["image_count"] == 27
              and seal["arms"] == list(c.ARMS) and not record["offline_attempts"]
              and len(record["images"]) == 27 and len(record["queries"]) == 24
              and type(seal["comparator_valid"]) is bool
              and seal["comparator_valid"] == record["comparator_valid"],
              "incomplete prediction population/comparator declaration")

    # Reuse the existing frontend verifier, then strengthen its receipt checks.
    predictor = c.module("_evr0028_evaluation_prediction", c.HERE / "predict.py")
    raw_record, raw_binding = predictor.admit_frontend(job, job_path, commit, sources)
    raw_seal = c.read(_file(raw_root, "seal.json"))
    _header(raw_record, "evr0028.frontend.v1", job_hash, commit, sources)
    _inventory(raw_root, raw_seal)
    c.require(not raw_record["offline_attempts"], "raw frontend attempted offline operation")
    c.require(record["raw_frontend_binding"] == raw_binding, "raw frontend binding changed")

    # This existing loader verifies source, installation/model, input acquisition
    # and cache inventories by metadata and opaque hashes; no array reader runs.
    old, frontend, fa, prior, old_outer, frozen, cache, verified = c.original_frontend(job)
    images, edges = c.selected_layout(verified)
    c.require(record["old_frontend_binding"] == old_outer
              and record["source_input_files"] == verified["file_sha256"],
              "old input cache/provenance changed")
    qa = frontend.qualification_args(fa)
    _, _, _, model_bindings = frontend.q.verify_files(qa, frozen)
    c.require(raw_record["input_bindings"] == dict(
        frontend=old_outer, plan_sha256=c.PLAN_SHA, model=model_bindings,
        cache_file_sha256=verified["file_sha256"]), "raw model/cache binding changed")
    c.require(raw_record["selected_images"] == [r["filename"] for r in images]
              and raw_record["selected_edges"] == [
                  {k: e[k] for k in ("scene", "source", "target", "edge_index")} for e in edges],
              "raw frontend allocation differs")
    native_binding = old.native_binding(Path(job["native_runtime"]), frontend.q)
    c.require(record["native_binding"] == native_binding, "native runtime binding changed")
    native_assets = {}
    for name in ("native_library", "native_receipt"):
        p = Path(job[name])
        c.require(p.is_file() and not p.is_symlink()
                  and c.sha(p) == job[name + "_sha256"], "TC native asset changed")
        native_assets[name + "_sha256"] = c.sha(p)
    c.require(record["runtime"]["tc_library_sha256"] == native_assets["native_library_sha256"],
              "TC loaded library differs")

    expected_files, raw_expected_files = {"receipt.json"}, {"receipt.json"}
    image_flags = []
    for expected, raw, image in zip(images, raw_record["images"], record["images"]):
        index, name = expected["image_index"], expected["filename"]
        c.require(type(index) is int and index >= 0, "invalid image index")
        c.require(all(r["image_index"] == index and r["filename"] == name
                      and r["status"] == "completed" for r in (raw, image)),
                  "image identity or completeness differs")
        image_name = "image-%03d.json" % index
        c.require(c.read(_file(raw_root, image_name)) == raw
                  and c.read(_file(root, image_name)) == image, "image receipt changed")
        raw_names = ("raw-%03d.npz" % index, "processed-%03d.npz" % index)
        for field, filename in zip(("raw", "processed"), raw_names):
            _array_meta(raw_root, raw[field], filename)
        sample_name = "samples-%03d.npz" % index
        _array_meta(root, image["samples_file"], sample_name)
        c.require(type(image["comparator_valid"]) is bool
                  and image["comparator_valid"] == (image["full"]["status"] == "completed"),
                  "image comparator declaration differs")
        image_flags.append(image["comparator_valid"])
        expected_files.update((image_name, sample_name))
        raw_expected_files.update((image_name, *raw_names))
    c.require(set(raw_seal["files"]) == raw_expected_files, "unexpected raw frontend files")

    rows, pair_flags = [], []
    for index, (edge, listed) in enumerate(zip(edges, record["queries"])):
        identity = dict(scene=edge["scene"], query="B" + str(index % 8 + 1),
                        anchor=edge["source"], target=edge["target"],
                        edge_index=edge["edge_index"], pair_index=index)
        name = "pair-%02d.json" % index
        c.require(all(listed[k] == v for k, v in identity.items())
                  and listed["filename"] == name and listed["sha256"] == c.sha(_file(root, name)),
                  "prediction identity/list binding differs")
        row = c.read(_file(root, name))
        c.require(row["schema"] == "evr0028.pair.v1" and row["status"] == "completed"
                  and all(row[k] == v for k, v in identity.items())
                  and row["original_match_indices"] == listed["original_match_indices"]
                  and row["selected_match_ids"] == listed["selected_match_ids"]
                  and set(row["arms"]) == set(c.ARMS)
                  and type(row["comparator_valid"]) is bool, "incomplete pair or identity mismatch")
        for arm in c.ARMS:
            outcome = row["arms"][arm]
            c.require(isinstance(outcome, dict) and outcome["status"] in metrics.STATUSES
                      and "R" in outcome and "t" in outcome, "incomplete arm outcome")
        input_name, partial_name = "pair-%02d-input.npz" % index, "pair-%02d.partial.json" % index
        _array_meta(root, row["inputs"], input_name)
        partial = c.read(_file(root, partial_name))
        c.require(all(partial["identity"][k] == v for k, v in identity.items())
                  and partial["progress"]["status"] == "completed", "final pair checkpoint incomplete")
        expected_files.update((name, input_name, partial_name))
        pair_flags.append(row["comparator_valid"])
        rows.append(row)
    c.require(set(seal["files"]) == expected_files, "unexpected prediction files")
    computed_comparator = all(image_flags) and all(pair_flags)
    c.require(computed_comparator == seal["comparator_valid"], "comparator aggregation mismatch")
    timing = {
        "frontend_outer_wall_seconds": _seconds(raw_binding["outer"]["wall_seconds"]),
        "frontend_child_wall_seconds": _seconds(raw_record["wall_seconds"]),
        "prediction_outer_wall_seconds": _seconds(prediction_outer["wall_seconds"]),
        "prediction_child_wall_seconds": _seconds(record["wall_seconds"]),
        "matched_complete_pipeline_time_claim": False,
        "scope": ("Stage timings are reported separately. This is a support-conditioned "
                  "opportunity test, not a matched complete-pipeline cost comparison."),
    }
    bindings = dict(
        prediction_outer=prediction_outer, prediction_seal_sha256=c.sha(root / "seal.json"),
        prediction_receipt_sha256=c.sha(root / "receipt.json"), prediction_files=seal["files"],
        raw_frontend=raw_binding, raw_frontend_files=raw_seal["files"],
        old_frontend=old_outer, source_input_files=verified["file_sha256"],
        model=model_bindings, native=native_binding, **native_assets)
    return dict(rows=rows, comparator_valid=computed_comparator, bindings=bindings,
                timing_summary=timing)


def load_historical_truth(job, rows):
    """Called only after prediction admission; no other truth file is allowed."""
    path = Path(job["ground_truth_evaluation"])
    c.require(job["ground_truth_evaluation_sha256"] == GROUND_TRUTH_SHA256
              and path.is_file() and not path.is_symlink()
              and c.sha(path) == GROUND_TRUTH_SHA256, "historical truth file changed")
    saved = c.read(path)
    c.require(saved["schema"] == "evr0023.evaluation.v1" and saved["query_count"] == 24
              and len(saved["rows"]) == 24, "historical evaluation population differs")
    expected = {(r["scene"], r["query"]): {k: r[k] for k in IDENTITY} for r in rows}
    c.require(len(expected) == 24, "duplicate admitted query")
    truth, preserved = {}, {}
    for row in saved["rows"]:
        key = row["scene"], row["query"]
        c.require(key in expected and key not in truth
                  and all(row[k] == v for k, v in expected[key].items()),
                  "historical truth query identity differs")
        relative = row["ground_truth"]["relative"]
        c.require(isinstance(relative, list) and len(relative) == 2, "relative [R,t] required")
        pose = dict(R=relative[0], t=relative[1])
        metrics.checked_pose(pose)
        truth[key] = pose
        preserved[key] = dict(expected[key], **pose)
    c.require(set(truth) == set(expected), "historical truth coverage differs")
    return truth, [preserved[(r["scene"], r["query"])] for r in rows]


def run(args):
    sources = c.verify_source(args.expected_commit)  # First source barrier.
    job = c.read(args.job)
    c.require(job["expected_commit"] == args.expected_commit, "job freeze")
    out = Path(args.output)
    c.require(not out.resolve().is_relative_to(c.ROOT), "private output required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(schema="evr0028.evaluation.v1", status="started", eligible=False,
                  stage="prediction_admission", expected_commit=args.expected_commit,
                  source_sha256=sources, job_sha256=c.sha(args.job),
                  ground_truth_access=False, manuscript_admission=False)
    def save():
        c.write(out / "receipt.json", record, replace=True)
    save()
    try:
        admission = admit_predictions(job, args.job, args.expected_commit, sources)
        record.update(stage="historical_truth", ground_truth_access=True,
                      prediction_bindings=admission["bindings"])
        save()
        truth, truth_rows = load_historical_truth(job, admission["rows"])
        record["stage"] = "metrics"; save()
        result = metrics.evaluate_records(admission["rows"], truth,
                                          comparator_valid=admission["comparator_valid"])
        result.update(expected_commit=args.expected_commit, source_sha256=sources,
                      job_sha256=record["job_sha256"], bindings=admission["bindings"],
                      ground_truth_evaluation_sha256=GROUND_TRUTH_SHA256,
                      truth_rows=truth_rows, timing_summary=admission["timing_summary"],
                      data_use="All three places and all24 queries remain development data.")
        # This evaluator installs no model/offline import guard. Recheck hashes
        # and complete chains without reading any numerical cache.
        final = admit_predictions(job, args.job, args.expected_commit, sources)
        c.require(final["bindings"] == admission["bindings"]
                  and final["rows"] == admission["rows"]
                  and c.source_hashes() == sources
                  and c.sha(args.job) == record["job_sha256"]
                  and c.sha(job["ground_truth_evaluation"]) == GROUND_TRUTH_SHA256,
                  "prediction/source/job/truth changed during evaluation")
        c.require(time.perf_counter() - started <= 1800, "evaluation deadline")
        c.write(out / "evaluation.json", result)
        record.update(status="completed", eligible=True, stage="completed",
                      query_count=24, image_count=27, evaluation_sha256=c.sha(out / "evaluation.json"),
                      gate_passed=result["gate"]["passed"], wall_seconds=time.perf_counter() - started)
        save()
        seal = dict(schema="evr0028.evaluation-seal.v1", status="completed",
                    expected_commit=args.expected_commit, source_sha256=sources,
                    job_sha256=record["job_sha256"], receipt_sha256=c.sha(out / "receipt.json"),
                    query_count=24, image_count=27, ground_truth_access=True,
                    files={name: c.sha(out / name) for name in ("evaluation.json", "receipt.json")})
        c.write(out / "seal.json", seal)
        print(c.serial(dict(status="completed", eligible=True, gate_passed=record["gate_passed"],
                            wall_seconds=record["wall_seconds"])))
        return 0
    except BaseException as exc:
        record.update(status="failed", eligible=False, failure_stage=record["stage"],
                      error_type=type(exc).__name__, error=str(exc),
                      wall_seconds=time.perf_counter() - started)
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    sys.exit(run(parser.parse_args()))
