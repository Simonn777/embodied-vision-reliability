"""Evaluate all48 sealed predictions after the separately admitted GT phase."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OWN = "experiments/0029-memory-information/"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def gt_contract(h, old, root, commit, proof, gt_root, gt_spec, admission, context):
    command = [str(context["installation"] / "venv/bin/python"), "-S",
               str(root / OWN / "acquire_query_gt.py"), "--repo-root", str(root),
               "--expected-commit", commit, "--proof", str(proof), "--output-dir", str(gt_root)]
    binding = context["new_outer_contract"](h, root, commit, gt_spec, gt_root / "receipt.json", command)
    record = h.read_json(gt_root / "receipt.json")
    h.require(record.get("schema") == "evr0029.query-gt-operation.v1"
              and record.get("status") == "completed" and record.get("eligible") is True
              and record.get("expected_commit") == commit
              and record.get("media_decoded") is False and record.get("query_gt_decoded") is False
              and record.get("source_sha256") == context["sources"]
              and record.get("prediction_admission") == admission and record.get("query_count") == 48,
              "GT operation admission changed")
    directory = gt_root / "acquisition"
    receipt = directory / "receipt.json"
    acquired = h.read_json(receipt)
    h.require(h.sha(receipt) == record.get("acquisition_receipt_sha256")
              and acquired.get("schema") == "evr0029.davis-acquisition.v1"
              and acquired.get("status") == "completed" and acquired.get("phase") == "evaluator_gt"
              and acquired.get("expected_commit") == commit
              and acquired.get("source_sha256") == context["sources"]
              and acquired.get("manifest_sha256") == admission["manifest_sha256"]
              and acquired.get("archive") == context["plan"]["archive"]
              and acquired.get("gt_admission") == admission
              and acquired.get("media_decoded") is False
              and acquired.get("hidden_retries") is False and acquired.get("full_zip_fallback") is False,
              "query-mask acquisition identity")
    wanted = [row for row in context["plan"]["members"] if row["stage"] == "evaluator_gt"]
    h.require(len(wanted) == len(acquired.get("members", [])) == 48
              and acquired.get("selected_names") == [row["filename"] for row in wanted]
              and acquired.get("phase_body_cap_bytes") == 500000
              and acquired.get("response_body_bytes_exact") is True
              and h.number(acquired.get("response_body_bytes"), 0, 500000)
              and acquired["response_body_bytes"] == record.get("response_body_bytes"),
              "fixed query-mask inventory or transfer cap")
    fingerprints = dict(binding["file_sha256"])
    fingerprints[str(receipt)] = h.sha(receipt)
    masks = {}
    for actual, expected in zip(acquired["members"], wanted):
        h.require(all(actual.get(k) == v for k, v in expected.items()), "query-mask metadata changed")
        relative = "members/" + expected["output_relative"]
        h.require(actual.get("saved_relative") == relative, "query-mask path changed")
        path = directory / relative
        h.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                  and path.stat().st_size == actual["bytes"] == expected["uncompressed_bytes"]
                  and h.sha(path) == actual["sha256"], "query-mask bytes changed")
        identity = expected["sequence"], expected["frame_index"]
        h.require(identity not in masks, "duplicate query-mask identity")
        masks[identity] = path
        fingerprints[str(path)] = actual["sha256"]
    h.require(acquired.get("output_bytes") == sum(r["uncompressed_bytes"] for r in wanted),
              "query-mask output accounting")
    binding["file_sha256"] = fingerprints
    return masks, binding


def load_predictions(h, context, np):
    """Verify numeric arrays and all choices before any GT pixels are decoded."""
    rows = []
    for source, directory in zip(context["rows"], context["query_prediction_dirs"]):
        row = {key: source[key] for key in ("sequence", "frame", "target_id", "status", "actionable",
                                           "selection", "witnesses", "costs")}
        row["candidates"] = {}
        archives = {}
        for name, candidate in source["candidates"].items():
            desc = candidate["output"]
            path = directory / desc["filename"]
            h.require(path.stat().st_size == desc["bytes"] and h.sha(path) == desc["sha256"],
                      "candidate archive changed")
            if path not in archives:
                with np.load(path, allow_pickle=False) as saved:
                    h.require(set(saved.files) == set(desc["arrays"]), "candidate array inventory")
                    arrays = {key: saved[key] for key in saved.files}
                for key, value in arrays.items():
                    expected = desc["arrays"][key]
                    h.require(value.dtype.kind in "bifu" and np.isfinite(value).all()
                              and list(value.shape) == expected["shape"]
                              and str(value.dtype) == expected["dtype"]
                              and hashlib.sha256(value.tobytes(order="C")).hexdigest()
                                  == expected["raw_c_order_sha256"], "candidate numeric array changed")
                archives[path] = arrays
            value = archives[path][candidate["array_key"]]
            channel = candidate["channel"]
            h.require(value.ndim == 4 and value.shape[0] == 1 and type(channel) is int
                      and 0 <= channel < value.shape[1]
                      and list(value.shape[2:]) == source["witnesses"]["current_shape"],
                      "candidate resolution/channel")
            row["candidates"][name] = value[0, channel].copy()
        rows.append(row)
    h.require(len(rows) == 48, "complete prediction population required")
    return rows


def complete_video_costs(h, context):
    result = []
    for video in context["videos"]:
        spec = h.read_json(video["spec"])
        path = Path(spec["operation_output"]) / "receipt.json"
        h.require(h.sha(path) == video["outer"]["outer_sha256"], "bound video outer receipt changed")
        outer = h.read_json(path)
        result.append(dict(sequence=video["sequence"], wall_seconds=outer["wall_seconds"],
            peak_group_rss_kib=outer["peak_group_rss_kib"],
            calls=video["record"]["native_video"]["calls"]))
    return result


def run(args):
    root, proof = Path(args.repo_root).resolve(), Path(args.proof).resolve()
    out = Path(args.output_dir).absolute()
    if out.exists() or out.is_symlink() or out.resolve() == root or root in out.resolve().parents:
        raise ValueError("fresh private evaluation output required")
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0029.query-evaluation-operation.v1", status="started", eligible=False,
        expected_commit=args.expected_commit, query_masks_decoded=0, model_calls=0,
        offline_attempts=[], compatibility_probes=[])
    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")
    save()
    try:
        seals = module("evr0029_evaluation_seals", root / OWN / "seal_queries.py")
        admission, context = seals.verify_queries(root, args.expected_commit, proof)
        h, old = context["h"], context["old_admission"]
        own = h.frozen(root, args.expected_commit, [OWN + "run_evaluation.py", OWN + "evaluate_queries.py"])
        h.require(h.sha(Path(__file__)) == own[OWN + "run_evaluation.py"], "wrong evaluator wrapper")
        record.update(source_sha256={**context["sources"], **own}, prediction_admission=admission)
        masks, record["gt_binding"] = gt_contract(h, old, root, args.expected_commit, proof,
            Path(args.gt_root).resolve(), Path(args.gt_spec).resolve(), admission, context)
        seals.require_runtime(context)
        q = module("evr0029_evaluation_guard", root / "experiments/0022-frontend/qualify.py")
        q.install_guard(record["offline_attempts"], record["compatibility_probes"])
        sys.dont_write_bytecode = True
        import site
        site.main()
        h.require(Path(sys.prefix).resolve() == context["installation"] / "venv", "wrong evaluation prefix")
        import numpy as np
        from PIL import Image
        runtime = context["installation"] / "venv/lib/python3.12/site-packages"
        h.require(Path(np.__file__).resolve().is_relative_to(runtime)
                  and Path(Image.__file__).resolve().is_relative_to(runtime), "wrong numeric/decoder origin")
        core = module("evr0029_evaluation_core", root / OWN / "diagnostic_core.py")
        evaluator = module("evr0029_query_evaluator", root / OWN / "evaluate_queries.py")
        rows = load_predictions(h, context, np)
        h.require(core.evidence_coverage(rows)["pass"] and set(masks) == set(core.QUERY_IDENTITIES),
                  "fixed all-query population")
        for row in rows:
            evaluator.validate_completed(row, core)
        record["prediction_numeric_preflight_complete"] = True
        save()
        annotations = {}
        for identity in core.QUERY_IDENTITIES:
            with Image.open(masks[identity]) as image:
                h.require(image.format == "PNG" and image.mode in ("P", "L", "I", "I;16"),
                          "query annotation must retain integer IDs")
                annotations[identity] = np.asarray(image).copy()
            record["query_masks_decoded"] += 1
        result = evaluator.evaluate_queries(rows, annotations, core=core)
        h.require(result["eligible"] is True and result["denominator"] == 48, "evaluation incomplete")
        record["evaluation"] = result
        record["complete_costs"] = dict(
            video_operations=complete_video_costs(h, context),
            scope="Complete per-video outer times; query-stage times are overlapping descriptions. "
                  "Frontend/acquisition costs remain in separately bound original receipts.")
        seals.recheck(context)
        old.unchanged(h, record["gt_binding"]["file_sha256"])
        h.require(all(h.sha(root / name) == digest for name, digest in own.items()), "evaluator source changed")
        h.require(not record["offline_attempts"] and record["query_masks_decoded"] == 48,
                  "offline/full-evaluation contract")
        record.update(status="completed", eligible=True)
    except BaseException as error:
        record.update(status="failed", eligible=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
        print(json.dumps({k: record[k] for k in ("status", "eligible", "wall_seconds")}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "proof", "gt-root", "gt-spec", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
