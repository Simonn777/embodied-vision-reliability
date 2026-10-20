"""Fetch only 48 fixed query PNGs after verifying every completed video seal.

No image or annotation is decoded. No blanket offline guard is installed:
the original fixed-URL/ETag/206 RangeReader must perform its bounded HTTP.
The original acquisition default remains closed; only this frozen callback
admits evaluator_gt, with its unchanged 500,000-byte phase cap.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OWN = "experiments/0029-memory-information/"
SELF = OWN + "acquire_query_gt.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run(args):
    root, out = Path(args.repo_root).resolve(), Path(args.output_dir).absolute()
    if out.exists() or out.is_symlink() or out.resolve() == root or root in out.resolve().parents:
        raise ValueError("fresh private query-GT output required")
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0029.query-gt-operation.v1", status="started", eligible=False,
        expected_commit=args.expected_commit, media_decoded=False, query_gt_decoded=False,
        http_scope="Only unchanged official DAVIS RangeReader; 48 fixed members; 500000 body bytes",
        blanket_offline_guard=False)
    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")
    save()
    try:
        seals = module("evr0029_query_gt_seal", root / OWN / "seal_queries.py")
        prepared = seals.prepare(root, args.expected_commit, args.proof)
        h, v, old = prepared["h"], prepared["video"], prepared["old"]
        h.require(h.sha(Path(__file__)) == prepared["sources"][SELF], "wrong GT wrapper")
        record["source_sha256"] = prepared["sources"]
        # The immutable manifest/index must be available to acquire_phase before
        # its callback. These are metadata only; the callback remains pre-HTTP.
        first = prepared["layouts"][0]["arguments"]
        prior, _ = v.evidence_arguments(h, root, prepared["proof"]["evidence_commit"],
                                        first["evidence_root"], first["evidence_spec"])
        acquisition = old.module("evr0029_query_gt_acquisition", root / OWN / "acquire_davis.py")
        helper = acquisition.load_helper(root / acquisition.HELPER)
        plan, index = acquisition.load_plan(root / acquisition.MANIFEST, prior["metadata_root"], helper)
        h.require(acquisition.PHASE_CAPS["evaluator_gt"] == 500000
                  and acquisition.PHASE_COUNTS["evaluator_gt"] == 48, "fixed query acquisition limits")
        state = {}

        def verifier(candidate_plan, manifest_sha256, proof_path):
            h.require(candidate_plan == plan and manifest_sha256 == index.fingerprints["manifest"]
                      and Path(proof_path).resolve() == prepared["proof_path"], "GT callback identity")
            if not state:
                admission, context = seals.verify_queries(root, args.expected_commit, proof_path,
                                                          prepared=prepared)
                h.require(context["admitted"]["plan"] == plan
                          and admission["manifest_sha256"] == manifest_sha256, "GT/prediction manifest differs")
                # No site.main()/NumPy/model import is needed for this stdlib operation.
                seals.require_runtime(context)
                state.update(admission=admission, context=context)
                record["prediction_admission"] = admission
                save()
            else:
                seals.recheck(state["context"])
            return state["admission"]

        def postcheck():
            h.require(state, "GT transport bypassed prediction verifier")
            seals.recheck(state["context"])
            _, current = acquisition.load_plan(root / acquisition.MANIFEST, prior["metadata_root"], helper)
            h.require(current.fingerprints == index.fingerprints, "GT source metadata changed")

        result = acquisition.acquire_phase(helper, plan, index, "evaluator_gt", out / "acquisition",
            expected_commit=args.expected_commit, source_sha256=prepared["sources"],
            source_check=postcheck, proof_path=prepared["proof_path"], gt_verifier=verifier)
        h.require(result.get("status") == "completed" and len(result.get("members", [])) == 48
                  and result.get("media_decoded") is False
                  and result.get("gt_admission") == state["admission"], "query acquisition incomplete")
        postcheck()
        h.require(time.perf_counter() - began <= 1800, "whole GT operation exceeded time cap")
        record.update(status="completed", eligible=True, query_count=48,
                      acquisition_receipt_sha256=h.sha(out / "acquisition/receipt.json"),
                      response_body_bytes=result["response_body_bytes"])
        save()
    except BaseException as exc:
        record.update(status="failed", eligible=False, error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "proof", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
