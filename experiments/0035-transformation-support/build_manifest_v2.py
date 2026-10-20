#!/usr/bin/env python3
"""Metadata-only correction of the unallocated v1 proposal; never transports."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = {
    "build_manifest.py": "d42448b5382df8fe62c1703c80046701416bbd6462ecc743c917ee787552e925",
    "proposed-manifest.json": "fc0bbbd85cd66878426956dc1c382826e56dbffd7e92a240b13d6f09d5dc03f1",
    "manifest-build-report.json": "296d32d0ca46903f96627ae0a233915e3f99b29f48fadf7343f86cefe2bdc364",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    output = HERE / "proposed-manifest-v2.json"
    report = HERE / "manifest-build-report-v2.json"
    if output.exists() or report.exists():
        raise FileExistsError("fresh v2 outputs required; preserve prior attempts")
    prior = {}
    for name, expected in PRIOR.items():
        raw = (HERE / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError("v1 changed: " + name)
        prior[name] = {"sha256": expected, "bytes": len(raw)}
    spec = importlib.util.spec_from_file_location("_vost_metadata_v1", HERE / "build_manifest.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)  # Verified stdlib metadata builder; main is not invoked.
    result = builder.build()
    saved = json.loads((HERE / "proposed-manifest.json").read_text())
    if result != saved:
        raise ValueError("v1 deterministic reproduction mismatch")
    unchanged = ("metadata_sources", "archive", "selection", "clips", "members", "totals", "phases", "phase_policy")
    result["schema"] = "evr0035.proposed-vost-manifest.v2"
    result["builder"] = {"path": "0035-transformation-support/build_manifest_v2.py",
                         "sha256": digest(Path(__file__).read_bytes())}
    review = (HERE / "metric-oracle-review.md").read_bytes()
    result["revision"] = {
        "prior_files": prior, "prior_files_preserved": True,
        "reason": "Correct paper-intent inference using the released pinned-author evaluator execution path.",
        "review": {"path": "0035-transformation-support/metric-oracle-review.md",
                   "sha256": digest(review), "bytes": len(review)},
        "changes": ["metric semantics", "explicit prospective three-request sentinel accounting",
                    "version/builder/revision metadata"],
        "transport_or_scientific_admission": False,
    }
    pin = "fe274574cb03c8a3ea83e121dd76e20b703781fd"
    prefix = "https://github.com/TRI-ML/VOST/blob/" + pin + "/evaluation/"
    result["metric_contract"] = {
        "status": "source-derived prospective contract, no evaluator execution",
        "author_commit": pin,
        "primary": ["J", "Jtr"],
        "label_255_behavior": "pinned dataset loader converts 255 to 0; caller passes void=None; treated as background",
        "prior_error": "v1 inferred ignored-255 behavior from paper wording; this does not describe the pinned caller",
        "sources": [prefix + "source/dataset.py#L80-L93",
                    prefix + "source/evaluation.py#L50-L69", prefix + "source/metrics.py",
                    prefix + "source/utils.py", prefix + "evaluation_method.py#L35-L52"],
        "frames": "positions [1:-1]; Jtr suffix starts floor(0.75 * evaluated_frame_count)",
        "aggregation": "equal sequence-object means, not equal videos if object counts differ",
        "target_ids": "all positive first-mask IDs except 255; check contiguous IDs before matching author 1..max convention",
        "read_scope": "parent/peer pinned-source review; this builder reads the saved review, not remote source or masks",
        "final_goal": "no official VOST JF claim; no replacement of the unchanged two-source +3 official JF gate",
    }
    result["scientific_limits"][0] = (
        "pinned-author VOST J/Jtr with loader 255->0 and caller void=None; no official JF claim")
    result["transport_requirements"]["prospective_three_read_accounting"] = {
        "assumption": "fixed local header, bounded name/extra, compressed member; no additional descriptor fetch",
        "response_body_bytes_cap": result["totals"]["response_body_bytes_cap"],
        "request_count": 3 * len(result["members"]),
        "length_plus_one_sentinel_bytes": 3 * len(result["members"]),
        "requested_body_upper_bound": result["totals"]["response_body_bytes_cap"] + 3 * len(result["members"]),
        "status": "derived only; request cap, retry/error budgets and allocation still unset",
    }
    if not all(result[k] == saved[k] for k in unchanged):
        raise ValueError("unexpected dataset/selection/descriptor change")
    raw = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    output.write_bytes(raw)
    for name, expected in PRIOR.items():
        if digest((HERE / name).read_bytes()) != expected:
            raise ValueError("v1 changed during build")
    record = {"status": "metadata_validation_passed", "allocated": False, "acquisition_authorized": False,
              "network_calls": 0, "scientific_payload_reads": 0, "prior_files": prior,
              "unchanged_fields": list(unchanged), "builder": result["builder"],
              "manifest_sha256": digest(raw), "manifest_bytes": len(raw),
              "members": 892, "frames": 446, "response_body_bytes_cap": 171951497,
              "requested_body_upper_bound_if_three_reads": 171954173}
    report.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
