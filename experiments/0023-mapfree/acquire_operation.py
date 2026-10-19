"""Frozen, explicit Map-free image/K acquisition operation; no GT or inference.

Preparation alone does not authorize a download. The reviewed protocol must
contain the exact private plan digest supplied on the command line. The original
candidate plan stays unchanged: adoption is recorded by the frozen protocol.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXED_LIMITS = dict(body_bytes=20 * 1024**2, requests=162,
                    wall_seconds=1800, request_seconds=30)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


planner = module("_evr0023_acquisition_planner", HERE / "propose_plan.py")
ranges = planner.ranges
supervisor = module("_evr0023_acquisition_source_verifier", HERE / "supervise.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return supervisor.sha256(path)


def bounded_json(path, limit=1024**2):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= limit,
            "bounded regular JSON file required")
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def source_paths(protocol):
    return [Path(__file__), HERE / "range_inputs.py", HERE / "propose_plan.py",
            Path(supervisor.__file__), protocol]


def verify_inputs(args):
    bindings = supervisor.verify_frozen(args.expected_commit, source_paths(args.protocol))
    require(re.fullmatch(r"[0-9a-f]{64}", args.plan_sha256) is not None,
            "full plan SHA256 required")
    # This binds private plan identity to the reviewed public protocol, rather
    # than letting a caller adopt a different same-sized cohort with a new hash.
    require(args.plan_sha256 in args.protocol.read_text(),
            "frozen protocol does not bind the requested plan SHA256")
    plan, plan_sha = bounded_json(args.plan)
    require(plan_sha == args.plan_sha256, "private plan SHA256 mismatch")
    index = ranges.load_index(args.members_index, args.central_directory,
                              args.index_receipt, plan["source"]["cached_index_sha256"])
    expected = planner.build_plan(index, plan["prior_resource_draft_sha256"])
    expected["planning_source_sha256"] = {
        "propose_plan.py": sha(HERE / "propose_plan.py"),
        "range_inputs.py": sha(HERE / "range_inputs.py")}
    require(plan == expected, "plan differs from the frozen deterministic metadata proposal")
    budget = plan["frontend_acquisition"]
    names = budget["member_names"]
    kinds = [ranges.frontend_kind(name) for name in names]
    require(len(names) == len(set(names)) == 54 and kinds.count("image") == 51
            and kinds.count("intrinsics") == 3, "exactly 51 images and three K members required")
    require(budget["range_requests"] == 162
            and budget["response_body_upper_bound"] <= FIXED_LIMITS["body_bytes"],
            "planned ranges exceed fixed admission caps")
    limits = ranges.Limits(**FIXED_LIMITS, output_bytes=budget["uncompressed_bytes"],
                           member_bytes=budget["maximum_member_uncompressed_bytes"])
    limits.validate()
    return plan, index, limits, bindings


def validate_completed(receipt, index, plan, limits, output):
    """Audit the finished opaque outputs; never decode an image or parse K."""
    budget = plan["frontend_acquisition"]
    names = budget["member_names"]
    require(receipt.get("schema") == ranges.SCHEMA and receipt.get("status") == "completed",
            "range helper did not complete")
    require(receipt.get("selected_names") == names and receipt.get("limits") == vars(limits)
            and receipt.get("source_sha256") == sha(HERE / "range_inputs.py")
            and receipt.get("index_sha256") == index.fingerprints
            and receipt.get("url") == index.url and receipt.get("etag") == index.etag
            and receipt.get("archive_bytes") == index.archive_bytes,
            "range helper identity/caps mismatch")
    require(receipt.get("pose_gt_members_allowed") is False
            and receipt.get("hidden_retries") is False and receipt.get("full_zip_fallback") is False
            and receipt.get("response_body_bytes_exact") is True,
            "unsupported acquisition semantics")
    requests = receipt.get("requests", [])
    body = receipt.get("response_body_bytes")
    require(type(body) is int and 0 <= body <= min(limits.body_bytes, budget["response_body_upper_bound"])
            and len(requests) == 162 and all(r.get("status") == 206
                    and type(r.get("body_bytes")) is int and r["body_bytes"] >= 0 for r in requests)
            and sum(r["body_bytes"] for r in requests) == body,
            "incomplete or over-budget response ledger")
    require(body >= sum(index.entries[name]["compressed_bytes"] + 30 + len(name) for name in names),
            "response ledger undercounts required header and compressed payload bytes")
    wall = receipt.get("wall_seconds")
    require(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall <= limits.wall_seconds,
            "range helper time limit exceeded")
    rows = receipt.get("members", [])
    require([row.get("filename") for row in rows] == names
            and receipt.get("output_bytes") == limits.output_bytes, "incomplete output inventory")
    for row in rows:
        entry = index.entries[row["filename"]]
        require(all(row.get(key) == entry[key] for key in ranges.CORE_FIELDS), "member metadata drift")
        path = output / row["filename"]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["uncompressed_bytes"]
                and sha(path) == row["sha256"], "opaque member output hash/size mismatch")


def verify_completed_chain(args, verified_bindings=None):
    """Frontend admission: require the completed operation AND its supervisor.

    The caller independently verifies opaque image/K output hashes. Here only
    JSON, frozen source bytes, metadata indices and bound process logs are read.
    """
    inner, inner_sha = bounded_json(args.input_operation)
    outer, outer_sha = bounded_json(args.input_supervision)
    require(inner.get("schema") == "evr0023.acquisition-operation.v1"
            and inner.get("status") == "completed" and inner.get("eligible") is True
            and inner.get("expected_commit") == args.expected_commit
            and inner.get("plan_sha256") == args.plan_sha256 and inner.get("sources_unchanged") is True
            and inner.get("acquisition_receipt_sha256") == args.input_receipt_sha256,
            "completed acquisition operation binding required")
    paths = source_paths(args.protocol)
    if verified_bindings is None:
        bindings = supervisor.verify_frozen(args.expected_commit, paths)
    else:
        # The frontend's offline guard forbids subprocesses after loading. Its
        # initial Git proof remains valid only while these exact bytes match.
        keys = [path.resolve().relative_to(ROOT).as_posix() for path in paths]
        require(all(key in verified_bindings for key in keys), "missing previously verified source")
        bindings = {key: sha(ROOT / key) for key in keys}
        require(bindings == {key: verified_bindings[key] for key in keys},
                "previously verified acquisition source changed")
    require(inner.get("source_sha256") == bindings, "acquisition operation source bindings")
    require(args.plan_sha256 in args.protocol.read_text(), "protocol plan adoption missing")
    require(args.input_receipt.resolve() == (args.input_operation.parent / "acquisition.json").resolve()
            and args.inputs.resolve() == (args.input_operation.parent / "inputs").resolve(),
            "acquisition output path binding")
    require(outer.get("schema") == "evr0023.supervision.v1" and outer.get("operation") == "acquisition"
            and outer.get("status") == "completed" and outer.get("exit_code") == 0
            and outer.get("expected_commit") == args.expected_commit and outer.get("termination") is None
            and outer.get("remaining_descendant_pids") == [] and outer.get("cleanup_errors") == [],
            "completed acquisition supervision required")
    keys = ["experiments/0023-mapfree/" + name for name in
            ("supervise.py", "acquire_operation.py", "range_inputs.py", "propose_plan.py")]
    require(outer.get("source_sha256") == {key: bindings[key] for key in keys},
            "acquisition supervisor source bindings")
    require(type(outer.get("wall_seconds")) in (int, float) and math.isfinite(outer["wall_seconds"])
            and 0 <= outer["wall_seconds"] < 1800 and outer.get("hard_timeout_seconds") == 1800
            and type(outer.get("peak_group_rss_kib")) is int and 0 <= outer["peak_group_rss_kib"] <= 8 * 1024**2
            and outer.get("group_rss_limit_kib") == 8 * 1024**2
            and outer.get("thread_environment") == {key: "4" for key in supervisor.THREAD_KEYS},
            "acquisition supervision resource limits")
    logs = {}
    for name in ("stdout", "stderr"):
        path = args.input_supervision.parent / (name + ".log")
        logs[name] = sha(path)
        require(logs[name] == outer.get(name + "_sha256"), "acquisition supervision log hash")
    command = outer.get("command")
    require(isinstance(command, list) and len(command) >= 4 and command[:2] ==
            ["$PYTHON", "$REPOSITORY/experiments/0023-mapfree/acquire_operation.py"]
            and len(command[2:]) % 2 == 0, "acquisition command")
    flags = dict(zip(command[2::2], command[3::2]))
    require(len(flags) * 2 == len(command[2:]) and flags.get("--expected-commit") == args.expected_commit
            and flags.get("--plan-sha256") == args.plan_sha256, "acquisition command bindings")
    def expanded(value):
        path = Path(value.replace("$REPOSITORY", str(ROOT)).replace("$PRIVATE", str(ROOT.parent)))
        return (path if path.is_absolute() else ROOT / path).resolve()
    for flag, path in {"--plan": args.plan, "--output": args.input_operation.parent,
                       "--protocol": args.protocol}.items():
        require(flag in flags and expanded(flags[flag]) == path.resolve(), "acquisition command path mismatch")
    for flag, kind in (("--members-index", "members"), ("--central-directory", "directory"),
                       ("--index-receipt", "receipt")):
        require(flag in flags and sha(expanded(flags[flag])) == inner["index_sha256"][kind],
                "acquisition directory metadata binding")
    lines = (args.input_supervision.parent / "stdout.log").read_text().splitlines()
    require(bool(lines) and json.loads(lines[-1]) == {key: inner[key] for key in
            ("status", "eligible", "wall_seconds")}, "acquisition stdout/operation mismatch")
    return dict(input_operation_sha256=inner_sha, input_supervision_sha256=outer_sha,
                input_supervision_log_sha256=logs)


def run(args):
    output = Path(args.output)
    require(not output.exists() and not output.is_symlink(), "fresh output directory required")
    require(not output.resolve().is_relative_to(ROOT), "acquired payloads must stay outside public repository")
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    receipt_path = output / "operation.json"
    acquisition_path = output / "acquisition.json"
    record = dict(schema="evr0023.acquisition-operation.v1", status="started", eligible=False,
                  stage="verification", expected_commit=args.expected_commit, plan_sha256=args.plan_sha256,
                  pose_gt_members_allowed=False, image_decoding=False, calibration_parsing=False,
                  scientific_inference=False, retries=False, sources_unchanged=None)

    def save():
        pending = output / "operation.pending"
        with pending.open("w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, receipt_path)

    save()
    try:
        plan, index, limits, bindings = verify_inputs(args)
        record.update(source_sha256=bindings, index_sha256=index.fingerprints,
                      source=dict(url=index.url, etag=index.etag, archive_bytes=index.archive_bytes),
                      selected_names=plan["frontend_acquisition"]["member_names"], limits=vars(limits),
                      protocol_adopts_private_plan=True, stage="acquisition")
        save()
        result = ranges.acquire(index, record["selected_names"], output / "inputs", acquisition_path, limits)
        record["stage"] = "postcheck"; save()
        saved, saved_sha = bounded_json(acquisition_path, 2 * 1024**2)
        require(saved == result, "returned helper record differs from durable receipt")
        record["acquisition_receipt_sha256"] = saved_sha
        validate_completed(saved, index, plan, limits, output / "inputs")
        # Re-read only metadata and frozen source bytes; no extra network work.
        final_plan, final_index, final_limits, final_bindings = verify_inputs(args)
        require(final_plan == plan and final_index == index and final_limits == limits
                and final_bindings == bindings, "bindings changed during acquisition")
        require(time.monotonic() - began <= FIXED_LIMITS["wall_seconds"], "operation wall limit exceeded")
        record.update(status="completed", eligible=True, sources_unchanged=True, stage="completed")
    except BaseException as exc:
        record.update(status="failed", eligible=False, failure_stage=record["stage"],
                      error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if acquisition_path.is_file():
            record["acquisition_receipt_sha256"] = sha(acquisition_path)
        record["wall_seconds"] = time.monotonic() - began
        save()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "plan", "members-index", "central-directory", "index-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--plan-sha256", required=True)
    result = run(parser.parse_args(argv))
    print(json.dumps({k: result[k] for k in ("status", "eligible", "wall_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
