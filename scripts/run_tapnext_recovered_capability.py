"""Separate recovery supervisor; preserves the failed v1 preparation unchanged.

Only stdlib imports occur here. Verification is evidence consistency, not an
attestation of the cloud operator. No network, installation or scientific logic
is implemented in this supervisor; the unchanged frozen adapter owns the run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import subprocess
import sys
import time

ORIGINAL_SUPERVISOR_SHA = "3475e043dd81efc843ba948b0a3c298b01627503681ed6c9a1c6a94d38f2c5a5"
ORIGINAL_FREEZE_SHA = "3e5aad0b0123affb3991997607c04310bd7d08a38172b8ffbc2c0f29b01f36ef"
ORIGINAL_INPUTS = (
    "configs/0005-native-capability.json", "protocols/0005-native-capability.md",
    "scripts/tapnext_native_adapter.py", "scripts/prepare_tapnext_assets.py",
    "scripts/run_tapnext_capability.py",
    "results/0003-network-followup-20261017/acquisition-ledger.json",
    "results/0002-development/decode/development-manifest.json",
)
CONFIG_PATH = "configs/0005-native-recovery.json"
FREEZE_PATH = "protocols/0005-native-recovery-freeze.json"
INPUTS = ORIGINAL_INPUTS + (
    "protocols/0005-native-capability-freeze.json", CONFIG_PATH,
    "protocols/0005-native-recovery.md", "scripts/prepare_tapnext_recovery.py",
    "scripts/run_tapnext_recovered_capability.py",
    "results/0005-native-capability/preparation-dependencies/preparation-dependencies.json",
    "results/0005-native-capability/review/postfailure-cache-inventory.json",
    "results/0005-native-capability/review/postfailure-overlay-manifest.json",
    "results/0005-native-capability/review/failure-review.json",
)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
NATIVE_MARKER = "0005-recovered-native-started.json"
LAUNCH_MARKER = "0005-recovered-supervisor-started.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def local(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("Unsafe relative artifact path")
    path = Path(root)
    if path.is_symlink():
        raise ValueError("Artifact root symlink")
    for part in rel.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Artifact symlink")
    return path


def load_json(path):
    return json.loads(path.read_text())


def verify_freeze(root, expected_sha):
    """Reuse the immutable v1 current/index/source verifier for both freezes."""
    path = local(root, "scripts/run_tapnext_capability.py")
    if digest(path) != ORIGINAL_SUPERVISOR_SHA:
        raise ValueError("Original supervisor changed")
    spec = importlib.util.spec_from_file_location("original_frozen_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_freeze(root, ORIGINAL_FREEZE_SHA)
    # This fresh private module is not shared; only the verifier's input tuple
    # and receipt path change. The original file and its seven inputs do not.
    module.INPUTS, module.FREEZE_PATH = INPUTS, FREEZE_PATH
    return module.verify_freeze(root, expected_sha)


def file_inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a regular directory")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Inventory symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Nonregular inventory entry")
        files[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size, "sha256": digest(path)}
    return files


def absent_native_markers(failed, recovery):
    # lexists behavior includes dangling links; they must never permit a retry.
    for path in (failed / "0005-native-started.json", recovery / NATIVE_MARKER):
        if path.exists() or path.is_symlink():
            raise ValueError("Existing native attempt marker; inspect, never repeat")


def validate_output(output, old, failed, recovery):
    resolved = output.resolve()
    protected = (old, failed, recovery / "source", recovery / "weights", recovery / "markers")
    if any(resolved == p or p in resolved.parents or resolved in p.parents for p in protected):
        raise ValueError("Output overlaps protected cache inputs")


def closed_child(record, recovery, prefix, seconds):
    if record != load_json(local(recovery, prefix + ".json")):
        raise ValueError("Recovery child receipt differs")
    elapsed = record.get("elapsed_seconds")
    if (type(record.get("exit_code")) is not int or record["exit_code"] != 0
            or type(record.get("raw_wait_status")) is not int or record["raw_wait_status"] != 0
            or record.get("terminated_by_parent") or not record.get("started_utc")
            or not record.get("finished_utc") or not isinstance(record.get("command"), list)
            or not record["command"] or type(elapsed) not in (int, float)
            or not math.isfinite(elapsed) or not 0 <= elapsed <= seconds):
        raise ValueError("Recovery child did not finish successfully")
    for stream in ("stdout", "stderr"):
        if digest(local(recovery, prefix + "." + stream + ".log")) != record.get(stream + "_sha256"):
            raise ValueError("Recovery child log differs")


def verify_recovery_preparation(root, old, failed, recovery):
    """Recheck the original failure, unchanged cache, and new closed receipts."""
    cfg_sha = digest(root / CONFIG_PATH)
    cfg = load_json(root / CONFIG_PATH)["recovery"]
    src_sha = digest(root / "scripts/prepare_tapnext_recovery.py")
    for ref in (cfg["original_config"], cfg["frozen_preparer"], *cfg["evidence"].values()):
        if digest(local(root, ref["path"])) != ref["sha256"]:
            raise ValueError("Bound recovery input differs")
    original = load_json(local(root, cfg["original_config"]["path"]))
    prep = original["preparation"]
    evidence = {k: load_json(local(root, v["path"])) for k, v in cfg["evidence"].items()}
    failure, review = evidence["failure_receipt"], evidence["failure_review"]
    if (failure.get("status") != "failed" or failure.get("stage") != "dependencies"
            or failure.get("config_sha256") != cfg["original_config"]["sha256"]
            or failure.get("source_sha256") != cfg["frozen_preparer"]["sha256"]
            or review.get("native") != "not_started" or review.get("checkpoint") != "not_started"
            or review.get("import_check") != "not_started"):
        raise ValueError("Expected preserved pre-native dependency failure")
    if file_inventory(failed) != evidence["cache_inventory"]:
        raise ValueError("Original failed cache was changed")
    if digest(local(failed, "preparation-dependencies.json")) != cfg["evidence"]["failure_receipt"]["sha256"]:
        raise ValueError("Failed receipt differs from published evidence")
    old_preservation = {**prep["runtime"]["old_init_sha256"], **review["old_markers_preserved"]}
    for name, want in old_preservation.items():
        if digest(local(old, name)) != want:
            raise ValueError("Old runtime or prior marker changed")
    expected_paths = {"old_cache": str(old), "failed_cache": str(failed),
                      "recovery_root": str(recovery), "source_root": str(recovery / "source"),
                      "overlay": str(failed / "overlay")}
    bindings = {"config_sha256": cfg_sha, "source_sha256": src_sha,
                "original_config_sha256": cfg["original_config"]["sha256"],
                "frozen_preparer_sha256": cfg["frozen_preparer"]["sha256"],
                "prior_failure_receipt_sha256": cfg["evidence"]["failure_receipt"]["sha256"],
                "prior_cache_inventory_sha256": cfg["evidence"]["cache_inventory"]["sha256"],
                "prior_overlay_inventory_sha256": cfg["evidence"]["overlay_inventory"]["sha256"]}
    receipts = {}
    for stage in ("dependencies", "checkpoint"):
        path = local(recovery, "recovery-" + stage + ".json")
        item = load_json(path)
        if (item.get("schema_version") != 1 or item.get("stage") != stage + "-recovery"
                or item.get("status") != "completed" or item.get("paths") != expected_paths
                or any(item.get(k) != v for k, v in bindings.items())
                or item.get("preserved_failed_cache") is not True
                or item.get("old_preservation") != old_preservation
                or item.get("affinity") != original["cpu_affinity"]
                or item.get("AS_bytes") != original["address_space_bytes"]
                or item.get("model_constructed") is not False
                or not item.get("started_utc") or not item.get("finished_utc")):
            raise ValueError("Completed bound recovery receipt required: " + stage)
        marker_path = local(recovery, "markers/" + stage + ".started.json")
        marker = load_json(marker_path)
        if (digest(marker_path) != item.get("marker_sha256")
                or marker.get("status") != "running" or marker.get("stage") != stage + "-recovery"
                or marker.get("paths") != expected_paths
                or any(marker.get(k) != v for k, v in bindings.items())):
            raise ValueError("Recovery stage marker differs")
        receipts[stage] = {"record": item, "sha256": digest(path)}
    dep, ckpt = receipts["dependencies"]["record"], receipts["checkpoint"]["record"]
    if (dep.get("new_network_payload_bytes") != 0 or dep.get("checkpoint_started") is not False
            or dep.get("model_constructed") is not False
            or dep.get("affinity") != original["cpu_affinity"]
            or dep.get("AS_bytes") != original["address_space_bytes"]):
        raise ValueError("Recovery dependency scope or limits differ")
    closed_child(dep["import_check"], recovery, "import-check", prep["import_timeout_seconds"])
    probe = load_json(local(recovery, "import-check.stdout.log"))
    if probe.get("status") != "passed" or probe.get("no_model_constructed_or_forward_called") is not True:
        raise ValueError("Expected successful import-only result")
    assets = prep["assets"]
    dependencies = [a for a in assets if a["kind"] != "checkpoint"]
    expected_assets = [{"id": a["id"], "relative_path": a["relative_path"], "bytes": a["bytes"],
                        "sha256": a["sha256"], "reused_verified": True} for a in dependencies]
    if dep["assets"] != expected_assets:
        raise ValueError("Recovery must reuse all original dependency assets")
    for asset in dependencies:
        path = local(failed, asset["relative_path"])
        if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
            raise ValueError("Reused dependency asset differs")
    expected_source = {a["relative_path"][len("source/"):]: {"bytes": a["bytes"], "sha256": a["sha256"]}
                       for a in dependencies if a["kind"] == "source"}
    for kind, directory, wanted in (("source", recovery / "source", expected_source),
                                    ("overlay", failed / "overlay", evidence["overlay_inventory"]["files"])):
        ref = dep[kind + "_manifest"]
        if ref.get("relative_path") != kind + "-manifest.json":
            raise ValueError("Unexpected recovery manifest path")
        path = local(recovery, ref["relative_path"])
        manifest = load_json(path)
        if (digest(path) != ref["sha256"] or manifest.get("schema_version") != 1
                or manifest.get("config_sha256") != cfg_sha or manifest.get("source_sha256") != src_sha
                or manifest.get("files") != wanted or file_inventory(directory) != wanted):
            raise ValueError("Recovery " + kind + " manifest differs")
    if ckpt.get("dependency_receipt_sha256") != receipts["dependencies"]["sha256"]:
        raise ValueError("Checkpoint dependency receipt binding differs")
    checkpoints = [a for a in assets if a["kind"] == "checkpoint"]
    if len(checkpoints) != 1 or ckpt.get("checkpoint_asset", {}).get("asset") != checkpoints[0]:
        raise ValueError("Expected the single original checkpoint")
    if ckpt.get("checkpoint_started") is not True or any(
            ckpt.get(kind + "_manifest") != dep[kind + "_manifest"] for kind in ("source", "overlay")):
        raise ValueError("Checkpoint recovery scope or manifest binding differs")
    asset = checkpoints[0]
    path = local(recovery, asset["relative_path"])
    if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
        raise ValueError("Recovered checkpoint bytes differ")
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("Checkpoint partial still exists after reported completion")
    # The remaining checkpoint record and inherited cumulative ledger checks are
    # deliberately separate from the dependency failure's preserved ledger.
    verify_checkpoint_record(recovery, failed, dep, ckpt, asset, cfg_sha, src_sha, prep)
    return {stage: value["sha256"] for stage, value in receipts.items()}


def verify_checkpoint_record(recovery, failed, dep, item, asset, cfg_sha, src_sha, prep):
    path = local(recovery, "asset-records/" + asset["id"] + ".json")
    record = load_json(path)
    if (digest(path) != item.get("checkpoint_asset_record_sha256")
            or record != item.get("checkpoint_asset")
            or record.get("asset") != asset or record.get("status") != "downloaded_verified"
            or record.get("config_sha256") != cfg_sha or record.get("source_sha256") != src_sha
            or record.get("sha256") != asset["sha256"] or record.get("partial_bytes_stored") != 0
            or record.get("http_payload_attempts") != 1
            or record.get("payload_bytes_observed") != asset["bytes"]
            or record.get("observed_payload_sha256") != asset["sha256"]
            or record.get("partial_exists") is not False
            or not record.get("finished_utc") or item.get("new_network_payload_bytes") != asset["bytes"]):
        raise ValueError("Expected one completed first checkpoint acquisition")
    ref = item["ledger_snapshot"]
    if ref.get("relative_path") != "acquisition-ledger-checkpoint.json":
        raise ValueError("Unexpected checkpoint ledger path")
    path = local(recovery, ref["relative_path"])
    ledger = load_json(path)
    previous = load_json(local(failed, "acquisition-ledger.json"))
    budget = prep["budget"]
    base = budget["historical_ceiling_bytes"] + budget["metadata_allowance_bytes"]
    expected_reservations = {a["id"]: a["bytes"] + 1 for a in prep["assets"] if a["kind"] != "checkpoint"}
    if previous.get("reservations") != expected_reservations:
        raise ValueError("Original dependency reservations differ")
    dep_ref = dep["ledger_snapshot"]
    dep_path = local(recovery, "acquisition-ledger-dependencies.json")
    expected_dep = dict(previous)
    expected_dep.update(inherited_ledger_sha256=digest(local(failed, "acquisition-ledger.json")),
                        prior_config_sha256=previous["config_sha256"],
                        prior_source_sha256=previous["source_sha256"],
                        config_sha256=cfg_sha, source_sha256=src_sha, new_payload_bytes_observed=0)
    if (dep_ref.get("relative_path") != "acquisition-ledger-dependencies.json"
            or digest(dep_path) != dep_ref.get("sha256") or load_json(dep_path) != expected_dep):
        raise ValueError("Recovery dependency ledger does not preserve prior accounting")
    expected_reservations[asset["id"]] = asset["bytes"] + 1
    planned = base + sum(expected_reservations.values())
    if (digest(path) != ref["sha256"] or planned >= budget["limit_bytes"]
            or ledger.get("config_sha256") != cfg_sha or ledger.get("source_sha256") != src_sha
            or ledger.get("base_ceiling_bytes") != base or ledger.get("planned_ceiling_bytes") != planned
            or ledger.get("limit_bytes") != budget["limit_bytes"]
            or ledger.get("reservations") != expected_reservations
            or any(ledger.get(k) != expected_dep[k] for k in (
                "inherited_ledger_sha256", "prior_config_sha256", "prior_source_sha256"))
            or ledger.get("new_payload_bytes_observed") != asset["bytes"]
            or ledger.get("payload_bytes_observed") != previous["payload_bytes_observed"] + asset["bytes"]
            or ledger.get("conservative_cumulative_ceiling_bytes") != planned
            or load_json(local(recovery, "acquisition-ledger.json")) != ledger):
        raise ValueError("Recovery cumulative acquisition ledger differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-cache", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--bootstapir-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze = verify_freeze(root, args.freeze_sha256)
    config = load_json(root / ORIGINAL_INPUTS[0])
    if args.verify_only:
        print(json.dumps({"status": "recovery_freeze_verified", "files": len(INPUTS),
                          "source_commit": freeze["source_commit"], "preparation_checked": False}))
        return 0
    if any(p.is_symlink() for p in (args.failed_cache, args.recovery_root, args.bootstapir_cache)):
        raise ValueError("Cache root symlink")
    failed, recovery, old = (p.resolve() for p in (args.failed_cache, args.recovery_root, args.bootstapir_cache))
    if len({failed, recovery, old}) != 3 or any(a in b.parents for a in (failed, recovery, old) for b in (failed, recovery, old) if a != b):
        raise ValueError("Cache roots must be distinct, non-nested directories")
    validate_output(args.output, old, failed, recovery)
    absent_native_markers(failed, recovery)
    preparation = verify_recovery_preparation(root, old, failed, recovery)
    marker = recovery / NATIVE_MARKER
    launch_path = local(recovery, LAUNCH_MARKER)
    launch = {"freeze_sha256": args.freeze_sha256, "source_commit": freeze["source_commit"],
              "preparation_receipt_sha256": preparation, "output": str(args.output.resolve())}
    if args.child:
        saved_launch = load_json(launch_path)
        if any(saved_launch.get(k) != v for k, v in launch.items()):
            raise ValueError("Child requires the matching exclusive supervisor launch marker")
        spec = importlib.util.spec_from_file_location("frozen_tapnext_adapter", root / "scripts/tapnext_native_adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run_capability(
            checkpoint=recovery / "weights/tapnextpp_ckpt.pt", development_npz=old / "development-000.npz",
            source_root=recovery / "source", output=args.output / "run", marker=marker,
            expected_packages=config["expected_packages"], overlay_root=failed / "overlay",
            base_runtime_root=old / "runtime-fixed",
            provenance={"freeze_sha256": args.freeze_sha256, "original_freeze_sha256": ORIGINAL_FREEZE_SHA,
                        "source_commit": freeze["source_commit"], "preparation_receipt_sha256": preparation})
        print(json.dumps({"status": result["status"], "frames_completed": result["frames_completed"]}))
        return 0 if result["status"] == "completed_native_capability" else 1
    if launch_path.exists() or launch_path.is_symlink():
        raise ValueError("Recovery supervisor already attempted; inspect, never repeat")
    args.output.mkdir(parents=True, exist_ok=False)
    if sys.platform != "linux":
        raise ValueError("Cloud Linux supervisor required")
    if not set(config["cpu_affinity"]).issubset(os.sched_getaffinity(0)):
        raise ValueError("Fixed four-CPU affinity unavailable")
    os.sched_setaffinity(0, config["cpu_affinity"])
    resource.setrlimit(resource.RLIMIT_AS, (config["address_space_bytes"],) * 2)
    command = ["timeout", "--kill-after=" + str(config["kill_after_seconds"]) + "s",
               str(config["outer_timeout_seconds"]), sys.executable, str(Path(__file__).resolve()),
               "--child", "--failed-cache", str(failed), "--recovery-root", str(recovery),
               "--bootstapir-cache", str(old), "--output", str(args.output.resolve()),
               "--freeze-sha256", args.freeze_sha256]
    env = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": str(old / "runtime-fixed") + ":" + str(old / "runtime")}
    env.update({name: str(config["threads"]) for name in THREAD_VARS})
    receipt = {"status": "started", "started_utc": now(), "command": command,
               "source_commit": freeze["source_commit"], "source_sha256": freeze["files"],
               "freeze_sha256": args.freeze_sha256, "original_freeze_sha256": ORIGINAL_FREEZE_SHA,
               "preparation_receipt_sha256": preparation,
               "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "environment": {**env, "PATH": "existing runtime PATH omitted"},
               "limits": {"affinity": config["cpu_affinity"], "AS_bytes": config["address_space_bytes"],
                          "timeout_seconds": config["outer_timeout_seconds"], "kill_after_seconds": config["kill_after_seconds"]},
               "exit_code": None, "raw_wait_status": None,
               "boundary": "parent wait4 observes timeout/child; retain the outer command exit separately"}
    with launch_path.open("x") as stream:
        json.dump({**launch, "started_utc": receipt["started_utc"], "pid": os.getpid()}, stream, indent=2)
        stream.write("\n")
    receipt["launch_marker_sha256"] = digest(launch_path)
    save(args.output / "outer.json", receipt)
    start = time.perf_counter()
    with (args.output / "stdout.log").open("xb") as stdout, (args.output / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
    receipt.update(status="finished", ended_utc=now(), wall_seconds=time.perf_counter() - start,
                   exit_code=child.returncode, raw_wait_status=status, peak_rss_kib=usage.ru_maxrss,
                   CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
                   stdout_sha256=digest(args.output / "stdout.log"), stderr_sha256=digest(args.output / "stderr.log"))
    save(args.output / "outer.json", receipt)
    print(json.dumps(receipt, indent=2))
    return child.returncode if 0 <= child.returncode <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
