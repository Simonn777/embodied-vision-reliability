"""Frozen-source supervisor for a single native TAPNext++ CPU capability attempt."""
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


INPUTS = (
    "configs/0005-native-capability.json",
    "protocols/0005-native-capability.md",
    "scripts/tapnext_native_adapter.py",
    "scripts/prepare_tapnext_assets.py",
    "scripts/run_tapnext_capability.py",
    "results/0003-network-followup-20261017/acquisition-ledger.json",
    "results/0002-development/decode/development-manifest.json",
)
FREEZE_PATH = "protocols/0005-native-capability-freeze.json"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_freeze(root, expected_sha):
    path = root / FREEZE_PATH
    if digest(path) != expected_sha:
        raise ValueError("Published freeze receipt differs")
    freeze = json.loads(path.read_text())
    if set(freeze["files"]) != set(INPUTS):
        raise ValueError("Freeze has missing or extra execution inputs")
    source = freeze["source_commit"]
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("Expected exact source commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"],
                   cwd=root, check=True)
    for ref in ("HEAD", ":", source):
        paths = list(INPUTS) + ([FREEZE_PATH] if ref in ("HEAD", ":") else [])
        for name in paths:
            spec = ":" + name if ref == ":" else ref + ":" + name
            blob = subprocess.check_output(["git", "show", spec], cwd=root)
            want = expected_sha if name == FREEZE_PATH else freeze["files"][name]
            if hashlib.sha256(blob).hexdigest() != want or digest(root / name) != want:
                raise ValueError("Frozen current/index/commit input differs: " + name)
    return freeze


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verify_preparation(root, cache):
    """Check closed preparation evidence, configured assets and installed bytes.

    These are consistency checks on preserved evidence, not signatures proving
    who executed a command. No runtime package is imported by this gate.
    """
    config_sha = digest(root / INPUTS[0])
    source_sha = digest(root / "scripts/prepare_tapnext_assets.py")
    config = json.loads((root / INPUTS[0]).read_text())["preparation"]

    def local(name):
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("Unsafe preparation artifact path")
        path = cache
        for part in rel.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("Preparation artifact symlink")
        return path

    def bound(record):
        return (record.get("config_sha256") == config_sha
                and record.get("source_sha256") == source_sha)

    def closed_child(record, prefix, seconds):
        if record != json.loads(local(prefix + ".json").read_text()):
            raise ValueError("Preparation child receipt differs: " + prefix)
        elapsed = record.get("elapsed_seconds")
        if (type(record.get("exit_code")) is not int or record["exit_code"] != 0
                or type(record.get("raw_wait_status")) is not int or record["raw_wait_status"] != 0
                or record.get("terminated_by_parent")
                or not record.get("started_utc") or not record.get("finished_utc")
                or not isinstance(record.get("command"), list) or not record["command"]
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or not 0 <= elapsed <= seconds):
            raise ValueError("Preparation child did not finish successfully: " + prefix)
        for stream in ("stdout", "stderr"):
            if digest(local(prefix + "." + stream + ".log")) != record.get(stream + "_sha256"):
                raise ValueError("Preparation child log differs: " + prefix)

    receipts = {}
    for stage in ("dependencies", "checkpoint"):
        path = local("preparation-" + stage + ".json")
        item = json.loads(path.read_text())
        if (item.get("status") != "completed" or item.get("stage") != stage
                or item.get("schema_version") != 1 or not bound(item)):
            raise ValueError("Successful frozen preparation receipt required: " + stage)
        receipts[stage] = {"sha256": digest(path), "record": item}
    dep, ckpt = receipts["dependencies"], receipts["checkpoint"]
    if ckpt["record"].get("dependency_receipt_sha256") != dep["sha256"]:
        raise ValueError("Checkpoint preparation dependency binding differs")
    closed_child(dep["record"]["installation"], "install", config["install_timeout_seconds"])
    closed_child(dep["record"]["import_check"], "import-check", config["import_timeout_seconds"])

    budget = config["budget"]
    base = budget["historical_ceiling_bytes"] + budget["metadata_allowance_bytes"]
    assets = config["assets"]
    if len({a["id"] for a in assets}) != len(assets):
        raise ValueError("Duplicate configured asset")
    planned = base + sum(a["bytes"] + 1 for a in assets)
    if planned >= budget["limit_bytes"]:
        raise ValueError("Frozen acquisition plan exceeds cumulative budget")
    reservations, payload = {}, 0
    for stage in ("dependencies", "checkpoint"):
        item = receipts[stage]["record"]
        expected = [a for a in assets if (a["kind"] == "checkpoint") == (stage == "checkpoint")]
        records = item["assets"]
        if len(records) != len(expected) or [a["asset"] for a in records] != expected:
            raise ValueError("Preparation assets differ: " + stage)
        for asset, record in zip(expected, records):
            downloaded = record.get("status") == "downloaded_verified"
            if (record.get("status") not in ("downloaded_verified", "reused_verified")
                    or not bound(record) or record.get("sha256") != asset["sha256"]
                    or record.get("partial_bytes_stored") != 0
                    or record.get("http_payload_attempts") != int(downloaded)
                    or record.get("payload_bytes_observed") != (asset["bytes"] if downloaded else 0)
                    or record.get("observed_payload_sha256") != (
                        asset["sha256"] if downloaded else hashlib.sha256(b"").hexdigest())
                    or not record.get("finished_utc")):
                raise ValueError("Unverified acquisition record: " + asset["id"])
            if record != json.loads(local("asset-records/" + asset["id"] + ".json").read_text()):
                raise ValueError("Acquisition record differs: " + asset["id"])
            path = local(asset["relative_path"])
            if not path.is_file() or path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
                raise ValueError("Prepared asset bytes differ: " + asset["id"])
            reservations[asset["id"]] = asset["bytes"] + 1
            payload += record["payload_bytes_observed"]
        ref = item["ledger_snapshot"]
        if ref.get("relative_path") != "acquisition-ledger-" + stage + ".json":
            raise ValueError("Unexpected ledger snapshot path")
        ledger_path = local(ref["relative_path"])
        if digest(ledger_path) != ref["sha256"] or item.get("ledger_sha256") != ref["sha256"]:
            raise ValueError("Preparation ledger hash differs")
        ledger = json.loads(ledger_path.read_text())
        if (not bound(ledger) or ledger.get("base_ceiling_bytes") != base
                or ledger.get("planned_ceiling_bytes") != planned
                or ledger.get("limit_bytes") != budget["limit_bytes"]
                or ledger.get("reservations") != reservations
                or ledger.get("payload_bytes_observed") != payload
                or ledger.get("conservative_cumulative_ceiling_bytes") != base + sum(reservations.values())):
            raise ValueError("Preparation cumulative ledger differs: " + stage)
    if json.loads(local("acquisition-ledger.json").read_text()) != ledger:
        raise ValueError("Current acquisition ledger changed after preparation")
    manifest_ref = dep["record"]["overlay_manifest"]
    if (manifest_ref.get("relative_path") != "overlay-manifest.json"
            or ckpt["record"].get("overlay_manifest") != manifest_ref):
        raise ValueError("Preparation overlay references differ")
    manifest_path = local("overlay-manifest.json")
    if digest(manifest_path) != manifest_ref["sha256"]:
        raise ValueError("Installed overlay manifest differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not bound(manifest):
        raise ValueError("Overlay provenance differs")
    overlay = cache / "overlay"
    if overlay.is_symlink() or not overlay.is_dir():
        raise ValueError("Expected regular overlay directory")
    observed = {}
    for path in overlay.rglob("*"):
        if path.is_symlink():
            raise ValueError("Installed overlay symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Unexpected overlay entry")
        observed[path.relative_to(overlay).as_posix()] = {
            "bytes": path.stat().st_size, "sha256": digest(path)}
    if (not observed or not any(name.endswith(".dist-info/RECORD") for name in observed)
            or observed != manifest["files"]):
        raise ValueError("Installed dependency bytes missing, added or changed")
    return {stage: item["sha256"] for stage, item in receipts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-cache", type=Path, required=True)
    parser.add_argument("--bootstapir-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze = verify_freeze(root, args.freeze_sha256)
    config = json.loads((root / INPUTS[0]).read_text())
    if args.verify_only:
        print(json.dumps({"status": "freeze_verified", "files": len(INPUTS),
                          "source_commit": freeze["source_commit"]}))
        return 0
    cache = args.asset_cache.resolve()
    old = args.bootstapir_cache.resolve()
    marker = cache / "0005-native-started.json"
    preparation = verify_preparation(root, cache)
    if args.child:
        spec = importlib.util.spec_from_file_location(
            "frozen_tapnext_adapter", root / "scripts/tapnext_native_adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run_capability(
            checkpoint=cache / "weights/tapnextpp_ckpt.pt",
            development_npz=old / "development-000.npz",
            source_root=cache / "source", output=args.output / "run", marker=marker,
            expected_packages=config["expected_packages"], overlay_root=cache / "overlay",
            base_runtime_root=old / "runtime-fixed",
            provenance={"freeze_sha256": args.freeze_sha256,
                        "source_commit": freeze["source_commit"],
                        "preparation_receipt_sha256": preparation})
        print(json.dumps({"status": result["status"],
                          "frames_completed": result["frames_completed"]}))
        return 0 if result["status"] == "completed_native_capability" else 1
    if marker.exists():
        raise ValueError("Existing scientific attempt marker; inspect, never repeat")
    args.output.mkdir(parents=True, exist_ok=False)
    if sys.platform != "linux":
        raise ValueError("Cloud Linux supervisor required")
    if not {0, 1, 2, 3}.issubset(os.sched_getaffinity(0)):
        raise ValueError("Fixed four-CPU affinity unavailable")
    os.sched_setaffinity(0, [0, 1, 2, 3])
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3,) * 2)
    command = ["timeout", "--kill-after=5s", "1800", sys.executable,
               str(Path(__file__).resolve()), "--child", "--asset-cache", str(cache),
               "--bootstapir-cache", str(old), "--output", str(args.output.resolve()),
               "--freeze-sha256", args.freeze_sha256]
    env = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(old / "runtime-fixed") + ":" + str(old / "runtime")}
    env.update({name: "4" for name in THREAD_VARS})
    receipt = {"status": "started", "started_utc": now(), "command": command,
               "source_commit": freeze["source_commit"], "source_sha256": freeze["files"],
               "freeze_sha256": args.freeze_sha256,
               "preparation_receipt_sha256": preparation,
               "checkout_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "environment": {**env, "PATH": "existing runtime PATH omitted"},
               "limits": {"affinity": [0, 1, 2, 3], "AS_bytes": 8 * 1024**3,
                          "timeout_seconds": 1800, "kill_after_seconds": 5},
               "exit_code": None, "raw_wait_status": None,
               "boundary": "parent wait4 observes timeout/child; outer command exit must also be retained"}
    save(args.output / "outer.json", receipt)
    start = time.perf_counter()
    with (args.output / "stdout.log").open("xb") as stdout, (args.output / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
    receipt.update(status="finished", ended_utc=now(), wall_seconds=time.perf_counter() - start,
                   exit_code=child.returncode, raw_wait_status=status,
                   peak_rss_kib=usage.ru_maxrss, CPU_user_seconds=usage.ru_utime,
                   CPU_system_seconds=usage.ru_stime,
                   stdout_sha256=digest(args.output / "stdout.log"),
                   stderr_sha256=digest(args.output / "stderr.log"))
    save(args.output / "outer.json", receipt)
    print(json.dumps(receipt, indent=2))
    return child.returncode if 0 <= child.returncode <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
