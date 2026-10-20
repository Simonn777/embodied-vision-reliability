"""Acquire only frozen 0029 new assets; no installation or implicit network call.

Existing installer inputs are outside these batches. Source manifests and this
wrapper must match the reviewed commit before any network operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import types

MANIFESTS = ("configs/0029-sam2-assets.json", "configs/0029-runtime-assets.json")
DIRECTORIES = ("sam2-assets", "runtime-assets")
PROTOCOL = "protocols/0029-runtime-assets.md"
HELPER = "experiments/0022-frontend/acquire.py"
HELPER_SHA256 = "5b09d48b3b81fcf1d62b90871475813e26544934711244dcff2d429d4ad7fa11"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen_inputs(root, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a complete lowercase commit SHA")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != commit:
        raise ValueError("Unexpected source commit")
    own_path = Path(__file__).resolve().relative_to(root).as_posix()
    bound = {}
    for relative in (*MANIFESTS, PROTOCOL, HELPER, own_path):
        path = root / relative
        if path.is_symlink():
            raise ValueError("Symlinked source: " + relative)
        actual = path.read_bytes()
        if actual != subprocess.check_output(["git", "show", commit + ":" + relative], cwd=root):
            raise ValueError("Frozen source mismatch: " + relative)
        bound[relative] = actual
    if sha(bound[HELPER]) != HELPER_SHA256:
        raise ValueError("Immutable 0022 helper changed")
    return bound


def run(args):
    root, out = Path(args.repo_root).resolve(), Path(args.outputdir).absolute()
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0029.asset-operation.v1", status="started",
                  source_commit=args.expected_commit, asset_or_dependency_only=True,
                  benchmark_payload_access=False, bindings={}, batches=[])
    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "operation.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "operation.json")
    save()
    try:
        bound = frozen_inputs(root, args.expected_commit)
        record["bindings"] = {p: sha(b) for p, b in bound.items()}
        helper = types.ModuleType("evr0029_verified_acquire")
        exec(compile(bound[HELPER], str(root / HELPER), "exec"), helper.__dict__)
        manifests = [json.loads(bound[p]) for p in MANIFESTS]
        record["body_cap_bytes"] = sum(helper.validate(m) for m in manifests)
        save()
        for path, directory, manifest in zip(MANIFESTS, DIRECTORIES, manifests):
            batch = dict(manifest=path, directory=directory, status="started")
            record["batches"].append(batch)
            save()
            result = helper.acquire_all(manifest, out / directory)
            ledger_path = out / directory / "ledger.json"
            if result.get("status") != "completed" or json.loads(ledger_path.read_bytes()) != result:
                raise ValueError("Batch did not produce a matching completed ledger")
            batch.update(status="completed", actual_body_bytes=result["actual_body_bytes"],
                         ledger_sha256=sha(ledger_path.read_bytes()))
            save()
        if frozen_inputs(root, args.expected_commit) != bound:
            raise ValueError("Source binding changed during acquisition")
        record["actual_body_bytes"] = sum(b["actual_body_bytes"] for b in record["batches"])
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for batch in record["batches"]:
            ledger = out / batch["directory"] / "ledger.json"
            if ledger.is_file():
                batch["ledger_sha256"] = sha(ledger.read_bytes())
            if batch["status"] == "started":
                batch["status"] = "failed"
        save()
        print(json.dumps({k: record[k] for k in ("status", "wall_seconds", "batches")}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--outputdir", required=True)
    run(parser.parse_args())
