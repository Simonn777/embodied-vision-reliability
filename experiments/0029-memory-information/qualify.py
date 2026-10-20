"""Verify a frozen installation, then run only the fixed manufactured fixture."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OWN = "experiments/0029-memory-information/"
PROTOCOL = "protocols/0029-native-runtime.md"
REVIEW = "configs/0029-sdist-build-review.json"
OLD_HELPER = "experiments/0022-frontend/qualify.py"
SUPERVISOR = "experiments/0027-affine-tangent/supervise.py"
MONITOR = "experiments/0025c-rs-noise/supervise.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def installation_contract(h, root, commit, installation, outer_path, spec_path):
    receipt_path = installation / "install-receipt.json"
    receipt, outer, spec = map(h.read_json, (receipt_path, outer_path, spec_path))
    h.require(receipt.get("schema") == "evr0029.install.v1" and receipt.get("status") == "completed"
              and receipt.get("expected_commit") == commit and receipt.get("model_invocations") == 0
              and receipt.get("scientific_input_access") is False, "installation incomplete or wrong scope")
    h.require(outer.get("schema") == "evr0026.operation.v1" and outer.get("status") == "completed"
              and outer.get("exit_code") == 0 and outer.get("termination") is None
              and outer.get("remaining_descendant_pids") == [] and outer.get("cleanup_errors") == [],
              "unclean installation supervision")
    h.require(outer.get("expected_commit") == commit and outer.get("spec_sha256") == h.sha(spec_path)
              and outer.get("output_sha256") == h.sha(receipt_path), "installation outer binding")
    h.require(outer.get("hard_timeout_seconds") == 1800 and outer.get("group_rss_limit_kib") == 8388608
              and h.number(outer.get("wall_seconds"), 0, 1800)
              and h.number(outer.get("peak_group_rss_kib"), 0, 8388608)
              and outer.get("thread_environment") == {k: "4" for k in h.THREAD_KEYS}, "installation resource limit")
    h.require(spec.get("commit") == commit and Path(spec["repo"]).resolve() == root
              and Path(spec["operation_output"]).resolve() == outer_path.parent
              and Path(spec["result_file"]).resolve() == receipt_path and spec.get("timeout") == 1800,
              "installation specification")
    command = spec.get("command")
    h.require(isinstance(command, list) and len(command) >= 3 and command[1:3] == ["-B", str(root / OWN / "install_runtime.py")],
              "installation command")
    tokens = command[3:]
    h.require(len(tokens) % 2 == 0 and len(set(tokens[::2])) == len(tokens[::2]), "installation flags")
    flags = dict(zip(tokens[::2], tokens[1::2]))
    required_flags = {"--repo-root", "--expected-commit", "--protocol", "--acquisition",
                      "--acquisition-supervision", "--acquisition-spec", "--prior-assets",
                      "--build-review", "--destination"}
    h.require(required_flags <= set(flags) <= required_flags | {"--python"}, "installation flag set")
    h.require(flags.get("--expected-commit") == commit and flags.get("--repo-root") == str(root)
              and flags.get("--destination") == str(installation)
              and flags.get("--protocol") == str(root / PROTOCOL)
              and flags.get("--build-review") == str(root / REVIEW), "installation input identity")
    expected_command = [x.replace(str(root), "$REPOSITORY").replace(str(root.parent), "$PRIVATE") for x in command]
    observed = outer.get("command")
    h.require(isinstance(observed, list) and observed[1:] == expected_command[1:]
              and observed[0] in (expected_command[0], "$PYTHON"), "outer command identity")
    for key, path in (("supervisor_sha256", SUPERVISOR), ("monitor_sha256", MONITOR)):
        h.require(outer.get(key) == hashlib.sha256(h.git_bytes(root, commit, path)).hexdigest(), "monitor source")
    h.require(outer.get("log_sha256") == {n: h.sha(outer_path.parent / n) for n in ("stdout.log", "stderr.log")}, "installation log changed")
    required_sources = {h.SELF, *h.MANIFESTS, *h.OLD_MANIFESTS, PROTOCOL, REVIEW}
    h.require(isinstance(receipt.get("source_sha256"), dict)
              and set(receipt["source_sha256"]) == required_sources, "installation source set")
    h.require(receipt["source_sha256"] == h.frozen(root, commit, required_sources), "installation source changed")
    h.require(receipt.get("build_review_sha256") == h.sha(root / REVIEW), "sdist review changed")
    h.require(receipt.get("installed_inventory") == h.inventory(installation), "installed files changed")
    return receipt, {"receipt_sha256": h.sha(receipt_path), "supervision_sha256": h.sha(outer_path),
                     "spec_sha256": h.sha(spec_path), "inventory_entries": len(receipt["installed_inventory"])}


def run(args):
    root = Path(args.repo_root).resolve()
    installation = Path(args.installation).resolve()
    out = Path(args.output_dir).absolute()
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = {"schema": "evr0029.native-qualification.v1", "status": "started", "eligible": False,
              "expected_commit": args.expected_commit, "scientific_input_access": False,
              "warnings": [], "offline_attempts": [], "compatibility_probes": [], "timings": {}}

    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")

    save()
    try:
        h = module("evr0029_install", root / OWN / "install_runtime.py")
        paths = [OWN + p for p in ("qualify.py", "native_fixture.py", "install_runtime.py")]
        paths += [PROTOCOL, REVIEW, OLD_HELPER, *h.MANIFESTS, *h.OLD_MANIFESTS, SUPERVISOR, MONITOR]
        bindings = h.frozen(root, args.expected_commit, paths)
        record["source_sha256"] = bindings
        installed, record["installation_bindings"] = installation_contract(
            h, root, args.expected_commit, installation,
            Path(args.installation_supervision).resolve(), Path(args.installation_spec).resolve())
        h.require(sys.version_info[:3] == (3, 12, 13), "CPython 3.12.13 required")
        h.require(sys.flags.no_site == 1, "launch with -S before the offline guard")
        h.require(Path(sys.executable).absolute() == installation / "venv/bin/python", "wrong interpreter")
        q = module("evr0022_qualification_helpers", root / OLD_HELPER)
        q.install_guard(record["offline_attempts"], record["compatibility_probes"])
        record["offline_guard_scope"] = "Python audit events; not an OS sandbox for arbitrary native calls"
        sys.dont_write_bytecode = True
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
        for key in h.THREAD_KEYS:
            h.require(os.environ.get(key) == "4", "thread environment")
        # The dedicated interpreter starts with -S; process installed .pth only after guarding.
        import site
        site.main()
        runtime = installation / "venv/lib/python3.12/site-packages"
        h.require(Path(sys.prefix).resolve() == (installation / "venv").resolve(), "venv prefix")
        sys.path[:0] = [str(installation / "sources/sam2"), str(installation / "sources/lightglue")]
        fixture = module("evr0029_native_fixture", root / OWN / "native_fixture.py")
        inputs = dict(qualify_helper=root / OLD_HELPER, sam2_source=installation / "sources/sam2",
                      lightglue_source=installation / "sources/lightglue", runtime=runtime,
                      sam2_checkpoint=installation / "weights/sam2.1_hiera_tiny.pt",
                      superpoint_checkpoint=installation / "weights/superpoint_v1.pth",
                      lightglue_checkpoint=installation / "weights/superpoint_lightglue.pth")
        result = fixture.run(inputs, out / "native-fixture", record)
        h.require(result.get("status") == "completed", "native fixture incomplete")
        h.require(not record["offline_attempts"], "forbidden network/process attempt")
        h.require(installed["installed_inventory"] == h.inventory(installation), "installed input mutated")
        # No git child is needed after the guard: compare the previously frozen bytes directly.
        h.require(all(h.sha(root / rel) == digest for rel, digest in bindings.items()), "source changed")
        record.update(status="completed", eligible=True)
    except BaseException as error:
        record.update(status="failed", eligible=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
        print(json.dumps({k: record[k] for k in ("status", "eligible", "wall_seconds")}), flush=True)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("repo-root", "expected-commit", "installation", "installation-supervision", "installation-spec", "output-dir"):
        p.add_argument("--" + arg, required=True)
    run(p.parse_args())
