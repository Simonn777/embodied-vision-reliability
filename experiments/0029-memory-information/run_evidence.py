"""Admit qualified native runtime and RGB/initial-mask inputs before matching.

No SAM sequence or query annotation is used by this first necessary screen.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OWN = "experiments/0029-memory-information/"
PROTOCOL = "protocols/0029-signed-memory-diagnostic.md"
MANIFEST = "configs/0029-davis-development.json"
SUPERVISOR = "experiments/0027-affine-tangent/supervise.py"
MONITOR = "experiments/0025c-rs-noise/supervise.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def snapshot(h, paths):
    return {str(path): h.sha(path) for path in paths}


def unchanged(h, fingerprints):
    h.require(all(h.sha(Path(path)) == digest for path, digest in fingerprints.items()),
              "admitted file changed during evidence run")


def distinct_media_roots(h, rgb, initial):
    h.require(rgb != initial and rgb not in initial.parents and initial not in rgb.parents,
              "RGB and initial supervision must have disjoint roots")


def outer_contract(h, root, commit, spec_path, receipt_path, command):
    spec = h.read_json(spec_path)
    outer_path = Path(spec["operation_output"]) / "receipt.json"
    outer = h.read_json(outer_path)
    h.require(spec.get("commit") == commit and Path(spec["repo"]).resolve() == root
              and Path(spec["result_file"]).resolve() == receipt_path.resolve()
              and spec.get("command") == command and spec.get("timeout") == 1800,
              "operation specification identity")
    expected = [x.replace(str(root), "$REPOSITORY").replace(str(root.parent), "$PRIVATE") for x in command]
    observed = outer.get("command")
    h.require(isinstance(observed, list) and observed[1:] == expected[1:]
              and observed[0] in (expected[0], "$PYTHON"), "outer command identity")
    h.require(outer.get("schema") == "evr0026.operation.v1"
              and outer.get("status") == "completed" and outer.get("exit_code") == 0
              and outer.get("termination") is None
              and outer.get("remaining_descendant_pids") == []
              and outer.get("cleanup_errors") == [], "unclean operation")
    h.require(outer.get("expected_commit") == commit
              and outer.get("spec_sha256") == h.sha(spec_path)
              and outer.get("output_sha256") == h.sha(receipt_path), "outer artifact binding")
    h.require(outer.get("hard_timeout_seconds") == 1800
              and outer.get("group_rss_limit_kib") == 8388608
              and h.number(outer.get("wall_seconds"), 0, 1800)
              and h.number(outer.get("peak_group_rss_kib"), 0, 8388608)
              and outer.get("thread_environment") == {k: "4" for k in h.THREAD_KEYS},
              "operation resource contract")
    for key, rel in (("supervisor_sha256", SUPERVISOR), ("monitor_sha256", MONITOR)):
        h.require(outer.get(key) == hashlib.sha256(h.git_bytes(root, commit, rel)).hexdigest(),
                  "operation monitor source")
    h.require(outer.get("log_sha256") == {n: h.sha(outer_path.parent / n)
              for n in ("stdout.log", "stderr.log")}, "operation logs changed")
    return {"receipt_sha256": h.sha(receipt_path), "outer_sha256": h.sha(outer_path),
            "spec_sha256": h.sha(spec_path), "file_sha256": snapshot(h,
                [receipt_path, spec_path, outer_path,
                 outer_path.parent / "stdout.log", outer_path.parent / "stderr.log"])}


def native_contract(h, root, native_root, commit):
    installation = native_root / "installation"
    qualify = module("evr0029_native_admission", root / OWN / "qualify.py")
    installed, installation_binding = qualify.installation_contract(h, root, commit, installation,
        native_root / "installation-supervision/receipt.json", native_root / "installation-spec.json")
    record_path = native_root / "qualification/receipt.json"
    record = h.read_json(record_path)
    h.require(record.get("schema") == "evr0029.native-qualification.v1"
              and record.get("status") == "completed" and record.get("eligible") is True
              and record.get("expected_commit") == commit
              and record.get("scientific_input_access") is False
              and record.get("offline_attempts") == []
              and record.get("installation_bindings") == installation_binding,
              "native qualification not eligible")
    paths = [OWN + p for p in ("qualify.py", "native_fixture.py", "install_runtime.py")]
    paths += [qualify.PROTOCOL, qualify.REVIEW, qualify.OLD_HELPER,
              *h.MANIFESTS, *h.OLD_MANIFESTS, SUPERVISOR, MONITOR]
    h.require(record.get("source_sha256") == h.frozen(root, commit, paths), "native source changed")
    fixture = record.get("native_fixture", {})
    h.require(fixture.get("status") == "completed" and fixture.get("manufactured_only") is True
              and fixture.get("carrier_removal_and_retention_verified") is True
              and fixture.get("future_after_discarded_probes", {}).get("exact_complete_state_parity") is True,
              "native interface checks incomplete")
    command = [str(installation / "venv/bin/python"), "-S", str(root / OWN / "qualify.py"),
               "--repo-root", str(root), "--expected-commit", commit,
               "--installation", str(installation),
               "--installation-supervision", str(native_root / "installation-supervision/receipt.json"),
               "--installation-spec", str(native_root / "installation-spec.json"),
               "--output-dir", str(native_root / "qualification")]
    binding = outer_contract(h, root, commit, native_root / "qualification-spec.json", record_path, command)
    binding["file_sha256"].update(snapshot(h, [root / rel for rel in paths]))
    binding["file_sha256"].update(snapshot(h, [native_root / rel for rel in
        ("installation/install-receipt.json", "installation-spec.json",
         "installation-supervision/receipt.json", "installation-supervision/stdout.log",
         "installation-supervision/stderr.log")]))
    return installation, installed, binding


def media_contract(h, root, commit, plan, phase, phase_root, spec_path, metadata_root):
    acquisition = module("evr0029_davis_admission", root / OWN / "acquire_davis.py")
    receipt_path = phase_root / "receipt.json"
    record = h.read_json(receipt_path)
    h.require(record.get("schema") == acquisition.SCHEMA and record.get("status") == "completed"
              and record.get("phase") == phase and record.get("expected_commit") == commit
              and record.get("manifest_sha256") == h.sha(root / MANIFEST)
              and record.get("media_decoded") is False and record.get("gt_admission") is None
              and record.get("hidden_retries") is False and record.get("full_zip_fallback") is False,
              "wrong or incomplete acquisition")
    paths = [acquisition.SELF, acquisition.HELPER, MANIFEST, PROTOCOL]
    h.require(record.get("source_sha256") == h.frozen(root, commit, paths), "acquisition source changed")
    command = ["/opt/homebrew/bin/python3.12", "-B", str(root / acquisition.SELF),
               "--repo-root", str(root), "--expected-commit", commit,
               "--metadata-root", str(metadata_root), "--phase", phase,
               "--output-dir", str(phase_root)]
    binding = outer_contract(h, root, commit, spec_path, receipt_path, command)
    binding["file_sha256"].update(snapshot(h, [root / rel for rel in paths]))
    expected = [row for row in plan["members"] if row["stage"] == phase]
    h.require(len(expected) == acquisition.PHASE_COUNTS[phase]
              and len({row["filename"] for row in expected}) == len(expected),
              "fixed acquisition phase coverage")
    h.require(record.get("archive") == plan["archive"]
              and record.get("selected_names") == [row["filename"] for row in expected]
              and len(record.get("members", [])) == len(expected), "media inventory identity")
    verified = {}
    for actual, wanted in zip(record["members"], expected):
        h.require(all(actual.get(k) == v for k, v in wanted.items()), "member metadata changed")
        rel = "members/" + wanted["output_relative"]
        h.require(actual.get("saved_relative") == rel, "member path changed")
        path = phase_root / rel
        h.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                  and path.stat().st_size == actual["bytes"] == wanted["uncompressed_bytes"]
                  and h.sha(path) == actual["sha256"], "media payload changed")
        verified[(wanted["sequence"], wanted["frame_index"])] = {
            "path": path, "sha256": actual["sha256"], "bytes": actual["bytes"],
            "frame": wanted["frame_index"]}
    h.require(record.get("response_body_bytes_exact") is True
              and h.number(record.get("response_body_bytes"), 0, acquisition.PHASE_CAPS[phase])
              and record.get("output_bytes") == sum(x["uncompressed_bytes"] for x in expected),
              "media accounting changed")
    return verified, binding


def run(args):
    root = Path(args.repo_root).resolve()
    out = Path(args.output_dir).absolute()
    if out.exists() or out.is_symlink() or root == out.resolve() or root in out.resolve().parents:
        raise ValueError("fresh private evidence output required")
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = {"schema": "evr0029.evidence-operation.v1", "status": "started", "eligible": False,
              "expected_commit": args.expected_commit, "native_commit": args.native_commit,
              "noninitial_annotation_access": False, "sam_sequence_invocations": 0,
              "offline_attempts": [], "compatibility_probes": []}
    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")
    save()
    try:
        h = module("evr0029_installer", root / OWN / "install_runtime.py")
        paths = [OWN + p for p in ("run_evidence.py", "frontend_evidence.py", "diagnostic_core.py", "acquire_davis.py")]
        paths += [PROTOCOL, MANIFEST, SUPERVISOR, MONITOR]
        bindings = h.frozen(root, args.expected_commit, paths)
        h.require(h.sha(Path(__file__)) == bindings[OWN + "run_evidence.py"], "wrong executing wrapper")
        record["source_sha256"] = bindings
        installation, installed, record["native_binding"] = native_contract(
            h, root, Path(args.native_root).resolve(), args.native_commit)
        acquisition = module("evr0029_davis_plan", root / OWN / "acquire_davis.py")
        ranges = acquisition.load_helper(root / acquisition.HELPER)
        plan, _ = acquisition.load_plan(root / MANIFEST, Path(args.metadata_root).resolve(), ranges)
        distinct_media_roots(h, Path(args.rgb_root).resolve(), Path(args.initial_root).resolve())
        verified = {}
        record["input_bindings"] = {}
        for phase, prefix in (("predictive_rgb", "rgb"), ("initial_supervision", "initial")):
            verified[phase], record["input_bindings"][phase] = media_contract(h, root, args.expected_commit,
                plan, phase, Path(getattr(args, prefix + "_root")).resolve(),
                Path(getattr(args, prefix + "_spec")).resolve(), Path(args.metadata_root).resolve())
        h.require(sys.version_info[:3] == (3, 12, 13) and sys.flags.no_site == 1
                  and Path(sys.executable).absolute() == installation / "venv/bin/python",
                  "pinned guarded interpreter required")
        q = module("evr0022_helpers", root / "experiments/0022-frontend/qualify.py")
        q.install_guard(record["offline_attempts"], record["compatibility_probes"])
        sys.dont_write_bytecode = True
        for key in h.THREAD_KEYS:
            h.require(os.environ.get(key) == "4", "thread environment")
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
        import site
        site.main()
        h.require(Path(sys.prefix).resolve() == installation / "venv", "wrong runtime prefix")
        sys.path[:0] = [str(installation / "sources/lightglue"), str(root / OWN)]
        worker = module("evr0029_frontend_evidence", root / OWN / "frontend_evidence.py")
        inputs = {"qualify_helper": root / "experiments/0022-frontend/qualify.py",
                  "diagnostic_core": root / OWN / "diagnostic_core.py",
                  "lightglue_source": installation / "sources/lightglue",
                  "runtime": installation / "venv/lib/python3.12/site-packages",
                  "superpoint_checkpoint": installation / "weights/superpoint_v1.pth",
                  "lightglue_checkpoint": installation / "weights/superpoint_lightglue.pth",
                  "source_sha256": bindings, "cohort": []}
        for cohort in plan["cohorts"]:
            sequence = cohort["sequence"]
            frames = [0, *cohort["query_frame_indices"]]
            inputs["cohort"].append({"sequence": sequence, "frame_count": cohort["frame_count"],
                "initial_mask": {k: verified["initial_supervision"][(sequence, 0)][k]
                                 for k in ("path", "sha256", "bytes")},
                "images": [verified["predictive_rgb"][(sequence, frame)] for frame in frames]})
        record["evidence"] = worker.run(inputs, out / "evidence", record)
        h.require(record["evidence"].get("status") == "completed", "evidence operation incomplete")
        h.require(not record["offline_attempts"], "forbidden process/network attempt")
        h.require(installed["installed_inventory"] == h.inventory(installation), "runtime files changed")
        h.require(all(h.sha(root / rel) == digest for rel, digest in bindings.items()), "source changed")
        unchanged(h, record["native_binding"]["file_sha256"])
        for binding in record["input_bindings"].values():
            unchanged(h, binding["file_sha256"])
        for phase in verified.values():
            h.require(all(row["path"].stat().st_size == row["bytes"]
                          and h.sha(row["path"]) == row["sha256"] for row in phase.values()),
                      "input payload changed during evidence run")
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
    for name in ("repo-root", "expected-commit", "native-root", "native-commit", "metadata-root",
                 "rgb-root", "rgb-spec", "initial-root", "initial-spec", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
