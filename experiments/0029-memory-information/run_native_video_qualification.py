"""Guarded manufactured qualification of the complete native-video worker."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OWN = "experiments/0029-memory-information/"
PROTOCOL = "protocols/0029-native-video-comparison.md"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run(args):
    root = Path(args.repo_root).resolve()
    out = Path(args.output_dir).absolute()
    if out.exists() or out.is_symlink() or root == out.resolve() or root in out.resolve().parents:
        raise ValueError("fresh private manufactured output required")
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0029.native-video-qualification-operation.v1", status="started",
        eligible=False, manufactured_only=True, scientific_eligible=False,
        expected_commit=args.expected_commit, native_commit=args.native_commit,
        scientific_input_access=False, offline_attempts=[], compatibility_probes=[])

    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")

    save()
    try:
        h = module("evr0029_install_admission", root / OWN / "install_runtime.py")
        admission = module("evr0029_evidence_admission", root / OWN / "run_evidence.py")
        paths = [OWN + p for p in ("run_native_video_qualification.py", "qualify_native_video.py",
            "native_video.py", "diagnostic_core.py", "run_evidence.py")]
        paths += [PROTOCOL, "protocols/0029-signed-memory-diagnostic.md",
                  admission.SUPERVISOR, admission.MONITOR]
        bindings = h.frozen(root, args.expected_commit, paths)
        h.require(h.sha(Path(__file__)) == bindings[OWN + "run_native_video_qualification.py"],
                  "wrong executing qualification wrapper")
        record["source_sha256"] = bindings
        installation, installed, record["native_binding"] = admission.native_contract(
            h, root, Path(args.native_root).resolve(), args.native_commit)
        h.require(sys.version_info[:3] == (3, 12, 13) and sys.flags.no_site == 1
                  and Path(sys.executable).absolute() == installation / "venv/bin/python",
                  "pinned guarded interpreter required")
        q = module("evr0022_offline_guard", root / "experiments/0022-frontend/qualify.py")
        q.install_guard(record["offline_attempts"], record["compatibility_probes"])
        sys.dont_write_bytecode = True
        for key in h.THREAD_KEYS:
            h.require(os.environ.get(key) == "4", "thread environment")
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
        import site
        site.main()
        h.require(Path(sys.prefix).resolve() == installation / "venv", "wrong runtime prefix")
        sys.path[:0] = [str(installation / "sources/sam2"), str(root / OWN)]
        worker = module("evr0029_manufactured_native_video", root / OWN / "qualify_native_video.py")
        inputs = dict(qualify_helper=root / "experiments/0022-frontend/qualify.py",
            native_fixture=root / OWN / "native_fixture.py", native_video=root / OWN / "native_video.py",
            diagnostic_core=root / OWN / "diagnostic_core.py", sam2_source=installation / "sources/sam2",
            runtime=installation / "venv/lib/python3.12/site-packages",
            sam2_checkpoint=installation / "weights/sam2.1_hiera_tiny.pt", source_sha256=bindings)
        record["qualification"] = worker.run(inputs, out / "manufactured", record)
        h.require(record["qualification"].get("status") == "completed"
                  and record["qualification"].get("complete") is True
                  and record["qualification"].get("manufactured_only") is True
                  and record["qualification"].get("scientific_eligible") is False,
                  "manufactured worker incomplete")
        h.require(not record["offline_attempts"], "forbidden process/network attempt")
        h.require(installed["installed_inventory"] == h.inventory(installation), "runtime files changed")
        h.require(all(h.sha(root / rel) == digest for rel, digest in bindings.items()), "source changed")
        admission.unchanged(h, record["native_binding"]["file_sha256"])
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
    for name in ("repo-root", "expected-commit", "native-root", "native-commit", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
