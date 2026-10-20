"""Run only manufactured D37 unit checks under the separately bounded outer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(spec_path):
    spec_raw = Path(spec_path).read_bytes()
    spec = json.loads(spec_raw)
    out = Path(spec["output"])
    if out.exists():
        raise ValueError("fresh output required")
    out.mkdir(parents=True)
    began = time.monotonic()
    record = dict(schema="evr0037.manufactured-qualification.v1", status="started",
        spec_sha256=hashlib.sha256(spec_raw).hexdigest(), source_sha256=spec["source_sha256"],
        manufactured_only=True, scientific_payload_reads=0, native_model_calls=0,
        offline_attempts=[], compatibility_probes=[])
    try:
        if not all(sha(p) == h for p,h in spec["source_sha256"].items()):
            raise ValueError("bound source changed")
        if not sys.flags.no_site or Path(sys.executable).absolute() != Path(spec["interpreter"]).absolute():
            raise ValueError("qualified interpreter -S required")
        guard_spec = importlib.util.spec_from_file_location("d37_qualification_guard", spec["guard_source"])
        guard = importlib.util.module_from_spec(guard_spec)
        guard_spec.loader.exec_module(guard)
        guard.install_guard(record["offline_attempts"], record["compatibility_probes"])
        import site
        site.main()
        import torch
        import cv2
        import unittest
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(4)
        here = Path(__file__).resolve().parent
        for p in here.glob("*.py"):
            compile(p.read_text(), str(p), "exec")
        sys.path.insert(0,str(here))
        suite = unittest.defaultTestLoader.discover(str(here), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        record.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                      skipped=len(result.skipped), successful=result.wasSuccessful())
        if not result.wasSuccessful() or result.skipped or record["offline_attempts"]:
            raise ValueError("manufactured qualification failed or incomplete")
        if not all(sha(p) == h for p,h in spec["source_sha256"].items()) or Path(spec_path).read_bytes()!=spec_raw:
            raise ValueError("source/spec mutated")
        record.update(status="completed", sources_unchanged=True)
    except BaseException as exc:
        record.update(status="failed", error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-began
        (out/"receipt.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return record


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",required=True)
    print(json.dumps(run(parser.parse_args().spec)))
