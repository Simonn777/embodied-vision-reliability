"""One fixed favorable overhead-admission fixture, with every timing retained."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np
from grid_backend import GridScorer, sha256
from test_grid_backend import favorable_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve()
    receipt = json.loads(args.build_receipt.read_text())
    if receipt["source_sha256"] != sha256(source.with_name("native_grid.cpp")):
        raise ValueError("grid source differs from receipt")
    for name, checksum in receipt["included_source_sha256"].items():
        if sha256(source.with_name(name)) != checksum:
            raise ValueError("included source differs from receipt: " + name)
    if sha256(receipt["library"]) != receipt["library_sha256"]:
        raise ValueError("library differs from receipt")
    report = {"schema": "evr0012d.synthetic.v1", "admission_passed": False,
        "utc": datetime.now(timezone.utc).isoformat(), "fixture": "10000 correlated points,64 models; fixed favorable protocol fixture",
        "source_sha256": {name: sha256(source.with_name(name)) for name in
            (source.name, "test_grid_backend.py", "grid_backend.py", "native_backend.py", "native_grid.cpp", "native_score.cpp", "run_grid.py")},
        "protocol_sha256": sha256(source.parents[2] / "protocols/0012d-lazy-grid-screening.md"),
        "build_receipt_sha256": sha256(args.build_receipt),
        "library_sha256": receipt["library_sha256"], "compiler": receipt["compiler"],
        "flags": receipt["command"][1:-3],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parents[2], text=True).strip(),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "native_threads": 1, "thread_limits": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "repetitions": 9, "mode_order": "(repetition+offset)%5", "warmups": [], "timed_calls": [],
        "confirmation": False, "pose_accuracy_evaluated": False}
    with args.output.open("x") as stream:
        try:
            backend = GridScorer(receipt["library"])
            x, y, matrices = favorable_fixture()
            scores = backend.scores(x, y, matrices, 1.0)
            validation = backend.validate(x, y, matrices, 1.0)
            report.update(scores=scores.tolist(), validation=validation)
            if np.argmin(scores) != 0 or validation["violations"] or validation["cell_checks"] != 128 or validation["aggregate_checks"] != 128:
                raise RuntimeError("artificial reference or containment validation failed")
            start = time.perf_counter()
            order = np.random.default_rng(20261020).permutation(len(x)).astype(np.uint64)
            shuffle_seconds = time.perf_counter() - start
            report["shuffle_seconds"] = shuffle_seconds
            for mode in range(5):
                result = backend.select(x, y, matrices, 1.0, order, mode)
                report["warmups"].append({"mode": mode, **result})
                if result["winner"] != 0 or result["score"] != scores[0]:
                    raise RuntimeError("warmup selection mismatch")
            for repetition in range(9):
                for position in range(5):
                    mode = (repetition + position) % 5
                    start = time.perf_counter()
                    result = backend.select(x, y, matrices, 1.0, order, mode)
                    call_seconds = time.perf_counter() - start
                    charged = shuffle_seconds if mode == 1 else 0.0
                    entry = {"mode": mode, "repetition": repetition, "position": position,
                        **result, "call_seconds": call_seconds, "charged_shuffle_seconds": charged,
                        "primary_total_seconds": call_seconds + charged,
                        "winner_agrees": result["winner"] == 0, "score_agrees": result["score"] == scores[0]}
                    report["timed_calls"].append(entry)
                    if not entry["winner_agrees"] or not entry["score_agrees"]:
                        raise RuntimeError("timed selection mismatch")
            medians = {str(mode): statistics.median(entry["primary_total_seconds"] for entry in
                report["timed_calls"] if entry["mode"] == mode) for mode in range(5)}
            ratios = {str(mode): medians["4"] / medians[str(mode)] for mode in range(4)}
            report.update(status="completed", medians=medians, candidate_ratios=ratios,
                          admission_passed=all(ratio <= 2 / 3 for ratio in ratios.values()))
        except Exception as error:
            report.update(status="error", error_class=type(error).__name__, error=str(error))
            raise
        finally:
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
    print(json.dumps({key: report[key] for key in ("status", "medians", "candidate_ratios", "admission_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
