"""Exploratory 0012b replay of frozen hypotheses on the same development pairs.

Only selected x1/x2 datasets are read. Original scientific outputs and sources
are never changed. Historical times are not simultaneous speed comparisons.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import run_development as original  # caps caller thread limits before NumPy
import score_bounds as core
import mean_variance_bounds as addition
np, h5py = original.np, original.h5py
LEAF_SIZE, INITIAL_CELLS, MAX_REFINEMENTS, TAU2 = 32, 128, 128, 1.0
SHUFFLE_SEED, BLOCK_SIZE = 20261020, 256


@contextmanager
def added_bound():
    """Process-local injection only; preserve the frozen implementation on disk."""
    saved = core.cell_bounds
    def replacement(cell, matrix, tau2, *, mode="moments", secant=True):
        if mode != "moments" or not secant:
            raise ValueError("0012b fixes moment mode with secant enabled")
        return addition.bound(cell, matrix, tau2)
    core.cell_bounds = replacement
    try:
        yield
    finally:
        core.cell_bounds = saved


def residuals(matrix, x, y):
    xh = np.column_stack((x, np.ones(len(x))))
    yh = np.column_stack((y, np.ones(len(y))))
    # Warning origin is unresolved. Suppress flags only here, then validate all
    # product components (including unused ety[:, 2]) and consumed reductions.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ex, ety = xh @ matrix.T, yh @ matrix
        numerator = np.sum(yh * ex, axis=1) ** 2
        denominator = np.sum(ex[:, :2] ** 2 + ety[:, :2] ** 2, axis=1)
    if not all(np.isfinite(v).all() for v in (ex, ety, numerator, denominator)):
        raise OverflowError("nonfinite partial residual arithmetic")
    costs = np.full(len(x), TAU2)
    np.divide(numerator, denominator, out=costs, where=denominator > 0)
    return np.minimum(costs, TAU2)


def partial_exact(matrices, x, y, order):
    """Model 0 incumbent; original model order; no oracle scores enter selection."""
    n, best = len(x), 0
    best_score = core.exact_msac(matrices[0], x, y, TAU2)
    evaluated, blocks, pruned = n, 0, 0
    models = [{"index": 0, "residuals": n, "score": best_score, "pruned": False}]
    guard = core._pad(n * TAU2, n + 32)
    for index, matrix in enumerate(matrices[1:], 1):
        matrix, costs, block_sums = core._matrix(matrix), np.empty(n), []
        used, stopped = 0, False
        for start in range(0, n, BLOCK_SIZE):
            indices = order[start:start + BLOCK_SIZE]
            values = residuals(matrix, x[indices], y[indices])
            costs[indices] = values
            block_sums.append(float(np.sum(values, dtype=np.float64)))
            used += len(indices)
            evaluated += len(indices)
            blocks += 1
            partial = math.fsum(block_sums)
            if partial > best_score + guard:
                stopped = True
                pruned += 1
                break
        result = {"index": index, "residuals": used, "pruned": stopped,
                  "partial_sum": math.fsum(block_sums)}
        if not stopped:
            # Completed sums use original point order, preserving exact ties.
            score = float(np.sum(costs, dtype=np.float64))
            result["score"] = score
            if (score, index) < (best_score, best):
                best, best_score = index, score
        models.append(result)
    return {"index": best, "score": best_score, "exact_residuals": evaluated,
            "block_evaluations": blocks, "pruned_models": pruned,
            "early_pruned_models": sum(m["pruned"] and m["residuals"] < n for m in models),
            "completed_models": len(matrices) - pruned, "guard": guard,
            "models": models}


def check_initial(cells, matrices, x, y, truth):
    intervals, violations, lifts = [], [], []
    for index, matrix in enumerate(matrices):
        bounds = []
        for cell_index, cell in enumerate(cells):
            baseline = core.cell_bounds(cell, matrix, TAU2)
            bound = addition.bound(cell, matrix, TAU2)
            score = core.exact_msac(matrix, x[cell.indices], y[cell.indices], TAU2)
            lifts.append(bound.lower - baseline.lower)
            if not bound.lower <= score <= bound.upper:
                violations.append({"model": index, "cell": cell_index,
                    "lower": bound.lower, "score": score, "upper": bound.upper})
            bounds.append(bound)
        lo = float(np.nextafter(math.fsum(b.lower for b in bounds), -np.inf))
        hi = float(np.nextafter(math.fsum(b.upper for b in bounds), np.inf))
        intervals.append([lo, hi])
        if not lo <= truth[index] <= hi:
            violations.append({"model": index, "lower": lo,
                               "score": truth[index], "upper": hi})
    return intervals, violations, {"cells_strictly_lifted": sum(v > 0 for v in lifts),
        "cells_checked": len(lifts), "sum_lower_lift": math.fsum(lifts),
        "max_lower_lift": max(lifts, default=0.0), "min_lower_lift": min(lifts, default=0.0)}


def replay_pair(saved, x, y, rng):
    proposals = saved["proposals"]
    valid = [p for p in proposals if "valid_index" in p]
    if len(proposals) != 64 or [p["valid_index"] for p in valid] != list(range(64)):
        raise ValueError("expected exactly 64 saved valid hypotheses in original order")
    matrices = [np.asarray(p["matrix"], dtype=np.float64) for p in valid]
    if any(m.shape != (3, 3) or not np.isfinite(m).all() for m in matrices):
        raise ValueError("invalid saved matrix")
    truth = saved["exact_scores"]
    if len(truth) != 64 or not np.isfinite(truth).all():
        raise ValueError("invalid saved reference scores")
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or len(x) != saved["matches"]:
        raise ValueError("coordinates differ from frozen workload")
    winner, times = int(np.argmin(truth)), {}
    record = {"type": "pair", "key": saved["key"], "matches": len(x),
        "exact_scores": truth, "exact_winner": winner, "proposals": proposals,
        "historical": {k: saved[k] for k in ("rankings", "times_seconds", "initial_bounds", "direct_residuals")}}
    started = time.perf_counter()
    tree = core.MatchTree(x, y, leaf_size=LEAF_SIZE)
    cells = tree.partition(INITIAL_CELLS)
    times["tree_setup"] = time.perf_counter() - started
    started = time.perf_counter()
    intervals, violations, lift = check_initial(cells, matrices, x, y, truth)
    times["containment_validation"] = time.perf_counter() - started
    record.update(initial_bounds=intervals, bound_violations=violations, lower_lift=lift,
                  validation_residuals=len(x) * len(matrices), initial_cells=len(cells))
    if violations:
        return {**record, "status": "stop_bound_violation", "times_seconds": times}
    started = time.perf_counter()
    with added_bound():
        result = tree.best_hypothesis(matrices, TAU2, initial_cells=INITIAL_CELLS,
                                     max_refinements=MAX_REFINEMENTS, mode="moments")
    times["mean_variance_ranking"] = time.perf_counter() - started
    times["mean_variance_including_setup"] = times["tree_setup"] + times["mean_variance_ranking"]
    record["mean_variance"] = asdict(result)
    if result.index != winner:
        return {**record, "status": "stop_wrong_winner", "times_seconds": times}
    started = time.perf_counter()
    order = rng.permutation(len(x))
    times["partial_shuffle_setup"] = time.perf_counter() - started
    started = time.perf_counter()
    partial = partial_exact(matrices, x, y, order)
    times["partial_ranking"] = time.perf_counter() - started
    times["partial_including_setup"] = times["partial_shuffle_setup"] + times["partial_ranking"]
    record.update(partial_exact=partial, shuffle_sha256=original.hashlib.sha256(order.tobytes()).hexdigest())
    status = "completed" if partial["index"] == winner else "stop_partial_wrong_winner"
    record.update(status=status, times_seconds=times,
        residual_ratio_vs_partial=result.exact_residuals / partial["exact_residuals"],
        residual_ratio_vs_full=result.exact_residuals / saved["direct_residuals"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    reference = [json.loads(line) for line in args.reference.read_text().splitlines()]
    pairs = [r for r in reference if r.get("type") == "pair"]
    keys = manifest["selected_pair_keys"]
    if len(keys) != 16 or keys != sorted(set(keys)) or [p["key"] for p in pairs] != keys:
        raise ValueError("reference must contain the same 16 selected pairs in order")
    if reference[0]["type"] != "start" or reference[-1].get("completed_pairs") != 16 or reference[-1]["type"] != "complete":
        raise ValueError("reference run incomplete")
    if any(p["status"] != "completed" for p in pairs) or reference[0]["manifest"] != manifest:
        raise ValueError("reference statuses/manifest mismatch")
    if any(reference[0].get(k) != v for k, v in {"attempts": 64, "leaf_size": 32,
            "initial_cells": 128, "max_refinements": 128, "tau2": 1.0}.items()):
        raise ValueError("reference workload parameters changed")
    if manifest.get("schema") != "evr0012.development.v1" or manifest.get("coordinate_convention") != "raw_pixels" or args.data.stat().st_size != manifest["dataset_bytes"]:
        raise ValueError("dataset convention/size mismatch")
    source = Path(__file__).resolve()
    hashes = {name: original.file_sha(source.with_name(name)) for name in
              (source.name, "run_development.py", "score_bounds.py", "mean_variance_bounds.py")}
    if any(hashes[k] != v for k, v in reference[0]["source_sha256"].items()):
        raise ValueError("frozen original source changed")
    rng, completed = np.random.default_rng(SHUFFLE_SEED), 0
    with args.output.open("x") as stream:
        original.emit(stream, {"type": "start", "schema": "evr0012b.exploratory.v1",
            "utc": datetime.now(timezone.utc).isoformat(), "manifest": manifest,
            "manifest_sha256": original.file_sha(args.manifest),
            "protocol_sha256": original.file_sha(source.parents[2] / "protocols/0012b-mean-variance-exploration.md"),
            "reference_sha256": original.file_sha(args.reference), "source_sha256": hashes,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parents[2], text=True).strip(),
            "python": sys.version, "numpy": np.__version__, "h5py": h5py.__version__,
            "platform": platform.platform(), "leaf_size": LEAF_SIZE, "initial_cells": INITIAL_CELLS,
            "max_refinements": MAX_REFINEMENTS, "tau2": TAU2,
            "shuffle_seed": SHUFFLE_SEED, "block_size": BLOCK_SIZE,
            "thread_limits": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "historical_timing_limit": "Original selectors are not rerun; their times are historical, not a controlled concurrent comparison."})
        try:
            with h5py.File(args.data, "r") as data:
                for saved in pairs:
                    key = saved["key"]
                    x = np.asarray(data[key]["x1"][:], dtype=np.float64)
                    y = np.asarray(data[key]["x2"][:], dtype=np.float64)
                    record = replay_pair(saved, x, y, rng)
                    original.emit(stream, record)
                    print(json.dumps({"pair": completed + 1, "status": record["status"]}), flush=True)
                    if record["status"] != "completed":
                        original.emit(stream, {"type": "stop", "completed_pairs": completed, "reason": record["status"]})
                        return 2
                    completed += 1
            original.emit(stream, {"type": "complete", "completed_pairs": completed,
                                  "pose_accuracy_evaluated": False, "confirmation": False})
        except Exception as error:
            original.emit(stream, {"type": "error", "completed_pairs": completed,
                                  "class": type(error).__name__, "message": str(error)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
