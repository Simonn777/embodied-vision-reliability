"""One-shot numerical diagnostic; selected x1/x2 only, never camera poses.

The 64 random eight-point hypotheses are a numerical workload, not a pose
baseline. Output is exclusively created JSONL; an existing output stops a run.
No dataset hash is recomputed here: the acquisition receipt owns that check.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Set before NumPy import; do not change the caller's Python installation.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    limit = int(os.environ.get(variable, "4"))
    if limit < 1:
        raise ValueError(variable + " must be positive")
    os.environ[variable] = str(min(4, limit))
import numpy as np
import h5py
from score_bounds import MatchTree, cell_bounds, exact_msac

SEED, ATTEMPTS, LEAF_SIZE, INITIAL_CELLS, TAU2 = 20261019, 64, 32, 128, 1.0
MAX_REFINEMENTS = 128


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(stream, value):
    stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def normalize(points):
    center = points.mean(axis=0)
    shifted = points - center
    scale_distance = float(np.linalg.norm(shifted, axis=1).mean())
    if not math.isfinite(scale_distance) or scale_distance <= 0:
        raise ValueError("zero or nonfinite normalization scale")
    scale = math.sqrt(2.0) / scale_distance
    transform = np.array([[scale, 0, -scale * center[0]],
                          [0, scale, -scale * center[1]], [0, 0, 1]])
    return np.column_stack((shifted * scale, np.ones(len(points)))), transform


def eight_point(x, y):
    xn, tx = normalize(x)
    yn, ty = normalize(y)
    design = (yn[:, :, None] * xn[:, None, :]).reshape(8, 9)
    _, singular, vt = np.linalg.svd(design, full_matrices=True)
    if singular[-1] <= np.finfo(float).eps * 9 * singular[0]:
        raise ValueError("rank-deficient eight-point sample")
    f = vt[-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(f)
    s[-1] = 0
    f = ty.T @ ((u * s) @ vt) @ tx
    scale = float(np.max(np.abs(f)))
    if not np.isfinite(f).all() or scale == 0:
        raise ValueError("invalid denormalized rank-two matrix")
    return f / scale


def make_hypotheses(x, y, rng):
    matrices, records = [], []
    for attempt in range(ATTEMPTS):
        indices = rng.choice(len(x), size=8, replace=False)
        record = {"attempt": attempt, "sample_indices": indices.tolist()}
        try:
            f = eight_point(x[indices], y[indices])
            record.update(valid_index=len(matrices), matrix=f.tolist())
            matrices.append(f)
        except (ValueError, np.linalg.LinAlgError) as error:
            record["failure"] = str(error)
        records.append(record)
    return matrices, records


def prepare_dms(cells, x, y):
    """Same partition, nearest actual match to its mean; not official DMS."""
    representatives = []
    for cell in cells:
        coordinates = np.column_stack((x[cell.indices], y[cell.indices]))
        delta = coordinates - coordinates.mean(axis=0)
        index = int(cell.indices[np.argmin(np.sum(delta * delta, axis=1))])
        representatives.append((np.r_[x[index], 1.0], np.r_[y[index], 1.0]))
    return representatives


def dms_form(f, cells, representatives):
    e, costs = f.ravel(), []
    for cell, (x, y) in zip(cells, representatives):
        denominator = float(np.sum((f @ x)[:2] ** 2) + np.sum((f.T @ y)[:2] ** 2))
        q = float(e @ cell.moment @ e)
        if not math.isfinite(q) or q < 0:
            raise ValueError("DMS-form moment evaluated negative/nonfinite")
        costs.append(min(q / denominator, cell.n * TAU2) if denominator > 0
                     else cell.n * TAU2)
    return math.fsum(costs)


def initial_intervals(cells, matrices, truth, x, y):
    intervals, violations = [], []
    for index, f in enumerate(matrices):
        bounds = [cell_bounds(cell, f, TAU2) for cell in cells]
        for cell_index, (cell, bound) in enumerate(zip(cells, bounds)):
            score = exact_msac(f, x[cell.indices], y[cell.indices], TAU2)
            for mode, check in (("moments", bound),
                    ("culling", cell_bounds(cell, f, TAU2, mode="culling"))):
                if not check.lower <= score <= check.upper:
                    violations.append({"index": index, "cell": cell_index, "mode": mode,
                        "lower": check.lower, "score": score, "upper": check.upper})
        lo = float(np.nextafter(math.fsum(b.lower for b in bounds), -np.inf))
        hi = float(np.nextafter(math.fsum(b.upper for b in bounds), np.inf))
        intervals.append([lo, hi])
        if not lo <= truth[index] <= hi:
            violations.append({"index": index, "lower": lo, "score": truth[index], "upper": hi})
    return intervals, violations


def pair_diagnostic(key, x, y, rng):
    if x.ndim != 2 or x.shape[1] != 2 or x.shape != y.shape or len(x) < 8:
        raise ValueError("selected pair needs equal N x 2 coordinates, N >= 8")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite selected coordinates")
    times, started = {}, time.perf_counter()
    matrices, proposals = make_hypotheses(x, y, rng)
    times["proposals"] = time.perf_counter() - started
    record = {"type": "pair", "key": key, "matches": len(x), "proposals": proposals}
    if not matrices:
        record.update(status="stop_no_valid_hypothesis", times_seconds=times)
        return record
    started = time.perf_counter()
    truth = [exact_msac(f, x, y, TAU2) for f in matrices]
    times["direct_score"] = time.perf_counter() - started
    winner = int(np.argmin(truth))
    started = time.perf_counter()
    tree = MatchTree(x, y, leaf_size=LEAF_SIZE)
    cells = tree.partition(INITIAL_CELLS)
    times["tree_setup"] = time.perf_counter() - started
    started = time.perf_counter()
    intervals, violations = initial_intervals(cells, matrices, truth, x, y)
    times["containment_diagnostic"] = time.perf_counter() - started
    record.update(exact_scores=truth, exact_winner=winner, initial_bounds=intervals,
                  initial_cells=len(cells), bound_violations=violations,
                  containment_cell_checks_per_mode=len(cells) * len(matrices),
                  containment_residuals=len(x) * len(matrices),
                  direct_residuals=len(x) * len(matrices))
    if violations:
        record.update(status="stop_bound_violation", times_seconds=times)
        return record
    rankings = {}
    for mode in ("culling", "moments"):
        started = time.perf_counter()
        result = tree.best_hypothesis(matrices, TAU2, initial_cells=INITIAL_CELLS,
                                     max_refinements=MAX_REFINEMENTS, mode=mode)
        times[mode + "_ranking"] = time.perf_counter() - started
        rankings[mode] = asdict(result)
        rankings[mode]["agrees_with_exact_winner"] = result.index == winner
        times[mode + "_including_setup"] = times["tree_setup"] + times[mode + "_ranking"]
        if result.index != winner:
            record.update(status="stop_wrong_winner", rankings=rankings, times_seconds=times)
            return record
    started = time.perf_counter()
    representatives = prepare_dms(cells, x, y)
    times["dms_extra_setup"] = time.perf_counter() - started
    started = time.perf_counter()
    approximate = [dms_form(f, cells, representatives) for f in matrices]
    times["dms_score"] = time.perf_counter() - started
    times["dms_including_setup"] = times["tree_setup"] + times["dms_extra_setup"] + times["dms_score"]
    record.update(status="completed", rankings=rankings, dms_form_scores=approximate,
                  dms_form_winner=int(np.argmin(approximate)), times_seconds=times,
                  dms_form_agrees_with_exact_winner=int(np.argmin(approximate)) == winner,
                  residuals_saved_vs_culling=rankings["culling"]["exact_residuals"] - rankings["moments"]["exact_residuals"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    keys = manifest["selected_pair_keys"]
    if manifest.get("schema") != "evr0012.development.v1" or len(keys) != 16:
        raise ValueError("wrong manifest schema or pair count")
    if keys != sorted(set(keys)) or manifest.get("coordinate_convention") != "raw_pixels":
        raise ValueError("manifest keys/conventions invalid")
    if args.data.stat().st_size != manifest["dataset_bytes"]:
        raise ValueError("dataset size differs from acquisition manifest")
    source = Path(__file__).resolve()
    rng = np.random.default_rng(SEED)
    with args.output.open("x") as stream:
        emit(stream, {"type": "start", "utc": datetime.now(timezone.utc).isoformat(),
                      "manifest": manifest, "manifest_sha256": file_sha(args.manifest),
                      "source_sha256": {source.name: file_sha(source),
                          "score_bounds.py": file_sha(source.with_name("score_bounds.py"))},
                      "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parents[2], text=True).strip(),
                      "python": sys.version, "numpy": np.__version__, "h5py": h5py.__version__,
                      "platform": platform.platform(), "seed": SEED,
                      "attempts": ATTEMPTS, "leaf_size": LEAF_SIZE,
                      "initial_cells": INITIAL_CELLS, "tau2": TAU2,
                      "max_refinements": MAX_REFINEMENTS,
                      "thread_limits": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
                      "dataset_hash_reverified_by_driver": False,
                      "timing_limit": "Shared tree setup includes moments unused by culling; Python numerical diagnostic only."})
        completed = 0
        try:
            with h5py.File(args.data, "r") as data:
                for key in keys:
                    # Deliberately do not enumerate values or read R/t/calibration.
                    x = np.asarray(data[key]["x1"][:], dtype=np.float64)
                    y = np.asarray(data[key]["x2"][:], dtype=np.float64)
                    record = pair_diagnostic(key, x, y, rng)
                    emit(stream, record)
                    print(json.dumps({"pair": completed + 1, "status": record["status"]}), flush=True)
                    if record["status"] != "completed":
                        emit(stream, {"type": "stop", "completed_pairs": completed, "reason": record["status"]})
                        return 2
                    completed += 1
            emit(stream, {"type": "complete", "completed_pairs": completed, "pose_accuracy_evaluated": False})
        except Exception as error:
            emit(stream, {"type": "error", "completed_pairs": completed,
                          "class": type(error).__name__, "message": str(error)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
