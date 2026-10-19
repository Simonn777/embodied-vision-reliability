"""0012d: frozen lazy-grid replay, five modes, three cyclic timed repetitions.

Used development pairs only. Saved references are validation inputs, never
selection inputs. No calibration, poses, images or unselected arrays are read.
"""
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

import run_development as original  # respect caller thread limits before NumPy
from grid_backend import GridScorer, sha256
np, h5py = original.np, original.h5py
MODES = {0: "direct", 1: "random_partial", 2: "grid_partial",
         3: "geometry_partial", 4: "lazy_moment_partial"}
SHUFFLE_MODES, SHUFFLE_SEED, REPETITIONS, TAU2 = {1}, 20261020, 3, 1.0


def score_matches(value, reference):
    return bool(abs(value - reference) <= 1e-8 + 1e-12 * abs(reference))


def validate_artificial_admission(admission, library_hash):
    if admission.get("schema") != "evr0012d.synthetic.v1" or admission.get("status") != "completed" or admission.get("admission_passed") is not True:
        raise ValueError("complete favorable artificial admission required")
    if admission.get("library_sha256") != library_hash or admission.get("original_library_sha256") != library_hash:
        raise ValueError("native binary differs from original/retry artificial admission")
    checks = admission.get("validation", {})
    if checks.get("violations") != 0 or checks.get("cell_checks") != 128 or checks.get("aggregate_checks") != 128:
        raise ValueError("artificial containment qualification incomplete")
    entries = admission.get("timed_calls", [])
    if len(entries) != 45 or len(admission.get("warmups", [])) != 5:
        raise ValueError("artificial timing record incomplete")
    for index, entry in enumerate(entries):
        repetition, position = divmod(index, 5)
        if (entry.get("repetition"), entry.get("position"), entry.get("mode")) != (repetition, position, (repetition + position) % 5):
            raise ValueError("artificial timing schedule changed")
        if entry.get("winner_agrees") is not True or entry.get("score_agrees") is not True:
            raise ValueError("artificial selection agreement failed")
        if not np.isfinite(entry["primary_total_seconds"]) or entry["primary_total_seconds"] <= 0:
            raise ValueError("invalid artificial duration")
    medians = {str(mode): statistics.median(e["primary_total_seconds"] for e in entries if e["mode"] == mode) for mode in range(5)}
    ratios = {str(mode): medians["4"] / medians[str(mode)] for mode in range(4)}
    if medians != admission.get("medians") or ratios != admission.get("candidate_ratios") or not all(r <= 2 / 3 for r in ratios.values()):
        raise ValueError("artificial gate does not recompute")


def mode_order(pair_index, repetition):
    return [(pair_index + repetition + offset) % 5 for offset in range(5)]


def public_build_receipt(receipt):
    """Keep local provenance on disk while omitting workspace paths in JSONL."""
    source, library = Path(receipt["source"]), Path(receipt["library"])
    replacements = [(str(source), "experiments/0012-score-bounds/native_grid.cpp"),
        (str(library), "<local-build>/" + library.name),
        (str(source.parents[2]), "<repository>"), (str(source.parents[3]), "<local-work>")]
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
        return value
    return clean(receipt)


def run_pair(backend, saved, x, y, pair_index, rng):
    valid = [p for p in saved["proposals"] if "valid_index" in p]
    if len(saved["proposals"]) != 64 or [p["valid_index"] for p in valid] != list(range(64)):
        raise ValueError("expected unchanged 64 valid saved hypotheses")
    matrices = np.asarray([p["matrix"] for p in valid], dtype=np.float64)
    reference = np.asarray(saved["exact_scores"], dtype=np.float64)
    if reference.shape != (64,) or not np.isfinite(reference).all():
        raise ValueError("invalid saved scores")
    if x.shape != y.shape or x.shape != (saved["matches"], 2):
        raise ValueError("coordinates differ from frozen workload")
    expected = int(np.argmin(reference))
    record = {"type": "pair", "key": saved["key"], "matches": len(x),
              "reference_scores": reference.tolist(), "reference_winner": expected,
              "proposals": saved["proposals"], "timed_calls": []}
    started = time.perf_counter()
    scores = backend.scores(x, y, matrices, TAU2)
    differences = np.abs(scores - reference)
    score_failures = np.flatnonzero(differences > 1e-8 + 1e-12 * np.abs(reference)).tolist()
    native_winner = int(np.argmin(scores))
    record.update(native_scores=scores.tolist(), max_score_difference=float(differences.max()),
                  native_direct_winner=native_winner, score_mismatch_indices=score_failures)
    if score_failures or native_winner != expected:
        return {**record, "status": "stop_native_reference_mismatch"}
    validation = backend.validate(x, y, matrices, TAU2)
    record.update(containment=validation, validation_seconds=time.perf_counter() - started)
    if validation["violations"] or validation["aggregate_checks"] != 128:
        return {**record, "status": "stop_containment_violation"}
    started = time.perf_counter()
    order = rng.permutation(len(x)).astype(np.uint64)
    shuffle_seconds = time.perf_counter() - started
    record.update(shuffle_seconds=shuffle_seconds,
                  shuffle_sha256=original.hashlib.sha256(order.tobytes()).hexdigest())
    for repetition in range(REPETITIONS):
        for position, mode in enumerate(mode_order(pair_index, repetition)):
            started = time.perf_counter()
            result = backend.select(x, y, matrices, TAU2, order, mode)
            call_seconds = time.perf_counter() - started
            agreement = result["winner"] == expected
            score_agreement = score_matches(result["score"], reference[expected])
            charged_shuffle = shuffle_seconds if mode in SHUFFLE_MODES else 0.0
            entry = {"mode": mode, "name": MODES[mode], "repetition": repetition,
                "position": position, **result, "call_seconds": call_seconds,
                "charged_shuffle_seconds": charged_shuffle,
                "native_total_seconds": result["setup_seconds"] + result["scoring_seconds"],
                "primary_total_seconds": call_seconds + charged_shuffle,
                "winner_agrees": agreement, "score_agrees": score_agreement}
            record["timed_calls"].append(entry)
            if not agreement or not score_agreement:
                return {**record, "status": "stop_timed_result_mismatch"}
    medians = {}
    for mode in MODES:
        entries = [r for r in record["timed_calls"] if r["mode"] == mode]
        medians[str(mode)] = {field: statistics.median(r[field] for r in entries) for field in
            ("primary_total_seconds", "native_total_seconds", "setup_seconds", "scoring_seconds", "call_seconds")}
    record.update(status="completed", medians=medians,
        primary_candidate_ratios={str(mode): medians["4"]["primary_total_seconds"] /
            medians[str(mode)]["primary_total_seconds"] for mode in (0, 1, 2, 3)})
    return record


def load_inputs(args):
    manifest = json.loads(args.manifest.read_text())
    reference = [json.loads(line) for line in args.reference.read_text().splitlines()]
    pairs = [r for r in reference if r.get("type") == "pair"]
    keys = manifest["selected_pair_keys"]
    if manifest.get("schema") != "evr0012.development.v1" or manifest.get("coordinate_convention") != "raw_pixels":
        raise ValueError("invalid manifest schema/convention")
    if len(keys) != 16 or keys != sorted(set(keys)) or [p["key"] for p in pairs] != keys:
        raise ValueError("expected the same 16 pairs in original order")
    if reference[0]["type"] != "start" or reference[0]["manifest"] != manifest:
        raise ValueError("reference header/manifest mismatch")
    if any(reference[0].get(k) != v for k, v in {"attempts": 64, "leaf_size": 32,
            "initial_cells": 128, "max_refinements": 128, "tau2": 1.0}.items()):
        raise ValueError("reference workload configuration changed")
    if reference[-1]["type"] != "complete" or reference[-1].get("completed_pairs") != 16:
        raise ValueError("reference run incomplete")
    if any(p["status"] != "completed" or p["matches"] != 10000 for p in pairs):
        raise ValueError("reference pair status/size mismatch")
    if args.data.stat().st_size != manifest["dataset_bytes"]:
        raise ValueError("dataset size changed")
    return manifest, reference[0], pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "reference", "build-receipt", "synthetic-admission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest, reference_header, pairs = load_inputs(args)
    source = Path(__file__).resolve()
    receipt = json.loads(args.build_receipt.read_text())
    if receipt["source_sha256"] != sha256(source.with_name("native_grid.cpp")):
        raise ValueError("native source differs from build receipt")
    if receipt["library_sha256"] != sha256(receipt["library"]):
        raise ValueError("native library differs from build receipt")
    for filename, checksum in reference_header["source_sha256"].items():
        if sha256(source.with_name(filename)) != checksum:
            raise ValueError("frozen original source changed: " + filename)
    for name, checksum in receipt["included_source_sha256"].items():
        if sha256(source.with_name(name)) != checksum:
            raise ValueError("included native source differs from receipt: " + name)
    admission = json.loads(args.synthetic_admission.read_text())
    validate_artificial_admission(admission, receipt["library_sha256"])
    for name, checksum in admission["source_sha256"].items():
        if sha256(source.with_name(name)) != checksum:
            raise ValueError("source differs from artificial admission: " + name)
    if admission["protocol_sha256"] != sha256(source.parents[2] / "protocols/0012d-lazy-grid-screening.md"):
        raise ValueError("protocol differs from artificial admission")
    backend, rng = GridScorer(receipt["library"]), np.random.default_rng(SHUFFLE_SEED)
    totals, completed = {str(mode): 0.0 for mode in MODES}, 0
    with args.output.open("x") as stream:
        original.emit(stream, {"type": "start", "schema": "evr0012d.grid.v1",
            "utc": datetime.now(timezone.utc).isoformat(), "manifest": manifest,
            "manifest_sha256": sha256(args.manifest), "reference_sha256": sha256(args.reference),
            "synthetic_admission_sha256": sha256(args.synthetic_admission),
            "build_receipt": public_build_receipt(receipt), "build_receipt_sha256": sha256(args.build_receipt),
            "protocol_sha256": sha256(source.parents[2] / "protocols/0012d-lazy-grid-screening.md"),
            "source_sha256": {f: sha256(source.with_name(f)) for f in
                (source.name, "grid_backend.py", "native_grid.cpp", "native_backend.py", "native_score.cpp", "run_development.py", "run_grid_synthetic.py", "test_grid_backend.py")},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parents[2], text=True).strip(),
            "python": sys.version, "numpy": np.__version__, "h5py": h5py.__version__,
            "platform": platform.platform(), "native_threads": 1,
            "thread_limits": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "tau2": TAU2, "grid_width_pixels": 256, "minimum_cell_points": 64,
            "block_size": 256, "shuffle_seed": SHUFFLE_SEED, "repetitions": REPETITIONS,
            "mode_order": "(pair_index + repetition + offset) % 5", "modes": MODES,
            "score_tolerance": "1e-8 + 1e-12*abs(reference); exact winner index required",
            "primary_total": "outer wrapper-call duration (input checks and ctypes included) + shuffle setup charged to mode1",
            "shuffle_charge": "generated once per pair; same measured cost charged to each fresh call using it",
            "confirmation": False, "pose_accuracy_evaluated": False})
        try:
            with h5py.File(args.data, "r") as data:
                for pair_index, saved in enumerate(pairs):
                    key = saved["key"]
                    x = np.asarray(data[key]["x1"][:], dtype=np.float64)
                    y = np.asarray(data[key]["x2"][:], dtype=np.float64)
                    record = run_pair(backend, saved, x, y, pair_index, rng)
                    original.emit(stream, record)
                    print(json.dumps({"pair": pair_index + 1, "status": record["status"]}), flush=True)
                    if record["status"] != "completed":
                        original.emit(stream, {"type": "stop", "completed_pairs": completed, "reason": record["status"]})
                        return 2
                    completed += 1
                    for mode in MODES:
                        totals[str(mode)] += record["medians"][str(mode)]["primary_total_seconds"]
            ratios = {str(mode): totals["4"] / totals[str(mode)] for mode in (0, 1, 2, 3)}
            original.emit(stream, {"type": "complete", "completed_pairs": completed,
                "sum_pair_median_primary_seconds": totals, "candidate_ratios": ratios,
                "numerical_speed_gate_passed": all(ratios[str(m)] <= 2 / 3 for m in (0, 1, 2, 3)),
                "confirmation": False, "pose_accuracy_evaluated": False})
        except Exception as error:
            original.emit(stream, {"type": "error", "completed_pairs": completed,
                                  "class": type(error).__name__, "message": str(error)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
