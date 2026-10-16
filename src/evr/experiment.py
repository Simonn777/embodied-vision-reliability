"""Orchestration: keep rendered truth outside the deployed estimator/scorers."""

import math
import time

from .metrics import METHODS, calibrate, paired_bootstrap, risk_coverage_curve, summarize
from .scenes import CONDITIONS, make_scene, render_frame, split_manifest, tie_key, trajectory
from .tracker import FixedTracker


def evaluate_sequence(sequence_id, partition, condition, frames=32):
    scene = make_scene(sequence_id)
    truth = trajectory(scene, condition, frames)
    first_frame = render_frame(scene, truth[0], 0, condition)
    # Explicit oracle initialization only; subsequent GT is used after scoring.
    tracker = FixedTracker(first_frame, scene.initial)
    rows = []
    for frame in range(1, frames):
        image = render_frame(scene, truth[frame], frame, condition)
        detection, scores = tracker.step(image, frame)
        error_norm = None if detection is None else math.hypot(
            detection.x - truth[frame, 0], detection.y - truth[frame, 1]) / math.hypot(96, 96)
        rows.append({
            "sequence_id": sequence_id, "partition": partition, "condition": condition,
            "frame": frame, "gt_x": float(truth[frame, 0]), "gt_y": float(truth[frame, 1]),
            "detected": detection is not None,
            "pred_x": detection.x if detection else None,
            "pred_y": detection.y if detection else None,
            "observed_area": detection.area if detection else 0,
            "anisotropy": detection.anisotropy if detection else None,
            "initial_observed_area": tracker.initial.area,
            "score_static": scores["static"] if detection else None,
            "score_temporal": scores["temporal"] if detection else None,
            "innovation_px": scores["innovation"], "error_norm": error_norm,
            "tie_key": tie_key(sequence_id, frame),
        })
    return rows


def run(smoke=False, bootstrap_repeats=1000, timeout_seconds=1800, progress=None):
    start = time.monotonic()
    rows = []
    manifest = split_manifest(smoke)
    for record in manifest:
        for condition in CONDITIONS:
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError("Experiment exceeded the 30-minute bound; no incomplete result is a success")
            rows.extend(evaluate_sequence(record["sequence_id"], record["partition"], condition))
        if progress:
            progress(record)
    calibration = [row for row in rows if row["partition"] == "cal" and row["condition"] == "clean"]
    thresholds = {method: calibrate(calibration, method) for method in METHODS if method != "accept_all"}
    test_rows = [row for row in rows if row["partition"] == "test"]
    conditions = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in test_rows if row["condition"] == condition]
        methods = {method: summarize(condition_rows, method, thresholds.get(method)) for method in METHODS}
        curves = {method: risk_coverage_curve(condition_rows, method) for method in ("static", "temporal", "random", "oracle")}
        conditions[condition] = {"methods": methods, "curves": curves}
    summary = {
        "experiment": "0001-synthetic-localization", "diagnostic_pilot": True, "smoke": smoke,
        "thresholds": thresholds,
        "calibration": {method: summarize(calibration, method, thresholds.get(method)) for method in METHODS},
        "test": conditions,
        "bootstrap": paired_bootstrap(test_rows, thresholds, repeats=bootstrap_repeats),
        "notes": [
            "Initial query point is ground-truth oracle initialization; frame zero is excluded.",
            "Methods share exactly the same raw predictions; only rejection differs.",
            "Missing detections remain in coverage and all-frame failure denominators.",
            "AURC integrates only reachable detected coverage; oracle is descriptive, not deployable.",
            "Synthetic diagnostic only; no robotics, novelty, or arbitrary-shift guarantee.",
        ],
    }
    return rows, summary, time.monotonic() - start
