#!/usr/bin/env python3
"""Run frozen protocol 0001, or a separately identified tiny CI smoke run."""

import os

THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def configure_threads(environ):
    """Respect the strictest inherited limit, with a local default/ceiling of four."""
    limit = 4
    for variable in THREAD_VARIABLES:
        if variable not in environ:
            continue
        try:
            value = int(environ[variable])
        except ValueError:
            raise ValueError(f"{variable} must be a positive integer") from None
        if value < 1:
            raise ValueError(f"{variable} must be a positive integer")
        limit = min(limit, value)
    # Validate every input before changing any environment variable.
    for variable in THREAD_VARIABLES:
        environ[variable] = str(limit)
    return limit


# Set before importing numerical libraries. This CPU experiment needs no GPU.
try:
    THREADS_CAP = configure_threads(os.environ)
except ValueError as error:
    raise SystemExit(str(error)) from None

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evr import __version__
from evr.experiment import run
from evr.metrics import FAILURE_TOLERANCE, METHODS, accepted, sequence_totals
from evr.scenes import CONDITIONS, MASTER_SEED, make_scene, render_frame, split_manifest, trajectory
from evr.tracker import FixedTracker


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def portable_path(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "<external-path>"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_examples(rows, destination, smoke):
    """Prespecified cases, selected without inspecting error or score values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sequence_id = 10006 if smoke else 120
    frames = (0, 8, 14, 16, 23, 31)
    scene = make_scene(sequence_id)
    lookup = {(row["condition"], row["frame"]): row for row in rows if row["sequence_id"] == sequence_id}
    figure, axes = plt.subplots(4, 6, figsize=(11, 8))
    for condition, row_axes in zip(CONDITIONS, axes):
        truth = trajectory(scene, condition)
        for frame, axis in zip(frames, row_axes):
            rgb = render_frame(scene, truth[frame], frame, condition)
            axis.imshow(rgb, interpolation="nearest")
            axis.scatter(*truth[frame], marker="+", color="#00cfff", s=70, linewidth=1.4)
            if frame == 0:
                initial = FixedTracker(rgb, scene.initial).initial
                prediction = (initial.x, initial.y)
            else:
                record = lookup[(condition, frame)]
                prediction = (record["pred_x"], record["pred_y"]) if record["detected"] else None
            if prediction is not None:
                axis.scatter(*prediction, marker="o", facecolors="none", edgecolors="#ff4dc3", s=65, linewidth=1.3)
            else:
                axis.text(.5, .06, "no detection", color="white", fontsize=7, ha="center", transform=axis.transAxes)
            if condition == CONDITIONS[0]:
                axis.set_title(f"Frame {frame}" + (" (init)" if frame == 0 else ""), fontsize=9)
            if frame == frames[0]:
                axis.set_ylabel(condition.replace("_", " "), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    handles = [Line2D([], [], marker="+", linestyle="", color="#00cfff", label="Ground-truth center"),
               Line2D([], [], marker="o", markerfacecolor="none", linestyle="", color="#ff4dc3", label="Raw centroid prediction")]
    figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    figure.suptitle(f"Prespecified synthetic examples — base sequence {sequence_id:03d}", fontsize=13)
    figure.tight_layout(rect=(0, .045, 1, .96))
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def make_figure(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"static": "#9d4d26", "temporal": "#21685b", "random": "#6d7082", "oracle": "#a8ad8b"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, condition in zip(axes.flat, CONDITIONS):
        content = summary["test"][condition]
        for method, curve in content["curves"].items():
            label = "oracle (GT ranking)" if method == "oracle" else method
            axis.plot(curve["coverage"], curve["risk"], label=label, color=colors[method],
                      linestyle="--" if method in ("random", "oracle") else "-", linewidth=1.6)
        for method in ("static", "temporal", "random"):
            metric = content["methods"][method]
            if metric["selected_failure_rate"] is not None:
                axis.scatter([metric["coverage"]], [metric["selected_failure_rate"]], color=colors[method], s=24, zorder=3)
        axis.set_title(condition.replace("_", " "))
        axis.set_xlim(0, 1); axis.set_ylim(-.03, 1.03); axis.grid(alpha=.2)
        axis.set_xlabel("Accepted frames / all scored frames")
        axis.set_ylabel("Failure among accepted predictions")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    figure.suptitle("Synthetic localization diagnostic — shared predictions, different rejection", fontsize=13)
    figure.text(.5, .015, "Failure: error > 2.5% of image diagonal. Dots use frozen clean-calibration thresholds. Missing detections remain in the denominator.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, .95))
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/0001-synthetic-localization.md"))
    parser.add_argument("--smoke", action="store_true", help="Use separate IDs and ten sequences; not a final-test experiment")
    args = parser.parse_args()
    destination = args.output.resolve()
    protocol = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    if destination.exists():
        parser.error("Output already exists; choose a fresh directory. Original runs are never overwritten.")
    revision = git_value("rev-parse", "HEAD")
    if not args.smoke:
        if not protocol.is_file():
            parser.error("Full run requires a saved and committed protocol")
        protocol_relative = portable_path(protocol)
        if protocol_relative == "<external-path>" or git_value("cat-file", "-e", f"HEAD:{protocol_relative}") is None:
            parser.error("Full-run protocol must be committed in this repository")
        if git_value("diff", "HEAD", "--", protocol_relative):
            parser.error("Protocol has uncommitted changes; freeze it before a full run")
    source_files = sorted([ROOT / "pyproject.toml", *list((ROOT / "src").rglob("*.py")), *list((ROOT / "scripts").glob("*.py")), *list((ROOT / "tests").glob("*.py"))])
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files}
    source_dirty = bool(git_value("status", "--porcelain", "--", "src", "scripts", "tests", "pyproject.toml"))
    config = {
        "protocol": portable_path(protocol), "protocol_sha256": sha256(protocol) if protocol.exists() else None,
        "master_seed": MASTER_SEED, "bootstrap_seed": 20261017,
        "bootstrap_repeats": 100 if args.smoke else 1000,
        "smoke": args.smoke, "manifest": split_manifest(args.smoke),
        "frames_per_sequence": 32, "scored_frames": "1..31; frame 0 initializes the tracker",
        "conditions": list(CONDITIONS), "image_size": [96, 96], "channels": "RGB uint8",
        "color_distance_threshold": .18, "patch_size": 5, "covariance_eigenvalue_floor": .25,
        "static_anisotropy_weight": .25, "temporal_weight": 1., "temporal_cap": 10.,
        "calibration_condition": "clean", "target_calibration_coverage": .8,
        "failure_tolerance_image_diagonal": FAILURE_TOLERANCE,
        "score_direction": "lower is more reliable", "threads_cap": THREADS_CAP, "device": "CPU",
    }
    destination.mkdir(parents=True)
    write_json(destination / "config.json", config)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    clock_start = time.monotonic()
    def hard_timeout(_signal, _frame):
        raise TimeoutError("Hard wall-clock limit of 1800 seconds exceeded")
    signal.signal(signal.SIGALRM, hard_timeout)
    signal.alarm(1800)
    try:
        rows, summary, computation_seconds = run(smoke=args.smoke, bootstrap_repeats=config["bootstrap_repeats"],
            progress=lambda record: print(f"Completed sequence {record['sequence_id']} ({record['partition']})", flush=True) if record["sequence_id"] % 20 == 0 else None)
        for row in rows:
            for method in METHODS:
                row[f"accepted_{method}"] = accepted(row, method, summary["thresholds"].get(method))
        write_csv(destination / "predictions.csv", rows)
        write_csv(destination / "sequence_totals.csv", sequence_totals(rows, summary["thresholds"]))
        write_json(destination / "summary.json", summary)
        make_figure(summary, destination / "figure.png")
        make_examples(rows, destination / "examples.png", args.smoke)
        cpu = platform.processor() or platform.machine()
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        provenance = {
            "status": "completed", "started_at_utc": started,
            "elapsed_seconds": time.monotonic() - clock_start, "computation_seconds": computation_seconds,
            "git_revision": revision, "source_sha256": source_hashes, "source_dirty": source_dirty,
            "protocol_commit": git_value("log", "-1", "--format=%H", "--", portable_path(protocol)) if protocol.exists() else None,
            "protocol_sha256": config["protocol_sha256"], "config_sha256": sha256(destination / "config.json"),
            "command": f"python scripts/run_experiment.py --output {portable_path(destination)}" + (" --smoke" if args.smoke else "") + f" --protocol {portable_path(protocol)}",
            "python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            "threads_cap": THREADS_CAP,
            "thread_environment": {variable: os.environ[variable] for variable in THREAD_VARIABLES},
            "package_version": __version__,
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "matplotlib")},
            "artifact_sha256": {name: sha256(destination / name) for name in ("predictions.csv", "sequence_totals.csv", "summary.json", "figure.png", "examples.png")},
        }
        write_json(destination / "provenance.json", provenance)
        print(f"Completed {len(rows)} scored frames in {provenance['elapsed_seconds']:.2f}s; output {portable_path(destination)}")
    except Exception as error:
        write_json(destination / "failure.json", {"status": "failed", "error_type": type(error).__name__,
            "message": str(error).replace(str(ROOT), "<repo>"), "elapsed_seconds": time.monotonic() - clock_start})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
