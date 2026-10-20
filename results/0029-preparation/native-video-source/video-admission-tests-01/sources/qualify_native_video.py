"""Manufactured-only qualification of the unchanged fixed-video worker.

Source preparation only. Nothing is generated or imported from ML on import.
The caller owns frozen-source/runtime/checkpoint admission, the offline guard,
sys.path setup and the 4-thread/8-GiB/1,800-second process supervisor.

run(inputs, output, record) accepts the worker's verified helper/source/runtime/
checkpoint bindings plus native_video. No video/media paths are accepted.
It builds a separate 25-frame manufactured fixture from q.fixture and calls the
UNCHANGED native_video.run. Its internal dog-agility/25 identity exercises the
worker's fixed contract only: this is not DAVIS data or scientific evidence.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

INTERNAL_SEQUENCE = "dog-agility"
FRAME_COUNT = 25
SHIFT_PIXELS = 8
QUERY_FRAMES = (4, 7, 10, 13, 15, 18, 21, 24)
POSITIVE_XY = ((224, 136), (252, 136), (224, 164), (252, 164))
NEGATIVE_XY = ((40, 40), (100, 40), (40, 300), (100, 300))
INPUT_KEYS = {
    "qualify_helper", "native_fixture", "native_video", "diagnostic_core",
    "sam2_source", "runtime", "sam2_checkpoint", "source_sha256",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def file_record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify_worker(result):
    """Check only technical behavior, not segmentation or selector accuracy."""
    require(result["status"] == "completed" and result["complete"] is True
            and result["sequence"] == INTERNAL_SEQUENCE and result["frame_count"] == FRAME_COUNT
            and result["query_frames"] == list(QUERY_FRAMES) and result["query_count"] == 8,
            "manufactured worker did not complete the exact sequence")
    require(result["noninitial_annotation_access"] is False
            and result["actionable_count"] == 7, "manufactured input scope changed")
    expected_counts = dict(image_encoder_intent=25, image_encoder_completed=25,
        track_step_intent=63, track_step_completed=63, baseline_frames=25,
        verification_intent=8, verification_completed=8,
        deletion_intent=16, deletion_completed=16,
        prompt_intent=14, prompt_completed=14, native_bank_extra_calls=0)
    require(result["calls"] == expected_counts, "manufactured native call accounting")
    require([row["frame"] for row in result["baseline"]] == list(range(FRAME_COUNT)),
            "baseline frame omission")
    require([row["frame"] for row in result["queries"]] == list(QUERY_FRAMES),
            "query order/coverage changed")
    pointer_batch_sizes = []
    position_difference_count = 0
    for index, row in enumerate(result["queries"]):
        require(row["status"] == "completed" and row["target_id"] == 1
                and row["actionable"] is (index != 0), "query status/identity/actionability")
        require(row["witness_counts"] == {"positive": 4, "negative": 3 if index == 0 else 4},
                "manufactured witness counts changed")
        require(row["baseline_state_sha256_before"] == row["baseline_state_sha256_after"],
                "branch altered complete baseline state")
        expected_candidates = {"B", "D1", "D3", "M0", "M1", "M2"}
        if index != 0:
            expected_candidates |= {"P", "R"}
        require(set(row["candidates"]) == expected_candidates
                and all(x["status"] == "completed" for x in row["candidates"].values()),
                "native candidate completeness")
        require(row["native_bank"]["count"] == 3
                and row["native_bank"]["captured_existing_call"] is True
                and row["native_bank"]["extra_calls"] == 0
                and row["native_bank"]["selected_low_exact"] is True
                and row["native_bank"]["selected_original_exact"] is True,
                "original native decoder-bank parity")
        baseline = row["baseline_trace"]["trace"]
        pointer_batch_sizes.append(len(baseline["pointer_frames"]))
        require(baseline["frame"] == row["frame"] and baseline["num_frames"] == FRAME_COUNT
                and baseline["pointer_limit"] == 16
                and baseline["pointer_distance_denominator"] == 15
                and baseline["actual_batch_formula_exact"] is True,
                "general pointer range/time normalization not exercised")
        require([b["name"] for b in row["branches"]] ==
                ["NOOP", "D1", "D3"] + (["P", "R"] if index != 0 else []),
                "branch order/coverage")
        for branch in row["branches"]:
            require(branch["status"] == "completed" and branch["call_attempted"] is True
                    and branch["call_completed"] is True
                    and branch["baseline_state_unchanged"] is True
                    and branch["trace"]["trace"]["actual_batch_formula_exact"] is True,
                    "branch execution/isolation check failed")
            if branch["name"] == "NOOP":
                require(branch["exact_baseline_original_parity"] is True
                        and branch["exact_baseline_low_parity"] is True, "no-op parity missing")
            if branch["name"] != "P":
                require(branch["fork_state_changed"] is False, "read-only fork changed")
            if branch["name"] in ("D1", "D3"):
                deletion = branch["deletion"]
                require(deletion["retained_features_exact"] is True
                        and deletion["retained_spatial_positions_exact"] is True
                        and deletion["retained_pointer_temporal_coordinates_exact"] is True
                        and deletion["actual_batch_formulas_exact"] is True
                        and deletion["no_backfill"] is True
                        and deletion["removed_kinds"] == ["spatial", "pointer"],
                        "both-carrier deletion verification missing")
                position_difference_count += len(deletion["pointer_position_cross_batch_differences"])
            if branch["name"] == "R":
                trace = branch["trace"]
                require(trace["trace"]["spatial_frames"] == [0]
                        and trace["trace"]["pointer_frames"] == [0]
                        and trace["previous_mask_sha256"] is None, "reset scope mismatch")
        require(row["selection"]["actionable"] is (index != 0), "selection availability changed")
        if index == 0:
            require(all(x == "B" for x in row["selection"]["choices"].values())
                    and row["selection"]["signed_action"] is False,
                    "nonactionable query did not preserve all baseline choices")
    require(pointer_batch_sizes[0] == 4 and pointer_batch_sizes[-1] == 16,
            "early and saturated pointer batches were not covered")
    return dict(technical_checks_passed=True, scientific_metrics_computed=False,
        query_count=8, actionable_queries=7, pointer_batch_sizes=pointer_batch_sizes,
        pointer_position_cross_batch_differences=position_difference_count,
        cross_batch_position_difference_is_failure=False,
        no_mask_accuracy_or_selector_gain_gate=True, expected_calls=expected_counts)


def run(inputs, output, record):
    require(set(inputs) == INPUT_KEYS, "qualification accepts only verified source/runtime/model bindings")
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "fresh manufactured output required")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="evr0029.native-video-qualification.v1", status="started", complete=False,
        manufactured_only=True, scientific_eligible=False, davis_data_access=False,
        noninitial_annotation_access=False, source_sha256=inputs["source_sha256"],
        fixture=dict(internal_worker_sequence=INTERNAL_SEQUENCE, frame_count=FRAME_COUNT,
            internal_identity_is_not_data_provenance=True, shape=[384, 512],
            right_shift_per_frame=SHIFT_PIXELS, padding="zero",
            jpeg_quality=95, jpeg_subsampling=0,
            quantization="floor(255*clip(x,0,1)+0.5)",
            initial_mask_yx=[118, 182, 207, 271], query_frames=list(QUERY_FRAMES),
            positive_initial_xy=[list(x) for x in POSITIVE_XY],
            negative_initial_xy=[list(x) for x in NEGATIVE_XY],
            first_query_omits_last_negative=True,
            witness_matches="manufactured identity correspondence, score1; unchanged select_witnesses",
            no_superpoint_or_lightglue_inference=True),
        inputs=[], worker_record={}, verification=None)
    record["native_video_qualification"] = result
    began = time.perf_counter()

    def save():
        result["wall_seconds"] = time.perf_counter() - began
        pending = output / "qualification-record.pending"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(pending, output / "qualification-record.json")

    save()
    try:
        source_keys = ("qualify_helper", "native_fixture", "native_video", "diagnostic_core")
        result["observed_helper_sha256"] = {key: sha(inputs[key]) for key in source_keys}
        q = module("_evr0029_manufactured_video_q", inputs["qualify_helper"])
        core = module("_evr0029_manufactured_video_core", inputs["diagnostic_core"])
        worker = module("_evr0029_manufactured_video_worker", inputs["native_video"])
        import numpy as np
        from PIL import Image
        require(core.query_schedule(FRAME_COUNT) == list(QUERY_FRAMES), "fixed schedule changed")
        result["stage"] = "manufacture_inputs"
        save()
        start = time.perf_counter()
        images, description = q.fixture(np)
        base = images[0]
        require(base.shape == (384, 512, 3), "qualified artificial texture shape changed")
        result["fixture"]["base"] = description
        frames = output / "manufactured-frames"
        frames.mkdir()
        for frame in range(FRAME_COUNT):
            image = np.zeros_like(base)
            shift = SHIFT_PIXELS * frame
            if shift == 0:
                image[:] = base
            else:
                image[:, shift:] = base[:, :512 - shift]
            pixels = np.floor(255 * np.clip(image, 0, 1) + .5).astype(np.uint8)
            path = frames / ("%05d.jpg" % frame)
            with path.open("xb") as stream:
                Image.fromarray(pixels).save(stream, format="JPEG", quality=95, subsampling=0)
            result["inputs"].append(dict(kind="manufactured_rgb", frame=frame, **file_record(path)))
        labels = np.zeros((384, 512), dtype=np.uint8)
        labels[118:182, 207:271] = 1
        initial_path = output / "manufactured-initial.png"
        with initial_path.open("xb") as stream:
            Image.fromarray(labels).save(stream, format="PNG")
        initial = file_record(initial_path)
        result["inputs"].append(dict(kind="manufactured_initial_mask", **initial))
        source_xy = np.asarray(POSITIVE_XY + NEGATIVE_XY, dtype=np.float64)
        witnesses = []
        for index, frame in enumerate(QUERY_FRAMES):
            target_xy = source_xy + np.asarray([SHIFT_PIXELS * frame, 0.])
            used = 7 if index == 0 else 8
            ids = np.stack((np.arange(used), np.arange(used)), axis=1).astype(np.int64)
            witness = core.select_witnesses(labels, source_xy, target_xy, ids,
                                           np.ones(used), (384, 512), target_id=1)
            witness.update(sequence=INTERNAL_SEQUENCE, frame=frame)
            require(len(witness["positive"]) == 4
                    and len(witness["negative"]) == (3 if index == 0 else 4)
                    and witness["actionable"] is (index != 0), "manufactured witness construction failed")
            path = output / ("manufactured-witness-%05d.json" % frame)
            with path.open("x") as stream:
                json.dump(witness, stream, indent=2, allow_nan=False)
                stream.write("\n")
            item = file_record(path)
            witnesses.append(dict(frame=frame, **item))
            result["inputs"].append(dict(kind="manufactured_witness", frame=frame, **item))
        result["manufacture_seconds"] = time.perf_counter() - start
        video = dict(sequence=INTERNAL_SEQUENCE, frame_count=FRAME_COUNT, frames_dir=str(frames),
                     initial_mask=initial, witnesses=witnesses)
        worker_inputs = {key: value for key, value in inputs.items() if key != "native_video"}
        worker_inputs["video"] = video
        result["stage"] = "unchanged_native_video_worker"
        save()
        start = time.perf_counter()
        try:
            completed = worker.run(worker_inputs, output / "worker", result["worker_record"])
        finally:
            result["worker_seconds"] = time.perf_counter() - start
            save()
        result["stage"] = "technical_verification"
        save()
        result["verification"] = verify_worker(completed)
        for item in result["inputs"]:
            path = Path(item["path"])
            require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                    "manufactured input changed during worker")
        require(result["observed_helper_sha256"] ==
                {key: sha(inputs[key]) for key in source_keys}, "qualification helper source changed")
        worker_record = output / "worker" / "video-record.json"
        result["worker_receipt"] = file_record(worker_record)
        result.update(status="completed", complete=True, stage="completed")
    except BaseException as exc:
        result.update(status="failed", complete=False, error_type=type(exc).__name__,
                      error=str(exc), failed_stage=result.get("stage"))
        raise
    finally:
        save()
    return result
