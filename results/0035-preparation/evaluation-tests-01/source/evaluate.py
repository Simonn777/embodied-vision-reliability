"""Saved-only D35 evaluation. No model imports, weight loading, or prediction.

CLI --spec JSON. admit.evaluation_context verifies the complete prediction,
qualification, acquisition and seal-before-GT chains BEFORE scientific imports.
It returns prediction_receipt/root, annotation_paths, installation, and bindings;
admit.postcheck(context) rechecks bound files/runtime without subprocesses.
Only a separately clean outer operation can finalize the continuation decision.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

COHORT = (("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69))
COMMON_SHA256 = "751355a33b6ed468208d8f5aa92c8b4b72522f534159814928799e56634407c7"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
NPZ_KEYS = {"low_bank", "native_original_logits", "object_ids", "selected_slots",
            "alternative_slots", "predicted_quality", "object_score_logits", "labels"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(loader)
    sys.modules[name] = result
    loader.loader.exec_module(result)
    return result


def frame_path(root, descriptor):
    root, relative = Path(root).resolve(), Path(descriptor["path"])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts and
            path.is_relative_to(root), "prediction output path escapes root")
    require(type(descriptor["bytes"]) is int and descriptor["bytes"] > 0 and
            path.stat().st_size == descriptor["bytes"] and
            sha(path) == descriptor["sha256"], "prediction file bytes changed")
    return path


def reconstruct_frame(path, row, frame_record, *, np, torch, metrics):
    """Restore native O×1 interpolation, then verify all sealed categorical maps."""
    ids, shape = row["object_ids"], tuple(row["shape"])
    objects, frame = len(ids), frame_record["frame"]
    count = 1 if frame == 0 else 3
    require(frame_record["bank_count"] == count and
            frame_record["selected_low_exact"] is True and
            frame_record["selected_original_exact"] is True and
            frame_record["initial_diagnostic_only"] is (frame == 0),
            "producer parity/initial contract")
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == NPZ_KEYS, "exact native bank NPZ keys")
        bank = {name: saved[name] for name in NPZ_KEYS}
    expected = {
        "low_bank": (np.dtype("float32"), (objects, count, 256, 256)),
        "native_original_logits": (np.dtype("float32"), (objects, *shape)),
        "object_ids": (np.dtype("int64"), (objects,)),
        "selected_slots": (np.dtype("int64"), (objects,)),
        "alternative_slots": (np.dtype("int64"), (objects, count - 1)),
        "predicted_quality": (np.dtype("float32"), (objects, count)),
        "object_score_logits": (np.dtype("float32"), (objects,)),
        "labels": (np.dtype("uint8"), (4, *shape)),
    }
    for key, (dtype, sizes) in expected.items():
        require(bank[key].dtype == dtype and bank[key].shape == sizes and
                np.isfinite(bank[key]).all(), "bank finite dtype/shape: " + key)
    require(bank["object_ids"].tolist() == ids == list(range(1, objects + 1)),
            "complete contiguous object IDs")
    selected = bank["selected_slots"]
    require(np.array_equal(selected, np.argmax(bank["predicted_quality"], axis=1)),
            "native quality argmax/first tie")
    alternatives = np.asarray([[j for j in range(count) if j != int(winner)]
                               for winner in selected], dtype=np.int64).reshape(objects, count - 1)
    require(np.array_equal(bank["alternative_slots"], alternatives), "original slot alternative order")
    for key in ("selected_slots", "alternative_slots", "predicted_quality", "object_score_logits"):
        require(np.array_equal(bank[key], np.asarray(frame_record[key], dtype=bank[key].dtype)),
                "receipt/NPZ metadata mismatch: " + key)
    absent = bank["object_score_logits"] <= 0
    require(not np.any(bank["low_bank"][absent] > 0), "absent raw masks must stay gated")

    def resize(slots):
        # This O×1 layout matches the producer, not an O×3 interpolation.
        low = bank["low_bank"][np.arange(objects), slots]
        tensor = torch.from_numpy(low).reshape(objects, 1, 256, 256).contiguous()
        if shape != (256, 256):
            tensor = torch.nn.functional.interpolate(
                tensor, size=shape, mode="bilinear", align_corners=False)
        return tensor[:, 0].numpy()

    with torch.inference_mode():
        require(np.array_equal(resize(selected), bank["native_original_logits"]),
                "selected original-resolution native parity")
        original = np.empty((objects, 3, *shape), dtype=np.float32)
        original[:, 0] = bank["native_original_logits"]
        for index in range(2):
            original[:, index + 1] = original[:, 0] if frame == 0 else resize(alternatives[:, index])
    require(np.isfinite(original).all(), "finite restored original bank")
    require((original[:, 0] > 0).sum(axis=(1, 2)).tolist() ==
            frame_record["native_positive_pixels"], "native positive support count")
    controls = metrics.compose_controls(original, ids)
    for index, name in enumerate(metrics.CONTROLS):
        require(np.array_equal(controls[name], bank["labels"][index]),
                "sealed categorical composition mismatch: " + name)
    return original, controls


def continuation(primary, controls):
    """Only prospective numerical clauses; clean evaluator outer remains required."""
    methods, delta = primary["methods"], primary["oracle_decomposition"]
    strongest = {metric: max(methods[name][metric] for name in controls) for metric in ("J", "Jtr")}
    margins = {metric: methods["O"][metric] - strongest[metric] for metric in strongest}
    clip_deltas = {row["sequence"]:
        row["oracle_decomposition"]["Jtr"]["additional_support_O_minus_O_B"]
        for row in primary["per_sequence"]}
    clauses = {
        "total_ceiling_J_at_least_5pp": delta["J"]["total_O_minus_B"] >= .05,
        "total_ceiling_Jtr_at_least_5pp": delta["Jtr"]["total_O_minus_B"] >= .05,
        "extra_support_J_at_least_2pp": delta["J"]["additional_support_O_minus_O_B"] >= .02,
        "extra_support_Jtr_at_least_3pp": delta["Jtr"]["additional_support_O_minus_O_B"] >= .03,
        "ceiling_over_best_real_J_at_least_3pp": margins["J"] >= .03,
        "ceiling_over_best_real_Jtr_at_least_3pp": margins["Jtr"] >= .03,
        "extra_Jtr_at_least_1pp_in_three_clips": sum(v >= .01 for v in clip_deltas.values()) >= 3,
    }
    return dict(numerical_clauses=clauses, numerical_passed=all(clauses.values()),
                clean_evaluation_outer_required=True,
                strongest_real_by_metric=strongest, ceiling_over_best_real=margins,
                per_clip_extra_support_Jtr=clip_deltas,
                interpretation="necessary information screen only; not method success or paper gate")


def run(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    here, common_path = Path(__file__).resolve().parent, Path(spec["common_path"]).resolve()
    sources, spec_hash = spec["source_sha256"], sha(spec_path)
    guard_path = Path(spec["repo"]).resolve() / "experiments/0022-frontend/qualify.py"
    require(Path(spec["metrics_path"]).resolve() == here / "metrics.py" and
            Path(spec["guard_source"]).resolve() == guard_path, "fixed metrics and guard source")
    for path in (Path(__file__).resolve(), here / "metrics.py", here / "admit.py",
                 here / "ADMISSION.md", common_path, guard_path):
        require(str(path) in sources and sha(path) == sources[str(path)], "bound evaluator dependency")
    require(sha(common_path) == COMMON_SHA256, "unchanged serialization/lifecycle helper")
    require(all(sha(path) == digest for path, digest in sources.items()), "all source bytes")
    common = module("d35_evaluate_common", common_path)
    spec, out, record, began = common.start(spec_path, "transformation-support-evaluation")
    record.update(schema="evr0035.evaluation.v1", model_calls=0, image_encoder_calls=0,
                  native_calls=0, label_decode_attempts=0, label_decodes=0,
                  completed_frames=0, completed_sequences=0, parity_frames=0,
                  sealed_predictions_admitted_before_gt=False, offline_attempts=[],
                  compatibility_probes=[], outputs=[])
    try:
        admission = module("d35_evaluation_admission", here / "admit.py")
        context = admission.evaluation_context(spec)
        producer, root = context["prediction_receipt"], Path(context["prediction_root"])
        installation = Path(context["installation"]).resolve()
        rows = producer["rows"]
        require([(r["sequence"], r["frames"]) for r in rows] == list(COHORT) and
                producer["completed_frames"] == 446 and producer["completed_sequences"] == 4,
                "all four complete prediction sequences")
        require(producer["outputs"] == [p for r in rows for p in r["predictions"]],
                "complete flat prediction inventory")
        require(sys.flags.no_site == 1 and sys.version_info[:3] == (3, 12, 13) and
                Path(sys.executable).absolute() == installation / "venv/bin/python",
                "qualified CP312 interpreter with -S")
        require(all(os.environ.get(k) == "4" for k in THREAD_KEYS), "four-thread envelope")
        guard = module("d35_evaluation_guard", guard_path)
        guard.install_guard(record["offline_attempts"], record["compatibility_probes"])
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        import site
        site.main()
        import numpy as np
        import torch
        from PIL import Image
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        metrics = module("d35_saved_metrics", here / "metrics.py")
        record.update(sealed_predictions_admitted_before_gt=True,
                      admission=context.get("bindings", {}),
                      prediction_spec_sha256=producer["spec_sha256"])
        common.save(out / "receipt.json", record)
        summaries = {policy: [] for policy in metrics.POLICIES}
        for row in rows:
            name, count, ids = row["sequence"], row["frames"], row["object_ids"]
            require(row["status"] == "completed" and row["completed_frames"] == count and
                    row["object_count"] == len(ids) and 0 < len(ids) <= 254 and
                    len(row["shape"]) == 2 and min(row["shape"]) > 0 and
                    [p["frame"] for p in row["predictions"]] ==
                    [p["frame"] for p in row["frame_records"]] == list(range(count)),
                    "complete per-frame metadata and identities")
            frame_scores = {policy: [] for policy in metrics.POLICIES}
            void_counts = []
            for descriptor, frame_record in zip(row["predictions"], row["frame_records"]):
                frame = frame_record["frame"]
                require(descriptor["sequence"] == name and descriptor["kind"] == "native-bank-frame",
                        "prediction identity")
                path = frame_path(root, descriptor)
                original, controls = reconstruct_frame(
                    path, row, frame_record, np=np, torch=torch, metrics=metrics)
                record["parity_frames"] += 1
                # Annotation encoded bytes and seal-before-acquisition proof were
                # admitted globally; no truth is used to reconstruct real arms.
                gt_path = Path(context["annotation_paths"][name][frame])
                require(sha(gt_path) == context["fingerprints"][str(gt_path)],
                        "annotation bytes changed before decode")
                record["label_decode_attempts"] += 1
                common.save(out / "receipt.json", record)
                with Image.open(gt_path) as image:
                    require(image.format == "PNG" and image.mode in ("P", "L"), "indexed GT PNG")
                    truth = np.array(image)
                require(truth.dtype == np.uint8 and truth.shape == tuple(row["shape"]), "GT type/shape")
                if frame == 0:
                    require(list(metrics.initial_object_ids(truth)) == ids, "initial object set unchanged")
                record["label_decodes"] += 1
                void_counts.append(int(np.count_nonzero(truth == 255)))
                for policy in metrics.POLICIES:
                    oracle = metrics.support_oracles(original, truth, ids, void_policy=policy)
                    scores = metrics.score_frame(truth, {**controls, **oracle}, ids, void_policy=policy)
                    for index in range(len(ids)):
                        require(all(scores[real][index] <= scores["O"][index]
                                    for real in metrics.CONTROLS) and
                                scores["B"][index] <= scores["O_B"][index] <= scores["O"][index],
                                "support-ceiling consistency")
                    frame_scores[policy].append(scores)
                record["completed_frames"] += 1
                del original, controls, truth, oracle, scores
            for policy in metrics.POLICIES:
                summary = metrics.summarize_sequence(name, frame_scores[policy], ids, void_policy=policy)
                summary.update(original_255_pixels=void_counts, shape=row["shape"],
                               recording_id=row.get("recording_id"))
                summaries[policy].append(summary)
            record["completed_sequences"] += 1
            common.save(out / "receipt.json", record)
        require(record["completed_frames"] == record["parity_frames"] ==
                record["label_decodes"] == record["label_decode_attempts"] == 446 and
                record["completed_sequences"] == 4, "complete finite all-frame evaluation")
        reports = {policy: metrics.aggregate_sequences(items) for policy, items in summaries.items()}
        decision = continuation(reports["author_background"], metrics.CONTROLS)
        report = dict(schema="evr0035.metrics.v1", metric_scale="[0,1]; times100 is percentage points",
            official_benchmark_result=False, deployed_oracle=False, F_computed=False,
            reference_commit="fe274574cb03c8a3ea83e121dd76e20b703781fd",
            primary=reports["author_background"], void_ignore_sensitivity=reports["void_ignore"],
            decision=decision, source_sha256=sources, spec_sha256=spec_hash)
        admission.postcheck(context)
        require(not record["offline_attempts"] and sha(spec_path) == spec_hash and
                all(sha(path) == digest for path, digest in sources.items()),
                "offline/input/source immutability")
        common.save(out / "evaluation.json", report)
        result_path = out / "evaluation.json"
        record.update(status="completed", numeric_gate_passed=decision["numerical_passed"],
            clean_outer_required=True, outputs=[dict(path="evaluation.json",
                sha256=sha(result_path), bytes=result_path.stat().st_size)],
            wall_seconds=time.perf_counter() - began)
    except BaseException as error:
        record.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        common.save(out / "receipt.json", record)
    common.save(out / "seal.json", dict(schema="evr0035.evaluation-seal.v1", complete=True,
        receipt_sha256=sha(out / "receipt.json"), spec_sha256=spec_hash,
        source_sha256=sources, outputs=record["outputs"], completed_frames=446, completed_sequences=4))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    run(parser.parse_args().spec)
