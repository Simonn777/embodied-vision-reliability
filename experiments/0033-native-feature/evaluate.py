"""Sealed two-arm predictions: descriptive TRAIN and full DEV J/F, no Torch."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

TRAIN = ("dog-agility", "planes-water", "dogs-scale", "dog-gooses")
DEV = ("cat-girl", "upside-down")
ARMS = ("native", "visual", "logit_only")
PREDICTION_KEYS = {"prediction_receipt", "prediction_outer", "prediction_operation", "prediction_worker_spec"}


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    sys.modules[name] = value
    loader.loader.exec_module(value)
    return value


def run(spec_path):
    spec_path = Path(spec_path).resolve()
    preliminary = json.loads(spec_path.read_text())
    common_path = Path(preliminary["campaign_root"]) / "common.py"
    if preliminary["source_sha256"].get(str(common_path)) != hashlib.sha256(common_path.read_bytes()).hexdigest():
        raise ValueError("serialization helper changed")
    c = module("feature_metric_common", common_path)
    spec, out, record, began = c.start(spec_path, "native-feature-evaluation")
    record.update(schema="evr0033.native-feature-evaluation.v1", model_calls=0, image_encoder_calls=0,
        native_calls=0, final_test_access=False, label_decode_attempted=0, label_decodes=0,
        predictions_sealed_before_gt=False)
    fingerprints = dict(spec["source_sha256"]); fingerprints[str(spec_path)] = record["spec_sha256"]
    try:
        here, evaluation = Path(__file__).resolve().parent, Path(spec["evaluation_root"])
        required = {str(here / n) for n in ("evaluate.py", "METRIC_SCOPE.md", "predict.py", "train.py",
                                           "head.py", "SCOPE.md", "ADMISSION.md")}
        required |= {str(common_path), str(Path(spec["campaign_root"]) / "cache_baselines.py")}
        required |= {str(evaluation / n) for n in ("evaluate_jf.py", "evaluate_baseline.py", "evaluate_rollout.py")}
        c.require(required <= set(fingerprints), "required metric source closure")
        for path, digest in fingerprints.items():
            c.require(c.sha(path) == digest, "source/spec changed")
        p = module("feature_metric_provenance", here / "predict.py")  # stdlib-only, no head import
        c.require(sys.version_info[:3] == (3, 12, 13)
            and Path(sys.executable).absolute() == Path(spec["python"]).absolute()
            and all(os.environ.get(k) == "4" for k in p.THREAD_KEYS), "qualified interpreter/threads")
        sys.path.insert(0, str(evaluation))
        e = module("evaluate_jf", evaluation / "evaluate_jf.py")
        admission = module("feature_metric_admission", evaluation / "evaluate_rollout.py")
        e.cv2.setNumThreads(0)
        c.require(set(spec["bindings"]) == p.OLD_KEYS | p.FIT_KEYS | PREDICTION_KEYS,
                  "exact fourteen metadata bindings")
        b = {k: p.check(d, fingerprints) for k, d in spec["bindings"].items()}
        baseline_path = p.check(spec["baseline_reference"], fingerprints)
        cache, fit, _ = p.admit_fit(admission, b, spec["bindings"], fingerprints, here)
        prediction, worker = admission.admit(b, "prediction", fingerprints)
        c.require(prediction["schema"] == "evr0033.native-feature-prediction.v1"
            and prediction["arms"] == list(p.ARMS)
            and prediction["completed_frames"] == prediction["native_parity_checks"] == 241
            and prediction["completed_arm_predictions"] == prediction["zero_head_parity_checks"] == 482
            and prediction["current_label_decodes"] == 0 and not prediction["offline_attempts"]
            and prediction["native_calls"] == prediction["image_encoder_calls"] == 0
            and prediction["native_weights_loaded"] is False and prediction["learned_weights_unchanged"] is True
            and prediction["native_state_hash_reference"] == cache["model_final_hash"],
            "complete GT-free two-arm prediction")
        inherited = {k: spec["bindings"][k] for k in p.OLD_KEYS | p.FIT_KEYS}
        c.require(prediction["bindings"] == worker["bindings"] == inherited
            and prediction["normalization"] == fit["normalization"]
            and prediction["terminal_checkpoints"] ==
                {a: fit["arms"][a]["terminal_checkpoint"] for a in p.ARMS}, "different fit/cache admission")
        operation = json.loads(b["prediction_operation"].read_text())
        c.require(Path(operation["command"][-2]).resolve() == here / "predict.py", "prediction command script")
        for path, digest in worker["source_sha256"].items():
            c.require(c.sha(path) == digest and (path not in fingerprints or fingerprints[path] == digest),
                      "prediction source changed")
            fingerprints[path] = digest
        prediction_root, cache_root = b["prediction_receipt"].parent, b["cache_receipt"].parent
        seal_path = prediction_root / "seal.json"
        seal = json.loads(seal_path.read_text()); fingerprints[str(seal_path)] = c.sha(seal_path)
        c.require(seal["schema"] == "evr0033.native-feature-seal.v1" and seal["complete"] is True
            and seal["receipt_sha256"] == c.sha(b["prediction_receipt"])
            and seal["source_sha256"] == prediction["source_sha256"]
            and seal["spec_sha256"] == prediction["spec_sha256"]
            and seal["frame_count"] == 241 and seal["arm_prediction_count"] == 482
            and seal["arms"] == list(p.ARMS) and seal["outputs"] == prediction["outputs"],
            "complete producer seal before GT")
        c.require(len(prediction["outputs"]) == len({d["path"] for d in prediction["outputs"]}) == 482,
                  "unique complete prediction files")
        for d in prediction["outputs"]:
            p.check(p.relative(prediction_root, d), fingerprints)
        mapping, index = (json.loads(b[k].read_text()) for k in ("mapping", "data_index"))
        c.require(cache["data_index"]["sha256"] == c.sha(b["data_index"]), "cache/index identity")
        plans = p.plans(cache, index, mapping)
        rows = {r["sequence"]: r for r in prediction["rows"]}
        bases = {r["sequence"]: r for r in cache["rows"]}
        truths = {(r["sequence"], r["frame"]): r for r in mapping["mappings"]}
        cache_files = {d["path"]: d for d in cache["outputs"]}
        c.require([r["sequence"] for r in prediction["rows"]] == list(TRAIN + DEV)
            and len(rows) == 6 and len(truths) == len(mapping["mappings"]) == 386
            and len(cache_files) == len(cache["outputs"]), "complete unique identities")
        selected, events, expected_outputs = {}, [], []
        for plan in plans:
            name, role = plan["sequence"], plan["role"]
            row, base = rows[name], bases[name]
            c.require(row["status"] == "completed"
                and all(row[k] == plan[k] for k in ("parent_id", "role", "frames", "shape", "target_id", "anchors"))
                and set(row["predictions"]) == set(p.ARMS)
                and all([d["frame"] for d in row["predictions"][a]] == plan["requested_frames"] for a in p.ARMS)
                and [d["frame"] for d in row["frame_records"]] == plan["requested_frames"]
                and all(d["status"] == "completed" and d["native_parity"]
                        and all(d["arms"][a]["zero_head_parity"] for a in p.ARMS) for d in row["frame_records"]),
                "parent prediction/parity completeness")
            c.require(len(row["shape"]) == 2 and min(row["shape"]) == 480
                and type(row["target_id"]) is int and 0 < row["target_id"] < 255,
                "original-resolution target identity")
            if role == "TRAIN":
                events.extend((name, a, h, a + h) for a in row["anchors"] for h in (1, 3))
            for i, frame in enumerate(plan["requested_frames"]):
                native = base["predictions"][frame]
                c.require(native == cache_files[native["path"]], "native inventory")
                paths = {"native": p.check(p.relative(cache_root, native), fingerprints)}
                for arm in p.ARMS:
                    d = row["predictions"][arm][i]
                    c.require(d["sequence"] == name and d["arm"] == arm and d["kind"] == "prediction",
                              "learned prediction identity")
                    expected_outputs.append(d)
                    paths[arm] = p.check(p.relative(prediction_root, d), fingerprints)
                selected[name, frame] = (paths, p.check(truths[name, frame], fingerprints), row["frame_records"][i])
        c.require(expected_outputs == prediction["outputs"] and len(selected) == 241
                  and len(events) == 96 and len({(n, f) for n, _, _, f in events}) == 87,
                  "both arms on complete fixed population")
        # Validate all predictions and preserved no-ops before first numeric label access.
        for (name, frame), (paths, _, detail) in selected.items():
            args = ([rows[name]["target_id"]], tuple(rows[name]["shape"]))
            native = e._prediction(paths["native"], *args)
            c.require(math.isfinite(detail["native_presence"]), "finite saved native presence")
            reason = ("DEV_initial" if name in DEV and frame == 0 else
                      "native_nonpositive_presence" if detail["native_presence"] <= 0 else
                      "native_empty" if not native.any() else None)
            c.require(detail["noop_reason"] == reason and all(
                detail["arms"][a]["head_called"] == (reason is None) for a in p.ARMS),
                "fixed native no-op rule")
            for arm in p.ARMS:
                mask = e._prediction(paths[arm], *args)
                if detail["noop_reason"] is not None:
                    c.require(e.np.array_equal(mask, native), "declared no-op changed native output")
            if name in DEV and frame == 0:
                c.require(detail["noop_reason"] == "DEV_initial", "DEV initial output not protected")
        c.require(all(prediction["learned_head_calls"][a] ==
                      sum(d["arms"][a]["head_called"] for r in rows.values() for d in r["frame_records"])
                      for a in p.ARMS), "complete head-call accounting")
        record.update(predictions_sealed_before_gt=True, prediction_seal_sha256=c.sha(seal_path),
                      verified_unique_frames=241, verified_arm_predictions=482,
                      bindings=spec["bindings"], baseline_reference=spec["baseline_reference"])
        c.save(out / "receipt.json", record)
        metrics = {}
        for (name, frame), (paths, label_path, _) in selected.items():
            row = rows[name]; shape, target = tuple(row["shape"]), row["target_id"]
            record["label_decode_attempted"] += 1; c.save(out / "receipt.json", record)
            truth = e._labels(label_path)
            c.require(tuple(truth.shape) == shape, "GT shape changed")
            record["label_decodes"] += 1
            entry = dict(sequence=name, parent_id=name, frame=frame, target_id=target,
                final_frame=frame == row["frames"] - 1,
                void_pixels_as_background=int(e.np.count_nonzero(truth == 255)))
            for arm in ARMS:
                j, f = e.jf(e._prediction(paths[arm], [target], shape)[0], truth == target)
                entry[arm] = dict(J=j, F=f, JF=(j + f) / 2)
            entry["delta_from_native"] = {a: {k: entry[a][k] - entry["native"][k] for k in ("J", "F", "JF")}
                                          for a in p.ARMS}
            metrics[name, frame] = entry

        def summarize(rr, names):
            parents = []
            for name in names:
                chosen = [r for r in rr if r["parent_id"] == name and r["included"]]
                parent = dict(parent_id=name, included_count=len(chosen),
                              **{a: e._means([r[a] for r in chosen]) for a in ARMS})
                parent["delta_from_native"] = {a: {k: parent[a][k] - parent["native"][k] for k in ("J", "F", "JF")}
                                               for a in p.ARMS}
                parents.append(parent)
            result = dict(rows=rr, parents=parents, **{a: e._means([x[a] for x in parents]) for a in ARMS})
            result["delta_from_native"] = {a: {k: result[a][k] - result["native"][k] for k in ("J", "F", "JF")}
                                           for a in p.ARMS}
            result["visual_minus_logit_only"] = {k: result["visual"][k] - result["logit_only"][k] for k in ("J", "F", "JF")}
            return result
        train_rows = [dict(metrics[n, f], anchor=a, horizon=h, included=True) for n, a, h, f in events]
        dev_rows = [dict(metrics[n, f], included=0 < f < rows[n]["frames"] - 1)
                    for n in DEV for f in range(rows[n]["frames"])]
        train, dev = summarize(train_rows, TRAIN), summarize(dev_rows, DEV)
        reference = json.loads(baseline_path.read_text())
        c.require(reference["schema"] == "evr0030.development-jf.cohort.v1"
                  and [x["parent_id"] for x in reference["parents"]] == list(DEV), "native reference identity")
        for observed, expected in [(dev["native"], reference["mean"])] + [
                (x["native"], y["mean"]) for x, y in zip(dev["parents"], reference["parents"])]:
            c.require(all(math.isclose(observed[k], expected[k], abs_tol=1e-12, rel_tol=0)
                          for k in ("J", "F", "JF")), "published native baseline parity")
        delta = dev["delta_from_native"]["visual"]
        clauses = dict(visual_DEV_JF_gain_at_least_1pp=delta["JF"] >= .01,
            visual_both_DEV_parents_positive=all(x["delta_from_native"]["visual"]["JF"] > 0 for x in dev["parents"]),
            visual_mean_JF_above_logit_only=dev["visual"]["JF"] > dev["logit_only"]["JF"],
            visual_mean_J_loss_at_most_half_pp=delta["J"] >= -.005,
            visual_mean_F_loss_at_most_half_pp=delta["F"] >= -.005)
        report = dict(schema="evr0033.native-feature-metrics.v1", official_benchmark_result=False,
            scope="known supervised capacity baseline; no innovation or paper-gate claim",
            metric_scale="[0,1]; deltas times100 are absolute percentage points",
            reference_commit=e.REFERENCE, boundary_fraction=e.BOUNDARY_FRACTION,
            void_policy="official_background", target_policy="native initial target ID; no union",
            train=dict(train, event_count=96, unique_future_frames=87,
                scope="descriptive fit-subset horizon events", weighting="24 events per parent, four equal parents",
                endpoint_policy="all fixed events including4 final-frame events; duplicates retained"),
            dev=dict(dev, raw_frames=154, included_frames=150,
                endpoint_policy="exclude first/last from means; all raw metrics retained",
                weighting="equal two parents; equal interior frames within parent", native_parity_atol=1e-12),
            continuation=dict(clauses=clauses, passed=all(clauses.values()),
                meaning="known-baseline capacity continuation only; not new method or final evidence"),
            producer_costs={k: {x: json.loads(b[k + "_outer"].read_text())[x]
                for x in ("wall_seconds", "peak_group_rss_kib")} for k in ("cache", "fit", "prediction")},
            cost_note="Inherited cache, complete two-arm fit and prediction costs separate; nested times not added twice.")
        c.require(record["label_decodes"] == 241 and sum(r["included"] for r in dev_rows) == 150
                  and sum(r["final_frame"] for r in train_rows) == 4, "complete metric population")
        c.save(out / "evaluation.json", report)
        for path, digest in fingerprints.items():
            c.require(c.sha(path) == digest, "source/input changed during metrics")
        record.update(status="completed", input_sha256=fingerprints,
            evaluation_sha256=c.sha(out / "evaluation.json"), train_events=96, train_unique_frames=87,
            dev_raw_frames=154, dev_interior_frames=150, continuation_passed=report["continuation"]["passed"])
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        import time
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)


if __name__ == "__main__":
    run(sys.argv[1])
