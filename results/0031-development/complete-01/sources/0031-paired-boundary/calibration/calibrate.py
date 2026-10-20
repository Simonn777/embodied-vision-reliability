"""One global TRAIN-selected output threshold; no model construction or native calls."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

TRAIN = ("dog-agility", "planes-water", "dogs-scale", "dog-gooses")
DEV = ("cat-girl", "upside-down")
THRESHOLDS = (-2, -1, 0, 1, 2)
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    sys.modules[name] = value
    loader.loader.exec_module(value)
    return value


def run(spec_path):
    preliminary = json.loads(Path(spec_path).read_text())
    common_path = Path(preliminary["campaign_root"]) / "common.py"
    import hashlib
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if preliminary["source_sha256"].get(str(common_path)) != digest(common_path):
        raise ValueError("changed serialization helper")
    c = module("calibration_common", common_path)
    spec, out, record, began = c.start(spec_path, "global-output-calibration")
    record.update(model_calls=0, image_encoder_calls=0, native_calls=0, optimizer_steps=0,
                  final_test_access=False, history_modified=False, tensor_loads=0,
                  zero_threshold_parity=[], thresholds=list(THRESHOLDS))
    fingerprints = dict(spec["source_sha256"])
    fingerprints[str(Path(spec_path).resolve())] = record["spec_sha256"]
    try:
        evaluation = Path(spec["evaluation_root"])
        native_source = Path(spec["sam2_source"]) / "sam2"
        required = {str(Path(__file__).resolve()), str(Path(__file__).with_name("SCOPE.md").resolve()), str(common_path),
            str(evaluation / "evaluate_jf.py"), str(evaluation / "evaluate_baseline.py"),
            str(evaluation / "evaluate_rollout.py"),
            str(Path(spec["campaign_root"]) / "cache_baselines.py"),
            str(native_source / "sam2_video_predictor.py"),
            str(native_source / "modeling/sam2_base.py")}
        c.require(set(fingerprints) - {str(Path(spec_path).resolve())} == required, "exact source set")
        c.require(sys.version_info[:3] == (3, 12, 13) and
            Path(sys.executable).absolute() == Path(spec["python"]).absolute(), "qualified interpreter")
        c.require(all(os.environ.get(k) == "4" for k in THREAD_KEYS), "four-thread environment")
        for path, expected in fingerprints.items():
            c.require(c.sha(path) == expected, "source/spec changed: " + path)
        c.require(spec["thresholds"] == list(THRESHOLDS) and
            spec["train_sequences"] == list(TRAIN) and spec["dev_sequences"] == list(DEV),
            "fixed candidate set and split")
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(evaluation))
        e = module("evaluate_jf", evaluation / "evaluate_jf.py")
        admission = module("calibration_admission", evaluation / "evaluate_rollout.py")
        e.cv2.setNumThreads(0)

        def check(d):
            path = e._descriptor(d)
            fingerprints[str(path)] = d["sha256"]
            return path

        keys = {"cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec",
                "mapping", "data_index"}
        c.require(set(spec["bindings"]) == keys, "exact metadata bindings")
        b = {k: check(d) for k, d in spec["bindings"].items()}
        cache, _ = admission.admit(b, "cache", fingerprints, legacy=True)
        c.require(cache["completed_frames"] == 386 and cache["completed_sequences"] == 6 and
            not cache["offline_attempts"] and cache["model_initial_hash"] == cache["model_final_hash"],
            "complete immutable native baseline")
        mapping, index = (json.loads(b[k].read_text()) for k in ("mapping", "data_index"))
        c.require(mapping["status"] == "completed" and mapping["member_count"] == 386 and
            cache["data_index"]["sha256"] == c.sha(b["data_index"]), "complete annotation mapping")
        rows = {r["sequence"]: r for r in cache["rows"]}
        inputs = {r["sequence"]: r for r in index["sequences"]}
        labels = {(r["sequence"], r["frame"]): r for r in mapping["mappings"]}
        c.require(len(rows) == len(cache["rows"]) == len(inputs) == len(index["sequences"]) == 6
            and set(rows) == set(inputs) == set(TRAIN + DEV) and len(labels) ==
            len(mapping["mappings"]) == 386, "fixed complete identity sets")
        files = {r["path"]: r for r in cache["outputs"]}
        c.require(len(files) == len(cache["outputs"]), "duplicate cache output")
        cache_root = b["cache_receipt"].parent
        selected, train_events = {}, []
        for name in TRAIN + DEV:
            row, source = rows[name], inputs[name]
            n, role = row["frames"], "TRAIN" if name in TRAIN else "DEV"
            c.require(row["status"] == "completed" and row["role"] == source["role"] == role and
                row["parent_id"] == source["parent_id"] == name and
                [p["frame"] for p in row["predictions"]] == list(range(n)) and
                [p["frame"] for p in source["labels"]] == list(range(n)), "complete parent frames")
            if name in TRAIN:
                anchors = [1 + i * (n - 5) // 11 for i in range(12)]
                c.require(row["anchors"] == anchors and len(set(anchors)) == 12, "fixed TRAIN anchors")
                events = [(a, h, a + h) for a in anchors for h in (1, 3)]
                train_events.extend((name, a, h, f) for a, h, f in events)
                frames = sorted({f for _, _, f in events})
            else:
                frames = list(range(n))
            for frame in frames:
                label = labels[name, frame]
                c.require(label["role"] == role and all(label[k] == source["labels"][frame][k]
                    for k in ("path", "sha256", "bytes")), "label identity")
                rel = f"{name}/frame-{frame:05d}.pt"
                tensor = dict(files[rel], path=str(cache_root / rel))
                pred = row["predictions"][frame]
                c.require(pred == files[pred["path"]] and pred["sequence"] == name, "prediction identity")
                pred = dict(pred, path=str(cache_root / pred["path"]))
                selected[name, frame] = (check(tensor), check(pred), check(label))
        c.require(len(train_events) == 96 and len({(n, f) for n, _, _, f in train_events}) == 87
            and sum(rows[n]["frames"] for n in DEV) == 154, "96/87 TRAIN and full154 DEV")
        record["bindings"] = spec["bindings"]

        def evaluate_frame(name, frame, thresholds):
            row = rows[name]
            shape, target = tuple(row["shape"]), row["target_id"]
            c.require(len(shape) == 2 and min(shape) == 480 and 0 < target < 255,
                      "original480p target-object contract")
            tensor_path, pred_path, label_path = selected[name, frame]
            item = torch.load(tensor_path, map_location="cpu", weights_only=True)
            record["tensor_loads"] += 1
            low = item["out"]["pred_masks"]
            presence = item["out"]["object_score_logits"]
            c.require(low.device.type == "cpu" and low.dtype == torch.float32 and not low.requires_grad and
                tuple(low.shape) == (1, 1, 256, 256) and torch.isfinite(low).all().item()
                and presence.numel() == 1 and torch.isfinite(presence).all().item(),
                "finite native post-presence logits")
            absent = float(presence.reshape(-1)[0]) <= 0
            if absent:
                c.require(bool((low == -1024).all()), "native hard-absence gate not preserved")
            logits = torch.nn.functional.interpolate(
                low, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()
            baseline = e._prediction(pred_path, [target], shape)[0]
            differences = int(e.np.count_nonzero((logits > 0) != baseline))
            record["zero_threshold_parity"].append(
                dict(sequence=name, frame=frame, differing_pixels=differences, native_absent=absent))
            c.require(differences == 0, "zero threshold differs from saved native prediction")
            # Do not decode this frame's GT unless exact native parity passed.
            truth = e._labels(label_path)
            c.require(tuple(truth.shape) == shape, "original GT dimensions")
            values = {}
            for threshold in thresholds:
                mask = logits > threshold  # Stored hard-absence logits are never replaced.
                c.require(not absent or not bool(mask.any()), "threshold bypassed native absence")
                j, f = e.jf(mask, truth == target)
                values[str(threshold)] = dict(J=j, F=f, JF=(j + f) / 2)
            return dict(sequence=name, frame=frame, target_id=target, native_absent=absent,
                final_frame=frame == row["frames"] - 1, metrics=values)

        train_frames = {(name, f): evaluate_frame(name, f, THRESHOLDS)
            for name in TRAIN for f in sorted({r[3] for r in train_events if r[0] == name})}
        events = [dict(train_frames[name, frame], anchor=anchor, horizon=horizon)
                  for name, anchor, horizon, frame in train_events]
        candidates = []
        for threshold in THRESHOLDS:
            parents = [dict(parent_id=name, event_count=24,
                mean=e._means([r["metrics"][str(threshold)] for r in events if r["sequence"] == name]))
                for name in TRAIN]
            candidates.append(dict(threshold=threshold, parents=parents,
                equal_parent=e._means([r["mean"] for r in parents])))
        chosen = min(candidates, key=lambda r: (-r["equal_parent"]["JF"],
                                               abs(r["threshold"]), r["threshold"]))["threshold"]
        train_report = dict(scope="TRAIN horizon-set target-object J/F; not official benchmark",
            weighting="24 events per parent, including repeated future frames; four equal parents",
            endpoint_policy="all96 declared events including4 final-frame events",
            unique_future_frames=87, event_count=96, rows=events, candidates=candidates,
            selected_threshold=chosen, tie_rule="max JF; closest zero; smaller threshold")
        c.save(out / "train-selection.json", train_report)
        locked = c.sha(out / "train-selection.json")
        record.update(selected_threshold=chosen, train_selection_sha256=locked,
                      train_complete_before_dev_decode=True)
        c.save(out / "receipt.json", record)
        dev_rows, parents = [], []
        for name in DEV:
            rr = [evaluate_frame(name, frame, sorted({0, chosen})) for frame in range(rows[name]["frames"])]
            for r in rr:
                r.update(included=0 < r["frame"] < rows[name]["frames"] - 1,
                    native=r["metrics"]["0"], selected=r["metrics"][str(chosen)])
                del r["metrics"]
                r["delta"] = {k: r["selected"][k] - r["native"][k] for k in ("J", "F", "JF")}
            parents.append(dict(parent_id=name, raw_frames=len(rr), included_frames=len(rr)-2,
                **{arm: e._means([r[arm] for r in rr if r["included"]])
                   for arm in ("native", "selected", "delta")}))
            dev_rows.extend(rr)
        report = dict(scope="object-subset DEV J/F; one globally TRAIN-selected output threshold",
            official_benchmark_result=False, selected_threshold=chosen, raw_frames=154,
            included_frames=150, endpoint_policy="exclude first and last; retain all raw metrics",
            void_policy="official_background", boundary_fraction=e.BOUNDARY_FRACTION,
            metric_reference_commit=e.REFERENCE, rows=dev_rows, parents=parents,
            **{arm: e._means([r[arm] for r in parents]) for arm in ("native", "selected", "delta")})
        c.save(out / "evaluation.json", report)
        c.require(c.sha(out / "train-selection.json") == locked and record["tensor_loads"] ==
            len(record["zero_threshold_parity"]) == 241, "complete unique-frame parity and locked selection")
        for path, expected in fingerprints.items():
            c.require(c.sha(path) == expected, "source/input changed: " + path)
        record.update(status="completed", input_sha256=fingerprints,
            evaluation_sha256=c.sha(out / "evaluation.json"), completed_train_events=96,
            completed_dev_raw_frames=154, completed_dev_interior_frames=150)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)


if __name__ == "__main__":
    run(sys.argv[1])
