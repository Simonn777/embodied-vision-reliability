"""Privileged TRAIN-only fixed-context target replay; no work/native imports on import."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

TRAIN = ("dog-agility", "planes-water", "dogs-scale", "dog-gooses")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    campaign = Path(spec["campaign_root"])
    required = [Path(__file__).resolve(), campaign / "common.py", campaign / "train.py",
                Path(spec["family_root"]) / "family.py", Path(spec["family_root"]) / "head.py",
                Path(spec["evaluator"])]
    for path in required:
        if spec["source_sha256"].get(str(path)) != sha(path):
            raise ValueError("missing/changed diagnostic source: " + str(path))
    loader = importlib.util.spec_from_file_location("common", campaign / "common.py")
    c = importlib.util.module_from_spec(loader)
    sys.modules["common"] = c
    loader.loader.exec_module(c)
    spec, out, record, began = c.start(spec_path, "teacher-task-alignment")
    runtime, original, encoder_hook = None, None, None
    record.update(scope="privileged TRAIN target-object fixed-context horizon-set J/F",
                  deployment_result=False, official_benchmark_result=False, final_test_access=False,
                  gradients=False, optimization_steps=0, image_encoder_calls=0, track_step_calls=0,
                  outputs=[], history_hashes=[], bindings=spec["bindings"])
    fingerprints = dict(spec["source_sha256"])
    fingerprints[str(Path(spec_path).resolve())] = record["spec_sha256"]

    def descriptor(d):
        path = Path(d["path"])
        c.require(path.stat().st_size == d["bytes"] and sha(path) == d["sha256"], "input changed")
        fingerprints[str(path)] = d["sha256"]
        return path

    def output(path, **metadata):
        row = dict(path=str(path.relative_to(out)), sha256=sha(path),
                   bytes=path.stat().st_size, **metadata)
        record["outputs"].append(row)
        return row

    def checkpoint():
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)

    try:
        c.require(tuple(spec["train_sequences"]) == TRAIN and spec["seed"] == 0, "fixed TRAIN scope")
        b = {k: descriptor(d) for k, d in spec["bindings"].items()}
        c.require(set(b) == {"teacher_receipt", "targets", "mapping", "data_index"}, "bindings")
        teacher = json.loads(b["teacher_receipt"].read_text())
        mapping = json.loads(b["mapping"].read_text())
        index = json.loads(b["data_index"].read_text())
        cache_path = Path(spec["cache"])
        receipt_path = cache_path / "receipt.json"
        c.require(sha(receipt_path) == spec["cache_receipt_sha256"], "cache receipt changed")
        fingerprints[str(receipt_path)] = spec["cache_receipt_sha256"]
        saved_cache = json.loads(receipt_path.read_text())
        c.require(teacher["status"] == mapping["status"] == saved_cache["status"] == "completed"
                  and len(teacher["rows"]) == 48 and teacher["rho"] == spec["rho"]
                  and saved_cache["completed_frames"] == mapping["member_count"] == 386
                  and saved_cache["completed_sequences"] == 6
                  and saved_cache["data_index"]["sha256"] == sha(b["data_index"]), "complete inputs")
        td = teacher["targets"]
        c.require(b["targets"] == b["teacher_receipt"].parent / td["path"] and
                  all(td[k] == spec["bindings"]["targets"][k] for k in ("sha256", "bytes")),
                  "teacher target identity")
        cache_rows = {r["sequence"]: r for r in saved_cache["rows"]}
        source_rows = {r["sequence"]: r for r in index["sequences"]}
        labels = {(r["sequence"], r["frame"]): r for r in mapping["mappings"]}
        identifiers, selected = [], {}
        for sequence in TRAIN:
            row, source = cache_rows[sequence], source_rows[sequence]
            n = row["frames"]
            anchors = [1 + i * (n - 5) // 11 for i in range(12)]
            c.require(row["status"] == "completed" and row["role"] == source["role"] == "TRAIN"
                      and row["parent_id"] == source["parent_id"] == sequence
                      and row["anchors"] == anchors and len(set(anchors)) == 12
                      and [p["frame"] for p in row["predictions"]] == list(range(n)), "TRAIN cache")
            for anchor in anchors:
                identifiers.append((sequence, anchor))
                for horizon in (1, 3):
                    frame = anchor + horizon
                    label = labels[sequence, frame]
                    c.require(label["role"] == "TRAIN" and all(
                        label[k] == source["labels"][frame][k] for k in ("path", "sha256", "bytes")),
                        "TRAIN annotation mapping")
                    pred = dict(row["predictions"][frame])
                    pred["path"] = str(cache_path / pred["path"])
                    descriptor(label); descriptor(pred)  # All 96 encoded pairs before any decode.
                    selected[sequence, anchor, horizon] = (label, pred)
        c.require([(r["sequence"], r["anchor"]) for r in teacher["rows"]] == identifiers,
                  "48 ordered teacher anchors")
        runtime = c.load_runtime(spec, record)
        torch, np, model = runtime["torch"], runtime["np"], runtime["model"]
        train = c.module("alignment_train", campaign / "train.py")
        e = c.module("alignment_metrics", spec["evaluator"])
        family, head_module = train.modules(spec)
        cache = train.Cache(cache_path, spec["cache_receipt_sha256"], torch)
        targets = torch.load(b["targets"], map_location="cpu", weights_only=True)
        c.require([(r["sequence"], r["anchor"]) for r in targets] == identifiers, "target coverage")
        state_hash = lambda value: runtime["nf"].state_hash(value, torch)
        capture = []
        original = model._track_step

        def track(*args, **kwargs):
            record["track_step_calls"] += 1
            result = original(*args, **kwargs)
            capture.append(result[1][3].detach().clone())  # Native selected POST-presence logits.
            return result

        def count_encoder(*unused):
            record["image_encoder_calls"] += 1

        model._track_step = track
        encoder_hook = model.image_encoder.register_forward_hook(count_encoder)
        parents = []
        with torch.no_grad():
            for parent_index, sequence in enumerate(TRAIN):
                meta = cache.meta(sequence)
                history = cache.history(sequence)
                before = state_hash(history)
                anchor_reports = []
                for target_index in range(parent_index * 12, (parent_index + 1) * 12):
                    item, saved = targets[target_index], teacher["rows"][target_index]
                    anchor = item["anchor"]
                    ex = train.example(spec, cache, runtime, family, head_module, sequence, anchor)
                    c.require(state_hash(ex["context"]) == state_hash(item["context"]),
                              "saved target uses a different memory family")
                    target_before = state_hash(item)
                    q = item["multistep"].detach()
                    c.require(q.shape == (64, 2) and q.dtype == torch.float32
                              and bool(torch.isfinite(q).all()), "saved multistep coefficients")
                    rows, losses = [], []
                    for horizon, future in zip((1, 3), ex["futures"]):
                        tick = time.perf_counter()
                        capture.clear()
                        loss = train.future_loss(runtime, ex, q, family, future)
                        c.require(len(capture) == 1 and not capture[0].requires_grad
                                  and bool(torch.isfinite(capture[0]).all()), "single native output")
                        losses.append(float(loss))
                        low = capture.pop()
                        c.require(tuple(low.shape) == (1, 1, 256, 256), "native lowres shape")
                        logits = torch.nn.functional.interpolate(
                            low, size=meta["shape"], mode="bilinear", align_corners=False)
                        mask = (logits[0, 0] > 0).cpu().numpy()
                        path = out / f"{sequence}-anchor-{anchor:05d}-h{horizon}.npz"
                        np.savez_compressed(path, masks=mask[None],
                                            object_ids=np.array([meta["target_id"]], dtype=np.int64))
                        pred_out = output(path, sequence=sequence, anchor=anchor, horizon=horizon)
                        label_d, base_d = selected[sequence, anchor, horizon]
                        truth = e._labels(Path(label_d["path"]))
                        c.require(tuple(truth.shape) == tuple(meta["shape"]), "original GT dimensions")
                        baseline = e._prediction(Path(base_d["path"]), [meta["target_id"]],
                                                 tuple(meta["shape"]))[0]
                        j, f = e.jf(mask, truth == meta["target_id"])
                        bj, bf = e.jf(baseline, truth == meta["target_id"])
                        row = dict(sequence=sequence, parent_id=meta["parent_id"], anchor=anchor,
                            horizon=horizon, frame=future["frame"], target_id=meta["target_id"],
                            final_frame=future["frame"] == meta["frames"] - 1,
                            teacher=dict(J=j, F=f, JF=(j+f)/2),
                            baseline=dict(J=bj, F=bf, JF=(bj+bf)/2),
                            delta=dict(J=j-bj, F=f-bf, JF=(j+f-bj-bf)/2),
                            replay_surrogate_loss=float(loss), prediction=pred_out,
                            baseline_prediction=base_d, annotation=label_d,
                            wall_seconds=time.perf_counter()-tick)
                        rows.append(row); record["rows"].append(row); checkpoint()
                    c.require(state_hash(item) == target_before, "saved target mutated")
                    anchor_reports.append(dict(sequence=sequence, anchor=anchor,
                        teacher=e._means([r["teacher"] for r in rows]),
                        baseline=e._means([r["baseline"] for r in rows]),
                        delta=e._means([r["delta"] for r in rows]),
                        saved_selected_loss=saved["selected_loss"], saved_baseline_loss=saved["baseline_loss"],
                        replay_mean_surrogate_loss=math.fsum(losses)/2, saved_best_step=saved["best_step"]))
                after = state_hash(history)
                record["history_hashes"].append(dict(sequence=sequence, before=before, after=after))
                c.require(after == before, "native cached history/positions/pointers mutated")
                parents.append(dict(parent_id=sequence, anchors=anchor_reports,
                    **{k: e._means([r[k] for r in anchor_reports]) for k in ("teacher", "baseline", "delta")}))
                del ex, history
                cache.histories.clear(); cache.shared.clear()
        c.require(len(record["rows"]) == 96 and record["track_step_calls"] ==
                  runtime.get("future_forward_calls") == 96 and record["image_encoder_calls"] == 0,
                  "complete 48-anchor/96-future replay required")
        report = dict(scope=record["scope"], official_benchmark_result=False,
            metric_scale="[0,1]", boundary_fraction=e.BOUNDARY_FRACTION, void_policy="official_background",
            aggregation="equal two horizons per anchor, 12 anchors per parent, four parents",
            endpoint_policy="all 96 declared anchor futures, including four marked final frames",
            unique_future_frames=len({(r["sequence"], r["frame"]) for r in record["rows"]}),
            parents=parents, rows=record["rows"],
            **{k: e._means([r[k] for r in parents]) for k in ("teacher", "baseline", "delta")})
        c.save(out / "evaluation.json", report); output(out / "evaluation.json")
        c.finish(runtime, record)
        for relative in cache.verified:
            descriptor(dict(cache.files[relative], path=str(cache_path / relative)))
        for filename, expected in fingerprints.items():
            c.require(sha(filename) == expected, "source/input changed during replay")
        record.update(status="completed", completed_anchors=48, completed_parents=4,
                      input_sha256=fingerprints, metric_reference_commit=e.REFERENCE)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if runtime:
            if original is not None:
                runtime["model"]._track_step = original
            if encoder_hook is not None:
                encoder_hook.remove()
            record["future_forward_calls"] = runtime.get("future_forward_calls", 0)
            record["zero_parity_details"] = runtime.get("zero_parity_details", [])
        checkpoint()


if __name__ == "__main__":
    main(sys.argv[1])
