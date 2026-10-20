"""Privileged TRAIN full-GT-mask/presence write replay; no native imports on import."""
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


def full_gt_future(runtime, ex, future, require):
    """Single cached BF16 write replacement; same native call/loss as train.future_loss."""
    torch, model = runtime["torch"], runtime["model"]
    original = future["history"]
    history = dict(cond_frame_outputs=original["cond_frame_outputs"],
                   non_cond_frame_outputs=dict(original["non_cond_frame_outputs"]))
    history["non_cond_frame_outputs"][ex["anchor"]] = dict(
        history["non_cond_frame_outputs"][ex["anchor"]], maskmem_features=ex["gt_memory"])
    capture = {}
    hook = model.sam_mask_decoder.register_forward_hook(
        lambda m, a, values: capture.update(values=values))
    runtime["future_forward_calls"] = runtime.get("future_forward_calls", 0) + 1
    try:
        result = model._track_step(future["frame"], False, future["vision"], future["pos"],
            future["sizes"], None, None, history, future["count"], False, None)
    finally:
        hook.remove()
    raw, quality, _, presence = capture["values"]
    index = int(quality[0].argmax())
    F = torch.nn.functional
    mask_loss = F.binary_cross_entropy_with_logits(raw[:, index:index+1], future["target"])
    presence_loss = F.binary_cross_entropy_with_logits(
        presence, torch.full_like(presence, float(future["presence"])))
    loss = mask_loss + presence_loss
    require(bool(torch.isfinite(loss)), "nonfinite full-GT future surrogate")
    # No zero-coefficient assertion: this is a complete cached write, not a coefficient target.
    return result[1][3].detach().clone(), float(loss)


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
    spec, out, record, began = c.start(spec_path, "full-gt-write")
    runtime, original, encoder_hook, memory_hook = None, None, None, None
    record.update(scope="privileged TRAIN full-GT-mask/presence fixed-context horizon-set J/F",
                  deployment_result=False, official_benchmark_result=False, final_test_access=False,
                  gradients=False, optimization_steps=0, image_encoder_calls=0, memory_encoder_calls=0,
                  track_step_calls=0, outputs=[], history_hashes=[], anchor_diagnostics=[],
                  bindings=spec["bindings"], projection_applied=False,
                  confounds=["cached native hard GT-mask encoding", "cached GT presence embedding",
                             "unbounded full replacement; rho only defines diagnostic reference radius"])
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
        c.require(tuple(spec["train_sequences"]) == TRAIN and spec["seed"] == 0
                  and spec["rho"] == .15, "fixed TRAIN scope/reference radius")
        b = {k: descriptor(d) for k, d in spec["bindings"].items()}
        c.require(set(b) == {"mapping", "data_index", "cache_receipt"}, "bindings")
        mapping = json.loads(b["mapping"].read_text())
        index = json.loads(b["data_index"].read_text())
        cache_path = Path(spec["cache"])
        receipt_path = cache_path / "receipt.json"
        c.require(b["cache_receipt"] == receipt_path
                  and sha(receipt_path) == spec["cache_receipt_sha256"], "cache receipt identity")
        saved_cache = json.loads(receipt_path.read_text())
        c.require(mapping["status"] == saved_cache["status"] == "completed"
                  and saved_cache["completed_frames"] == mapping["member_count"] == 386
                  and saved_cache["completed_sequences"] == 6
                  and saved_cache["gt_memory_calls"] == 48
                  and saved_cache["data_index"]["sha256"] == sha(b["data_index"]), "complete inputs")
        cache_rows = {r["sequence"]: r for r in saved_cache["rows"]}
        source_rows = {r["sequence"]: r for r in index["sequences"]}
        labels = {(r["sequence"], r["frame"]): r for r in mapping["mappings"]}
        c.require(len(labels) == len(mapping["mappings"]) == 386, "unique complete label mapping")
        identifiers, selected, anchor_labels = [], {}, {}
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
                current_label = labels[sequence, anchor]
                c.require(current_label["role"] == "TRAIN" and all(
                    current_label[k] == source["labels"][anchor][k] for k in ("path", "sha256", "bytes")),
                    "current TRAIN annotation mapping")
                descriptor(current_label)
                anchor_labels[sequence, anchor] = current_label
                for horizon in (1, 3):
                    frame = anchor + horizon
                    label = labels[sequence, frame]
                    c.require(label["role"] == "TRAIN" and all(
                        label[k] == source["labels"][frame][k] for k in ("path", "sha256", "bytes")),
                        "future TRAIN annotation mapping")
                    pred = dict(row["predictions"][frame])
                    pred["path"] = str(cache_path / pred["path"])
                    descriptor(label); descriptor(pred)
                    selected[sequence, anchor, horizon] = (label, pred)
        c.require(len(identifiers) == 48 and len(selected) == 96, "fixed complete coverage")
        # Every encoded annotation/prediction descriptor is checked before any PNG/NPZ decode.
        runtime = c.load_runtime(spec, record)
        torch, np, model = runtime["torch"], runtime["np"], runtime["model"]
        train = c.module("full_gt_train", campaign / "train.py")
        e = c.module("full_gt_metrics", spec["evaluator"])
        family, head_module = train.modules(spec)
        cache = train.Cache(cache_path, spec["cache_receipt_sha256"], torch)
        state_hash = lambda value: runtime["nf"].state_hash(value, torch)
        original = model._track_step

        def track(*args, **kwargs):
            record["track_step_calls"] += 1
            return original(*args, **kwargs)

        def count_encoder(*unused):
            record["image_encoder_calls"] += 1

        def count_memory(*unused):
            record["memory_encoder_calls"] += 1

        model._track_step = track
        encoder_hook = model.image_encoder.register_forward_hook(count_encoder)
        memory_hook = model.memory_encoder.register_forward_hook(count_memory)
        parents = []
        with torch.no_grad():
            for sequence in TRAIN:
                meta = cache.meta(sequence)
                history = cache.history(sequence)
                before = state_hash(history)
                anchor_reports = []
                for name, anchor in identifiers:
                    if name != sequence:
                        continue
                    ex = train.example(spec, cache, runtime, family, head_module, sequence, anchor)
                    gt_memory = ex["gt_memory"]
                    c.require(gt_memory.shape == ex["memory"].shape
                              and tuple(gt_memory.shape) == (1, 64, 64, 64)
                              and gt_memory.dtype == torch.bfloat16
                              and bool(torch.isfinite(gt_memory).all()), "cached full GT memory")
                    gt_before = state_hash(gt_memory)
                    delta = gt_memory.float() - ex["memory"]
                    delta_energy = float(delta.double().flatten(2).square().mean(2).sum())
                    delta_rms = math.sqrt(delta_energy / 64)
                    radius = ex["context"]["radius"]
                    current_label_d = anchor_labels[sequence, anchor]
                    current_truth = e._labels(Path(current_label_d["path"]))
                    c.require(tuple(current_truth.shape) == tuple(meta["shape"]), "current GT dimensions")
                    gt_present = bool(np.any(current_truth == meta["target_id"]))
                    native_logit = ex["current"]["out"]["object_score_logits"]
                    c.require(native_logit.numel() == 1 and bool(torch.isfinite(native_logit).all()),
                              "native current presence logit")
                    native_present = bool(native_logit.item() > 0)
                    diagnostics = dict(sequence=sequence, anchor=anchor,
                        gt_memory_sha256=gt_before, original_memory_sha256=state_hash(ex["memory"]),
                        full_delta_rms=delta_rms, full_delta_sum_channel_mean_square=delta_energy,
                        reference_rho=spec["rho"], reference_family_radius=radius,
                        reference_family_rms_radius=radius / math.sqrt(64),
                        full_delta_to_family_radius_ratio=math.sqrt(delta_energy)/radius if radius > 0 else None,
                        radius_ratio_defined=radius > 0,
                        radius_ratio_undefined_reason=None if radius > 0 else "zero_family_radius",
                        norm_scope="stored BF16 tensors lifted to float32; compare to original pre-BF16 family bound",
                        changed_bf16_entries=int((gt_memory != ex["memory"].to(torch.bfloat16)).sum()),
                        gt_present=gt_present, native_present=native_present,
                        native_presence_logit=float(native_logit.item()),
                        gt_native_presence_agreement=gt_present == native_present,
                        current_annotation=current_label_d)
                    record["anchor_diagnostics"].append(diagnostics)
                    rows, losses = [], []
                    for horizon, future in zip((1, 3), ex["futures"]):
                        tick = time.perf_counter()
                        record["pending_call"] = dict(sequence=sequence, anchor=anchor, horizon=horizon,
                                                      frame=future["frame"])
                        checkpoint()
                        low, loss = full_gt_future(runtime, ex, future, c.require)
                        c.require(not low.requires_grad and bool(torch.isfinite(low).all())
                                  and tuple(low.shape) == (1, 1, 256, 256), "native selected output")
                        losses.append(loss)
                        logits = torch.nn.functional.interpolate(
                            low, size=meta["shape"], mode="bilinear", align_corners=False)
                        mask = (logits[0, 0] > 0).cpu().numpy()
                        path = out / f"{sequence}-anchor-{anchor:05d}-h{horizon}.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, masks=mask[None],
                                                object_ids=np.array([meta["target_id"]], dtype=np.int64))
                        pred_out = output(path, sequence=sequence, anchor=anchor, horizon=horizon)
                        label_d, base_d = selected[sequence, anchor, horizon]
                        truth = e._labels(Path(label_d["path"]))
                        c.require(tuple(truth.shape) == tuple(meta["shape"]), "original future GT dimensions")
                        baseline = e._prediction(Path(base_d["path"]), [meta["target_id"]],
                                                 tuple(meta["shape"]))[0]
                        j, f = e.jf(mask, truth == meta["target_id"])
                        bj, bf = e.jf(baseline, truth == meta["target_id"])
                        row = dict(sequence=sequence, parent_id=meta["parent_id"], anchor=anchor,
                            horizon=horizon, frame=future["frame"], target_id=meta["target_id"],
                            final_frame=future["frame"] == meta["frames"] - 1,
                            full_gt_write=dict(J=j, F=f, JF=(j+f)/2),
                            baseline=dict(J=bj, F=bf, JF=(bj+bf)/2),
                            delta=dict(J=j-bj, F=f-bf, JF=(j+f-bj-bf)/2),
                            surrogate_loss=loss, prediction=pred_out,
                            baseline_prediction=base_d, annotation=label_d,
                            wall_seconds=time.perf_counter()-tick)
                        rows.append(row); record["rows"].append(row)
                        record["pending_call"] = None
                        checkpoint()
                    c.require(state_hash(gt_memory) == gt_before, "cached GT memory mutated")
                    anchor_reports.append(dict(sequence=sequence, anchor=anchor, diagnostics=diagnostics,
                        **{k: e._means([r[k] for r in rows]) for k in ("full_gt_write", "baseline", "delta")},
                        mean_surrogate_loss=math.fsum(losses)/2))
                after = state_hash(history)
                record["history_hashes"].append(dict(sequence=sequence, before=before, after=after))
                c.require(after == before, "native cached history/positions/pointers/current outputs mutated")
                parents.append(dict(parent_id=sequence, anchors=anchor_reports,
                    **{k: e._means([r[k] for r in anchor_reports])
                       for k in ("full_gt_write", "baseline", "delta")}))
                del ex, history, gt_memory, delta
                cache.histories.clear(); cache.shared.clear()
        c.require(len(record["rows"]) == record["track_step_calls"] ==
                  runtime.get("future_forward_calls") == 96
                  and len(record["anchor_diagnostics"]) == 48
                  and record["image_encoder_calls"] == record["memory_encoder_calls"] == 0
                  and sum(r["final_frame"] for r in record["rows"]) == 4, "complete fixed replay required")
        report = dict(scope=record["scope"], official_benchmark_result=False, deployment_result=False,
            metric_scale="[0,1]", boundary_fraction=e.BOUNDARY_FRACTION, void_policy="official_background",
            confounds=record["confounds"],
            aggregation="equal two horizons per anchor, 12 anchors per parent, four parents",
            endpoint_policy="all 96 declared anchor futures, including four marked final frames",
            unique_future_frames=len({(r["sequence"], r["frame"]) for r in record["rows"]}),
            parents=parents, rows=record["rows"], anchor_diagnostics=record["anchor_diagnostics"],
            **{k: e._means([r[k] for r in parents]) for k in ("full_gt_write", "baseline", "delta")})
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
            if memory_hook is not None:
                memory_hook.remove()
            record["future_forward_calls"] = runtime.get("future_forward_calls", 0)
        checkpoint()


if __name__ == "__main__":
    main(sys.argv[1])
