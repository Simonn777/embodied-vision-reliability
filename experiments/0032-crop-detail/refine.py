"""One fixed current-frame crop baseline. Import is stdlib-only; no GT decoder."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

COHORT = (("dog-agility", 25, "TRAIN"), ("planes-water", 38, "TRAIN"),
          ("dogs-scale", 83, "TRAIN"), ("dog-gooses", 86, "TRAIN"),
          ("cat-girl", 89, "DEV"), ("upside-down", 65, "DEV"))
BINDINGS = ("cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec",
            "mapping", "data_index")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(loader)
    sys.modules[name] = result
    loader.loader.exec_module(result)
    return result


def anchors_for(n):
    return [1 + i * (n - 5) // 11 for i in range(12)]


def frame_plan(n, role):
    require(role in ("TRAIN", "DEV"), "unknown role")
    return sorted({a + h for a in anchors_for(n) for h in (1, 3)}) \
        if role == "TRAIN" else list(range(n))


def crop_geometry(mask, cv2):
    """Inclusive foreground extrema, ceil 10% padding, exterior-aware EDT point."""
    import numpy as np
    require(mask.ndim == 2 and mask.dtype == np.bool_, "boolean original mask required")
    if not mask.any():
        return None
    y, x = np.nonzero(mask)
    h, w = mask.shape
    px = math.ceil(0.10 * (int(x.max()) - int(x.min()) + 1))
    py = math.ceil(0.10 * (int(y.max()) - int(y.min()) + 1))
    left, top = max(0, int(x.min()) - px), max(0, int(y.min()) - py)
    right, bottom = min(w, int(x.max()) + 1 + px), min(h, int(y.max()) + 1 + py)
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1),
                                    cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    iy, ix = np.unravel_index(int(np.argmax(distance)), mask.shape)
    require(bool(mask[iy, ix]) and float(distance[iy, ix]) > 0, "EDT point outside foreground")
    return dict(bounds_xyxy=[left, top, right, bottom],
                point_original_xy=[int(ix), int(iy)],
                point_crop_xy=[int(ix) - left, int(iy) - top],
                box_crop_xyxy=[0, 0, right - left - 1, bottom - top - 1],
                edt_distance=float(distance[iy, ix]), padding_xy=[px, py])


def stitch(native, crop_mask, bounds):
    import numpy as np
    left, top, right, bottom = bounds
    require(native.ndim == 2 and native.dtype == crop_mask.dtype == np.bool_
            and 0 <= left < right <= native.shape[1] and 0 <= top < bottom <= native.shape[0]
            and crop_mask.shape == (bottom - top, right - left), "crop inverse dimensions")
    answer = native.copy()
    answer[top:bottom, left:right] = crop_mask
    return answer


def select(piou):
    import numpy as np
    require(piou.shape == (3,) and np.isfinite(piou).all(), "three finite predicted IoUs")
    return int(np.argmax(piou))  # Exact ties select the first native candidate.


def metadata_plans(cache, index, mapping, cache_root):
    """Only metadata: no label payload is opened, hashed, or decoded here."""
    require(cache["schema"] == "evr0030.development.baseline-cache.v1"
            and cache["completed_frames"] == 386 and cache["completed_sequences"] == 6
            and not cache["offline_attempts"]
            and cache["model_initial_hash"] == cache["model_final_hash"]
            and mapping["status"] == "completed" and mapping["member_count"] == 386,
            "complete native cache and mapping required")
    bases = {r["sequence"]: r for r in cache["rows"]}
    inputs = {r["sequence"]: r for r in index["sequences"]}
    require(len(cache["rows"]) == len(index["sequences"]) == len(bases) == len(inputs) == 6
            and set(bases) == set(inputs) == {r[0] for r in COHORT}, "exact six parents")
    inventory = {d["path"]: d for d in cache["outputs"]}
    require(len(inventory) == len(cache["outputs"]), "duplicate cache outputs")

    def cached(relative):
        require(relative in inventory, "cache file absent from sealed inventory")
        d = inventory[relative]
        path = Path(d["path"])
        require(not path.is_absolute() and ".." not in path.parts, "unsafe cache relative path")
        return dict(d, path=str(cache_root / path))

    plans = []
    for name, n, role in COHORT:
        base, source = bases[name], inputs[name]
        require(base["status"] == "completed" and base["completed_frames"] == n
                and base["frames"] == n and base["role"] == source["role"] == role
                and base["parent_id"] == source["parent_id"] == name
                and base["anchors"] == (anchors_for(n) if role == "TRAIN" else [])
                and [d["frame"] for d in source["rgb"]] == list(range(n))
                and [d["frame"] for d in source["labels"]] == list(range(n))
                and [d["frame"] for d in base["predictions"]] == list(range(n)),
                "cohort/frame/anchor metadata changed")
        labels = [d for d in mapping["mappings"] if d["sequence"] == name]
        require([d["frame"] for d in labels] == list(range(n))
                and all(d["role"] == role and all(d[k] == source["labels"][i][k]
                        for k in ("path", "bytes", "sha256")) for i, d in enumerate(labels)),
                "complete mapping identity")
        frames = []
        for frame in frame_plan(n, role):
            pred = base["predictions"][frame]
            require(pred == inventory[pred["path"]], "native prediction inventory mismatch")
            frames.append(dict(frame=frame, rgb=source["rgb"][frame],
                native_prediction=cached(pred["path"]),
                tensor=cached(f"{name}/frame-{frame:05d}.pt")))
        plans.append(dict(sequence=name, parent_id=name, role=role, frames=n,
            shape=base["shape"], target_id=base["target_id"], anchors=base["anchors"],
            inputs=frames))
    require(sum(len(r["inputs"]) for r in plans[:4]) == 87
            and sum(len(r["inputs"]) for r in plans[4:]) == 154, "fixed 241-frame plan")
    return plans


def run(spec_path):
    began = time.perf_counter()
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="evr0032.crop-detail.v1", status="started",
        spec_sha256=sha(spec_path), source_sha256=spec["source_sha256"],
        bindings=spec["bindings"], rows=[], outputs=[], diagnostics=[], warnings=[],
        current_label_decodes=0, final_test_access=False, scientific_gain_claim=False,
        completed_frames=0, train_unique_frames=87, dev_frames=154,
        image_encoder_calls=0, mask_decoder_calls=0, native_parity_checks=0,
        noop_initial=0, noop_empty=0, eligible_frames=0,
        observation="current RGB and cached causal native mask/logits; output only",
        object_presence_policy="native image API: record object score, do not hard-gate output",
        cost_scope="incremental refinement; original full-video baseline cost inherited, not free")
    fingerprints = dict(spec["source_sha256"])
    fingerprints[str(spec_path)] = record["spec_sha256"]
    runtime, hooks, original_track = None, [], None
    counts = dict(encoder=0, decoder=0)
    decoder_presence = []

    def checkpoint():
        record["wall_seconds"] = time.perf_counter() - began
        save(out / "receipt.json", record)

    def descriptor(d):
        path = Path(d["path"])
        require(path.is_file() and path.stat().st_size == d["bytes"] and sha(path) == d["sha256"],
                "input file changed: " + str(path))
        fingerprints[str(path)] = d["sha256"]
        return path

    def write_npz(path, kind, **arrays):
        with path.open("xb") as stream:
            runtime["np"].savez_compressed(stream, **arrays)
        return dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
                    sha256=sha(path), kind=kind)

    checkpoint()
    try:
        common_path = Path(spec["campaign_root"]) / "common.py"
        evaluation = Path(spec["evaluation_root"])
        required_sources = [Path(__file__).resolve(), Path(__file__).with_name("SCOPE.md").resolve(),
            Path(__file__).with_name("ADMISSION.md").resolve(), common_path] + [evaluation / n for n in
                           ("evaluate_rollout.py", "evaluate_baseline.py", "evaluate_jf.py")]
        require(all(spec["source_sha256"].get(str(p)) == sha(p) for p in required_sources),
                "required worker/helper source bindings")
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/spec changed")
        require(set(spec["bindings"]) == set(BINDINGS) and spec.get("seed", 0) == 0,
                "fixed input bindings and seed")
        binding = {k: descriptor(d) for k, d in spec["bindings"].items()}
        common = module("crop_common", common_path)
        # This verifies the qualified runtime, installs its guard, and loads one frozen model.
        runtime = common.load_runtime(spec, record)
        sys.path.insert(0, str(evaluation))
        admission = module("crop_admission", evaluation / "evaluate_rollout.py")
        cache, _ = admission.admit(binding, "cache", fingerprints, legacy=True)
        require(cache["data_index"]["sha256"] == sha(binding["data_index"])
                and cache["model_final_hash"] == record["model_initial_hash"], "cache/runtime binding")
        plans = metadata_plans(cache, json.loads(binding["data_index"].read_text()),
                               json.loads(binding["mapping"].read_text()),
                               binding["cache_receipt"].parent)
        for plan in plans:
            for frame in plan["inputs"]:
                for key in ("rgb", "native_prediction", "tensor"):
                    descriptor(frame[key])
        record["baseline_history_cost"] = dict(
            scope="entire previously completed six-video 386-frame native cache, including cache/GT-memory work",
            receipt=spec["bindings"]["cache_receipt"], outer=spec["bindings"]["cache_outer"],
            wall_seconds=json.loads(binding["cache_outer"].read_text())["wall_seconds"],
            not_a_matched_complete_pipeline_time=True)
        model, torch, np, cv2, Image = (runtime[k] for k in ("model", "torch", "np", "cv2", "Image"))
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(model, mask_threshold=0, max_hole_area=0, max_sprinkle_area=0)
        F = torch.nn.functional

        def encode_hook(_module, _args):
            counts["encoder"] += 1

        def decode_hook(_module, _args, answer):
            counts["decoder"] += 1
            require(len(answer) == 4 and bool(torch.isfinite(answer[3]).all()),
                    "finite native image object-presence output")
            decoder_presence.append(answer[3].detach().float().cpu().reshape(-1).tolist())

        def forbidden(*_args, **_kwargs):
            raise RuntimeError("crop baseline must not call video tracking or memory encoding")

        hooks = [model.image_encoder.register_forward_pre_hook(encode_hook),
                 model.sam_mask_decoder.register_forward_hook(decode_hook),
                 model.memory_encoder.register_forward_pre_hook(forbidden)]
        original_track = model._track_step
        model._track_step = forbidden
        with torch.inference_mode(), runtime["guard"].capture_warnings(record):
            for plan in plans:
                sequence_started = time.perf_counter()
                name = plan["sequence"]
                dest = out / name
                dest.mkdir()
                row = {k: plan[k] for k in
                       ("sequence", "parent_id", "role", "frames", "shape", "target_id", "anchors")}
                row.update(status="started", predictions=[], frame_records=[])
                record["rows"].append(row)
                for item in plan["inputs"]:
                    started = time.perf_counter()
                    frame = item["frame"]
                    detail = dict(frame=frame, status="started", inputs=item,
                                  image_encoder_calls=0, mask_decoder_calls=0)
                    row["frame_records"].append(detail)
                    checkpoint()
                    before = dict(counts)
                    native = admission.e._prediction(Path(item["native_prediction"]["path"]),
                                [plan["target_id"]], tuple(plan["shape"]))[0]
                    packed = torch.load(item["tensor"]["path"], weights_only=True, map_location="cpu")
                    low = packed["out"]["pred_masks"]
                    require(low.dtype == torch.float32 and tuple(low.shape) == (1, 1, 256, 256)
                            and bool(torch.isfinite(low).all()), "cached finite native selected logits")
                    full = F.interpolate(low, size=tuple(plan["shape"]),
                                         mode="bilinear", align_corners=False)[0, 0]
                    require(np.array_equal((full > 0).numpy(), native), "native zero-logit mask parity")
                    old_presence = packed["out"]["object_score_logits"]
                    require(bool(torch.isfinite(old_presence).all()), "native object presence finite")
                    detail["native_object_score_logits"] = old_presence.float().reshape(-1).tolist()
                    del packed, low, old_presence
                    record["native_parity_checks"] += 1
                    detail["native_parity"] = True
                    if plan["role"] == "DEV" and frame == 0:
                        answer = native.copy()
                        detail.update(operation="native_initial_noop", selected_index=None)
                        record["noop_initial"] += 1
                    elif not native.any():
                        answer = native.copy()
                        detail.update(operation="native_empty_noop", selected_index=None)
                        record["noop_empty"] += 1
                    else:
                        geometry = crop_geometry(native, cv2)
                        left, top, right, bottom = geometry["bounds_xyxy"]
                        detail.update(operation="crop_refinement", crop=geometry, phase="RGB_decode")
                        checkpoint()
                        with Image.open(item["rgb"]["path"]) as image:
                            require(image.size == (plan["shape"][1], plan["shape"][0]),
                                    "original RGB size differs from native prediction")
                            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                        crop = np.ascontiguousarray(rgb[top:bottom, left:right])
                        prompt = F.interpolate(full[top:bottom, left:right][None, None],
                            size=(256, 256), mode="bilinear", align_corners=False,
                            antialias=True).clamp(-32, 32)[0].numpy()
                        require(np.isfinite(prompt).all(), "finite dense prompt")
                        detail["dense_prompt"] = dict(shape=list(prompt.shape),
                            sha256=hashlib.sha256(prompt.tobytes()).hexdigest(),
                            min=float(prompt.min()), max=float(prompt.max()),
                            resize="float32 bilinear align_corners=False antialias=True; clamp [-32,32]")
                        detail["phase"] = "image_encoder"
                        checkpoint()
                        call_started = time.perf_counter()
                        predictor.set_image(crop)
                        detail["image_encoder_seconds"] = time.perf_counter() - call_started
                        detail["phase"] = "mask_decoder"
                        checkpoint()
                        call_started = time.perf_counter()
                        logits, piou, low_candidates = predictor.predict(
                            point_coords=np.asarray([geometry["point_crop_xy"]], dtype=np.float32),
                            point_labels=np.ones(1, dtype=np.int32),
                            box=np.asarray(geometry["box_crop_xyxy"], dtype=np.float32),
                            mask_input=prompt, multimask_output=True, return_logits=True,
                            normalize_coords=True)
                        detail["mask_decoder_seconds"] = time.perf_counter() - call_started
                        require(logits.shape == (3, bottom - top, right - left)
                                and low_candidates.shape == (3, 256, 256)
                                and np.isfinite(logits).all() and np.isfinite(low_candidates).all(),
                                "three finite native crop candidates")
                        selected = select(piou)
                        crop_masks = logits > 0
                        answer = stitch(native, crop_masks[selected], geometry["bounds_xyxy"])
                        diagnostic = write_npz(dest / f"candidates-{frame:05d}.npz", "crop-candidates",
                            crop_masks=crop_masks, low_logits=low_candidates, predicted_iou=piou)
                        diagnostic.update(sequence=name, frame=frame)
                        record["diagnostics"].append(diagnostic)
                        detail.update(selected_index=selected, predicted_iou=piou.tolist(),
                            image_object_score_logits=decoder_presence[-1],
                            candidate_foreground_pixels=[int(m.sum()) for m in crop_masks],
                            diagnostic=diagnostic, output_empty=not bool(answer.any()))
                        record["eligible_frames"] += 1
                        predictor.reset_predictor()
                        del rgb, crop, logits, low_candidates, prompt, crop_masks
                    detail["image_encoder_calls"] = counts["encoder"] - before["encoder"]
                    detail["mask_decoder_calls"] = counts["decoder"] - before["decoder"]
                    expected = int(detail["operation"] == "crop_refinement")
                    require(detail["image_encoder_calls"] == detail["mask_decoder_calls"] == expected,
                            "one encoder and one decoder per eligible frame, none for no-op")
                    prediction = write_npz(dest / f"prediction-{frame:05d}.npz", "prediction",
                        masks=answer[None], object_ids=np.asarray([plan["target_id"]], dtype=np.int64))
                    prediction.update(sequence=name, frame=frame)
                    row["predictions"].append(prediction)
                    record["outputs"].append(prediction)
                    detail.update(status="completed", phase="saved",
                        native_foreground_pixels=int(native.sum()), output_foreground_pixels=int(answer.sum()),
                        changed_pixels=int(np.count_nonzero(native != answer)),
                        wall_seconds=time.perf_counter() - started)
                    record["completed_frames"] += 1
                    record["image_encoder_calls"], record["mask_decoder_calls"] = counts["encoder"], counts["decoder"]
                    checkpoint()
                    del native, full, answer
                row.update(status="completed", wall_seconds=time.perf_counter() - sequence_started)
                checkpoint()
        common.finish(runtime, record)
        require(record["completed_frames"] == record["native_parity_checks"] == 241
                and record["noop_initial"] == 2
                and record["eligible_frames"] + record["noop_empty"] + 2 == 241
                and counts["encoder"] == counts["decoder"] == record["eligible_frames"]
                and record["outputs"] == [p for r in record["rows"] for p in r["predictions"]],
                "complete fixed cohort and call accounting")
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/input changed during refinement")
        record.update(status="completed", input_sha256=fingerprints,
                      metric_stage="not executed; all current GT decoding deferred to separate stage")
        checkpoint()
        save(out / "seal.json", dict(schema="evr0032.crop-detail-seal.v1", complete=True,
            receipt_sha256=sha(out / "receipt.json"), outputs=record["outputs"],
            diagnostics=record["diagnostics"], source_sha256=record["source_sha256"],
            spec_sha256=record["spec_sha256"], frame_count=241))
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      image_encoder_calls=counts["encoder"], mask_decoder_calls=counts["decoder"])
        checkpoint()
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if runtime is not None and original_track is not None:
            runtime["model"]._track_step = original_track


if __name__ == "__main__":
    run(sys.argv[1])
