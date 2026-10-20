"""One fixed complete video, observation-only SAM2 branches; no work on import.

run(inputs, output, record) assumes the caller has admitted all source, runtime,
checkpoint, JPEG-directory and witness bindings and installed the offline guard.
Inputs are qualify_helper, native_fixture, diagnostic_core, sam2_source, runtime,
sam2_checkpoint, source_sha256, and video. The video has sequence, frame_count,
frames_dir, initial_mask:{path,sha256,bytes}, witnesses:[{frame,path,sha256,bytes}].
Only the initial annotation is decoded. No query annotation path is accepted.

All branches use deep copies of the complete native state. Full state hashes
include the image tensor. These deliberately expensive checks are charged and
do not constitute an efficiency claim. Only original baseline propagation
advances history; a transparent observer captures its existing decoder bank.
"""
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_file(item):
    path = Path(item["path"])
    require(path.is_file() and not path.is_symlink()
            and type(item["bytes"]) is int and path.stat().st_size == item["bytes"]
            and sha(path) == item["sha256"], "admitted input changed")
    return path


def module_origins(inputs):
    runtime = Path(inputs["runtime"]).resolve(strict=True)
    checked = {}
    for name, root_key, distribution in (
        ("sam2", "sam2_source", None), ("numpy", "runtime", "numpy"),
        ("torch", "runtime", "torch"), ("cv2", "runtime", "opencv-python"),
        ("PIL", "runtime", "Pillow"), ("hydra", "runtime", "hydra-core"),
        ("omegaconf", "runtime", "omegaconf"), ("iopath", "runtime", "iopath"),
    ):
        loaded = importlib.import_module(name)
        root = Path(inputs[root_key]).resolve(strict=True)
        filename = getattr(loaded, "__file__", None)
        paths = [Path(filename).resolve(strict=True)] if filename else [
            Path(p).resolve(strict=True) for p in getattr(loaded, "__path__", ())]
        require(paths and all(p.is_relative_to(root) for p in paths), "wrong import origin: " + name)
        item = dict(root_key=root_key, relative_origins=[str(p.relative_to(root)) for p in paths])
        if filename:
            item["module_sha256"] = sha(paths[0])
        if distribution:
            dist = importlib.metadata.distribution(distribution)
            require(Path(dist.locate_file("")).resolve().is_relative_to(runtime),
                    "wrong distribution metadata: " + name)
            item.update(distribution=distribution, version=dist.version)
        checked[name] = item
    return checked


def carrier_hook(model, outputs, frame, num_frames, traces, torch, tensor_hash):
    """Verify actual native attention tokens, without supplying attention inputs.

    The pinned case is forward, one immutable cond0, seven spatial slots and
    stride one. Pointer positions retain native num_frames normalization. The
    position projection is evaluated for the ACTUAL pointer batch, including
    reset's singleton batch; cross-reset bit identity is not assumed.
    """
    from sam2.modeling.sam2_utils import get_1d_sine_pe
    cond, noncond = outputs["cond_frame_outputs"], outputs["non_cond_frame_outputs"]
    require(set(cond) == {0} and frame > 0
            and all(type(k) is int and 0 < k <= frame for k in noncond),
            "unexpected conditioning/future history")
    spatial = [(0, 0)]
    for position in range(1, 7):
        previous = frame - (7 - position)
        if previous in noncond:
            spatial.append((previous, position))
    pointer_limit = min(num_frames, model.max_obj_ptrs_in_encoder)
    pointers = [0] + [frame - age for age in range(1, pointer_limit)
                      if frame - age >= 0 and frame - age in noncond]

    def hook(_module, args, kwargs):
        require(not args, "unexpected attention positional arguments")
        memory, position = kwargs["memory"], kwargs["memory_pos"]
        cursor, blocks = 0, []

        def block(kind, source, values, pos):
            nonlocal cursor
            values, pos = values.to(memory), pos.to(position)
            end = cursor + values.shape[0]
            require(torch.equal(memory[cursor:end], values), "carrier feature mismatch")
            require(torch.equal(position[cursor:end], pos), "carrier position mismatch")
            blocks.append(dict(kind=kind, frame=source, start=cursor, end=end,
                               values_sha256=tensor_hash(memory[cursor:end], torch),
                               position_sha256=tensor_hash(position[cursor:end], torch)))
            cursor = end

        for source, temporal_position in spatial:
            item = cond[0] if source == 0 else noncond[source]
            values = item["maskmem_features"].flatten(2).permute(2, 0, 1)
            pos = item["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
            pos = pos + model.maskmem_tpos_enc[6 - temporal_position]
            block("spatial", source, values, pos)
        ptrs = torch.stack([(cond[0] if source == 0 else noncond[source])["obj_ptr"]
                            for source in pointers], dim=0)
        distances = torch.tensor([frame - source for source in pointers], device=ptrs.device)
        dim = model.hidden_dim if model.proj_tpos_enc_in_obj_ptrs else model.mem_dim
        pos = get_1d_sine_pe(distances / (pointer_limit - 1), dim=dim)
        pos = model.obj_ptr_tpos_proj(pos).unsqueeze(1).expand(-1, 1, model.mem_dim)
        tokens = model.hidden_dim // model.mem_dim
        require(model.hidden_dim % model.mem_dim == 0, "unsupported pointer dimensions")
        ptrs = ptrs.reshape(-1, 1, tokens, model.mem_dim).permute(0, 2, 1, 3)
        for index, source in enumerate(pointers):
            block("pointer", source, ptrs[index], pos[index:index + 1].expand(tokens, -1, -1))
        require(cursor == memory.shape[0] and memory.shape == position.shape
                and kwargs["num_obj_ptr_tokens"] == len(pointers) * tokens,
                "unaccounted attention carriers")
        traces.append(dict(frame=frame, num_frames=num_frames,
                           spatial_frames=[source for source, _ in spatial],
                           pointer_frames=pointers, pointer_limit=pointer_limit,
                           pointer_distance_denominator=pointer_limit - 1,
                           num_obj_ptr_tokens=kwargs["num_obj_ptr_tokens"],
                           total_tokens=cursor, blocks=blocks))
    return hook


def compare_deleted_trace(full, branch, removed):
    old = {(x["kind"], x["frame"]): x for x in full["blocks"]}
    new = {(x["kind"], x["frame"]): x for x in branch["blocks"]}
    require(set(new) == {key for key in old if key[1] != removed},
            "deleted frame not removed from both carriers, or backfill occurred")
    for key, item in new.items():
        require(all(item[k] == old[key][k] for k in ("values_sha256", "position_sha256")),
                "retained deletion-branch carrier changed")
    return dict(removed_frame=removed, consumed_in_full=any(k[1] == removed for k in old),
                removed_kinds=[kind for kind in ("spatial", "pointer") if (kind, removed) in old],
                retained_carriers_exact=True, no_backfill=True)


def run(inputs, output, record):
    """No labels beyond the supplied initial mask; all eight rows stay present."""
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="evr0029.native-video.v1", status="started", complete=False,
                  source_sha256=inputs["source_sha256"], warnings=[], timings=[], loads={},
                  queries=[], baseline=[], outputs=[], noninitial_annotation_access=False,
                  calls=dict(image_encoder_intent=0, image_encoder_completed=0,
                             track_step_intent=0, track_step_completed=0,
                             baseline_frames=0, verification_intent=0, verification_completed=0,
                             deletion_intent=0, deletion_completed=0,
                             prompt_intent=0, prompt_completed=0, native_bank_extra_calls=0))
    record["native_video"] = result
    began = time.perf_counter()
    handles, original_track_step, model = [], None, None

    def save():
        result["wall_seconds"] = time.perf_counter() - began
        pending = output / "video-record.pending"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(pending, output / "video-record.json")

    @contextlib.contextmanager
    def timed(stage, category, row=None):
        item = dict(stage=stage, category=category, status="started")
        result["stage"] = stage
        result["timings"].append(item)
        if row is not None:
            row["status"] = "started"
        save()
        start = time.perf_counter()
        try:
            yield
            item["status"] = "completed"
            if row is not None:
                row["status"] = "completed"
        except BaseException as exc:
            item.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            if row is not None:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            item["seconds"] = time.perf_counter() - start
            save()

    save()
    try:
        q = module("_evr0029_video_q", inputs["qualify_helper"])
        fixture = module("_evr0029_video_qualified_fixture", inputs["native_fixture"])
        with timed("imports", "setup"), q.capture_warnings(result):
            import numpy as np
            import torch
            import cv2
            from PIL import Image
            from sam2.build_sam import build_sam2_video_predictor
            core = module("_evr0029_video_arithmetic", inputs["diagnostic_core"])
            result["module_origins"] = module_origins(inputs)
        video = inputs["video"]
        require(set(video) == {"sequence", "frame_count", "frames_dir", "initial_mask", "witnesses"},
                "unexpected video input fields")
        sequence, frame_count = video["sequence"], video["frame_count"]
        require((sequence, frame_count) in core.COHORT, "video outside fixed cohort")
        query_frames = core.query_schedule(frame_count)
        require([x["frame"] for x in video["witnesses"]] == query_frames
                and all(set(x) == {"frame", "path", "sha256", "bytes"} for x in video["witnesses"])
                and set(video["initial_mask"]) == {"path", "sha256", "bytes"},
                "fixed witness/initial-mask contract")
        result.update(sequence=sequence, frame_count=frame_count, query_frames=query_frames,
                      frames_dir=str(video["frames_dir"]),
                      initial_mask={**video["initial_mask"], "path": str(video["initial_mask"]["path"])},
                      method_names=list(core.METHODS), descriptive_controls=list(core.DESCRIPTIVE_CONTROLS))
        rows = {}
        for media in video["witnesses"]:
            row = dict(sequence=sequence, frame=media["frame"], status="not_attempted",
                       input={**media, "path": str(media["path"])}, actionable=None,
                       candidates={}, branches=[], selection=None)
            rows[media["frame"]] = row
            result["queries"].append(row)
        save()
        with timed("initial_mask_and_witnesses", "input"):
            with Image.open(bound_file(video["initial_mask"])) as image:
                require(image.format == "PNG", "initial annotation is not PNG")
                labels = np.asarray(image).copy()
            target = core.choose_target(labels)
            require(target is not None, "invalid initial target; no object replacement")
            bound_file(video["initial_mask"])
            result["object_id"] = target
            mask = labels == target
            witnesses = {}
            for media in video["witnesses"]:
                item = json.loads(bound_file(media).read_text())
                frame = media["frame"]
                require(item["sequence"] == sequence and item["frame"] == frame
                        and item["target_id"] == target and item["initial_shape"] == list(mask.shape)
                        and type(item["actionable"]) is bool
                        and item["actionable"] == (len(item["positive"]) >= 4 and len(item["negative"]) >= 4)
                        and max(len(item["positive"]), len(item["negative"])) <= 32,
                        "witness identity/availability changed")
                for name, label in (("positive", 1), ("negative", 0)):
                    require(all(point["label"] == label for point in item[name]), "witness class mismatch")
                witnesses[frame] = item
                rows[frame].update(actionable=item["actionable"], target_id=target, witnesses=item,
                                   witness_counts={name: len(item[name]) for name in ("positive", "negative")})
            del labels
        with timed("execution_configuration", "setup"):
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            cv2.setNumThreads(0)
            result["execution"] = dict(device="cpu", parameter_dtype="float32",
                torch_intraop=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads(),
                opencv_threads=cv2.getNumThreads(), opencv_parallelism_disabled=True,
                authored_compact_memory_dtype="bfloat16", autocast=False, vos_optimized=False,
                isolation="copy.deepcopy complete native state, inclusive image tensor",
                full_state_hash_includes_images=True)
            require((torch.get_num_threads(), torch.get_num_interop_threads(), cv2.getNumThreads())
                    == (4, 1, 1), "execution thread configuration")
        with timed("strict_native_loading", "setup"), q.capture_warnings(result):
            model = build_sam2_video_predictor(fixture.CONFIG, ckpt_path=None, device="cpu",
                mode="eval", apply_postprocessing=False,
                hydra_overrides_extra=list(fixture.OVERRIDES), vos_optimized=False)
            checkpoint = torch.load(inputs["sam2_checkpoint"], map_location="cpu", weights_only=True)
            result["loads"]["sam2"] = q.strict_state(model, checkpoint["model"], torch)
            del checkpoint
            model.eval().to(device="cpu", dtype=torch.float32)
            require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
                    "native parameter device/dtype")
            decoder = model.sam_mask_decoder
            require(model.fill_hole_area == 0 and model.binarize_mask_from_pts_for_mem_enc
                    and decoder.dynamic_multimask_via_stability
                    and decoder.dynamic_multimask_stability_delta == .05
                    and decoder.dynamic_multimask_stability_thresh == .98
                    and model.num_maskmem == 7 and model.memory_temporal_stride_for_eval == 1
                    and model.use_obj_ptrs_in_encoder and model.add_tpos_enc_to_obj_ptrs
                    and model.max_obj_ptrs_in_encoder >= 9
                    and not model.clear_non_cond_mem_around_input
                    and not model.add_all_frames_to_correct_as_cond,
                    "native configuration differs from qualified branch contract")
            result["config"] = dict(name=fixture.CONFIG, overrides=list(fixture.OVERRIDES),
                image_size=model.image_size, num_maskmem=7, memory_stride=1,
                max_obj_ptrs_in_encoder=model.max_obj_ptrs_in_encoder,
                checkpoint_sha256=sha(inputs["sam2_checkpoint"]),
                native_bank="capture original baseline _track_step; zero additional calls")

        active, captured = {}, []

        def encoder_enter(_module, _args):
            require(not active or active["name"] == "B", "branch attempted new image encoding")
            result["calls"]["image_encoder_intent"] += 1

        def encoder_exit(_module, _args, _output):
            result["calls"]["image_encoder_completed"] += 1

        handles.extend((model.image_encoder.register_forward_pre_hook(encoder_enter),
                        model.image_encoder.register_forward_hook(encoder_exit)))
        original_track_step = model._track_step

        def observe(frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                    feat_sizes, point_inputs, mask_inputs, output_dict, num_frames,
                    track_in_reverse, prev_sam_mask_logits):
            result["calls"]["track_step_intent"] += 1
            tracing = bool(active)
            traces, hook = [], None
            if tracing:
                require(frame_idx == active["frame"] and not is_init_cond_frame
                        and not track_in_reverse and num_frames == frame_count,
                        "observed query route changed")
                hook = model.memory_attention.register_forward_pre_hook(
                    carrier_hook(model, output_dict, frame_idx, num_frames, traces, torch,
                                 fixture.tensor_hash), with_kwargs=True)
            try:
                answer = original_track_step(frame_idx, is_init_cond_frame, current_vision_feats,
                    current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict,
                    num_frames, track_in_reverse, prev_sam_mask_logits)
            finally:
                if hook is not None:
                    hook.remove()
            result["calls"]["track_step_completed"] += 1
            if tracing:
                require(len(traces) == 1 and len(answer[1]) == 7, "query attention/SAM tuple count")
                tensors = tuple(x.detach().clone() for x in answer[1])
                require(all(torch.isfinite(x).all().item() for x in tensors), "nonfinite native SAM tuple")
                meta = dict(name=active["name"], frame=frame_idx, trace=traces[0],
                    vision_sha256=[fixture.tensor_hash(x, torch) for x in current_vision_feats],
                    vision_position_sha256=[fixture.tensor_hash(x, torch) for x in current_vision_pos_embeds],
                    feature_sizes=[list(x) for x in feat_sizes],
                    previous_mask_sha256=None if prev_sam_mask_logits is None
                        else fixture.tensor_hash(prev_sam_mask_logits, torch),
                    point_coords=None if point_inputs is None
                        else point_inputs["point_coords"].detach().cpu().tolist(),
                    point_labels=None if point_inputs is None
                        else point_inputs["point_labels"].detach().cpu().tolist())
                captured.append((tensors, meta))
            return answer

        model._track_step = observe

        def take_capture():
            require(len(captured) == 1, "exactly one captured query step required")
            return captured.pop()

        def array(logits, shape):
            value = logits.detach().cpu().float().numpy()
            require(value.ndim == 4 and value.shape[0] == 1 and tuple(value.shape[-2:]) == shape
                    and np.isfinite(value).all(), "invalid original-resolution native logits")
            return value

        def save_candidate(frame, name, logits, tensors, item, witness):
            low_all, high_all, quality, low_selected, high_selected, pointer, object_score = tensors
            require(quality.shape == low_all.shape[:2] and low_all.shape[0] == 1
                    and low_all.shape[:2] == high_all.shape[:2], "native decoder shapes")
            selected = int(torch.argmax(quality[0]).item())
            require(torch.equal(low_selected, low_all[:, selected:selected + 1])
                    and torch.equal(high_selected, high_all[:, selected:selected + 1]),
                    "native selected mask/quality mismatch")
            values = array(logits, tuple(witness["current_shape"]))
            require(values.shape[1] == 1, "candidate must be one mask")
            saved = q.save_arrays(output / ("query-%05d-%s.npz" % (frame, name)),
                                  {"logits": values, "low_logits": low_selected.detach().cpu().numpy(),
                                   "predicted_quality": quality.detach().cpu().numpy(),
                                   "object_score_logits": object_score.detach().cpu().numpy()}, np)
            result["outputs"].append(saved)
            item.update(status="completed", output=saved, array_key="logits", channel=0,
                        selected_native_index=selected, predicted_iou=float(quality[0, selected]),
                        low_sha256=fixture.tensor_hash(low_selected, torch),
                        loss=core.balanced_logloss(values[0, 0], witness) if witness["actionable"] else None)
            return values[0, 0]

        with torch.inference_mode(), q.capture_warnings(result):
            with timed("native_init_state", "baseline"):
                state = model.init_state(str(video["frames_dir"]), offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=False)
                require(state["num_frames"] == frame_count
                        and (state["video_height"], state["video_width"]) == mask.shape,
                        "complete video shape/count mismatch")
                for witness in witnesses.values():
                    require(witness["current_shape"] == list(mask.shape), "witness/video dimensions")
            with timed("native_initial_mask", "baseline"):
                answer = model.add_new_mask(state, frame_idx=0, obj_id=target, mask=mask)
                require(answer[0] == 0 and answer[1] == [target], "initial identity mismatch")
            iterator = iter(model.propagate_in_video(state))
            for frame in range(frame_count):
                timing_start = len(result["timings"])
                calls_before = dict(result["calls"])
                active.clear()
                if frame in rows:
                    active.update(name="B", frame=frame)
                with timed("baseline-%05d" % frame, "baseline"):
                    got, ids, logits = next(iterator)
                    require(got == frame and ids == [target], "baseline order/identity mismatch")
                    result["calls"]["baseline_frames"] += 1
                    result["baseline"].append(dict(frame=frame,
                        original_logits_sha256=fixture.tensor_hash(logits, torch)))
                active.clear()
                if frame not in rows:
                    continue
                row, witness = rows[frame], witnesses[frame]
                row["status"] = "started"
                baseline_tensors, baseline_meta = take_capture()
                with timed("capture-bank-%05d" % frame, "observation"):
                    row["baseline_trace"] = baseline_meta
                    low_all, high_all, quality, low_selected, high_selected, _, _ = baseline_tensors
                    require(low_all.shape[1] == 3, "qualified native three-mask bank changed")
                    require(torch.equal(state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame]
                                        ["pred_masks"], low_selected), "captured/stored baseline low mismatch")
                    _, selected_video = model._get_orig_video_res_output(state, low_selected)
                    require(torch.equal(selected_video, logits), "captured selected/baseline output mismatch")
                    row["candidates"]["B"] = {}
                    baseline_array = save_candidate(frame, "B", logits, baseline_tensors,
                                                    row["candidates"]["B"], witness)
                    _, all_video = model._get_orig_video_res_output(state, low_all)
                    values = array(all_video, tuple(witness["current_shape"]))
                    saved = q.save_arrays(output / ("query-%05d-native-bank.npz" % frame),
                        {"logits": values, "low_logits": low_all.detach().cpu().numpy(),
                         "predicted_quality": quality.detach().cpu().numpy()}, np)
                    result["outputs"].append(saved)
                    for index in range(3):
                        row["candidates"]["M%d" % index] = dict(status="completed", output=saved,
                            array_key="logits", channel=index, predicted_iou=float(quality[0, index]),
                            loss=core.balanced_logloss(values[0, index], witness)
                                if witness["actionable"] else None)
                    row["native_bank"] = dict(count=3, captured_existing_call=True,
                        extra_calls=0, selected_index=row["candidates"]["B"]["selected_native_index"],
                        selected_low_exact=True, selected_original_exact=True)
                    del all_video, values
                with timed("state-checkpoint-%05d" % frame, "verification"):
                    state_before = fixture.state_hash(state, torch)
                    row["baseline_state_sha256_before"] = state_before
                    require(frame in state["cached_features"], "current baseline features missing")
                    baseline_cache_hash = fixture.state_hash(state["cached_features"], torch)
                branch_arrays = {"B": baseline_array}
                branch_names = ["NOOP", "D1", "D3"] + (["P", "R"] if witness["actionable"] else [])
                for name in branch_names:
                    category = "verification" if name == "NOOP" else "method_fork"
                    counter = "verification" if name == "NOOP" else ("deletion" if name in ("D1", "D3") else "prompt")
                    branch = dict(name=name, status="not_attempted", call_attempted=False, call_completed=False)
                    row["branches"].append(branch)
                    with timed("query-%05d-%s" % (frame, name), category, branch):
                        fork = copy.deepcopy(state)
                        fork_before = fixture.state_hash(fork, torch)
                        require(fork_before == state_before, "deepcopy changed native checkpoint")
                        history = fork["output_dict_per_obj"][0]
                        drop = frame - (1 if name == "D1" else 3) if name in ("D1", "D3") else None
                        selected = dict(cond_frame_outputs=dict(history["cond_frame_outputs"]),
                            non_cond_frame_outputs={} if name == "R" else {
                                i: value for i, value in history["non_cond_frame_outputs"].items()
                                if i < frame and i != drop})
                        point_rows = witness["positive"] + witness["negative"]
                        points = np.asarray([p["target_xy"] for p in point_rows], dtype=np.float32).reshape(-1, 2)
                        point_labels = np.asarray([p["label"] for p in point_rows], dtype=np.int32)
                        before_encoders = result["calls"]["image_encoder_intent"]
                        active.update(name=name, frame=frame)
                        result["calls"][counter + "_intent"] += 1
                        branch["call_attempted"] = True
                        save()
                        try:
                            if name == "P":
                                answer = model.add_new_points_or_box(fork, frame, target,
                                    points=points, labels=point_labels)
                                require(answer[0] == frame and answer[1] == [target], "prompt identity")
                                candidate_logits = answer[2]
                            else:
                                point_inputs = None
                                if name == "R":
                                    point_inputs = dict(point_coords=torch.tensor(points).unsqueeze(0) /
                                        torch.tensor([state["video_width"], state["video_height"]]) * model.image_size,
                                        point_labels=torch.tensor(point_labels).unsqueeze(0))
                                _, low = model._run_single_frame_inference(fork, selected, frame, 1,
                                    False, point_inputs, None, False, False, prev_sam_mask_logits=None)
                                _, candidate_logits = model._get_orig_video_res_output(fork, low)
                        finally:
                            active.clear()
                        branch["call_completed"] = True
                        result["calls"][counter + "_completed"] += 1
                        tensors, meta = take_capture()
                        branch["trace"] = meta
                        require(result["calls"]["image_encoder_intent"] == before_encoders
                                and fixture.state_hash(fork["cached_features"], torch) == baseline_cache_hash,
                                "branch changed/re-encoded current features")
                        require(all(meta[key] == baseline_meta[key] for key in
                                    ("vision_sha256", "vision_position_sha256", "feature_sizes")),
                                "branch did not use identical current features")
                        if name == "NOOP":
                            require(torch.equal(candidate_logits, logits)
                                    and torch.equal(tensors[3], low_selected)
                                    and meta["trace"] == baseline_meta["trace"], "no-op exact parity failed")
                            branch.update(exact_baseline_original_parity=True, exact_baseline_low_parity=True)
                        elif name in ("D1", "D3"):
                            branch["deletion"] = compare_deleted_trace(baseline_meta["trace"], meta["trace"], drop)
                        elif name == "P":
                            require(meta["previous_mask_sha256"] == fixture.tensor_hash(
                                torch.clamp(low_selected, -32., 32.), torch)
                                and meta["trace"] == baseline_meta["trace"],
                                "official current-prompt baseline memory/prior-mask mismatch")
                        else:
                            require(meta["trace"]["spatial_frames"] == [0]
                                    and meta["trace"]["pointer_frames"] == [0]
                                    and meta["previous_mask_sha256"] is None, "reset history/prior mismatch")
                            old = {(b["kind"], b["frame"]): b for b in baseline_meta["trace"]["blocks"]}
                            require(all(b["values_sha256"] == old[(b["kind"], 0)]["values_sha256"]
                                        for b in meta["trace"]["blocks"]), "reset altered original memory values")
                        if name in ("P", "R"):
                            expected_coords = (torch.tensor(points).unsqueeze(0) /
                                torch.tensor([state["video_width"], state["video_height"]]) * model.image_size).tolist()
                            require(meta["point_coords"] == expected_coords
                                    and meta["point_labels"] == [point_labels.tolist()],
                                    "prompt coordinate/complete witness mismatch")
                        else:
                            require(meta["point_coords"] is None and meta["previous_mask_sha256"] is None,
                                    "unprompted branch received prompt/prior")
                        if name != "NOOP":
                            row["candidates"][name] = {}
                            branch_arrays[name] = save_candidate(frame, name, candidate_logits, tensors,
                                                                 row["candidates"][name], witness)
                        fork_after = fixture.state_hash(fork, torch)
                        branch.update(state_sha256_before=fork_before, state_sha256_after=fork_after,
                                      fork_state_changed=fork_after != fork_before)
                        if name != "P":
                            require(fork_after == fork_before, "read-only branch mutated native state")
                        require(fixture.state_hash(state, torch) == state_before, "fork mutated baseline state")
                        branch["baseline_state_unchanged"] = True
                        del fork, selected, history, tensors, candidate_logits
                with timed("selection-%05d" % frame, "selection"):
                    for name in ("B", "D1", "D3"):
                        row["candidates"][name]["unsigned_change"] = (
                            0. if name == "B" else core.unsigned_change(branch_arrays[name], baseline_array))
                    bank = row["candidates"]
                    row["selection"] = core.select_controls(sequence=sequence, object_id=target, frame=frame,
                        actionable=witness["actionable"],
                        losses={name: bank[name]["loss"] for name in core.BANK},
                        predicted_ious={name: bank[name]["predicted_iou"] for name in core.BANK},
                        unsigned_changes={name: bank[name]["unsigned_change"] for name in core.BANK},
                        native_losses=[bank["M%d" % i]["loss"] for i in range(3)],
                        prompt_loss=bank.get("P", {}).get("loss"), reset_loss=bank.get("R", {}).get("loss"))
                    row["baseline_state_sha256_after"] = fixture.state_hash(state, torch)
                    require(row["baseline_state_sha256_after"] == state_before, "query altered future baseline state")
                    row.update(status="completed", future_baseline_scope=
                        "complete native state unchanged after forks; original iterator alone continues")
                    save()
                query_times = result["timings"][timing_start:]
                row["costs"] = dict(
                    seconds={category: sum(item.get("seconds", 0.) for item in query_times
                                           if item["category"] == category)
                             for category in sorted({item["category"] for item in query_times})},
                    counts={key: value - calls_before[key] for key, value in result["calls"].items()})
                row["cost_scope"] = (
                    "this query frame and its probes/checks only; excludes setup and intervening "
                    "baseline frames, which remain included in complete-video operation timing")
                save()
                del baseline_tensors, baseline_array, branch_arrays
            require(next(iterator, None) is None, "unexpected extra native frame")
            with timed("final_checks", "verification"):
                for media in video["witnesses"]:
                    bound_file(media)
                bound_file(video["initial_mask"])
                calls = result["calls"]
                actionable = sum(row["actionable"] for row in result["queries"])
                require(calls["baseline_frames"] == frame_count
                        and calls["image_encoder_intent"] == calls["image_encoder_completed"] == frame_count
                        and calls["verification_intent"] == calls["verification_completed"] == 8
                        and calls["deletion_intent"] == calls["deletion_completed"] == 16
                        and calls["prompt_intent"] == calls["prompt_completed"] == 2 * actionable
                        and calls["track_step_intent"] == calls["track_step_completed"]
                            == frame_count + 24 + 2 * actionable
                        and all(row["status"] == "completed" for row in result["queries"]),
                        "incomplete native operation/call accounting")
                history = state["output_dict_per_obj"][0]
                require(set(history["cond_frame_outputs"]) == {0}
                        and set(history["non_cond_frame_outputs"]) == set(range(1, frame_count)),
                        "baseline memory history changed")
                dtypes = sorted({str(item["maskmem_features"].dtype) for section in history.values()
                                 for item in section.values()})
                require(dtypes == ["torch.bfloat16"], "native compact memory dtype changed")
                result.update(final_baseline_state_sha256=fixture.state_hash(state, torch),
                              observed_compact_memory_dtypes=dtypes, query_count=8,
                              actionable_count=actionable, status="completed", complete=True,
                              stage="completed")
    except BaseException as exc:
        result.update(status="failed", complete=False, error_type=type(exc).__name__,
                      error=str(exc), failed_stage=result.get("stage"))
        for row in result["queries"]:
            if row["status"] == "started":
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if model is not None and original_track_step is not None:
            model._track_step = original_track_step
        for handle in handles:
            handle.remove()
        result["timing_by_category"] = {
            category: sum(item.get("seconds", 0.) for item in result["timings"] if item["category"] == category)
            for category in sorted({item["category"] for item in result["timings"]})}
        result["timing_scope"] = (
            "one complete video; includes deep copies, full state hashes and verification no-ops; "
            "input evidence extraction belongs to the separately charged frontend operation")
        save()
    return result
