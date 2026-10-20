"""Draft manufactured native SAM2/SP/LG qualification; no work on import.

run(installation, output, record) expects already verified local paths:
qualify_helper, sam2_source, lightglue_source, sam2_checkpoint,
superpoint_checkpoint, lightglue_checkpoint, runtime. The caller prepares dependency
sys.path and the offline guard. Output must be fresh. No installation, Git,
acquisition, receipt admission or benchmark logic belongs in this module.
"""
import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
# build_sam2_video_predictor(True) appends fill_hole_area=8 after user overrides.
# Reproduce its other four authored overrides explicitly, with hole filling off.
OVERRIDES = [
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    "++model.binarize_mask_from_pts_for_mem_enc=true",
    "++model.fill_hole_area=0",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor_hash(value, torch):
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    h = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(raw)
    return h.hexdigest()


def state_hash(state, torch):
    """Hash all native state values, including caches and bfloat16 storage."""
    def describe(value):
        if isinstance(value, torch.Tensor):
            return {"tensor": tensor_hash(value, torch)}
        if isinstance(value, dict):
            return [[describe(k), describe(v)] for k, v in value.items()]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [describe(v) for v in value]]
        if isinstance(value, torch.device):
            return {"device": str(value)}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError("Unsupported native state value: " + type(value).__name__)
    return hashlib.sha256(json.dumps(describe(state), allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def module_origins(installation):
    """Check actual imported packages; exact dependency closure is caller-owned."""
    runtime = Path(installation["runtime"]).resolve(strict=True)
    packages = {
        "sam2": ("sam2_source", None), "lightglue": ("lightglue_source", None),
        "numpy": ("runtime", "numpy"), "torch": ("runtime", "torch"),
        "cv2": ("runtime", "opencv-python"), "PIL": ("runtime", "Pillow"),
        "kornia": ("runtime", "kornia"), "hydra": ("runtime", "hydra-core"),
        "omegaconf": ("runtime", "omegaconf"), "iopath": ("runtime", "iopath"),
    }
    checked = {}
    for name, (key, distribution) in packages.items():
        module = importlib.import_module(name)
        root = Path(installation[key]).resolve(strict=True)
        file = getattr(module, "__file__", None)
        paths = [Path(file).resolve(strict=True)] if file else [
            Path(path).resolve(strict=True) for path in getattr(module, "__path__", ())]
        require(bool(paths), "missing import origin: " + name)
        require(all(path.is_relative_to(root) for path in paths), "wrong import origin: " + name)
        item = dict(root_key=key, relative_origins=[str(path.relative_to(root)) for path in paths],
                    namespace=file is None)
        if file:
            item["module_sha256"] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
        if distribution:
            dist = importlib.metadata.distribution(distribution)
            location = Path(dist.locate_file("")).resolve(strict=True)
            require(location.is_relative_to(runtime), "wrong distribution metadata: " + name)
            item.update(distribution=distribution, version=dist.version)
        else:
            item["version_binding"] = "verified source commit and file inventory, not installed distribution"
        checked[name] = item
    return checked


def carrier_hook(model, outputs, traces, torch):
    """Check the actual attention input against fixed frame-7 carrier layout.

    This is a token-accounting check for the pinned native 7-memory/stride-1,
    9-video-frame configuration, not a replacement attention implementation.
    """
    from sam2.modeling.sam2_utils import get_1d_sine_pe
    cond, noncond = outputs["cond_frame_outputs"], outputs["non_cond_frame_outputs"]
    require(set(cond) == {0} and set(noncond) <= set(range(1, 7)), "unexpected probe history")
    spatial = [0] + sorted(noncond)
    pointers = [0] + sorted(noncond, reverse=True)
    def hook(module, args, kwargs):
        require(not args, "unexpected attention positional arguments")
        memory, position = kwargs["memory"], kwargs["memory_pos"]
        blocks, cursor = [], 0
        def block(kind, frame, values, pos):
            nonlocal cursor
            values, pos = values.to(memory), pos.to(position)
            end = cursor + values.shape[0]
            require(torch.equal(memory[cursor:end], values), "carrier feature mismatch")
            require(torch.equal(position[cursor:end], pos), "carrier position mismatch")
            blocks.append(dict(kind=kind, frame=frame, start=cursor, end=end,
                               values_sha256=tensor_hash(memory[cursor:end], torch),
                               position_sha256=tensor_hash(position[cursor:end], torch)))
            cursor = end
        for frame in spatial:
            item = cond[0] if frame == 0 else noncond[frame]
            values = item["maskmem_features"].flatten(2).permute(2, 0, 1)
            pos = item["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
            pos = pos + model.maskmem_tpos_enc[6 - frame]
            block("spatial", frame, values, pos)
        ptrs = torch.stack([(cond[0] if f == 0 else noncond[f])["obj_ptr"]
                            for f in pointers], dim=0)
        distances = torch.tensor([7 - f for f in pointers], device=ptrs.device)
        dim = model.hidden_dim if model.proj_tpos_enc_in_obj_ptrs else model.mem_dim
        # Native max_obj_ptrs=min(num_frames=9,max_obj_ptrs_in_encoder):
        # even the frame-7 probe divides temporal distances by 9-1.
        pos = get_1d_sine_pe(distances / 8, dim=dim)
        pos = model.obj_ptr_tpos_proj(pos).unsqueeze(1).expand(-1, 1, model.mem_dim)
        tokens_per_pointer = model.hidden_dim // model.mem_dim
        require(model.hidden_dim % model.mem_dim == 0, "pointer dimensions")
        ptrs = ptrs.reshape(-1, 1, tokens_per_pointer, model.mem_dim).permute(0, 2, 1, 3)
        for i, frame in enumerate(pointers):
            block("pointer", frame, ptrs[i], pos[i:i + 1].expand(tokens_per_pointer, -1, -1))
        require(cursor == memory.shape[0] and memory.shape == position.shape,
                "unaccounted attention carriers")
        require(kwargs["num_obj_ptr_tokens"] == len(pointers) * tokens_per_pointer,
                "incorrect pointer token count")
        traces.append(dict(spatial_frames=spatial, pointer_frames=pointers,
                           num_obj_ptr_tokens=kwargs["num_obj_ptr_tokens"],
                           total_tokens=cursor, blocks=blocks))
    return hook


def run(installation, output, record):
    """Run only after the parent's source/install admission and offline guard."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="evr0029.native-fixture.v1", status="started",
                  manufactured_only=True, warnings=[], timings=[], outputs=[],
                  baseline=[], probes=[], local_hub_calls=[], loads={})
    record["native_fixture"] = result
    began, old_path = time.perf_counter(), list(sys.path)
    def save():
        result["wall_seconds"] = time.perf_counter() - began
        pending = output / "fixture-record.pending"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(pending, output / "fixture-record.json")
    @contextlib.contextmanager
    def timed(stage):
        result["stage"] = stage
        item = dict(stage=stage, status="started")
        result["timings"].append(item)
        save()
        start = time.perf_counter()
        try:
            yield
            item["status"] = "completed"
        except BaseException:
            item["status"] = "failed"
            raise
        finally:
            item["seconds"] = time.perf_counter() - start
            save()
    save()
    try:
        spec = importlib.util.spec_from_file_location("verified_0022_helpers", installation["qualify_helper"])
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        sys.path[:0] = [str(installation["sam2_source"]), str(installation["lightglue_source"])]
        with timed("imports"), q.capture_warnings(result):
            import numpy as np
            import torch
            import cv2
            from PIL import Image
            from kornia.config import InstallationMode, kornia_config
            kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
            from lightglue import SuperPoint, LightGlue
            from sam2.build_sam import build_sam2_video_predictor
            result["module_origins"] = module_origins(installation)
        with timed("execution_configuration"):
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            cv2.setNumThreads(0)
            result["execution"] = dict(device="cpu", parameter_dtype="float32",
                torch_intraop=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads(),
                opencv_threads=cv2.getNumThreads(), opencv_parallelism_disabled=True,
                authored_compact_memory_dtype="bfloat16", autocast=False, vos_optimized=False)
            require((result["execution"]["torch_intraop"], result["execution"]["torch_interop"],
                     result["execution"]["opencv_threads"]) == (4, 1, 1), "thread configuration")
        with timed("strict_native_loading"), q.capture_warnings(result):
            model = build_sam2_video_predictor(CONFIG, ckpt_path=None, device="cpu", mode="eval",
                apply_postprocessing=False, hydra_overrides_extra=list(OVERRIDES), vos_optimized=False)
            checkpoint = torch.load(installation["sam2_checkpoint"], map_location="cpu", weights_only=True)
            result["loads"]["sam2"] = q.strict_state(model, checkpoint["model"], torch)
            del checkpoint
            model.eval().to(device="cpu", dtype=torch.float32)
            require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
                    "native parameter device/dtype")
            result["native_model"] = dict(parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_tensors=sum(1 for _ in model.parameters()),
                buffer_count=sum(b.numel() for b in model.buffers()),
                state_tensors=len(model.state_dict()),
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
            decoder = model.sam_mask_decoder
            require(model.fill_hole_area == 0 and model.binarize_mask_from_pts_for_mem_enc
                    and decoder.dynamic_multimask_via_stability
                    and decoder.dynamic_multimask_stability_delta == 0.05
                    and decoder.dynamic_multimask_stability_thresh == 0.98, "effective postprocessing")
            require(model.num_maskmem == 7 and model.memory_temporal_stride_for_eval == 1
                    and model.use_obj_ptrs_in_encoder and model.add_tpos_enc_to_obj_ptrs
                    and model.max_obj_ptrs_in_encoder >= 9, "unsupported fixture memory configuration")
            result["config"] = dict(name=CONFIG, overrides=OVERRIDES,
                                   num_maskmem=model.num_maskmem, memory_stride=1)
        with timed("manufactured_jpeg_inputs"):
            images, metadata = q.fixture(np)
            base = images[0]
            result["fixture"] = dict(base=metadata, frames=9, right_shift_per_frame=12,
                padding="zero", jpeg_quality=95, jpeg_subsampling=0,
                quantization="floor(255*clip(x,0,1)+0.5)", initial_mask_yx=[118, 182, 207, 271],
                probe_frame=7, future_frame=8, lightglue_frames=[0, 8])
            frames = output / "frames"
            frames.mkdir()
            result["inputs"] = []
            for i in range(9):
                image = np.zeros_like(base)
                shift = 12 * i
                image[:, shift:] = base[:, :512 - shift]
                pixels = np.floor(255 * np.clip(image, 0, 1) + 0.5).astype(np.uint8)
                path = frames / ("%05d.jpg" % i)
                with path.open("xb") as f:
                    Image.fromarray(pixels).save(f, format="JPEG", quality=95, subsampling=0)
                result["inputs"].append(dict(filename="frames/" + path.name,
                    sha256=q.sha(path), bytes=path.stat().st_size))
            mask = np.zeros((384, 512), dtype=np.bool_)
            mask[118:182, 207:271] = True
            result["outputs"].append(q.save_arrays(output / "initial-mask.npz", {"mask": mask}, np))
        def save_logits(name, logits):
            array = logits.detach().cpu().float().numpy()
            require(array.shape[-2:] == (384, 512) and np.isfinite(array).all(),
                    "invalid original-resolution logits")
            item = q.save_arrays(output / (name + ".npz"), {"logits": array}, np)
            result["outputs"].append(item)
            return item
        with torch.inference_mode(), q.capture_warnings(result):
            with timed("native_init_state"):
                state = model.init_state(str(frames), offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=False)
            with timed("native_initial_mask"):
                initial = model.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
                require(initial[0] == 0 and initial[1] == [1], "initial object identity")
                save_logits("initial-prompt", initial[2])
            iterator = iter(model.propagate_in_video(state))
            for i in range(9):
                with timed("native_baseline_frame_%d" % i):
                    frame, ids, logits = next(iterator)
                    require(frame == i and ids == [1], "baseline order/identity")
                    saved = save_logits("baseline-%02d" % i, logits)
                    result["baseline"].append(dict(frame=frame, output=saved["filename"]))
                    if i == 7:
                        baseline_query = logits.detach().clone()
                        query_state = copy.deepcopy(state)
                    if i == 8:
                        baseline_future = logits.detach().clone()
            require(next(iterator, None) is None, "unexpected baseline frame")
            history = state["output_dict_per_obj"][0]
            memory_dtypes = sorted({str(item["maskmem_features"].dtype)
                for section in ("cond_frame_outputs", "non_cond_frame_outputs")
                for item in history[section].values()})
            require(memory_dtypes == ["torch.bfloat16"], "native compact-memory dtype")
            result["native_model"]["observed_compact_memory_dtypes"] = memory_dtypes
            original_hash = state_hash(state, torch)
            query_hash = state_hash(query_state, torch)
            result["state_after_baseline_sha256"] = original_hash
            result["state_after_frame_7_sha256"] = query_hash
            seen, traces_by_drop = {}, {}
            for order, drop in enumerate((None, 6, 4, 4, 6, None)):
                name = "full" if drop is None else "remove_%d" % drop
                with timed("probe_%d_%s" % (order, name)):
                    fork = copy.deepcopy(query_state)
                    fork_before = state_hash(fork, torch)
                    history = fork["output_dict_per_obj"][0]
                    selected = dict(cond_frame_outputs=dict(history["cond_frame_outputs"]),
                        non_cond_frame_outputs={i: value for i, value in history["non_cond_frame_outputs"].items()
                                                if i < 7 and i != drop})
                    traces = []
                    hook = model.memory_attention.register_forward_pre_hook(
                        carrier_hook(model, selected, traces, torch), with_kwargs=True)
                    try:
                        _, low = model._run_single_frame_inference(fork, selected, 7, 1, False,
                            None, None, False, False, prev_sam_mask_logits=None)
                        _, logits = model._get_orig_video_res_output(fork, low)
                    finally:
                        hook.remove()
                    require(len(traces) == 1, "attention invocation count")
                    saved = save_logits("probe-%d-%s" % (order, name), logits)
                    probe = dict(order=order, removed_frame=drop, output=saved["filename"],
                                 trace=traces[0], status="checking")
                    result["probes"].append(probe)
                    reference = baseline_query if drop is None else seen.get(drop, logits)
                    delta = float((logits - reference).abs().max())
                    require(torch.equal(logits, reference), "CPU no-op or branch-order parity")
                    require(state_hash(fork, torch) == fork_before, "probe mutated its inference state")
                    require(state_hash(state, torch) == original_hash, "probe mutated baseline state")
                    require(state_hash(query_state, torch) == query_hash, "probe mutated query checkpoint")
                    if drop in traces_by_drop:
                        require(traces_by_drop[drop] == traces[0], "branch-order carrier mismatch")
                    seen[drop], traces_by_drop[drop] = logits.detach().clone(), traces[0]
                    probe.update(status="completed", exact_parity=True,
                                 maximum_absolute_parity_error=delta, state_unchanged=True)
                    del fork
            full_blocks = {(b["kind"], b["frame"]): b for b in traces_by_drop[None]["blocks"]}
            for drop in (6, 4):
                blocks = traces_by_drop[drop]["blocks"]
                require({(b["kind"], b["frame"]) for b in blocks} ==
                        {key for key in full_blocks if key[1] != drop}, "carrier deletion coverage")
                for b in blocks:
                    old = full_blocks[(b["kind"], b["frame"])]
                    require(all(b[k] == old[k] for k in ("values_sha256", "position_sha256")),
                            "retained carrier changed")
            result["carrier_removal_and_retention_verified"] = True
            points = np.asarray([[323, 150], [108, 40]], dtype=np.float32)
            labels = np.asarray([1, 0], dtype=np.int32)
            with timed("current_point_prompt"):
                fork = copy.deepcopy(query_state)
                answer = model.add_new_points_or_box(fork, 7, 1, points=points, labels=labels)
                require(answer[0] == 7 and answer[1] == [1], "point prompt identity")
                save_logits("current-point-prompt", answer[2])
                del fork
            with timed("initial_memory_only_and_prompt"):
                fork = copy.deepcopy(query_state)
                fork_before = state_hash(fork, torch)
                selected = dict(cond_frame_outputs=dict(fork["output_dict_per_obj"][0]["cond_frame_outputs"]),
                                non_cond_frame_outputs={})
                # Exact native add_new_points_or_box original-pixel conversion;
                # bypass its previous-mask lookup, without changing the decoder.
                point_inputs = dict(point_coords=torch.tensor(points).unsqueeze(0) /
                                    torch.tensor([512, 384]) * model.image_size,
                                    point_labels=torch.tensor(labels).unsqueeze(0))
                traces = []
                hook = model.memory_attention.register_forward_pre_hook(
                    carrier_hook(model, selected, traces, torch), with_kwargs=True)
                try:
                    _, low = model._run_single_frame_inference(fork, selected, 7, 1, False,
                        point_inputs, None, False, False, prev_sam_mask_logits=None)
                    _, logits = model._get_orig_video_res_output(fork, low)
                finally:
                    hook.remove()
                save_logits("initial-memory-point-prompt", logits)
                require(len(traces) == 1 and traces[0]["spatial_frames"] == [0]
                        and traces[0]["pointer_frames"] == [0], "initial-only carriers")
                require(state_hash(fork, torch) == fork_before, "initial-only prompt mutated state")
                result["initial_memory_prompt_trace"] = traces[0]
                result["reset_scope"] = "retain conditioning frame 0; drop all nonconditioning outputs; same points; no previous mask; no memory encoding"
                del fork
            with timed("native_same_history_video_decoder_bank"):
                fork = copy.deepcopy(query_state)
                fork_before = state_hash(fork, torch)
                history = fork["output_dict_per_obj"][0]
                selected = dict(cond_frame_outputs=dict(history["cond_frame_outputs"]),
                    non_cond_frame_outputs={i: value for i, value in history["non_cond_frame_outputs"].items()
                                            if i < 7})
                require(7 in fork["cached_features"], "decoder probe must use cached frame-7 features")
                _, _, vision, vision_pos, sizes = model._get_image_feature(fork, 7, 1)
                traces = []
                hook = model.memory_attention.register_forward_pre_hook(
                    carrier_hook(model, selected, traces, torch), with_kwargs=True)
                try:
                    # Call the original video path unchanged. _track_step invokes
                    # native _use_multimask and _forward_sam_heads, but does not
                    # encode/write new memory (that belongs to track_step).
                    _, sam_outputs, _, _ = model._track_step(
                        frame_idx=7, is_init_cond_frame=False,
                        current_vision_feats=vision, current_vision_pos_embeds=vision_pos,
                        feat_sizes=sizes, point_inputs=None, mask_inputs=None,
                        output_dict=selected, num_frames=fork["num_frames"],
                        track_in_reverse=False, prev_sam_mask_logits=None)
                finally:
                    hook.remove()
                require(len(sam_outputs) == 7, "unexpected native SAM output schema")
                low_all, high_all, quality, low_selected, high_selected, pointer, object_score = sam_outputs
                require(low_all.ndim == high_all.ndim == 4 and low_all.shape[0] == 1
                        and low_all.shape[:2] == high_all.shape[:2]
                        and quality.shape == low_all.shape[:2] and low_all.shape[1] >= 1,
                        "native decoder bank shape")
                _, all_video = model._get_orig_video_res_output(fork, low_all)
                _, chosen_video = model._get_orig_video_res_output(fork, low_selected)
                tensors = dict(low_resolution_logits=low_all, high_resolution_logits=high_all,
                    original_resolution_logits=all_video, predicted_quality=quality,
                    selected_low_resolution_logits=low_selected,
                    selected_high_resolution_logits=high_selected,
                    selected_original_resolution_logits=chosen_video,
                    object_pointer=pointer, object_score_logits=object_score)
                arrays = {key: value.detach().cpu().float().numpy() for key, value in tensors.items()}
                result["outputs"].append(q.save_arrays(output / "native-video-decoder-bank.npz", arrays, np))
                require(all(np.isfinite(value).all() for value in arrays.values()), "nonfinite decoder bank")
                require(all_video.shape[-2:] == (384, 512), "decoder bank original dimensions")
                chosen = int(torch.argmax(quality[0]).item())
                require(torch.equal(low_selected, low_all[:, chosen:chosen + 1])
                        and torch.equal(high_selected, high_all[:, chosen:chosen + 1]),
                        "native decoder selection mismatch")
                require(torch.equal(chosen_video, baseline_query), "native decoder baseline parity")
                require(len(traces) == 1 and traces[0] == traces_by_drop[None],
                        "decoder bank changed video memory carriers")
                require(state_hash(fork, torch) == fork_before, "decoder probe mutated inference state")
                result["native_video_decoder_bank"] = dict(frame=7, count=int(low_all.shape[1]),
                    native_multimask_requested=bool(model._use_multimask(False, None)),
                    selected_index=chosen, selected_output_exact_baseline_parity=True,
                    same_history_carriers=True, state_unchanged=True, forced_decoder_alternatives=False,
                    scope="original video _track_step outputs; a singleton bank is permitted")
                del fork, sam_outputs, tensors, arrays
            require(state_hash(state, torch) == original_hash, "control mutated baseline state")
            require(state_hash(query_state, torch) == query_hash, "control mutated query checkpoint")
            with timed("native_future_frame_8_after_discarded_probes"):
                fork = copy.deepcopy(query_state)
                # Native propagation recomputes frame 8 and its memory encoder.
                # Only this isolated fork may update its image-feature cache.
                continuation = iter(model.propagate_in_video(fork, start_frame_idx=8,
                                                             max_frame_num_to_track=0))
                frame, ids, logits = next(continuation)
                require(frame == 8 and ids == [1] and next(continuation, None) is None,
                        "future continuation identity")
                save_logits("future-after-probes-08", logits)
                delta = float((logits - baseline_future).abs().max())
                require(torch.equal(logits, baseline_future), "future logit parity")
                require(state_hash(fork, torch) == original_hash, "future complete-state parity")
                require(state_hash(query_state, torch) == query_hash
                        and state_hash(state, torch) == original_hash, "future fork leaked")
                result["future_after_discarded_probes"] = dict(frame=8, exact_logit_parity=True,
                    maximum_absolute_parity_error=delta, exact_complete_state_parity=True,
                    query_checkpoint_unchanged=True, original_baseline_unchanged=True)
                del fork
            with timed("strict_sp_lg_loading"):
                paths = {q.SP_URL: Path(installation["superpoint_checkpoint"]),
                         q.LG_URL: Path(installation["lightglue_checkpoint"])}
                with q.local_hub(torch, paths, result["local_hub_calls"]):
                    sp = SuperPoint(**q.SP_OPTIONS).eval().to(device="cpu", dtype=torch.float32)
                    raw = torch.load(paths[q.SP_URL], map_location="cpu", weights_only=True)
                    result["loads"]["superpoint"] = q.strict_state(sp, raw, torch)
                    lg = LightGlue(**q.LG_OPTIONS).eval().to(device="cpu", dtype=torch.float32)
                    raw = q.rename_lightglue(torch.load(paths[q.LG_URL], map_location="cpu",
                                                       weights_only=True), lg.conf.n_layers)
                    result["loads"]["lightglue"] = q.strict_state(lg, raw, torch, allow_confidence=True)
                    del raw
                require([c["url"] for c in result["local_hub_calls"]] == [q.SP_URL, q.LG_URL],
                        "unexpected local weight loading")
            features = []
            for frame in (0, 8):
                with timed("superpoint_frame_%d" % frame):
                    with Image.open(frames / ("%05d.jpg" % frame)) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
                    feature = sp.extract(tensor, resize=None)
                    raw = {k: v.detach().cpu().numpy() for k, v in feature.items()}
                    result["outputs"].append(q.save_arrays(output / ("features-%d.npz" % frame), raw, np))
                    require(all(np.isfinite(v).all() for v in raw.values()), "nonfinite features")
                    require(raw["keypoints"].shape[0] == 1 and raw["keypoints"].shape[2] == 2
                            and 0 < raw["keypoints"].shape[1] <= 2048, "feature count")
                    require(np.array_equal(raw["image_size"], [[512, 384]]), "feature image size")
                    features.append(feature)
            with timed("lightglue_first_last"):
                matches = lg(dict(image0=features[0], image1=features[1]))
                raw = {k: (v[0].detach().cpu().numpy() if isinstance(v, list) else v.detach().cpu().numpy())
                       for k, v in matches.items() if k != "stop"}
                raw["stop"] = np.asarray(matches["stop"], dtype=np.int64)
                result["outputs"].append(q.save_arrays(output / "first-last-matches.npz", raw, np))
                pairs = raw["matches"][0] if raw["matches"].ndim == 3 else raw["matches"]
                require(all(np.isfinite(v).all() for v in raw.values())
                        and pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) >= 1
                        and np.issubdtype(pairs.dtype, np.integer), "invalid or zero matches")
                for column in (0, 1):
                    require((pairs[:, column] >= 0).all() and
                            (pairs[:, column] < features[column]["keypoints"].shape[1]).all(),
                            "match index bounds")
                require(len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1])),
                        "nonmutual compact matches")
                require(np.array_equal(raw["matches0"][0, pairs[:, 0]], pairs[:, 1])
                        and np.array_equal(raw["matches1"][0, pairs[:, 1]], pairs[:, 0]),
                        "compact/dense matches disagree")
                result["match_count"] = len(pairs)
        result.update(status="completed", stage="completed")
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      failed_stage=result.get("stage"))
        raise
    finally:
        sys.path[:] = old_path
        save()
    return result
