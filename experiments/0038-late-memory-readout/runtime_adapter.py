"""D38 source-prepared native observation and side-readout adapter.

No Torch/SAM imports, file access or model work on import. Torch and an already
qualified CPU/eval predictor are injected. The caller owns all admission, model
loading, inference_mode, source/state hashes, supervision and original resizing.
Only one object is supported. Side readouts never call track_step or write state.
"""
from contextlib import contextmanager
import time


ARMS = ("early_mean", "flat_histories", "late_mean", "output_ensemble")
COMPACT = ("maskmem_features", "maskmem_pos_enc", "pred_masks", "obj_ptr",
           "object_score_logits")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def clone_tree(value, torch):
    """Private detached tensor copies, with no alias into native state."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(v, torch) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unsupported snapshot type: " + type(value).__name__)


def equal_tree(a, b, torch):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.shape == b.shape
                and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y, torch) for x, y in zip(a, b))
    return a == b


def _finite(value, torch, name, shape=None):
    require(isinstance(value, torch.Tensor) and value.is_floating_point()
            and value.device.type == "cpu" and value.numel() > 0
            and (shape is None or tuple(value.shape) == tuple(shape))
            and bool(torch.isfinite(value).all().item()), name + " finite CPU layout")


def _idle(model):
    observer = getattr(model, "_d38_observer", None)
    require(observer is None or not observer.active, "side work inside native observation")
    require(not model.training, "eval predictor required")


@contextmanager
def _replace_attribute(obj, name, value):
    own = name in obj.__dict__
    old = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if own:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


class NativeObserver:
    """Transparent single-object capture, including initial mask-output route.

    Recording is active only inside the original _track_step. Side decoder and
    context materialization calls are ignored. pop removes ownership from this
    bounded queue; callers should retain only previous/current captures.
    """
    def __init__(self, model, torch, max_pending=2):
        require(type(max_pending) is int and max_pending >= 1, "positive capture bound")
        self.model, self.torch, self.max_pending = model, torch, max_pending
        self._pending, self._active, self._entered = {}, False, False
        self._tokens, self._attention, self._handles = [], [], []

    @property
    def active(self):
        return self._active

    @property
    def pending(self):
        return tuple(self._pending)

    def pop(self, frame):
        require(type(frame) is int and frame in self._pending, "missing captured frame")
        return self._pending.pop(frame)

    def _decoder_hook(self, _module, _args, _kwargs, answer):
        if self._active:
            require(isinstance(answer, tuple) and len(answer) == 4, "native decoder tuple")
            self._tokens.append(clone_tree(answer[2], self.torch))

    def _attention_hook(self, _module, args, kwargs, answer):
        if self._active:
            require(not args and set(kwargs) == {"curr", "curr_pos", "memory",
                    "memory_pos", "num_obj_ptr_tokens"}, "native attention kwargs")
            self._attention.append((clone_tree(kwargs, self.torch),
                                    clone_tree(answer, self.torch)))

    def __enter__(self):
        require(not self._entered and not hasattr(self.model, "_d38_observer"),
                "observer cannot be nested/reused")
        _idle(self.model)
        self._entered = True
        self._old_track = self.model._track_step
        self._own_track = "_track_step" in self.model.__dict__
        self._old_track_attr = self.model.__dict__.get("_track_step")
        self.model._d38_observer = self
        try:
            self._handles.append(self.model.sam_mask_decoder.register_forward_hook(
                self._decoder_hook, with_kwargs=True))
            self._handles.append(self.model.memory_attention.register_forward_hook(
                self._attention_hook, with_kwargs=True))
            self.model._track_step = self._observed
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_exception):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._own_track:
            self.model._track_step = self._old_track_attr
        else:
            self.model.__dict__.pop("_track_step", None)
        self.model.__dict__.pop("_d38_observer", None)
        self._active = False

    def _observed(self, frame_idx, is_init_cond_frame, current_vision_feats,
                  current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                  output_dict, num_frames, track_in_reverse, prev_sam_mask_logits):
        torch = self.torch
        require(not self._active and len(self._pending) < self.max_pending
                and frame_idx not in self._pending, "nested/overflow/duplicate capture")
        require(type(frame_idx) is int and 0 <= frame_idx < num_frames
                and not track_in_reverse and point_inputs is None
                and prev_sam_mask_logits is None
                and bool(is_init_cond_frame) == (frame_idx == 0)
                and (mask_inputs is not None) == (frame_idx == 0)
                and current_vision_feats[-1].shape[1] == 1,
                "single-object initial-mask forward-only route")
        started = time.perf_counter()
        self._tokens, self._attention, self._active = [], [], True
        try:
            answer = self._old_track(frame_idx, is_init_cond_frame,
                current_vision_feats, current_vision_pos_embeds, feat_sizes,
                point_inputs, mask_inputs, output_dict, num_frames,
                track_in_reverse, prev_sam_mask_logits)
        finally:
            self._active = False
        require(len(answer) == 4 and len(answer[1]) == 7 and len(self._tokens) == 1,
                "native track/decoder result count")
        bank = 1 if frame_idx == 0 else 3
        low, high, quality, selected_low, selected_high, pointer, presence = answer[1]
        for value, shape, name in (
            (low, (1, bank, 256, 256), "low bank"),
            (high, (1, bank, 1024, 1024), "high bank"),
            (quality, (1, bank), "quality"), (pointer, (1, 256), "pointer"),
            (presence, (1, 1), "presence"),
            (self._tokens[0], (1, bank, 256), "decoder tokens")):
            _finite(value, torch, name, shape)
            require(value.dtype == torch.float32, name + " FP32")
        selected = int(torch.argmax(quality[0]).item())
        require(torch.equal(selected_low, low[:, selected:selected + 1])
                and torch.equal(selected_high, high[:, selected:selected + 1]),
                "native selected bank parity")
        require(len(self._attention) == (0 if frame_idx == 0 else 1),
                "initial/ordinary native attention count")
        context, stack = self._attention[0] if self._attention else (None, None)
        self._pending[frame_idx] = dict(frame=frame_idx,
            is_init_cond_frame=bool(is_init_cond_frame), num_frames=num_frames,
            history=clone_tree(output_dict, torch),
            sam_outputs=clone_tree(answer[1], torch), decoder_tokens=self._tokens[0],
            current_vision_feats=clone_tree(current_vision_feats, torch),
            current_vision_pos_embeds=clone_tree(current_vision_pos_embeds, torch),
            feat_sizes=clone_tree(feat_sizes, torch),
            high_res_features=clone_tree(answer[2], torch), pix_feat=clone_tree(answer[3], torch),
            attention_context=context, attention_output=stack, selected_index=selected,
            capture_seconds=time.perf_counter() - started)
        return answer  # the original object, unmodified


def project_pointer(model, torch, token, presence):
    """Native per-token MLP and exact presence/no-object operations (one row)."""
    _idle(model)
    _finite(token, torch, "one candidate token", (1, model.hidden_dim))
    _finite(presence, torch, "presence", (1, 1))
    pointer = model.obj_ptr_proj(token)
    if model.pred_obj_scores:
        appearing = (presence.sigmoid() if model.soft_no_obj_ptr else
                     (presence > 0).float())
        if model.fixed_no_obj_ptr:
            pointer = appearing * pointer
        pointer = pointer + (1 - appearing) * model.no_obj_ptr
    _finite(pointer, torch, "projected pointer", token.shape)
    return pointer.detach().clone()


def candidate_writes(model, torch, prior_capture, native_compact_write):
    """Original three decoder-slot order; selected stored write is reused exactly."""
    _idle(model)
    started = time.perf_counter()
    require(prior_capture["frame"] >= 1 and model.use_multimask_token_for_obj_ptr
            and model.use_obj_ptrs_in_encoder and not model.non_overlap_masks_for_mem_enc,
            "coherent ordinary-frame candidate configuration")
    require(set(COMPACT) <= set(native_compact_write), "complete compact native write")
    sam = prior_capture["sam_outputs"]
    chosen = prior_capture["selected_index"]
    for key, value in (("pred_masks", sam[3]), ("obj_ptr", sam[5]),
                       ("object_score_logits", sam[6])):
        require(equal_tree(native_compact_write[key], value, torch), "native write parity: " + key)
    memory = native_compact_write["maskmem_features"]
    _finite(memory, torch, "native compact memory")
    require(memory.dtype == torch.bfloat16, "native BF16 compact memory required")
    writes, durations = [], []
    for slot in range(3):
        begin = time.perf_counter()
        write = clone_tree({k: native_compact_write[k] for k in COMPACT}, torch)
        if slot != chosen:
            pointer = project_pointer(model, torch, prior_capture["decoder_tokens"][:, slot], sam[6])
            features, positions = model._encode_new_memory(
                current_vision_feats=prior_capture["current_vision_feats"],
                feat_sizes=prior_capture["feat_sizes"],
                pred_masks_high_res=sam[1][:, slot:slot + 1],
                object_score_logits=sam[6], is_mask_from_pts=False)
            _finite(features, torch, "encoded candidate memory", memory.shape)
            require(equal_tree(positions, native_compact_write["maskmem_pos_enc"], torch),
                    "alternate native spatial position parity")
            write.update(maskmem_features=features.to(dtype=memory.dtype, device=memory.device).detach().clone(),
                         pred_masks=sam[0][:, slot:slot + 1].clone(), obj_ptr=pointer)
        write.update(candidate_index=slot, is_native_selected=(slot == chosen))
        writes.append(write)
        durations.append(dict(slot=slot, reused_native=(slot == chosen),
                              seconds=time.perf_counter() - begin))
    return writes, dict(frame=prior_capture["frame"], selected_index=chosen,
        calls=dict(memory_encoders=2, pointer_projections=2), timings=durations,
        elapsed_seconds=time.perf_counter() - started)


def context_for_write(model, torch, capture, write):
    """Use native layout/time/pointer code, but do not execute an attention stack.

    Module hooks can see this capture-only dispatch; it is NOT a real stack call.
    The native output is returned only to satisfy the native reshape, discarded.
    """
    _idle(model)
    frame = capture["frame"]
    require(frame >= 2 and capture["attention_output"] is not None,
            "latest noninitial write requires t>=2")
    history = clone_tree(capture["history"], torch)
    require(frame - 1 in history["non_cond_frame_outputs"], "previous compact write missing")
    replacement = history["non_cond_frame_outputs"][frame - 1]
    replacement.update(clone_tree({k: write[k] for k in COMPACT if k in write}, torch))
    captured = []

    def collect(*args, **kwargs):
        require(not args and set(kwargs) == set(capture["attention_context"]),
                "native context kwargs changed")
        captured.append(clone_tree(kwargs, torch))
        return capture["attention_output"].clone()

    with _replace_attribute(model.memory_attention, "forward", collect):
        model._prepare_memory_conditioned_features(frame_idx=frame,
            is_init_cond_frame=False,
            current_vision_feats=capture["current_vision_feats"][-1:],
            current_vision_pos_embeds=capture["current_vision_pos_embeds"][-1:],
            feat_sizes=capture["feat_sizes"][-1:], output_dict=history,
            num_frames=capture["num_frames"], track_in_reverse=False)
    require(len(captured) == 1, "one native context materialization required")
    return captured[0]


def readouts(model, torch, capture, writes, operators, *, reuse_selected=False,
             selected_branch_qualified=False, identity_fastpath=False):
    """Return (four low-logit tensors, JSON metadata); never writes history.

    All work is outside the observer's active baseline call. Default costing
    executes selected branch and verifies its native exact stack/decoder parity.
    Qualified reuse must be explicitly requested. Identity fast path is opt-in.
    """
    _idle(model)
    started = time.perf_counter()
    counts = dict(context_materializations=0, attention_stacks=0, sam_heads=0)
    metadata = dict(frame=capture["frame"], calls=counts, timings=[],
                    selected_reuse=bool(reuse_selected), identity_fastpath_used=False)
    native_low = capture["sam_outputs"][3]
    if capture["frame"] < 2:
        metadata.update(noop_reason="initial_or_first_noninitial", elapsed_seconds=time.perf_counter()-started)
        return {arm: native_low.clone() for arm in ARMS}, metadata
    require(not reuse_selected or selected_branch_qualified, "reuse requires prior exact qualification")
    require(len(writes) == 3 and [w["candidate_index"] for w in writes] == [0, 1, 2]
            and sum(bool(w["is_native_selected"]) for w in writes) == 1,
            "three original slots with one native selected marker")
    chosen = next(i for i, w in enumerate(writes) if w["is_native_selected"])

    def timed(name, function):
        begin = time.perf_counter()
        result = function()
        metadata["timings"].append(dict(name=name, seconds=time.perf_counter()-begin))
        return result

    def context(write):
        counts["context_materializations"] += 1
        return context_for_write(model, torch, capture, write)

    def stack(kwargs):
        counts["attention_stacks"] += 1
        result = model.memory_attention(**kwargs)
        _finite(result, torch, "branch stack", capture["attention_output"].shape)
        return result

    def decode(output):
        counts["sam_heads"] += 1
        spatial = output.permute(1, 2, 0).view(1, model.hidden_dim, *capture["feat_sizes"][-1])
        result = model._forward_sam_heads(backbone_features=spatial,
            point_inputs=None, mask_inputs=None,
            high_res_features=capture["high_res_features"], multimask_output=True)
        require(len(result) == 7, "native side decoder tuple")
        for tensor in result:
            _finite(tensor, torch, "side SAM output")
        _finite(result[3], torch, "selected side low", native_low.shape)
        return result

    contexts = [timed("context_slot_" + str(i), lambda w=w: context(w))
                for i, w in enumerate(writes)]
    require(equal_tree(contexts[chosen], capture["attention_context"], torch),
            "selected replacement complete context differs from original native")
    metadata["selected_context_exact"] = True
    identical = operators.contexts_identical(contexts, torch=torch)
    metadata["contexts_identical"] = identical
    if identical and identity_fastpath:
        require(selected_branch_qualified, "native identity reuse requires prior qualification")
        metadata.update(identity_fastpath_used=True, selected_stack_exact=None,
            selected_sam_exact=None, elapsed_seconds=time.perf_counter()-started)
        return {arm: native_low.clone() for arm in ARMS}, metadata

    outputs, branch_sam = [], []
    for i, kwargs in enumerate(contexts):
        if i == chosen and reuse_selected:
            output, sam = capture["attention_output"].clone(), clone_tree(capture["sam_outputs"], torch)
        else:
            output = timed("stack_slot_" + str(i), lambda k=kwargs: stack(k))
            sam = timed("decoder_slot_" + str(i), lambda z=output: decode(z))
        if i == chosen:
            require(equal_tree(output, capture["attention_output"], torch), "native selected stack parity")
            require(equal_tree(sam, capture["sam_outputs"], torch), "native selected SAM parity")
            metadata.update(selected_stack_exact=True, selected_sam_exact=True)
        outputs.append(output)
        branch_sam.append(sam)

    mean_write, mean_info = operators.mean_write(writes, torch=torch)
    mean_context = timed("context_early_mean", lambda: context(mean_write))
    early_stack = timed("stack_early_mean", lambda: stack(mean_context))
    early = timed("decoder_early_mean", lambda: decode(early_stack))[3]
    flat_context, flat_info = operators.flatten_contexts(contexts, torch=torch)
    flat_stack = timed("stack_flat_histories", lambda: stack(flat_context))
    flat = timed("decoder_flat_histories", lambda: decode(flat_stack))[3]
    late_stack, late_info = operators.mean_stack_outputs(outputs, torch=torch)
    late = timed("decoder_late_mean", lambda: decode(late_stack))[3]
    ensemble = torch.stack([sam[3].float() for sam in branch_sam]).mean(0)
    result = dict(early_mean=early, flat_histories=flat, late_mean=late, output_ensemble=ensemble)
    for name, tensor in result.items():
        _finite(tensor, torch, name, native_low.shape)
    metadata.update(operators=dict(early=mean_info, flat=flat_info, late=late_info),
        branch_selected_indices=[int(torch.argmax(sam[2][0]).item()) for sam in branch_sam],
        branch_presence=[float(sam[6].item()) for sam in branch_sam],
        branch_quality=[sam[2][0].detach().tolist() for sam in branch_sam],
        elapsed_seconds=time.perf_counter()-started)
    return {k: v.detach().clone() for k, v in result.items()}, metadata
