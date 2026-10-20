"""Operator-equivalent D38 implementation candidate; not yet qualified.

No ML imports or work on import. Selected native output is reused exactly.
Batching and multiplicity-compressed SDPA are numerically, not bitwise,
equivalent to the frozen adapter. See ADMISSION; no scientific run is admitted.
"""
from contextlib import ExitStack, contextmanager
import math
import time


EQUIVALENCE_ATOL = 1e-4
EQUIVALENCE_RTOL = 1e-5
ARMS = ("early_mean", "flat_histories", "late_mean", "output_ensemble")


def require(ok, message):
    if not ok:
        raise ValueError(message)


@contextmanager
def _replace(obj, name, value):
    own, old = name in obj.__dict__, obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if own:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def cow_context(model, torch, capture, write):
    """Native read-only context construction; copies maps, never whole tensors.

    The pinned native preparation reads compact tensors, creates cat/stack/pos
    sums, and does not assign into them. Only the separately copied latest entry
    is replaced. The capture-only forward does not execute any stack.
    """
    frame = capture["frame"]
    require(frame >= 2 and not model.training, "ordinary eval context required")
    observer = getattr(model, "_d38_observer", None)
    require(observer is None or not observer.active, "side context during native observation")
    source = capture["history"]
    require(set(source) == {"cond_frame_outputs", "non_cond_frame_outputs"}, "history map keys")
    history = {key: dict(value) for key, value in source.items()}
    history["non_cond_frame_outputs"][frame - 1] = dict(source["non_cond_frame_outputs"][frame - 1])
    entry = history["non_cond_frame_outputs"][frame - 1]
    for key in ("maskmem_features", "maskmem_pos_enc", "obj_ptr", "pred_masks", "object_score_logits"):
        if key in write:
            entry[key] = write[key]
    captured = []

    def collect(*args, **kwargs):
        require(not args and set(kwargs) == set(capture["attention_context"]), "native kwargs changed")
        captured.append(dict(kwargs))
        return capture["attention_output"]

    with _replace(model.memory_attention, "forward", collect):
        model._prepare_memory_conditioned_features(frame_idx=frame,
            is_init_cond_frame=False,
            current_vision_feats=capture["current_vision_feats"][-1:],
            current_vision_pos_embeds=capture["current_vision_pos_embeds"][-1:],
            feat_sizes=capture["feat_sizes"][-1:], output_dict=history,
            num_frames=capture["num_frames"], track_in_reverse=False)
    require(len(captured) == 1, "one capture-only attention dispatch")
    return captured[0]


def compress_contexts(contexts, *, spatial_grid_tokens, pointer_width, torch, operators):
    """Compress repeated COMMON full-history tokens using log multiplicities.

    Restricted native layout: one initial conditioning frame, latest noninitial
    spatial grid last, and latest pointer immediately after the initial pointer.
    All complete spatial grids keep their within-grid RoPE phase. Pointer tokens
    remain last and excluded from spatial RoPE. Common tokens get log(3); every
    candidate gets zero, even when candidate values happen to be identical.
    """
    operators.contexts_identical(contexts, torch=torch)  # validates equal layouts/positions
    require(type(spatial_grid_tokens) is int and spatial_grid_tokens > 0
            and type(pointer_width) is int and pointer_width > 0, "positive layout units")
    first = contexts[0]
    pointers = first["num_obj_ptr_tokens"]
    spatial = first["memory"].shape[0] - pointers
    require(spatial >= 2 * spatial_grid_tokens and spatial % spatial_grid_tokens == 0
            and pointers >= 2 * pointer_width and pointers % pointer_width == 0,
            "complete initial/latest grids and pointers required")
    variable_start = spatial - spatial_grid_tokens
    p0, p1 = spatial + pointer_width, spatial + 2 * pointer_width
    for context in contexts[1:]:
        for left, right in ((0, variable_start), (spatial, p0), (p1, spatial + pointers)):
            require(torch.equal(first["memory"][left:right], context["memory"][left:right]),
                    "nonvariable full-history tokens differ")
    out = {"curr": first["curr"], "curr_pos": first["curr_pos"]}
    for key in ("memory", "memory_pos"):
        pieces = [first[key][:variable_start]]
        pieces += [context[key][variable_start:spatial] for context in contexts]
        pieces += [first[key][spatial:p0]]
        pieces += [context[key][p0:p1] for context in contexts]
        pieces += [first[key][p1:]]
        out[key] = torch.cat(pieces, dim=0)
    out["num_obj_ptr_tokens"] = pointers + 2 * pointer_width
    common = torch.full((out["memory"].shape[0],), math.log(3),
                        dtype=torch.float32, device=first["memory"].device)
    common[variable_start:variable_start + 3 * spatial_grid_tokens] = 0
    new_spatial = spatial + 2 * spatial_grid_tokens
    common[new_spatial + pointer_width:new_spatial + 4 * pointer_width] = 0
    return out, common, dict(operator="flat_full_history_exact_multiplicity_compression",
        common_multiplicity=3, candidate_multiplicity=1,
        original_flat_spatial_tokens=3 * spatial,
        compressed_spatial_tokens=new_spatial,
        original_flat_pointer_tokens=3 * pointers,
        compressed_pointer_tokens=out["num_obj_ptr_tokens"],
        native_grid_tokens=spatial_grid_tokens, pointer_width=pointer_width,
        numeric_equivalence="real arithmetic identity; FP32 qualification required")


def _compressed_forward(module, torch, log_multiplicity):
    """Exact native projections/RoPE, with SDPA additive log-multiplicity mask."""
    original = module.forward
    function = getattr(original, "__func__", None)
    rotary = getattr(function, "__globals__", {}).get("apply_rotary_enc")
    require(callable(rotary), "pinned RoPE apply_rotary_enc unavailable")
    require(not module.training and module.rope_k_repeat, "eval repeating native RoPE required")

    def forward(q, k, v, num_k_exclude_rope=0):
        q = module._separate_heads(module.q_proj(q), module.num_heads)
        k = module._separate_heads(module.k_proj(k), module.num_heads)
        v = module._separate_heads(module.v_proj(v), module.num_heads)
        require(q.dtype == k.dtype == v.dtype == torch.float32
                and q.device.type == "cpu", "qualified FP32 CPU attention required")
        require(log_multiplicity.ndim == 1 and log_multiplicity.numel() == k.shape[-2],
                "multiplicity per key")
        require(module.freqs_cis.shape[0] == q.shape[-2], "fixed native RoPE query grid")
        n = k.shape[-2] - num_k_exclude_rope
        require(n > 0 and n % q.shape[-2] == 0, "complete RoPE grids and pointer tail")
        q, k[:, :, :n] = rotary(q, k[:, :, :n],
            freqs_cis=module.freqs_cis.to(q.device), repeat_freqs_k=True)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
            attn_mask=log_multiplicity.to(dtype=q.dtype, device=q.device).view(1, 1, 1, -1),
            dropout_p=0.0)
        return module.out_proj(module._recombine_heads(out))
    return forward


def compressed_stack(model, torch, context, log_multiplicity):
    """Preserve all four native layers; restore all patched forwards on failure."""
    require(not model.training and len(model.memory_attention.layers) == 4,
            "exact four-layer eval stack required")
    layers = [layer.cross_attn_image for layer in model.memory_attention.layers]
    require(len({id(layer) for layer in layers}) == 4, "independent native layer modules")
    with ExitStack() as stack:
        for module in layers:
            replacement = _compressed_forward(module, torch, log_multiplicity)
            stack.enter_context(_replace(module, "forward", replacement))
        return model.memory_attention(**context)


def batch_contexts(contexts, torch):
    """Native sequence x batch x channels; hypotheses never share attention."""
    require(len(contexts) >= 1, "nonempty context batch")
    first = contexts[0]
    for context in contexts:
        require(set(context) == set(first)
                and context["num_obj_ptr_tokens"] == first["num_obj_ptr_tokens"], "batch layout keys")
        for key in ("memory", "memory_pos"):
            require(context[key].shape == first[key].shape and context[key].shape[1] == 1,
                    "same single-object context shape")
        for key in ("curr", "curr_pos"):
            require(isinstance(context[key], list) and len(context[key]) == 1
                    and context[key][0].shape == first[key][0].shape
                    and context[key][0].shape[1] == 1, "one native current level")
    out = {key: [torch.cat([context[key][0] for context in contexts], dim=1)]
           for key in ("curr", "curr_pos")}
    out.update({key: torch.cat([context[key] for context in contexts], dim=1)
                for key in ("memory", "memory_pos")})
    out["num_obj_ptr_tokens"] = first["num_obj_ptr_tokens"]
    return out


def decode_batch(model, torch, capture, stack_outputs):
    """Five independent native SAM heads; high-res current features repeated."""
    count = len(stack_outputs)
    require(count > 0, "nonempty head batch")
    spatial = [output.permute(1, 2, 0).reshape(1, model.hidden_dim,
                *capture["feat_sizes"][-1]) for output in stack_outputs]
    high = capture["high_res_features"]
    if high is not None:
        require(all(level.shape[0] == 1 for level in high), "one current high-res feature batch")
        high = [level.expand(count, -1, -1, -1) for level in high]
    outputs = model._forward_sam_heads(backbone_features=torch.cat(spatial, dim=0),
        point_inputs=None, mask_inputs=None, high_res_features=high, multimask_output=True)
    require(len(outputs) == 7 and all(value.shape[0] == count for value in outputs),
            "native batched seven SAM fields")
    return [tuple(value[i:i + 1] for value in outputs) for i in range(count)]


def optimized_readouts(model, torch, capture, writes, operators, adapter, *, audit_tensors=None):
    """Same four arms; COW context, batch3 stacks, compressed flat, batch5 heads.

    Caller must have separately qualified selected-branch reuse. This source
    candidate always checks exact selected complete context. No identity skip is
    used, so manufactured absence does not hide the full cost. audit_tensors is
    optional caller-owned tensor storage, never included in JSON metadata.
    """
    adapter._idle(model)
    started = time.perf_counter()
    calls = dict(context_materializations=0, attention_stack_calls=0,
                 attention_stack_examples=0, sam_head_calls=0, sam_head_examples=0)
    meta = dict(frame=capture["frame"], calls=calls, timings=[], selected_reuse=True,
                history_tensor_copy_bytes=0, history_mode="copy-on-write maps; borrowed read-only tensors",
                equivalence_atol=EQUIVALENCE_ATOL, equivalence_rtol=EQUIVALENCE_RTOL)
    native = capture["sam_outputs"]
    if capture["frame"] < 2:
        meta.update(noop_reason="initial_or_first_noninitial", elapsed_seconds=time.perf_counter()-started)
        return {name: native[3].clone() for name in ARMS}, meta
    require(len(writes) == 3 and [w["candidate_index"] for w in writes] == [0, 1, 2]
            and sum(bool(w["is_native_selected"]) for w in writes) == 1, "original three slots")
    require(set(capture["history"]["cond_frame_outputs"]) == {0}
            and model.memory_temporal_stride_for_eval == 1
            and model.max_cond_frames_in_attn in (-1, 1)
            and model.use_obj_ptrs_in_encoder and model.only_obj_ptrs_in_the_past_for_eval,
            "fixed causal initial-only conditioning layout")
    chosen = next(i for i, write in enumerate(writes) if write["is_native_selected"])
    others = [i for i in range(3) if i != chosen]
    grid = capture["feat_sizes"][-1][0] * capture["feat_sizes"][-1][1]
    require(model.hidden_dim % model.mem_dim == 0, "native pointer width")
    pointer_width = model.hidden_dim // model.mem_dim

    def timed(name, function):
        begin = time.perf_counter()
        result = function()
        meta["timings"].append(dict(name=name, seconds=time.perf_counter()-begin))
        return result

    def context(write):
        calls["context_materializations"] += 1
        return cow_context(model, torch, capture, write)

    contexts = [timed("context_slot_" + str(i), lambda w=w: context(w)) for i, w in enumerate(writes)]
    require(adapter.equal_tree(contexts[chosen], capture["attention_context"], torch),
            "selected native complete context parity")
    meta["selected_context_exact"] = True
    expected_grids = 1 + min(capture["frame"] - 1, model.num_maskmem - 1)
    max_pointers = min(capture["num_frames"], model.max_obj_ptrs_in_encoder)
    expected_ptr_tokens = pointer_width * min(capture["frame"], max_pointers)
    for value in contexts:
        require(value["memory"].shape[0] == grid * expected_grids + expected_ptr_tokens
                and value["num_obj_ptr_tokens"] == expected_ptr_tokens, "native token topology")

    mean_write, mean_info = operators.mean_write(writes, torch=torch)
    early_context = timed("context_early_mean", lambda: context(mean_write))
    batch = timed("batch_context_assembly", lambda: batch_contexts(
        [contexts[i] for i in others] + [early_context], torch))
    calls["attention_stack_calls"] += 1
    calls["attention_stack_examples"] += 3
    stacked = timed("stack_batch_two_alternatives_plus_early", lambda: model.memory_attention(**batch))
    branch_outputs = [None, None, None]
    branch_outputs[chosen] = capture["attention_output"]
    for batch_index, slot in enumerate(others):
        branch_outputs[slot] = stacked[:, batch_index:batch_index + 1]
    early_output = stacked[:, 2:3]
    flat_context, bias, flat_info = timed("flat_compression", lambda: compress_contexts(contexts,
        spatial_grid_tokens=grid, pointer_width=pointer_width, torch=torch, operators=operators))
    calls["attention_stack_calls"] += 1
    calls["attention_stack_examples"] += 1
    flat_output = timed("stack_compressed_flat", lambda: compressed_stack(model, torch, flat_context, bias))
    late_output, late_info = operators.mean_stack_outputs(branch_outputs, torch=torch)

    head_names = ["slot" + str(i) for i in others] + ["early", "flat", "late"]
    head_inputs = [branch_outputs[i] for i in others] + [early_output, flat_output, late_output]
    calls["sam_head_calls"] += 1
    calls["sam_head_examples"] += 5
    decoded = timed("heads_batch_five", lambda: decode_batch(model, torch, capture, head_inputs))
    sam_by_name = dict(zip(head_names, decoded))
    sam_by_name["slot" + str(chosen)] = native
    stack_by_name = {"slot" + str(i): branch_outputs[i] for i in range(3)}
    stack_by_name.update(early=early_output, flat=flat_output, late=late_output)
    for name, sam in sam_by_name.items():
        require(len(sam) == 7, "seven SAM output fields")
        for tensor in sam:
            adapter._finite(tensor, torch, name + " SAM")
    for name, tensor in stack_by_name.items():
        adapter._finite(tensor, torch, name + " stack", capture["attention_output"].shape)
    ensemble = torch.stack([sam_by_name["slot" + str(i)][3].float() for i in range(3)]).mean(0)
    low = dict(early_mean=sam_by_name["early"][3], flat_histories=sam_by_name["flat"][3],
               late_mean=sam_by_name["late"][3], output_ensemble=ensemble)
    for name, value in low.items():
        adapter._finite(value, torch, name, native[3].shape)
    if audit_tensors is not None:
        require(isinstance(audit_tensors, dict) and not audit_tensors, "empty audit dictionary required")
        audit_tensors.update(stack_by_name=adapter.clone_tree(stack_by_name, torch),
                             sam_by_name=adapter.clone_tree(sam_by_name, torch))
    meta.update(operators=dict(early=mean_info, flat=flat_info, late=late_info),
        selected_stack_reused_exact=True, selected_sam_reused_exact=True,
        selected_indices={name: int(torch.argmax(sam[2][0]).item()) for name, sam in sam_by_name.items()},
        presence_signs={name: bool(sam[6].item() > 0) for name, sam in sam_by_name.items()},
        elapsed_seconds=time.perf_counter()-started)
    return {name: value.detach().clone() for name, value in low.items()}, meta
