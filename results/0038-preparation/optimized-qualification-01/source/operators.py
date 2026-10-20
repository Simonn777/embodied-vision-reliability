"""Known uniform merge operators; no model loading or posterior interpretation.

Inputs are three coherent, already encoded/presence-handled candidate writes or
complete native attention kwargs. All helpers return (new_value, JSON info).
The caller owns native pointer projection, positional construction and history.
Torch is injected or imported only when a helper is called.
"""


def _torch(torch):
    if torch is None:
        import torch as module
        return module
    return torch


def _three(items):
    if not isinstance(items, (list, tuple)) or len(items) != 3:
        raise ValueError("exactly three native candidate slots required")


def _tensor(x, torch, name, ndim=None):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError(name + " must be a floating tensor")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(name + " rank")
    if x.numel() == 0 or not bool(torch.isfinite(x).all()):
        raise ValueError(name + " must be nonempty and finite")


def _same_layout(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device


def _equal(a, b, torch):
    if isinstance(a, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_equal(x, y, torch) for x, y in zip(a, b)))
    return isinstance(b, torch.Tensor) and _same_layout(a, b) and torch.equal(a, b)


def _clone(x):
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(t) for t in x)
    return x.clone()


def _info(operator, identical, fast, **counts):
    return dict(operator=operator, candidate_count=3,
                identical_inputs=bool(identical),
                identity_fastpath_used=bool(fast), **counts)


def mean_write(writes, *, torch=None, identity_fastpath=False):
    """Mean compact spatial features and ALREADY projected pointers in float32.

    Keys: maskmem_features[B,C,H,W], obj_ptr[B,D], maskmem_pos_enc[list].
    Cast each mean back to that field's native dtype. Position tensors must agree
    exactly and are cloned, never averaged. No other output-dict fields copied.
    """
    torch = _torch(torch)
    _three(writes)
    for w in writes:
        _tensor(w["maskmem_features"], torch, "maskmem_features", 4)
        _tensor(w["obj_ptr"], torch, "obj_ptr", 2)
        if w["maskmem_features"].shape[0] != w["obj_ptr"].shape[0]:
            raise ValueError("write batch mismatch")
        if w["maskmem_features"].device != w["obj_ptr"].device:
            raise ValueError("write device mismatch")
        pos = w["maskmem_pos_enc"]
        if not isinstance(pos, (list, tuple)) or not pos:
            raise ValueError("native spatial position list required")
        for p in pos:
            _tensor(p, torch, "maskmem_pos_enc", 4)
            if p.shape != w["maskmem_features"].shape or p.device != w["maskmem_features"].device:
                raise ValueError("spatial position shape/device mismatch")
    ref = writes[0]
    for w in writes[1:]:
        for k in ("maskmem_features", "obj_ptr"):
            if not _same_layout(ref[k], w[k]):
                raise ValueError("candidate write layout mismatch: " + k)
        if not _equal(ref["maskmem_pos_enc"], w["maskmem_pos_enc"], torch):
            raise ValueError("candidate spatial position metadata differs")
    identical = all(all(_equal(ref[k], w[k], torch) for k in
                       ("maskmem_features", "obj_ptr")) for w in writes[1:])
    fast = bool(identity_fastpath and identical)
    out = {"maskmem_pos_enc": _clone(ref["maskmem_pos_enc"])}
    for k in ("maskmem_features", "obj_ptr"):
        out[k] = (ref[k].clone() if fast else
                  torch.stack([w[k].float() for w in writes]).mean(0).to(ref[k].dtype))
        _tensor(out[k], torch, "mean " + k)
    return out, _info("early_stored_write_mean", identical, fast)


def _validate_contexts(contexts, torch):
    _three(contexts)
    for c in contexts:
        if isinstance(c["curr"], tuple) or isinstance(c["curr_pos"], tuple):
            raise ValueError("native current levels must be lists, not tuples")
        if isinstance(c["curr"], list) != isinstance(c["curr_pos"], list):
            raise ValueError("current and position container types differ")
        for k in ("memory", "memory_pos"):
            _tensor(c[k], torch, k, 3)
        if c["memory"].shape != c["memory_pos"].shape or c["memory"].device != c["memory_pos"].device:
            raise ValueError("memory position shape/device mismatch")
        n = c["num_obj_ptr_tokens"]
        if type(n) is not int or not 0 <= n < c["memory"].shape[0]:
            raise ValueError("pointer-tail count must leave spatial tokens")
        for k in ("curr", "curr_pos"):
            tensors = c[k] if isinstance(c[k], (list, tuple)) else [c[k]]
            if len(tensors) != 1:
                raise ValueError("native attention accepts one current level")
            _tensor(tensors[0], torch, k, 3)
            if tensors[0].shape[1] != c["memory"].shape[1] or tensors[0].device != c["memory"].device:
                raise ValueError("current/memory batch or device mismatch")
        current = c["curr"][0] if isinstance(c["curr"], (list, tuple)) else c["curr"]
        position = c["curr_pos"][0] if isinstance(c["curr_pos"], (list, tuple)) else c["curr_pos"]
        if current.shape != position.shape:
            raise ValueError("current position shape mismatch")
    ref = contexts[0]
    for c in contexts[1:]:
        for k in ("curr", "curr_pos", "memory_pos"):
            if not _equal(ref[k], c[k], torch):
                raise ValueError("complete contexts differ in common " + k)
        if (not _same_layout(ref["memory"], c["memory"])
                or ref["num_obj_ptr_tokens"] != c["num_obj_ptr_tokens"]):
            raise ValueError("complete context layout/pointer count mismatch")
    return all(_equal(ref["memory"], c["memory"], torch) for c in contexts[1:])


def contexts_identical(contexts, *, torch=None):
    """Exact full-context identity; permits the caller's shared native fast path."""
    return bool(_validate_contexts(contexts, _torch(torch)))


def flatten_contexts(contexts, *, torch=None, identity_fastpath=False):
    """Repeat three COMPLETE contexts: all spatial prefixes, then pointer tails.

    Equal repeated complete contexts have equal attention mass mathematically;
    floating-point bit parity requires the separately recorded identity fast path.
    """
    torch = _torch(torch)
    identical = _validate_contexts(contexts, torch)
    fast = bool(identity_fastpath and identical)
    ref = contexts[0]
    out = {k: _clone(ref[k]) for k in ("curr", "curr_pos")}
    n = ref["num_obj_ptr_tokens"]
    if fast:
        out.update(memory=ref["memory"].clone(), memory_pos=ref["memory_pos"].clone(),
                   num_obj_ptr_tokens=n)
    else:
        for k in ("memory", "memory_pos"):
            spatial = [c[k][:-n] if n else c[k] for c in contexts]
            tails = [c[k][-n:] for c in contexts] if n else []
            out[k] = torch.cat(spatial + tails, dim=0)
        out["num_obj_ptr_tokens"] = sum(c["num_obj_ptr_tokens"] for c in contexts)
    return out, _info("flat_complete_histories", identical, fast,
                      spatial_tokens=int(out["memory"].shape[0] - out["num_obj_ptr_tokens"]),
                      pointer_tokens=out["num_obj_ptr_tokens"])


def mean_stack_outputs(outputs, *, torch=None, identity_fastpath=False):
    """Uniform float32 mean AFTER each complete unchanged native stack."""
    torch = _torch(torch)
    _three(outputs)
    for x in outputs:
        _tensor(x, torch, "stack output", 3)
        if not _same_layout(outputs[0], x):
            raise ValueError("stack output layout mismatch")
    identical = all(torch.equal(outputs[0], x) for x in outputs[1:])
    fast = bool(identity_fastpath and identical)
    result = (outputs[0].float().clone() if fast else
              torch.stack([x.float() for x in outputs]).mean(0))
    _tensor(result, torch, "mean stack output", 3)
    return result, _info("late_complete_stack_mean", identical, fast)
