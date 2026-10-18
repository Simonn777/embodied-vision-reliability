"""Prospective 0007 CPU gradient measurement. No data I/O or training on import.

Only NumPy is required for input/geometry checks. Torch is imported lazily.
This module never creates an optimizer, updates parameters, or reads clean data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

METHODS = ("independent", "random_antithetic", "mean_antithetic", "same_patch")
REGIMES = ("A_fixed", "B_redraw")
PARAMETER_NAMES = ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
CENTER = 24


class DeadlineExceeded(RuntimeError):
    pass


class _Deadline:
    def __init__(self, seconds, clock=time.monotonic):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("deadline must be finite and positive")
        self.clock, self.started, self.seconds = clock, clock(), float(seconds)

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise DeadlineExceeded("cooperative case deadline reached; external hard timeout remains required")


def array_sha256(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({"shape": list(a.shape), "dtype": a.dtype.str}, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()


def state_dict_sha256(state):
    """Identical evr-mlp-state-v1 encoding to saved 0006 weight content."""
    h = hashlib.sha256(b"evr-mlp-state-v1\0")
    for name, value in sorted(state.items()):
        a = np.asarray(value)
        if not isinstance(name, str) or a.dtype != np.dtype("float32"):
            raise ValueError("weight names and float32 arrays required")
        a = np.ascontiguousarray(a).astype("<f4", copy=False)
        header = json.dumps({"name": name, "shape": list(a.shape), "dtype": "<f4"}, sort_keys=True, separators=(",", ":")).encode("ascii")
        payload = a.tobytes()
        h.update(len(header).to_bytes(8, "little")); h.update(header)
        h.update(len(payload).to_bytes(8, "little")); h.update(payload)
    return h.hexdigest()


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) < high:
        raise ValueError(name + " is outside its integer range")
    return int(value)


def _validate_weights(state, widths=(49, 64, 32, 1)):
    if set(state) != set(PARAMETER_NAMES):
        raise ValueError("exact six Sequential state_dict names required")
    result = {}
    for layer, left, right in zip((0, 2, 4), widths[:-1], widths[1:]):
        for suffix, shape in (("weight", (right, left)), ("bias", (right,))):
            key = str(layer) + "." + suffix
            a = np.asarray(state[key])
            if a.dtype != np.dtype("float32") or a.shape != shape or not np.isfinite(a).all():
                raise ValueError("invalid weight tensor: " + key)
            result[key] = np.array(a, copy=True, order="C")
    return result


def validate_covariance(covariance):
    c = np.asarray(covariance)
    if c.dtype != np.dtype("float64") or c.shape != (49, 49) or not np.isfinite(c).all() or c[CENTER, CENTER] <= 0:
        raise ValueError("finite float64 [49,49] covariance with positive center variance required")
    scale = float(np.max(np.abs(c)))
    if np.max(np.abs(c - c.T)) > 64 * np.finfo(np.float64).eps * scale:
        raise ValueError("asymmetric covariance; no repair")
    try:
        factor = np.linalg.cholesky(c).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise ValueError("positive-definite covariance required; no jitter") from exc
    if not np.isfinite(factor).all():
        raise ValueError("factor overflow")
    c = np.array(c, copy=True, order="C")
    f64 = factor.astype(np.float64)
    effective = np.sum(f64[:, None, :] * f64[None, :, :], axis=2)
    return c, factor, float(np.max(np.abs(effective - c)))


def extract_patches(noisy):
    y = np.asarray(noisy)
    if y.dtype != np.dtype("float32") or y.shape != (128, 128) or not np.isfinite(y).all():
        raise ValueError("finite noisy float32 [128,128] required")
    return np.array(np.lib.stride_tricks.sliding_window_view(y, (7, 7)).reshape(-1, 49), dtype=np.float32, order="C", copy=True)


def mean_pair_order(rows):
    """Stable occurrence-slot ties; no normalization, target labels, or noise draws."""
    a = np.asarray(rows)
    if a.ndim != 2 or a.shape[1] != 49 or a.shape[0] % 2 or a.dtype != np.dtype("float32") or not np.isfinite(a).all():
        raise ValueError("finite even-row float32 [B,49] required")
    return np.argsort(np.mean(a, axis=1, dtype=np.float64), kind="stable").astype(np.int64)


def _torch_runtime():
    for name in THREAD_ENV:
        os.environ[name] = "4"
    import torch
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def _new_model(torch, state, widths):
    # Constructor initialization is discarded without changing the global CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(widths[0], widths[1], device="cpu", dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(widths[1], widths[2], device="cpu", dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(widths[2], widths[3], device="cpu", dtype=torch.float32),
        )
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()}, strict=True)
    model.eval()
    return model


def _export_state(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def seed_plan(torch, case_index, regime_index, replicates):
    """Master overhead is outside step timers and separately recorded by caller."""
    base = 700000 + 10000 * case_index + 1000 * regime_index
    result = {"master_seeds": {"indices": base, "augmentation": base + 1, "pairing": base + 2}}
    for stream, name in enumerate(("indices", "augmentation", "pairing")):
        gen = torch.Generator(device="cpu").manual_seed(base + stream)
        result[name] = torch.randint(0, 2**63 - 1, (replicates,), generator=gen, dtype=torch.int64).tolist()
    if regime_index == 0:
        result["fixed_index_seed"] = result["indices"][0]
    return result


def _estimate_gradient(torch, model, patches, factor, *, method, batch_size, index_seed,
                       augmentation_seed, pairing_seed, fixed_indices=None,
                       deadline=None, capture_debug=False, clock=time.monotonic):
    """One B-row forward/backward; ALL estimator-specific work is timed."""
    if method not in METHODS or batch_size <= 0 or batch_size % 2:
        raise ValueError("unknown estimator or invalid even batch size")
    if deadline is not None:
        deadline.check()
    started = clock()
    gen_i = torch.Generator(device="cpu").manual_seed(index_seed)
    # Regime B draws B even for same_patch. Regime A copies its fixed setup batch.
    indices = (torch.randint(patches.shape[0], (batch_size,), generator=gen_i, dtype=torch.int64)
               if fixed_indices is None else fixed_indices.clone())
    rows = patches[indices]
    used_indices = indices
    order = torch.arange(batch_size, dtype=torch.int64)
    if method == "same_patch":
        rows = rows[:batch_size // 2].repeat_interleave(2, dim=0)
        used_indices = indices[:batch_size // 2].repeat_interleave(2)
    elif method == "random_antithetic":
        gen_p = torch.Generator(device="cpu").manual_seed(pairing_seed)
        order = torch.randperm(batch_size, generator=gen_p)
        rows = rows[order]; used_indices = indices[order]
    elif method == "mean_antithetic":
        order = torch.from_numpy(mean_pair_order(rows.numpy()))
        rows = rows[order]; used_indices = indices[order]
    gen_a = torch.Generator(device="cpu").manual_seed(augmentation_seed)
    draw_rows = batch_size if method == "independent" else batch_size // 2
    epsilon = torch.randn((draw_rows, 49), generator=gen_a, dtype=torch.float32, device="cpu")
    noise = epsilon @ factor.T
    if method != "independent":
        noise = torch.stack((noise, -noise), dim=1).reshape(batch_size, 49)
    inputs = rows + 0.5 * noise
    target = rows[:, CENTER] - 2.0 * noise[:, CENTER]
    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs).squeeze(-1), target, reduction="mean")
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite loss")
    loss.backward()
    parameters = list(model.named_parameters())
    if tuple(name for name, _ in parameters) != PARAMETER_NAMES:
        raise RuntimeError("parameter flatten order changed")
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _, p in parameters):
        raise FloatingPointError("missing or nonfinite gradient")
    gradient = torch.cat([p.grad.reshape(-1) for _, p in parameters]).detach().clone().numpy().copy()
    loss_value = float(loss.detach())
    wall = float(clock() - started)
    # A completed gradient is returned even if this step exceeded the deadline.
    # The caller stores it, then checks the deadline before doing another step.
    # Hashing is audit work, explicitly outside estimator wall time.
    audit_started = clock()
    record = {"status": "completed", "method": method, "wall_seconds": wall, "loss": loss_value,
              "index_seed": index_seed, "augmentation_seed": augmentation_seed, "pairing_seed": pairing_seed,
              "batch_occurrences": batch_size, "gaussian_rows": draw_rows,
              "base_indices_sha256": array_sha256(indices.numpy()), "used_indices_sha256": array_sha256(used_indices.numpy()),
              "pair_order_sha256": array_sha256(order.numpy()), "epsilon_sha256": array_sha256(epsilon.numpy()),
              "gradient_sha256": array_sha256(gradient), "forward_calls": 1, "backward_calls": 1}
    record["audit_seconds"] = float(clock() - audit_started)
    if capture_debug:
        record["debug"] = {"base_indices": indices.numpy().copy(), "used_indices": used_indices.numpy().copy(),
                           "order": order.numpy().copy(), "epsilon": epsilon.numpy().copy(),
                           "noise": noise.numpy().copy(), "rows": rows.numpy().copy()}
    return gradient, record


def _run_patches(patches, covariance, weights, *, case_index, replicates, batch_size,
                 widths=(49, 64, 32, 1), deadline_seconds=90.0, clock=time.monotonic,
                 extra_setup=None):
    """Private artificial-fixture interface. Production uses run_case below."""
    deadline = _Deadline(deadline_seconds, clock)
    start = deadline.started
    case_index = _integer(case_index, "case_index", 0, 8)
    replicates = _integer(replicates, "replicates", 1, 65)
    batch_size = _integer(batch_size, "batch_size", 2, 513)
    if batch_size % 2 or widths[0] != 49 or widths[-1] != 1:
        raise ValueError("even batch and 49-input scalar-output MLP required")
    p = np.asarray(patches)
    if p.ndim != 2 or p.shape[1] != 49 or p.shape[0] == 0 or p.dtype != np.dtype("float32") or not np.isfinite(p).all():
        raise ValueError("finite float32 nonempty [N,49] patches required")
    c, f, factor_error = validate_covariance(covariance)
    state = _validate_weights(weights, widths)
    torch = _torch_runtime()
    model = _new_model(torch, state, widths)
    p_tensor = torch.from_numpy(np.array(p, copy=True, order="C"))
    factor = torch.from_numpy(f)
    count = sum(v.size for v in state.values())
    before_sha = state_dict_sha256(_export_state(model))
    gradients = np.full((2, 4, replicates, count), np.nan, dtype=np.float32)
    walls = np.full((2, 4, replicates), np.nan, dtype=np.float64)
    losses = np.full_like(walls, np.nan)
    completed = np.zeros(walls.shape, dtype=np.bool_)
    records, plans, order_log = [], [], []
    receipt = {"schema": "0007-gradient-core-v1", "case_index": case_index, "regimes": list(REGIMES), "methods": list(METHODS),
               "replicates": replicates, "batch_size": batch_size, "sampling": "with-replacement-occurrences",
               "alpha": 0.5, "objective": "unhalved mean squared R2R loss", "parameter_names": list(PARAMETER_NAMES),
               "parameter_count": count, "architecture": list(widths), "optimizer_steps": 0,
               "torch_version": str(torch.__version__), "numpy_version": np.__version__, "device": "cpu",
               "numeric_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
               "patches_sha256": array_sha256(p), "covariance_sha256": array_sha256(c), "factor_sha256": array_sha256(f),
               "factor_max_absolute_covariance_error": factor_error, "weight_before_sha256": before_sha,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "case_deadline_seconds": deadline_seconds, "deadline_kind": "cooperative; requires external hard timeout",
               "shared_setup_seconds": float(clock() - start), "master_rng_seconds": 0.0, "fixed_index_setup_seconds": 0.0,
               "A_variance_includes_random_pairing": True, "clean_reference_accessed": False,
               "extra_setup": extra_setup, "status": "completed", "failure": None}
    attempted = None
    try:
        deadline.check()
        for regime in range(2):
            before = clock(); plan = seed_plan(torch, case_index, regime, replicates)
            receipt["master_rng_seconds"] += float(clock() - before); plans.append(plan)
            fixed = None
            if regime == 0:
                before = clock()
                fixed = torch.randint(p_tensor.shape[0], (batch_size,), generator=torch.Generator(device="cpu").manual_seed(plan["fixed_index_seed"]), dtype=torch.int64)
                receipt["fixed_index_setup_seconds"] += float(clock() - before)
            for rep in range(replicates):
                rotation = (case_index + rep) % 4
                order = list(range(rotation, 4)) + list(range(rotation))
                order_log.append({"regime": regime, "replicate": rep, "method_indices": order})
                for method_index in order:
                    attempted = {"regime": regime, "replicate": rep, "method_index": method_index}
                    gradient, record = _estimate_gradient(torch, model, p_tensor, factor,
                        method=METHODS[method_index], batch_size=batch_size,
                        index_seed=(plan["fixed_index_seed"] if regime == 0 else plan["indices"][rep]),
                        augmentation_seed=plan["augmentation"][rep], pairing_seed=plan["pairing"][rep],
                        fixed_indices=fixed, deadline=deadline, clock=clock)
                    gradients[regime, method_index, rep] = gradient
                    walls[regime, method_index, rep] = record["wall_seconds"]
                    losses[regime, method_index, rep] = record["loss"]
                    completed[regime, method_index, rep] = True
                    record.update(attempted); records.append(record)
                    deadline.check()
    except (DeadlineExceeded, FloatingPointError, RuntimeError) as exc:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc), "attempted": attempted}
    after_sha = state_dict_sha256(_export_state(model))
    receipt.update({"weight_after_sha256": after_sha, "weight_unchanged": after_sha == before_sha,
                    "seed_plans": plans, "execution_order": order_log, "records": records,
                    "completed_measurements": int(completed.sum()), "expected_measurements": 2 * 4 * replicates,
                    "gradient_array_shape": list(gradients.shape),
                    "gradient_array_sha256": array_sha256(gradients), "completed_mask_sha256": array_sha256(completed),
                    "walls_sha256": array_sha256(walls), "losses_sha256": array_sha256(losses),
                    "estimator_wall_seconds": float(sum(record["wall_seconds"] for record in records)),
                    "per_measurement_audit_seconds": float(sum(record["audit_seconds"] for record in records))})
    if not receipt["weight_unchanged"]:
        receipt["status"] = "failed"; receipt["failure"] = {"type": "WeightMutation", "message": "saved parameters changed"}
    if receipt["status"] == "completed":
        try:
            deadline.check()
        except DeadlineExceeded as exc:
            receipt["status"] = "failed"
            receipt["failure"] = {"type": type(exc).__name__, "message": str(exc), "attempted": attempted}
    receipt["total_seconds"] = float(clock() - start)
    json.dumps(receipt, allow_nan=False)
    return {"gradients": gradients, "walls": walls, "losses": losses, "completed": completed, "receipt": receipt}


def run_case(noisy, covariance, weights, *, case_index):
    """Production: two regimes × four estimators ×64, B512, alpha.5; no file I/O.

    Return float32 gradients [2,4,64,5313], float64 walls/losses, bool completion
    mask and JSON-safe receipt. Caller must persist arrays/receipt and enforce an
    independent hard timeout. Failed/incomplete measurements are never successful.
    """
    started = time.monotonic()
    patches = extract_patches(noisy)
    setup = {"noisy_sha256": array_sha256(np.asarray(noisy)), "patch_extraction_seconds": time.monotonic() - started}
    remaining = 90.0 - (time.monotonic() - started)
    if remaining <= 0:
        raise DeadlineExceeded("production input preparation exhausted cooperative case budget")
    result = _run_patches(patches, covariance, weights, case_index=case_index, replicates=64,
                         batch_size=512, deadline_seconds=remaining, extra_setup=setup)
    result["receipt"]["production_entry_total_seconds"] = time.monotonic() - started
    result["receipt"]["production_case_deadline_seconds"] = 90.0
    return result
