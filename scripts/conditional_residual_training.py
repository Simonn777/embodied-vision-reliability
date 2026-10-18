"""Unfrozen CPU-only conditional-residual development runner.

Original implementation, not an official R2R reproduction. The public fit_one
interface fixes the requested scientific dimensions and budgets. It consumes
only a noisy image and a supplied covariance, never a clean image or metrics.
Private patch helpers permit tiny interface tests without research runs.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

import numpy as np

METHODS = ("blindspot", "residual", "r2r_alpha05", "r2r_alpha1", "safe_blindspot")
CENTER = 24
NEIGHBORS = tuple(i for i in range(49) if i != CENTER)
PRODUCTION_STEPS = 500
PRODUCTION_BATCH = 512
PRODUCTION_LR = 1e-3
PRODUCTION_FIT_SECONDS = 90.0
PRODUCTION_CHUNK = 1024
PRODUCTION_MC = 50
_INTEROP_CONFIGURED = False


def _torch_runtime():
    # Importing this file needs only NumPy; CI without Torch can test validation.
    import torch

    global _INTEROP_CONFIGURED
    torch.set_num_threads(4)
    if not _INTEROP_CONFIGURED:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        _INTEROP_CONFIGURED = True
    torch.use_deterministic_algorithms(True)
    return torch


def _seed(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not 0 <= int(value) < 2**63:
        raise ValueError(name + " must be in [0, 2**63)")
    return int(value)


def _array_digest(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(json.dumps({"shape": list(a.shape), "dtype": a.dtype.str},
                        sort_keys=True, separators=(",", ":")).encode("ascii"))
    h.update(b"\x00")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _validate_covariance(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(covariance)
    if c.shape != (49, 49) or c.dtype != np.dtype("float64"):
        raise ValueError("covariance must be float64 [49,49] in row-major patch order")
    if not np.isfinite(c).all() or c[CENTER, CENTER] <= 0:
        raise ValueError("covariance must be finite with positive center variance")
    scale = float(np.max(np.abs(c)))
    if float(np.max(np.abs(c - c.T))) > 64 * np.finfo(np.float64).eps * scale:
        raise ValueError("covariance must be symmetric; no automatic repair")
    try:
        factor64 = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("R2R requires positive-definite covariance; no jitter repair") from exc
    factor = factor64.astype(np.float32)
    if not np.isfinite(factor).all():
        raise ValueError("covariance factor cannot be represented as float32")
    return np.array(c, copy=True, order="C"), factor


def _prepare_image(noisy: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    y = np.asarray(noisy)
    if y.dtype != np.dtype("float32") or y.shape != (128, 128):
        raise ValueError("noisy must be a float32 grayscale [128,128] array")
    if not np.isfinite(y).all():
        raise ValueError("noisy must be finite; no clipping or normalization is applied")
    c, factor = _validate_covariance(covariance)
    patches = np.lib.stride_tricks.sliding_window_view(y, (7, 7))
    patches = np.array(patches.reshape(122 * 122, 49), dtype=np.float32, order="C", copy=True)
    gains = (c[:, CENTER] / c[CENTER, CENTER]).astype(np.float32)
    if not np.isfinite(gains).all():
        raise ValueError("residual gains cannot be represented as float32")
    factor64_rounded = factor.astype(np.float64)
    # Explicit small reductions avoid backend-specific BLAS warning-state issues.
    effective = np.sum(factor64_rounded[:, None, :] * factor64_rounded[None, :, :], axis=2)
    return {
        "patches": patches,
        "gains": gains,
        # Exact finite-kernel oracle zeros only; no threshold is estimated.
        "correlated_mask": c[:, CENTER] != 0.0,
        "factor": factor,
        "noisy_sha256": _array_digest(y),
        "covariance_sha256": _array_digest(c),
        "factor_sha256": _array_digest(factor),
        "factor_max_absolute_covariance_error": float(np.max(np.abs(effective - c))),
    }


def _fixed_features(patches: np.ndarray, gains: np.ndarray, method: str,
                    correlated_mask: np.ndarray | None = None) -> np.ndarray:
    if method not in METHODS:
        raise ValueError("unknown method")
    p = np.asarray(patches)
    if p.ndim != 2 or p.shape[1] != 49 or p.dtype != np.dtype("float32"):
        raise ValueError("patches must be float32 [N,49]")
    if not np.isfinite(p).all():
        raise ValueError("patches must be finite")
    features = p.copy()
    if method == "residual":
        g = np.asarray(gains)
        if g.shape != (49,) or g.dtype != np.dtype("float32") or not np.isfinite(g).all():
            raise ValueError("gains must be finite float32 [49]")
        # Every row is centered on that row's SAME y_i. No cross-row operation.
        features -= p[:, CENTER, None] * g[None, :]
    if method in ("blindspot", "residual"):
        features[:, CENTER] = 0.0
    if method == "safe_blindspot":
        mask = np.asarray(correlated_mask)
        if mask.shape != (49,) or mask.dtype != np.dtype("bool") or not mask[CENTER]:
            raise ValueError("safe_blindspot requires exact C[:,24]!=0 boolean mask including center")
        features[:, mask] = 0.0
    if not np.isfinite(features).all():
        raise ValueError("features overflowed float32")
    return features


def _new_model(torch, init_seed: int):
    # Preserve the caller's CPU RNG while obeying torch.manual_seed initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(49, 64, device="cpu", dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32, device="cpu", dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1, device="cpu", dtype=torch.float32),
        )
    return model


def state_dict_sha256(state_dict_np: dict[str, np.ndarray]) -> str:
    """Hash parameter names/shapes and little-endian float32 bytes, not a ZIP."""
    h = hashlib.sha256(b"evr-mlp-state-v1\x00")
    for name, value in sorted(state_dict_np.items()):
        if not isinstance(name, str) or np.asarray(value).dtype != np.dtype("float32"):
            raise ValueError("state entries must have string names and float32 arrays")
        array = np.ascontiguousarray(value).astype("<f4", copy=False)
        header = json.dumps({"name": name, "shape": list(array.shape), "dtype": "<f4"},
                            sort_keys=True, separators=(",", ":")).encode("ascii")
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        payload = array.tobytes(order="C")
        h.update(len(payload).to_bytes(8, "little"))
        h.update(payload)
    return h.hexdigest()


def _state_digest(model) -> str:
    return state_dict_sha256({name: value.detach().cpu().contiguous().numpy()
                              for name, value in model.state_dict().items()})


def _alpha(method: str) -> float | None:
    return {"r2r_alpha05": 0.5, "r2r_alpha1": 1.0}.get(method)


def _fit_patches(torch, patches, features, factor, method, *, init_seed, batch_seed,
                 augment_seed, steps, batch_size, learning_rate, max_fit_seconds,
                 clock: Callable[[], float] = time.monotonic):
    """Private interface fixture hook. Public fit_one always requests 500 steps."""
    started = clock()
    model = _new_model(torch, init_seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                                 amsgrad=False, foreach=False)
    batch_rng = torch.Generator(device="cpu").manual_seed(batch_seed)
    augmentation_rng = torch.Generator(device="cpu").manual_seed(augment_seed)
    batch_digest = hashlib.sha256()
    augmentation_digest = hashlib.sha256()
    record = {
        "requested_steps": steps, "completed_steps": 0,
        "initial_weight_sha256": _state_digest(model),
        "loss_every_50": [], "last_loss": None, "failure_reason": None,
        "augmentation_standard_normal_draws": 0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    alpha = _alpha(method)
    for step in range(1, steps + 1):
        if clock() - started >= max_fit_seconds:
            record["failure_reason"] = "fit_time_cap"
            break
        indices = torch.randint(patches.shape[0], (batch_size,), generator=batch_rng,
                                device="cpu", dtype=torch.int64)
        batch_digest.update(indices.numpy().astype("<i8", copy=False).tobytes())
        inputs = features[indices]
        target = patches[indices, CENTER]
        if alpha is not None:
            epsilon = torch.randn((batch_size, 49), generator=augmentation_rng,
                                  dtype=torch.float32, device="cpu")
            augmentation_digest.update(epsilon.numpy().astype("<f4", copy=False).tobytes())
            record["augmentation_standard_normal_draws"] += epsilon.numel()
            noise = epsilon @ factor.T
            inputs = inputs + alpha * noise
            target = target - noise[:, CENTER] / alpha
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs).squeeze(-1), target,
                                           reduction="mean")
        if not bool(torch.isfinite(loss)):
            record["failure_reason"] = "nonfinite_loss"
            break
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
               for p in model.parameters()):
            record["failure_reason"] = "nonfinite_gradient"
            break
        optimizer.step()
        record["completed_steps"] = step
        record["last_loss"] = float(loss.detach())
        if step % 50 == 0:
            record["loss_every_50"].append({"step": step, "mean_batch_loss": record["last_loss"]})
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            record["failure_reason"] = "nonfinite_weight"
            break
        if clock() - started >= max_fit_seconds:
            # Even an over-budget 500th step is not reported as a successful fit.
            record["failure_reason"] = "fit_time_cap"
            break
    record["weight_sha256"] = _state_digest(model)
    record["batch_indices_sha256"] = batch_digest.hexdigest()
    record["augmentation_draws_sha256"] = augmentation_digest.hexdigest() if alpha is not None else None
    record["fit_seconds"] = float(clock() - started)
    if record["fit_seconds"] >= max_fit_seconds and record["failure_reason"] is None:
        record["failure_reason"] = "fit_time_cap"
    record["status"] = ("completed" if record["failure_reason"] is None
                        and record["completed_steps"] == steps else "failed")
    return model, record


def _predict(torch, model, features, factor, method, *, infer_seed, chunk_size,
             mc_samples, clock: Callable[[], float] = time.monotonic):
    """Pointwise patch inference: no spatial CNN or cross-row normalization."""
    model.eval()
    n = features.shape[0]
    alpha = _alpha(method)
    inference_rng = torch.Generator(device="cpu").manual_seed(infer_seed)
    primary = np.empty(n, dtype=np.float32)
    direct = np.empty(n, dtype=np.float32) if alpha is not None else None
    record = {"primary_seconds": 0.0, "direct_seconds": 0.0,
              "primary_forward_calls": 0, "direct_forward_calls": 0,
              "standard_normal_draws": 0, "mc_accumulation_dtype": "float64" if alpha else None}
    started = clock()
    with torch.inference_mode():
        for offset in range(0, n, chunk_size):
            inputs = features[offset:offset + chunk_size]
            if alpha is None:
                output = model(inputs).squeeze(-1)
                record["primary_forward_calls"] += 1
            else:
                accumulated = torch.zeros(inputs.shape[0], dtype=torch.float64, device="cpu")
                for _ in range(mc_samples):
                    epsilon = torch.randn(inputs.shape, generator=inference_rng,
                                          dtype=torch.float32, device="cpu")
                    noise = epsilon @ factor.T
                    accumulated += model(inputs + alpha * noise).squeeze(-1).to(torch.float64)
                    record["primary_forward_calls"] += 1
                    record["standard_normal_draws"] += epsilon.numel()
                output = (accumulated / mc_samples).to(torch.float32)
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("nonfinite primary prediction")
            primary[offset:offset + inputs.shape[0]] = output.numpy()
    record["primary_seconds"] = float(clock() - started)
    if alpha is not None:
        started = clock()
        with torch.inference_mode():
            for offset in range(0, n, chunk_size):
                inputs = features[offset:offset + chunk_size]
                output = model(inputs).squeeze(-1)
                record["direct_forward_calls"] += 1
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError("nonfinite direct prediction")
                direct[offset:offset + inputs.shape[0]] = output.numpy()
        record["direct_seconds"] = float(clock() - started)
    return primary, direct, record


def fit_one(noisy: np.ndarray, covariance: np.ndarray, method: str, *,
            init_seed: int, batch_seed: int, augment_seed: int, infer_seed: int,
            steps: int = 500, batch_size: int = 512, learning_rate: float = 1e-3,
            max_fit_seconds: float = 90.0, inference_chunk_size: int = 1024,
            mc_samples: int = 50) -> dict[str, Any]:
    """Return status/prediction/extras/receipt/state_dict_np; maps are [122,122].

    A failed fit returns no predictions. Unsupported input/specification raises
    ValueError. The caller must enforce an outer process deadline/memory limit,
    save local predictions, and independently score them. No files are written.
    """
    total_started = time.monotonic()
    if method not in METHODS:
        raise ValueError("method must be one of " + repr(METHODS))
    for name, value in (("steps", steps), ("batch_size", batch_size),
                        ("inference_chunk_size", inference_chunk_size), ("mc_samples", mc_samples)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    fixed = (("steps", steps, 500), ("batch_size", batch_size, 512),
             ("learning_rate", learning_rate, 1e-3), ("max_fit_seconds", max_fit_seconds, 90.0),
             ("inference_chunk_size", inference_chunk_size, 1024), ("mc_samples", mc_samples, 50))
    if any(isinstance(actual, (bool, np.bool_)) or actual != required
           for _, actual, required in fixed):
        raise ValueError("public fit_one requires exactly the frozen 500/512/1e-3/90s/1024/T50 settings")
    steps, batch_size = int(steps), int(batch_size)
    inference_chunk_size, mc_samples = int(inference_chunk_size), int(mc_samples)
    learning_rate, max_fit_seconds = float(learning_rate), float(max_fit_seconds)
    seeds = {name: _seed(value, name) for name, value in
             (("init_seed", init_seed), ("batch_seed", batch_seed),
              ("augment_seed", augment_seed), ("infer_seed", infer_seed))}
    prepared = _prepare_image(noisy, covariance)
    features_np = _fixed_features(prepared["patches"], prepared["gains"], method,
                                 prepared["correlated_mask"])
    torch = _torch_runtime()
    patches = torch.from_numpy(prepared["patches"])
    features = torch.from_numpy(features_np)
    factor = torch.from_numpy(prepared["factor"])
    receipt = {
        "schema_version": 1, "scope": "development-adapted-pointwise-MLP",
        "method": method, "alpha": _alpha(method), "seeds": seeds,
        "image_shape": [128, 128], "valid_shape": [122, 122], "patch_shape": [7, 7],
        "center_index": CENTER, "feature_count": 49, "architecture": [49, 64, 32, 1],
        "activation": "ReLU", "device": "cpu", "dtype": "float32",
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "optimizer": {"name": "Adam", "learning_rate": learning_rate,
                      "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False},
        "batch_size": batch_size, "sampling": "with-replacement-local-CPU-generator",
        "fit_time_cap_seconds": max_fit_seconds, "inference_chunk_size": inference_chunk_size,
        "mc_samples": mc_samples if _alpha(method) is not None else 1,
        "normalization": "none", "training_clipping": "none", "prediction_clipping": "none",
        "weight_hash_schema": "evr-mlp-state-v1", "clean_input_accessed": False,
        "data_shape": list(prepared["patches"].shape),
        "safe_blindspot_zero_feature_indices": (np.flatnonzero(prepared["correlated_mask"]).tolist()
                                                 if method == "safe_blindspot" else None),
        **{key: prepared[key] for key in ("noisy_sha256", "covariance_sha256", "factor_sha256",
                                        "factor_max_absolute_covariance_error")},
    }
    receipt["preparation_seconds"] = float(time.monotonic() - total_started)
    model, fit = _fit_patches(torch, patches, features, factor, method,
                             init_seed=seeds["init_seed"], batch_seed=seeds["batch_seed"],
                             augment_seed=seeds["augment_seed"], steps=steps, batch_size=batch_size,
                             learning_rate=learning_rate, max_fit_seconds=max_fit_seconds)
    receipt["fit"] = fit
    receipt["inference"] = None
    result = {"status": fit["status"], "prediction": None, "extras": {},
              "receipt": receipt, "state_dict_np": None}
    if fit["status"] == "completed":
        inference_started = time.monotonic()
        try:
            primary, direct, infer = _predict(torch, model, features, factor, method,
                                            infer_seed=seeds["infer_seed"],
                                            chunk_size=inference_chunk_size, mc_samples=mc_samples)
        except FloatingPointError as exc:
            result["status"] = "failed"
            receipt["inference"] = {"failure_reason": str(exc),
                                    "total_seconds": float(time.monotonic() - inference_started)}
        else:
            infer["total_seconds"] = float(time.monotonic() - inference_started)
            receipt["inference"] = infer
            postprocess_started = time.monotonic()
            result["prediction"] = primary.reshape(122, 122)
            receipt["prediction_sha256"] = _array_digest(result["prediction"])
            receipt["finite_output"] = bool(np.isfinite(primary).all())
            if direct is not None:
                result["extras"]["direct_prediction"] = direct.reshape(122, 122)
                receipt["direct_prediction_sha256"] = _array_digest(result["extras"]["direct_prediction"])
                receipt["finite_direct_output"] = bool(np.isfinite(direct).all())
            receipt["postprocess_seconds"] = float(time.monotonic() - postprocess_started)
    export_started = time.monotonic()
    result["state_dict_np"] = {
        name: value.detach().cpu().contiguous().numpy().astype(np.float32, copy=True)
        for name, value in model.state_dict().items()
    }
    receipt["weight_export_seconds"] = float(time.monotonic() - export_started)
    receipt["status"] = result["status"]
    receipt["total_seconds"] = float(time.monotonic() - total_started)
    # Safety check: receipts must be JSON-safe, including failure records.
    json.dumps(receipt, allow_nan=False)
    return result
