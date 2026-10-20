"""Fixed ordinary CPU classifier for the D37 local comparison.

No file access, model downloads, normalization, label renewal, or native fallback.
The caller owns feature meanings, fold isolation, thread limits, and execution.
"""

import hashlib
import math


# Admission tolerance only, not an adjustment to the optimization weights.
# 16 * np.finfo(np.float32).eps, written without importing NumPy at module load.
WEIGHT_SUM_ROUNDOFF_TOLERANCE = 16 * 2.0 ** -23

CONFIG = {
    "dtype": "float32",
    "device": "cpu",
    "hidden": 32,
    "activation": "relu",
    "epochs": 20,
    "batch_size": 8192,
    "seed": 0,
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "selection": "terminal_epoch_only",
    "weight_sum_roundoff_tolerance": WEIGHT_SUM_ROUNDOFF_TOLERANCE,
    "weight_renormalization": False,
}


class FitError(RuntimeError):
    """An invalid input or failed fit, with JSON-safe partial evidence."""

    def __init__(self, message, trace, model=None, code="fit_failed"):
        super().__init__(message)
        self.trace = trace
        self.model = model
        self.code = code


def _features(x):
    import numpy as np

    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] < 1 or x.dtype != np.float32:
        raise ValueError("x must be a float32 array of shape (N,D), D >= 1")
    if not np.isfinite(x).all():
        raise ValueError("x contains nonfinite values")
    return np.array(x, dtype=np.float32, order="C", copy=True)


def _training_arrays(x, y, w, active):
    import numpy as np

    x = _features(x)
    n = len(x)
    vectors = []
    for name, value in (("y", y), ("w", w)):
        value = np.asarray(value)
        if value.shape != (n,) or value.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be a real vector of length N")
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        vectors.append(value)
    y, w = vectors
    if np.any(y > 1):
        raise ValueError("y must lie in [0,1]")
    weight_sum = float(np.sum(w, dtype=np.float64))
    if weight_sum > 1.0 + WEIGHT_SUM_ROUNDOFF_TOLERANCE:
        raise ValueError("sum(w) exceeds 1 plus the fixed roundoff tolerance; "
                         "weights are never renormalized")
    if active is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.asarray(active)
        if active.dtype != np.bool_ or active.shape != (n,):
            raise ValueError("active must be a boolean vector of length N")
        active = active.copy()
    y = np.array(y, dtype=np.float32, copy=True)
    w = np.array(w, dtype=np.float32, copy=True)
    w[~active] = 0
    return x, y, w, active, weight_sum


def _network(d):
    import torch

    # Restore the caller's CPU RNG state; initialization and shuffles are separate.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CONFIG["seed"])
        return torch.nn.Sequential(
            torch.nn.Linear(d, 32, device="cpu", dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1, device="cpu", dtype=torch.float32),
        )


def _state(network):
    import torch

    if not all(torch.isfinite(p).all().item() for p in network.parameters()):
        return None
    return {key: value.detach().cpu().tolist()
            for key, value in network.state_dict().items()}


def _model(d, kind, state, epoch):
    return {
        "schema": "evr0037.mlp.v1", "kind": kind, "input_dim": d,
        "config": dict(CONFIG), "state_dict": state, "terminal_epoch": epoch,
        "constant_probability": 1.0 if kind == "constant_keep" else None,
    }


def _objective(network, tx, ty, tw):
    import torch

    terms = []
    with torch.no_grad():
        for start in range(0, len(tx), CONFIG["batch_size"]):
            stop = start + CONFIG["batch_size"]
            logits = network(tx[start:stop]).squeeze(1)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, ty[start:stop], reduction="none")
            value = float((losses * tw[start:stop]).sum().item())
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite fixed-state weighted BCE")
            terms.append(value)
    return math.fsum(terms)


def fit(x, y, w, active=None):
    """Return (JSON model, JSON trace); effective weights are w * active.

    Every row stays in the seeded permutation, N, and minibatch schedule. Zero
    weight minibatches still take Adam steps, including its fixed weight decay.
    Only a globally zero effective weight returns constant_keep without fitting.
    Intended sum(w) <= 1 is admitted within 16 float32 epsilons; this tolerance
    never changes weights. The original sum and any admitted excess are recorded.
    """
    trace = {
        "schema": "evr0037.mlp-trace.v1", "status": "started",
        "config": dict(CONFIG), "epochs": [], "steps": 0, "batch_attempts": 0,
    }
    network = None
    d = None
    try:
        x, y, w, active, input_weight_sum = _training_arrays(x, y, w, active)
        import numpy as np

        n, d = x.shape
        effective_sum = float(np.sum(w, dtype=np.float64))
        trace.update({
            "n": n, "input_dim": d, "active_count": int(active.sum()),
            "positive_effective_weight_count": int(np.count_nonzero(w)),
            "input_weight_sum": input_weight_sum,
            "input_weight_excess_over_one": max(0.0, input_weight_sum - 1.0),
            "weight_sum_roundoff_admitted": input_weight_sum > 1.0,
            "effective_float32_weight_sum": effective_sum,
        })
        if effective_sum == 0:
            trace.update({
                "status": "completed", "reason": "no_positive_eligible_weight",
                "actual_fit": False, "completed_epochs": 0,
                "initial_weighted_bce": None, "terminal_weighted_bce": None,
            })
            return _model(d, "constant_keep", None, 0), trace

        import torch

        tx, ty, tw = (torch.from_numpy(a) for a in (x, y, w))
        network = _network(d)
        initial = _state(network)
        trace.update({
            "actual_fit": True, "parameter_count": sum(
                p.numel() for p in network.parameters()),
            "initial_state_dict": initial,
            "initial_weighted_bce": _objective(network, tx, ty, tw),
        })
        optimizer = torch.optim.Adam(
            network.parameters(), lr=CONFIG["learning_rate"],
            betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"],
            weight_decay=CONFIG["weight_decay"])
        generator = torch.Generator(device="cpu").manual_seed(CONFIG["seed"])
        for epoch in range(CONFIG["epochs"]):
            order = torch.randperm(n, generator=generator, device="cpu")
            online_terms, scaled_terms, gradient_norms = [], [], []
            epoch_steps = 0
            trace["current_epoch"] = epoch + 1
            for start in range(0, n, CONFIG["batch_size"]):
                ids = order[start:start + CONFIG["batch_size"]]
                trace["batch_attempts"] += 1
                optimizer.zero_grad(set_to_none=True)
                logits = network(tx[ids]).squeeze(1)
                per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, ty[ids], reduction="none")
                weighted_bce = (tw[ids] * per_row).sum()
                loss = weighted_bce * (n / len(ids))
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("nonfinite minibatch objective")
                trace["last_batch"] = {
                    "epoch": epoch + 1, "start": start, "size": len(ids),
                    "weighted_bce": float(weighted_bce.detach().item()),
                    "optimizer_loss": float(loss.detach().item()),
                }
                loss.backward()
                gradients = [p.grad for p in network.parameters()]
                if any(g is None or not torch.isfinite(g).all().item()
                       for g in gradients):
                    raise FloatingPointError("missing or nonfinite gradient")
                gradient_norm = math.sqrt(math.fsum(
                    float(g.double().square().sum().item()) for g in gradients))
                optimizer.step()
                trace["steps"] += 1
                epoch_steps += 1
                if any(not torch.isfinite(p).all().item()
                       for p in network.parameters()):
                    raise FloatingPointError("nonfinite parameters after Adam")
                online_terms.append(trace["last_batch"]["weighted_bce"])
                scaled_terms.append(trace["last_batch"]["optimizer_loss"])
                gradient_norms.append(gradient_norm)
            trace["epochs"].append({
                "epoch": epoch + 1, "steps": epoch_steps, "rows": n,
                "order_sha256": hashlib.sha256(
                    order.numpy().astype("<i8", copy=False).tobytes()).hexdigest(),
                # Online sums use successive parameter states, not final weights.
                "online_weighted_bce": math.fsum(online_terms),
                "mean_optimizer_loss": math.fsum(scaled_terms) / epoch_steps,
                "maximum_data_gradient_norm": max(gradient_norms),
            })
        terminal = _state(network)
        if terminal is None:
            raise FloatingPointError("nonfinite terminal state")
        trace.update({
            "status": "completed", "reason": "fixed_epochs_completed",
            "completed_epochs": CONFIG["epochs"],
            "terminal_weighted_bce": _objective(network, tx, ty, tw),
            "convergence_claim": False,
        })
        return _model(d, "mlp", terminal, CONFIG["epochs"]), trace
    except Exception as error:
        trace.update({"status": "failed",
                      "failure": {"type": type(error).__name__, "message": str(error)}})
        partial = None
        if network is not None:
            state = _state(network)
            if state is not None:
                partial = _model(d, "partial_mlp", state, len(trace["epochs"]))
        code = "invalid_input" if network is None else "fit_failed"
        raise FitError(str(error), trace, partial, code) from error


def predict(model, x):
    """Return float32 probabilities. A partial/failed model cannot be deployed."""
    import numpy as np

    x = _features(x)
    if (not isinstance(model, dict) or model.get("schema") != "evr0037.mlp.v1"
            or model.get("config") != CONFIG
            or type(model.get("input_dim")) is not int
            or model["input_dim"] != x.shape[1]):
        raise ValueError("incompatible serialized model")
    if model.get("kind") == "constant_keep":
        if (model.get("state_dict") is not None
                or model.get("terminal_epoch") != 0
                or model.get("constant_probability") != 1.0):
            raise ValueError("invalid constant_keep model")
        return np.ones(len(x), dtype=np.float32)
    if model.get("kind") != "mlp" or model.get("terminal_epoch") != CONFIG["epochs"]:
        raise ValueError("only terminal completed models may predict")
    import torch

    network = _network(x.shape[1])
    saved = model.get("state_dict")
    expected = network.state_dict()
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("invalid state keys")
    state = {}
    for key, template in expected.items():
        value = np.asarray(saved[key], dtype=np.float64)
        if (value.shape != tuple(template.shape) or not np.isfinite(value).all()
                or np.any(np.abs(value) > np.finfo(np.float32).max)):
            raise ValueError("invalid serialized parameter")
        state[key] = torch.tensor(value, dtype=torch.float32, device="cpu")
    network.load_state_dict(state, strict=True)
    answer = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), CONFIG["batch_size"]):
            stop = start + CONFIG["batch_size"]
            logits = network(torch.from_numpy(x[start:stop])).squeeze(1)
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("nonfinite prediction logits")
            answer[start:stop] = torch.sigmoid(logits).numpy()
    return answer
