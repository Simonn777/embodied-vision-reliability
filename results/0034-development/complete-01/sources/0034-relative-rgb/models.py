"""Fixed float64 pointwise readouts. No filesystem, native model or eager ML imports."""

import math

DIMENSIONS = {
    "logit": 3, "absolute": 50, "prototype": 5, "nn": 5,
    "ordinary_relations": 17, "bilateral": 4,
    "signed_no_gram": 4, "signed_gram": 14,
    "displaced_gram": 14, "rgb256": 14,
}
GATED = frozenset(("signed_no_gram", "signed_gram", "displaced_gram", "rgb256"))
MODEL_SCHEMA = "evr0034.pointwise-model.v1"
TRACE_SCHEMA = "evr0034.pointwise-fit-trace.v1"
LBFGS = {
    "lr": 1.0, "max_iter": 80, "max_eval": 400,
    "tolerance_grad": 1e-6, "tolerance_change": 1e-12,
    "history_size": 20, "line_search_fn": "strong_wolfe",
}
REGULARIZATION = 1e-3
WEIGHT_SUM_TOLERANCE = 16 * 2.220446049250313e-16


class FitError(RuntimeError):
    """Failure evidence is JSON-safe; partial parameters are never a fitted model."""

    def __init__(self, reason_code, message, partial_model, closure_trace):
        super().__init__(message)
        self.reason_code = reason_code
        self.partial_model = partial_model
        self.closure_trace = closure_trace


def _array(value, name, ndim):
    import numpy as np
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or raw.ndim != ndim:
        raise ValueError(f"{name}: expected a real numeric {ndim}-D array")
    array = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite value")
    return array


def _design(name, x, evidence):
    if name not in DIMENSIONS:
        raise ValueError(f"unknown model: {name!r}")
    array = _array(x, "x", 2)
    if array.shape[1] != DIMENSIONS[name]:
        raise ValueError(f"x: {name} requires {DIMENSIONS[name]} columns")
    if not (array[:, 0] == 1.0).all():
        raise ValueError("x: first column must be exactly one")
    if name in GATED:
        if evidence is None:
            raise ValueError("signed model requires evidence")
        e = _array(evidence, "evidence", 1)
        if len(e) != len(array) or (abs(e) > 1.0).any():
            raise ValueError("evidence: require N entries in [-1, 1]")
    else:
        if evidence is not None:
            raise ValueError("ordinary model does not consume separate evidence")
        e = None
    return array, e


def _delta(torch, name, design, coefficients, evidence):
    activation = design @ coefficients
    if not bool(torch.isfinite(activation).all()):
        raise ArithmeticError("nonfinite linear activation")
    if name in GATED:
        return 4.0 * evidence * torch.sigmoid(activation)
    return 4.0 * torch.tanh(activation)


def _finite_number(torch, value):
    return float(value.detach()) if bool(torch.isfinite(value).all()) else None


def _objective(torch, name, design, coefficients, evidence, logits, target, weights):
    delta = _delta(torch, name, design, coefficients, evidence)
    data = (weights * torch.nn.functional.binary_cross_entropy_with_logits(
        logits + delta, target, reduction="none")).sum()
    energy = (weights * delta.square()).sum()
    penalty = REGULARIZATION * energy
    return data + penalty, data, penalty, energy


def fit_model(name, x, native_logits, target, weights, evidence=None):
    """One terminal LBFGS fit. Returns (JSON model, JSON closure trace).

    Weights are used verbatim, including sums below one. Empty/zero-weight fits
    fail distinctly from numerical failures. Inputs are copied and never mutated.
    """
    model = {
        "schema": MODEL_SCHEMA, "name": name, "status": "preparing",
        "dtype": "float64", "input_dim": DIMENSIONS.get(name),
        "family": "signed_gate" if name in GATED else "ordinary",
        "coefficients": None, "initial_coefficients": None,
        "config": {"correction_bound": 4.0, "regularization": REGULARIZATION,
                   "gate_initial_intercept": -4.0, "lbfgs": dict(LBFGS)},
    }
    trace = {
        "schema": TRACE_SCHEMA, "name": name, "status": "preparing",
        "closures": [], "terminal_evaluations": 0,
        "configured_max_eval": LBFGS["max_eval"],
        "max_eval_is_hard_cap": False,
    }

    def fail(code, message):
        model.update(status="failed", reason_code=code, reason=str(message))
        trace.update(status="failed", reason_code=code, reason=str(message))
        if trace["closures"] and trace["closures"][-1]["status"] == "started":
            trace["closures"][-1].update(status="failed", reason_code=code)
        raise FitError(code, str(message), model, trace)

    try:
        design, signed = _design(name, x, evidence)
        logits = _array(native_logits, "native_logits", 1)
        labels = _array(target, "target", 1)
        sample_weights = _array(weights, "weights", 1)
        n = len(design)
        if any(len(v) != n for v in (logits, labels, sample_weights)):
            raise ValueError("x, logits, target and weights must have equal N")
        if not ((labels == 0.0) | (labels == 1.0)).all():
            raise ValueError("target must already be binary 0/1")
        if (sample_weights < 0.0).any():
            raise ValueError("negative training weight")
        weight_sum = math.fsum(float(v) for v in sample_weights)
        if weight_sum > 1.0 + WEIGHT_SUM_TOLERANCE:
            raise ValueError("weight sum exceeds one")
        model["training"] = {
            "sample_count": n, "positive_weight_count": int((sample_weights > 0).sum()),
            "weight_sum": weight_sum, "weights_renormalized": False,
            "weight_sum_roundoff_tolerance": WEIGHT_SUM_TOLERANCE,
        }
    except (ValueError, TypeError, OverflowError) as exc:
        fail("invalid_input", exc)
    if n == 0 or weight_sum == 0.0:
        fail("no_positive_training_weight", "no eligible positive-weight training samples")

    # Import/runtime failures are retained without pretending optimization ran.
    try:
        import torch
        tensors = [torch.from_numpy(v) for v in (design, logits, labels, sample_weights)]
        tx, tl, ty, tw = tensors
        te = None if signed is None else torch.from_numpy(signed)
        coefficient = torch.zeros(DIMENSIONS[name], dtype=torch.float64, requires_grad=True)
        if name in GATED:
            with torch.no_grad():
                coefficient[0] = -4.0
        model["initial_coefficients"] = coefficient.detach().tolist()
        model["coefficients"] = coefficient.detach().tolist()
        model["torch_version"] = str(torch.__version__)
        optimizer = torch.optim.LBFGS([coefficient], **LBFGS)
    except Exception as exc:
        fail("runtime_setup_exception", f"{type(exc).__name__}: {exc}")
    last_finite = coefficient.detach().tolist()

    def closure():
        nonlocal last_finite
        optimizer.zero_grad(set_to_none=True)
        row = {"evaluation": len(trace["closures"]) + 1, "status": "started"}
        trace["closures"].append(row)
        if not bool(torch.isfinite(coefficient).all()):
            fail("nonfinite_coefficients", "nonfinite optimizer trial coefficients")
        loss, data, penalty, energy = _objective(
            torch, name, tx, coefficient, te, tl, ty, tw)
        row.update(loss=_finite_number(torch, loss),
                   data_loss=_finite_number(torch, data),
                   penalty_loss=_finite_number(torch, penalty),
                   weighted_delta_squared=_finite_number(torch, energy))
        if any(row[k] is None for k in ("loss", "data_loss", "penalty_loss",
                                       "weighted_delta_squared")):
            fail("nonfinite_objective", "nonfinite trial objective component")
        loss.backward()
        gradient = coefficient.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            row.update(status="failed", gradient_norm=None, gradient_max_abs=None)
            fail("nonfinite_gradient", "missing or nonfinite trial gradient")
        norm = torch.linalg.vector_norm(gradient)
        row.update(gradient_norm=_finite_number(torch, norm),
                   gradient_max_abs=_finite_number(torch, gradient.abs().max()))
        if row["gradient_norm"] is None:
            fail("nonfinite_gradient", "trial gradient norm overflow")
        row["status"] = "finite"
        last_finite = coefficient.detach().tolist()
        return loss

    try:
        optimizer.step(closure)
        # This is the returned terminal state, never the minimum-loss trial.
        if not bool(torch.isfinite(coefficient).all()):
            fail("nonfinite_coefficients", "nonfinite terminal coefficients")
        loss, data, penalty, energy = _objective(
            torch, name, tx, coefficient, te, tl, ty, tw)
        trace["terminal_evaluations"] = 1
        terminal = {
            "loss": _finite_number(torch, loss), "data_loss": _finite_number(torch, data),
            "penalty_loss": _finite_number(torch, penalty),
            "weighted_delta_squared": _finite_number(torch, energy),
        }
        if any(v is None for v in terminal.values()):
            fail("nonfinite_objective", "nonfinite terminal objective component")
        gradient, = torch.autograd.grad(loss, coefficient)
        if not bool(torch.isfinite(gradient).all()):
            fail("nonfinite_gradient", "nonfinite terminal gradient")
        terminal.update(gradient=gradient.detach().tolist(),
                        gradient_norm=_finite_number(torch, torch.linalg.vector_norm(gradient)),
                        gradient_max_abs=_finite_number(torch, gradient.abs().max()))
        if terminal["gradient_norm"] is None:
            fail("nonfinite_gradient", "terminal gradient norm overflow")
        state = optimizer.state[coefficient]
        n_iter, func_evals = int(state.get("n_iter", 0)), int(state.get("func_evals", 0))
        if func_evals != len(trace["closures"]):
            fail("optimizer_accounting", "Torch func_evals differs from observed closures")
        accounting = {
            "n_iter": n_iter, "func_evals": func_evals,
            "observed_closure_evals": len(trace["closures"]),
            "terminal_loss_gradient_evals": 1,
            "configured_max_iter": LBFGS["max_iter"],
            "configured_max_eval": LBFGS["max_eval"],
            "max_iter_reached": n_iter >= LBFGS["max_iter"],
            "max_eval_reached": func_evals >= LBFGS["max_eval"],
            "max_eval_is_hard_cap": False,
            "returned_successfully": True, "convergence_claimed": False,
            "terminal_gradient_tolerance_satisfied":
                terminal["gradient_max_abs"] <= LBFGS["tolerance_grad"],
        }
        model.update(status="completed", coefficients=coefficient.detach().tolist(),
                     terminal=terminal, optimizer=accounting)
        trace.update(status="completed", optimizer=dict(accounting))
        return model, trace
    except FitError:
        raise
    except Exception as exc:
        fail("optimizer_exception", f"{type(exc).__name__}: {exc}")
    finally:
        if model["status"] != "completed":
            # Last finite is diagnostic only; it is not selected or deployable.
            model["last_finite_trial_coefficients"] = last_finite
            model["coefficients"] = (coefficient.detach().tolist()
                                     if bool(torch.isfinite(coefficient).all()) else None)
            state = optimizer.state.get(coefficient, {})
            trace["optimizer_at_failure"] = {
                "n_iter": int(state.get("n_iter", 0)),
                "func_evals": int(state.get("func_evals", 0)),
                "observed_closure_evals": len(trace["closures"]),
            }


def predict_delta(model, x, evidence=None):
    """Return float64 NumPy[N] corrections; invalid/failed models raise ValueError."""
    import numpy as np
    import torch
    if not isinstance(model, dict) or model.get("schema") != MODEL_SCHEMA:
        raise ValueError("invalid model schema")
    if model.get("status") != "completed":
        raise ValueError("only completed terminal fits are deployable")
    name = model.get("name")
    design, signed = _design(name, x, evidence)
    if model.get("input_dim") != DIMENSIONS[name]:
        raise ValueError("model input dimension mismatch")
    if model.get("family") != ("signed_gate" if name in GATED else "ordinary"):
        raise ValueError("model family mismatch")
    coefficients = _array(model.get("coefficients"), "coefficients", 1)
    if len(coefficients) != DIMENSIONS[name]:
        raise ValueError("coefficient dimension mismatch")
    tx, tc = torch.from_numpy(design), torch.from_numpy(coefficients)
    te = None if signed is None else torch.from_numpy(signed)
    with torch.no_grad():
        activation = tx @ tc
        if not bool(torch.isfinite(activation).all()):
            raise ValueError("nonfinite prediction activation")
        correction = _delta(torch, name, tx, tc, te).numpy().copy()
    if not np.isfinite(correction).all() or (abs(correction) > 4.0).any():
        raise ValueError("nonfinite or out-of-bound correction")
    if signed is not None:
        if not (correction[signed == 0.0] == 0.0).all():
            raise ValueError("zero-evidence correction is not exactly zero")
        if ((correction * signed) < 0.0).any():
            raise ValueError("signed correction reverses evidence")
        if (abs(correction) > 4.0 * abs(signed)).any():
            raise ValueError("signed correction exceeds evidence-scaled bound")
    return correction
