# 0034 pointwise fitting and prediction contract

Source preparation only. No scientific inputs or execution are admitted by this
file. `ADMISSION.md` and the final runner scope govern the experiment. This is a
known-baseline comparison, not an algorithmic novelty claim.

`models.py` has no filesystem access, native-model import or eager NumPy/Torch
import. It does not set global threads, random seeds, dtype or runtime paths.
Root supplies the qualified CPU runtime and complete-operation supervisor.

## API and operators

`fit_model(name, x, native_logits, target, weights, evidence=None)` returns
`(model_dict, trace_dict)`, both directly JSON-serializable with `allow_nan=False`.
Inputs are real numeric NumPy arrays (numeric array-like values also work);
the function copies them to float64 and never mutates caller arrays.

`x` is `[N,D]`, including an **exactly-one first column**; logits, labels and
weights are `[N]`. Binary labels must already be 0/1. The runner maps label255
to background under the admitted convention, constructs supports/samples and
fold weights, and normalizes only the absolute arm's native-feature channels
using its three training parents. This module never receives label IDs, parent
identities, feature statistics or held-parent labels.

| Name | D | Correction |
|---|---:|---|
| logit | 3 | ordinary |
| absolute | 50 | ordinary |
| prototype | 5 | ordinary |
| nn | 5 | ordinary |
| ordinary_relations | 17 | ordinary |
| bilateral | 4 | ordinary |
| signed_no_gram | 4 | signed gate |
| signed_gram | 14 | signed gate |
| displaced_gram | 14 | signed gate |
| rgb256 | 14 | signed gate |

Ordinary: `delta = 4*tanh(x @ coefficients)`, coefficients initially all zero.
Signed gate: `delta = 4*evidence*sigmoid(x @ coefficients)`, coefficients initially
zero **except intercept −4**. The gate therefore begins with a small nonzero
correction where evidence is nonzero; it is not native-identity initialization.
Separate evidence is required for gate arms, shape `[N]`, finite and in `[-1,1]`;
ordinary arms reject it. Evidence construction belongs to the feature provider.
Zero evidence produces exact zero correction. The gate cannot reverse its sign.

`predict_delta(model, x, evidence=None)` returns a copied float64 NumPy `[N]`
array. It requires a completed model, matching schema/name/family/dimensions and
finite parameters and linear activations. It enforces correction bounds and
signed zero/polarity/evidence-scaled bounds. Empty prediction arrays are valid.
The caller adds the correction to native logits only at admitted sites and owns
all native no-op behavior elsewhere.

## Objective, optimizer and accounting

The objective is exactly

`sum_i W_i*BCEWithLogits(L_i + delta_i, Y_i)
 + 1e-3*sum_i W_i*delta_i**2`.

Weights are finite and nonnegative, with sum at most one; **there is no
renormalization**. A 16-float64-epsilon allowance around one admits only summation
roundoff, and the actual sum is recorded unchanged. Unavailable frames retain
their zero data-loss mass through the runner's parent/frame weights. N=0 or
sumW=0 raises `FitError` with reason `no_positive_training_weight`; this is missing
eligible training evidence, not an alleged numerical bug.

One `torch.optim.LBFGS.step(closure)` uses fixed lr1, max_iter80, max_eval400,
tolerance_grad1e-6, tolerance_change1e-12, history_size20, strong_wolfe. There is
no restart, data-dependent setting, best-trial selection or checkpoint search.
The returned coefficients are the actual terminal optimizer state. The model
records the Torch version and settings.

Every closure records its index, status, weighted BCE, regularization, weighted
delta-squared, total loss, gradient L2 norm and max absolute gradient. There is
one separately counted terminal loss/gradient evaluation. The returned model
contains its full final gradient and objective components. Actual Torch `n_iter`,
`func_evals`, observed closure count, configured limits and limit-reached flags
are retained. Strong-Wolfe evaluation can overshoot Torch's configured max_eval;
400 is **not advertised as a strict cap**. The whole fit-and-prediction operation's
wall/RSS limits are enforced outside this module.

Finite terminal state or a successful optimizer return is not called convergence.
The terminal gradient-tolerance flag is descriptive; `convergence_claimed` is
always false. Reaching max_iter with finite values remains an explicitly
reported terminal fit, as admitted.

## Failure evidence and tests

`FitError` exposes `reason_code`, `partial_model` and `closure_trace`. Validation,
missing training weight, runtime setup, nonfinite objective/gradient/coefficients,
optimizer-accounting and optimizer-exception reasons remain distinct. On an
optimizer failure the evidence includes current finite coefficients if available,
last finite trial coefficients for diagnosis only, and all observed closures.
These are marked failed and cannot be passed to `predict_delta` for deployment.
There is no silent parameter repair or numerical fallback inside the module.
The runner records failure, native fallback for coverage, and failed continuation.

Nine manufactured tests are prepared, **not executed here**: hand-computed
ordinary output; signed polarity/zero/bounds; a small finite fit with the weight
sum below one; terminal gradient versus an independent finite-difference objective;
gate initial loss/gradient and interrupted-optimizer evidence; empty/zero-weight
fits; malformed inputs; finite loss with overflowing gradient; and invalid or
overflowing prediction. No model weights, scientific arrays or filesystem
fixtures are required. The tests do not import or call SAM2.
