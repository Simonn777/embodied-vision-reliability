# 0018: admission design for a native depth-refinement trace

2026-10-19. Source-only design; no build, estimator execution, cache-array read,
image read, or new reference-pose read was performed for this note. This is not
an experimental protocol or an admitted new estimator. It proposes a separately
frozen diagnostic following 0017. Parent-reported 0017 outcomes establish a
trigger, not causality: depth-scale outputs with masked inactive directions and
material pose harm occurred in Lincoln/London; additional-depth oracle headroom
was approximately 9.20 AUC5 percentage points. No method achieves that oracle.

**Recommendation:** admit a log-only replay of all 80 allocated pairs and both
depth arms, conditional on artificial qualification and final-output parity.
Do not restrict the cohort to the four trigger pairs, add seeds, change an
optimizer setting, or implement a rescue rule in this diagnostic. It can locate
where harm enters. A trajectory alone cannot prove that an alternative objective
would prevent it.

## Source contract and precise hooks

All code claims below come from direct reads of PoseLib revision
`6a636b1d84273d5e3ba97676463476569611572a`, the source used for 0017. Preserve its
algorithm, input order, random generator, options, floating-point expressions,
termination checks, loss callbacks, and support masks. An instrumented binary
necessarily has a new hash; call it the same algorithm, not the same binary.

| Hook | Required observation and interpretation |
|---|---|
| [`RelativePoseMonoDepthEstimator::generate_models/refine_model`](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/estimators/relative_pose.cc#L112) | Identify iteration, minimal sample, model index and raw proposal. Internal refinement uses all input matches, TRUNCATED loss, 25 iterations, and the scaled epipolar loss threshold. Record the input/output of every invoked LO. |
| [`score_models` and terminal refinement](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/ransac_impl.h#L109) | Log the already computed proposal/refined MSAC scores, inlier counts, incumbent updates and model lineage. The last qualifying proposal within a sample is refined. Its result is adopted only if its score improves the incumbent. A separate terminal LO is also conditionally adopted. Do not infer adoption from an LM cost decrease. |
| [`estimate_monodepth_relative_pose`](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust.cc#L383) | Capture the post-RANSAC/pre-final geometry and the actual returned epipolar-mask indices. When `stats.num_inliers>3`, the wrapper runs a third stage on that fixed subset and returns its result without an outer MSAC-revert test. Record a skipped stage explicitly. |
| [`refine_monodepth_relpose`](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/bundle.cc#L576) | Attach stage context and the actual effective options, weights, `scale_reproj`, `weight_sampson`, and `refine_shift` to each invocation. The wrapper's final stage uses the native bundle defaults and its own scaled loss threshold; it is not identical to internal LO. |
| [`lm_impl`](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/optim/lm_impl.h#L57) | Record initial state/cost, every attempted step's cost and native accept/reject decision, accepted states, and every exit path. Log before early breaks. Preserve all computations and decisions. |
| [`MonoDepthRelPoseRefiner`](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/optim/monodepth_relpose.h#L133) and [`NormalAccumulator`](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/optim/jacobian_accumulator.h#L69) | Observe cost numerator/count, direction-specific active counts and nonfinite flags. Preserve the literal `Z.z>0` predicate. Never reset or recompute the live accumulator for logging. |

The existing `IterationCallback(BundleStats, RobustLoss*)` is insufficient: it
does not expose geometry, misses several termination paths, and can change a
graduated loss. Keep it intact. Add a separate read-only C++ observer, disabled
by default, with a scoped context for the monodepth calls. No Python callbacks,
GIL transitions, formatting, or file writes inside the LM iterations. Copy
compact records into owned buffers and serialize after the native call. If a
buffer limit is reached, finish the algorithm unchanged but mark the entire
trace incomplete and ineligible for causal interpretation.

The pybind wrapper currently returns geometry and stats/inliers. A separate
trace entry point can return those unchanged plus the trace. Do not use its
`initial_pose` argument as a substitute for a matched refinement-only binding:
the downstream RANSAC initializer resets pose components, and a fresh RANSAC
call is not a replay of a particular LO. No standalone calibrated monodepth
refinement binding was found in the inspected binding source.

## Important correction: native cost is an active-block mean

The prior source note's gated-sum expression describes the numerator only. The
actual `NormalAccumulator.get_residual()` returns

\[
C(\theta)=S(\theta)/\max(1,n(\theta)),\qquad
S=\sum_{b\in A(\theta)}w_b\rho(\|r_b(\theta)\|^2).
\]

Each scalar Sampson residual counts as one block. Each two-dimensional
directional reprojection also counts as one block, not two. Both the numerator
and denominator change when a transformed depth stops passing `Z.z>0`.
Consequently, dropping a low-cost block can *increase* native cost. Counting
inactive directions alone cannot establish that dropping terms helped a step.

For old/new active sets with positive counts, define common-block change D,
old lost-block cost L, and new gained-block cost G. Then

\[
S_1-S_0=D-L+G,
\quad C_1-C_0=(D-L+G)/n_1+S_0(1/n_1-1/n_0).
\]

Report all four contributions. Use the actual scaled residuals, loss type and
weights: internal TRUNCATED uses `min(r_squared,tau_squared)`; final CAUCHY uses
`tau_squared*log1p(r_squared/tau_squared)`. The independent fixed-denominator
16-pixel diagnostic from 0017 is a different objective and cannot replace this
accounting. The all-inactive case uses the native `max(1,n)` convention and must
be handled explicitly rather than applying the displayed positive-count form.

There is an additional implementation distinction to preserve. Jacobian
accumulation increments its count only for nonzero robust weights; cost
accumulation counts all active blocks. A shared `residual_count` is overwritten
by the candidate cost evaluation before Nielsen's predicted-decrease calculation.
After a rejected trial, the next solve retains the previous JtJ/Jtr while using
the most recently evaluated candidate's count. Thus record separately the
Jacobian-construction count, count actually used by each solve, and candidate
cost count. These are source facts, not demonstrated causes of 0017 error.
Do not silently fix this count reuse or change damping during instrumentation.

## Minimal trace and diagnostic replay

Retain every case and every invoked LO, including discarded LOs and zero-step
convergence. Each record needs:

- Pair/arm, stage, LO ID, RANSAC iteration and proposal identity, support size
  and support-index hash; store each actual fixed support once.
- Raw quaternion, raw translation, depth scale and both shifts at entry,
  accepted steps and exit. Do not normalize translation for depth arithmetic.
  Candidate states can also be retained for rejected trials at low storage cost.
- Attempt number, native iteration counter, acceptance, numerator/count/mean,
  lambda before/after, gradient/step norms, invalid-step count, actual termination
  reason, all three count roles above, and nonfinite status.
- Existing native model scores and adoption/incumbent lineage at every outer
  update. The terminal RANSAC code does not update `stats.model_score` when it
  adopts its final refinement; record its already computed local score and do
  not treat the returned stats value as an independently verified final score.

The observer should copy native scalars, not recompute a second objective inside
the optimizer. A separate, charged diagnostic pass can reconstruct directional
support and per-block losses from saved states plus the same allocated inputs.
Audit every accepted transition, with aggregate counts for unchanged-support
transitions and the full D/L/G/denominator decomposition for all support-changing
transitions. Use fixed loops and explicit nonfinite handling. Reconcile this
diagnostic cost with the recorded native numerator/count before interpreting it.
No discrepancy may be hidden by clipping or dropping a state. Keep only two
adjacent states' block arrays in memory; do not persist a per-point array at
every iteration. The pass does not optimize or choose a new pose.

Before any new label-based interpretation, seal the complete 160 replay outputs
and traces, verify bindings and final parity, then evaluate all recorded initial,
accepted and exit poses with the existing directed-translation metric. These are
already-exposed development scenes, not new blinded validation. Show each scene,
arm and stage, all failures and no-support-change controls. Retain continuous
error changes; the already declared 1-degree/5-degree material-change definitions
may be used for descriptive counts without introducing tuned cutoffs.

## Parity, artificial qualification, and resources

Use a separate source copy/build directory and package path. Preserve the 0017
binary and outputs. The existing build is Apple clang, Release `-O3 -DNDEBUG`,
GNU C++17, arm64, `MARCH_NATIVE=OFF`, with local Eigen/pybind dependencies. Copy
the actual existing compiler, SDK, definitions and flags; do not add fast-math,
change contraction settings or upgrade dependencies. The inspected command also
has `-Wall -fPIC -Wno-sign-compare -Werror -Wfatal-errors`. Changing the generic
LM header can rebuild many refinement instantiations. Two build jobs and the
existing 8 GiB process limit are conservative proposed limits, not measured cost.

Artificial qualification must compare observer disabled/enabled within the new
build, and both against the pinned baseline: zero-step exits, rejected/accepted
steps, positive-to-nonpositive crossings, support gain, TRUNCATED zero weights,
both shift modes, and final-stage omission. Verify original outputs, callback
sequence, RNG/proposal sequence and decision counters. Explicitly exercise an
early relative-cost termination so the last accepted state is not missing.

For all 160 real replays, require exact mask, iteration/refinement counts and
discrete outcomes, with bitwise floating-output parity preferred. Compare the
saved R, raw t, scale and shifts: 0017 did not save raw quaternion coefficients.
Quaternion parity is available for the artificial observer-on/off comparison.
If a numerical
tolerance is needed, freeze it before execution; a proposed strict alternative
is componentwise `abs(a-b)<=1e-12*max(1,abs(a),abs(b))` for raw geometry, scores
and stats, plus identical discrete outcomes. Do not use quaternion sign or
translation normalization to hide a changed raw gauge. Do not relax a failed
parity test after seeing outcomes. Any mismatch blocks causal interpretation;
it does not become an excluded case. Record the patched source/binary hashes
and complete compile commands independently of the original identities.

With the fixed 1,000 RANSAC iteration cap, at most one proposal LO is invoked per
iteration, plus terminal LO and final wrapper refinement: at most
`25*(1000+1)+100 = 25,125` attempted LM iterations per case (no supplied initial
model). Across 160 cases this is 4.02 million possible attempts. At a compact
256 bytes per attempt, the loose cohort storage ceiling is about 1.03 GB before
headers/serialization; JSON can be much larger. Use compact storage with explicit
byte accounting, per-case flushing, and no scientific worker concurrency. Do not
store N-point residual arrays per attempt. C++ logging and diagnostic replay
overheads are unmeasured; source inspection cannot guarantee a five-minute run.
Propose reusing the 90-second worker / 1,800-second cohort / 8-GiB limits, with
incomplete coverage blocking admission. Charge build, native replay, observation,
diagnostic reconstruction and label evaluation separately. An artificial budget
check is necessary but cannot prove the real cohort will fit.

## What the trace can decide, and what remains known prior

Distinguish: bad proposal entering LO; an accepted LO trajectory worsening its
input; a worsened LO being discarded; a worsened LO replacing the incumbent;
and unconditional final-refinement harm. Only adopted lineage can explain a
returned model, and a poor initial state with no further harm is not an LO
failure. If no accepted transition loses support, or support-loss transitions
do not worsen pose, reject this specific omission mechanism despite the 0017
final-state association. A cost reduction attributed arithmetically to deletion
is stronger evidence than an inactive final point, but still is not an
interventional causal result.

A later separately admitted control could replay exactly the saved LO initial
state with fixed constructed residuals, scalar depth/scale bounds, or both,
holding observations, robust loss and budget fixed. Neither a fresh RANSAC call
nor selecting the best intermediate pose with GT is a matched control. Invalid
projection behavior and gauge-scaled bounds must be specified in advance;
simply holding the denominator fixed while invalid residuals cost zero is not
a repair. These interventions are **known controls**, not the proposed novelty.

[MADPose, CVPR 2025, full relevant §§3–4.4](https://arxiv.org/html/2501.05446v3#S4)
already combines affine-depth calibration, hybrid proposals and geometric/depth
residuals. Its reviewed
[optimizer setup](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/optimizer.h)
constructs fixed Ceres residual blocks and applies scale/minimum-depth bounds;
its calibrated functors directly project those blocks. These do not guarantee
positive transformed depth, but preempt a claim that fixed support plus positive
scalar calibration is itself a new invention. RePoseD's
[analysis discussion](https://arxiv.org/html/2501.07742v3) also acknowledges
physically impossible solutions with favorable scores. Any subsequent method
must beat these closest treatments and explain a distinct identifiable benefit.
If a known bound, ordinary accumulator correction, or fixed-support change
exhausts the gain, report it as that result rather than inventing a new theory.
