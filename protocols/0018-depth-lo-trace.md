# 0018: prospective native depth-refinement trace

Prospective scientific protocol,2026-10-19. No scientific execution until this
protocol, implementation, patched source/binary, all configurations and baseline
bindings are frozen in a published commit and its artificial CI passes.
This is a log-only diagnostic following0017; no new estimator or paper is admitted.

## Question and fixed cohort

0017's scale-depth harm/native-inactivity condition occurred in Lincoln and
London; four-output oracle headroom was9.202524pp, while deployable score
selection gained only0.032132pp. Final inactivity did not establish causality.
Determine whether refinement introduces harm on the final returned model's
lineage, and whether state-dependent residual support contributes to the native
objective decrease. Poor proposals and discarded refinements must be distinguished.

Replay ALL80 existing0017manifest pairs with BOTH depth_scale and depth_affine:
160calls, one seed0, identical options/input order, no geometric-arm rerun.
These five scenes are permanently development and their labels are already
exposed. No case/seed selection, parameter search, extra replacement pairs or
selective outcome reporting. Original0017binary, sources and outputs are immutable.
The native source base remains6a636b1d84273d5e3ba97676463476569611572a.
The source-only design is `docs/literature/0018-depth-lo-trace-design.md`.

## Instrumentation boundary

Use a fresh source/build/package directory and preserve the original compiler,
SDK, Release/O3 flags, Eigen/pybind dependencies and MARCH_NATIVE=OFF. No fast-math,
dependency upgrade, loss/threshold/iteration/RNG change or existing callback
replacement. The instrumented binary has a distinct hash. Record exact source
patches, base inventory, commands, build receipts, environment and binary metadata
in configs/0018-libraries.json under `instrumented`; retain the0017pin as `baseline`.

Observe existing scalars/states in a separate bounded C++ buffer, disabled by
default. No Python callback, formatting/file write, extra objective evaluation
or reset of the live optimizer/accumulator inside LM. Do not fix the native
shared-count reuse, damping, stale returned model_score or visibility behavior.
Serialize after the native call. Freeze a50,000event native buffer limit percase;
on overflow finish the unchanged algorithm but mark the trace incomplete.
Serialized traces are capped at64MiB/case and512MiB/cohort; any incompleteness
blocks interpretation of the entire160case diagnostic, not just a difficult case.

Retain three explicitly labelled stages: record-breaking-proposal LO,
terminal RANSAC LO, and unconditional wrapper final refinement (or explicit skip).
Internal stages use all input matches, native TRUNCATED loss/25iteration settings;
wrapper final refinement uses its supplied pre-final epipolar mask, native CAUCHY
defaults/100iteration setting. Record actual effective settings and support once
per LO. Do not silently assume identical support or losses across stages.

Each invoked LO receives a unique ID and raw entry/accepted/exit states, including
quaternion, rotation, raw translation,scale,shift1,shift2. Retain all attempted
steps, accept/reject and termination information, actual numerator/count/mean,
lambda/step/gradient quantities available without extra math, and all count roles:
Jacobian construction count, count used by the solve and candidate cost count.
Rejected nonfinite trials remain explicit invalid records, not removed successes.
Capture already computed native scores and proposal/incumbent/adoption identity;
derive the final winning ancestry. An LO adopted temporarily and later replaced
by an unrelated proposal cannot explain the final returned pose.
An `attempt` event means a candidate state and its objective were evaluated.
Gradient or small-step termination before a trial is represented by the exit
reason and recorded norms/count roles; it is not a missing evaluated candidate.

## Qualification and complete parity barrier

Artificial qualification compares observer enabled/disabled in the new build
and both with the pinned baseline. Exercise both shift modes, accepted/rejected
steps, zero-step/early termination, visibility loss/gain, TRUNCATED zero weights,
final support and trace overflow. Artificial stress cases do not tune on real
image outcomes. Record coverage gaps honestly; unresolved observer effects block
scientific replay. Native-call qualification is separate from this160case study.

After a complete160case seal, require final raw R,t,scale,shift1,shift2 and numeric
stats to match saved0017values componentwise:
`abs(a-b)<=1e-12*max(1,abs(a),abs(b))`. Masks, discrete iterations/refinements and
statuses must be identical. Prefer/report bitwise equality as well. Do not
normalize away raw-gauge changes or relax tolerance after outcomes. Baseline
source/seal/input and every new process/log/output/trace hash must match.
Any parity/coverage failure blocks all new label-based interpretation.

Workers read only the inherited0017coordinate/camera/depth allowlist; no cache
R/t reads. A separate evaluator may use the already saved0017reference poses only
after the complete parity barrier. One thread,90s/worker,1800s/cohort, sampled
worker RSS<=8GiB. Build uses at most2jobs. Include all failed/timeout cases in
receipts and never rerun automatically. Report build,native replay,observation,
diagnostic reconstruction and label-evaluation costs separately; instrumentation
timing is not a method speed comparison. No new asset download or paid service.
The separate reconstruction pass has a90s/case and1800s/cohort deadline and
512MiB serialized-output cap; exceeding any cap fails the entire diagnostic.

## Independent native-cost reconstruction

Reconstruct each LO's initial and accepted states from its fixed observations,
raw gauge and recorded effective loss/settings. This separate charged pass does
not optimize or modify a pose. Use only two adjacent states' per-block arrays in
memory; do not persist N-point arrays for each iteration. Reconcile native sums
and means with tolerance `1e-12+1e-9*max(abs(replay),abs(native))`; block counts
must agree exactly. A reconstruction/count mismatch or nonfinite accepted-state
arithmetic blocks the whole case and thus the complete-cohort interpretation.
Native invalid rejected trials are reported as such, without requiring a finite
counterfactual value. No clipping, point deletion or post-hoc tolerance repair.

The native objective is S/max(1,n), with one block per scalar Sampson residual
and one per2D directional residual. It is not the capped16px diagnostic used in
0017. For adjacent states with positive counts, retain common-block change D,
old lost-block sum L,new gained-block sum G,and exact old/new counts:

    S1-S0 = D-L+G
    C1-C0 = D/n1 + (-L+G)/n1 + S0*(1/n1-1/n0).

Use actual TRUNCATED/CAUCHY kernels, thresholds and weights, not proxy MSAC.
Handle native max(1,n) explicitly if zero counts occur. Report loss-only,
gain-only,mixed and unchanged-support transitions, including discarded LOs.
The support contribution combines the last two terms. Do not call a tiny
contribution negative/positive evidence unless its magnitude exceeds10times the
propagated reconstruction tolerance. Report the values/bounds regardless.

## Interpretation and next-stage gate

After parity, compute existing0013directed rotation/translation/max errors at
every valid entry,accepted and exit state; sign-folded errors are secondary.
Do not select a best intermediate pose with GT as a deployable output. Report
all80pairs,all five scenes,both arms,each stage, discarded/adopted/final-ancestry
counts and continuous changes. Material harm retains the0017definition:
>=1degree increase or crossing from<5to>=5degrees. Distinguish net LO-entry/exit
harm from individual step changes. An inactive final direction alone is insufficient.

Primary necessary trigger for a separately frozen matched-initialization
intervention: the scale-depth arm has a material-harm LO on the final returned
ancestry in at least two scenes, with at least one accepted support-losing
transition IN THAT SAME LO that itself causes material pose harm (>=1degree
increase or crossing<5to>=5degrees) and whose support contribution lowers native
mean cost beyond the noise bound. Weaker co-occurrence is descriptive only and
cannot substitute for this primary gate. Report whether common-block changes
alone could explain the decrease. This same-step requirement was specified
before instrumented construction or any scientific trace, following source-only
review; it must not be relaxed after observations.
Affine-depth observations are a parallel known-method diagnostic and do not
retroactively substitute for the primary scale condition. This arithmetic and
trajectory evidence is still association, not interventional causal proof.

If the primary condition fails, close this specific support-loss explanation;
do not rescue it by selecting only the four0017examples, altering bounds, or
calling any earlier temporarily adopted model the final ancestor. The distinct
additional-information opportunity remains a separate research question.
If it passes, a new prospective experiment must use exactly matched LO initial
states/support/loss/budget to compare explicit known interventions. A fresh
RANSAC run or GT-chosen intermediate state is not such a control.

MADPoseCVPR2025 already uses fixed residual blocks and scalar scale/depth bounds.
Ordinary accumulator corrections or cheirality fixes cannot automatically be
labelled a novel method. No manuscript before a distinct contribution passes
the unchanged>=2AUC5pp/matchedcompletepipeline/twoindependentheldoutsource gate,
scene-levelconfidence,harm limits,strongcontrols,ablations andreplication.
