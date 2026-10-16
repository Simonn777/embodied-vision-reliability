# Protocol 0002-E — exploratory RGB-Stacking pilot (DRAFT, NOT FROZEN)

2026-10-16. AI-assisted proposal informed by development-only CPU feasibility.
**No calibration/final inference is authorized by this draft.** Publish a later
frozen revision, implementation and complete identity manifest before evaluation
in a subsequent session. Protocol 0001 and its results stay unchanged.

## Narrowed question and evidence boundary

For identical frozen Online BootsTAPIR coordinates, does an additional causal
motion residual rank large localization errors better than the model's native
temporal confidence? Both native refined logits already contain history. This
does **not** answer the original strict single-image-versus-temporal question.
RGB-Stacking is synthetic; upstream training/model-selection exposure is not
fully audited. No real-video, real-robot, competitive benchmark or novelty claim.

## Identity and partitions — FINAL FREEZE BLOCKED

- TAPNet code: `730cda1c730877cfedbe01bf87fb1cadb78a565d`.
- Checkpoint SHA-256: `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`.
- RGB-Stacking ZIP SHA-256: `ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc`.
- Entire video index 0, all 250 frames/30 points, is permanently development.
  Only its first 16 frames were inferred in this feasibility session. Other
  arrays were materialized by restricted container decoding but not visualized,
  scored, or inferred. Indices 1–49 remain unassigned; do not call them a frozen
  final set. Pixel hashes and group relationships for them are not yet audited.
- Proposed capacity: 10 calibration videos, 20 evaluation videos, remaining
  videos reserved; actual identities/counts must follow the manifest audit, not
  outcome-based selection. All queries, crops and frames of a video/group stay
  together. Scene independence is unknown. Without defensible scene grouping,
  label the resulting study an **exploratory video-disjoint pilot**, never
  scene-independent confirmation. A separate real source is still needed.

## Planned bounded execution

CPU torch 2.10.0+cpu, NumPy 2.0.2, dm-tree 0.1.9 and einshape 1.0; strict state-dict
load with weights_only=True, eval, no gradients, inference mode. Verify source,
checkpoint/data and array/manifest digests first. Explicit use_casual_conv=True,
one current 256×256 frame, initial recurrent state per arrival group, local query
t=0 and absolute-arrival mask; no future query features or image access.

Proposed clips are the first 64 frames of each allocated video, at most the first
two point indices visible at frame 0 (no outcome selection). If fewer exist,
retain/report the available count or zero; do not replace the video. This choice
excludes late-born points and long trajectories from the primary scope; an
arrival-frame correctness test exists separately on development data. Freeze
this policy and any necessary revision before evaluation. Do not select which
points to retain using tracking error or visibility after initialization.

At most four full-video prefixes per invocation, timeout 900s, two threads/CPU
affinity and 6 GiB address-space cap. Check observed RSS too. The 16-frame
development measurement is not a certified latency/RAM bound for longer clips.
If a frozen run exceeds the cap, preserve failure; do not silently shorten its
evaluation clips or discard slow/hard videos. Resume only under a documented
protocol revision, or report an incomplete exploratory pilot.

## Fixed candidate scores — must be implemented/tested before freeze

Use the last refinement coordinates and raw native logits on those same
coordinates. Let p_visible=sigmoid(-occlusion) and p_local=sigmoid(-expected_dist).
They are model scores, not newly calibrated error probabilities.

- Native product risk score: `s_native=-log(max(p_visible*p_local, 1e-8))`.
- Augmented score: `s_aug=s_native+min(d/8,10)`, where d is current position's
  pixel distance from a constant-velocity prediction from the two previous valid
  raw coordinates at their true timestamps. With fewer than two past positions,
  residual is zero; rejected valid predictions still update history. No future
  frame, annotation, renderer state or visibility truth enters either score.
- Controls: visible-only and localization-only native scores, deterministic
  label-free random ranking, accept all finite outputs, and labeled oracle
  localization-error ranking. No weights/scale selected on evaluation results.

The eight-pixel residual scale and unit weight are prospective exploratory
choices, not an optimized or efficacy-supported method. A change after seeing
evaluation outcomes cannot be presented as the same untouched experiment.

## Opportunities, endpoints and matched coverage

Primary evaluation opportunities are ground-truth-visible point/frames after
the initialization frame within each fixed prefix. Ground-truth visibility is
used for evaluation/calibration accounting only, never an inference input. Report
all-frame visibility accuracy separately. Occluded-coordinate error is excluded
unless annotation validity is separately established; do not treat it as zero.

Localization failure is Euclidean error **>4 pixels** at 256×256 (strict >).
An invalid/nonfinite prediction counts as missing and cannot be accepted. Coverage
denominator includes all visible opportunities, including missing predictions.
Report missing rate and estimator failure including missing outputs. Conditional
selected risk uses accepted opportunities as denominator and is undefined/null
with zero accepts; never replace it with zero.

Primary ranking comparison: at prespecified 80% coverage of all visible
opportunities, sort finite predictions by score and deterministic hash tie key,
and take the same `ceil(0.8*N)` count for both methods. Labels only evaluate risk;
scores/weights are not refit. If finite availability is insufficient, primary
risk is unreachable/null and attainable coverage is reported. This descriptive
fixed-coverage test ranking is **not a deployable threshold chosen without test
access**. Report curves too; do not select a favorable coverage afterward.

Operational secondary comparison: choose separate native/augmented thresholds
targeting 80% calibration visible opportunities using a score/hash pair,
then lock them for evaluation. Report realized calibration coverage, evaluation
coverage AND conditional risk; unequal coverage cannot support a matched-coverage
superiority claim. Hash ties are fixed by video identity/point/frame and method
independent. No conformal guarantee is claimed.

Practical exploratory target: augmented minus native selected-failure risk at
matched 80% coverage **<= -0.02** (two percentage points). Report effect and
interval even when absent/negative. A useful signal requires the interval's upper
endpoint below zero plus this practical point-estimate target; these criteria
are not a confirmatory power calculation or license to claim general efficacy.

Use paired video-cluster bootstrap, 2,000 draws, seed 20261018. Carry all selected
queries/frames and both methods together and recompute pooled counts/ratios and
fixed-coverage rankings in every draw. Keep operational calibration thresholds
fixed. Report 95% percentile intervals, defined replicate counts and null bounds
unless >=95% of draws are defined. This conditions on the observed calibration
sample and video-disjoint exploratory set; frames/queries are not independent
replicates and unknown scene dependence limits population interpretation.

## Outstanding freeze checklist

Resolve grouping or explicitly adopt video-disjoint exploratory status; publish
complete calibration/evaluation/reserve identity hashes; implement and test
metrics/ties/missingness/cluster resampling; profile a development-only 64-frame
two-query clip; audit any upstream benchmark exposure; publish exact source,
protocol/config, safe-loader and dependency hashes. Only then run the allocated
evaluation videos in a later session. This draft does not imply those gates pass.
