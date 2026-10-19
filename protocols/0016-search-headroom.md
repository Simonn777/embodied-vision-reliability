# 0016: distinguish remaining pose error from a tested search remedy

This is a prospective **development diagnostic**, not a new method, accuracy
claim, or manuscript result. The0015 sparse photometric method remains stopped.
The question is whether broader previously unread development pairs exhibit
substantial errors, whether four fixed official-estimator restarts contain better
poses, and whether a label-blind final-pose score chooses those poses. Stable
error does not prove input bias: common local minima, model mismatch, calibration,
reference errors and geometric ambiguity remain alternatives.

## Allocation and information barrier

Use the existing immutable332175160-byte MegaDepth1500 DKM cache, SHA256
22ee1720378add43932741fe36c58f65e4b11b9343995df3aa9b9fbee7bf9f2e.
Read root names only to allocate64 lowest SHA256 hashes of
`evr0016:development:2026-10-19\0` followed by the pair key, among scene0015's600
keys after excluding the16 keys in configs/0012-development-manifest.json.
The resulting configs/0016-development-manifest.json is frozen before prediction.
Allocation is not outcome-stratified; selected pairs can share photographs and
all belong to one development landmark. They are not independent scenes.
Scene0022 remains numerically unread. No image, confidence or depth value is read.

Predictions read only each allocated pair's x1/x2,K1/K2,camera1/camera2 through
the unchanged0013 allowlist and calibration validation. All10000 raw pixel matches
are used. No resizing, coordinate shift, initial pose, synthetic perturbation,
or match filtering is introduced. Selected R/t may be read only by the separate
evaluator after verifying all192 completed worker records, four terminal
candidate records each, source/library bindings and actual process exits.
Before numerical evaluation, verify the opaque cache size/hash again. A broken
binding, invalid pose/score, calibration failure, worker timeout or incomplete
cohort makes the whole diagnostic ineligible; preserve its outputs without a
scientific retry. An ordinary caught native estimator exception is a terminal
unavailable candidate, retained with its elapsed cost. All64pairs remain in every
reported metric, with180degree coverage penalties when no eligible pose exists.

## Three official backends, four fixed seeds

1. old_norm: unchanged official PoseLib2.0.5 wheel, pinned in0013.
2. current_norm: official PoseLib master6a636b1d84273d5e3ba97676463476569611572a,
   whose VERSION.txt and built module report3.0.0, normalized Sampson.
3. current_ts: exactly the same3.0.0 build with tangent_sampson=true.

Use seeds0,1,2,3 in that order within each worker. Each native call uses
min_iterations1000,max_iterations100000,dyn_num_trials_mult3,
success_prob0.9999,progressive_samplingfalse,max_prosac_iterations100000,
one-pixel inlier threshold,real_focal_checkfalse,and no initial pose.
Old settings use the legacy six-argument API; current settings use the new nested
`ransac` dictionary and top-level `max_error`. Do not silently pass old option
names to the new API. Artificial native qualification must show the nested
iteration fields are honored before any real data is opened.

Retain each version's official final bundle defaults. In2.0.5 the wrapper sets
Cauchy loss scale0.5pixel with gradient tolerance1e-10; in3.0.0 its default is
1pixel with tolerance1e-12,relative-cost tolerance1e-10,Nielsen lambda updates
and Levenberg damping. Both cap100final iterations. Therefore old-versus-current
is a combined version/default/build comparison, not a causal TS experiment or
fair compiler-speed comparison. Current normalized-versus-current TS isolates
the TS option in one build. Build metadata and exact runtime hashes are in
configs/0016-libraries.json and results/0016-preparation/.

No cross-backend candidate pool or additional long single-seed arm is used.
Rotate the order of the three backends by pair index. Serial workers each perform
four calls with one native/BLAS thread;90seconds per worker,1800seconds overall,
8GiB sampled RSS ceiling. Exclusive outputs and process logs preserve failures.

## Selection, cost, and interpretation

Save each FINAL R/t, native pre-refinement stats/mask, actual API duration,
score duration and cumulative charged time. Recompute final scores; do not select
using returned pre-final-refinement model_score. The common fixed selector
minimizes the sum of truncated squared **pixel F-Sampson** residuals at1pixel,
with the official algebraic cheirality gate at positive unit-baseline depth.01.
Ties choose the smaller seed. It is an independently defined common selector,
not claimed identical to normalized or Tangent Sampson. Compute those other
two final objectives as descriptive columns, with each residual normalized by
its own threshold; do not use their ground-truth performance to change selection.
Zero residual denominators saturate to1, while nonfinite input/arithmetic is an
explicit technical failure. Use unoptimized einsum contractions under strict
floating-point guards to avoid the previously established Apple BLAS flag issue.

For each backend report seed0 and best-scored completed prefixes at charged
budgets0.1,0.25,0.5,1,and2seconds, plus all four candidates with full actual cost.
Charged time includes pair input loading/calibration/geometry preparation, every
preceding native call, validation and all final-score computations. JSON writing,
Python imports and supervisor overhead are excluded from this charged measure
and separately represented by worker/outer wall times. A candidate is eligible
only after its charged_end_seconds is<=the budget. Native calls cannot be
interrupted to return an interim pose: this is an **offline completed-prefix
comparison**, not a measured real-time deadline implementation or exactly equal
spent compute. Feature matching is cached and absent from the measured scope;
this cannot satisfy the final complete-pipeline time requirement.

Freeze all deployed selections before reading R/t. Use the unchanged0013
SVD-projected rotation reference and directed translation convention; report
secondary sign-folded translation separately. Main pair error=max(R,t), with
trapezoidal AUC@5/10/20 over all64. Preserve individual R/t errors and coverage.
Report candidate pose dispersion and per-pair error ranges descriptively. A
ground-truth-best-of-four table is an explicitly nondeployable oracle upper bound
for this fixed candidate bank, with matched candidate count across backends;
more candidates can mechanically improve it. It is never an improvement claim.
No population confidence interval is supported by one physical scene.

## Next decision and unchanged final gate

This diagnostic has no invented success gate: it maps headroom and distinguishes
tested restart/selector/version effects. Before implementing an ordinal-depth
selector, require a current-backend four-candidate oracle to exceed both
current full-four deployed selectors by at least2pp AUC@5 on this development
bank. A large gap against a weaker old-version selector alone is insufficient. This is only a necessary search
opportunity, not sufficient information, novelty, or efficacy evidence. If absent,
do not add depth inference merely to choose among nearly identical candidates.
If present, a later separately frozen study must verify stable depth-order
disagreement, compare RePoseD/MADPose and other strong controls, and count all
additional feature/depth costs. Ordinal consistency is locally flat within strict
rank cells and cannot alone establish fine refinement. No depth operation is
authorized by this protocol itself.

A manuscript still requires a distinct defensible contribution, at least2absolute
AUC@5points at matched complete-pipeline time on two independent held-out sources,
positive paired scene-level95% intervals, no more than0.5pp AUC@20 loss, strong
official baselines, ablations and replication. A baseline diagnostic or its GitHub
publication does not meet the owner's stopping criterion.
