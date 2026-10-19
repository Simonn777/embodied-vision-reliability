# 0017: additional-depth information and failure diagnostic

Prospective protocol,2026-10-19. This is an80-pair comparison of four **known
methods**, not a proposed new estimator. Its purpose is to determine whether
depth supplies useful candidate poses and exposes an unresolved failure mechanism
after0016 found negligible restart-bank headroom. No manuscript is admitted.
Source review: `docs/literature/0017-depth-diagnostic-review.md` and the0016
monodepth resource/consumer contracts. Pinned PoseLib source is
`6a636b1d84273d5e3ba97676463476569611572a` (declared3.0.0); use the already built
binary in `configs/0017-libraries.json`, not a new/reconfigured implementation.

## Allocation and leakage barrier

The separate `0017-depth-assets.md` plan fixes five public outdoor RoMa/MoGe
caches and16hash-ranked keys each, before reading values. The80manifest rows in
`configs/0017-development-manifest.json` are immutable; every one of the five
physical scenes is permanently development, including unused keys. No pair
replacement, outcome-conditioned exclusion or same-scene final-test claim.
Frontend checkpoints/exporter provenance remain unverified; this compares known
estimators on the same authored cached inputs, not a neural pipeline reproduction.

Publish all production source, protocol, library config and allocated manifest,
and pass artificial tests before scientific values are accessed. Prediction reads
only x1,x2,K1,K2,camera1,camera2,depth1,depth2 at allocated keys. Read authored
axial-depth proxies unchanged. Finite negative or zero depths remain in the
input; count them, do not repair or prune them. Unsupported camera/calibration,
nonfinite input, binding mismatch or malformed output blocks evaluation rather
than silently changing the cohort. There is no ground-truth reader in a worker.

All320arm outcomes and process receipts must be sealed, including failures,
before the evaluator can read only selected R/t labels. Final-pose selection is
fixed from saved scores before labels. Source, binary, config, manifest, input
file, output and process hashes are verified. Do not rerun a failed estimator
or timeout. A technical blocker is preserved and reviewed before any separate
prospective repair; no old output is overwritten.

## Four fixed arms

Arm order `e_norm,e_ts,depth_scale,depth_affine`; rotate execution order by global
pair_index modulo4. One seed0 per arm, serial processes, one native thread. All
arms: min_iterations100,max_iterations1000,success_prob0.9999,
dyn_num_trials_mult3,progressive_sampling=false,score_initial_model=false;
nested ransac/bundle dictionaries, bundle={} native defaults, no initial pose.
No restarts, extra iteration arms or selected seeds.

| Arm | Explicit native setting |
|---|---|
| e_norm | estimate_relative_pose; max_error2px; tangent_sampson=false; real_focal_check=false |
| e_ts | same API; max_error2px; tangent_sampson=true; real_focal_check=false |
| depth_scale | estimate_monodepth_relative_pose; max_errors=[16,2]px; estimate_shift=false; weight_sampson1 |
| depth_affine | same API and thresholds; estimate_shift=true; weight_sampson1 |

RePoseD uses depth-driven proposals, epipolar scoring and joint refinement. This
API does not guarantee retaining a five-point solution. MADPose and other
published hybrid methods remain strong mandatory controls before a future
contribution claim. A software default change or combination of these arms is
not novel. Separate singular max_error and plural max_errors deliberately.

Per-worker watchdog90s, entire cohort1800s, worker-process sampled RSS<=8GiB,
at most4CPUthreads globally. Imports/loading, native call, validation/diagnostics
and worker/outer wall times are reported separately. Equal iteration budgets are
not equal compute. Cached feature/depth inference is excluded and unavailable;
no complete-pipeline efficiency claim follows. Preserve all terminal logs and
commands, return codes, package/environment and source bindings.
The charged total is outer supervised worker wall time including imports,
bindings, file hashing and output overhead, not merely the component sum.
Interrupted/native-failed component timings may be null when unknowable; their
full observed worker wall remains charged. Successful input/native/diagnostic
components are finite and their sum cannot exceed that outer wall (1microsecond
rounding tolerance). Workers spawn no scientific child processes.

## Saved outputs and diagnostics

Retain final R and raw t, raw depth scale/shift1/shift2, all returned native stats
and mask. The returned mask is **pre-final-refinement** support, not recomputed
final inliers or the final active depth support. Count final quantities on every
input and separately on that mask with explicit denominators. Never normalize
raw t for depth reprojection; its gauge is coupled to depth parameters.

All four final poses receive the same independent pixel-F Sampson MSAC at2px,
including the existing0.01unit-baseline cheirality check and saturated
degeneracies. The optional external selector chooses the minimum saved common
pixel_msac, then fixed arm order. Its work cost is the sum of all four arms,
including failed calls and diagnostics, not just the chosen arm. It is a control,
not native RePoseD, and cannot guarantee true-error non-harm.

For depth outputs, with z-one rays h1,h2, c1=d1+shift1,c2=d2+shift2, let
Z12=R(c1*h1)+t and Z21=R.T(scale*c2*h2-t). Count c1<=0,c2<=0,scale*c2<=0,
Z12.z<=0,Z21.z<=0, unions, finite active and nonfinite cases, for ALL and mask.
For every fixed set S compute both-direction16px capped reprojection error;
each transformed point with nonpositive z or nonfinite projection costs1.
Divide by2|S| without dropping any invalid direction. Empty mask is unavailable.
This reference is not the native normalized Cauchy objective. Any analogous
omission-score gap is an arithmetic identity, not proof of harmful optimization.
Also store literal native branch counts for `Z.z>0`. A nonfinite projection
is invalid for the fixed-denominator reference, but positive-infinite z passes
that native branch. Do not conflate projection invalidity with branch omission.

## Evaluation and decisions

Use existing0013metric functions: validate proper predicted rotations, normalize
translation only for angles, project GT rotation as U@Vt (no reflection repair),
directed translation primary and sign-folded secondary. Report rotation,
translation,max error and DMS left-search trapezoid AUC5/10/20, all80 and each
16-pair scene. Unavailable estimates get180degrees in the full denominator;
show successful-only metrics/coverage separately. Invalid GT blocks scoring.

Report each depth arm against both geometric arms: all per-pair differences,
wins/ties/losses, rescue/damage across5degrees, and material change defined as
at least1degree or crossing5degrees. Define nondeployable GT oracles O_G=min
of two geometric arms,O_D=min of two depth arms,O_ALL=min of all four. Report
AUC5(O_ALL)-AUC5(O_G) and AUC5(O_ALL)-max(single-arm AUC5), explicitly labelled
oracle diagnostics. Descriptively report every scene and leave-one-scene-out
ranges; no80-independent-trial confidence interval or five-scene generalization.

Two distinct necessary opportunities are fixed:

1. A new four-output selector is worth considering only if additional-depth
   oracle headroom AUC5(O_ALL)-AUC5(O_G)>=2absolute percentage points. This
   neither establishes a usable selector nor admits generic ordinal selection.
2. A state-dependent depth-residual omission audit is triggered only if the SAME
   depth arm has a completed pair in each of>=2/5scenes with material harm against
   BOTH completed geometric arms, and at least one inactive transformed-depth
   direction on its nonempty returned mask, specifically a direction failing
   the literal native `Z.z>0` condition. Material harm here is at least1degree
   worse than each geometric arm, OR depth error>=5while both geometric errors<5.
   Use final masked support counts, not all-match outliers, for this trigger.

Report both conditions even when false. Final counts cannot identify causality.
If the second passes, a **separately frozen** matched-initialization LO trace is
needed to distinguish poor proposals from damage introduced by refinement.
Cheirality/fixed-support fixes are known and may be ordinary bug fixes. No
modified estimator, fitted gate or subsequent native rerun is part of0017.

Preserve prior gates and negative results. A manuscript remains conditional on a
distinct defensible method with>=2absolute AUC5points at matched complete-pipeline
time on TWO independent held-out sources, positive paired scene-level95%intervals,
<=0.5pp AUC20loss, strong official baselines, ablations and independent replication.
The current80pairs, oracle numbers and known-method benefits cannot satisfy that.
