# 0003 v1 — motion versus localization-error curvature: DESIGN FREEZE

October 17, 2026. AI-assisted prospective design following the already disclosed
0002 negative result. No old-test recalculation or tuning, new predictor run,
formal effect estimate, threshold fit or experiment is performed here.
**DESIGN_ONLY; BLOCKED_NO_FINAL_COHORT.** A separate receipt will bind this design
input commit. This is not a final cohort or execution freeze and cannot authorize
inference. Missing source/data evidence is a gate, not an assumed success.

## Scientific question and identifiable quantities

Can a fixed image-trajectory residual distinguish genuine motion changes from
localization drift? A 2D predicted track cannot identify 3D localization drift.
The identifiable error term here is **2D localization-error curvature**. Under
verified simulator GT, 3D world motion is a separate diagnostic, not real-robot
truth. No physical or causal attribution follows solely from image curvature.

At exactly the same three timestamps t0<t1<t2 and original prediction-history
anchors, define q=(t2-t1)/(t1-t0) and signed L(x)=x2-x1-q*(x1-x0).
With p=g+e, r_pred=L(p)=L(g)+L(e)=r_GT_image+r_error in vector coordinates.
**Norms do not add.** Perspective projection is nonlinear: there is no additive
camera/world split asserted. K and a verified world-to-camera T_cw project world
homogeneous columns with positive camera z: u=K*(T_cw*X)_xyz, xy=u_xy/u_z.
This is an implementation convention awaiting validation against the actual
PointOdyssey annotation/export code; it is NOT claimed verified on sample data.

Compute world velocity changes separately as
2*[(X2-X1)/(t2-t1)-(X1-X0)/(t1-t0)]/(t2-t0). Report world-coordinate units per
timestamp-unit squared. Do not label m/s² without verified metric units and time
scale/FPS. Do not interpret frame-index second differences as physical acceleration.
The same camera pose and K at each timestamp must be used, including a moving
camera. Static world points can have image curvature due to camera/projective effects.

## Acquisition and independence gate

Primary candidate: PointOdyssey v1.2. Official project and repository are linked
in `results/0003-data-design-20261017/sources.json`. Dataset terms are conservatively
CC BY-NC-SA 4.0 despite the author HF card's conflicting MIT label. No raw content
will be redistributed. Code rights do not imply unrestricted data rights.
Proposed HF revision `4a2ec9492ca732dc3611adbb2a94248cc2bbe3eb` and generator revision
`a053342a72fed32af4134b4ca0c44a78a3ac0fdb` are unverified exact-byte leads in this
cloud. Proposed sample is 3,324,284,510 bytes, SHA256
`436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f`.
A future authorized acquisition must verify both; a partial digest is never a
whole-object digest. Current direct requests failed DNS, received zero bytes,
and did not reach GET after sample HEAD failed. No repeated full acquisition.
Full test/train/val archives exceed the budget and remain excluded. RoboTAP and
TAPVid-3D are source-audit alternatives only; see limitations in the source audit.

Total data/weight/dependency acquisitions stay under 10 GiB. Existing ceiling is
1,482,283,622 bytes; additional source-text accounting is conservative and explicit
in the ledger. No paid service, renderer/GPU, install, training stack or gated
license/login workflow is enabled. Existing subscription CPU only, at most two
CPU slots/numeric threads, 6 GiB AS, 900 s per child. No automatic download fallback.

A successful later archive audit must stream inventory without extraction; reject
absolute/traversal/backslash/NUL paths, empty components, links/devices, duplicate member paths, >200,000
members (one terminal slash normalized only for directory entries), any single annotation >512 MiB, or total unpacked content >20 GiB. Read
only numeric NPZ/NPY arrays with allow_pickle=False and bounded decompression;
reject object dtype. Record member names/sizes/hashes, schema, timestamps, units,
all scene/story/source-session/asset identities and multiview links. Inspect only
one lexicographically first audit scene, its geometry and at most 3 RGB frames
(0,1,2) initially. Any inspected scene's entire connected family is permanently
AUDIT/DEVELOPMENT. Access ledger distinguishes metadata, RGB, GT and predictions.
Future archive metadata inventory alone does not expose predictor errors; geometry
or RGB inspection still quarantines its entire family. No scene was inspected here.

Union exact video/prefix/shared-frame duplicates plus shared session/story,
synchronized views, animation/motion source and reused asset/environment families
transitively. Unresolved links conservatively join into an UNKNOWN family; such a
family cannot contribute independent held-out units. Different hashes never prove
independence. Family evidence, not video count, must establish at least 20 independent
evaluation families and 10 calibration families. Source independence from old
RGB-Stacking is not checkpoint pretraining independence; full upstream exposure is
UNKNOWN and must remain declared. No globally unseen-data claim is permitted.

## Final cohort and fixed sampling rules (not yet allocated)

Keep all audited/unknown families development. For verified untouched families,
sort SHA256 UTF8 `0003|20261020|family_id`, then family ID. Allocate first 2 additional
families development, next 10 calibration, next 20 evaluation, remainder reserve.
All linked views/clips stay together; never split whole families or refill a role
based on predictions/errors. A complete identity/access/family/partition manifest,
verified geometry, runtime/source/model hashes and executable evaluation pipeline
must be separately published as **FINAL COHORT FREEZE** before any calibration or
evaluation prediction. None exists now. Unknown independence means blocked, not
20 invented groups. No current adapter is run-ready for PointOdyssey.

One preselected clip/view per family: lexicographically first canonical view and
first 64 consecutive original frames, start frame 0, no temporal resampling; shorter
clips are ineligible before prediction. Save exact source frame IDs/timestamps,
FPS/units status, dimensions and resize transformation. Select at most the first
two original point indices frame-0-visible with finite provided 2D GT and valid
positive-depth 3D projection at frame 0; arrival=0. No future visibility screening,
point replacement or model-error selection. Zero-query families remain allocated.
Use raster xy*[256/width,256/height] at 256x256, no crop/255 scaling/clipping; K
uses the same row scaling. Verify projection against provided 2D annotations in
original raster coordinates: >=99.9% of finite visible validation points within
0.5 original px, no convention fitting on evaluation. Failures block acquisition
eligibility until source ambiguity resolved on development and a revised design
is explicitly frozen, rather than flipping conventions to improve scores.

Before predictions, >=95% of source-visible post-query opportunities per family
must have valid finite GT2D, world GT and camera projection. Record all excluded
families/points/reasons before final freeze; no replacement based on error. After
freeze, **primary** opportunities are all finite-GT2D source-visible frames t>0,
including coordinate-missing predictions as failures. Report invalid-GT denominator
and percentage separately; do not remove primary rows because a diagnostic GT3D
anchor is invalid. Diagnostics use the predictor's actual two previous finite raw
coordinate anchors and SAME times/GT. Invalid GT anchors make that diagnostic
undefined, with counts, without picking other anchors or changing primary denominator.
No future data enters a score; GT is accessed for diagnostics only after predictions
and their hashes are frozen. Rejected/occluded/invalid-logit coordinates still update
raw finite-coordinate history. Nonfinite coordinates skip history with real gaps.

## Reference policies, reliability endpoints and planning goal

Keep Online BootsTAPIR checkpoint SHA256
`87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`, upstream
`730cda1c730877cfedbe01bf87fb1cadb78a565d`, causal/eval/inference-mode CPU gates
and dependency identities as the existing reference. No new policy or weights
are introduced; source/runtime adapter must be verified before final cohort freeze.
Native is temporal confidence, NOT single-image confidence. Finite occ/expected
logits are required before stable softplus sums; native=min(sum,-log(1e-8)).
Augmented=native+min(norm(r_pred)/8,10), weight 1. No GT, errors, geometry,
visibility labels or future frames enter either score/history. Initial two finite
predictions have zero residual; frame 0 is not a scored opportunity. Nonfinite
logits make both policies unavailable even if coordinates are finite. Finite
out-of-raster predictions are not clipped. Missing coordinates are failures.

Strict visible error >4 px at 256 defines failure. Jointly available mask is shared
by native/augmented matched comparisons. Original tie is SHA256 UTF8
`0003|dataset_revision|family_id|view_id|clip_start_original_frame|point_index|frame_id`;
this key is method/outcome independent and preserved for bootstrap copies.
Matched target is ceil(0.8*N) of ALL primary visible opportunities, not just available
rows. If unavailable target, report null risk/INCONCLUSIVE, never lower coverage.
Risk=accepted failures / accepts; visible coverage=accepts/N. Also report coverage
over all post-query rows, raw coordinate missingness, GT support and score availability.
Zero opportunities or zero accepts imply null risk. No silent occluded-error endpoint.
Fit score+tie operational boundaries ONCE on 10 calibration families at target 80%,
apply unchanged to evaluation and bootstrap. No finite calibration rows: undefined,
reject-all; unreachable: largest finite observed boundary, marked unreachable.
Operational unequal coverage is distinct from matched comparison. No refits.
Random hash rank and GT-error oracle are descriptive controls, oracle non-deployable.

The prospectively chosen **pilot continuation** goal is 20% relative conditional
error reduction: C=R_aug-0.8*R_native, paired 95% cluster CI upper <0 (and C<=0).
Also report both absolute risks, absolute delta, relative change, and coverage.
This is a planning judgment, not estimated efficacy, power, confirmatory validation
or robotics utility. Zero observed native risk cannot pass. Support requires >=20
verified independent evaluation families, >=5 families bearing native matched
accepted errors, >=20 total native matched failures, valid GT support gates, reachable
matched coverage, and >=1,900/2,000 defined bootstrap replicates. Otherwise classify
INCONCLUSIVE; no adaptive hard-case addition, endpoint shift or test retuning.

Bootstrap seed 20261021, 2,000 paired complete-family draws with replacement from
all allocated evaluation families (including zero-query/zero-opportunity units).
Retain repeated copies; for each paired draw recompute Ndraw, ceil(.8*Ndraw), both
ranks/risks and C with original ties/copy IDs. Do not deduplicate original rows.
Zero-opportunity/unreachable draws undefined; a finite zero-native-risk draw's C
remains defined, but zero point-baseline cannot pass continuation. Percentile CI
uses linear-interpolated 2.5%/97.5% quantiles only with >=95% defined draws. No frame-iid
uncertainty, no claim of checkpoint non-exposure or statistical power.

## Mechanism estimands and controls, kept separate from continuation

Freeze low GT image curvature <=0.05 px; high >=0.5 px; middle reported separately.
These cutoffs are planning judgments and cannot be moved based on predictor errors.
For correct visible predictions (EPE<=4), M1 is each family's mean d_pred(high)
minus mean d_pred(low), averaged equally over families with BOTH strata; report
support/undefined families. For failing low-GT-curvature predictions, M2 is each
family's fraction with d_pred<=0.05, averaged over families with that stratum.
Report signed r_GT_image/r_error/r_pred components, their norms/dot products and
cross-term 2*r_GT·r_error (not additive norm attribution). The mechanism pilot can
be descriptive with >=3 verified development families and at least 2 families per
required stratum; below this, only per-family/fixture descriptions, no mechanism
population or efficacy inference. It is not a smaller substitute confirmatory test.
World-coordinate velocity changes and camera pose changes are reported separately,
with units status; no perspective-additive camera/world attribution or 3D predicted
localization-error estimate. GT strata are diagnostic only after prediction freeze.

Hand-computed negative/positive controls at t=0,1,2: GT=pred x=(0,1,4) gives r=2
and zero error; constant +10 offset and linear error (0,3,6) can have large EPE
with zero innovation; quadratic error gives positive innovation. Cancellation
GT=(0,1,8),pred=(0,1,2) yields rGT=6,rerror=-6,rpred=0,EPE6. Orthogonal signed
components (3,0)+(0,4) have norm5, not7; EPE4 is NOT failure. Irregular times
(0,2,5),x=(0,4,10) give zero. Static-world/moving-camera projection tests distinguish
world motion from image curvature. These injections are algebra controls, not
measured native tracker drift or new model performance results.

Before final cohort freeze, a new-development-only procedural precision assessment
must use these fixed rules to simulate 2,000 family resamples at the planned group
counts and compare CI width/defined fraction under stated error sparsity scenarios;
save all assumptions/negative outcomes. No old RGB-Stacking outcomes or held-out
labels may estimate/tune scales, goals, strata or sample additions. It has NOT run
here. If budget/independence/headroom insufficient, retain blocked/inconclusive status;
a genuinely revised study requires a new prospectively published design/cohort freeze.

## Binding and readiness

`0003-design-freeze-v1.json` is the sole canonical DESIGN receipt. Publish input
files first, then a separate receipt naming that real input commit and all nine
mandatory file SHA256 values. Receipt does not include its own SHA/commit in inputs.
Verifier rejects changed blobs/current/index/HEAD, unsafe paths/symlinks, duplicate
JSON keys, missing/empty/extra maps, alternate config/protocol paths, self-containing
input commits and unpublished receipts. Exact receipt must exist on freshly fetched
origin/main or the fixed 0003 review branch. Local refs are publication evidence,
not cryptographic remote attestation. This verifies design provenance only; no flag
or CLI mode starts an experiment. After merge, next work is source/acquisition and
new-development geometry/provenance evidence, then a separately reviewed final cohort
freeze. No inference/evaluation is authorized by this artifact's existence.
