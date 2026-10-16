# Protocol 0002-E v1: exact-duplicate-group-disjoint exploratory pilot

2026-10-16. AI-assisted specification. The separate `0002-freeze-v1.json` receipt
binds the input commit and every executable/config/manifest/dependency/spec hash.
It excludes its own commit/hash, preventing circular provenance. Existing drafts
and development protocols remain historical records. This session performs no
calibration/evaluation inference. Future execution is gated on this freeze input
commit being merged into origin/main and unchanged frozen input hashes.

## Question and scope

On identical frozen Online BootsTAPIR coordinates, does an extra causal constant-
velocity residual improve error ranking over native **temporal** confidence?
This does not resolve the strict single-image confidence question. It is a small
exploratory comparison, not confirmation, published-paper reproduction, novel
algorithm, real-robot validation, SOTA or a safety guarantee.

RGB-Stacking's 50 episodes share triplet-4 object assets and a front-left camera.
Scene/episode identifiers and dependencies are unknown; near-duplicates are not
excluded. Different exact hashes do not establish scene independence. No new-
scene/object/viewpoint generalization is claimed. The first two visible annotated
indices are fixed few points, not representative random samples. Published
benchmark data may have influenced upstream work; complete checkpoint training/
model-selection non-exposure is not established. Our evaluation videos have no
project inference/error scoring or score tuning before freeze, not global novelty.
See the [primary-source audit](../docs/0002-exposure-and-scope.md).

## Frozen identities and allocation

Use [complete manifest](../results/0002-preflight/identity/manifest.json) and
[all RGB frame hashes](../results/0002-preflight/identity/frame-hashes.csv).
Each of 50 entries has archive ordinal, all video/points/occluded array shapes,
dtypes and SHA256, full-video RGB and actual first64 RGB-prefix SHA256. Audit
access was metadata/byte hashing only for videos 1–49, not model/error outcomes.
There are no scene/episode fields. All arrays are 250-frame videos with 30 points;
coordinate arrays are float64, not the earlier generic float32 schema assumption.

Union ANY inter-video exact frame, prefix or full-video match transitively. The
observed 12,500-frame audit yields 50 singleton groups and no cross-video exact
edges. Whole group containing video 0 is permanently development/quarantined,
only video 0 may be inferred for development. All other videos keep allocated
calibration/evaluation/reserve roles; no later tuning may relabel them development.

Allocation was prespecified independently of outcomes: group ID encodes sorted
zero-padded ordinals; sort SHA256 UTF8 `0002|20261019|group_id`, then group ID. Take
complete non-development groups until >=10 calibration videos, then >=20
evaluation videos, rest reserve. Actual counts: development 1, calibration 10,
evaluation 20, reserve 19 (same group counts). Exact identities are in the manifest,
not selected later. Any allocation changes require a new published freeze before
evaluation. Unknown cross-group scene dependence remains a limitation.

## Fixed execution and queries

Official TAPNet `730cda1c730877cfedbe01bf87fb1cadb78a565d`, checkpoint SHA256
`87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`, archive
SHA256 `ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc`.
Version pins/wheel hashes and actual imported module path/hash gates are frozen.
Code/weights Apache-2.0, video/annotations CC BY 4.0; preserve upstream notices,
publish original scripts and small derived results, never raw inputs/weights.

First 64 consecutive frames, 256x256 RGB, at most first two frame-0-visible point
indices. Arrival is frame 0; no frame-4/future visibility screening or replacement
for a zero-query video. Keep the whole video in its partition even with zero
queries/opportunities. Provided initial query is privileged and explicit.
Initialize each query's descriptor from frame 0 with local t=0 and a constructed
initial recurrent state; no future features/other ground-truth queries enter a
point's output. Query batching passed a development check at atol 1e-4, rtol0.

CPU torch 2.10.0+cpu, strict weights_only state-dict load, explicit causal flag,
eval/no gradients/inference_mode, seed 20261016. Two CPU affinity/numeric/torch
intra/inter-op threads, external timeout900, 6GiB address-space cap. Restricted
NumPy-only dataset decoding; no unsafe loading fallback. Release unrelated video
arrays before loading torch. Infer one entire duplicate group (max four videos)
per process; do not split a group or omit slow/hard members. Refuse reused output
directories; flush raw xy/logits/time every frame and receipts after every video.
Missing/nonfinite estimator outputs stay explicit in complete prediction grids.
Resource failure stops the group; preserve incomplete artifacts, never substitute
shorter/easier clips. A necessary change requires a new freeze. The 64-frame
development profile supports this scope, not realtime/all-query guarantees.

## Causal scores, common availability and history

The scorer receives timestamps/raw xy/two logits only, no labels/visibility truth.
Use float64 after checking finiteness. Coordinate-valid means both xy finite;
logit-valid means BOTH occlusion o and expected_dist u finite. Native/augmented
and ranking controls share joint availability = both conditions. Inf logits do
not turn into valid probabilities. Finite out-of-image xy are not clipped.

Let L=-log(1e-8) and softplus(z)=logaddexp(0,z). Lower is better:

- native = min(softplus(o)+softplus(u),L), implemented equivalently by capping
  each nonnegative term before addition to avoid finite-sum overflow;
- visibility-only = min(softplus(o),L), local-only = min(softplus(u),L);
- augmented = native + min(d/8,10), unit weight;
- random = zero score plus fixed original tie; oracle = true EPE, labeled as
  unavailable at deployment and used only in descriptive ranking.

d uses the two previous coordinate-valid raw predictions of that point/video:
extrapolate p1+(p1-p0)*(t-t1)/(t1-t0), then Euclidean distance to current raw xy.
Timestamp units are original frame indices (no physical acquisition timestamps
are released); gaps span actual indices, not valid-row count or compute latency.
With <2 past valid coordinates d=0. Score current before append. Valid coordinates
always update raw history despite invalid logits, rejection or GT occlusion.
Points/videos have separate histories; rejected gates do not alter model/scorer
state. Scale/weight/cap remain prospective fixed exploratory choices, not tuned
on held-out errors. The uncertainty head was trained for >6px, not our >4px target;
none of these scores is claimed to be a calibrated four-pixel probability.

Original method/outcome-independent tie is SHA256 UTF8
`0002|archive_index|video_sha256|point_index|frame_index`. Sort (score,tie).
Bootstrap copies keep the original tie and every copy; no deduplication by ID.

## Opportunities, endpoints and controls

Exclude frame 0. Primary opportunities are GT-visible selected point/frames in
the fixed prefix. GT normalized xy multiply [256,256], not 255. EPE >4px strictly
is failure; EPE=4 is not. Missing coordinate predictions count as estimator failure
and remain in the all-visible-opportunity coverage denominator. Invalid logits
separately reduce score availability; they do not delete coordinate errors.
Do not score occluded-coordinate error without established validity. No extra
six-pixel sensitivity endpoint is added.

Report coordinate missing rate, joint-score unavailable rate, estimator failure
including missing coordinates; coordinate-only accept-all coverage and conditional
risk; six ranking controls above. All-rejected conditional risk is null, never
zero. Zero-opportunity metrics are null. Visibility accuracy separately covers
ALL post-query frames: head-only predicted-visible iff finite o<0 (equivalent to
sigmoid(-o)>0.5); invalid o is incorrect even when GT is occluded. Coordinate and
u validity do not gate this head-only diagnostic. No probability guarantee.

Primary comparison is selected-failure risk at matched 80% of ALL visible
opportunities, m=ceil(.8*N). Each method selects its first m jointly available
ranked rows. If insufficient, matched risk/coverage are null and attainable
coverage is reported; do not lower the target. This is a fixed-coverage descriptive
ranking evaluation, not a deployable threshold selected without test access.
Report full risk-coverage curves to attainable coverage without choosing a new
coverage. Oracle uses the same joint availability and fixed coordinates.

Secondary operational policies fit separate thresholds using calibration visible
opportunities only: m=ceil(.8*Ncal); use the mth finite (score,original tie). If
unreachable, use largest finite observed pair, mark unreachable. No finite rows
means undefined threshold/reject all, not +Inf. Freeze thresholds once; apply
unchanged to evaluation. Report realized calibration/evaluation coverage and
conditional risk together. Unequal operational coverages are not matched-risk
superiority. Accept-all is explicitly separate and coordinate-only.

## Paired exact-group uncertainty and practical target

Sampling frame is the COMPLETE evaluation-group allocation, including groups
with no queries, rows or visible opportunities. Draw G groups with replacement
for 2,000 paired replicates, seed20261018. Carry every group's videos, queries,
frames and both methods together. Count multiple sampled copies fully; recompute
Ndraw, ceil(.8*Ndraw), rankings, ratios and native/augmented differences each draw.
Do not refit calibration thresholds. No-opportunity/zero-accept ratios undefined.
Report defined replicate counts and percentile95% intervals only with >=1900
defined replicates; otherwise null. This conditions on calibration data and the
published shared capture domain; unknown scene dependence limits interpretation.

Prospective practical signal: augmented-minus-native matched-risk point estimate
<= -0.02 and interval upper bound <0. Report effects/intervals regardless of sign;
this is exploratory, with no confirmatory power or general efficacy claim.

## Future entry and freeze lifecycle

`run_frozen_pilot.py infer --group-id ...` accepts only frozen calibration/evaluation
groups after origin/main contains the freeze input commit. It verifies frozen
files, versions, source origins, archive/video/prefix and checkpoint digests.
`evaluate --predictions ...` requires every allocated group's complete bound grid,
including header-only zero-query videos, then executes scores, calibration,
controls and paired bootstrap. No missing video is silently dropped. The complete
pipeline was exercised on synthetic fixtures, never held-out predictions here.

After source/spec/config/manifest changes, retain this receipt/version and publish
an explicit new freeze revision BEFORE evaluation. Do not overwrite a published
freeze or call reused test data untouched after outcome-informed revisions.
