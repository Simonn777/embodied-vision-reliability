# 0004 Stage 0: does short-term state prevent recoverable point tracking?

2026-10-18. Development-only mechanism test. AI-assisted design and implementation.
This protocol precedes new inference. No positive result or novelty is asserted.
The owner explicitly requested a results-oriented pivot; prior 0001–0003 protocols,
thresholds, negative findings and cache failures remain unchanged. The blocked
0003 geometry operation is deferred, not executed or replaced by this experiment.

## Question and fixed intervention

Online TAPIR already matches the original query globally on every frame. We test
whether its short-term causal refinement state can nevertheless hinder recovery.
All branches use the same frozen weights, original query descriptors, query batch,
current RGB feature grids and chunk size. Only causal convolution context differs.
No query re-extraction, prediction-based initialization, future image, GT-based
action, model training, threshold search or selection policy is permitted.

For each frame, the baseline continues its own context. A diagnostic zero branch
receives an independent all-zero copy of the baseline's pre-frame context and is
discarded after that frame. At starts 32,64,96,128,160,192,224, an additional shadow
resets context once, then evolves independently for eight frames. It never affects
the baseline. At frames 0,32,128,224,249 a same-state clone control must match the
baseline at absolute tolerance 1e-4 (relative tolerance zero). All leaves are cloned
separately, avoiding the upstream constructor's repeated-dictionary aliases.
Matched controls must also have exactly identical next-context SHA256 digests.

Every branch is evaluated with the same immutable original descriptors. Query and
current-feature fingerprints are checked before/after branch calls. Context keys,
shapes, dtypes and devices must remain unchanged; all predictions/context tensors
must be finite. Keep and zero must also agree at frame zero; each shadow's first
output must match that frame's zero branch at the same tolerance. Any failure stops
the attempt and preserves partial evidence; no silent retry or threshold change.

## Data and exposure

Only the already cached TAP-Vid RGB-Stacking development video index 0 is allowed:
250 frames, 256×256 RGB, first eight original point indices visible at frame zero.
Frame-zero GT is used solely to initialize the specified task query. Full labels
are hash-validated on load, but the inference callback receives no labels; scoring
occurs only after the complete raw prediction file has been closed and hashed.
Video 0 is permanently development and synthetic, with previously inspected frames.
No original calibration/evaluation/reserve video or PointOdyssey cache is accessed.
New inference is necessary because saved 0002 predictions contain no branch state
or counterfactual predictions. It is not a rerun of the old frozen comparison.

Cached NPZ and checkpoint digests are fixed in `configs/0004-stage0.json`.
Upstream is google-deepmind/tapnet commit
`730cda1c730877cfedbe01bf87fb1cadb78a565d`; three actual imported source digests
must match the preserved audited helper. No data/weights/dependencies are downloaded.
Only derived coordinates, logits, error summaries and execution evidence are public;
do not redistribute RGB, checkpoint, or upstream paper text. Data provenance and
licenses remain those recorded in the [0002 audit](../docs/data-model-audit-2026-10-16.md).

## Measurements fixed before inference

Save every branch's x,y,occlusion logit,expected-distance logit, frame, original
point identity, and shadow start/offset. Save encoder, each refiner call, whole-frame
and total process wall time; actual call counts and peak RSS. Expected calls:
250 keep +250 zero +5 clones +56 shadow =561; 250 encoder calls. This deliberately
expensive diagnostic is not a deployed compute-budget claim. Encoder sharing and
clone/fingerprint overhead are included in whole-run timing.

After inference, exclude query frame 0. Report visible-coordinate error rates at
1,2,4,8,16 pixels; visibility and visible-correct proportions; threshold Jaccards
and their mean using native visibility `(1-sigmoid(occ))*(1-sigmoid(dist)) > 0.5`.
At each threshold, TP is GT-visible/predicted-visible/correct; FP is predicted-visible
and either GT-occluded or incorrect. Jaccard is TP/(GT-visible+FP). These are
eight-query development diagnostics, not full official-benchmark scores.

Report paired visible baseline-wrong→zero-correct and baseline-correct→zero-wrong
counts at 4 px regardless of predicted visibility, so rejection cannot masquerade
as coordinate recovery. Report all visible post-query paired EPE differences,
including signed zero-minus-keep EPE for each original query. Also report the paired 8-frame shadow return for each
start/query. Loss is 1 unless (GT-visible and predicted-visible and error≤4px) or
(GT-occluded and predicted-occluded). Return is mean(keep loss − shadow loss).
Keep individual beneficial, neutral and harmful episodes; no GT oracle is deployed.
Do not bootstrap frames, points or windows as independent scenes. There is one
video; no population confidence interval, significance or transfer claim is valid.

## Decision rules

This small test can reject a broken implementation or reveal possible headroom.
If fewer than 20 baseline visible errors >4px occur, label it **undersensitive**;
do not claim that the mechanism succeeds or cannot work. If there are ≥20 such
errors, require at least five coordinate recoveries, a positive aggregate shadow
return, and beneficial shadow episodes for at least two original queries to justify
testing the same operator on representative independent development videos. Report
all harms even if this continuation rule passes. Failure stops this exact candidate
from advancing on this dataset; a changed mechanism needs a dated new protocol.
Passing is only permission for another diagnostic, not a learned policy, innovation,
real-world improvement, generalization, or permission to write a positive paper.
The broader gates are in [research direction](../docs/0004-research-direction.md).

## Execution and provenance

Use the restored existing cloud CPU environment, two threads/cores, 6 GiB address
space and a 900-second hard timeout. Require pinned torch 2.10.0+cpu (no CUDA),
NumPy 2.0.2, dm-tree 0.1.9 and einshape 1.0; save actual imports/environment. No
paid service or subscription upgrade. Exclusive output directory and a shared
0004 cache marker prevent accidental repeat launches; neither may be overwritten.
Inspect existing processes/output before launch. Do not touch old 0003 markers.

Publish and independently review code/config/protocol before this attempt; a
separate freeze JSON binds their committed bytes, source revision and dependencies.
Runner checks both commit and working bytes. Save command, seed, input hashes,
versions, timing, partial rows/failure trace, observed outer exit and artifact hashes.
Run meaningful fake-backend tests before release. Independently recompute summaries
from saved raw rows (no new inference) and review claims before publishing results.
No claim of completion if timeout, missing frames or control/integrity failure.
