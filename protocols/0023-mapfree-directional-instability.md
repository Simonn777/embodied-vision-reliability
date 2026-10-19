# 0023 — directional reference instability in a small Map-free development graph

Date: 2026-10-19. Prospective development protocol; no method result, novelty
claim, official Map-free benchmark reproduction or manuscript admission.

## Admission, allocation and unchanged research gate

This protocol conditionally adopts the preserved private candidate plan with
literal SHA256
`cc9b02cac8946107f1c945643b316695e5ebbf6339b34baf737541f8cec886d5`.
The plan remains byte-for-byte unchanged, including its historical
`candidate_only`/`allocated=false` fields. Its adoption is recorded by this
protocol and the subsequent frozen operation receipts, not by rewriting it.

Allocation/acquisition may begin only after a complete eligible 0022 artificial
frontend qualification **and its successful resource-supervision receipt**, and
publication of the exact 0023 implementation/protocol commit with passing CI for
that commit. Bind the admitted 0022 freeze, installation, qualification, assets,
logs and receipts; a failed outer operation cannot be rescued by an inner
`completed` record. Preserve every earlier compatibility failure. No qualifying
model, checkpoint, image or setting is chosen using scientific outcomes.

The prerequisite 0022 qualification05 has now completed under freeze
`fa454585e67a8f0395ad5f67a03d5a28cc33a66a`, after exact CI run `39339776188`.
Its successful outer receipt records 23.377363625 seconds and peak sampled
group RSS 2,129,712 KiB. This establishes the manufactured frontend runtime
qualification only. Exact 0023 source/protocol CI and the admission checks above
are still required; no 0023 image acquisition or allocation is implied by this
qualification receipt.

At that admission the three places, in order `s00513`, `s00475`, `s00478`, become
**development forever**, including their unselected frames. The other 62 places
in the cached archive inventory remain unallocated and their image,
calibration/depth and pose payloads unread. Earlier metadata-only enumeration
does not establish foundation-model training independence. Do not silently
replace a place or frame after discovering poor overlap, parallax or performance.

Passing this small study admits only a separately frozen continuation. It does
not meet the unchanged [0012 outcome gate](0012-score-bounds-development.md) or
the [research charter](../docs/RESEARCH_CHARTER.md). Before a manuscript, a
defensible distinct method still needs at least +2 absolute AUC@5 percentage
points at matched complete-pipeline cost on **two independent held-out sources**,
positive paired scene-level 95% intervals, no more than 0.5-point AUC@20 loss,
strong official controls, ablations and independent replication. Previous
closed hypotheses and their results remain closed. Neither algebra, runtime
qualification nor infrastructure completion substitutes for pose improvement.

## Fixed source, images, graph and acquisition

The only source is the official paired Map-free validation ZIP:
`https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/dataset/val.zip`,
archive size 7,907,389,876 bytes, ETag
`"efdc0e742961dbe955226543ad213d35"`. The immutable plan binds the three cached
directory/index/receipt SHA256s; no second archive or full-archive fallback is
allowed. The cached index is metadata, not proof that all payloads are usable.

Per place use anchor `seq0/frame_00000.jpg` and 16 seq1 frames at uniform
chronological member ranks
`floor(k*(N-1)/15 + 1/2)`, k=0,...,15. Sort source frames by actual frame number,
then filename. Odd one-based selected positions are B1,...,B8; even positions
are C1,...,C8. References are A0→each C and each B→the three C images nearest by
**actual frame-number difference**, ties by lower C frame number then filename.
This is 32 reference edges and eight A0→B query edges per place: 51 images,
96 references, 24 queries and 120 unique match caches overall. Temporal index
proximity is not an image-overlap guarantee.

Initial acquisition admits exactly the plan's 51 JPEG members and three
`intrinsics.txt` members. Their compressed total is 11,058,077 bytes,
uncompressed total 11,169,983 bytes and largest member 293,393 bytes. The
metadata-derived response-body ceiling including headers and one-byte probes is
14,600,405 bytes, below the hard 20 MiB cap. Exactly 162 bounded Range requests
are planned (three per member), with 30 seconds per request and no hidden retry.
Require HTTP206, exact Content-Range/archive size/ETag, local ZIP header/name/
method/size agreement, bounded decompression and CRC32; record output SHA256 and
every response body. A Range-ignored 200 or inconsistent source fails closed.
No pose member is accepted by the frontend acquisition helper. All payloads
stay in fresh private work directories, never in the public repository.

## Shared frontend and fixed match selection

Use the source-pinned 0022 CPU float32 frontend and its exact learned weights,
strict key/configuration checks, local-only loading and recorded compatibility
exceptions. No additional weights, package installation, training or implicit
network/model loading is admitted here. The Python audit guard is not an OS
network sandbox; inspected pinned code, local verified inputs and explicit
offline guards jointly support this runtime contract.

Decode each selected image once as RGB; reject unexpected dimensions and
nonidentity EXIF orientation. Do not transpose EXIF. Downsize only if needed to
maximum edge 512 using Pillow bilinear with round-half-up output dimensions.
Use authored per-image PINHOLE calibration, with `fx'=sx*fx`, `fy'=sy*fy`,
`cx'=(cx+.5)*sx-.5`, `cy'=(cy+.5)*sy-.5`. Features and projection use zero-based
array pixel centers; do not add .5 to projection coordinates.

Run complete MoGe inference once per image with num_tokens=1200,
use_fp16=False, apply_mask=False, force_projection=True and the admitted native
PyTorch SDPA configuration. Keep focal/shift/metric postprocessing and raw depth,
mask and other outputs. SuperPoint uses max_num_keypoints=2048 and
`extract(resize=None)` once per image. LightGlue uses features=superpoint,
flash=False, mp=False, depth_confidence=-1, width_confidence=-1 once per fixed
edge. Preserve all 51 image and 120 edge outcomes, including empty matches.

Normalize raw depth by the median positive valid depth of that **whole image**
once. At keypoints, bilinear depth is valid only when all four distinct
neighbors are in bounds, model-valid, finite and positive. Save raw/normalized
samples and a separate validity flag. Invalid depth or an empty valid-depth
population does not remove a feature or geometric match. Do not replace invalid
measurements with usable synthetic depths.

The input adapter verifies the complete acquisition/installation/qualification/
frontend source and receipt chain, all 171 output files and their per-output
JSON records, before loading any canonical array. NPZ loading uses
allow_pickle=False and a fixed key allowlist; array shapes/dtypes/raw hashes
and edge copies of image coordinates/depth/flags must agree. It never reads
poses or GT and does not load raw image pixels/descriptors for fitting.

For each edge choose at most 128 reference or 256 query matches. Source-image
8×8 cell IDs are `floor(8*x/W), floor(8*y/H)`. Within each cell sort descending
LightGlue score, ties by original compact-match index. Fill round-robin across
row-major cells until the cap; save original match indices and all empty/failing
edge outcomes. No depth, pose residual, GT or post-fit criterion affects this
selection. All methods start from these same selected matches.

## Shared reference reconstruction, state and objective

Build reference tracks by confidence-sorted union with deterministic
image/feature-ID ties. Reject a union that creates two different features from
one image in a track; keep counts and all retained edge indices. Query additions
cannot change already accepted reference unions. Each deletion fold rebuilds
its graph from its own retained raw reference edges. Geometry BA counts each
retained track observation once; field BA counts the fixed directed edge
measurements. This known weighting difference is reported, not hidden as exact
objective equality.

Initialize geometry without query edges: choose the anchor-incident partner
with most retained matches, ties by image ID, estimate relative pose, then
triangulate/incrementally register remaining images using PnP. Use the pinned
current PoseLib3 backend with max_error=2 pixels, nested RANSAC seed=0,
min_iterations=100 and max_iterations=1000, otherwise its frozen defaults.
Relative initialization needs at least eight supported matches and a nonzero
baseline; PnP needs at least six supported tracks. Align the positive global
scale by `sum(Z*d)/sum(Z^2)` from positive triangulated anchor depths and valid
whole-image-normalized anchor samples (at least three). No GT alignment.
If fewer than three such samples exist, geometric initialization remains
available: rescale **all** reconstructed translations and points together to
unit norm of the chosen anchor-partner baseline, record
`depth_scale_available=false` and its reason, and permit geometry BA. This
does not add a native call or discard a geometric observation. A malformed
depth/validity input remains a technical failure. When valid depth-scale
support exists, use the original positive LS path unchanged. Field reference
fits require `depth_scale_available=true`; a unit-gauge geometric fallback is
not silently treated as comparable cross-fold depth-field calibration.

Let `u=2*(x+.5)/W-1`, `v=2*(y+.5)/H-1`,
`Phi=[d,u*d,v*d,1,u,v]`, and corrected axial depth `z=Phi*theta`.
Every image has six field coefficients and a world-to-camera pose. Initialize
theta=[1,0,0,0,0,0]. Fix A0 pose exactly identity and theta_A0[0]=1; other first
coefficients have a log-ratio update, other field coefficients additive updates.
The genuine global positive scale is fixed; a common depth shift is **not** a
geometric gauge. Pose charts use left SO(3) increments and additive W2C
translation in the shared normalized-depth scale. Deletion differences use
Log(R_delete*R_full^T), not subtraction of absolute rotation vectors.

Each retained field match contributes both reprojection directions. For each
2D pixel-error block with squared norm s, sum
`rho(s)=log1p(min(s,256)/4)`. Invalid source depth, corrected depth at/below
1e-6, nonpositive/too-small transformed Z, or nonfinite projection receives
the full constant cap, never an omitted term. Preserve all match mass and
invalid counts. Nonfinite declared inputs/state are technical failures.

The solver uses the exact continuous radial residual embedding with squared
norm rho and analytic derivatives; its capped tangential derivative need not
vanish. Separately the information Jacobian is the documented IRLS proxy
`J_pixel/sqrt(4+s)` on valid uncapped blocks, **zero on capped/invalid blocks**.
Do not treat capped embedding derivatives as information or call this proxy a
calibrated covariance. Geometry BA uses the same pixel loss and a sphere
exponential translation chart fixing its initial positive baseline norm.

All fits use the same sparse SciPy TRF/LSMR implementation with analytic
Jacobians, at most 30 accepted updates and 100 objective evaluations, fixed
ftol/xtol/gtol=1e-8. The initial point is not an accepted update. Retain the
exact last accepted finite iterate at a limit and report the limit as
**not converged**. Finite retained accepted iterates, including budget-limited
ones, are used consistently in all arms/folds; do not select only converged
fits after observing outcomes.

## Reference variability and six fixed arms

Fit the complete reference field graph and all eight delete-C graphs per place,
using independent reference-only initializations: 27 field reference fits.
Also fit three complete reference geometry BAs. There is no query warm-start
of a reference fit and no reuse of a full-reference fit as a deletion
initialization. Deleting C removes its node and every incident reference edge.
Each B has only three direct C neighbors; the eight deletion estimates are
dependent and are not eight independent noise samples.

For each query keep the final reference-local 17 coordinates
`[query rotation3, translation3, A0 other-field5, query log-a+other-field6]`.
Project nuisance Jacobian columns away and obtain the identified SVD basis U,
singular values sigma, and B=U/ sigma. Use fixed relative rank threshold1e-8,
with the original kept-Jacobian scale as a floor before nuisance projection.
Flat directions have zero **precision**, never zero variance; all 17 variables
remain free in conditional optimization. Preserve pose–field coupling.

For eight common-chart deletion deltas, set
`V=(7/8)*sum((delta_k-mean(delta))*(delta_k-mean(delta))^T)`,
`W=diag(sigma)*U^T`, `E=PSD_plus(W*V*W^T-I)`.
At valid query endpoint depth samples, form the joint field derivative D and
project each endpoint's output away from span[current corrected depth,1].
This removes global affine recalibration only for the **depth-functional
comparison**, not as a physical pose gauge or pose-observability claim. Let Pi
project onto rowspace(projected_D*B), with rank threshold1e-8 relative to the
largest singular value of **unprojected_D*B**. Empty/affine-only support gives
Pi=0. Invalid depth is excluded only from this operator, not from fitting mass.
Set `E_shape=Pi*E*Pi`, gamma=trace(E_shape)/rank(Pi), or zero for rank zero.

The six arms are:

1. `geometry_multiview`: nonlinear calibrated track/point BA using references
   plus the current query, without a depth field; depth fixes arbitrary scale
   only at initialization when available, otherwise a unit geometric baseline
   fixes scale. It is not made unavailable by missing anchor depth.
2. `joint_field`: full nonlinear same-field robust BA using references plus
   the current query, without covariance intervention.
3. `hessian`: query-only nonlinear field factors plus one joint reference
   precision factor corresponding to B*B^T on its identified subspace.
4. `scalar_deletion`: the same compressed prior using
   B*(I+gamma*Pi)*B^T, trace-matched in whitened shape coordinates.
5. `residual_scalar`: the same prior inflated by
   alpha=max(1,median(min(s,256))/(4*2*log(2))); all reference blocks count and
   invalid blocks have s=256.
6. `directional`: candidate B*(I+E_shape)*B^T.

The candidate changes a local misspecification model; it supplies no new
observations. Shared stable bias is invisible to deletion variability. Full
same-information BA may make it unnecessary. No generic novelty is claimed for
depth fields, multi-view BA, jackknife variability, covariance propagation,
Schur compression, spectral inflation or GLS.

For each of the 24 queries every arm sees **32 references plus that one query**;
the other seven query edges never affect that fit, its prior or initialization.
Compressed arms include the reference prior exactly once and do not also add
reference residuals. Use up to two starts per arm: its reference solution and
the retained-query native-E rotation/direction aligned to the reference
translation norm, when that alternative is defined. Charge/save both; choose
the minimum own objective with start-order ties. Geometry uses its geometry
reference solution; field arms use the shared field reference solution. Thus
the maximum planned optimizer workload is 27+3 reference fits and 24×6×2 query
attempts, not a claim that failures create two available starts.

Also report the seventh estimator, labelled exactly
`native_E_reference_filtered_query`: the fixed PoseLib E estimate on the
query matches retained after reference-track consistency. It is **not strictly
query-only**. No new E estimate or extra random seed is used per arm.

## Failure, observability and complete coverage

Keep all 24 query outcomes in every metric. Insufficient reference support,
unavailable deletion initialization, undefined reference translation direction,
or locally unobservable final rotation/direction uses the frozen native-E
fallback for that arm; if fallback also fails, assign 180 degrees. Geometry
query initializations are handled per start: preserve failed attempts and
choose an available survivor; if all fail, fallback. Ordinary API, source,
shape, arithmetic-contract or serialization errors are technical failures and
stop the operation, not selectively removed observations.
An unavailable full-reference depth scale triggers fallback for the five field
arms while geometry reference/query BA still runs when geometrically possible.
An unavailable depth scale in any deletion fold makes only the two deletion-
covariance arms unavailable; do not compare their parameter differences across
an arbitrary unmatched gauge. Record these separately from geometric failures.

Report local IRLS rank/null diagnostics. For geometry, eliminate each track's
point columns separately, then use the narrow pose Jacobian and its
pre-projection scale floor. At the final pose test rotation and translation
**direction** functional-null ratios separately, both <=1e-8; harmless
translation-length nulls may remain. Recenter geometry information at its final
state. Conditional fixed-rotation charts must also be nonsingular. A zero or
too-small translation has undefined direction. These are numerical diagnostics
with stated coordinate units, not calibrated uncertainty or global
identifiability guarantees. Convergence flags, cap/invalid counts and
nonzero/null shape information remain visible regardless of fallback.

## Prediction seal, GT barrier and evaluation

Prediction inputs include only the 51/120 admitted frontend caches and pinned
sources/library provenance. Save the complete 27 field-reference fits, three
geometry-reference outcomes, 24 query rows with all seven estimators, attempted
starts, native-call records, masks/support, initialization and convergence
reasons, scalar/PSD/rank diagnostics and timing. Bound the exact plan, protocol,
all executed source files, library binary and runtime provenance, frontend and
input receipts, all output/process/log hashes, and the outer supervision.

**No pose download, parsing or GT access before every prediction is sealed and
the outer prediction operation is completed/eligible.** Evaluation first
verifies the entire source/input/output/seal/process/log chain, complete fixed
populations and outer resource success. A partial seal, changed hash or failed
outer run blocks labels. Recheck bindings after evaluation as well.

Only then fetch the three exact `val/s00513/poses.txt`,
`val/s00475/poses.txt`, `val/s00478/poses.txt` members, with the same pinned
archive identity, strict range/header/CRC/SHA checks, at most 2 MiB total
response-body bytes and nine Range requests. Each request remains <=30 seconds.
Do not acquire any other pose member. Parse numeric poses only for the selected
51 image names; derive each A0→B relative pose using the documented authored
pose convention. Calibration and GT are never fitted or offset-tuned to the
predictions. The evaluator implementation and metric convention must pass
artificial checks and be frozen before this access; do not improvise a
convention from observed scientific errors.

The primary error is max(rotation angular error, **directed** translation-
direction angular error), using W2C relative GT. Do not fold the translation
angle or identify t with -t. Failure is 180 degrees. Report standard sorted-
error trapezoidal AUC@5/@10/@20 in
percentage points over all 24 queries and each fixed eight-query scene. This is
a custom calibrated relative-pose development diagnostic, **not the official
Map-free relocalization benchmark or its original metric**. No 24-independent-
sample confidence claim is justified by three places.

## Frozen development decision and resources

Let the comparison set contain the other five arms **and**
native_E_reference_filtered_query. The directional candidate advances only if
all these conditions hold on the complete fallback-inclusive population:

- Overall AUC@5 is at least 2.0 percentage points higher than the best comparison
  estimator's overall AUC@5.
- Against the best comparison AUC@5 **within each scene**, its gain is strictly
  positive in at least two of three scenes, and no scene loses more than 1.0
  percentage point.
- Overall AUC@20 is no more than 0.5 percentage point below the best comparison
  estimator's overall AUC@20.
- All required provenance, coverage and resource checks pass.

Report every arm/scene, harm, fallback rate and descriptive mechanism quantity;
no post-hoc subset, deletion, support cap or scalar/PSD/rank threshold changes.
Failure of the efficacy gate closes this tested idea on these 24 queries. Do
not tune and re-test on the same labels. Technical/resource incompletion is
ineligible, not evidence of efficacy or a licence to loosen the cap silently.
Passing does not admit paper writing or a claim of held-out generalization.

No fees, paid cloud/API calls or new assets beyond the explicit members above.
Each acquisition, frontend, prediction and evaluation operation gets its own
fresh output/receipt/log directory and at most **1,800 seconds**, four CPU
threads and 8 GiB sampled process-group RSS. Account for child processes,
timeouts, actual exits and cleanup. No hidden retry, partial-result rescue,
checkpoint resume or enlarged budget inside this frozen run. Preserve failed
artifacts and any separately justified compatibility revision explicitly.

Charge frontend inference, image decoding/resizing, matching, cache/source
verification, graph/initializer setup, all reference/deletion fits, both query
starts, covariance construction and final validation. Save stage and total
wall times; shared caches do not make their creation free. This development
study does not establish matched-cost superiority, and cached-input fitting
times cannot support a complete-pipeline speed claim. Keep upstream payloads
private; publish only original source, compact derived outcomes, complete
failure accounting and attributed source/protocol notes.
