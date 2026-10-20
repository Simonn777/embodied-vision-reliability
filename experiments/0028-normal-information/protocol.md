# 0028: normal information after camera calibration

2026-10-20. **Draft before scientific execution.** This is a known-control
opportunity test, not an admitted new method or manuscript. The mathematical
relation, angular normal consistency, robust regression, similarity alignment
and normal-guided pose all have prior work. Code preparation and artificial
tests may proceed; scientific processing requires the reviewed complete source
freeze and exact-commit CI. No scientific output has been inspected under this
protocol. The failed 0027 variant remains closed.

## Question and smallest useful result

Does restricting an existing normal-guided depth correction to one image-wide
axial offset preserve useful shape better than full-map normal refinement, and
does that restriction improve actual relative pose after the depth producer
has already used the known camera calibration? The primary is angular scalar
calibration (ANG); ordinary scalar least squares (LS), robust scalar regression
(IRLS), and the shipped full-map normal helper are mandatory controls.

Unnormalized tangency can spuriously reward shrinking depth toward zero. The
angular objective removes this magnitude incentive: on a constant-depth plane
the depth-derived normal direction is constant for every positive offset,
even when predicted normals are tilted. This is a source/mathematical reason
to compare the objectives, not evidence of accuracy or a priority claim.

The 27 images and 24 A0-to-B queries in the existing three Map-free development
places are fixed by the 0023 plan, SHA256
`cc9b02cac8946107f1c945643b316695e5ebbf6339b34baf737541f8cec886d5`.
Places are s00513, s00475 and s00478, eight queries each. All are development
forever. No images, weights, packages or data are downloaded; no held-out
source is opened. Reference C images/edges are not used for fitting this study.
All 51 image/120 edge cache identities are verified before reading the selected
27 image/24 edge arrays, to reuse the already qualified frontend provenance.

## One fresh raw forward, three camera controls

The old cache stores projected points, losing the original learned X/Y. It
cannot faithfully redo the producer's known-FoV shift recovery. Therefore run
exactly one new raw MoGe2 forward per selected image, with the same original RGB
preparation, frozen checkpoint, CPU float32 and 1,200 tokens. Save raw points,
normal, mask probability and metric scale before postprocessing. No new
SuperPoint or LightGlue call; reuse the fixed matching outputs.

Each raw output supplies three separately recorded controls without further
network execution:

- A: official inferred-FoV focal/shift recovery.
- B: official known horizontal FoV, `2 atan(W/(2 fx))`, using authored resized
  fx. The official implementation assumes a centered principal point and
  square pixels; B does not claim support for the complete authored K.
- C: a general-authored-K adaptation of the same upstream scalar shift fit.
  Use the exact same nearest 64-by-64 sampling indices and mask, actual
  pixel-center rays from K, and upstream `solve_optimal_shift(ray_xy, raw_xyz,
  focal=1)`. This fits normalized-ray reprojection, not a new learned model
  or a new solver. K has no skew in this source. Record numerical convergence
  limitations of the upstream fit and reject nonfinite outputs.

Depth is `(raw_z+shift)*metric_scale`. Preserve the probability mask and each
positive-depth output mask separately; normals are never differentiated from
these postprocessed depths. Model sharing does not establish independent error.
All three point controls and the source functions are bound by hashes. Any
upstream/API failure is retained and cannot silently remove a baseline.

## Scalar corrections and full-map control

Apply every correction to C. Normalize its depth by the median of all
model-mask-valid finite positive values. Use authored pixel K, rays r with
third coordinate one, and E equal to the first two columns of inverse K.
Sample a fixed grid starting at (2,2), stride four, through W/H minus three.
Each eligible sample needs a full 3-by-3 stencil of model-valid finite positive
depth and finite nonzero normals, and stencil maximum/minimum depth <=1.1.
Use central +/-1 pixel differences for depth gradient g. Normalize each
observed normal; use identical eligible sample IDs for LS, IRLS and ANG.

Let k=E_x cross E_y and ell=g_y(E_x cross r)+g_x(r cross E_y).
The corrected surface normal is parallel to v(b)=(d+b)k+ell. The shared domain
is `[-min(all valid normalized depths)+1e-6, 4]`, with zero required inside it.
All original valid depths must remain positive; no mask is changed after fit.

- ANG minimizes the mean of `1-(unit(v(b)) dot n)^2`, invariant to normal sign.
  Evaluate 65 uniformly spaced domain points plus zero, then one bounded
  scalar minimization in the neighboring grid interval of the best point,
  maxiter64/xatol1e-8. Keep all losses and convergence status, use fixed
  generation order for ties except a flat curve chooses zero. A total grid
  loss spread <=1e-12 records uninformative/b=0, not successful calibration.
- LS uses a=n^T E and y=-d*a-(n dot r)*g, and
  b=sum(a dot y)/sum(||a||^2). Out-of-domain or degenerate fits are unavailable;
  no clipping, mask rescue or outcome-selected thresholds.
- IRLS starts at LS and performs five Huber updates on each sample's 2-vector
  residual. Scale is 1.4826 times the median absolute deviation of residual
  norms around their median; Huber delta=1.345 times scale. Zero scale stops
  updates with a recorded reason. Final inadmissibility remains unavailable.
- FULL calls the pinned MoGe `refine_depth_with_normal` unchanged with its
  defaults: ten iterations, damp1e-3, kernel5. Convert actual pixel K to
  normalized half-pixel K correctly; copy inputs because the helper mutates
  depth during clamping. Keep its complete warnings/nonfinite/validity record.
  Source review found that the pinned helper calls a missing utils3d
  `sliding_window_2d` name. A scoped compatibility alias to the existing
  equivalent last-two-axis `sliding_window` is allowed only after an explicit
  manufactured window-equivalence check and real-helper qualification. Preserve
  the original missing-name observation; do not edit pinned files on disk.
  Record the alias in every helper invocation. This changes callable API
  compatibility, not the numerical update. Any other repair requires review
  before scientific execution and must not silently weaken the comparator.
  The helper does not accept a validity mask; do not conceal that limitation
  or replace it with a different algorithm under the upstream name. If this
  causes a failed comparator, a positive result cannot pass the opportunity
  gate until a separately specified trustworthy same-information comparator
  is available. No weakening a comparator to claim a gain.

Raw/full corrected maps are sampled by the frozen four-neighbor bilinear rule.
Full maps and input pixels remain private. Publish original code, per-image
calibration records, per-pair poses, failure-inclusive errors and operation
receipts; do not rehost upstream code, images, GT arrays or weights.

## Shared pair support and pose controls

Use the original 0023 8-by-8 source-cell, score-ordered round-robin selection,
cap256 per query, independent of depths/normals/errors. Recompute one native E
pose per query on all selected matches using current pinned PoseLib, seed0,
max_error2px, min_iterations1000/max_iterations100000, normalized Sampson
(tangent_sampson=false), success_prob.9999, dyn_num_trials_mult3, no PROSAC,
otherwise the recorded current defaults. This is query-only support; do not
reuse the earlier reference-filtered 0023 E mask and imply equivalent inputs.

The unmodified E output remains a separate full-information baseline. The
common correction-fit support is its returned inlier mask intersected ONCE
with original A/B/C four-neighbor depth eligibility and predicted-normal
eligibility at both endpoints. Corrected values never alter that support.
Retain the mask and every support index. At least four supported matches are
required for the common fixed-support comparison. A method-unavailable query
falls back to this same E pose, explicitly counted; an unavailable E gives
180-degree error. No query is dropped.

Raw A/B/C, C+LS, C+IRLS, C+ANG and C+FULL all use the same closed-form proper
similarity alignment on that common support. For X=z1*r1 and Y=z2*r2, minimize
||R X+t-a Y||, a>0, using centered proper SVD, a=proper_trace/||Y_centered||^2,
t=a meanY-R meanX. No new RANSAC is attributed to this fitting stage.

Mandatory same-normal controls: proper normal SVD, followed by the smallest
singular vector of the epipolar-plane system for translation; select the sign
by positive triangulated depths, with deterministic ties. Refine normal and E
starts using joint normal-chord residuals divided by sin(5 degrees) and signed
pixel Sampson residuals divided by2px, max_nfev100. Both translation signs and
all starts/status/costs are retained. This is known normal-guided pose geometry.

Mandatory free-affine control: use the already qualified 0026 native
TRUNCATED_CAUCHY refiner with max100 iterations and exactly its recorded source
settings. Starts include E (unit translation, a=1,b=c=0), C raw similarity,
and every admissible LS/IRLS/ANG similarity converted into the raw C gauge.
The FULL output is not globally affine in C and does not supply a falsely
equivalent calibration seed. Keep all returned branches and source/native
errors; a native ABI/programming failure fails the operation. Numerical
unavailability is retained, not rescued by hidden retries. The free-affine
arm includes the E incumbent and every admissible refined result, choosing
the same pixel MSAC score on ALL selected original matches, threshold2px,
then generation order. Normal-refinement starts use that same selection rule.

This support-conditioned study cannot establish robust sampling superiority.
Scalar offsets would be an exact reparameterization if subsequently freed in
the same estimator; any gain here must be attributed to the restriction and
its source of information, not a new pose family. All frontend, correction,
support, fitting, selection, serialization and failed-call costs are recorded.
Cached feature/match timing is historical; it is not a new complete-pipeline
matched-speed measurement.

## Prospective decision and execution boundaries

Complete all27 raw forwards and all24 query outcomes before evaluation. Seal
the inputs, source revisions, all method predictions and operation receipts;
require successful outer completion before the evaluator reads the already
development 0023 ground-truth poses. Report directed max(rotation,translation)
error, AUC@5/10/20 using the existing DMS trapezoid convention, success/fallback
coverage, every per-scene result and paired differences. No population CI from
three development places is claimed. Independent saved-output code recomputes
the metrics and gate without refitting any estimate.

All following conditions are necessary to continue this fixed ANG candidate:

1. All required controls are technically trustworthy and complete; no win can
   be explained by a missing or invalid stronger comparator.
   Prespecified shared-support failure (unavailable E or fewer than four common
   matches) permits the same fallback for every non-E arm and cannot create
   an ANG advantage on that query. It is distinct from a method-specific FULL
   map/fit failure, which invalidates the comparator clause.
2. ANG has at least +2 absolute AUC@5 percentage points over the strongest
   prespecified control in the pooled24 queries, including raw C and the
   free-affine/same-normal arms. At least20/24 ANG outcomes must be genuine
   fitted results rather than fallback.
3. At least two of three scene margins against the strongest pooled control
   are positive, and no scene loses more than2 AUC@5 points.
4. Pooled AUC@20 trails no prespecified control by more than0.5 points.

Failure closes this fixed formulation without changing its sample grid,
bounds, losses, seeds, camera controls, scene/query set or thresholds. A pass
only justifies a distinct contribution assessment and separately frozen larger
validation; it is not novelty proof, final held-out evidence or manuscript
admission. The main gate remains >=2 AUC@5 points at matched complete-pipeline
cost on two independent held-out sources, positive scene-paired95% intervals,
<=0.5 AUC@20 loss, strong official controls, ablations and replication.

Use existing runtimes only, four CPU threads,8GiB process-group limit. Each
operation has an outer hard limit <=1800seconds and private fresh output.
Qualification uses manufactured fixtures only, never the scientific grid.
Raw inference, pose prediction, evaluation and independent audit are separate
operations. No paper is drafted from this preparation or a negative result.

## Preparation access disclosure

Before this source freeze, a read-only schema inspection of the existing 0023
evaluation file used JSON decoding for the full document and printed only
top-level keys and first-row field types. That operation necessarily decoded
the historical numeric ground-truth subobjects, although none were printed,
copied, used for fitting, or used to compare 0028 predictions. These are already
development labels with previous published evaluations; this study is not blind
to the researchers' historical development results. The future predictor must
still exclude GT access, and the new evaluator must still require a complete
clean prediction seal. No held-out payload or 0028 accuracy was inspected.
