# 0015: does image alignment add useful pose information?

Prospective development diagnostic; not yet executed. This is a classical
image-information control before any proposed dependence-aware method. It is
not claimed novel. The final research target remains ≥2 AUC@5 percentage points
at matched complete-pipeline time on independent held-out sources; this reused
one-scene pilot cannot satisfy that target or admit a manuscript.

## Hypothesis, fixed observations and conventions

Some remaining target-coordinate error may be visible in the actual image
intensities, beyond the cached dense matches. Local photometric alignment can
therefore change the observation mean; covariance weighting alone cannot
identify pose-aligned bias. Lower photometric loss is not sufficient evidence.
Compare actual pose estimates using exactly the same selected points and also
the full10000-match baseline. A shared-pixel covariance method is not run here.

Use only the existing16 scene0015 pairs and their23 acquired images. Keep the
0012 manifest, original0013 pose/mask/settings and camera calibrations unchanged.
Use the complete acquired receipt and each selected image's SHA/size/dimensions.
No confidence, depth, R/t HDF5 field, new image, weights or held-out values are
read by prediction. GT evaluation uses the already frozen0013 evaluation only,
after the entire new prediction cohort is sealed.

The documented COLMAP/DKM corner-origin convention gives array sample positions
p_cache−0.5 and q_cache−0.5. Add0.5 back to the refined target before native pose
estimation; preserve originalK. This is a source-grounded prospective choice;
the exact PoseBench cache-producing script remains unlocated. No offset sweep
or result-dependent resize is permitted. Decode originalJPEG with Pillow,
without EXIF transpose or resizing; grayscale is Pillow convert('L') divided
by255 in float64. The decoder version is recorded. Grid centers below are in
cache coordinates; patch/interpolation bounds are in array coordinates.

## Fixed query allocation and affine initialization

Use the saved pre-final RANSAC inlier mask. Divide source image extent into16×16
cells. In every occupied cell choose the inlier closest to its center, with
original index as tie-breaker; at most256 queries. Do not replace a failed query
with another point. Source coordinates outside the paired image extent stop
that pair as an input incompatibility, not a reason to guess a scale.

For each chosen query, take up to32 nearest other baseline inliers within64px
in the source image, tie-breaking by original index. At least8 are required.
Fit A0 by centered least squares from neighbor source offsets to target offsets.
Require source normal condition≤1e6, finite positive determinant of A0 and
condition(A0)≤100. Failures remain explicit. These are engineering choices for
this precision diagnostic, not published defaults or a new matching method.

## Fixed photometric adaptation

For each initialized query, fit I2(q+δ+A u)−g I1(p+u)−b using all961 offsets
u∈[-15,15]². Source p stays fixed. Bilinear samples and their exact same
interpolant derivatives are used. This forward asymmetric adaptation differs
from Making Affine Work's symmetric latent-patch fit; original dense matching
may have bidirectional origins. Do not claim a faithful reproduction of that
paper or DKM training.

Eight parameters are δ(2),15*A(4), gain and bias. Initial δ=0,g=1,b=0,A=A0.
C=0.5sum(r²), g_normal=Jᵀr,H=JᵀJ. Batch at most64. Use float64 and unoptimized
einsum for observation-dependent contractions, with strict finite checks.
LM allows30 total attempts, initialλ1e-3, diagonalH damping floor1e-12;
strict cost decrease accepts a trial and dividesλ by10; rejection multiplies
λ by10, clipped to[1e-10,1e10]. Trials require every sample in bounds,
||δ||₂≤4px, gain∈[.25,4], positive det(A A0⁻¹), and singular values of A A0⁻¹
inside[.5,2]. Invalid trials are rejected, not clipped.

Stop on gradient∞≤1e-8, scaled step norm≤1e-6, or relative accepted cost decrease
≤1e-10. Admission additionally requires no30-attempt exhaustion, positive
Cholesky curvature, diagonal-normalized condition(H)≤1e8, and local undamped
GN decrease g_normalᵀH⁻¹g_normal/(2max(1,C))≤1e-8. Valid stationary zero-update
fits are admitted; count actual target-coordinate changes separately from
changes in gain/bias/A. These criteria check local numerical behavior, not
correspondence correctness or a calibrated error bound. No false floating-point
warning suppression, pseudo-inverse rescue or NaN coercion is allowed.

## Comparisons and complete coverage

First seal every pair's image-correction record, including all query failures,
unchanged points, costs and iterations. Then estimate exactly three outputs per
pair with the pinned original0013 PoseLib2.0.5 and exact0013.SETTINGS:

1. Subset-original: original coordinates of exactly the admitted queries.
2. Subset-refined: the identical indices with admitted target updates.
3. Full-refined: all original10000 pairs, replacing only admitted target values.

The fourth reference is the already sealed0013 full-original output; do not
rerun or alter it. The two subset outputs are conditional on photometric
admission; they diagnose the coordinate update, not an unconditional selection
policy. If fewer than8 queries are admitted, both subset outputs are unavailable
and both receive180degree error in all16-pair metrics. Failed photometric fits
retain original targets in the full arm. No failed pair may disappear from
reporting. A native/full-cohort operational failure makes the final seal
ineligible; preserve partial evidence without a favorable-cohort score.

Each native call runs once, with its input-count and full call cost recorded.
No new mask selection, parameter sweep, extra native continuation or output
selection occurs outside the estimator's unchanged native RANSAC behavior.
After all16 outputs are sealed and hashes/source/calibration verified, compute
signed translation angle, projected-GT rotation angle, max error and the same
AUC@5/10/20 as0013 from the pinned saved labels. Report all per-pair outcomes,
failures, query/admission/change counts, and costs including initialization,
image reading, fitting and native calls. Original baseline API costs remain
visible. This development diagnostic is not a matched end-to-end timing claim.

## Prospective decision

Expansion of this exact sparse replacement configuration requires all of:
complete16-pair full outputs; at least12 pairs with≥8 admitted queries;
full-refined gain≥0.25 absolute AUC@5 points over full-original;
full AUC@20 loss≤0.5point; at least10 of16 full-pair errors reduced by>1e-9degree;
and subset-refined gain≥2 absolute AUC@5 points over subset-original, with all16
pairs in both denominators. No confidence interval from these dependent pairs
is a population claim. Passing only a subset or photometric metric is not enough.

These are early expansion criteria, not a weakened final research target.
At most2.56% of full matches can change; failure rejects this sparse budget/config,
not every possible use of photometric information. Do not immediately tune patch
size, bounds, damping, sample count or positive-pair selection to rescue it.
A different follow-up requires a specific explanation and prospective design.
A positive result would still need strong full-density/direct/affine controls,
novelty evidence and independent validation before any manuscript.

## Operation and preservation

Freeze all source/config/protocol before real fitting. One fresh output directory
`results/0015-photometric/` contains correction records/seal, pose records/seal,
all process logs and evaluation. Serial workers,90seconds per pair process,
30minutes total across both phases, sampled RSS cap8GiB,≤4CPUthreads. Parent
terminates and waits for a timed-out child. No scientific retry is implicit.
Public artifacts contain derived numerical records, no images or patch pixels.
Artificial Jacobian/recovery/degeneracy, query-allocation, binding and cohort
barrier tests plus independent review precede execution. No fee is incurred.
