# 0019 prospective screen: depth calibration transfer across spatial groups

2026-10-19. Bounded primary-source screen; **not an experiment admission or a
new-method claim**. No scientific arrays, images, reference poses, estimator
calls, or builds were accessed. No frozen 0018 file was changed. Leave this
prospective note untracked until the parent separately decides its disposition.

**Verdict:** there is a precise, inexpensive diagnostic question, but currently
no defensible distinct CVPR method. Generic affine-depth rescoring is already
MADPose; generic marginal-likelihood scoring is also covered by a very close
2026 paper. Spatially cross-fitted calibration could expose a failure that these
simple controls miss, but that specific benefit has not been demonstrated and
no absence of search hits establishes novelty. The approximately 9.20-point
0017 oracle gap motivates investigation; it is not evidence that any available
score can recover it.

## Closest primary sources, read depth, and overlap

1. **MADPose, CVPR 2025.** Read [arXiv v3 §§4.3–4.4, Eqs. 7–10 and Table 7](https://arxiv.org/html/2501.05446v3#S4.SS3).
   Its point-only proposals are triangulated and their depths fitted to each
   monocular prior by least squares. Both directional depth reprojection costs
   and Sampson cost then score all correspondences. Its hybrid sampling and LO
   use both information sources. Thus “calibrate depth for every geometric
   candidate and use depth to choose among candidates” is already explicit.
   The potentially additional question below concerns *transfer between spatial
   groups*, not these ingredients. Table 7 already isolates hybrid scoring as a
   useful component, so the direct same-bank version is an essential control.

2. **Pritts, Seegräber and Köser, RANSAC Scoring Done Right.** Read [arXiv v1 §§2–3, Eq. 7 and sort-and-sweep argument](https://arxiv.org/html/2606.27385v1#S3).
   It analytically integrates the inlier noise variance under an inverse-gamma
   prior for a fixed inlier partition, then maximizes over partitions. It also
   discusses label priors and the difference between profiling and marginalizing.
   This directly preempts an unspecific “Bayesian score that removes the noise
   threshold” contribution. It does not, in the inspected formulation, eliminate
   the two images' affine depth-calibration parameters or validate their spatial
   transfer. The [author's current page](https://prittjam.github.io/) reports
   NeurIPS 2026 acceptance; this review inspected the v1 paper, not an independently
   verified camera-ready version or reproduced implementation.

3. **Shekhovtsov, RANSAC Scoring Functions: Analysis and Reality Check.** Read
   [v1 §§6–8 and the stated limitations](https://arxiv.org/html/2512.19850v1#S8).
   The paper isolates scoring from proposal/LO changes and reports similar
   performance for several residual-only scores under its validation protocol.
   It explains why a large pose-angle oracle gap need not be recoverable by
   another residual transform. Its cross-validation is across validation/test
   image pairs for threshold selection, not the proposed within-pair calibration
   split. These are the author's findings for the evaluated settings, not a
   theorem that every score is equivalent.

4. **RePoseD, ICCV 2025.** The [reviewed v3 paper](https://arxiv.org/html/2501.07742v3)
   and pinned implementation already jointly handle relative pose and depth
   scale/shift, and acknowledge favorable scoring of physically impossible
   solutions. This blocks a broad claim that recognizing unreliable affine-depth
   solutions is itself new. Positive-depth/fixed-support repairs remain the
   known controls discussed in the separate 0017/0018 notes.

5. **Two-View Geometry Scoring Without Correspondences, CVPR 2023.** Read the
   [official abstract and PDF §3](https://openaccess.thecvf.com/content/CVPR2023/papers/Barroso-Laguna_Two-View_Geometry_Scoring_Without_Correspondences_CVPR_2023_paper.pdf).
   FSNet already uses image evidence to rank pose hypotheses when correspondence
   scoring is inadequate. It requires a learned scorer and image features;
   cached depth is a different input channel, but “new information beyond
   epipolar residuals helps candidate selection” is not a new general claim.

A newly located related constraint on future LO claims is [Robust Local
Optimization Done Right, arXiv 2610.05743](https://arxiv.org/abs/2610.05743).
Only its primary abstract and author-page summary were read here. It explicitly
separates ranking and refinement objectives and evaluates degradation of accurate
seeds. The author reports ACCV 2026 acceptance. Its existence is a warning against
claiming objective matching or a broader robust-loss basin as the new mechanism;
this review does not claim it studies affine-depth visibility omission.

## One concrete formulation, not yet a proposed algorithm

The question is: **when competing poses have similar epipolar support, does the
correct pose admit an image-wide affine depth calibration that transfers across
spatial regions, while a wrong pose fits one region's correlated depth errors?**

For a fixed candidate `(R,t)`, set the translation baseline gauge to unit length
only for this newly defined score; do not overwrite an estimator's raw geometry.
Let h1,h2 be z-one calibrated rays. Write corrected depths in this gauge as
`a1*d1+b1` and `a2*d2+b2`. A fixed-pose calibration residual is linear in four
nuisance coefficients beta:

\[
A_i(R)\beta+t,
\quad A_i=[d_{1i}Rh_{1i},\ Rh_{1i},\ -d_{2i}h_{2i},\ -h_{2i}],
\quad\beta=(a_1,b_1,a_2,b_2)^T.
\]

Given a candidate-independent spatial partition B1,...,BK, fit beta on all
groups except Bk and evaluate its bidirectional depth-induced **pixel**
reprojection on Bk. A simple falsification version uses full-rank linear least
squares for the four coefficients, followed by the existing fixed 16-pixel
capped reprojection diagnostic; invalid projections receive the cap and remain
in the fixed denominator. Use both directions with equal weight. The point-only
LS fit is deliberately a known control, not a claim of robust optimality. Robust
fitting, positivity constraints, fold boundaries and tie/failure behavior would
all require a prospective specification before any execution.

This avoids unstable per-point triangulation as the fitting primitive, but
linear fixed-pose calibration is an ordinary least-squares elimination, not a
novel theorem. The proposed information is the *spatial transfer gap* between
same-point and disjoint-group calibration, with cached `(d1,d2)` values supplying
information absent from a pure epipolar score. If the held-out score only changes
because it uses fewer points or reweights dense regions, the claimed mechanism
fails. If all image-wide depth distortions are affine, a global affine fit may
already capture essentially everything and splitting merely adds variance.

Do not call this a calibrated likelihood. Image-derived depth errors can be
spatially correlated, heteroscedastic, non-affine and correlated with matching
errors. A product of per-point densities would overcount evidence without a
validated model. Marginalizing affine coefficients requires proper,
gauge-consistent priors and the associated normalization/Jacobian terms; an
arbitrary flat prior or dropping determinant terms can change rankings under a
change of depth units. Merely appending a log-determinant is not a justified
Bayes factor. Noise-scale marginalization alone is already covered above.

## Critical identifiability and leakage limits

- **Nuisance-only splitting is not independent pose validation.** All four saved
  0017 candidates used the complete correspondence set; the depth candidates
  also used the full depth inputs. Holding out points only from beta fitting
  cannot undo that dependence. It is a transfer-sensitivity diagnostic. A genuine
  out-of-sample pose test would regenerate candidates without each validation
  group, changing both the candidate bank and total cost; it is not authorized
  by this note and must not be quietly called the same experiment.
- A narrow depth range, pure rotation, near-zero parallax or nearly dependent
  columns of A leaves calibration/translation poorly identifiable. A small
  fitted residual then need not imply an accurate translation direction.
  Rank failure must be explicit; pseudoinverse regularization is not evidence.
- If a wrong pose and its calibrated depth field explain the same observed
  rays/depths, no scorer using only those quantities can distinguish it. Coherent
  scene-wide non-affine bias can transfer between folds and defeat this test.
  Shared frontend errors and near-planar scenes are plausible examples, not
  established diagnoses of these 80 pairs.
- Independent spatial folds may be impossible if matches cover one small
  textured region. Repeated measurements in neighboring cells are not independent
  evidence. Random point splitting is a useful negative control because it
  typically shares the same local error field between train and test.

## Decisive test if separately admitted

First use manufactured 3D scenes with frozen candidate poses: (a) only global
affine depth changes, (b) a localized smooth depth distortion, (c) the same
distortion made global, (d) a planar/low-parallax ambiguity. Hold candidate poses
and matched points fixed across scores. Apply arbitrary positive depth-unit
changes and corresponding shifts to confirm invariance. Duplicate matches
within one spatial region and separately permute depth-to-coordinate assignments
while preserving each image's depth histogram. The score should not manufacture
confidence from duplicates; any genuine depth-association benefit should be
destroyed by the permutation. Do not select perturbation amplitudes based on the
80 real labels.

Required same-bank controls are: pixel MSAC; global affine-calibrated
bidirectional depth score; MADPose-style hybrid score with published settings;
random-point cross-fitting; spatial weighting/subsampling without cross-fitting;
and the actual spatial transfer score. A later full-pipeline claim also requires
official MADPose and the recent marginal-score baseline, not only an imitation
on four saved outputs. Keep the bank, work accounting and denominators fixed.

The diagnostic is worthwhile only if a prospective test shows that spatial
transfer contributes beyond those controls, with all cases retained. Reject it
if the apparent gain disappears with same-point weighting, if random folds do
equally well, if wrong candidates transfer just as well, or if selected accuracy
improves only by tuning folds/losses on exposed labels. This would still be a
development observation; the unchanged substantial-gain and independent-source
requirements apply before a paper claim.

For four candidates and a fixed small number of folds, the fit uses four-column
QR/normal-equation sufficient statistics and O(KMN) work, with O(N) working
memory. No neural inference or new images are intrinsically required. That is a
plausible low-cost implementation, not a measured runtime promise. Robust
partitions, uncertainty propagation or pose regeneration can dominate the cost.
At present, the strongest justified next action is to benchmark the existing
MADPose scoring ingredient before introducing a new transfer mechanism, under a
separate frozen plan. **No experiment or distinct method is admitted here.**
