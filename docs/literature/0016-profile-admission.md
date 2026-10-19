# 0016 bounded cost-profile proposal: NO_GO

Decision: do not run the proposed 49-point translation-grid / ground-truth-nearest-candidate diagnostic. This is a mathematical admission rejection, not an experimental negative result. No images, labels, scientific predictions or result files were read, and no implementation or experiment was performed for this review. The frozen 0015 experiment is unchanged.

The considered proposal fixes the initialized affine warp A0, profiles gain and bias, evaluates the 49 integer translation offsets with Euclidean norm at most four pixels, and uses a deterministic query cohort. These choices remove some tuning freedom, but do not resolve the following identification problems.

## Candidate multiplicity produces oracle gains without image information

If the original coordinate is among the alternatives, selecting the smallest ground-truth epipolar error can never make that error worse. More alternatives create more opportunities to approach the line, even if their positions are unrelated to image evidence.

A simple counterexample makes the problem explicit. Suppose K candidate signed distances to a known epipolar line are independent uniform samples on [-a,a]. Their absolute distances are uniform on [0,a], and

\[
\mathbb E\big[\min_{1\le j\le K}|r_j|\big]=\frac{a}{K+1}.
\]

This is a toy calculation, not a stochastic model of the proposed grid. It proves that a decreasing oracle residual alone is not evidence of recoverable photometric information. A finite grid does not prevent this effect, nor does calling its selected candidates modes.

If such an oracle analysis is ever used as a secondary diagnostic, all basin representatives must first be determined from image evidence without ground truth. Candidate-count, displacement-radius, support and boundary-matched nulls are necessary: for example, shuffling profile costs among offsets, or rotating/permuting candidate directions while retaining their radii and multiplicities. Even beating those nulls would establish only an association with the epipolar direction, not correct correspondence or improved pose.

## Epipolar compatibility cannot identify the correct point

For a fixed source point and relative pose, its target correspondence lies on a one-dimensional epipolar line. Different positions on that line represent different latent scene structure. Cheirality can exclude part of the line, but generally does not identify a unique point. Repeated texture can give equal photometric costs at multiple valid positions; wrong matches on the same line can satisfy both tests. Epipolar costs also cannot distinguish the sign of translation by themselves.

Consequently, choosing the point closest to the ground-truth line is not an oracle for the true correspondence. It can preferentially choose an arbitrary small shift or the wrong repetition. Independent correspondence truth, an additional view with suitable visibility, or another independently justified constraint would be needed to resolve those cases. No such evidence is supplied by this proposal.

Constraining every refined correspondence to the *initial predicted* epipolar line does not solve the problem. It makes that initial pose exactly consistent with the adjusted subset. The resulting geometric agreement is circular; the adjusted coordinates do not independently demonstrate that the pose should change. A genuine joint objective must compare photometric evidence as the pose changes.

## Required comparison with a quadratic / JET control

Generic joint optimization of photometric patches and two-view pose is already covered by [Joint Epipolar Tracking, WACV 2017](https://arxiv.org/pdf/1703.05065), full-text §§3.2–3.3, Eqs. 5–15, reviewed previously. Local cost-surface caching and featuremetric geometry refinement are also established in [PixSfM, ICCV 2021](https://arxiv.org/html/2108.08291v1), §§4.3–4.4. The proposal's novelty is not established.

For a target-coordinate quadratic cost with positive-definite curvature H,

\[
L(q)=\tfrac12(q-\mu)^TH(q-\mu),\qquad a(\theta)^Tq+c(\theta)=0,
\]

elimination gives

\[
\min L(q)=\frac{(a^T\mu+c)^2}{2a^TH^{-1}a}.
\]

This is covariance-weighted geometric refinement. A future nonquadratic-information claim must beat a control retaining the full two-dimensional coordinate curvature and the same coordinate optimum. At a regular local optimum, profiling nuisance variables gives the Schur-complement curvature; treating the coordinates as isotropic would be an artificially weak control.

All arms must use the same pixel support, coordinate convention, initialized affine treatment, photometric loss, gain/bias bounds and displacement domain. Comparing a fixed-A0 full profile with a quadratic approximation from an independently optimized affine model confounds likelihood shape with nuisance treatment. JET's repeated relinearization must be distinguished from a single frozen quadratic. A one-pixel grid also cannot exclude narrower subpixel modes; any negative conclusion would be limited to the sampled domain and resolution.

## What could falsify a narrower claim, without authorizing a run

Artificial fixtures with known correspondences can test whether a proposed profile representation detects recoverable separated modes at all. Necessary negative cases include a single quadratic basin, independent localization noise with no spatial bias, flat/aperture-limited texture, and repeated patterns whose competing positions are indistinguishable photometrically and epipolarly. The last case must remain ambiguous rather than produce a confident correction. Any image-only candidate selector must be frozen before ground-truth evaluation, with every failure included.

For pose relevance, a diagnostic would need image-driven selection among pose hypotheses proposed without ground-truth assistance, or another independently grounded prediction task. Merely minimizing ground-truth point residuals is insufficient. A profile's departure from a quadratic may instead reflect occlusion, incorrect affine warping or illumination-model failure; nonquadratic shape is not by itself useful information.

These requirements describe the missing evidence. They do not admit an alternative 0016 experiment. Given the failed 0015 sparse control and the lack of a non-oracle mechanism that separates useful ambiguity from these confounds, the current 49-grid proposal should be closed rather than expanded into joint optimization or correction propagation.
