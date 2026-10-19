# 0013: conditional pose-failure mechanism screen

Primary-source review, 2026-10-19. No actual data or saved prediction values
were read for this note; no candidate was implemented or run. The failed0012
scorer family remains closed. These are two conditional research questions,
not admitted methods, novelty findings, or evidence of pose improvement.

## Sources and reading depth

| Primary source/version | Inspected material and relevant limitation |
|---|---|
| [SIMFIT++, Less Biased Noise Scale Estimation for Threshold-Robust RANSAC](https://arxiv.org/html/2503.13433v1), arXiv v1, 2025; CVPR2025 Image Matching Workshop | Full-text §2.3–2.4 and Algorithm2: separate fitting/validation matches, correction for truncated residuals, and filtered scale estimates. Splitting data to reduce fitting bias is already explicit prior work. |
| [VSAC](https://arxiv.org/html/2106.10240v2), arXiv v2 | Full-text §2–3: independent/dependent inliers, spatial proximity, epipole and epipolar-line dependence, cheirality, and DEGENSAC+. Spatially redundant support is already addressed. |
| [Dense Match Summarization](https://arxiv.org/html/2506.02893v1), arXiv v1; CVPR2025 | Full-text methods and RANSAC-integration ablations: representative-match sampling/scoring and moment-based approximate refinement. DDD/CCC/CCA are relevant complete-pipeline controls. |
| [NONSAC](https://arxiv.org/html/2604.22518v1), arXiv v1, 24 April2026 preprint | Full-text §3.1–3.2 and §4.1: nonminimal subsets, robust estimation per subset, pair/triplet consistency, residual statistics and truncated-Lp selection. Relative-pose experiments in that section use synthetic data and rotation mAA. This does not establish a novel gap for directed translation. |
| [Deploy DINO with Many-to-Many Association](https://arxiv.org/html/2604.23670v1), arXiv v1, 26 April2026 preprint | Full-text §II–III-B, IV-D and V-C: matching-cardinality likelihood, symmetric Harmonic Consensus Maximization (HCM), angular geometric residuals and directed pose evaluation. Its association graph contains alternatives unavailable in our raw matched-coordinate cache. |
| [C2P](https://arxiv.org/html/2312.05995v2), arXiv v2; CVPR2024 | Full-text direction/cheirality construction, pure-rotation experiment and AppendixA.3: pose constraints and near-pure-rotation treatment are prior work; the paper states its method does not handle outliers. |
| [PoseLib2.0.5 scoring](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/utils.cc#L81-L120), [RANSAC acceptance](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/ransac_impl.h#L125-L133), [public-API refinement](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust.cc#L169-L206), [refinement residual](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/jacobian_impl.h#L383-L401) | Read the linked function bodies at tag v2.0.5: scoring checks cheirality; internal final LO is rescored; an additional public-API refinement returns without that acceptance comparison; its objective is robust Sampson loss without a depth term. This is a source-level distinction, not an observed failure. |

Combining known pieces can be useful. A contribution would require a specific
increment beyond these controls and substantial task gains, rather than novelty
being assumed or rejected solely because its ingredients already exist.

## Question1: validation outside correlated spatial support

**Hypothesis.** A coherent wrong region can appear in both sides of a random
point split, making a wrong pose or underestimated noise scale look reliable.
Holding out whole spatial regions, with a buffer around their boundaries, may
test a model's ability to explain evidence beyond the region that generated it.
Available inputs are x1/x2 and K. Spatial arrangement and cross-region residuals
provide usable structure; they are not new observations, known matcher lineage,
or proof of statistical independence.

**Falsifiable increment.** With equal correspondence count, marginal noise and
total hypothesis/optimization budget, vary only spatial error correlation in a
prospective synthetic test. Does buffered regional validation select better
poses than random validation, and does any effect survive independent real
scenes? Model generation must exclude the validation region; scoring an
all-data fit on a nominally held-out region would not test this explanation.

**Closest controls.** Full PoseLib; SIMFIT++ random split and corrected scale;
VSAC independent support; DMS/CCA; matched spatial subsampling without
validation; and random splits with identical budgets. NONSAC is an additional
control if several subset-generated pose hypotheses are retained. Stop if the
gain is explained by subsampling or a different budget, or fails to improve
complete-pipeline directed pose errors. Generic cross-validation/group weighting
is likely not novel; the proposed dependency-specific increment is unestablished.

## Question2: directional validity during final refinement

**Hypothesis.** The additional PoseLib public-API refinement can, in principle,
reduce sign-invariant Sampson loss while losing physically supported direction.
Its input inliers were selected with cheirality, but the final objective does not
keep that constraint. Whether this happens materially in our workload is unknown.

**Falsifiable increment.** A separately authorized, instrumented experiment
would retain the same pre/post refinement states and test whether a noise-aware
positive-depth margin prevents directed translation harms. Use the same initial
poses, matches, optimization budget and output coverage. Compare against the
unchanged public API, no final refinement, simple post-refinement cheirality/MSAC
rescore-and-revert, and standard positive-depth-constrained refinement. C2P and
HCM's geometric residual are relevant prior mechanisms, not evidence that a
new loss is needed. A simple revert is an engineering baseline, likely not a
paper contribution. Pure rotation cannot yield identifiable translation from
these observations merely by changing the loss.

## Prospective saved-output diagnostic: sign versus broad direction error

This diagnostic can use only the frozen0013 evaluation JSON and its sealed
prediction records after their hashes are checked. It needs no estimator call,
HDF5 access, new calibration, or unselected pair. Preserve the original files
and write any approved analysis to a new artifact. It is not executed here.

For each successful row, use its saved directed translation angle theta and
rotation angle r, both in degrees, and report:

- `opposite_translation_error = 180 - theta`;
- `axis_error = min(theta, 180 - theta)`;
- `original_max_error = max(r, theta)`;
- `oracle_sign_only_max_error = max(r, axis_error)`.

Recompute theta from the saved prediction/GT unit translations as an arithmetic
cross-check. The oracle quantity uses GT to select a sign; it is a diagnostic
bound on a hypothetical sign-only correction, never a deployable result or a
replacement for DMS's directed metric. Report all individual continuous values.

Use only the already frozen5/10/20degree thresholds as descriptive strata:
`r < h` and `180-theta < h` identifies a near-sign-only failure at threshold h;
`r < h` and `axis_error >= h` identifies a broad translation-direction error;
`r >= h` records a rotation-coupled error regardless of the translation class.
The remaining `r < h, theta < h` rows meet the original pose threshold. These
four categories partition successful rows for each h in {5,10,20}; near-sign-only
does not mean an exact180degree flip. Keep timeout/error rows and denominator16
visible, with no invented translation angle for missing estimates. An oracle
all-pair AUC, if reported, must retain the original missing-estimate penalty and
be prominently labeled diagnostic only; the frozen baseline AUC stays primary.

This separates error geometry but cannot identify its cause. Final predictions
do not reveal the pre-refinement pose, and returned RANSAC masks/stats are not
that missing pose. Neither small axis error nor the norm of GT translation
establishes pure rotation or low parallax without scene-depth information.
Positive-depth support and causal attribution require a separate prospective
selected-data/instrumentation protocol; they cannot be inferred from this
saved-output classification alone. One scene provides no population confidence
interval or evidence of an independent method gain.

## Observed applicability: saved-output inspection, 2026-10-19

After the preceding prospective note, the parent authorized reading only
[`evaluation.json`](../../results/0013-baseline-calibration-compatibility/evaluation.json)
from the calibration-compatibility continuation. Its inspected SHA256 is
`f8fe46392810b78d485a8dd90ae1849eae6491eaba620d2c9454b6c0f15cb5a4`.
No HDF5, new estimator call or unselected pair was accessed for this update.

All16 estimates succeeded and have max error below5degrees. AUC fractions at
5/10/20degrees are0.8116962248/0.9058481124/0.9529240562. Max errors range from
0.126850 to2.358317degrees; translation dominates15/16 rows. Rotation errors
range from0.036084 to1.014796degrees, and directed translation errors from
0.126850 to2.358317degrees. The proposed5/10/20degree classification gives zero
near-sign-only, broad-translation or rotation-coupled failures at every threshold;
all16 meet each original threshold. Axis error equals directed translation error
throughout, so a GT-oracle sign correction has no effect. These observations
do not support a catastrophic-failure/sign-rescue mechanism on this cohort.
They show fine-grained remaining error, not yet a systematic signed bias.

**Next diagnostic to propose, not run:** explain the objective and support of
the saved final solution before choosing a method. A new bounded protocol
could admit only the same16 pairs' x1/x2 and their already saved calibration,
poses and pre-final RANSAC masks. It would evaluate residuals and derivatives
at the saved poses, without generating hypotheses or rerunning RANSAC:

1. Reconstruct the actual final Cauchy objective on its saved input mask and
   measure its five-dimensional tangent gradient. A near-zero gradient suggests
   objective stationarity; a substantial gradient first warrants an optimizer
   convergence/numerical check. Neither outcome alone establishes pose accuracy.
2. At the identical pose, hold the mask fixed and compare Cauchy weights with
   the already specified truncated-consensus weights; separately hold Cauchy
   weights fixed as a rule and refresh only the final-pose Sampson/cheirality
   mask. This separates loss and support changes. Use their existing nominal
   scales, not a threshold sweep chosen from visible errors. Returned RANSAC
   statistics must not be treated as recomputed final-pose statistics.
3. Decompose signed residuals and tangent-gradient contributions by prospectively
   fixed spatial groups, comparing with count-matched random groups. Coherent
   regional contributions would motivate a dependence question; they could also
   arise from calibration/reference error or model mismatch. They do not by
   themselves establish correlated matcher noise. This is distinct from merely
   reranking hypotheses or counting a dense region fewer times.
4. For diagnosis only, project the translation gradient onto the tangent
   direction from the saved estimate toward GT, reporting the rotation component
   separately. This GT-derived directional derivative asks whether each fixed
   objective locally rewards correction or opposes it. It is an oracle analysis,
   not an inference rule, optimized candidate, or evidence of achievable gain.

Such a fixed-pose derivative audit can distinguish potential loss/support bias
from optimization failure with little computation, but it cannot reconstruct
the missing pre-final pose or prove that the final refinement caused the error.
Causal pre/post attribution would need separate instrumentation. Saved poses and
GT translation norms provide no point parallax or baseline-to-depth ratio;
no low-parallax claim follows. The small errors also make reference-pose accuracy
a relevant unresolved limit. Both candidate questions remain unadmitted.
