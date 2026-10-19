# 0020: spatial transfer of affine depth calibration

Prospective development design, 2026-10-19. Scientific execution requires the separately published source freeze, its passing exact-commit CI, and the preparation review. No 0018/0019 source, result, protocol, or candidate is changed. All numbers below are fixed engineering choices or inherited thresholds, not selected by testing the 80 labels.

## Question, scope, and information

Can a pose be selected more accurately when its affine depth calibration must transfer across image regions, instead of fitting and scoring the same points? The additional observable is regional transfer of the cached depth–correspondence relation. The proposed contribution is its validated use for pose selection; ordinary least squares and cross-fitting are not claimed as new mathematics.

Use exactly the existing 0017 manifest: 80 pairs, five permanently development scenes, and the four sealed poses in fixed order `e_norm,e_ts,depth_scale,depth_affine`. No pose refinement, restarts, new depth inference, images, confidence model, or candidate replacement. The two fixed 0019 author-loss outputs enter only the benchmark comparison table after a complete eligible 0019 evaluation; they never enter the candidate bank or any selector. Candidate-bank expansion requires a new prospective protocol. This comparison requirement was set before any scientific 0019 call. All candidates already saw the full input, so this is **nuisance-calibration transfer**, not independent validation of pose estimation or held-out scene evaluation.

Verify the 0017 seal, manifest, source/library provenance, all 320 records and input hashes before reading allocated x1/x2/K/camera/depth values. Read no R/t labels in scoring. Only the selected 80 keys are permitted. Use the existing PINHOLE/axial-depth consumer contract, including finite negative/zero raw depths. Nonfinite input or provenance failure is a technical blocker, not an excluded pair.

## Fixed geometry, folds, and weights

1. Validate each saved R and nonzero t; use `t_unit=t/||t||` only inside this new scorer. Preserve the saved raw geometry. Let h1,h2 be calibrated z-one rays.
2. Compute each candidate's common 0017 per-point 2px pixel-F Sampson cost c_g in [0,1], including its fixed 0.01-unit-baseline cheirality check and saturated arithmetic degeneracy. Set G=mean(c_g). Require the recomputed sum to match the sealed pixel_msac within `1e-10*max(1,N)` before continuing. Do not use native returned masks. The common fit-support mask is c_g<1; it depends on candidate geometry, never on depth or labels, and is identical across all calibration controls for that candidate. Validation always includes all matches.
3. Form four **candidate-independent source-image rectangular folds**. Let u0,v0 be the lower medians of all source x/y coordinates: sorted element index floor((N-1)/2). Set B_i=2*1[x1_i.x>u0]+1[x1_i.y>v0]. Equal-coordinate ties stay together. Do not adapt boundaries using depth, pose, residuals, target positions, or outcomes. This asymmetric choice tests source-image regional transfer; target regions need not be disjoint.
4. If every fold is nonempty, define fixed spatial weights w_i=1/(4*n[B_i]), summing to one. Otherwise record `spatial_partition_unavailable`; all three spatial-weighted selectors below use D=1 for every candidate, so their real-arithmetic ordering is equivalent to G. The declared floating-point argmin remains unchanged: adding one can collapse sub-ULP differences into ties, which use the fixed arm order. The uniform-global control can still operate. No pair is dropped.
5. Create one random-point four-fold control with exactly the same fold sizes n0,...,n3. Sort original match indices by SHA256 of UTF-8 `evr0020-random-v1\0` + pair key + `\0` + decimal index, breaking hash ties by index; assign consecutive chunks of those sizes. It uses the SAME w_i from the spatial partition, not new random-fold weights. No seed search.

## Affine normalization and the sole fitting primitive

For each image j, once per pair, compute mean mu_j and population RMS deviation sigma_j of all raw depths. Set z_ji=(d_ji-mu_j)/sigma_j. All computations use float64, explicit finite checks, and non-BLAS contractions (`einsum(...,optimize=False)`) where required by the documented local NumPy issue. A zero sigma or nonrepresentable arithmetic makes calibration unavailable, with D=1; never replace it by an arbitrary epsilon. Save normalization and failure status. Full-input normalization is only a coordinate change, not a held-out-data independence claim.

The corrected depth is c_ji=alpha_j*z_ji+beta_j. For any declared training set T, fit theta=(alpha1,beta1,alpha2,beta2) by

\[
\min_\theta \sum_{i\in T}q_i\|A_i\theta+t_{unit}\|^2,
\quad A_i=[z_{1i}Rh_{1i},\ Rh_{1i},\ -z_{2i}h_{2i},\ -h_{2i}].
\]

T is the specified training fold complement intersected with the common fit-support mask. q_i is either uniform or the specified w_i, normalized over T; normalization does not change the minimizer. Use ordinary SVD least squares on the stacked sqrt(q_i)-weighted 3-by-4 blocks and repeated target -sqrt(q_i)*t_unit. No IRLS, ridge, trimming, nonlinear optimization, or constraint search.

Require at least four scalar equations, finite factors/solution, and numerical rank four using `s_min > eps64*max(3*|T|,4)*s_max`; equality fails. Require alpha1>0 and alpha2>0. Rank deficiency, nonpositive fitted slope, empty fit support, or numerical failure means a failed fit; all its validation directional costs equal one. Record the coefficients, singular values, rank, support count/mass, training residual and exact failure reason. Do not silently use a pseudoinverse solution, relax the threshold, or substitute another pose.

This normalization preserves the score mathematically under independent positive affine changes d'_j=a_j*d_j+b_j, a_j>0: normalized z is unchanged, as are the fit, positive-slope decision, projections and score. Translation magnitude is absent after unit-gauge normalization. Floating-point tests cover representable transformations, not extreme shifts that erase input information through rounding.

## Fixed validation cost and one selector

For a successful fit, compute

\[
Z_{12}=R(c_1h_1)+t_{unit},\qquad
Z_{21}=R^T(c_2h_2-t_{unit}).
\]

Forward cost is min(||pi2(Z12)-x2||/16,1)^2, with reverse analogous. A direction costs exactly one when its source corrected depth is nonpositive/nonfinite, any transformed coordinate is nonfinite, transformed z<=0, or projection/distance is nonfinite. Invalid directions remain in the denominator; no visibility or validation-mask refresh. The requirement on source corrected depth is an explicit physical guard shared by every control, not claimed identical to native RePoseD.

For each spatial fold k, fit on all other folds and score only B_k. Let ell_i be half the sum of the two capped directional costs obtained from i's omitted-fold fit. Define

\[
D_{spatial}=\sum_i w_i\ell_i,\qquad S_{spatial}=G+D_{spatial}.
\]

Choose the candidate minimizing `(S_spatial, fixed_arm_index)`; no numerical tie band. D is always finite in [0,1], S in [0,2]. A failed fold contributes the full cost of its entire validation mass. All-failed calibration adds the same unit penalty to every candidate, without deleting the pair; real-arithmetic ordering is equivalent to G, subject to the same floating-point tie qualification above. No threshold on the transfer gap, no post-selection refinement, and no oracle fallback.

## Four mandatory same-bank controls

All controls share raw inputs, candidate geometry, per-candidate fit-support mask, calibration primitive, normalization, positivity/rank checks and 16px invalid-cost convention. The geometric component is the identical uniform G in every hybrid score.

| Selector | Fit and validation | Final ranking |
|---|---|---|
| `pixel_msac` | Existing independent geometric cost only | G |
| `global_affine_ls` | One uniform fit on all fit-support points; validate every match uniformly | G + mean(ell) |
| `spatial_weight_only` | One global fit weighted by w; validate every match with w; no omitted region | G + sum(w*ell) |
| `random_point_transfer` | Four omitted random-fold fits using the SAME w; validate each point once with w | G + sum(w*ell) |

The primary `spatial_transfer` differs from `random_point_transfer` only in region membership. It differs from `spatial_weight_only` only in disjoint-region calibration fitting. This controls nominal omitted-fold size and spatial density weighting. Candidate-dependent geometric fit support can still give spatial and random folds different actual training counts, masses and rank failures; report those for every fit. A score gain alone therefore cannot establish that transfer, rather than effective sample size or fit failure, caused the gain. A later mechanism claim requires a separately specified support-matched diagnostic. Save every control, not merely the weakest comparator. Also report all four fixed candidate arms.

[MADPose, CVPR 2025, inspected v3 §§4.3–4.4/Eqs.7–10](https://arxiv.org/html/2501.05446v3#S4.SS3) already calibrates depth for point-based poses, combines bidirectional reprojection and Sampson scores, and performs hybrid LO. Its point-based calibration triangulates points and then fits affine depth to those depths. **Our fixed-pose 3D-consistency LS is a different fitting objective; `global_affine_ls` is not an exact MADPose reproduction.** The equal normalized geometric/depth weighting inherits the published hybrid form with lambda_s=1 and the established 2px/16px thresholds. Full official MADPose remains mandatory before a later contribution claim. A gain over this bank alone does not supersede that comparison.

## Artificial qualification and execution budget

Before any scientific read, manufactured fixed-pose tests must check affine-depth unit/shift invariance, positive rescaling of saved t, independently known exact calibration, positive/negative direction cases, zero-depth spread, rank deficiency, empty spatial folds, fixed denominator, random-fold determinism, finite serialization, tie order, and score-seal tamper barriers. Compare batched/SVD outputs to a direct small reference. Invariance tolerances are numerical tests (relative 1e-10 for finite costs away from ties), not permission to alter real-data ranking. A constructed regional-depth-distortion example may illustrate behavior, but cannot tune fold rules, thresholds or parameters; a failed illustration remains recorded.

Use serial CPU execution with one numerical thread, <=8GiB RSS, <=30s per pair and <=600s for the complete scoring cohort. No download or new dependency. At most 10 four-parameter fits per candidate (1 uniform-global,1 spatial-global,4 spatial-transfer,4 random-transfer), hence 3200 fits for the fixed bank. Stream pairs; O(N) workspace and O(N) arithmetic per fixed fit, with only four columns. These are operation bounds, not measured speed claims. Charge source/input verification, loading, all controls, failed fits, serialization and outer wall time separately; also report the saved cost of generating all four candidate arms. Candidate-generation cost is common and cannot be attributed only to the selected pose.

Any technical interruption produces an immutable failure receipt and stops the cohort; no favorable partial scientific score or silent retry. Mathematical unavailability follows the declared saturated-cost rules and keeps the pair. Seal all 80 scoring records, including all folds/candidates/control scores and selected identities, before a separate evaluator reads the saved 0017 evaluation labels. No HDF5 GT is needed and no labels may influence choices. Source/protocol/input/bank/result hashes bind the seal. Before label decoding, the evaluator also requires the completed outer prediction receipt, exit zero, no termination or residual process groups, wall time at most 600 seconds, sampled peak group RSS at most 8 GiB, the frozen helper/source identity, and an intact prediction log hash. It rechecks those bindings after evaluation.

## Single development gate and stop rule

Use the unchanged directed-translation/max-pose-error DMS AUC@5/10/20 definitions and all 80 pairs. All five scenes remain development. Report per-pair winners, candidate score margins, fold failures, selected-arm frequencies, each scene, wins/ties/losses and all fixed-control scores. No 80-independent-sample confidence interval or held-out claim.

For a conservative **benchmark comparator only**, B* is whichever of the four prespecified controls, four fixed candidate arms, and two fixed 0019 author-loss arms has the greatest all-80 AUC5; break ties by the order listed above, then original fixed arm order, then 0019 scale/affine order. These ten comparators include the stronger official controls without changing any selected candidate. An ineligible 0019 evaluation blocks this comparison; it cannot be omitted to obtain an easier gate. B* is not a learned deployable selector. Admission to any subsequent expansion requires ALL:

1. AUC5(spatial_transfer) - AUC5(B*) >= **2.0 absolute percentage points**.
2. AUC5(spatial_transfer) exceeds B* in at least **two of the five scenes**, using that same aggregate B* in every scene.
3. AUC20(spatial_transfer) >= max AUC20 over those same ten comparators - **0.5 absolute percentage points**.
4. Complete binding/coverage and the fixed resource limits pass. All same-bank control outcomes are retained; no choosing a favorable loss, fold seed, fallback or candidate bank afterward.

If any condition fails, close this fixed formulation without expanding the bank or retuning on these 80 labels. If all pass, the result is a development admission only: next work needs official strong baselines, independent-scene/source allocation, matched complete-pipeline cost and separate freezing. Neither success of known ingredients nor their combination alone establishes novelty; a contribution would be supported by the demonstrated transfer mechanism and substantial reproducible gains.
