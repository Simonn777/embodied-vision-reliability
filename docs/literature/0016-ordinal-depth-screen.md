# Ordinal depth for calibrated relative-pose adjudication

Source-only screen, 2026-10-19. No images, HDF5 arrays, model weights, inference, or new experiments were accessed for this note. This is a conditional research hypothesis, not an experiment admission or a novelty/result claim.

**Recommendation:** test whether depth ordering can distinguish existing, geometrically competitive pose hypotheses before acquiring a depth model. The narrow mechanism is to use independent image-derived near/far relations to adjudicate hypotheses whose epipolar scores are similar. Its possible advantage is invariance to per-image nonlinear monotone depth calibration. Its major weakness is that finitely many orders are locally constant in pose: they can reject separated alternatives but cannot generally determine fine pose precision. If candidate orders do not disagree reliably, stop this direction.

## Closest primary work and novelty boundaries

| Work and inspected evidence | What is already established; consequence here |
|---|---|
| **MADPose, CVPR 2025**, [paper v2](https://arxiv.org/html/2501.05446v2), full method §§3–4 and Appendix D inspected | Models corrected depths as `D1 + b1` and `a(D2 + b2)`; combines depth-aware and point solvers. Its scoring/local refinement mix bidirectional depth reprojection and Sampson errors. Appendix D explicitly identifies spatially varying depth errors and limitations of a global affine correction, and suggests region-specific modeling. Neither non-affine miscalibration nor a generic regional remedy is a new observation. Importantly, spatially varying errors need not preserve a global order either. |
| **RePoseD, ICCV 2025**, [paper v3](https://arxiv.org/html/2501.07742v3), full §3 inspected; [official current project page](https://kocurvik.github.io/reposed/) inspected | Eq. (5), `s(beta_i+v) q_i = (alpha_i+u) R p_i+t`, uses numerical depths with unknown scale/shifts. Metric, scale, and affine variants are relevant controls. The official page now describes extended results with updated joint Sampson/reprojection optimization, implemented in PoseLib. A later strong comparison must pin that current implementation; comparing only with an older paper setting would be insufficient. |
| **Depth2Pose, 2026**, [paper v1](https://arxiv.org/html/2605.19797v1), full §3, §5.1 and Appendix B inspected | Distinguishes a hybrid pose estimator with Sampson scoring from a reprojection-only estimator that more directly exposes depth quality. Consequently, good hybrid pose accuracy alone does not establish that a depth prior is informative. The ordinal diagnostic should expose its incremental selection information separately from its geometric fallback. |
| **Zero-Shot Metric Depth Estimation via Monocular Visual-Inertial Rescaling**, 2025, [paper v1](https://arxiv.org/html/2509.08159v1), full §III-B inspected | Fits a monotonic B-spline mapping from relative to metric disparity using sparse visual-inertial reconstruction. Monotonic nonlinear recalibration itself is already a published mechanism. This work has geometric anchors supplied by VINS; it does not establish ordinal-only selection of an unknown two-view camera pose. |
| **Monocular 3D Human Pose Estimation by Generation and Ordinal Ranking**, ICCV 2019, [primary paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sharma_Monocular_3D_Human_Pose_Estimation_by_Generation_and_Ordinal_Ranking_ICCV_2019_paper.pdf) | Abstract and indexed method text, not a complete paper reading: generated body-pose candidates are evaluated using predicted joint-depth orders. Thus “generate pose hypotheses, then rank them with ordinal depth” is not a new abstract construction. Human articulation differs from calibrated camera motion; this is a scope boundary, not an exact camera-pose equivalence. |
| **FSNet, CVPR 2023**, [primary paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Barroso-Laguna_Two-View_Geometry_Scoring_Without_Correspondences_CVPR_2023_paper.pdf) | Abstract and indexed §3 inspected, not the entire paper: learns image-based two-view geometry scoring and candidate ranking. Independent image evidence for pose verification is already established. The proposed information restriction must have a demonstrated advantage, not merely a new score name. |

Terminology warning: *Fast Relative Pose Estimation using Relative Depth*, 3DV 2024, uses cross-view relative distances to the **same** scene point, according to its [official author poster](https://vlarsson.github.io/assets/pdf/astermark2024fast_poster.pdf). Those ratios are more informative than ordinal comparisons between different points. The title alone is not evidence of an ordinal-only solver.

This bounded search did not establish an exact modern calibrated two-view verifier using only arbitrary-monotone depth orders. That is an unresolved literature question, not evidence of novelty. The mathematical construction below is elementary geometry and is not claimed as a new theorem.

## A precise constraint and its limits

Use normalized homogeneous rays `x_i = K1^-1 [p_i;1]`, `y_i = K2^-1 [q_i;1]`, with third coordinate one, and the convention

`z2_i y_i = z1_i R x_i + t`, with `||t|| = 1`.

Let `u_i = R x_i`, `v_i = y_i`, and `c_i = u_i cross v_i`. For nonparallel rays, the ordinary least-squares ray-intersection depths are

```
z1_i = c_i · (v_i cross t) / ||c_i||^2 = a1_i(R)^T t,
z2_i = c_i · (u_i cross t) / ||c_i||^2 = a2_i(R)^T t,
a1_i = (c_i cross v_i) / ||c_i||^2,
a2_i = (c_i cross u_i) / ||c_i||^2.
```

These are exact for consistent rays; for noisy matches they are a specified algebraic triangulation, not optimal pixel-reprojection triangulation. Small `||c_i||` makes them unstable.

Suppose an image-derived output satisfies `d_vi = f_v(z_vi)` for an unknown strictly increasing function, independently in each view. A predicted sign `s_vij = sign(d_vi-d_vj)` implies

`s_vij (a_vi(R)-a_vj(R))^T t >= 0`.

For fixed rotation, these are homogeneous half-space constraints on translation direction, together with positive-depth constraints. They are invariant to any strictly increasing reparameterization of either depth map. Inverse depth requires its known orientation to be reversed. Camera-axis depth and Euclidean range must not be confused: the formulas above use camera-axis depth.

Compare each predicted order with triangulated depths **in that same camera**. Camera motion can change the order of two points; equality of image-1 and image-2 orders is not a valid general constraint. No metric translation scale is recovered.

The identifiability limitation is decisive. For a finite set of nondegenerate points with strict depth differences, continuity gives an open neighborhood of poses with the same orders. An ordinal score is therefore piecewise constant. It can select between sufficiently distinct candidates; it cannot promise arbitrary subdegree refinement. Adding a smooth ranking loss introduces a choice of margins or scale that does not create new measured information.

A profiled isotonic control would fit a nondecreasing map `g_v` by minimizing `sum_i (z_vi-g_v(d_vi))^2` for each candidate. If all orders agree, an unrestricted monotone interpolant can fit every sampled depth exactly. This creates entire pose regions with zero training residual. Any such control needs fixed translation normalization and separate fit/evaluation correspondences; better training fit than an affine map is not evidence of better pose.

## Minimum discriminating diagnostic

The question is: **does monotone-invariant information resolve actual candidate ambiguity that affine depth calibration resolves incorrectly?** This is more specific than asking whether depth improves pose.

1. **Freeze a common candidate bank without depth or GT.** Preserve the baseline winner, all candidate failures, original indices and deterministic tie behavior. Generate alternatives from the same calibrated geometric backend and matches under one prospective budget. Freeze a common correspondence set and a sparse, bounded-degree set of within-view point pairs. Do not let a candidate improve its score by discarding its unfavorable comparisons.
2. **Test observability before depth inference.** Report oracle pose-selection headroom, distinct order signatures, and order stability under a fixed coordinate-noise perturbation rule. Separately count comparisons on which competitive candidates disagree. Orders shared by every candidate cannot select anything, however high the overall rank correlation may be. If needed, use depths triangulated under the reference pose as an explicitly optimistic *proxy* order evaluator. With noisy or wrong correspondences, these are not ground-truth scene depths and not a real method result. GT stays evaluator-only.
3. **Only if headroom survives, admit one fixed depth model.** Verify output semantics, image-coordinate sampling, resource cost and license first. Evaluate a pure ordinal mismatch score on the frozen comparisons in each view. Keep a fixed denominator and explicit invalid/cheirality penalties. Quantify accuracy specifically on candidate-disagreement comparisons, not just easy global near/far pairs. Model inference cost is part of the method even if cached across pose hypotheses.
4. **Required controls:** geometric selection on the identical bank; affine-depth numerical scoring on that bank; an isotonic numerical control with separate fit/evaluation points if pursued; and the actual current RePoseD/MADPose pipelines for any subsequent competitive claim. A comparison against point-only PoseLib cannot establish a new state of the art. Keep the same depth model and image observations across depth controls.

Candidate scoring can be `O(H(N+M))` for `H` hypotheses, `N` rays and `M` frozen comparisons. This is a feasibility estimate, not a measured runtime. Counting all `N(N-1)/2` pair relations as independent evidence would exaggerate support because they reuse points and share ordering structure.

## Kill tests and decision

- **No candidate headroom:** the best candidate under reference pose error cannot reach the separately declared minimum gain over the baseline. More scoring complexity cannot fix absent good hypotheses.
- **No ordinal observability:** relevant alternatives retain the same stable order signature. Especially plausible for the already accurate development poses; ordinal-only refinement is then inappropriate.
- **Wrong model assumption:** the depth model flips the orders that distinguish candidates. A nonlinear monotone calibration cannot repair spatially varying order reversals. Large predicted gaps do not by themselves certify correct ordering.
- **Hidden numerical-depth dependence:** applying strictly increasing transformations to the same predicted depths changes the purported ordinal selection. The construction is not actually ordinal-only.
- **No incremental advantage:** ordinal selection loses to affine controls, or isotonic fitting only reduces its own training loss. Close the hypothesis rather than adding GT-tuned thresholds or changing the candidate bank after inspecting results.

The potentially useful research object is a verifier whose information survives unknown monotone calibration **and demonstrably changes otherwise ambiguous decisions**. The present screen establishes a constraint and a cheap way to reject its applicability. It establishes neither a novel pose estimator nor a measurable gain, and does not justify paper drafting.
