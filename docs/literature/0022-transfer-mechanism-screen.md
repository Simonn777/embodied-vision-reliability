# 0022 transfer mechanism: equivalence, closest prior and falsification

2026-10-19. Mathematical reasoning and bounded primary-source reading only. No scientific arrays, payload acquisition, model inference, pose experiment, or new artificial experiment was run for this note. The artificial checks proposed below are not results. The `paper-novelty-design-v1` review framework was used to separate the problem, intervention and falsifiable claim.

**Verdict:** an image-attached spatial depth field is already expressible in depth-constrained bundle adjustment (BA). Reference-only calibration followed by ordinary Gaussian predictive marginalization is not a new estimator in the linearized, correctly specified case. Cutting query feedback is a genuinely different estimator, but modular/cut inference and predictive selection of feedback strength are established. The more specific candidate—query-excluded leave-reference-view errors estimating local prior misspecification, while preserving query-visible null directions—was not found as an exact complete method in this bounded search. Its ingredients have close prior art, and it currently lacks an identifiable target and a defensible calibration guarantee. This is neither a claim of novelty nor a conclusion that an exact matching paper must exist.

## 1. Exact equivalence and its boundary

Fix the coordinate gauge and linearization. After eliminating reference-pose nuisance parameters, let the reference estimate be `theta_R` with positive definite field information `H`. For example, if the whitened reference model is `r = A theta + P alpha + noise`, then `H = A^T P_perp_P A`. On identifiable coordinates the reference Gaussian is `N(theta_R,H^-1)`. Query measurements have the local model

\[
y=B\theta+C\xi+\epsilon+b,\qquad \epsilon\sim N(0,I),
\]

where `xi` is query pose, `b` is systematic model error, and `C` has full column rank. Define `U=P_perp_C B`. Same-field joint least squares minimizes

\[
(\theta-\theta_R)^T H(\theta-\theta_R)
 +\|y-B\theta-C\xi\|^2.
\]

Eliminating query pose gives

\[
\theta_J=(H+U^TU)^{-1}
  [H\theta_R+B^TP^\perp_Cy],\qquad
\xi_J=C^+(y-B\theta_J).
\]

Alternatively, marginalize the reference Gaussian into the query likelihood:

\[
y\mid\xi,R\sim N(B\theta_R+C\xi,
                  I+B H^{-1}B^T).
\]

Generalized least squares for `xi` with that covariance has exactly the joint solution. This statement requires a fixed linear model, matching Gaussian assumptions and priors, and correct gauges. It is not an assertion about arbitrary nonlinear optimizers or every form of uncertainty propagation.

In particular, a **cut posterior** is different:
`q(theta,xi)=p(theta|reference) p(xi|theta,query)`.
With a flat query-pose prior its mean is the frozen-field solution
`xi_cut=C^+(y-B theta_R)`, but it still propagates field uncertainty through

\[
\operatorname{Cov}_{cut}(\xi)
=(C^TC)^{-1}+C^+B H^{-1}B^T(C^+)^T.
\]

Thus “forward uncertainty propagation without feedback” is possible; it must not be confused with predictive-likelihood marginalization. Generic directed feedback control is established by [Carmona and Nicholls, AISTATS 2020, §§2 and 4](https://proceedings.mlr.press/v108/carmona20a/carmona20a.pdf). Their §4.3 also uses cross-validation/WAIC to select influence by predictive performance. Transplanting that choice into BA does not itself establish a new principle.

Under correctly specified independent linear-Gaussian errors, the full joint estimator uses the available information efficiently; freezing cannot create additional information. For unbiased references and a fixed query bias `b`, its feedback contribution is

\[
\delta\theta_J=(H+U^TU)^{-1}U^Tb,\qquad
\delta\xi_J-\delta\xi_{cut}
=-C^+B(H+U^TU)^{-1}U^Tb.
\]

Two decisive null cases follow. If `U=0`, query observations add no field information and joint and frozen solutions coincide. If `b` lies in the query pose column space, `U^T b=0`: freezing cannot remove this pose-compatible bias. A claimed feedback mechanism needs a nonzero field update **and** a nonzero induced pose change. Their existence does not imply the change is harmful; evaluate the error direction and variance.

If `H` is singular, do not replace it by a pseudoinverse and interpret the resulting zero variance in null directions as certainty. Work on identified coordinates and explicitly retain query-visible unidentified directions as unbounded or bounded only by declared priors. Different reference edges need their own positive baseline scales. Arbitrarily unit-normalizing all translations does not put their depth samples in one metric gauge.

## 2. Fixed inexpensive falsification before a larger scientific allocation

Use the existing 0021 manufactured full-rank, planar and pure-rotation constructions, then prospectively specify a small query Jacobian and all bias vectors before execution. No images, pretrained models or real scenes are required. Compare four procedures on identical observations, field family, gauges and priors:

1. Frozen-field/cut inference, including its propagated covariance.
2. Gaussian predictive GLS from the reference posterior.
3. Same-field joint Gaussian estimation; this must agree with procedure 2.
4. A strong robust joint estimator with the same support and initialization budget, if the analytic stage survives.

Start with exact small linear solves. Compute pose risk as squared bias plus covariance trace, without Monte Carlo seed selection or local-optimizer differences. Declare five regimes: correct independent noise; no spatial deformation; query bias within the pose column space; a fixed non-pose field-related query bias; and reference-edge bias. Keep degenerate cases and failures. Do not choose the most favorable bias direction after looking at the result.

Reject the proposed advantage if it depends on unequal priors/support, an artificially weak joint initialization, omitted calibration uncertainty, or an oracle choice of which module is corrupted. Reject a universal claim on the pose-compatible-bias null case. An advantage only for one deliberately favorable misspecification is a mechanism witness, not evidence that the mechanism occurs in the intended data. After the algebra stage, any useful policy must outperform robust joint inference without access to true corruption labels. This note admits no implementation or execution.

## 3. More specific candidate and nearest inspected work

Candidate under review: fit a shared per-image field on reference edges excluding the query; predict each reference view from other references; use those out-of-view errors to estimate field misspecification; preserve query-support null directions and abstain where transfer is unidentified; use the resulting prior in query pose inference.

| Primary source and inspected depth | Overlap and remaining distinction |
|---|---|
| [Pak, Reichel and Burke, *Machine-Learning-Inspired Workflow for Camera Calibration*, Sensors 2022, §§6–8](https://www.mdpi.com/1424-8220/22/18/6804) | Already replaces reliance on internal calibration variance with testing on held-out poses, K-fold variability and downstream forward-projection reliability maps. It calibrates camera models with a controlled target, not a monocular spatial depth field from unknown reference poses. The publisher-indexed sections were inspected; direct article opening was unavailable. |
| [Carmona and Nicholls, *Semi-Modular Inference*, AISTATS 2020, §§2, 4.3](https://proceedings.mlr.press/v108/carmona20a/carmona20a.pdf) | Already distinguishes joint, cut and intermediate directed feedback, selecting influence via predictive criteria. This covers the generic “trust reference module, limit query feedback using prediction” idea. It is not a geometric implementation or an answer to shared correspondence bias. |
| [Chu, Pan and Shi, *View-Structured Conformal Prediction for 3D Gaussian Splatting*, arXiv:2609.10307v1, 9 September 2026, §§3–5 and Algorithm 1](https://arxiv.org/html/2609.10307v1) | Separates a spatial residual field from view difficulty, uses separated data roles and held-out view quantiles, and distinguishes view coverage from pooled pixels. It predicts RGB uncertainty with observed calibration-view images and known camera descriptors. It does not establish depth uncertainty from unknown-pose geometric pseudo-labels or the proposed nullspace-aware depth-prior transfer. Preprint status only. |
| [Dong and Li, *Conformalized confidence calibration for multi-view 3D reconstruction*, Pattern Recognition, DOI 10.1016/j.patcog.2026.114249](https://www.sciencedirect.com/science/article/pii/S0031320326012148) | The publisher-indexed abstract/introduction explicitly combines geometric/photometric consistency scores, grouped calibration and reconstruction error radii. It requires held-out ground-truth depth and assumes scene-level exchangeability. The listing gives a December 2026 issue; early-online date was not verified. Direct full-text opening failed, so this is a verified close lead, not an exhaustive methods comparison. |
| [Yang and Pavone, *Object Pose Estimation With Statistical Guarantees*, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Object_Pose_Estimation_With_Statistical_Guarantees_Conformal_Keypoint_Detection_and_CVPR_2023_paper.pdf) | Already propagates calibrated keypoint sets to a pose uncertainty set and worst-case pose bounds. It uses object keypoint calibration and known object geometry, not unlabeled relative-depth calibration. A generic conformal-to-pose wrapper is therefore insufficient differentiation. Abstract and formulation overview inspected. |
| [Poggi et al., *On the Uncertainty of Self-Supervised Monocular Depth Estimation*, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.pdf) | Studies depth uncertainty without depth-label training and explicitly identifies pose-estimation error as a confounder. Its self-teaching/network setting differs from per-instance leave-reference-view field calibration. Indexed methods/results/conclusion inspected. No-GT uncertainty by itself is not new. |
| [Wilson and Wehrwein, *Visualizing Spectral Bundle Adjustment Uncertainty*, 3DV 2020](https://swehrwein.github.io/files/sfmflex.pdf) | Analyzes collective covariance modes rather than isolated diagonal variances, including deformation and parameter-scaling issues. Preserving coupled weak directions is established uncertainty practice; the proposed query-transfer null condition is an application-specific requirement. Abstract/introduction and author project inspected. |

Additional adjacent leads were checked only to bound the search: [Bayesian Deep Basis Fitting, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Qu_Bayesian_Deep_Basis_Fitting_for_Depth_Completion_With_Uncertainty_ICCV_2021_paper.html) already provides basis-fitting depth variance; [ProBA, inspected arXiv v1 §3](https://arxiv.org/html/2505.20858v1) propagates 3D/2D uncertainty through reprojection; [Achieving Risk Control in Online Learning Settings, TMLR 2023](https://arxiv.org/abs/2205.09095) includes image-depth intervals but uses observed sequential outcomes. None of these inspected portions establishes the exact candidate. This search is not exhaustive and does not support a “first” claim. Existing MP-SfM/MBA controls and their input differences remain as recorded in the [0021 screen](0021-identifiability-screen.md).

## 4. The unresolved target is more serious than the name

Leave-reference-view prediction error is an **observable compatibility error**, not automatically true depth error. It combines field misspecification, feature localization, correspondence error, reference-pose error and visibility. If the held-out view's pose is itself fitted using its scored correspondences, it is not a fully out-of-sample prediction: pose fitting can absorb field error. Either estimate that pose from disjoint observations with explicit uncertainty, or define and analyze the profiled residual honestly. Fitting a shared image does not make the reference views independent.

A direct indistinguishability counterexample survives leave-view validation. Choose an alternative positive field and camera configuration and generate all reference matches consistently from it. The observed graph can be interpreted either as accurate observations of that alternative geometry, or as coherently biased matches of a different geometry. All leave-view predictions can agree, despite different true depth at query support. More folds do not distinguish those worlds. Model-error uncertainty cannot universally cover that shared bias without additional assumptions, trusted measurements, or explicit error bounds. Retaining a model's Jacobian nullspace addresses geometric non-identification; it does not reveal every coherent error outside that model.

Ordinary conformal guarantees also do not transfer automatically. The target of calibration must be observable, and the exchangeable unit must be justified; thousands of pixels sharing a depth map are not thousands of independent reference views. Reusing overlapping leave-one-out fits is not the same protocol as split conformal. For the usual split order statistic `ceil((m+1)(1-alpha))`, fewer than nine calibration units at 90% nominal coverage gives an infinite threshold; fewer than nineteen does so at 95%. These are arithmetic feasibility limits, not guarantees that nine or nineteen correlated views suffice. The view-level construction and explicit infinity case are also visible in VSCP Eq.16. No depth or pose coverage is claimed here.

## 5. Narrow testable distinction that remains

The potentially testable claim is not “we estimate uncertainty” or “cross-validation is new.” It is: **under stated reference-error assumptions, query-excluded out-of-view compatibility identifies when a spatial monocular prior will damage query pose more reliably than Hessian uncertainty and in-sample residuals, at matched coverage, support and total compute.** This is currently a hypothesis, not an established contribution.

Its minimal artificial screen should add to Section 2: (a) a shared field error visible in withheld reference directions; (b) the exact planar query-visible null counterexample; (c) a coherent common-bias counterexample with small leave-view error; and (d) wrong-image reference transfer. Compare the proposed score with Hessian-only, in-sample residual, ordinary leave-view scalar inflation, global-affine transfer, and the same-field robust joint estimator. All policies need the same fallbacks and all queries retained. If the richer method only improves uncertainty ranking without improving pose at matched abstention, it does not satisfy the intended research outcome. If an ordinary leave-view scalar plus BA achieves the same behavior, the elaborate proposal has not isolated a contribution.

No exact complete prior was located, so “fully covered” would overstate this search. Conversely, the broad version is already covered by modular inference, cross-validation calibration and geometric uncertainty propagation. A defensible advance would require solving the geometric pseudo-label/confounding problem with a specific assumption or new identifiable measurement, then demonstrating substantial benefit over the same-field strong controls. Relabeling residual variance as misspecification uncertainty or adding a conformal quantile does not meet that burden. Resource acquisition and scientific admission remain separate decisions.
