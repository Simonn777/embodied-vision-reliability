# 0019 prospective screen: depth updates in weak geometric directions

Status: source/math review only, 2026-10-19. No implementation, scientific input, model call, or experiment. This new note is deliberately untracked and does not change the frozen 0018 experiment. The parent-reported 0017 oracle gap motivates investigation but does not show that a deployable spectral rule can recover it; no 0018 outcome is assumed.

## Decision

**Not yet a distinct method admission.** Restricting an auxiliary estimate to weak geometric directions has a direct general precedent, not merely a vaguely related fusion precedent. Eliminating affine-depth nuisance and imposing a geometric trust region are mathematically sensible, inexpensive controls. Their combination could still yield an empirical contribution, but the present proposal supplies no observable reason that depth is reliable specifically in the directions where geometry cannot check it. It must beat nuisance-aware weighted fusion, solution remapping, and a plain geometric trust region before any stronger contribution claim.

## Primary sources and exact overlap

| Source and inspected version | Read depth and relevant overlap | What it does not establish |
|---|---|---|
| Zhang, Kaess, Singh, [On Degeneracy of Optimization-based State Estimation Problems](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.pdf), ICRA 2016 | Full text, introduction and §IV.B, Eqs. 15–20. Uses eigenvector subspaces to retain the measured solution in well-conditioned directions and a prediction in degenerate directions. The introduction explicitly motivates limiting extra constraints to degenerate directions. | Does not analyze affine monocular-depth calibration. Nevertheless, “depth supplies only weak directions” is an application of its central principle, not a new principle. |
| Tuna et al., [X-ICP](https://arxiv.org/html/2211.16335v3), inspected arXiv v3, 28 July 2023 | Full text §§V–VI, especially Eq. 10 and constrained optimization. Localizability categories determine linear constraints on pose updates along principal directions. | LiDAR registration, not calibrated two-view depth fusion; neither its categories nor thresholds transfer automatically. |
| Yu et al., [Relative Pose Estimation through Affine Corrections of Monocular Depth Priors](https://arxiv.org/html/2501.05446v3#S4.SS3), MADPose, CVPR 2025, v3 | Full text §§4.3–4.4, Eqs. 7–10. Combines affine-corrected depth reprojection with Sampson terms, jointly optimizes pose/calibration, and calibrates depths for point-only hypotheses through triangulation and least squares. | The inspected formulation does not impose a pose-subspace restriction or geometric cost budget. This difference is an operational comparison to test, not sufficient novelty by itself. |
| [OpenVINS MSCKF nullspace projection](https://docs.openvins.com/update-null.html), official implementation documentation | Full short derivation and QR implementation. Removes nuisance-feature sensitivity by left-nullspace projection. | Affine calibration is a different nuisance, but applying the same elimination algebra is standard. |
| Barbas Laina et al., [Scalix: Uncertainty-Aware Scale-Consistent Monocular SLAM](https://arxiv.org/html/2608.17553v1), preprint v1, 18 August 2026 | Full text §§IV–V: separate frame-scale and pixel-depth uncertainty, correlated depth covariance in Eq. 4, learned uncertainty heads, joint scale-aware optimization, and Schur-complement pose–scale factors in Eqs. 18–21. | Not the proposed training-free two-view affine-nuisance projection. It substantially narrows claims that separating calibration uncertainty from useful depth information is new. No acceptance status is inferred. |
| Cros et al., [Revisiting Split Covariance Intersection: Correlated Components and Optimality](https://arxiv.org/html/2501.07915v1), v1, 14 January 2025; abstract page reports IEEE TAC in press | Full text introduction/§2 conceptual setup, not all proofs. Conservative fusion explicitly handles unknown cross-correlation and distinguishes known correlated/independent components. | Covariance intersection needs credible error covariance bounds; inverse robust Hessians from biased, correlated matches do not acquire that interpretation by being called covariance. It provides no per-pair pose-error guarantee under arbitrary systematic bias. |

This bounded search did not identify a paper implementing exactly the proposed affine-nuisance, epipolar-subspace recipe. Absence from these searches is not evidence of novelty. The first three sources already make the principal ingredients close prior art.

## Concrete local derivation and equivalence

Fix translation magnitude first. Let the pose tangent be \(u\in\mathbb R^5\) for \(SO(3)\times S^2\), and let \(\eta=(a_1,b_1,a_2,b_2)\) calibrate axial depths as \(a_jd_j+b_j\). Fixing the baseline and allowing four nuisance values is locally equivalent to retaining baseline magnitude and choosing one depth-scale gauge. Leaving both gauges free creates a spurious null direction.

At a fixed pose, nuisance fit, support, whitening, and robust weights, write the local residuals as

\[
r_g+J_gu,\qquad r_d+J_du+J_\eta v.
\]

Whiten before elimination. With \(P_\perp=I-J_\eta J_\eta^\dagger\), minimizing over the nuisance increment gives

\[
\bar H_d=J_d^TP_\perp J_d,
\quad \bar g_d=J_d^TP_\perp r_d,
\quad H_g=J_g^TJ_g,\quad g_g=J_g^Tr_g.
\]

This is the usual Schur complement, implemented with rank-aware QR/SVD rather than constructing a dense projector. A pose direction \(u\) contains **no local depth information beyond calibration** when \(J_du\in\operatorname{col}(J_\eta)\). Damping or a nuisance prior changes this information and cannot be silently used to turn rank deficiency into evidence. These are local weighted-least-squares statements; changing robust weights or support requires a new linearization.

At a geometric stationary point, a proposed depth step could solve

\[
\min_u\ \bar g_d^Tu+\tfrac12u^T\bar H_du,
\quad \tfrac12u^TH_gu\le\epsilon,
\quad \tfrac12u^TMu\le\Delta,
\]

with a declared positive-definite tangent metric \(M\). Its KKT system is

\[
(\bar H_d+\lambda H_g+\mu M)u=-\bar g_d.
\]

Thus the soft geometric trust region is **locally an adaptive weighting of depth, geometry, and damping**, not intrinsically different information. Away from stationarity, the geometric constraint must contain \(g_g^Tu\) too; its multiplier contributes \(\lambda g_g\) on the right-hand side. Omitting that term is incorrect. A hard restriction \(u=U_wz\) adds subspace truncation, closely related to solution remapping. A generalized eigenproblem relative to \(M\) avoids confusing arbitrary coordinate units with weak information, but does not solve statistical calibration.

An actual nonlinear acceptance check on the same full geometric objective can enforce \(C_g(\theta\oplus u)\le C_g(\theta)+\epsilon\). This protects the measured objective only. It does not protect ground-truth translation, select the right essential-matrix basin, or make an inverse Hessian a calibrated uncertainty. All invalid projections need a fixed accounting convention; silently dropping them would invalidate the intended protection.

## Why the central reliability assumption is unproved

Consider a two-coordinate local problem with geometric curvature \(H_g=\operatorname{diag}(L,\varepsilon)\), \(L\gg\varepsilon\). The geometric estimate is the origin; the true correction is \((0,a)\). Suppose nuisance-free depth prefers \((b_s,a+b_w)\). Keeping only the weak direction gives \((0,a+b_w)\): it removes strong-direction depth bias \(b_s\), but the final error is \(|b_w|\), versus initial error \(|a|\). A small geometric-cost increase is compatible with worse pose error whenever \(|b_w|>|a|\). In the exact nullspace, geometry cannot discriminate these alternatives at all.

This is the testable assumption, not a theorem to claim: useful depth corrections must survive nuisance elimination and be more accurate than the geometric estimate in its weak subspace. A low depth residual alone does not establish that assumption. Affine nuisance can absorb some depth errors while non-affine, pose-aligned bias survives. Correlated dense matches can also make \(H_g\) spuriously large; duplicating identical rows increases its eigenvalues without adding independent observations. Local curvature should be described as conditioning unless a noise/correlation model justifies observability or covariance language.

## One decisive falsification, before any real-data admission

Use manufactured two-view correspondences with a fixed known pose and controlled low-parallax direction. Fix a single initial geometric pose, support, affine-depth parameterization, robust loss, and computational budget for every arm. Construct paired depth perturbations of equal size **after removing the affine-nuisance component**: one primarily aligned with a strong geometric direction and one with a weak direction. Include unbiased depth and depth perturbations entirely absorbed by nuisance as controls. Ground truth controls the synthetic construction only, never a deployable gate.

Compare: geometric-only continuation; full joint MADPose-style refinement; Schur-reduced uncertainty-weighted fusion; 2016-style solution remapping with the same weak subspace; geometric trust region without subspace truncation; and the proposed combined rule. Record true rotation/directed-translation error, the fixed full geometric/depth costs, and rejected/unchanged cases. A projected-step method that wins only against unprotected joint refinement, or only under strong-direction bias, has demonstrated the expected old remapping behavior, not a distinct reliability mechanism.

Required algebraic controls are invariant outputs under invertible affine reparameterization of input depths, and invariance to exact correspondence duplication when duplicated rows share the original total weight. No claim of invariance to additional independent observations is intended. If nuisance elimination removes the prospective improvement, if weak-direction bias passes the gate and causes equal or greater harm, or if the plain trust region achieves the same tradeoff, the current mechanism should be rejected or retained only as a known baseline.

The arithmetic is small: accumulating the pose/nuisance system is linear in matches, and the final spectral/Schur operations involve at most a 9-by-9 matrix. Cheap feasibility is plausible, not timed evidence. Neither an eigenvalue threshold nor a geometric budget is admitted by this note. A meaningful contribution would need a demonstrably useful, observable rule for when nuisance-free depth is trustworthy, plus substantial independent-scene accuracy gains over these controls; the current recipe has not supplied that rule.
