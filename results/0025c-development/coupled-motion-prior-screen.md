# Coupled-motion RS follow-up: source-only admission screen

Private, 2026-10-19. **NO_GO for a new “physical RS affine-depth two-point” method justified by trajectory coupling or minimal sample size alone.** No new experiment is admitted. The closed 0025c outputs are not retuned; their failure does not establish that the coupled model would improve accuracy. This review read primary papers and existing source notes only, not scientific arrays, GT, images or model outputs.

## Closest primary evidence

1. **Su et al., ICCV 2025, “A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks.”** [Official publication](https://openaccess.thecvf.com/content/ICCV2025/html/Su_A_Linear_N-Point_Solver_for_Structure_and_Motion_from_Asynchronous_ICCV_2025_paper.html); [full arXiv v1](https://arxiv.org/html/2507.22733v1); [author code](https://github.com/suhang99/AsyncTrack-Motion-Solver). Read §§3.1–3.4 and Appendix §§6.5–6.6, not only the abstract. Equations (3)–(8) use known rotation, timestamped bearings and constant translational velocity; they eliminate structure through a Schur complement. Section 3.4 explicitly includes two tracks with two observations each. Appendix Eq.(49) generalizes to known-angular higher-order translation. Equations (61)–(64) already insert known constant IMU acceleration as an inhomogeneous linear system. These are direct precedents for both proposed coupling routes. The paper's assumption of supplied acceleration is not an established raw-accelerometer calibration pipeline. Code is linked by the authors; this screen did not inspect or execute its implementation.

2. **Zhuang, Cheong and Lee, ICCV 2017, “Rolling-Shutter-Aware Differential SfM and Image Rectification.”** [Official publication](https://openaccess.thecvf.com/content_iccv_2017/html/Zhuang_Rolling-Shutter-Aware_Differential_SfM_ICCV_2017_paper.html); [full text, arXiv v1 posted 2019](https://arxiv.org/html/1903.03943v1). Read §§3–4.2. Its small-motion constant-velocity model already ties scanline translation to inter-frame translation: Eq.(6a) multiplies the latter by a known row-dependent scalar. Equations (8)–(13) extend to acceleration constrained parallel to velocity, with one unknown acceleration factor. That restricted differential model is not an exact finite SO(3)/SE(3) model or a general measured-acceleration solver; nevertheless, merely sharing within-readout and between-frame motion is established.

3. **RePoseD, “Relative Pose Estimation through Affine Corrections of Monocular Depth Priors.”** [Full arXiv v1](https://arxiv.org/html/2501.07742v1), §§2–3.1 read. Eq.(4) is the existing two-view affine-depth relation after fixing the first depth scale. Conditioning it on known rotation removes rotational unknowns. Adding a familiar trajectory constraint to that calibration equation is a possible combination, but the linear algebra alone does not establish a new source of directional information.

The following deductions are our algebraic screening, not claims quoted from those papers. No absence-of-search-hit argument is used to establish novelty.

## Why the two-point direction is already depth-free

Start with the frozen row-ray model

\[
(a_2d_{2i}+b_2)\bar r_{2i}-(d_{1i}+b_1)R\bar r_{1i}
=t-RB_{1i}v_1+B_{2i}v_2.
\]

For the angular-only control, the right side is `t`. Let
\(n_i=\bar r_{2i}\times R\bar r_{1i}\).
Every exact affine solution satisfies \(n_i^Tt=0\). Two independent rows therefore determine the unoriented translation direction \(t\parallel n_1\times n_2\), without depth. Affine-depth values fix calibration/length and can reject a physically incompatible sample; they do not supply a different generic two-match direction. Directed sign still requires cheirality, which the depth-free comparator can also test.

For **constant camera-center velocity in a common world frame**, rotate each observed ray into that frame: \(q_{ji}=R(\tau_{ji})r_{ji}\). With \(C(\tau)=C_0+V\tau\),

\[
z_{2i}q_{2i}-z_{1i}q_{1i}=(\tau_{1i}-\tau_{2i})V,
\quad (q_{2i}\times q_{1i})^TV=0.
\]

The nonzero known time difference cancels from the epipolar equation. Two independent match normals already determine `V` direction. This is precisely the relevant known-rotation, asynchronous geometry family, not a new affine-depth minimal advantage.

**Constant left SE(3) twist** is a different motion assumption; it must not be called constant world translational velocity. In the frozen convention, take one shared \((\omega,v)\), \(R=\exp(\Delta[\omega]_\times)\), \(G_\Delta=\int_0^\Delta\exp(u[\omega]_\times)du\), and \(t=G_\Delta v\). The preceding equation becomes

\[
z_{2i}\bar r_{2i}-z_{1i}R\bar r_{1i}=K_i v,
\qquad K_i=G_\Delta-RB_{1i}+B_{2i}.
\]

Its depth-free rows are \(n_i^TK_i v=0\). Again, rank two determines the direction of `v`, and hence of `t` when nonzero. The six-variable affine system `(v,b1,a2,b2)` may solve from two correspondences, but that sample-size claim buys no generic directional advantage over the same-physics depth-free two-point control. Degenerate rank, zero translation and sign ambiguity remain explicit exceptions; equation counting does not guarantee conditioning. An independently perturbed `R` and endpoint angular rates cannot simultaneously be declared one exact shared twist unless a prospectively defined consistent trajectory is supplied to every method.

## Acceleration is neither free nor an unchanged gauge

Suppose the metric displacement forcing is actually known, so camera position is
\(C(\tau)=C_0+V_0\tau+h(\tau)\). Eliminating depths yields

\[
n_i^T\{(\tau_{1i}-\tau_{2i})V_0+h(\tau_{1i})-h(\tau_{2i})\}=0.
\]

This is a depth-free **inhomogeneous** three-vector problem, generically requiring three independent scalar constraints. Known metric forcing removes the free global length gauge. The affine-depth version must retain both depth scales: unknowns are `(V0,a1,b1,a2,b2)`, seven scalars, so two matches provide only six equations. Fixing `a1=1` without a known metric calibration would discard a necessary unknown. A three-point affine system can add overdetermined depth structure, but “metric acceleration plus affine depth still gives a generic two-point method” is not justified. Degenerate zero/collinear forcing can leave scale unobservable and must be treated separately.

An accelerometer supplies specific force, not world acceleration or velocity. Gravity, orientation over the interval, bias, timing, camera–IMU lever arm and the initial velocity contract matter. Independent gyro supplies rotation after integration/calibration; it does not measure a shared translation trajectory. Su et al.'s supplied-acceleration appendix means generic acceleration coupling is also prior art, even before those engineering ambiguities are resolved.

## What remains, and the admission decision

A combination of known components can still earn a contribution through a distinct, validated mechanism. Here the only remaining opening is an **overdetermined, uncertainty-aware use of learned depth that adds accuracy beyond identical temporal geometry and identical calibrated IMU inputs under model mismatch**. That is a research question, not a screened new mechanism. Neither ordinary regularization of `v1,v2`, a Schur complement, a constant-twist constraint, nor replacing free velocities with an IMU trajectory alone meets the requested distinction.

No implementation or new resource acquisition is justified for the proposed two-point claim. Before any later method admission, a concrete design must explain why its added depth information survives affine nuisance elimination and helps beyond: (i) same-physics known-rotation depth-free two-/three-point geometry, (ii) Su's overdetermined asynchronous incidence approach, and (iii) angular-only affine calibration. Exact equality of minimal directions is a mathematical veto of the sample-size story, not an empirical accusation and not a claim that all depth fusion must fail. The current noise failures remain closed; reusing them to tune a coupled model would not establish a fresh efficacy result.

## Scope and transfer accounting

No solver, fixture, model, test or dataset call ran. New directly metered HTTP response-body bytes: **0**; the web provider's page-transfer byte counts are unavailable, not asserted zero. Primary full texts were read through the web tool; no sensor/archive/image/depth/pose payload or new checkpoint was requested or saved. The source screen is bounded and does not prove that every published affine-depth/RS combination has been found.
