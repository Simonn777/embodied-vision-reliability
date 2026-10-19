# 0016: cached monocular depth as additional pose information

Date: 2026-10-19. Status: **source-only screen; no method or experiment admitted**. This is a literature comparison and elementary analytical assessment, not an experimental result or a new theorem. No cache values, ground truth, images, or models were read or executed for this screen.

**Decision:** depth can supply information absent from correspondence-only estimation, but “add cached depth, correct scale/shift, and fall back to geometry” is already RePoseD/MADPose. The inspected sources do not establish a distinct mechanism that warrants implementing a new method. Retain depth only as a possible next *baseline/headroom qualification* after a separate data/resource decision. Do not turn a successful reproduction of known depth gains into a contribution claim.

## What the closest methods already solve

### RePoseD: the affine-depth geometry is already solved

Ding et al., *RePoseD: Efficient Relative Pose Estimation With Known Depth Information*, ICCV 2025. Read: full-text arXiv v3, relevant §§2–4 and solver table; venue checked against the official proceedings. With homogeneous image points and bearings \(b_{1i}=K_1^{-1}p_i\), \(b_{2i}=K_2^{-1}q_i\), its Eqs. 1–4 give

\[
\lambda_i b_{2i}=\eta_i Rb_{1i}+T,\qquad
\eta_i=s_1(\alpha_i+u),\quad\lambda_i=s_2(\beta_i+v),
\]
\[
s(\beta_i+v)b_{2i}=(\alpha_i+u)Rb_{1i}+t,
\qquad s=s_2/s_1,\quad t=T/s_1.
\]

Translation has three unknown components in this gauge; imposing unit translation while retaining that depth parameterization changes the problem. Three depth-equipped matches give the calibrated nine-parameter minimal problem. Scale-only, independent/equal shifts, inverse-depth, and mixed 3D–3D/3D–2D/2D–2D variants are covered. Sampson-only scoring/refinement and depth reprojection are compared. Its §4.1 already discusses depth quality deciding whether P3P, shift-aware solvers, or five-point geometry works better, and explicitly proposes adding P3P to hybrid RANSAC. That combination is not an unoccupied idea. [Full text v3](https://arxiv.org/html/2501.07742v3), [official ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_RePoseD_Efficient_Relative_Pose_Estimation_With_Known_Depth_Information_ICCV_2025_paper.pdf).

### MADPose: robust geometric fallback is already part of the estimator

Yu et al., *Relative Pose Estimation through Affine Corrections of Monocular Depth Priors*, CVPR 2025. Read: full-text v3, §§3–4. Its Eq. 4 uses corrected depths \(\widehat D_1=D_1+\beta_1\), \(\widehat D_2=\alpha(D_2+\beta_2)\), equivalent to the preceding affine model. Let

\[
E_{12}=\|\Pi_2(RP_1+t)-p_2\|^2,\qquad
E_{21}=\|\Pi_1(R^{-1}(P_2-t))-p_1\|^2.
\]

For squared thresholds \(\tau_r,\tau_s\), its Eqs. 8–10 use

\[
\bar E=\min(E_{12},\tau_r)+\min(E_{21},\tau_r)
+2\lambda_s\frac{\tau_r}{\tau_s}\min(E_s,\tau_s).
\]

Local optimization minimizes the corresponding untruncated terms on three separate inlier sets. Hybrid LO-MSAC selects depth-aware or point-only minimal solvers; point-only proposals obtain compatible depth corrections through triangulation and least squares. Therefore independent directional rejection, geometric fallback, joint pose/calibration refinement, and mixing depth with Sampson error are existing controls. \(\lambda_s\) and threshold scaling are empirical weighting, not a demonstrated joint noise likelihood. [Full text v3, Eqs. 4, 7–10](https://arxiv.org/html/2501.05446v3), [official CVPR record](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Relative_Pose_Estimation_through_Affine_Corrections_of_Monocular_Depth_Priors_CVPR_2025_paper.html), [official implementation](https://github.com/MarkYu98/madpose).

### Uncertainty propagation and depth consistency are also occupied

Pataki et al., *MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion*, CVPR 2025. Read: full-text v1, §§3.1–3.5 and covariance discussion. Eqs. 2–5 optimize

\[
C_{\rm BA}+C_{\rm reg}+C_{\rm int},\quad
C_{\rm reg}=\sum\rho_C\big((\widehat D_i(X_k)-D_i^*(x_j))^2\big),
\]

where BA uses image covariance, and integration penalizes depth-prior and normal discrepancies in their covariance metrics. Alternation propagates uncertainties between structure and depth. The method uses depth for low-parallax initialization and depth consistency to reject incorrect registrations. Its §3.5 reports no significant end-to-end accuracy difference from substituting propagated refined-depth covariance for prior covariance in its experiments. Thus “propagate covariance into geometry” alone is neither a novelty argument nor evidence of accuracy gain. This is a multi-view system, not an interchangeable two-view implementation. [Full text](https://arxiv.org/html/2504.20040v1), [official CVPR record](https://openaccess.thecvf.com/content/CVPR2025/html/Pataki_MP-SfM_Monocular_Surface_Priors_for_Robust_Structure-from-Motion_CVPR_2025_paper.html).

Aung et al., *DGSfM: Depth-Guided Scale-Aware Global Structure-from-Motion*, July 2026 **preprint; acceptance not verified**. Read: full-text v1, §3. Eqs. 4–8 lift \(X_i=D_i(x_i)K_i^{-1}\bar x_i\), transform \(\tilde X_j=R_{ij}X_i+t_{ij}\), and reject using reprojection plus

\[
e^d_{ij}=\frac{|[\tilde X_j]_z-s_{ij}D_j(x_j)|}{s_{ij}D_j(x_j)}.
\]

Checks operate in both directions. Eq. 11 robustly averages log depth scales, \(\min_{\ell}\sum\rho_s(\ell_j-\ell_i-\ell_{ij})\), with a fixed scale gauge. Consequently, adding scale synchronization or depth-consistency pruning to a graph is not an unexplored rescue. Its full global pipeline needs more than isolated pairwise depth samples. [Full text v1](https://arxiv.org/html/2607.09507v1).

This targeted screen is not a proof that every possible residual-noise formulation has prior art. It does rule out claiming the combinations above as a distinct method.

## Where additional information can and cannot come from

The following is our analytical assessment, not an empirical finding.

Correspondence-only geometry permits separate unknown depths for every point. An externally predicted depth *shape*, tied together by only a few calibration parameters, restricts those freedoms. It can constrain weak geometric directions when the shape is reliable, or distinguish an epipolarly plausible false alignment from a correct one. A different pretrained depth predictor is an additional information channel; it is **not statistically independent** of a matcher using the same images, shared visual biases, or overlapping training sources.

An exact countercheck is to allow unrestricted per-point corrections. Starting from \(d_{2i}b_{2i}=Rd_{1i}b_{1i}+t\), eliminating both free depths leaves

\[
b_{2i}^{\mathsf T}[t]_\times Rb_{1i}=0.
\]

Thus freely “repairing” each predicted depth removes the additional depth constraint and returns to epipolar geometry. A smoothness or small-correction penalty can retain a prior, but then the benefit depends on the validity of that prior, not on a newly created observation.

A local identifiability test makes this more precise. For residual linearization

\[
r\simeq J_\theta\delta\theta+J_\nu\delta\nu+e,
\]

let \(\theta\) denote pose, \(\nu\) the depth nuisance parameters, and \(C\succ0\) a fixed noise covariance. Set \(X=C^{-1/2}J_\theta\), \(Z=C^{-1/2}J_\nu\), and \(P_Z=ZZ^\dagger\). Profiling the nuisance variables gives

\[
I_\theta=X^{\mathsf T}(I-P_Z)X.
\]

For a pose direction \(v\) with \(Xv\in\operatorname{col}(Z)\), the profiled residual contains no local information in that direction. Extra flexible depth corrections can therefore absorb the very pose change being estimated. The full stacked image/depth residual must use its joint covariance; adding two nominal information matrices without accounting for dependence is not justified.

With systematic residual bias \(b\), the first-order pose error includes

\[
\delta\theta_{\rm bias}
=-I_\theta^{-1}X^{\mathsf T}(I-P_Z)C^{-1/2}b
\]

when that inverse exists. Small final residuals and small reported covariance do not establish small pose bias. In particular, if a shared error lies in the pose Jacobian column space, reweighting cannot identify the true pose from that observation alone. Depth hallucination that supports the same false structure as the matcher is a decisive adverse regime.

One concrete noise interaction beyond global affine calibration is coordinate-dependent depth sampling. Locally,

\[
\widehat d(p+\delta p)\simeq\widehat d(p)+g^{\mathsf T}\delta p+\epsilon_d,
\quad g=\nabla\widehat d(p).
\]

Under an independent \(\epsilon_d\) approximation,

\[
\operatorname{Cov}(\delta p,\delta d)=\Sigma_p g,
\qquad \operatorname{Var}(\delta d)=g^{\mathsf T}\Sigma_p g+\sigma_d^2.
\]

This identifies why diagonal image/depth weights may be wrong near sloping surfaces; depth discontinuities additionally invalidate the linearization. It is ordinary uncertainty propagation, **not a proposed contribution**. The cache loader does not supply the gradient or a validated joint error law. Inferring one from a single set of neighboring, biased predictions does not establish its calibration. Likewise, disagreement between two depth models cannot detect their shared bias or identify which model is right.

## Decisive rejection conditions and appropriate controls

Reject a proposed extension if its only difference is scale/shift fitting, inverse-depth coordinates, a known hybrid score, P3P mixing, ordinary covariance propagation, or depth-consistency filtering. Those overlaps are explicit above. Also reject a noise-model claim if its distinguishing parameters cannot be identified from the available inputs, or if per-point correction merely absorbs pose error.

For any later, separately admitted qualification, compare the **same cached correspondences** with a current strong correspondence-only estimator, RePoseD scale-only and affine variants, one-sided P3P, and the actual MADPose hybrid. Sampson versus depth-reprojection refinement is an important existing ablation. If the depth advantage disappears against these controls, it is a reproduction, not an advance. Match complete solver time and report prior-generation cost as excluded/precomputed, rather than comparing only iteration counts. An installed old release is not evidence that the current implementation was tested.

The failure regimes must include incorrect depth *shape* after affine correction, systematic bias shared with matching, wrong correspondences, and poorly conditioned depth/point configurations. Constant or narrow depth ranges can make affine correction poorly conditioned; collinear lifted points leave rigid alignment ambiguous. At true zero translation, a translation direction is undefined. A variance-only improvement cannot be advertised as fixing bias or recovering an unobservable direction.

A potentially meaningful unanswered question is whether a *measurable, non-affine and pose-distinguishable* depth error component explains failures that remain after the above strong methods. This screen has not identified such a component in these assets, established a novel way to estimate it, or demonstrated sufficient independent observations. No implementation follows from merely naming that question.

## Cache and resource implications

The pinned PoseBench loader lists 24 monodepth files: five named outdoor groups plus ScanNet, crossed with RoMa/SP+LightGlue and MoGe/UniDepth. It reads correspondences, cameras, calibration, two depth fields, and pose labels. It already exposes calibrated RePoseD with/without shifts. ScanNet is an aggregate, not one independent scene; model variants are not independent scene replicates. File sizes, shapes, units, pair identity overlap, and actual locally available depth fields have **not** been verified here. The loader supplies no uncertainty/normal/gradient input; that does not prove the underlying archive lacks other fields. [Pinned loader](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/monodepth_relative_pose.py), [pinned estimator wrappers](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/estimators.py).

Future source comparison should distinguish the current PoseLib source pin `6a636b1d84273d5e3ba97676463476569611572a` from the unchanged 2.0.5 binary used in 0013–0015. The current header declares the monodepth API, but this screen did not build or validate it. [Pinned PoseLib declarations](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust.h).

Cached depth could make a small CPU qualification plausible under the current $0 / 8 GiB process limit. It does not make generating the depth free, prove Apple-hardware runtime, or establish that the remaining download allocation can hold the desired files. Do not import the benchmark to discover this: its normal workflow mixes input and label loading, and package entry paths can download data. If qualification is later chosen, inspect only archive/file metadata first, reserve independent scene/image identities before numeric access, and use a purpose-built prediction/label barrier. The already examined sixteen related pairs are not an untouched test set for this new direction.

**Result of this screen:** the extra-information rationale survives; the putative new method does not. No experimental improvement, novelty, or paper readiness is claimed.
