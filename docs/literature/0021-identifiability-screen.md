# 0021 identifiability and equivalence screen

2026-10-19. Independent derivation and primary-source reading only. No cache arrays, labels, models, scientific calls, implementation, or tests. The claims below are local mathematical statements, not empirical admission.

**Verdict:** reference-edge calibration can compress geometry and extend a monocular prior to otherwise untriangulated query pixels. It creates no observation beyond its reference correspondences and depth prior. Where reference geometry already determines the query points, the proposal reduces to triangulation followed by PnP/similarity alignment, with a restrictive depth fit. Where it interpolates new points, it is a depth-completion/regularization hypothesis already expressible in depth-constrained BA. Excluding the query from calibration may change robustness under misspecification; it is not, by itself, an information or novelty result.

## 1. The gauge is scale, not general affine depth

Write the proposed image field as

\[
 f_i=\phi_i^T\theta,\qquad
 \phi_i=(d_i,u_id_i,v_id_i,1,u_i,v_i)^T.
\]

For one calibrated reference edge, homogeneous rays satisfy
\[
 y_i\times\{R(\phi_i^T\theta\,x_i)+t\}=0.
\]

The transformation `(theta,t) -> (c theta,c t)`, `c>0`, leaves all projections unchanged. An additive depth change generally does not: adding `b` moves different points by `b x_i`, which cannot generally be absorbed by one camera translation. Thus, after fixing the source camera frame, **positive scale is the generic geometric gauge**. Deliberately quotienting transferred fields by `f -> a f+b` because the query solver re-estimates shifts discards one additional, potentially observable quantity. Call that a transfer convention/nuisance choice, not an extra geometric gauge. General gauge/parameterization issues are classical BA theory. [Triggs et al., verified introductory and parameterization text](https://thoth.inrialpes.fr/people/triggs/pubs/Triggs-va99.pdf).

For disconnected reference edges `A-C` and `B-D`, retain two independent scale conventions. The query may estimate their relative scale and its translation; never equate the two unit-baseline triangulation gauges. If several reference edges share A, let `z_e` be depths triangulated with unit baseline. Their noiseless constraints are `Phi_e theta = s_e z_e`. Stack these as `[Phi,-Z](theta,s)=0`, where each column of Z contains the depths of one edge. Identifiability up to common scale requires rank `6+E-1`; the necessary count is `sum n_e >= 6+E-1`, not sufficient geometry. Fix one positive scale or a positive depth normalization, and propagate pose uncertainty rather than treating triangulated depths as independent exact measurements.

## 2. Smallest local sufficient configuration

**One reference pair can suffice algebraically.** Conditional on a selected cheiral pose branch, six generic nondegenerate correspondences are locally sufficient when (i) the calibrated epipolar Jacobian has rank 5, (ii) there is nonzero triangulation parallax, and (iii) `rank(Phi)=6`. The first condition fixes rotation and translation direction locally; unit-baseline triangulation then yields six depths, and the invertible feature matrix determines theta. The remaining common scale is unavoidable. This does not prove unique global recovery among all essential-matrix branches, numerical stability, correct matches, or generalization beyond those six locations.

A second, distinct motion edge is therefore a useful independent validation constraint, not a universal algebraic necessity. Excluding the query edge also does not make reference errors statistically independent: the same image, monocular prediction, and matcher can share biases. If the query is absent from a connected reference reconstruction, its pose may already be determined by that reconstruction; this must be counted as an ordinary multi-view control.

## 3. Exact admissible counterexample

Suppose all reference support lies on a plane and its predicted depth is exact there:
\[
d_i=1/(\alpha u_i+\beta v_i+\gamma).
\]
Then
\[
\Delta\theta=(\gamma,\alpha,\beta,-1,0,0)^T,
\qquad \Phi_{\rm ref}\Delta\theta=0.
\]
Starting from the identity correction, `theta + epsilon Delta theta` produces **identical reference depths and projections for every epsilon**, even with perfectly known reference poses and arbitrarily many matches. At query pixels off that plane, the change is `epsilon[(gamma+alpha u+beta v)d-1]`, generally not a global affine transform of the query depth vector. On a bounded domain with strictly positive initial depth/slope, sufficiently small nonzero epsilon preserves both positivity requirements. Positivity therefore does not remove this ambiguity. Pure rotation is a stronger degeneracy: reference projections constrain no depth at all. A narrow support strip similarly leaves unobserved spatial directions.

This counterexample rejects an unconditional claim that many reference matches determine a transferable six-parameter spatial field. It does not reject a conditional, observability-aware estimator on full-rank support.

## 4. A mathematical veto before implementation

Use the reference reprojection Jacobian `J_ref=[J_pose,J_theta]`, fixing the true scale gauge. For the proposed affine-invariant transfer convention, define
\[
T\delta\theta=P^\perp_{[f_q,\mathbf1]}\Phi_q\delta\theta,
\quad f_q=\Phi_q\theta.
\]
The necessary local condition is
\[
\ker J_{\rm ref}\subseteq\ker[0,T].
\]
If a reference-invisible joint pose/field perturbation changes query shape after the permitted affine quotient, that shape is not identifiable. The plane construction above already violates this condition analytically for generic off-plane query support. A fit residual, number of matches, or regularizer cannot turn that invisible component into measured information. A regularizer can choose a value, but that is an extra prior and must be shared by the baseline. Near-null directions likewise demand uncertainty/abstention, rather than a confident frozen correction. This is the proposed pre-implementation veto; no numerical test has been run.

## 5. What comparison could reveal a real increment?

If query points are triangulated by reference tracks, use the same points directly for robust PnP; if both sides have separate reference reconstructions, use similarity alignment with the relative scale free. With exact, sufficiently constrained geometry these already recover the query pose. A six-parameter fit offers denoising or compression, not extra measurements. If query points have no such tracks, the useful hypothesis is specifically that the **same-image monocular shape plus a low-dimensional correction predicts their depths**. References and query points must share a verified depth realization and coordinate convention.

Minimum strong controls, with identical reference observations and charged costs:

- Geometry-only reference triangulation/registration and subsequent joint BA; report track availability instead of quietly restricting its support.
- Global-affine depth transfer, and a two-stage reference-only fit of the **same six-parameter field** followed by the same query solver. If this is exactly the proposed pipeline, it establishes equivalence, not a distinct method.
- Joint reference/query BA with the **same field family**, robust losses, support, and priors. Freeze-versus-joint is then an estimator comparison under model/match errors, not an additional-information claim.
- Where direct tracks are unavailable, an otherwise identical depth-lifting registration control. A win over geometry deprived of depth extrapolation does not isolate the spatial correction or the no-query-feedback mechanism.

Joint inference under a correctly specified model can use query information to improve the field. Freezing it is not theoretically superior; any gain needs a demonstrated error mechanism, such as query outliers corrupting a shared fit, without hiding calibration uncertainty. No current evidence establishes that mechanism.

## 6. Direct prior limits, checked beyond abstracts

[MP-SfM, CVPR 2025, §§3.1–3.3](https://arxiv.org/html/2504.20040v1#S3) already scales/refines depth against triangulated structure, lifts previously untriangulated pixels, registers another view by PnP, and alternates depth refinement with pose/structure BA. Its normals and confidence inputs differ from these caches. It is a direct conceptual predecessor to the proposed information path, not an identical six-coefficient implementation.

[MBA, 3DV 2026, §§3.1–3.3, Eq.1–4 and coarse graph discussion](https://arxiv.org/html/2602.18906v1#S3) jointly optimizes per-image affine depth corrections, poses and intrinsics over pairwise correspondence edges; it uses a dense-residual marginalized objective and star-subgraph/full-graph stages. Its correction is global scale/shift rather than the proposed spatial six-parameter field. It does not establish query-excluded calibration, but rules out novelty from sharing affine depth corrections across edges alone. The [Google Research page](https://research.google/pubs/marginalized-bundle-adjustment-multi-view-camera-pose-from-monocular-depth-estimates/) confirms the venue; the linked OpenReview forum challenged browser access, so the author arXiv full text was used.

[PTC-Depth, inspected v1 §§3.1–3.4](https://arxiv.org/html/2604.01791v1#S3) transfers a previous posterior through estimated motion and fuses triangulated observations into a spatial scale field, with reliability weighting and superpixel consolidation. It uses external metric displacement and targets temporally consistent depth. This is not the same scale-free query-pose problem, but it prevents describing cross-frame spatial correction or uncertainty rejection in general as new.

**Admission conclusion:** no new scientific experiment follows from this note. A viable next question would need verified shared-image inputs, observable query-shape transfer, and a reason to outperform the same-field triangulate-transfer/joint-BA controls. Without those, this direction is a low-dimensional formulation of existing depth-assisted multi-view geometry, not a substantiated new pose mechanism.
