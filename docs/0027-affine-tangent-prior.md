# 0027: one affine correspondence with affine-depth tangents

Source/math screen, 2026-10-19. No model, scientific arrays, labels, fitting or
experiment was used. This is a conditional hypothesis, not novelty proof or
experiment admission.

## Assessment

The precise **known-K, one-AC, two depth shifts plus relative scale** problem was
not found solved in the primary sources checked below. However, almost every
construction surrounding it is established: depth-gradient tangents, one-AC
pose recovery, Gram/angle invariants eliminating rotation, affine-depth
three-point calibration, and converting local frames into point triples.
The narrow possible increment is a stable, physically filtered infinitesimal
affine-depth hypothesis generator that improves actual pose at matched total
cost. Merely writing its three quadratics or using a modern depth network is
insufficient. No search absence establishes priority.

## Primary-source reading ledger

| Source; sections actually read | Direct overlap and bounded distinction |
|---|---|
| Eichhardt and Barath, **Relative Pose from Deep Learned Depth and a Single Affine Correspondence**, ECCV 2020; [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf), §§2.1–3, Eqs.4–11, and §4.2 | Differentiates lifted points, removes translation, and aligns two 3D tangents from one AC. It estimates a relative depth scale, with an SVD/Umeyama variant and a faster orientation construction. The stated model has no additive depth shifts. Thus one-AC+depth, first-order lifting and rotation recovery are direct prior art. |
| Hruby, Pollefeys and Barath, **Semicalibrated Relative Pose from an Affine Correspondence and Monodepth**, ECCV 2024; [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05645.pdf), §§2.1–2.4, Eqs.11–17, §§3.1–3.2 | Unknowns are two focal lengths, relative scale and pose. Tangent angle/length ratios remove rotation before polynomial solution. It assumes scale-only depth, not independent offsets. It explicitly handles a fronto-parallel degeneracy and evaluates AC-to-3PC. Its depth definition uses unit-bearing range; our proposed model uses axial depth. These semantics cannot be interchanged silently. |
| Yu et al., **Relative Pose Estimation through Affine Corrections of Monocular Depth Priors**, CVPR 2025 (MADPose); [paper](https://arxiv.org/html/2501.05446v3), §§3–4.4, especially §4.1 Eq.5 | Three calibrated point correspondences determine affine-depth corrections through distance preservation; hybrid sampling, scoring and local refinement already combine depth and epipolar information. It does not use an AC/depth-gradient input in the checked solver. |
| Ding et al., **RePoseD: Efficient Relative Pose Estimation With Known Depth Information**, ICCV 2025; [published paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_RePoseD_Efficient_Relative_Pose_Estimation_With_Known_Depth_Information_ICCV_2025_paper.pdf), checked [precursor](https://arxiv.org/html/2501.07742v1) §§2–3.1, Eqs.2–4 and 8–13 | The precursor is titled **Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation**. It already fixes the first depth scale, eliminates pose by three distances, and solves six lifted monomials. The title/version distinction matters; this review is not a new published-version code audit. |
| Chum, Matas and Obdrzalek, **Epipolar Geometry from Three Correspondences**, CVWW 2003; [author PDF](https://cmp.felk.cvut.cz/~matas/papers/chum-cvww03.pdf), abstract, §§1 and 3 | Three point correspondences per matched local frame and locally optimized robust estimation are old. This is not itself an affine-depth solver. The 2024 paper includes the corresponding AC-to-3PC family as a baseline. |
| Barath et al., **Making Affine Correspondences Work in Camera Geometry Computation**, ECCV 2020; [paper](https://arxiv.org/html/2007.10032v1), §3.2, Algorithm 1, §§4.4–4.5 | Symmetric photometric refinement, affine uncertainty and point-only local optimization are already established. Their essential-matrix uncertainty preemption was excluded for cost. Generic covariance weighting or photometric affine refinement is not a new component here. |
| Sun et al., **Learning Affine Correspondences by Integrating Geometric Constraints**, CVPR 2025; [paper](https://arxiv.org/html/2504.04834v1), §§3.2–3.4 | Dense matches locate 32×32 image patches; learned scale, orientation and residual shape produce the affine map. It is a concrete stronger affine frontend, not free information already contained in one matched pixel. |

Targeted current searches also checked the primary method text of Barath's
[Rolling Shutter Relative Pose Estimation Made Practical](https://arxiv.org/html/2606.26863v1),
§§3–4: its AC constraints address row motion, not the proposed GS affine-depth
model. No direct 2026 shifted-depth one-AC replacement was found in this bounded
search. Secondary summaries/search snippets were not used as proof of coverage.

## Algebra and what is actually different

Use pixel coordinates consistently. Let `r_i=K_i^{-1}[x_i,y_i,1]^T` (third entry
one), `J_i=∂r_i/∂(x_i,y_i)`, `g_i=∇d_i` in depth-per-pixel units, and let
`A=∂x_2/∂x_1`. Fix only the first *raw-depth coefficient* to one:

```
X = (d1+b) r1,                 Y = (a d2+c) r2,   a>0
U = (d1+b) J1 + r1 g1^T
V = [(a d2+c) J2 + a r2 g2^T] A
Y = R X+t,                    V = R U.
```

The supplied expression is consistent with the chain rule: the second gradient
must be transported by `A`. A constant depth shift has zero scalar gradient but
still changes a lifted tangent through `b J1` or `c J2`; dropping those terms
would change the problem. Pixel-normalized affine matrices/gradients must be
transformed together if using a different image chart.

Eliminating `R` gives the three entries of `U^T U−V^T V=0`. Write
`U=P+bJ1`, `V=aQ+cS`, where `P=d1 J1+r1 g1^T`,
`Q=(d2 J2+r2 g2^T)A`, `S=J2 A`. The equations are linear in
`[1,b,b²,a²,ac,c²]`, exactly the familiar affine-depth monomial pattern.
There is no independent justification for calling the lifting new.

When both tangents have rank two and the Gram equality holds, a proper rotation
is obtained from the two tangent columns plus their cross product, followed by
`t=Y−RX`. Positivity, real roots and residual checks are required. Counting three
equations/three calibration unknowns does not prove generic rank, unique physical
solution, useful translation direction, or stable numerical recovery. With noisy
inputs there may be several or no admissible roots; global point support must
resolve candidates without labels.

**Range warning.** Replacing `r` by the unit ray `q=r/||r||` requires
`λ=z||r||` and `∇λ=||r||∇z+z∇||r||`. Consequently an axial offset becomes a
spatially varying range offset with a derivative contribution. An unchanged
range-based implementation plus two constants is not the proposed axial model.

## AC-to-three-points is a mandatory same-input control

Generate `x1`, `x1+εe1`, `x1+εe2`, their affine-mapped partners, and depths
using the same first-order depth fields. Their corrected 3D differences obey
`ΔX_j=εU_j+O(ε²)`, `ΔY_j=εV_j+O(ε²)`. Divide the three pair-distance equations
by `ε²` and take the limit: they yield the two tangent norms and the mixed Gram
entry. Thus the candidate is the infinitesimal limit of the existing three-point
distance construction with the same local measurements.

It is not exactly the finite-ε ordinary three-point system: multiplying
`d(x)+εg_j` by `r(x)+εJ_j` introduces an `ε²g_jJ_j` term even when the scalar
depth approximation is linear. The exact derivative solver may avoid this
finite-step bias and tiny-triangle cancellation, but that is an empirical
numerical advantage to demonstrate. Synthetic neighboring points are correlated,
not three independent inliers. Choosing their spacing after seeing pose labels
would invalidate the comparison.

## Main practical risks and minimal falsifier

Small depth error does not imply small gradient error: adding
`η sin(kx)` changes depth by at most `η` but its derivative by `ηk`. Edge mixing,
upsampling, depth discontinuities and local warp curvature can dominate the
tangent equations. If `g1=0`, scaling `(d1+b, a, c)` together can leave a
continuous calibration family; its effect on `R` and normalized `t` must be
checked separately, since a pure scale ambiguity need not damage pose direction.
Nearly dependent tangents, a singular calibration Jacobian, small translation,
and a wrong but internally coherent local match are additional failure cases.

The next useful step, if separately admitted, is a **small fixed manufactured
qualification**, not a new full RANSAC framework: derive AC and axial gradients
from the same projected smooth surface; verify analytic Jacobians and exact
clean recovery; include a generic sloped patch, a fronto-parallel case and a
small-baseline case. Perturb point/affine/gradient measurements jointly in a
declared image-chart model. Report all physical roots, unavailable fraction,
rotation error and directed translation error, including functional nullspaces.
Do not select the best root with truth for the main result.

Necessary controls are: scale-only 1AC+D on identical tangents; finite-ε
AC-to-three-points with the existing affine-depth solver; ordinary three-point
affine-depth proposals; current RePoseD author-loss/PoseLib and MADPose hybrid
pipelines; and the same image/neighbor support used as point data. Any later
robust comparison must share candidate scoring/refinement and charge affine
extraction, depth inference, gradients, every root and failed attempt.

If physical pose sensitivity is already catastrophic under modest coherent
derivative errors, or the same-input synthetic-point baseline supplies the same
useful hypotheses at comparable cost, there is no reason to build the pipeline.
If it survives, the defensible question is whether shift-aware differential
initialization adds useful candidates beyond existing strong methods. A positive
algebra/noise check alone is neither a CVPR contribution nor a task-level gain.

## Source access

The ECCV 2024 primary PDF was read locally after web text retrieval timed out:
1,433,096 response-body bytes, SHA256
`d40f7a9c62cd08db7665b7dfa6c88d26aa1e5a69d33ca10c50f52649f294ab3a`.
Its private source receipt is `source-screen/source-receipt.json`; no PDF is
proposed for public redistribution. Web-tool paper/search traffic was not
separately metered. No model, dataset or image payload was acquired.
