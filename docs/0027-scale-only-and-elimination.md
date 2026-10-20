# 0027: axial scale-only comparator and alternative eliminations

Source/math only; no solver or benchmark was run for this note. These are prospective mathematical branches, not permission to retune the current generic implementation after its qualification.

## Scale-only 1AC+D in the current convention

Let X0=d1 r1, Y0=d2 r2, U0=d1 J1+r1 g1ᵀ and V0=(d2 J2+r2 g2ᵀ)A. Our pose convention maps camera 1 to camera 2. The scale-only model is

    R X0+t=a Y0,    R U0≈a V0,    a>0.

After eliminating t, minimize ||R U0−a V0||F² over proper R and positive a. If

    V0 U0ᵀ=P Σ Qᵀ,
    D=diag(1,1,det(PQᵀ)),

then

    R=P D Qᵀ,
    a=tr(ΣD)/||V0||F²,
    t=a Y0−R X0.

The minimized tangent objective is ||U0||F²−tr(ΣD)²/||V0||F². This follows by maximizing tr(V0ᵀRU0), then differentiating the scalar quadratic in a. A formulation with the scale multiplying R U0 instead uses a different denominator and length gauge; those conventions must not be mixed.

Require finite inputs, nonzero denominator and positive a, rank-two tangents for uniquely determined rotation, positive corrected depths on the declared evaluation support, and nonzero t for a translation-direction result. The SVD solves a least-squares scale-only problem even when the two Gram matrices are not exactly proportional. It does not estimate the two additive offsets.

This is the proper-rotation SVD/Umeyama comparator already developed in [Eichhardt–Barath, ECCV 2020, Eqs. (5)–(11)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf), expressed here with axial rays and our source-to-target convention. Existing unit-bearing/range code requires the corresponding depth and gradient conversion, not just renaming its input depth. The published ordered-frame approximation is a separate comparator, with its axis-order asymmetry retained explicitly.

The comparator does not depend on invertibility of the shift solver's M. At a principal point with two independent lifted tangents, it remains available. In the both-flat-gradient case, source depth is only a length gauge and the local affine model's pose can often be represented by this simpler scale-only model; failure to identify every nuisance calibration coefficient is not itself a reason to discard the pose.

## M singularity is not removed by changing pixel coordinates

For a consistent source reparameterization δu=L δu', U'=UL and V'=VL. Every Gram matrix changes by the same invertible congruence G↦LᵀGL, so rank(M) is invariant. A target coordinate change cancels between J2, g2 and A. Merely putting the matched pixel at coordinate zero does not make its calibrated ray the optical axis.

At the **physical** target principal point, r2=e3 and C=J2A has zero third row, hence Cᵀr2=0. The target Gram depends only on s² and a², where s=a d2+c. Consequently M has rank at most two even when the complete three calibration equations are locally isolated. Rotating the camera frame to center a ray changes the definition of axial depth: an axial offset becomes a spatially varying term. It cannot silently remove this singularity while preserving the same affine-depth model.

Near this special case, M can be ill-conditioned while the complete calibration Jacobian is not. Thus numerical failure of the chosen M inverse is not sufficient evidence that the physical problem is ill-conditioned.

## Alternative source-basis quartic

Write U=pB+r1g1ᵀ, with p=d1+b, and define the three-column source basis

    N=[vec(BᵀB),
       vec(Bᵀr1 g1ᵀ+g1 r1ᵀB),
       vec(||r1||² g1g1ᵀ)].

For q=c/a, V=a(V0+qV1), where V1=J2A. If N is invertible, let

    w(q)=N⁻¹ vec[(V0+qV1)ᵀ(V0+qV1)].

The equations are (p²,p,1)=a²w(q). Therefore

    w0(q)w2(q)−w1(q)²=0,
    a²=1/w2(q), p=w1(q)/w2(q), b=p−d1, c=a q.

This is another polynomial of degree at most four. Require w2>0, p>0, a(d2+q)>0, and original-equation checks. It can handle some singular/poorly conditioned M cases. It is not universally applicable: for example, at the source principal point the middle column of N vanishes. The two inverse-based branches therefore do not cover the entire identifiable set.

Choosing the better-conditioned elimination before solving is mathematically compatible with the same estimator, but its scaling, rank threshold, tie rule and failure policy must be specified before the new qualification. It is a prospective numerical implementation change, not a result-dependent rescue of an already failed variant.

## Direct rank-two branch and principal-point simplification

If rank(M)=2, let l span its left nullspace. Solvability first requires

    lᵀq(b)=0,

a scalar quadratic in b. For each root, solve the consistent linear system as y=y_particular+λ n, where n spans the right nullspace. Enforcing y0y2−y1²=0 is then at most quadratic in λ. Recover a,c and check the original constraints and positivity as before. This provides an exact branch without pretending M is invertible. If the compatibility polynomial vanishes identically, a continuous calibration family is generically possible; rank/conic singularities need explicit handling rather than an arbitrary pseudoinverse output.

An especially simple case occurs when **both** rays are on their optical axes. Cross terms vanish, so

    p² BᵀB + g1g1ᵀ = s² CᵀC + a² hhᵀ,
    h=Aᵀg2.

This is a 3×3 linear system in (p²,s²,a²). When its coefficient matrix is invertible and all three values are positive, positivity fixes p,s,a by their positive square roots, followed by b=p−d1 and c=s−a d2. An exact centered variant of the earlier witness uses K=I, both rays e3, d1=d2=1, g1=(1,0), g2=(1,4), A=[[1,−2/3],[0,1/3]], the same R, and t=(−1/3,2/3,1/3). Its square-variable coefficient matrix is [[1,−1,−1],[0,2/3,−2/3],[1,−5/9,−4/9]], with determinant 2/3 and solution (1,1,1). This additional branch calculation is analytical and was not part of the executed off-axis Fraction check. Thus a generic-M-only implementation can reject a fully identifiable and algebraically simple physical case.

When g1=0, N loses rank and there is a common local length/calibration scaling family; when g2=0, a and c enter only through s. These cases may still admit useful pose estimates from a reduced model or the scale-only comparator. Do not equate full-calibration rank loss with a null direction that changes R or normalized t.

## Exact-witness cross-check reviewed

Root independently implemented the already specified rational witness in `work/0027-preparation/exact_witness.py`, SHA256 `ae0e8786b3bc526a5f129c6544aa48da9d9dfff0fe3cf8d16f1af3016b956450`, and executed it once. I read its source and completed receipt/output without rerunning it. All constants and matrix operations remain `fractions.Fraction`; the checks derive the physical U,V, M, polynomial coefficients and Jacobian rather than only comparing a prewritten determinant. The saved output verifies det(M)=−85/54, calibration-Jacobian determinant −16/3, and the stated quartic factorization.

The recorded operation exited 0 in 0.17267375 seconds with empty stderr and no remaining descendants. This is an independently coded exact-arithmetic check of one analytical construction, not an independent-team replication, manufactured noise qualification, generic numerical stability test, or pose-accuracy result.
