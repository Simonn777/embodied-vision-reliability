# One affine correspondence with affine depth: mathematical screen

Status: analytical derivation only. No numerical fixture, solver, benchmark, image, depth array, or learned model was executed. The exact rational example below was derived by hand after the hypothesis was proposed. It establishes an existence statement, not robustness, novelty, or experimental admission.

## Model and three constraints

Use axial rays r=(x,y,1), calibrated pixel-to-ray Jacobians B=Jr1 and J=Jr2, and a nonsingular image-warp Jacobian A=df/du. All gradients must be in the same pixel-coordinate convention as their respective Jacobians. With g=gradient(d1), h=Aᵀgradient(d2), C=JA, p=d1+b and s=a d2+c,

    X=p r1,                 Y=s r2,
    U=p B+r1 gᵀ,            V=s C+a r2 hᵀ.

The physical equations R X+t=Y and R U=V give

    F(p,s,a)=UᵀU−VᵀV=0.

There are three scalar symmetric-Gram equations. Their expansion is

    p² BᵀB + p(Bᵀr1 gᵀ+g r1ᵀB) + ||r1||² ggᵀ
      = s² CᵀC + as(Cᵀr2 hᵀ+h r2ᵀC) + a²||r2||² hhᵀ.

They are quadratic in (p,s,a), equivalently in (b,a,c). At rank-two U,V, Gram equality is also sufficient for a unique **proper** rotation: map their two ordered tangent columns and their cross products. Then t=Y−RX. An improper normal flip is not a second SO(3) solution. Positive p,s and a are required; a zero translation has no defined translation direction. No metric scale is recovered: setting the first depth scale to one fixes the usual arbitrary length unit.

## Quartic reduction and degree limits

Write U=U0+bU1 and V=aV0+cV1, where

    U0=d1 B+r1 gᵀ, U1=B,
    V0=d2 C+r2 hᵀ, V1=C.

Using vec(S)=(S11,S12,S22), form

    q(b)=vec(U0ᵀU0)+b vec(U0ᵀU1+U1ᵀU0)+b²vec(U1ᵀU1),
    M=[vec(V0ᵀV0), vec(V0ᵀV1+V1ᵀV0), vec(V1ᵀV1)].

If det(M)≠0, y=M⁻¹q(b) must equal (a²,ac,c²). Hence

    y0(b)y2(b)−y1(b)²=0

is a polynomial of degree at most four. For each real root with y0>0, recover a=sqrt(y0), c=y1/a and explicitly check the original Gram equations, positive depths and proper pose. The sign pair (a,c) and (−a,−c) is removed by a>0. Degree at most four is not a claim that all four roots are real, admissible, distinct, or stable. Without this invertible-M reduction, three quadrics have at most eight isolated complex solutions by the usual Bézout bound; degenerate positive-dimensional solution sets are possible.

M nonsingularity is a real condition, not automatic. For example, at the target principal point with standard pinhole coordinates, Cᵀr2=0, so the target Gram depends only on s² and a² and M has rank at most two. Such data can still have an isolated solution in the full three-equation formulation. Rejecting this branch is an implementation limitation that must be declared or handled by a fallback.

## Exact physically realizable witness

Take K1=K2=I and

    r1=(0,0,1), r2=(1/2,0,1), d1=d2=1,
    g1=(1,0), g2=(2,8),
    A=[[1/2,-1],[0,1/3]], (b,a,c)=(0,1,0),
    R=[[2,-2,1],[2,1,-2],[1,2,2]]/3,
    t=(1/6,2/3,1/3).

RᵀR=I and det(R)=1. A source surface z1(u)=1+ux in a sufficiently small neighborhood of u=0 has tangent columns U1=(1,0,1), U2=(0,1,0). Applying this R and t gives Y0=(1/2,0,1), V1=(1,0,1), V2=(−2/3,1/3,2/3). Differentiating target perspective projection yields exactly A; differentiating target axial depth yields Aᵀg2=(1,2/3). Thus the supplied A and gradients arise from one rigidly transformed physical surface rather than independent invented observations. Both depths remain positive in a neighborhood.

For this example,

    M=[[2,1,1/4],[0,-4/3,-1/2],[1,14/9,10/9]],
    det(M)=−85/54.

Set x=(1+b)². Direct exact substitution gives

    M⁻¹q(b)=((47x+38)/85, −27(x−1)/85, 72(x−1)/85),
    y0y2−y1²=(x−1)(2655x+3465)/7225.

The only real, positive-source-depth solution is b=0,a=1,c=0. The Jacobian of (F11,F12,F22) with respect to (b,a,c) at this solution is

    [[2,-4,-1],[0,0,4/3],[2,-2,-14/9]],

whose determinant is −16/3. Therefore there is **no universal continuous gauge** beyond the chosen length scale for this input model. Generic local isolation is mathematically possible. This does not establish uniqueness for every physical observation, a numerical conditioning guarantee, or the reliability of measured affine/depth derivatives.

## Important degeneracies and gauge distinctions

- If g1=0, scaling (p,s,a) together scales both tangent matrices and translation. The source offset absorbs the scale change. Full calibration parameters have a continuous ambiguity even though rotation and translation direction may remain identifiable. Small g1 gives weak scale anchoring.
- If g2=0, V depends on a,c only through s=a d2+c. Their individual values are not identifiable from this patch, although a pose can still be identifiable. Small target gradients weaken this separation.
- With both gradients zero, tangent-Gram compatibility reduces to p²BᵀB=s²CᵀC; this is a special fronto-parallel configuration, not a generic full-calibration case. Parameter-rank failure must not automatically be relabeled task-direction failure.
- Singular/poorly conditioned A, nearly dependent tangent columns, det(M) near zero, repeated quartic roots, or nearly singular Gram Jacobians can amplify errors. A single patch also cannot validate that an affine correction remains valid over a whole image.
- Under raw-depth changes d1'=α1d1+β1 and d2'=α2d2+β2, with positive α1,α2 and correspondingly transformed gradients, choose b'=α1b−β1, a'=α1a/α2, c'=α1c−a'β2. Then U'=α1U, V'=α1V, R'=R and t'=α1t. Affine normalization alone adds no information. The extra observation here is the image-warp derivative together with depth derivatives.
- Radial range, inverse depth and axial z are not interchangeable. For example, an additive inverse-depth correction does not produce the U,V above. Model semantics and derivative units must be explicit before any empirical use.

## Prior overlap and what remains unproven

The differential point/tangent equations and proper rigid alignment are already central to Eichhardt–Barath's **1AC+D** (ECCV 2020), particularly Eqs. (3)–(6) and the scale/rotation solution. That paper assumes a relative depth scale; adding two offsets changes its unknown calibration model, not its differential geometry. [Primary paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf).

There is also **Semicalibrated Relative Pose from an Affine Correspondence and Monodepth** (ECCV 2024), addressing unknown focal lengths and fronto-parallel degeneracy. Its full overlap is being reviewed separately; this note does not infer absence of a scale-and-shift variant from its abstract. [Primary publication](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5645_ECCV_2024_paper.php).

The new Gram system is the differential limit of the three interpoint-distance equations in calibrated three-point affine-depth alignment. Take source points u, u+εe1, u+εe2 and corresponding first-order target/depth expansions. Divide their three squared-distance equalities by ε² and let ε→0: the limits give ||U1||²=||V1||², ||U2||²=||V2||² and ||U1−U2||²=||V1−V2||², equivalently all three Gram entries. Thus its algebra substantially overlaps MADPose's scale-and-shift distance elimination; it is not new rigid geometry. [MADPose primary paper](https://arxiv.org/html/2501.05446v3).

A direct derivative solver could still differ computationally and statistically from feeding nearby fabricated points into a three-point solver. That must be demonstrated, not assumed: finite-offset fabrication has higher-order error, while measured derivatives have their own correlated noise. In particular, estimating A from many sparse matches consumes those matches and does not create an independent one-point RANSAC observation or justify an inlier-probability advantage of q versus q³. No scientific execution is admitted by this mathematical existence result. Novelty, usable derivative accuracy, fair input cost, and downstream gain remain open.
