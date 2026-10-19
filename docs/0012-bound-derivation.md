# Independent mathematical audit: hierarchical bounds for truncated Sampson scoring

2026-10-19. Mathematical audit only. No model, dataset, downloaded asset, scientific inference, or historical repository modification. No novelty or practical speedup claim. The proposed real-arithmetic bounds are correct under stated conditions; numerical certification needs additional work.

## 1. Target, notation, and assumptions

For a nonempty match cluster C with n points, write homogeneous coordinates x=(u,v,1), x'=(u',v',1). For a 3x3 hypothesis E, define

    r_i = x_i'^T E x_i,
    q_i = r_i^2,
    d_i = (E x_i)_1^2 + (E x_i)_2^2
        + (E^T x_i')_1^2 + (E^T x_i')_2^2,
    t = tau^2 > 0,
    S_C = sum_i min(q_i/d_i, t).

First assume all actual d_i>0. Suppose valid uniform bounds are

    0 <= l <= q_i <= u,  0 <= Dlo <= d_i <= Dhi,
    Q = sum_i q_i.

With row-major e=vec(E), the feature vector is

    a_i = (u'u, u'v, u', v'u, v'v, v', u, v, 1),
    M = sum_i a_i a_i^T,  Q=e^T M e.

These are real-arithmetic identities. Numerically stored moments are not automatically exact.

## 2. Audit of the proposed bounds

**Upper: valid when Dlo>0.** Since each term is <=q_i/Dlo and <=t,

    S_C <= U = min(Q/Dlo, n*t).

If Dlo=0, retain only U=n*t; do not divide by zero. A zero box lower bound does not imply any actual point has zero denominator.

**Lower: valid for u>0 and Dhi>0.** Define K=max(Dhi,u/t). For every point, K>=d_i and K>=q_i/t, hence q_i/K is at most both q_i/d_i and t. Therefore

    S_C >= L = Q/max(Dhi,u/t).

Handle Q=u=0 directly as S_C=0 when actual denominators are positive. Do not evaluate 0/0.

**All-outlier shortcut: valid.** If l>=t*Dhi, every q_i>=t*d_i, so S_C=n*t. Use multiplication instead of l/Dhi. Boundary equality is fine because min(q_i/d_i,t)=t there. This is a real exact identity, not necessarily bitwise equality with a floating-point loop summing t repeatedly.

**All-inlier shortcut: only interval-exact.** If u<=t*Dlo with Dlo>0, truncation is inactive, but

    Q/Dhi <= S_C=sum_i q_i/d_i <= Q/Dlo.

It is generally false that a single moment matrix gives the exact all-inlier sum. Exactness follows only if all denominators are the same known value, or if a further exact rational summation is performed. Do not label an all-inlier node 'exact' merely because truncation is inactive.

## 3. A strictly useful inexpensive lower-bound improvement

Let f(q)=min(q/Dhi,t), which is nondecreasing and concave on [l,u]. Its chord lies below f. For u>l,

    B = (f(u)-f(l))/(u-l),
    L_chord = n*f(l) + B*(Q-n*l),
    S_C >= L_chord.

The original lower bound is the chord from 0 to u. Incorporating l>0 can tighten it substantially. For l=0 they coincide. Use the maximum of independently valid lower bounds, although in exact arithmetic the chord using l is at least as strong here. If u=l, use n*f(l).

This chord is the tight convex relaxation using only Q,n,l,u and the common denominator upper bound. There is a potentially tighter finite-n extreme-point construction: put as many q_i as possible at l or u and at most one in between. It requires floor/count boundary handling and is not recommended in an initial certified prototype.

A cheap independent lower bound is n*min(l/Dhi,t). It is mostly redundant with the chord but useful as a sanity check. Bounds that exploit actual q-d correlation require more information than Q and extrema; do not assume numerator and denominator are independent.

## 4. Bounding residuals on a 4D coordinate box

Write E=[[A,b],[c^T,f]], with A 2x2. Let p=(u,v), p'=(u',v'), box centers p0,p0' and nonnegative coordinate half-widths h,h'. Then

    r = p'^T A p + p'^T b + c^T p + f.

The residual is multiaffine in the four coordinate variables, so its minimum and maximum over the box occur among its **16 vertices**. A real-exact 16-corner evaluation therefore supplies the tight box range [rlo,rhi]. It is still an enclosure of the actual stored matches, because the box admits coordinate combinations absent from the cluster.

Convert to squared bounds:

    l = 0                         if rlo<=0<=rhi,
        min(rlo^2,rhi^2)          otherwise;
    u = max(rlo^2,rhi^2).

Cheaper center/extent alternative:

    r0 = p0'^T A p0 + p0'^T b + c^T p0 + f,
    wr = A^T p0' + c,
    wl = A p0 + b,
    B = abs(wr) dot h + abs(wl) dot h' + h'^T abs(A) h,
    r in [r0-B, r0+B].

All operations need outward error control for a numerical certificate. Store explicit minima/maxima rather than an accidentally shrunken center/radius box. Midpoint and radius arithmetic can otherwise exclude extremal input points by one ULP.

## 5. Denominator bounds can be stronger than generic 4D intervals

The denominator separates:

    d(p,p') = ||A p+b||^2 + ||A^T p'+c||^2.

Each summand is a convex quadratic on a 2D rectangle. Its maximum occurs at one of four corners; Dhi is the sum of the two maxima. Only **4+4 corner evaluations** are needed, and this is the exact real box maximum.

For a tight box minimum, solve each 2D convex box-constrained quadratic: evaluate four corners, each edge's clipped 1D stationary point, and any feasible unconstrained stationary point. Singular A needs explicit handling; skipping singular interior cases without proof can overestimate the minimum and invalidate the upper score bound.

A simpler always-safe interval lower bound is usually preferable initially. For each affine component g_j, derive center z_j and radius b_j>=0, then bound its square by

    max(abs(z_j)-b_j,0)^2 <= g_j^2 <= (abs(z_j)+b_j)^2.

Sum all four lower/upper terms. This loses correlations but requires no matrix inversion and safely handles singular hypotheses. Dlo=0 means 'upper bound is cap-only; subdivide if useful', not failure or permission to add an unreported epsilon.

## 6. Floating-point certification: mandatory distinction

The target must be stated first: (a) the mathematical score on exact values represented by input floats; or (b) bit-for-bit equality with a particular NumPy/BLAS/reference reduction. These are different targets. Aggregating n copies of t as n*t can change the last bits versus summing them one by one. The usual useful target is (a), with rigorous intervals enclosing the mathematical value and an explicit uncertainty policy for ties. A tolerance-only numerical comparison is not a proof of either target.

Required safeguards:

- Maintain certified intervals [Qlo,Qhi], [l,u], [Dlo,Dhi]. Evaluate lower bounds using Qlo and downward-rounded operations, and upper bounds with Qhi and upward-rounded operations. Intersect Q's enclosure with [n*l,n*u] only using safe rounding. If the intersection is empty, treat the node as invalid and fall back; never silently swap/clamp contradictory bounds into apparent success.
- e^T M e can suffer severe cancellation precisely near an excellent geometric fit. A float64 matrix product, Kahan sum, `max(Q,0)`, or arbitrary relative tolerance is not a certificate. The construction of M must account for rounding in coordinate products and accumulated outer products, and the query quadratic form must account for its own rounding. Interval arithmetic at build and evaluation, or derived forward-error bounds including input-feature errors, is required.
- One feasible conservative implementation uses outward-rounded scalar interval arithmetic for each primitive (+,-,*,/), with `nextafter` around correctly rounded float64 results. Do not delegate certified reductions to unspecified fast-math/BLAS operations. Overflow should widen to infinity or trigger safe fallback, not produce an invalid finite bound. Subnormal and cancellation cases must be covered.
- A fixed relative error allowance fails when Q is near zero. An absolute allowance based on sums of absolute products is needed if using analytic floating-error bounds. Such allowances can become too loose to accelerate good hypotheses. This is a practical limitation, not grounds to tighten the interval heuristically.
- Check all-outlier as a directed inequality between a lower enclosure for l and an upper enclosure for t*Dhi. Check all-inlier using an upper enclosure for u and a lower enclosure for t*Dlo. Do not base classification on midpoint values.
- Do not report the floating value n*t as an exact scalar unless representability is established. Return its outward interval, or a symbolic capped contribution combined with certified accumulation.
- Normalize finite nonzero E by a stable positive scale such as max(abs(E)) before arithmetic to reduce overflow/underflow. Reject an identically zero matrix. Treat the resulting floating coefficients as the pinned hypothesis for the mathematical score unless normalization-rounding equivalence is explicitly bounded.

Because 9x9 moment evaluation costs much more than a few point residuals, a good initial policy is to stop using moments below a moderate leaf size and evaluate original coordinates directly with certified scalar operations. A high-precision or exact-rational tiny synthetic reference can validate formulas, but no such test has been run in this audit.

## 7. Undefined denominators, invariance, and other edge cases

- Exact q>0,d=0 can be interpreted as infinite untruncated error, hence capped t, if this convention is declared. Exact q=d=0 is undefined for the Sampson approximation. Either reject/mark the hypothesis invalid, or define and disclose a convention shared by every comparator. Do not silently set it to zero for the candidate only.
- If the reference instead uses d_eps=max(d,eps), all bounds must target that exact function: replace denominator bounds by max(Dlo,eps), max(Dhi,eps). An absolute eps breaks E-scale invariance unless E normalization is fixed first. Exact E-scale invariance belongs to the unregularized quotient, not a differently clamped implementation.
- Under E -> alpha E (alpha !=0), q,d,l,u,Q all scale by alpha^2, so the real score and real bound ratios are invariant. E sign is harmless. Numerical values need not be bit-identical after arbitrary rescaling.
- Sampson pixel distance is not invariant under arbitrary image coordinate normalization. Similarity transformations need corresponding threshold scaling; anisotropic camera normalization changes the metric. Do not mix pixel-space tau with normalized-coordinate denominators.
- Empty clusters score zero. t=0 can be a trivial separately handled objective; negative/nonfinite t is invalid. Nonfinite coordinates/matrices and invalid boxes must not be selectively omitted. Duplicate correspondences retain their multiplicity in n and M.
- A rank-deficient E is not automatically invalid for these algebraic inequalities, but it increases the chance of zero denominators. Essential/fundamental rank constraints are a separate hypothesis-generation issue.

## 8. Ranking logic and likely computational limit

For each hypothesis, maintain a partition of its unrefined nodes and sum their score intervals; refined children replace their parent, never coexist with it. Hypothesis h can be pruned when L_h>U_best. Equality needs an explicit deterministic tie policy; strict comparison is the safe default. Refine the largest useful interval width per estimated cost. A winner can be certified without every score being exact if its upper bound is below every competitor's lower bound. If a reported exact winner score is needed, evaluate its remaining leaves or return its certified interval. For top-k, use the kth smallest available upper bound as the cutoff.

The strongest opportunity is many clearly bad hypotheses whose clusters quickly satisfy the all-outlier condition. Good hypotheses may have residual intervals crossing zero, tiny denominators, or q-d coupling, forcing substantial subdivision. Mixed clusters are not rare merely because the true model is good. A 9-dimensional quadratic form (~45 unique moments), 16 residual corners, denominator bounds, outward interval operations, and tree traversal can cost more than vectorized scoring of dozens of matches. Python branching is especially unfavorable. Store 45 symmetric moments rather than 81 if possible, use cheap bounds before tight corners, and compare complete end-to-end hypothesis ranking to a well-vectorized exact baseline.

**Audit decision:** mathematically viable as a bound-and-refine prototype. The supplied scalar inequalities pass with positive denominators. Do not claim exact all-inlier aggregation, floating-point certification, novelty, or practical savings yet. A numerically correct but slower implementation is an expected possible outcome, not a research result.
