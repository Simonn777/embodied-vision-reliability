# 0012d: lazy grid lower bounds and cost conditions

This is the mathematical companion to
[`0012d-lazy-grid-screening.md`](../protocols/0012d-lazy-grid-screening.md).
It describes an exploratory implementation after 0012c failed its runtime
gate. The algebra below concerns exact real arithmetic. Outward numerical
guards require independent implementation review and artificial checks; this
document does not claim a formal floating-point certificate, novelty, speed,
or improved pose quality.

## Fixed grouping and objective

For homogeneous image points x=(x0,x1,1), y=(y0,y1,1), define

    r = y^T E x
    d = (Ex)_0^2 + (Ex)_1^2 + (E^T y)_0^2 + (E^T y)_1^2
    cost = min(r^2/d, t),  where t = tau_squared > 0.

The shared implementation convention is cost=t whenever d=0, including
r=d=0. Every cell cost is therefore in [0,n*t]. E normalization, residual
evaluation and completed-score reduction must match all controls.

The fixed sparse grid key is floor((x0,x1,y0,y1)/256), in raw pixels. Occupied
cells follow first appearance in the input; their members retain original
input order. Only n>=64 cells are screened. Grouping and actual coordinate
extrema require one input pass. Each eligible cell's moments are built on
its first actual Cantelli request by a further pass over that cell's members,
then cached across hypotheses within the call. This is not a claim that all
statistics are obtained in one total pass.

Fixed 256-pixel width avoids a preceding global extent pass and fixes the
physical coordinate scale before observation. Four bins per axis would have
resolution- or extent-dependent widths and would be a different protocol.
The chosen grid remains resolution-sensitive and can have up to N occupied
cells; it is not limited to 256 cells. The size cutoff is a declared cost
heuristic, not an optimality result. There is no hierarchy, sorting, refinement,
or data-dependent model ordering.

## Cheap geometric bound

Let xc=(cx0,cx1,1), yc=(cy0,cy1,1) be the centers of a cell's actual coordinate
box and hx,hy its nonnegative two-dimensional halfwidths. Set

    p = E xc,   q = E^T yc,   r0 = yc^T p.

For x=xc+(dx0,dx1,0), y=yc+(dy0,dy1,0), the exact expansion is

    r = r0 + q[:2]^T dx + p[:2]^T dy + dy^T E[:2,:2] dx.

Consequently every residual is in [r0-R,r0+R], where

    R = sum_j |q_j| hx_j + sum_i |p_i| hy_i
        + sum_{i,j<2} |E_ij| hy_i hx_j.
    r_min = max(|r0|-R, 0).

An upper bound on every denominator in the cell is

    D = sum_{i<2} (|p_i| + sum_{j<2}|E_ij| hx_j)^2
        + sum_{j<2} (|q_j| + sum_{i<2}|E_ij| hy_i)^2.

For D>0 the geometric lower cost is

    L_geometry = n * min(r_min^2/D, t).

If a valid denominator upper bound is exactly zero, all cell costs equal t
under the shared convention. A nonfinite or invalid computed bound instead
supplies zero lower cost; it must not be interpreted as this exact-zero case.

The implementation must enclose the input extrema when forming centers and
halfwidths: an ordinary midpoint/radius calculation can shrink by an ULP.
It must also account for rounding in p, q, r0, the radius, squared terms and
their sums. For example, use outward affine intervals for p and q, their
upper absolute endpoints in R and D, and an outward residual interval to
obtain r_min. Check raw values for finiteness before max/min clamps.

## Signed moments without an absolute-moment matrix

For each cell member k, let A_k=vec(y_k x_k^T) in row-major order and
e=vec(E) in that same order. Store only the 45 distinct entries of
M=sum_k A_k A_k^T. Since A_k,8=1, M[:,8]=sum_k A_k, so the same summary
gives both the signed mean and total squared residual:

    mu = e^T M[:,8] / n,   Q = e^T M e = sum_k r_k^2.

The raw triangular Q evaluation retains the multiplication order
(e_i*M_ij)*e_j, with multiplicity two off the diagonal. Precomputing e_i*e_j
can underflow before multiplication by a large representable M_ij.

To bound absolute reduction scales without 45 additional absolute moments,
let D_j=sum_k A_kj^2 and B_ij=sum_k |A_ki A_kj|. Cauchy gives

    B_ij <= sqrt(D_i D_j).

With true outward upper bounds D_j_plus and upward-rounded
s_j=sqrt(D_j_plus), form S=sum_j |e_j| s_j outward. Then

    sum_{i,j} |e_i e_j| B_ij <= S^2,
    (1/n) sum_j |e_j| sum_k |A_kj| <= S/sqrt(n).

These are scales for enclosing the Q and mean computations, respectively.
The square, sum and reciprocal square root also need outward treatment.
Cache the nine s_j when the moments are first built, independently of E.
Do not assume the rounded stored matrix is positive semidefinite, or use its
raw diagonal as a true upper bound. The diagonal enclosure must include
feature-product formation, squaring and positive reduction error, with a
subnormal absolute error floor. The subsequent contraction guard must cover
moment construction and arithmetic reduction, not merely its last addition.
On nonfinite intermediate values, abandon this moment certificate.

The Cauchy scale can exceed the previously stored absolute-moment scale.
Removing that storage is therefore a changed numerical bound with a possible
loss in pruning, not an algebraically identical performance optimization.

## Cantelli lift and shared computation

Use the mean interval to obtain m_lower, its distance from zero. From an
upper Q enclosure obtain

    v_upper >= Q_upper/n - m_lower^2,

with the division rounded upward and squared subtraction rounded
conservatively. A nonfinite or negative conservative upper result abandons
Cantelli; it must not be repaired by clamping it to zero. This bounds the variance
of the signed residual of a uniformly selected cell member. It is a finite
population inequality; no independence or sampling assumption is required.

For any 0<a<m_lower, Cantelli applied after orienting the mean positively
gives a lower bound on the fraction of members whose |r| is at least a:

    fraction >= (m_lower-a)^2 / (v_upper + (m_lower-a)^2).

For D>0 this yields the valid cell lower cost

    L_Cantelli(a) = n * min(a^2/D, t)
                    * (m_lower-a)^2/(v_upper+(m_lower-a)^2).

Retain the previously declared threshold

    a = min(m_lower/2, sqrt(t)*sqrt(D)).

It is valid only when the computed a remains finite and strictly between zero
and m_lower; otherwise the extra lower bound is zero. Obtain a downward
delta=m_lower-a and squared value w. The ratio w/(v_upper+w) needs an upward
denominator and a downward result; using the same w preserves the monotonic
lower-bound substitution. Likewise a^2/D and the final product need lower
enclosures. Zero variance
is permitted when a<m_lower; zero mean gives no Cantelli contribution.
Sign-symmetric residuals can have large costs but zero mean, so this mechanism
cannot resolve every weak geometric box bound.

Compute geometry and D once per model/cell. If its lower bound already
reaches the cell cap, no moments are needed. Otherwise mode4 may request
cached moments and use

    L_cell = max(L_geometry, L_Cantelli).

Both terms bound the same cost and must never be added. The improvement
over geometry is their nonnegative difference. Shared geometry, denominator,
mean/Q and their error scales should not be redundantly evaluated.

## Lazy verification and matched controls

Model0 receives a full exact score as the initial incumbent; challengers keep
their original model order. Accumulate downward-enclosed lower costs over
eligible cells in first-seen order. The unvisited cells contribute zero.
Reject a challenger only when this prefix lower bound strictly exceeds the
guarded exact incumbent. This also protects deterministic lowest-index ties.

If the total eligible-cell cap cannot exceed the incumbent, screening cannot
reject. Likewise, when prefix lower plus remaining eligible cap cannot exceed
it, stop screening and start exact fallback. These are only exits from the
attempt to obtain a certificate; they never reject a challenger.

Fallback starts from point zero in concatenated grid order and evaluates
ordinary exact block256 partial costs. No unevaluated point costs are reused.
Completed scores are reduced in original input order to match the reference.
All grouping, moment construction, screening and fallback work is charged.

The five modes are full exact (0), frozen-random-order partial (1), same-grid-
order partial (2), lazy geometry plus that partial fallback (3), and geometry
plus Cantelli with the same fallback (4, primary). Modes2 and3 pay only for
the structures they require. Mode2 isolates grouping/order effects; mode3
isolates the actual value and cost of signed moments.

## Cost condition and stopping diagnostic

For a queried cell, let P be moment-building cost per point, B the extra
Cantelli query cost, V the number of actual queries, C exact residual cost,
and delta_R the residual evaluations saved against the matched control.
Ignoring common work, a necessary cost condition is approximately

    delta_R*C > n*P + V*B + additional grouping/traversal overhead.

A 45-entry signed summary requires roughly 45 products and additions per
member before any query; arithmetic counts are not wall-time estimates.
For N=10000, n>=64 permits at most 156 eligible cells, but 63 challengers
could still make 9828 bound queries per pair, repeatedly querying the same
cached summaries; at most 156 summaries are built. Thus eligibility alone does not
establish a useful speedup. Both cheap geometry and lazy stopping must avoid
substantial work; cached summaries must save enough subsequent exact work to
amortize their construction. The previous real-development target is about
8.70ms total over 16 pairs, or 0.544ms per pair including setup. A cheaper
tree replacement alone cannot establish that target.

The protocol first uses a deliberately favorable 10000-point, 64-model
artificial case: x=(0,s), y=(0,s+10), s in [0,240], all in one grid cell;
E has E[1,2]=-1, E[2,1]=1, E[2,2]=c and all other entries zero. Model0 uses
c=9 and challengers c=0. At t=1, their point costs are respectively 0.5 and
1. The independent box interval crosses zero, while signed residual variance
is zero. This tests whether Cantelli can repay its overhead when the box
bound misses a strong correlation. It does not estimate real-data quality.

After independent artificial correctness checks, require candidate median
outer-call time at most two-thirds of every control under the frozen timing
schedule. If this favorable case fails, stop this configuration before real
data replay; do not retune the grid, cutoff or fixture. A pass admits only
the already specified exploratory development replay. It establishes neither
held-out generalization nor a publishable contribution.
