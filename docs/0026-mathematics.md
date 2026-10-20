# Exact product-weight distance loss: independent mathematical review

Analytical review only. No arrays, fixtures, fits, models or prior experiment outputs were evaluated. The identity below is correct under its stated conditions; this review establishes neither novelty nor measured benefit.

## Objective, signs and the factor of two

Let `r_i,r'_i` be calibrated axial rays, and define

\[
X_i=(d_i+b)r_i,\qquad Y_i=(a d'_i+c)r'_i,\qquad a>0,
\]
\[
L=\sum_{i<j}w_iw_j\big(\|X_i-X_j\|^2-\|Y_i-Y_j\|^2\big)^2,
\quad w_i\ge0,\quad W=\sum_iw_i>0.
\]

The weights must be fixed for this objective evaluation. Write `h_i=(d_i r_i,r_i,d'_i r'_i,r'_i)` as four three-vector blocks, `mu=Σw_i h_i/W`, and `v_i=h_i−mu`. With `z=(1,b,b²,a²,ac,c²)`, the symmetric block matrix is

\[
A(z)=\begin{pmatrix}
 I&bI&0&0\\ bI&b^2I&0&0\\
 0&0&-a^2I&-acI\\0&0&-acI&-c^2I
\end{pmatrix}=\sum_{k=0}^{5}z_k A_k.
\]

Thus the cross-term basis matrices have **off-diagonal I and −I**, not half these values. Their quadratic forms contain `2 p·r` and `−2 p'·r'`. Pair differences remove mu exactly.

Set `C=Σw_i v_i v_i^T`, `q_ki=v_i^T A_k v_i`, and `m_k=tr(A_k C)`. For one pair,

\[
\delta_{k,ij}=q_{ki}+q_{kj}-2v_i^TA_kv_j.
\]

Expanding the **ordered** double sum and dividing by two yields

\[
\sum_{i<j}w_iw_j\delta_{k,ij}\delta_{l,ij}
=W\sum_iw_iq_{ki}q_{li}+m_km_l+2\operatorname{tr}(A_kCA_lC).
\]

The terms linear in a bilinear cross-product vanish because `Σw_i v_i=0`. The last term follows from
`Σij wi wj (vi^T Ak vj)(vi^T Al vj)=tr(Ak C Al C)` for symmetric Ak/Al. The diagonal pairs contribute zero. Therefore the proposed `L=z^T H z` is exact, with no missing outer factor of two. An ordered `i!=j` objective would instead be `2 z^T H z`.

H is positive semidefinite in exact arithmetic: it is the weighted Gram matrix of the six-vector `delta_ij`. Centering must use the same point weights as the product-weight objective. W-normalized weights are also valid if their overall scaling is tracked.

## What the compression saves, and what it does not

Two streaming passes can compute mu, C and `Σw_i q_i q_i^T` in O(N) time with fixed-size storage; h has dimension 12 and H is 6×6. The remaining trace operations have constant dimension. Subsequent evaluations and derivatives in `(a,b,c)` have cost independent of N. This is an exact reduction of the specified complete-pair quartic objective, not an approximation requiring additional observations.

It does **not** make the optimization convex. An unconstrained six-vector solution can leave the physical lift: `z0=1`, `z2=z1²`, `z3=a²>0`, `z4=ac`, `z5=c²` must hold together. Optimizing the three original variables or enforcing these relations is necessary; a smallest eigenvector of H is not generally a valid calibration.

The reduction is potentially useful for repeated evaluation or global search of this exact objective. Fair algorithmic evaluation would charge moment construction and compare the same objective, solution quality and numerical accuracy. It must also compare competitive existing estimators; beating a naïve O(N²) implementation alone does not establish that practical pose estimation becomes faster or better. Algorithmic benefit remains possible without claiming new geometric information.

## Gauge and affine reparameterization

For positive raw-depth changes `dtilde_i=u d_i+v` and `dtilde'_i=p d'_i+q`, choose

\[
\tilde b=ub-v,\quad \tilde a=ua/p,\quad
\tilde c=uc-uaq/p.
\]

Then both reconstructed point clouds scale by u, each squared-distance difference scales by `u²`, and `Ltilde=u⁴L`, for unchanged weights and correspondingly transformed constraints. Rotation and translation direction are unchanged; translation magnitude scales by u. Therefore the calibration is not new metric-scale information and raw global affine preprocessing alone does not alter the attainable pose family. If weights or parameter bounds are changed inconsistently, this equivalence need not hold.

Fixing the first depth coefficient to one removes the common multiplicative zero-scale gauge. It does not establish uniqueness of b/a/c or a nonzero camera baseline.

## Robustness and identifiability limits

* Arbitrary edge weights `w_ij`, correspondence-specific missing pairs, or robust losses such as `Σrho(delta_ij)` are not generally representable by fixed products `w_iw_j`. Pairwise IRLS produces residual-dependent edge weights that usually break this formula. Point-weight updates retain the identity for each new product-weight problem, but require rebuilding moments and do not silently become the same pairwise robust objective. Any approximation must be named.
* Corrected depths must be positive on the declared active support: `d_i+b>0` and `a d'_i+c>0`. For constant raw depths in both images, choosing shifts toward their negative constants collapses both clouds toward zero; strict positivity excludes the exact collapse but can leave a zero infimum at its boundary. Near-constant depth can be weakly constrained. This is a specific degeneracy, not a claim that all generic data collapse.
* Repeated rays, collinearity, insufficient angular/depth variation, zero-baseline identical views and special symmetric configurations can leave calibration or pose ambiguous. With identical rays and identical depth arrays in both views, `a=1,c=b` gives identical clouds for every admissible b. Product weighting cannot resolve missing identifiability.
* At an exact fit, local calibration identifiability can be checked by the rank-three Jacobian of the active pair discrepancies. Its rows are
  \[
  (\partial_b\delta,\partial_a\delta,\partial_c\delta)
  =\big(2\Delta r\cdot(\Delta(dr)+b\Delta r),
  -2\Delta(d'r')\cdot(a\Delta(d'r')+c\Delta r'),
  -2\Delta r'\cdot(a\Delta(d'r')+c\Delta r')\big).
  \]
  Full rank gives a locally isolated calibration under regularity; it does not rule out other separated roots or guarantee noise stability. A lifted H rank alone is not this test.
* Complete distance equality determines congruence in O(3), including reflection. For noncoplanar points, signed tetrahedral volumes or a proper-rotation fit must reject a reflected solution. For collinear geometry, rotational ambiguity remains. A zero recovered t leaves translation direction undefined.

## An estimator can use it, with explicit physical recovery

There is no mathematical prohibition on an estimator using this objective when the calibration is identifiable and a proper positive-depth solution exists. After enforcing the lift, positive scale and depths, recover the weighted rigid transform. If `S=Σw_i (Y_i−Ybar)(X_i−Xbar)^T=U diag(sigma)V^T`, use

\[
R=U\,\mathrm{diag}(1,1,\det(UV^T))V^T,\quad
t=Ybar-RXbar.
\]

Check the actual SO(3) alignment residual and nonzero translation, rather than treating small distance loss as sufficient. A noisy minimum can still fit squared 3D distances poorly in the task-relevant image geometry; this objective is not automatically a pixel-noise likelihood or an outlier-robust estimator.

Numerically, the centered fourth-moment construction and `z^T H z` can lose precision near cancellation, despite exact positive semidefiniteness. This review provides no floating-point certificate. Stable accumulation, explicit finite checks and objective-agreement evidence are separate implementation requirements; silently clipping a negative computed loss does not prove equality.

**Conclusion:** the factors, signs and O(N) identity are sound for the fixed complete-product-weight objective. The physical estimator and possible computational benefit are coherent, conditional hypotheses. General pairwise robustness, global uniqueness, reflection rejection, numerical accuracy and novelty are not supplied by the identity. The preceding distance-elimination/lift construction must not itself be renamed as a new contribution.

## Convex lower bound and its precise scope

Replace the physical monomials by `z=(1,b,B,A,C,D)` and impose

\[
\begin{pmatrix}B&b\\b&1\end{pmatrix}\succeq0,
\qquad
\begin{pmatrix}A&C\\C&D\end{pmatrix}\succeq0.
\]

The first constraint is equivalent to `B>=b²`; both are two-dimensional semidefinite cones with rotated second-order-cone representations. For exact PSD H, minimizing `z^T H z` is convex. Every physical calibration maps into this set, so dropping depth positivity and the rank constraints supplies a lower bound. In particular, relaxed `A=0` and nonpositive reconstructed depths are allowed in the lower-bound problem; they do not become physical estimates.

Exact equality `B=b²`, `AD=C²`, and `A>0` permits `a=sqrt(A), c=C/a`. If the reconstructed depths are positive on the entire declared support, this gives a feasible physical calibration. Equality between a valid dual lower bound and its physical objective then certifies a global optimum of **this fixed-weight distance subproblem**. A nonzero certified gap also bounds this subproblem's suboptimality. Neither conclusion certifies a robust outer iteration, a proper camera pose, a pixel-noise objective, or real-data accuracy. A floating-point primal/dual residual and gap are numerical diagnostics unless rounding and feasibility have been bounded; an eigensolver or clipping H to PSD does not supply that proof.

For implementation sign checking, set `x=(b,B,A,C,D)`, `f=H00+2 H0x x+x^T Hxx x`. A multiplier `lambda>=0` for `b²-B<=0` and a PSD cone dual S give

\[
Q=H_{xx}+\lambda e_b e_b^T,\quad
\ell=2H_{0x}+(0,-\lambda,-S_{11},-2S_{12},-S_{22})^T.
\]

If Q is positive definite, the dual value is `H00 - (1/4) ell^T Q^{-1} ell`. The factor two on the off-diagonal cone entry and the factor one quarter are required. Singular PSD Q has a finite dual value only when ell is in its range; declining that case is conservative and valid for a prototype, although it sacrifices potentially useful bounds.

## Necessary constant-time Procrustes comparator

The same centered 12-by-12 C also supplies a stronger computational comparator using only second moments. Denote its blocks by p=`d r`, r, p'=`d' r'`, r'. Then

\[
C_{XX}=C_{pp}+b(C_{pr}+C_{rp})+b^2C_{rr},
\]
\[
C_{YY}=a^2C_{p'p'}+ac(C_{p'r'}+C_{r'p'})+c^2C_{r'r'},
\]
\[
C_{XY}=aC_{pp'}+cC_{pr'}+abC_{rp'}+bcC_{rr'}.
\]

For `CXY=U diag(sigma) V^T`, the weighted point-alignment objective after eliminating proper rotation and translation is

\[
F=\operatorname{tr}C_{XX}+\operatorname{tr}C_{YY}
-2[\sigma_1+\sigma_2+\det(UV^T)\sigma_3].
\]

This uses a fixed-size 3-by-3 SVD and weighted means, hence also has O(N) preparation and O(1) evaluation in N. It is a different loss from the squared pair-distance discrepancy, but is a necessary strong estimator/computational control. Moment caching alone is therefore not a unique asymptotic improvement. A tractable, useful global lower bound or search for the specific quartic remains a separate hypothesis whose prior art, numerical behavior and task benefit must be established; the algebra here does not establish novelty.
