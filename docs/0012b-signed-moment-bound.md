# Exploratory signed-moment lower bound for capped Sampson cost

Status: mathematical extension and synthetic numerical checks only. This does
not establish speedup, held-out performance, or publication novelty. The original
0012 implementation and development result remain frozen. The new implementation
is `experiments/0012-score-bounds/mean_variance_bounds.py`; its entry point is
`bound(cell, E, tau2) -> Interval`.

## Objective and failure mechanism

For a cell of `n` matched homogeneous points, define

\[
r_i=y_i^T E x_i,\qquad
d_i=\|(E x_i)_{1:2}\|^2+\|(E^T y_i)_{1:2}\|^2,\qquad
S=\sum_i\min(r_i^2/d_i,t).
\]

Here `t = tau2 > 0`; the shared reference convention assigns cost `t` whenever
`d_i = 0`, including `r_i = 0`. Suppose the existing coordinate bounds give
`d_i <= D`. Independent coordinate intervals discard correspondence correlations.
Their signed residual interval can therefore cross zero even when the actual
residuals have a large common sign and small variance. The existing squared-moment
bound can also be weakened by a large interval maximum for `r_i^2`.

## Finite-cell derivation

Choose an index uniformly from the cell and let `R` be its signed residual.
This constructs a finite discrete distribution; it assumes neither independent
matches nor a random sampling process. Set

\[
\mu=\frac1n\sum_i r_i,\quad m=|\mu|,\quad
v=\frac1n\sum_i r_i^2-\mu^2.
\]

If necessary negate `R`, so its mean is `m`. For any `0 < a < m`, the one-sided
Chebyshev--Cantelli inequality gives

\[
\Pr(|R|\ge a)\ \ge\ \Pr(R\ge a)
\ \ge\ \frac{(m-a)^2}{v+(m-a)^2}.
\]

Each member of that event contributes at least `min(a^2/D,t)`. Hence, for `D > 0`,

\[
\boxed{S\ \ge\ L_C(a)
=n\min\!\left(\frac{a^2}{D},t\right)
\frac{(m-a)^2}{v+(m-a)^2}.}
\]

The zero-denominator convention is compatible: its assigned cost is the cap,
which is at least the expression used above. If `D = 0`, every denominator is
zero and the shared convention makes the score `nt`; the existing bound handles
that case. The new bound is combined as `max(existing_lower, L_C)` and keeps the
existing upper bound.

## Existing storage and numerical enclosure

The cell already stores rows `A_i = vec(y_i x_i^T)` and `M = A^T A`.
With row-major flattening, homogeneous coordinates make `A_i[8] = 1`, so

\[
M_{:,8}=\sum_i A_i,\qquad
\mu=\frac{e^T M_{:,8}}n,\qquad
Q=\sum_i r_i^2=e^T M e.
\]

No additional stored first-moment statistic is needed. The exploratory float64
implementation encloses the signed mean and `Q` using the existing absolute
moment scale. From a mean interval it obtains `m_lower`, the distance of that
interval from zero, and computes a conservative upper estimate

\[
v_{upper}\ \ge\ Q_{upper}/n-m_{lower}^2,
\]

including rounding allowances for the square, division, and subtraction.
If the mean interval includes zero, the additional lower bound is zero. Raw
nonfinite moments, scales, or intermediate calculations also produce zero
additional lower bound before any clamp; this preserves the original enclosure.
The implementation does not claim a formal floating-point proof.

The fixed threshold is

\[
a=\min(m_{lower}/2,\sqrt{t}\sqrt{D_{upper}}).
\]

For a fixed `a`, the probability factor increases with `m` and decreases with
`v`; the cost factor decreases with `D`. Thus substituting `m_lower`, `v_upper`,
and `D_upper` preserves the lower-bound direction. An analytic maximizer exists:
for exact positive `m,v`, let `b` be the unique positive root of
`b^3 + 2vb - vm = 0`, then use `a = min(m-b, sqrt(tD))`. The prototype deliberately
uses the stated fixed threshold; it contains no fitted threshold or numerical
optimizer. Its floating-point operations receive the same conservative guards
as the frozen prototype. If the new lower estimate exceeds the existing upper
estimate, the wrapper returns a full-cap interval instead of intersecting an
inconsistent interval.

## What the synthetic checks demonstrate

For

\[
E=\begin{pmatrix}0&0&0\\0&0&-1\\0&1&0\end{pmatrix},\quad
x=((0,0),(0,100)),\quad y=((0,9),(0,111)),\quad t=1,
\]

the actual residuals are `(-9,-11)` and both denominators are `2`.
The exact capped score is `2`, with `m=10`, `v=1`. The coordinate row box permits
residuals from `-111` to `91`; the frozen lower bound is below `0.02`.
The new bound exceeds `1.97` and remains below `2`. This is an artificial example
of recovered correspondence information, not evidence of practical runtime gain.

Five tests cover that strict improvement; a symmetric bimodal limitation;
random and correlated cells at multiple coordinate and cap scales; the prior
overflow regression and zero denominators; cancellation, matrix sign/scale, and
empty cells. Containment assertions add no test-side tolerance. Tests operate
only on generated arrays and do not load development or held-out match files.

## Limitations and prior art

* A symmetric bimodal cell with residuals `(-10,10)` has mean zero. Cantelli adds
  nothing even though both residuals are large. More generally, first and second
  signed moments alone cannot force substantial capped cost at mean zero:
  probability `1-epsilon` at zero and `epsilon/2` at each of
  `+/-sigma/sqrt(epsilon)` keeps variance `sigma^2`, while capped expectation can
  tend to zero. Additional bounded-support information or subdivision is needed.
* Loose denominator bounds remain a limitation. Tree subdivision may already
  recover the same correlations, and evaluating another bound adds overhead.
  Synthetic improvement cannot establish reduced total residual work or faster
  wall-clock selection; both require the separately frozen comparative replay.
* First/second-moment probability inequalities are classical. See
  [Vandenberghe, Boyd, and Comanor, *Generalized Chebyshev Bounds via Semidefinite
  Programming* (SIAM Review, 2007)](https://web.stanford.edu/~boyd/papers/pdf/prob_bnds.pdf).
  The introduction and moment-bound discussion were checked; this extension does
  not claim a new probability inequality or general moment method.
* Conservative spatial pruning in RANSAC is also prior art:
  [*Space-Partitioning RANSAC* (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920715.pdf).
  Any future claim would have to distinguish efficient aggregate capped-cost
  bounds from existing conservative cell culling and demonstrate an actual gain.
  A search without a direct identical result is insufficient to establish novelty.
* [Variance Shadow Maps, coauthor's GPU Gems chapter](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps)
  already stores first/second moments and applies exactly the complementary
  Cantelli factor. [Moment Shadow Mapping](https://momentsingraphics.de/Media/I3D2015/MomentShadowMapping.pdf)
  extends compressed-moment lower bounds with four moments; only the author
  abstract and indexed sections were accessible during this check. Compressed
  moments plus a probability inequality is therefore not a new general idea.
* [Capel's BMVC2005 bail-out paper](https://bmva-archive.org.uk/bmvc/2005/papers/224/paper.pdf),
  section3, already describes deterministic accumulated nonnegative-cost
  termination. This directly motivates the partial-score baseline. No inspected
  source supplied this exact signed-Sampson aggregate bound, but the narrow
  application-level distinction remains unverified, with substantial novelty risk.

The scientific question remains whether this additional constant-size statistic
reduces total verification work enough to compensate for its own cost. The
original diagnostic's negligible reduction does not answer that question for
this extension, and it supplies no positive result for a paper.
