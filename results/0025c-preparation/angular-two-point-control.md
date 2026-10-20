# Prospective control correction: angular-only affine RS can use two points

Source/algebra review only, before the 0025c cohort runs. No generated observations, numerical solver or experiment was executed for this note. This is a control requirement, not a new-method priority claim.

With measured middle rotation and row rotations known, the angular-only model is

\[
t=(a_2d_{2i}+b_2)\bar r_{2i}-(d_{1i}+b_1)R\bar r_{1i}.
\]

For two matches, subtract the first equation from the second. Writing the differences as Δ gives

\[
\underbrace{[-R\Delta\bar r_1,\;\Delta(d_2\bar r_2),\;\Delta\bar r_2]}_{M\in\mathbb R^{3\times3}}
\begin{bmatrix}b_1\\a_2\\b_2\end{bmatrix}
=R\Delta(d_1\bar r_1).
\]

If `det(M) != 0`, the three affine parameters and then `t` are unique in the declared length gauge. This condition is attainable, not just a degrees-of-freedom count. One exact physically consistent GS special case has `R=I`, `t=(1,0,0)`, source rays `(0,0,1)` and `(0,1,1)` at depths 1 and 2, target rays `(1,0,1)` and `(1/2,1,1)`, and `d1=d2=z`, zero shifts, and target scale 1. Then the columns of M are

\[
(0,-1,0)^T,\quad(0,2,1)^T,\quad(-1/2,1,0)^T,
\]

and its determinant is exactly `1/2`. Both point depths are positive. Nonzero determinant is preserved under sufficiently small physically consistent perturbations, including known row rotations; no promise is made for every motion, depth distribution or two-point sample. Near-degeneracy and measured-rotation errors still require the same prospective qualification as any other initializer.

Therefore the Stage A `D4 >= 9/70` rule remains only a necessary availability screen. Its hypothetical four-versus-eight sampling calculation does not establish an advantage over the strongest same-input angular-only alternative. At 50% hypothetical independent inliers, the two-point pure-inlier probability is four times the four-point probability. Whether its omitted readout translation creates enough bias to overcome that sampling advantage is an actual question, not something settled by the sample counts.

**Pre-execution root amendment:** Stage A now includes every one of the 28 two-subsets as A2, retaining the same full-eight-point positivity rule. A8 remains its nonminimal bias/accuracy control. The unchanged six D4 coverage gates are supplemented by a necessary optimistic condition: at least one of the three fixed sensor-noise groups G/S, N/S, C/S must have `p_D4 > 4*p_A2` at hypothetical inlier fraction 0.5. With 16 realizations, the respective denominators are 1120 and 448, so the strict comparison is exactly `D4_useful_count > 10*A2_useful_count`. All three comparisons are reported; no best subset is selected. The cohort has 132 records × 101 solves = 13,332 attempts. This was decided before any cohort outcome.

The rule deliberately ignores A2's cheaper solve cost and real outlier dependence. It therefore gives D4 an optimistic necessary screen, not a robust-efficiency claim. A separately admitted robust Stage B must retain A2, the same measured rotation, raw-depth coefficients, sign/positivity contract, complete-pool scorer and charged cost. No A3 arm is added at this stage. Passing the current necessary screen cannot itself justify a pose-improvement claim or an image campaign.
