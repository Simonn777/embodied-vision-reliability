# Rolling shutter + affine axial depth: source-only identifiability screen

Prepared without data, model calls, numerical rank experiments or fitting. This is a mathematical admission screen, not a proposed contribution or a minimal-solver result.

**Conclusion.** The unrestricted formulation is not uniformly identifiable, even with exact positive row-camera depths and arbitrarily many correspondences. There are exact continuous counterexamples affecting the twists, and sometimes the middle-row pose or the depth offsets. These counterexamples do **not** prove generic impossibility. A 21-dimensional gauge-fixed Jacobian rank test is needed before claiming generic local identifiability; seven matches is only a counting lower bound. The more immediate practical gap is that a GS-trained network has not supplied the assumed row-camera axial-depth observation model. Known angular velocities reduce the problem but do not remove the translational/offset counterexample.

## 1. Exact coordinate and observation contract

Use world-to-camera transforms. Let the first middle-row camera define world coordinates, `T1(0)=I`, and let `T2(0)=T=(R,t)`. Define

\[
G_k(s)=\exp(s\widehat\xi_k),\quad T_k(s)=G_k(s)T_k(0),\quad
\xi_k=(\omega_k,v_k).
\]

Here `s` is the **known signed exposure time relative to that camera's middle row**; `v` is the translational component of this left SE(3) twist, not automatically the camera-center velocity. For calibrated homogeneous rays `rki=Kk^-1 uki`, normalized to last component 1, assume actual row-camera axial depth

\[
z_{ki}=a_k d_{ki}+b_k>0,\qquad a_k>0.
\]

The exact three-component observation equation is

\[
\begin{bmatrix}z_{2i}r_{2i}\\1\end{bmatrix}
=G_2(s_{2i})T G_1(s_{1i})^{-1}
\begin{bmatrix}z_{1i}r_{1i}\\1\end{bmatrix}. \tag{A}
\]

There are 22 scalar unknowns: 6 pose, 12 twists, 4 depth parameters. The global length gauge is

\[
(t,v_1,v_2,a_1,b_1,a_2,b_2)\mapsto
\lambda(t,v_1,v_2,a_1,b_1,a_2,b_2),\quad\lambda>0,
\]

with `R,omega1,omega2` unchanged. Known seconds fix the velocity's time unit, **not the scene's length unit**. Fix `a1=1` or one positive baseline length, not both without independent metric information. Thus there are 21 gauge-fixed unknowns and `N>=7` is necessary by count only. Multiple isolated solutions and critical configurations remain possible.

Barath's [*Rolling Shutter Relative Pose Estimation Made Practical*, §3, Eqs. 1–3](https://arxiv.org/html/2606.26863v1#S3) uses a first-order row model with 5 pose and 12 motion degrees of freedom after fixing translation scale. It explicitly notes velocity–translation coupling. That supports retaining these nuisances; it is not a proof about this exact exponential model with affine-depth observations. Its §4.2 solver additionally approximates the pose-dependent elimination subspace. The exact SE(3) convention here must not silently be replaced by a different first-order translation convention.

## 2. Exact common-motion ambiguity when matched row times are affine

Suppose all matches obey `s2_i=A*s1_i+B` and a constant row-relative transform `T*` maps the two row-camera point clouds. For **any sufficiently small six-dimensional twist** `c`, define

\[
\xi_2=c,\quad \xi_1=A\operatorname{Ad}_{T_*^{-1}}c,
\quad T=\exp(-B\widehat c)T_*.
\]

Then, identically for every match,

\[
\exp(s_2\widehat c)\,\exp(-B\widehat c)T_*
\exp[-s_1 A\widehat{\operatorname{Ad}_{T_*^{-1}}c}]=T_*.
\]

All observed rays and both row depths are unchanged. The reconstructed world points change to `G1(s1_i)^-1(z1_i*r1_i)`, which is still a static finite point set. This is not an assumed moving scene.

- With `B=0`, there is a six-dimensional twist ambiguity while the middle-row pose remains `T*`. A rectified horizontal stereo configuration has equal rows for a nonplanar scene and realizes this case.
- With `B!=0`, the middle-row pose also varies. Different calibrated principal-point/readout origins can produce such an affine time correspondence even for rectified geometry; the claim does not require every ordinary same-camera pair to have this configuration.
- Near-affine row-time correspondence is a conditioning warning, not an exact gauge assertion for arbitrary data. Enforcing independent per-frame constant twists does not constrain motion in the time gap between frames.

## 3. Exact depth-offset / row-translation ambiguity

Take zero angular velocities, `R=I`, normalized rays `r_k=(x_k,y_k,1)`, and known row times `s_k=y_k` (other known units only rescale velocities). The equation becomes

\[
z_2r_2=z_1r_1+t+y_2v_2-y_1v_1.
\]

If `x2=x1`, then for every sufficiently small scalar `delta`,

\[
b_1'=b_1+\delta,\ b_2'=b_2+\delta,
\quad v_1'=v_1+\delta e_y,\ v_2'=v_2+\delta e_y
\]

leaves the equation exactly unchanged: the additional left-minus-right term is `delta*(r2-r1)=delta*(y2-y1)*e_y`. Positive depths stay positive for a small open interval of `delta`. This survives the scale choice `a1=1` and is **not** the global scale gauge.

A nonplanar scene under a vertical middle-row translation realizes `x2=x1` with nonzero vertical flow and varying depths. Thus this is not merely the constant-depth plane case. Importantly, **known angular velocities equal to zero do not remove it**. For `s_k=gamma_k*y_k+eta_k`, use `delta*v_k=delta*e_y/gamma_k` and adjust `delta*t=-delta*(eta2/gamma2-eta1/gamma1)*e_y`.

Additional elementary failures include a narrow/single-row sample (middle pose and row pose confound), constant predicted depth in an image (`a*d+b` cannot separate a and b), collinear ray configurations, and vanishing translation when a translation direction is requested.

## 4. What a valid local-identifiability check would test

At zero row twists, set `p1=z1*r1`, `p2=R*p1+t`, and residual `F=p2_observed-R*p1-t`. For left rotation perturbation of `R` and additive translation, its analytic Jacobian blocks, in order
`[deltaR,deltat,omega1,v1,omega2,v2,a1,b1,a2,b2]`, are

\[
J_i=\left[
[Rp_1]_\times,\ -I,\ -s_1R[p_1]_\times,\ s_1R,
\ s_2[p_2]_\times,\ -s_2I,
\ -d_1Rr_1,\ -Rr_1,\ d_2r_2,\ r_2
\right]. \tag{B}
\]

Remove one length-gauge column/coordinate and test rank 21 of the stacked Jacobian; away from zero twists use the exact SE(3) exponential derivative. Full column rank is a local sufficient condition at a regular solution, not global uniqueness. A rank witness must use a physically consistent static scene with the **implicit row-exposure equation solved**, not independently invented row times unrelated to projected y. The exact families above should remain analytic null directions. No rank computation was run here, and no generic-rank claim is made.

For the middle pose alone, inspect whether the Jacobian nullspace changes rotation or translation direction after removing length scale; demanding all velocity parameters be observable can be stronger than the task requires. Counterexample 2 still prevents recovering the proposed depth offsets even if its middle pose remains fixed.

## 5. GS-depth versus instantaneous row-camera depth is a different model

Write `G(s)=[Q(s),u(s)]`. A row-camera point `z_row*r` has middle-camera coordinates `Q(s)^T*(z_row*r-u(s))`, so

\[
z_{mid}=h(s,r)z_{row}-e_3^TQ(s)^Tu(s),\qquad
h(s,r)=e_3^TQ(s)^Tr.
\]

Even **if** a network gave middle-camera axial depth `a*d+b` associated with the same physical point,

\[
z_{row}=\frac{a d+b+e_3^TQ(s)^Tu(s)}{h(s,r)},
\]

which is generally not a single affine calibration of `d`. If the predictor instead renders a GS depth map at an anchor camera, its pixels also require reprojection/data association; equal pixel indices do not establish the same point. A network trained on GS imagery need not choose any coherent physical anchor on an RS image.

The author [*Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion* project abstract](https://focus.ece.ufl.edu/research/2026/rolling-shutter-anchor-point/) explicitly studies temporal/geometric anchoring and reports its importance. Only the author abstract was checked here; it does not establish the affine row-depth model for MoGe or validate a particular conversion. This distinction is a prerequisite to a useful experiment, not an extra gauge that can automatically be repaired with two scalar shifts.

## 6. Smaller variants and their limits

**Known angular velocities from synchronized, calibrated IMU.** With `Qk(s)` known, let `u_k(s)=J_SO3(s*omega_k)*s*v_k`. Define `rbar=Q^T*r` and `B=Q^T*J_SO3(s*omega)*s`. Then (A) is

\[
z_2\bar r_2-B_2v_2=R(z_1\bar r_1-B_1v_1)+t.
\]

There are 15 unknowns after the length gauge, hence five matches is only a counting lower bound. For fixed R, `t,v1,v2,a1,b1,a2,b2` enter linearly; this gives a plausible small conditional solver structure. It still has counterexample 3, needs measured camera–IMU rotation/time/readout conventions, and supplies **new measurements** unavailable in the current assets. It must not be compared as an equal-input RGB-only improvement or implemented with velocity GT.

**Known middle R as well as both angular rates.** Fixing `a1=1` leaves the 12 linear unknowns `[t(3),v1(3),v2(3),b1,a2,b2]`. For each match the matrix row block is
`[-I, R*B1, -B2, -R*rbar1, d2*rbar2, rbar2]`, with right-hand side `R*d1*rbar1`. Four matches is again only a counting lower bound. In fact, for `R=I`, zero angular rates and four matches, the matrix always has rank at most 11: eliminate the translation columns with a left-null vector `m` of `[1,s1,s2]`. Since each time is affine in its normalized image y, `m^T*r1` and `m^T*r2` have only x components. Thus the two offset columns after elimination are collinear, and the remaining three columns `(b1,a2,b2)` have rank at most two. This argument does not require `x2=x1`. Five points supply two such left-null vectors and remove this particular rank ceiling, but do not guarantee full rank. A nonidentity R with `R*e_x` not parallel to `e_x` also removes this particular collinearity argument, not every possible degeneracy. The planned manufactured check will test these separately, without calling a numerical witness a generic theorem. Dewarped rays must retain their third component; normalizing it without transforming every depth/offset coefficient changes the model.

**Known angular velocities plus negligible readout translation.** Setting `v1=v2=0` permits known rotation dewarping of the rays and returns the affine-depth rigid-alignment form. This is a narrower physical approximation; it does not follow from having a gyroscope. [Ding et al., *Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation*, §2 Eqs. 2–4 and §3.1](https://arxiv.org/html/2501.07742v1#S2) already solve the calibrated scale/shift alignment problem. Applying known row rotations to the known vectors is not, by itself, a new estimator principle.

**One twist over both readouts and the inter-frame gap.** A genuinely known middle-time separation and a constant trajectory over that entire gap can tie `T` and the two twists, removing the independent common-motion freedom. This is a stronger motion assumption than two separately constant readouts and overlaps existing RS differential-SfM modeling; neither timestamp data nor validity of the approximation should be invented.

**Admission recommendation.** Do not yet fund a full unknown-twist/affine-depth solver experiment. The ideal row-depth formulation is a mathematically meaningful conditional research problem, but no generic-rank witness or usable learned-depth contract has been established. The only defensible next mathematical qualification would fix the physical row-depth meaning, retain the exact null counterexamples, and demonstrate task-relevant Jacobian rank on a predeclared physically consistent configuration. That would establish algebraic feasibility only. A subsequent real method must beat strong current RS geometry and appropriately input-matched gyro/affine controls; merely estimating more nuisance variables, using angular GT, or treating GS network output as row depth does not establish a contribution.
