# 0025: rolling-shutter depth geometry, mathematical qualification

**Status: algebraically feasible conditional candidate; no accuracy experiment
or manuscript admitted.** This is a distinct rolling-shutter image-formation
problem. It does not reopen the failed 0024 virtual-rotation offset method.

## Concrete hypothesis

Given calibrated row exposure times and measured camera rotations, two learned
depth maps might supply enough independent information to estimate middle-frame
translation and both readout translation velocities from fewer correspondences
than depth-free rolling-shutter geometry. The depth maps have independent unknown
affine scale/offset. This is an initialization hypothesis, not an established
gain; inaccurate depth and gyro measurements can erase its potential advantage.

The conditional model knows middle rotation R and both angular velocities. It
does **not** estimate the full relative rotation from four points. With length
gauge a1=1, the unknowns are [t(3), v1(3), v2(3), b1, a2, b2]. Exact row rotations
give unnormalized rays rbar=Q^T*r and B=Q^T*J(s*omega)*s. Each match contributes

    [-I, R*B1, -B2, -R*rbar1, d2*rbar2, rbar2] beta = d1*R*rbar1.

The [derivation and counterexamples](0025-rs-depth-mathematics.md) retain axial
depth, time units, the scale gauge, and all translational readout terms. Depth
semantics on actual rolling-shutter input remain an unverified assumption.

## What has now been established

The first fixed manufactured invocation and an independent saved-output audit
completed successfully. The full unknown-twist model has numerical rank 21 in
the declared seven/eight-point generic fixtures. The given-rotation branch has
rank 12 in its generic four-point fixture. Known degeneracies remain, including
rank 11 with identity R and four points. Five points resolve that particular
fixture, but not every critical motion. Numerical rank is not a noise guarantee.

After inspecting those numerical results, a **separately declared exact-rational
extension** kept the first four points and other constants, replacing the
exponential R with a fixed rational Cayley R. Its physical observations, matrix,
nonzero determinant and unique complete solution were checked with exact
fractions. A root-run, algorithmically independent integer Bareiss cross-check agrees
exactly; this is not an independent-team replication. This
establishes the existence of a physically consistent noncritical four-point
configuration; it is not a search over fixtures or a replacement of the first
record. See the [complete evidence](../results/0025-identifiability/README.md).

The exact witness has zero generating twists during each separate readout, but
retains all unknown velocity columns and nonzero row times. The implicit row
equation has derivative 1 there. Continuity of its local solution and of the
nonzero determinant therefore supplies a neighborhood that includes nonzero
readout motions. This does not assert uniqueness over every physical branch.
Three points cannot determine all 12 unknowns by counting alone; this is not a
proof that translation direction itself requires four points.

The independent audit also separates task and nuisance ambiguity. In the
identity-R four-point fixture the nullspace changes translation direction. In
the vertical-flow four/five-point fixture the observed nullspace changes depth
offsets/velocities while leaving translation direction unchanged. The generic
four-point scaled matrix condition number is about 3815: invertibility alone
does not establish a practical robust estimator.

## Closest prior work and current limits

The [prior-work screen](literature/0025-rolling-shutter-prior.md) and
[depth-semantics review](literature/0025-rs-depth-semantics.md) distinguish the
following primary sources. The 2017 inertial RS formulation already has a
conditional linear translation/velocity block; the 2019 gyro five-point method
sets readout linear velocity to zero. Known global-shutter affine-depth solvers
already estimate depth shifts. Barath's 2026 RS solver with seven affine
correspondences is a strong contemporary comparison. Merely combining these
ingredients or obtaining full rank is not a demonstrated novel contribution.
The bounded full-text search has not established priority for this conditional
four-point system. A useful contribution would need a distinct, accurately
attributed estimator/critical-configuration analysis and measured benefit over
strong methods with the same information and complete-pipeline cost.

MoGe's projection step defines points on input pixel rays; it does not specify a
rolling-shutter temporal anchor. It neither proves instantaneous row-camera
depth nor proves middle-frame depth. A prospective physical test using
independent rendered depth, rather than another network as truth, is being
screened. A pure-rotation test can check depth semantics but cannot validate
translation robustness or establish a pose gain. No images, GT values, model
inference or real-data allocation occurred in this qualification.

The [dataset screen](literature/0025-rs-dataset-feasibility.md) and its
[follow-up](literature/0025-rs-dataset-followup.md) identify GrandTour HDR and a
possible TUM-RS grayscale domain. GrandTour's release timing chain and TUM-RS's
explicit data terms still need qualification. Grayscale could be declared
prospectively as a separate generalization domain; it must not be called RGB.
The later [complete size/mission screen](literature/0025-rs-dataset-budget.md)
found an official warning that GrandTour ETH-1/2/3 HDR uses arrival timestamps,
not hardware timestamps. These missions are therefore excluded from precise
row-time admission; the earlier ETH-1 bundle remains only a historical size
example. Smaller SPX-3/ICE-1 bundles make a budget possible, but have distinct
night/limited-motion or ice/GT-support caveats and are not allocated. No
two-source empirical evidence is implied.

## Next admission decision

Qualify independent depth format and bounded access, then freeze a small
physical falsifier with paired GS, same-point middle-depth and scale-only
controls. Predicted depth calibration may use only a predefined pixel training
partition for this diagnostic, with a fixed full holdout denominator and no
selection of the best temporal explanation after results. A pass would only
permit considering a separate robust-pose prototype with measured input IMU.
GT rotation/velocity cannot become estimator input. A failure closes the tested
observation approximation; no rescue tuning on those outputs.

The original final gate remains unchanged: at least +2 absolute AUC@5 points at
matched complete-pipeline time on two independent held-out sources, positive
paired scene-level 95% intervals, at most 0.5 point AUC@20 loss, ablations,
independent replication and a defensible novelty claim. No manuscript yet.
