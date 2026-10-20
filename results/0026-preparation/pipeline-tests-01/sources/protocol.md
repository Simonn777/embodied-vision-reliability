# 0026: moment calibration as a nonminimal initialization

Prospective development protocol, 2026-10-19. No target 0026 scientific arrays,
poses or outcomes have been computed at drafting. The 80 historical pairs and
their historical outcomes are already development forever. This is not a final
test or a manuscript admission.

## Question and prior work

Can a small convex lifted distance-calibration problem supply a useful basin
for affine-depth relative-pose refinement on an existing robust support?
Pose elimination through interpoint distances and polynomial lifting already
appear in RePoseD and MADPose. Moment compression alone, variable projection,
GNC, convex relaxation and certification are not new contributions. Proper
Procrustes F has the same O(N) preparation and constant-size evaluations as G.
The concrete unverified mechanism is a pose-independent nonminimal calibration
seed, followed by physical G polishing and the unchanged native TC optimizer.
The tested relaxation supplies numerical diagnostics, never a rigorous certificate.

## Inputs and hard boundaries

Use all 80 pairs in configs/0017-development-manifest.json, in its exact order,
16 per existing scene. Reuse cached author RoMa/MoGe HDF5s, named calibration,
x1/x2 and depth1/depth2 fields only. Rehash opaque files and compare converted
inputs to the old sealed records before fitting. Read no HDF5 R/t or new data.
Verify the full 0017 and 0019 prediction seals with their original source
verifiers. The sole support is each saved 0019 depth_affine pre-final epipolar
mask; w is exactly its 0/1 indicator. There is no learned/IRLS/GT-dependent
mask, point deletion, pair exclusion or depth inversion. Moments use axial
z-one rays from authored PINHOLE parameters. Supported minima define physical
positivity; unsupported points remain in full-array final scoring/diagnostics.

## Frozen proposals and controls

The incumbent is the saved final 0019 TC depth_affine geometry, with its raw
translation and calibration. First continue it once with the direct native TC
refiner on the fixed support. No RANSAC entry point may substitute for this call.

Build the exact H6x6 and centered C12x12 moments once. The three calibration
starts are: neutral (b,a,c)=(0,1,0); incumbent (shift1,scale,scale*shift2); and
the physical rounding of the one fixed lifted relaxation. A nonpositive or
unavailable start stays unavailable; do not clip, replace, or retry it.

For each of these starts, optimize G and F separately using the same positive
parameterization b=exp(u)-m1, a=exp(v), c=exp(w)-a*m2. Fixed L-BFGS-B limits are
maxiter=80/maxfun=400, ftol1e-12, gtol1e-7, no bounds/jitter. SciPy checks
maxfun after a line search, so400 is an option rather than a hard evaluation
cap; save and charge actual counts. The outer20s pair bound remains hard.
Only completed, finite optimizer outputs advance; failure states and last
finite evaluations remain saved. G and F raw signed costs remain unmodified.
Use proper Procrustes to recover R,t, check proper rotation/nonzero translation
and positive supported input depths, then exactly one common TC refinement.
This is 1 continued-TC plus at most 6 proposal refinements per pair, 560 total.
The conic solve uses its frozen 9-value barrier schedule, 40 Newton iterations
per value and at most 60 line-search halvings; only status=completed plus finite
positive physical recovery yields its seed. Loose rank slacks and negative
numerical gaps remain descriptive and do not rank or prune a pose.

If fewer than four points belong to the saved support, all seven new slots are
unavailable and the original incumbent remains in every union. A saved final
native incumbent can have nonpositive supported corrected depths; in that case
its continuation and incumbent-start proposals are unavailable, not an ABI or
input-integrity error. Preserve its original pose in the fallback unions.

The common direct native call uses the existing pinned PoseLib archive,
TRUNCATED_CAUCHY, 100 LM iterations, loss_scale=.25*2*(1/f1+1/f2),
scale_reproj=1/64, weight_sampson=1, estimate shifts, and unchanged other defaults.
Map b,a,c to native shift1=b, scale=a, shift2=c/a. Native post-refinement
negative-depth/omitted-direction behavior is retained and reported for all arms;
it is not silently fixed. Finite proper R and nonzero t are needed for scoring.

## Label-free selection and failure accounting

Score every valid post-refinement pose on ALL original matches using the exact
0017 raw-pixel 2px capped Sampson plus cheirality score. This is the common
selector, not the numerical distance objective or the native LM cost. Ties
favor the incumbent, then continuation, then the documented proposal order.

Primary method: best score among incumbent, continued-TC, G_lift.
Controls: incumbent; incumbent+continued-TC; incumbent+continued-TC+G_neutral+
G_incumbent; the corresponding F local union; incumbent+continued-TC+F_lift;
and the stronger union of all three F starts with incumbent/continuation.
Report individual proposals and G-all union descriptively, without promoting a
post-outcome winner. Also report unchanged historical E/TC-scale references.
Those single-method references also enter the best-control improvement gate;
no reference receives per-pair GT-dependent oracle selection.
Failed proposal attempts remain charged and fall back only through these fixed
unions; no pair is removed. Moment/relaxation preparation errors remain explicit
and make dependent proposals unavailable. Binding/input/native ABI faults stop
the cohort rather than converting a broken implementation into a method failure.

## Timing, execution and sealing

Record preparation, relaxation, each local solve/recovery/native refinement/
full scoring, plus total input and process time. Every union pays for every
attempt in it, including failures, plus its prerequisites. The old TC base time
is historical and labelled separately. A Python/SciPy prototype and cached
frontend cannot establish a full-pipeline speed claim. Report estimator time
without hiding this limitation; no manuscript can pass the later matched-time
gate with historical timings alone. Rotate the seven proposal execution slots
by pair index; do not adapt order to outcomes. Use no more than four CPU threads,
8GiB sampled RSS and 1800s per operation, with 20s per pair. A resource timeout
is retained; integrity and unexpected implementation exceptions stop evaluation.

Freeze source/protocol in Git and require exact-commit CI before target numerical
inputs. Run once, save all attempt records and complete the prediction seal and
clean outer receipt before a separate evaluator reads the previously exposed
0017 saved labels. No optimizer is callable from the evaluator. No outcome
retuning, extra restarts, threshold or subset changes in this experiment.

## Decision rule

Primary metric is directed max(rotation,translation) error with invalid pose
penalty180 degrees, and exact empirical AUC@tau=mean(max(0,1-error/tau)).
Recompute every comparison on the same metric; historical trapezoidal AUC may
be additionally reported and must not be mixed. Report all five scene metrics,
all-pair/raw-proposal availability, selection counts, losses and timings.

Continue this fixed method only if primary G_lift union exceeds the BEST frozen
control by at least2 absolute AUC@5 percentage points, is positive against that
control in at least3/5 scenes, no scene loses more than1 AUC@5 point, and pooled
AUC@20 loss is at most0.5 point. Ties do not count as positive. Failure closes
this fixed variant on this development regime; no tuning on its same outcomes.
A pass permits separately frozen held-out/time-matched validation, ablations and
replication, not a paper. The main goal still requires >=2 AUC@5 points at
matched complete-pipeline time on two independent held-out sources with positive
scene-paired95% intervals and <=0.5pp AUC@20 loss. No goal completion here.

Choose that one BEST control by pooled AUC@5, resolving ties by this fixed order:
incumbent, continued_tc, G_local, F_local, F_lift, F_all, historical_e_norm,
historical_e_ts, historical_TC_scale. Use the same selected control in every
scene and in the AUC@20 clause; do not choose a different oracle per endpoint.
