# 0011: oblique evidence for coordinate pairing — development pilot

Status: prospective; execution requires the reviewed source/config freeze on
main. This is a different candidate from the rejected full-covariance proposal
in `docs/literature/0011-covariance-prior-review.md`. No result or novelty claim
is implied. Substantial AI assistance is disclosed in `docs/CONTRIBUTIONS.md`.

## Question and mechanism

Can a second, oblique rasterization of a causal RGB history improve the pairing
of a frozen tracker's separate x/y coordinate distributions, beyond ordinary
fusion using exactly the same outputs? It supplies another network response,
not another camera, new visible surfaces, or an independent observation.

Let px, py be reference-view full-head probabilities. For reference bin-center
z=(x+.5,y+.5), evaluate auxiliary marginal probabilities at Taux inv(Tref) z.
Their product is G(z). With Q0(y,x)=py(y)px(x), compute

    Q* = argmin_Q < -log G, Q > + KL(Q || Q0)
         subject to row sums py and column sums px.

This is standard entropy-regularized transport / iterative proportional fitting,
not a new optimizer. The proposed use of oblique evidence for fixed-marginal
pairing is a hypothesis whose priority remains conditional. The marginals are
floor-smoothed native full heads, not the native truncated output distribution.

Axis-preserving transforms give separable cost and hence Q*=Q0. Oblique cost can
change the coupling. Fixed marginals force the same global coordinate mean;
the only possible gain here is in the predeclared mode-local readout. This does
not reconstruct a unique true posterior. Wrong marginals can prevent recovery.
See `docs/literature/0011-oblique-coupling-prior.md` for closest work and limits.

## New allocation, inputs and disclosure

Allocate **g-036 (video36), then g-002 (video2)** as development only in this new
study. They are the first two reserve groups in the immutable 0002 identity
manifest (SHA256 b5415fd3eab77806ec71a974e2d875734f052ed3e44087a6e5b0a6d543cbbba3).
Do not edit its historical reserve assignments. The other 17 reserve groups
remain unallocated here. Old development/calibration/evaluation outcomes cannot
be used to tune this method. Current bounded cache/marker inspection found no
inference record for these two groups; this is not proof across lost environments.
RGB-Stacking is simulated imagery with shared objects/camera; episode/scene and
checkpoint-training independence are unknown. This is not independent-scene or
real-robot confirmation.

Use only cached official archive 187291581 bytes, SHA256
ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc. No downloads.
Restricted NumPy unpickling in a separate bounded preparation process necessarily
materializes the 50-example container, including annotation arrays. Disclose this;
do not claim that future GT bytes never entered memory. Verify selected arrays
against their existing identity hashes. No other group is analyzed or inferred.

Select ascending first eight point IDs visible at frame0; require eight or stop.
Only frame0 visibility/coordinates determine query selection. Use fixed frame IDs
0,16,32,...,240, 16 observations each. This is deliberately sparse observation,
not ordinary-frame-rate tracking. Causal model steps are 0..15; no intervening or
future frames enter the model. Preparation separates predictor NPZ (RGB, initial
queries, IDs) from GT NPZ. Prediction/fusion processes must not open GT files.

## Fixed rendering, model and decoding

Frozen native TAPNext++ 256px checkpoint and the five cached official source files
are those verified in 0005, SHA/revisions in the machine config. CPU float32,
eval/inference mode, no compilation/autocast/adaptation/support queries. Eight
queries in each view, four independent recurrent states, unchanged across time.
Fusion is readout only and never feeds state or future frames.

Canvas 256x256, centers .5..255.5, center of rotation (128,128), x right/y down.
Positive angle uses R=[[cos,-sin],[sin,cos]]. Views in fixed order:
original=(0deg,1), reference=(0deg,.7), oblique=(45deg,.7), axis_control=(90deg,.7).
The .7 scale fits the whole rotated original support. Inverse warp RGB with four
zero-padded bilinear taps, float64 accumulation, float32 conversion, np.rint
ties-to-even, uint8 clip. This is not antialiased area downsampling. Query affine
is exact float64 then float32, no coordinate clipping. Each view observes the
same original RGB/queries; original full resolution is a separate strong control.

Core logits are [y256,x256]. Use softmax(.5*logits), floor each entry at 1e-12,
renormalize. Sample auxiliary probabilities (not logs) linearly at mapped bin
centers; outside [.5,255.5] each marginal likelihood equals 1e-12; floor inside
as well. G is a compatibility field, not a normalized density or calibrated
likelihood. Relative rotation is 45deg with no scale. Reference padding bins
are retained; outside auxiliary support is not edge-clamped.

Float64 log-domain Sinkhorn: entropy coefficient1, zero initial log-column
scaling, row then column updates, maximum1000 iterations, maximum absolute
row/column residual <=1e-9. Failure/nonfinite output stops the candidate; no
fallback, relaxed tolerance, or selective missing-case exclusion. Joint readout:
joint mode, exact ties lexicographic (x,y), then conditional mean in inclusive
axis-aligned +/-20-bin square, using centers+.5. Apply the identical readout to
independent/product/mixture/coupled grids. Inverse-map coordinates to original
pixels without clipping, including predictions in padding/outside image.

## Predeclared comparisons and execution order

Save all raw logits, native coordinates and visibility logits once. Compare:

1. Native original, reference, oblique and axis-control coordinates separately.
2. Independent reference joint with the declared joint decoder (parity control).
3. Mean of inverse-warped native reference/oblique coordinates.
4. Choose reference or oblique by lower sum of full-head x/y entropies; exact tie
   chooses reference. No label-informed selector.
5. Product Q0*normalize(G45), normalized; equal additive mixture of Q0 and
   normalize(G45). Both use the same two views as the candidate.
6. Oblique fixed-marginal coupling (candidate).
7. Axis-control fixed-marginal coupling (algebraic negative control).

One model load and 64 forwards per group, 128 total. Original/90deg forwards are
experimental controls; deployed candidate would need reference+45deg (two calls
per observation). Report actual four-view experiment cost separately.

First prepare the two groups once. Then predict each group once under separate
900-second hard supervisors. Seal raw output files. Then derive all eleven
methods for both groups from saved outputs, with no model calls, under one
1800-second supervisor. Seal all decoded predictions and diagnostics before one
scoring process opens evaluation GT. Missing/failed groups are not replaced.
Partial outputs and markers survive failures; no automatic scientific retries.

## Endpoints and continuation rule

Primary eligible rows: selected original frames16..240 with GT non-occluded,
regardless of predicted visibility. At least one eligible row per group is
required. All methods retain 100% of these rows, including bad/outside predictions.
Quality is mean of hit indicators distance<tau for tau={1,2,4,8,16} original
pixels. Average within each group, then equally across two groups. This is
coordinate accuracy, **not** TAP-Vid average Jaccard or an occlusion metric.
Record per-threshold counts, EPE, median EPE, all-point and occluded EPE, prediction
visibility, and paired corrections/harms at4px for each comparator. Exclude
initial query rows from quality but retain their raw outputs and diagnostics.

Continue only if all numerical checks pass AND candidate aggregate quality is
at least **0.05 absolute (5 percentage points)** above the strongest aggregate
predeclared non-candidate control AND candidate quality exceeds every control
in each of the two groups. Candidate macro EPE must not exceed the EPE of the
strongest-quality comparator (ties resolved by fixed method order). These are
developmental screening targets, not predicted results. A weaker-baseline win,
positive subgroup, or non-significant narrative cannot activate continuation.

Numerical gates: every coupling converges; global-mean difference from Q0 <=5e-5
pixels in reference coordinates; 90deg coupled-grid max absolute difference
from Q0 <=1e-7. Report native/reference-independent coordinate discrepancies;
require <=1e-3 pixels in original coordinates as a decoder-parity gate. Exact
mode ties may be numerically unstable; report them, do not silently change the
readout. Report maximum residuals/iterations and distribution entropy/change.

Two dependent simulated groups cannot support population confidence intervals
or a manuscript. Report both group outcomes rather than treating points/frames
as independent replications. A positive screen would require a separate frozen
larger independent-source evaluation, stronger comparisons, mechanism ablations,
cost analysis and replication before a manuscript. Negative/low-headroom results
stop this exact candidate/configuration without tuning on these two groups.

## Resource, provenance and publication

Existing no-cost Work cloud CPU only; four threads/affinity, address space <=8GiB,
preparation <=300s, inference <=900s/group, fusion/evaluation <=1800s each,
immediate process-group kill at the hard deadline. Credential-free child environment, no network/install or
shell subprocesses in scientific children. Log commands, UTC times, durations,
exit status, peak RSS, package/source/config/data hashes, per-forward counts,
failure markers and sealed output hashes. Original cache/source/history remain
unchanged. Downloads this study:0; prior acquisition accounting unchanged.

Only original code, brief attributed notes, and small derived predictions,
diagnostics and receipts may be published; no raw RGB, checkpoint, pickle, GT
NPZ or paper rehosting. Independent review reconstructs scores and invariants
from saved outputs. Publish exact commit/CI evidence. Neither passing software
fixtures nor a positive two-group pilot is a CVPR-level result.
