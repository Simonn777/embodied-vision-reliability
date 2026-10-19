# 0012d: lazy grid screening — prospective diagnostic

0012c failed: 84.168ms versus 13.052ms ordinary partial scoring. Its selected-call
setup alone was44.249ms; removing that setup would still fail. This distinct
computational design removes both recursive spatial construction and eager
all-model/all-cell bounds. It does not claim a new probability inequality or
admit pose estimation. All prior failures remain unchanged.

## Fixed design and controls

Use a sparse four-dimensional grid with fixed256 raw-pixel width on every
coordinate, floor-based integer keys and first-seen occupied-cell order. Keep
points within each cell in original input order. No sorting, tree, refinement,
learned order, score-derived model ordering or parameter search. Only cells
with at least64 points are screened. Other cells contribute zero to screening.
All geometry grouping and per-cell extrema costs are charged.

Compute model0's full exact score as incumbent, then process models1..63 in
their original stored order. For each challenger, visit eligible cells lazily.
Accumulate conservatively padded nonnegative lower costs and reject only if
the prefix lower bound exceeds the guarded incumbent; unseen cells contribute
zero. If screening cannot reject, fall back to ordinary block256 partial
scoring from the beginning of the same concatenated grid point order. Charge
all screening and fallback work; there is no pretend reuse of uncomputed costs.
Skip screening if the total eligible-cell cap cannot exceed the guarded
incumbent. During screening, fall back immediately when the accumulated lower
plus the remaining eligible-cell cap cannot exceed it. These exits only abandon
a certificate and perform exact partial work; they never reject a model.

Each cell first uses a geometric residual interval from the center and widths
of its4D box, and an upper Sampson denominator bound. The geometric lower cost
is n*min(r_min_squared / D_upper, tau_squared). Candidate mode4 additionally
uses the signed-moment Cantelli lower bound if geometry has not already reached
the cap. Build the45 signed symmetric moments only on the first actual request
for that eligible cell, caching them within this selection call across models.
Upper bounds on nine diagonal moments supply Cauchy bounds for absolute
quadratic and mean roundoff scales; no45-entry absolute-moment table is built.
Use a=min(m_lower/2, sqrt(tau_squared)*sqrt(D_upper)), as in0012b, with finite
checks on products instead of an overflow-prone direct square root of a product.
Combine lower bounds
with max, never sum two bounds for the same cell. Invalid/overflowing statistics
contribute zero additional lower bound. Guards remain conservative engineering,
not a formal certificate. See the separate derivation and independent review.

Five single-threaded modes share the exact residual and threshold convention:
0 full exact;1 original block256 partial with the frozen random permutation;
2 grid-order block256 partial without bounds or unused moments;
3 lazy geometric-only screening plus grid-order partial fallback;
4 lazy geometry plus Cantelli screening and the same fallback (primary).
Modes2 and3 charge only the preparation they require. No oracle input enters
selection. Track residuals, grid/eligible cells, geometric/Cantelli calls,
moment cells/points, pruned hypotheses, setup/scoring and outer-call time.

## Artificial cost admission before any real replay

Use N=10000, H=64, x=(0,s), y=(0,s+10), with s evenly spaced from0 to240.
All points occupy one256px cell. E(c) has only E[1,2]=-1, E[2,1]=1,
E[2,2]=c; model0 has c=9 and all63 challengers have c=0. With threshold1,
the incumbent point cost is0.5 and challenger cost1. The box residual interval
crosses zero while signed variance is zero. This deliberately favorable case
tests overhead feasibility, not real-data effectiveness, novelty or accuracy.

Validate every cell bound and all winner indices. Run nine fresh timed calls
per mode in cyclic order (repetition+offset)%5, with one separately disclosed
untimed call per mode for loading only. Shuffle seed20261020; charge measured
shuffle setup to mode1 on every repetition. Each call builds fresh needed
structures. Require mode4 median outer time <=2/3 of each of modes0,1,2,3 and
zero observed bound/winner violations. If it fails, stop this configuration
before reading any real pair; retain all outputs. Do not tune the fixture or
grid after the result. Independent numerical tests must also pass first.

## Conditional real development replay

Only if artificial admission passes and source/protocol are committed and
published: use exactly the same16 development pairs and64 saved hypotheses
from0012, with no resampling, calibration, poses, confidence values, labels,
images or held-out coordinate reads. Same frozen dataset/manifest/source
checks. This is repeated exploratory development, not independent confirmation.

Validate all1024 full scores against the saved originals with tolerance
1e-8+1e-12*abs(reference) and require exactly matching winner indices. Check
every eligible cell's geometric and combined lower bounds against its direct
score, and the summed bounds against each model's direct score; no added
containment tolerance. Any invalid output, observed violation or source
mismatch stops and retains the incomplete record.

Use three cyclic timed repetitions per mode/pair, order
(pair_index+repetition+offset)%5. Charge shuffle preparation to mode1; grid
construction and lazy summaries stay within each outer call. Report all calls,
per-pair medians, their aggregate sums, all counters and correct selections.
Require mode4 summed median total <=2/3 of each of modes0,1,2,3, all correct
winners and zero checked lower-bound violations before any pose integration.
No selected pair, best repeat, relaxed gate or same-configuration rescue.

## Resource and claim boundaries

System clang++ C++17 -O3, no fast-math or floating contraction, one native
thread, existing Python environment, no new packages/data/hardware or fees,
roughly8GiB limit and30-minute hard timeout per process. Build products remain
outside Git. Publish code, exact source/build hashes, complete small results,
failures and limitations. The main held-out two-source pose-quality and novelty
gates remain unchanged. Passing this diagnostic cannot itself authorize a paper.

## Pre-development serialization repair

The first artificial invocation at c2c48ed completed its loop but failed while
writing a NumPy boolean in JSON. Its output is truncated within the first timed
record; later timings cannot be recovered from disk. The early header's
`admission_passed: true` is not a valid gate result and cannot authorize replay.
Keep the incomplete output and error record unchanged, apart from disclosed
public workspace-path normalization in the traceback log. No real data was read.

One technical retry is admitted after a separate repair freeze and regression
test: convert agreement flags to built-in bool before serialization. Keep the
native binary, fixture, controls, repetitions, order, gates and all algorithmic
parameters unchanged. Save the retry under `synthetic-serialization-retry`;
never overwrite the first attempt. This is a disclosed repeated artificial
timing invocation caused by lost evidence, not the originally intended single
invocation. Only its complete, independently checked record may admit the real
development replay. No choice between the first and retry timings is possible
or permitted, and no further same-configuration timing retry is admitted here.
The unchanged original native library SHA256 is
`198c3a8c8841570c99c467323ea40f9abdddb24564d30922e29940138764e388`;
the retry requires this value as an explicit original-library hash input.
The real replay recomputes the artificial gate from all45 scheduled timing
entries and requires completed status, both128-check counts, zero violations,
all agreements, the original/retry/current binary link and source/protocol hashes.
