# 0009 Stage0: exact-tail headroom in synthetic translation ranking

2026-10-18. **Prospective development protocol; no outcome is asserted.** One
proposed score and one bounded screen. Execution requires the reviewed canonical
source/config/protocol freeze published on main and passing exact-head checks.
Pure-z remains rejected and is excluded. The existing artificial arithmetic
tests are not these results. No acquisition or manuscript is authorized here.

**This is an oracle-covered synthetic translation-ranking diagnostic.** Every
pool contains the exact true translation, plus 31 fixed alternatives. There is no
feature extractor, descriptor matching, RANSAC, DLT, model fitting, real imagery,
camera motion or learned model. Calling these matrices homographies does not make
this a general homography or end-to-end vision evaluation. A positive result means
only that this particular score merits a separately designed real-match check.

## Single question and investment boundary

Does the exact uniform-permutation upper tail choose the true transformation more
often than **each** of five cheap controls after common location deduplication,
when wrong correspondences have nonuniform endpoint marginals, without harming
uniform, compact-true or sparse-wrong-support controls? Does its exact solver cover
every declared candidate within the fixed CPU budget? Both are empirical questions;
neither follows from the classical counting formula. Sparse graphs can reduce the
score to support counting. Deduplication or a simple density score may remove all
headroom. This is an acceptable negative outcome, not grounds to alter the design.

No parameter search, support floor, fitted epsilon, z-score, approximate tail,
component pruning, fallback score or second candidate variant is permitted.

## Data, randomness and candidate pool

Coordinates are continuous float64 x,y in the square [0,512] x [0,512]. The matching
tolerance is **tau=3 pixels**, including equality. Every case has true H*=I. The
input observation is an ordered list of matched source/target coordinates; labels
are retained in an evaluator-only record. No scorer accepts true labels or the true
candidate index. The generator and scorer are separate functions.

There are four named families, each with seeds s=0,...,63: `uniform`,
`clustered_outlier`, `compact_true`, `sparse_isolated`. Thus 256 cases, without a
training/test split or inferential population claim. All are synthetic development.
Each starts with 128 base pairs; k=8 for s<32 and k=16 otherwise are true pairs.
The remaining 128-k are wrong pairs. No seed replacement or rejection of a whole
case based on scores, residual support, graph structure or GT quality is allowed.

For each family/seed/stream, derive a NumPy PCG64 seed from the first eight bytes,
little-endian, of SHA256 of the exact ASCII string

    evr0009|stage0|{family}|{s}|{stream}

Streams are `global-shift`, `inlier-xy`, `inlier-noise`, `outlier-source`,
`outlier-target`, `outlier-pairing`, `wrong-noise`, `clone-base`, `priority`, and
`candidate-order`. Use NumPy Generator(PCG64(seed)), not process-global RNG. Record
all ten integer seeds and the NumPy version. Draw one common two-dimensional shift
u~Uniform([-8,8]^2) and add it to all X and Y coordinates in the case. This keeps
true I and false translation (128,128) unchanged while varying alignment with the
fixed spatial grid; it is generator randomness, not a searched grid phase.

All 32 candidate matrices are translations H(t) with rows (1,0,tx), (0,1,ty),
(0,0,1). The canonical pool is:

1. Nine offsets (a,b), a,b in {-8,0,8}, Cartesian order with a outer and b inner.
2. Nine offsets (128+a,128+b), in the same Cartesian order.
3. These fourteen offsets, in this order:
   (-24,0), (24,0), (0,-24), (0,24), (-24,-24), (-24,24), (24,-24), (24,24),
   (32,0), (-32,0), (0,32), (0,-32), (32,32), (-32,-32).

The list is fixed before generating any coordinates. Permute its 32 canonical IDs
once using the `candidate-order` stream; every method uses this same order. Exact
score ties select the first candidate in this order, never the GT index, a larger
support count or a post-hoc secondary score. The true candidate therefore does not
receive a constant index-zero advantage. Save canonical and shuffled IDs.

There is no proposal/verification split because there are no fitted proposals:
none of these alternatives is obtained from the observed pairs. In particular,
including the false offset (128,128) is a declared oracle for an artificial
confounder, not evidence that a real proposer would discover it. Projection outside
the image is allowed when evaluating an alternative, with no H-dependent point
filtering. All input points remain in range and all candidates are invertible.

## Four generators, fixed before outcomes

A uniform disk of radius r is sampled using angle 2*pi*U and radius r*sqrt(V), with
independent uniforms. Construct base IDs in this fixed order: k true pairs, then
wrong pairs; in `sparse_isolated`, the first four wrong pairs are the explicit
sparse wrong relation. Apply the common shift after these constructions.

**True pairs.** In `uniform` and `clustered_outlier`, X~Uniform([32,480]^2).
In `compact_true` and `sparse_isolated`, X=(256,256)+uniform_disk(6). In all four,
Y=X+epsilon with independent epsilon~N(0,I_2), giving the same 1-pixel standard
deviation. If a Y after shift lies outside the image, redraw only that pair's
Gaussian noise, at most 64 attempts; do not clip. Exceeding this bound is an
explicit generator failure and invalidates completion, not an invitation to use
another seed. This rule also applies to the four sparse wrong pairs below.

**Uniform wrong pairs (`uniform` and `compact_true`).** Independently draw X and Y
from Uniform([32,480]^2), then add u. The compact-true family isolates possible
penalization of a concentrated correct explanation under otherwise uniform noise.

**Clustered accidental wrong pairs (`clustered_outlier`, the primary family).**
For even s use four centers (64,64), (160,64), (64,128), (160,128), with disk
radius 0.4 pixels. For odd s use the twelve Cartesian centers
{64,96,128,160} x {64,96,128}, with disk radius 2 pixels. For m=128-k wrong pairs,
source point j is center[j mod C] plus independent disk jitter. Independently make
m target points with the same balanced center-label counts, about center+(128,128)
with independent disk jitter. Uniformly permute this full target list using
`outlier-pairing` before assigning it to the source list. Then add u to both.
Source and target cluster membership must **not** be deliberately paired.

Each of the k-by-density settings has exactly 16 seeds. The ultra-dense four-cluster
condition should expose effects removed by shared location dedup; it is included
even if it leaves the controls essentially perfect. The twelve-cluster condition
permits distinct close points. Both are one declared family, not alternative
conditions from which a favorable result can be selected. Random coupling is
built into this synthetic generator and does not validate that assumption for
SIFT, mutual matches, or descriptor-filtered wrong correspondences. Moreover,
the common two-endpoint greedy deduplication itself depends on the observed
assignment; post-dedup selected matches (also containing true pairs) are not
claimed to obey uniform permutation even in this artificial screen.

**Sparse isolated wrong relation (`sparse_isolated`).** Four wrong source points are
(64,64), (320,64), (64,320), (320,320); each target equals source+(128,128) plus
independent N(0,I_2) from `wrong-noise`. The remaining 124-k wrong pairs are
independent uniform pairs as above. This tests whether a few isolated coherent
wrong pairs outrank concentrated true support. It is deliberately not a sample
from an independent-accident null. No periodicity or real physical-scene claim is
made. k=8 and k=16 are both retained even if one exposes an obvious limitation.

## One common deduplication, applied to all methods

Choose 16 distinct base IDs uniformly without replacement using `clone-base`, and
append two exact coordinate copies of each, yielding 160 raw pairs. Give all 160
raw pairs independent continuous random priorities from `priority`; sort ascending
by (priority, raw_id). Greedily retain a pair only if its source is farther than
**1 pixel** from every previously retained source AND its target is farther than
1 pixel from every retained target. Distance <=1 on either side rejects the pair.
Keep accepted pairs in this priority order, preserving their diagonal assignment.

This rule is independent of H, GT and true/wrong/clone labels. Priority is a
synthetic representative ordering, not a claimed descriptor confidence. Exact
copies are guaranteed to disappear; distinct nearby points can also disappear.
Record all accepted/rejected IDs and whether source, target, or both caused each
rejection. Never replenish the list or tune the tolerance to achieve a desired
count. The same nonempty list, of size n<=128, goes to all six scores. The first
pair is always accepted, so empty input would be an implementation failure.
Only this post-dedup list is evaluated. No pre-dedup result can rescue the gate.
“Raw support” below means unweighted count on this shared post-dedup input.

## Exact score and five controls

For every H define d_i=||H X_i-Y_i||, I_i=1[d_i<=3], N=sum I_i, and
A_ij=1[||H X_i-Y_j||<=3]. Let S=sum A_ij and mu=S/n. Use the geometry core's
inclusive Euclidean-distance convention; verify trace(A)=N. No extra H-specific
pruning, interpolation, local optimization or re-estimation is used.

- **Candidate:** minimize P_H=# {permutations pi: sum_i A_i,pi(i)>=N}/n!.
  Use exact integer numerator/denominator and integer cross-products to rank;
  floating log tails are optional diagnostics only. Because the observed identity
  assignment is included, 1<=numerator<=denominator must hold. There is no
  significance threshold and no interpretation as Pr(H is incorrect).
- **Raw:** maximize N.
- **MSAC:** maximize sum_i max(0,1-d_i^2/9). A boundary d=3 has zero MSAC weight.
- **8x8 cell normalization:** maximize sum_i I_i/sqrt(c_s(X_i)*c_t(Y_i)).
  Counts use all retained endpoints, their original coordinates, cells
  clip(floor(coordinate/64),0,7), origin (0,0); a coordinate exactly512 is in cell7.
  No H-projected cell assignment, grid phase search or neighborhood kernel.
- **Excess:** maximize N-mu. Ranking can use integer n*N-S because n is shared
  across H in a case. Report both this exact key and the descriptive float score.
- **Ratio:** maximize N/mu=N*n/S. If S=0 then necessarily N=0: define its key as
  zero, without an epsilon. Otherwise compare integer rational keys; no rounding
  tolerance for ties. Report numerator and denominator plus descriptive floats.

All five controls use identical candidates, observations, tau and tie order.
Their aggregate tie priority, solely to name a strongest control, is raw, MSAC,
cell, excess, ratio. These are transparent cheap controls, not implementations of
GMS, ORSA or MAGSAC++. Those legitimate baselines remain necessary for later real
pipeline claims; their absence cannot be disguised with a familiar method name.

## Exact solver coverage and resource failures

Freeze the reviewed `exact_permutation_tail` implementation: n<=512, exact
closed-form rook coefficients for complete bicliques; generic components allowed
only when their smaller bipartite side is <=12; **2,000,000 operation units per H**
as counted by that implementation. Component polynomials combine under the global
n!, including isolated vertices. No threshold-dependent exemption from component
coverage. State these caps as algorithmic implementation limits, not properties
of all radius graphs. Preserve every failure reason and attempted-H count.

If any of the 32 H is unsupported, exceeds the operation cap or hits its case
runtime limit, the candidate **fails that entire case**. Do not select from the
remaining supported subset, substitute p=1, change tau, treat a dense component
as complete, approximate, or omit the case from a denominator. It is permissible
to stop that case's remaining tail calls after its first failure, but all 32 cheap
control scores must already have been completed. Continue other cases. Candidate
failure has selected-H=null, transfer-error=null and success=false. Do not encode
Infinity as a JSON error. A software exception or invalid integer probability is
a study-validity failure, not ordinary declared unsupported coverage.

## Execution budget and accounting

One run only, existing free cloud CPU; no asset, API or dependency acquisition.
At most four CPU threads/cores and 8 GiB. Overall hard wall cap **900 seconds**, with
at most **2.5 active seconds per case**, including generation, deduplication,
geometry, all controls and candidate calls. The implementation must enforce the
case deadline, not merely measure it afterward; an outer supervisor enforces the
overall hard cap. Run cases in order s=0..63, and, within s, cycle the family order
by s mod4 from the fixed family list above. This interleaves groups across the
wall budget. No separate warm-up geometry benchmark or outcome-driven rerun.

Compute the complete five-control/32H record before tail calls in each case.
A case timeout after controls are complete is a declared tail failure; a timeout
before controls finish makes that case incomplete and prevents a GO decision.
The 256 case caps total640 active seconds, leaving a ceiling margin for startup,
serialization and review outputs; this is a budget allocation, not a measured
runtime prediction. Global timeout, missing cases or an incomplete control set
means **INCOMPLETE/NO-GO**, with all attempted and unattempted IDs retained. Do
not resume only favorable cases, increase limits, or replace seeds within this
protocol. Cooperative checks alone do not establish a hard limit.

Record actual total wall time, user/system CPU and peak RSS with platform units.
Keep generation/dedup, matched residuals, spatial-count construction, full graph
construction, cheap reductions and integer tail times separately. The candidate,
excess and ratio require the full A graph; raw/MSAC need only diagonal residuals,
and the cell control additionally needs the fixed cell counts. Shared-code timing
must not charge unused graph work to raw/MSAC or disguise shared work as isolated
execution. Any per-method allocation is explicitly an allocation, not a measured
standalone pipeline time. No speedup claim or runtime-ratio gate is made here.

## Quality measure and fixed decision

After selections and failures for all cases are locked, a separate evaluator reads
H* and evaluates on the **9x9 grid linspace(0,512,9) x linspace(0,512,9)**. The
one-way mean transfer error is mean_q ||H q-H* q|| over that fixed grid,
using the separate geometry evaluator. A selection is successful iff error<=3px.
For this pure-translation pool, the metric reduces to the selected offset's
Euclidean distance, and only I passes. State this equivalence in the report: it
is not evidence about nontranslation homography extrapolation. Also publish each
selected offset/error and family success totals, not only a combined score.

Every gate below is required for **GO-to-real-development-qualification**:

1. All256 cases and every cheap control complete; all256 tail cases support all32
   candidates within the fixed caps. Zero silent exclusions or failure fallbacks.
2. On the 64 `clustered_outlier` cases, candidate success count minus the largest
   success count of any one fixed cheap control is **at least8** (12.5 percentage
   points). This compares against the strongest aggregate fixed rule, not a
   per-case oracle mixture and not whichever baseline the candidate beat most.
3. In **each** of `uniform`, `compact_true`, and `sparse_isolated` (64 cases each),
   for **each** of the five fixed controls, at most2 cases may change from control
   success to candidate failure. These 15 paired-harm limits cannot be offset by
   successes elsewhere. Publish full paired wins/losses/ties and net counts.
4. Resource limits and independent saved-result arithmetic review pass. The review
   must not rerun science or modify a failure into a success.

If the strongest primary control already exceeds56/64, eight net gains are
impossible even for a perfect candidate. Report **insufficient headroom at the
predeclared target**, rather than increase difficulty after seeing outcomes.
A result with genuine paired improvements but more paired harms must not be
advertised by its improvements alone. Cases, seeds, mechanisms and solver failures
remain visible. This screen makes no confidence interval or population generality
claim from its four hand-designed families. It is intentionally a stop/go
investment decision, not a statistical test that a paper contribution exists.

Any failed gate is NO-GO for this single exact-tail candidate at this budget and
setting. It is not a theorem ruling out every possible geometry method. No new
radius, mixture score, support floor, “easy subset,” variant or manuscript follows
from this screen. A clean GO permits only a separately reviewed small real-view
development protocol with licensed data, physical scene separation, actual feature
matching, legitimate author baselines and equal measured CPU budgets. Novelty and
independent held-out scene gains remain unproved and necessary before a paper.

## Minimal evidence to retain

Freeze source/config/protocol digests before the run. Save environment, actual
command, source/freeze hashes, start/end/exit/timeout/RSS, all case IDs and RNG seeds,
raw and retained synthetic coordinates/IDs, dedup decisions, candidate order,
per-H N/S/n, A content hash and graph component summaries, exact tail integers,
operation counts, controls and tie decisions, selected IDs, statuses and timings.
Save GT/labels separately from scoring inputs and join only in the evaluator.
Coordinates plus pool and tau permit independent reconstruction of A; saving all
dense matrices is unnecessary. These are small original synthetic artifacts, not
third-party data. A saved-result checker should reproduce pool/dedup/score ranking,
GT errors and the fixed gates; numerical counting correctness is additionally
covered by the already separate exhaustive artificial matrix tests.

Report exact arithmetic under a stipulated **fixed-graph uniform-permutation**
model. Do not call these calibrated false-match p-values, a selected-H false-alarm
guarantee, a posterior correctness probability, a new counting identity, a real
SIFT result or a CVPR-level contribution. The synthetic random-coupling generator
cannot supply the exchangeability missing after real descriptor selection.
