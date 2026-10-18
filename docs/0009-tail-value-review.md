# Conditional permutation tails: value review, not a method approval

2026-10-18. Conceptual review only: existing lane-A/0009 notes and prototype.
No geometry performance, network, installation, data access or repository edit.
The separate agents own exact-tail arithmetic and prior-art verification.

**Verdict: unlike pure z, an exact tail is not already disqualified by the
k-block versus isolated-support counterexample. A single bounded headroom check
could be justified after arithmetic and close-prior review. It does not yet justify
a full estimator, dataset acquisition, CVPR claim or broad implementation effort.**
If the closest work already implements the same conditional empirical-marginal
tail, numerical exactness alone is not a reason to continue this as a new method.

## Where a real contribution could disappear

**Sparse graphs can reduce the proposal to ordinary support.** If the binary
compatibility graph consists of M disjoint edges, its rook numbers are
r_j=binomial(M,j). At fixed n and M the null distribution is identical across
hypotheses, so ranking its upper tail is exactly monotone in observed support N
(up to discrete ties). No extra geometric information is introduced. If M varies,
the tail can still change ranks: it also incorporates the compatibility opportunity
count/mean M/n. Thus sparse graphs do not prove universal equivalence to count,
but expected-support correction is the necessary stronger comparison. When most
plausible SIFT hypotheses have essentially isolated compatible edges, meaningful
headroom may be negligible despite mathematically different tiny tail values.

**Dense blocks can reflect feature replication rather than difficult geometry.**
Multiple orientations/scales at almost the same SIFT position remain separate
feature IDs and may survive descriptor mutual matching. They generate local
bicliques. An exact tail does not automatically remove their dependence: for a
k-block observed at its maximum, the tail is 1/binomial(n,k). Replicating one
physical correspondence changes n and k and can create strong apparent evidence.
Endpoint conditioning is therefore not a substitute for physical-location dedup.
The k-block example that defeats z is a mathematical diagnostic, not proof that
these blocks create a novel real-world failure mode. Conversely, useful nonduplicate
nearby points must not all be dismissed as fake replication.

**The hard-radius statistic still discards within-threshold residual quality.**
Two equally significant correspondences need not yield equally accurate transfer.
A score that wins against count but loses to MSAC or MAGSAC is not a task advance.
A coherent wrong plane/periodic shift can also be significant under this null.
That limits the claim without refuting the narrower accidental-outlier question.

## Minimum cheap controls, fixed before any results

Use one common proposal/verification split, candidate pool, residual convention,
valid-model rules, tolerance and fixed tie order. Split whole near-location feature
groups so orientation clones cannot straddle proposal and verification. Every
method must see the same retained correspondences; scores do not access GT.

1. Raw support and MSAC. They establish whether binary-tail ranking improves
   actual geometry rather than just the significance number.
2. Hypothesis-independent near-location dedup, using one fixed pixel tolerance and
   descriptor-quality tie rule. Apply it to **all** methods for the primary test;
   raw duplicate-rich matches can be a declared diagnostic. Do not tune tolerance
   or select representatives by a candidate H or GT.
3. One fixed 8-by-8 marginal cell-density control, for example
   sum_i I_i/sqrt(c_source(cell(x_i))*c_target(cell(y_i))). Cell counts and weights
   are fixed from the same verification endpoints, not fitted per hypothesis.
4. Expected-support excess N-mu and ratio N/mu using exactly the same radius graph
   as the tail; define mu=0/N=0 neutrally. These are cheap arithmetic once the graph
   exists and test whether higher-order matching conflicts matter beyond density.

These are controls, not additional proposed methods. GMS already represents
spatial/density support, and ORSA already represents a-contrario/random-coupling
reasoning; neither can be replaced by an informal namesake implementation. A
subsequent real full-pipeline claim requires a pinned legitimate ORSA/a-contrario
comparator, the relevant GMS filtering control, and MAGSAC++ on identical matches
with equal measured total CPU budgets. A same-pool study isolates ranking; it
cannot replace that end-to-end comparison. Current absence of cv2 is a capability
boundary, not permission to call a weaker implementation MAGSAC++.

## Probability claims that are and are not available

Allowed: “We calculate an exact upper-tail probability for a fixed binary graph
under uniform permutation of the observed endpoint labels,” subject to proven
arithmetic and overflow/resource handling. Empirical geometric accuracy/runtime
could later be reported with its actual protocol and limitations.

Not allowed: calibrated false-match p-values in SIFT/mutual/ratio-filtered data,
a guaranteed real-scene false-alarm rate, posterior probability that H is correct,
or an automatic familywise guarantee after selecting the best H. Descriptor
similarities, mutual constraints, ratio thresholds and spatial preprocessing mean
incorrect surviving assignments are generally not uniform conditional on endpoint
locations. One-to-one matching does not establish exchangeability. Disjoint proposal
pairs remove direct self-fit support but do not make descriptor matching uniform,
and reusing verification pairs to select H adds selection effects. A multiplicity
factor cannot repair a false null. “Exact” concerns the declared combinatorial
calculation, not truth of the scientific null or originality.

## Small go/no-go investment boundary

After the independent arithmetic/prior reviews, the only reasonable next investment
is **one** fixed-pool development headroom test, at most900 seconds, four CPU threads
and8GiB, with one tail definition and no fitted score variants. Freeze before GT
outcomes. Do not silently discard graphs that exceed the exact solver's component
or time cap, approximate their tails, or report only easy graphs as complete exact
coverage. Numerical feasibility and unsupported cases are separate outcomes.

First record graph diagnostics after common dedup: row/column degree distribution,
component sizes, edge count, location replication, and whether tail selection differs
from all cheap controls. Uniform sparse cases and compact true support are no-harm
controls; clustered accidental outliers are the primary mechanism target. A periodic
false explanation is a declared boundary, not a reason to deny accidental headroom.
Any planted-H pool is explicitly an oracle-coverage ranking experiment. Compare
selected-model transfer error against independent known geometry, not support or
negative log-tail. Charge graph construction and tail counting to the candidate;
raw/MSAC do not pay for a graph they do not use.

Stop if the candidate almost always selects the same H, if any improvement vanishes
after fixed dedup/cell normalization, if simple expected-support scores match it,
if it loses true-cluster accuracy, or if exact coverage/runtime fails the frozen
budget. A useful signal must be a prospectively sized task gain over the strongest
cheap control, not just nonzero rank disagreement or a less approximated number.
Do not rescue a failure by choosing another graph threshold, component subset or
feature granularity afterward.

Only a clean gain would justify checking a small licensed **real-view** development
subset and proper author baselines. Synthetic point clouds cannot establish what
actual SIFT/mutual compatibility graphs look like. HPatches eligibility can remain
a future data route; no acquisition is authorized here. On present evidence the
scientific value is unresolved and probably narrow: exact counting is a potential
implementation ingredient, not yet a CVPR contribution.
