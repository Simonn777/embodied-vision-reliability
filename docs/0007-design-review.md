# 0007 prospective design review

Read-only review, 2026-10-18. No scientific gradients, data/weights, cloud, new downloads, or repository modifications. Reviewed the two proposed memos and the seed/timing paths of the prospective gradient core; this is not a completed implementation or result audit.

**Disposition: no remaining conceptual blocker to one bounded falsification screen, provided the clarified definitions below are carried into the frozen protocol/scorer.** Mean sorting is a weak, unvalidated proxy. A pass supports only local gradient-estimation behavior at eight saved development checkpoints; it does not establish a better optimizer, denoiser, or new method. A failure must stop this exact pairing rule rather than initiate an unplanned representation search.

## Logic that holds

Condition on the noisy dataset, saved parameters, and sampled occurrences. Pairing based on original noisy rows, independently of fresh eta, preserves each row's Gaussian marginal. The target must retain the opposite sign from the augmentation: input `p+.5*eta`, target `y_i−2*eta_i`; the negative partner uses both reversed signs. One mean MSE over 512 rows retains normalization. Same-patch uses 256 sampled occurrences twice, not 512 independent examples. All controls spend 512 actual row evaluations.

The two regimes correctly address different quantities. In A, independent/random/mean have the same conditional gradient expectation; same-patch's fixed first half can have a different expectation. Random matching refreshed each replicate contributes matching randomness to A's reported variance. In B, all four have the same empirical-dataset expectation because each base draw remains uniform with replacement. B includes the sampling variance penalty of using half as many occurrences. Neither regime establishes population/scene uncertainty.

The small-alpha Jacobian expansion has regularity qualifications and cannot guarantee behavior for a finite-alpha ReLU model. Scalar patch means do not certify Jacobian proximity. The same-patch leading-term cancellation and even-noise counterexample correctly rule out universal dominance. The candidate consequently needs an empirical A mechanism check and the stronger B comparisons; lower A variance alone is insufficient.

## Resolved seed and timing ambiguities

Root's clarification and inspected core resolve two initial concerns:

- The three master seeds per case/regime are `700000+10000*c+1000*regime+stream`. Each independently seeds a Torch generator that produces 64 int64 replicate seeds. Each replicate's tuple is reused across methods, while its augmentation/pairing seeds vary across replicates. This does not reset all 64 rounds to the same augmentation. Record the generated schedule; do not redraw any coincidental collision based on outcomes. Bootstrap seed becomes `2000000+c`, replacing the memo's colliding `710000+c`; projection seed remains 709999.
- A draws its fixed 512 indices once in separately timed common setup, then each method pays for cloning fixed indices and selecting rows. B redraws the same 512 indices from each replicate's copied seed inside every method timer, even for same-patch. The inspected timer includes generator creation/seeding, selection, permutation or float64 mean/stable sort, own 512/256 Gaussian draws, covariance product, sign expansion, zeroing, one forward/backward, finite checks, flattening and gradient copy. No extra forward/warmup is supplied. Rotation balances all four execution positions over 64 replicates.

Candidate-independent model/input/factor preparation and master/fixed-index setup are separately reported. Audit hashing may be outside step timers, but full case/process wall and audit costs must remain available. The B cost gate uses sums of B step times over all cases; the V×time comparison uses each case's B variance times its 64-step sum. The fixed number 64 cancels in ratios, making this equivalent to using average step cost. Do not mix A's timing or between-case dispatch delays into that definition, or call the ratio an observed training speedup.

## Definitions to make explicit before freeze

1. **Mean diagnostics:** use paired replicate differences `d_r = uᵀ(g_method,r−g_control,r)`, with SE `sample_sd(d)/sqrt(64)`. Check all 3 method pairs in A among independent/random/mean; all 6 pairs in B, for each of 8 fixed unit Rademacher projections and 8 cases. The stated tolerance is a diagnostic warning boundary, not an equivalence certificate or an unbiasedness proof. A warning leaves integrity unresolved and disallows continuation; do not extend replicates until it disappears. Preserve a zero-SE discrepancy.
2. **Bootstrap:** within each fixed case, sample 64 whole replicate bundles with replacement, preserving shared-index/noise/method pairing, and recompute V for every method. Combine the 8 case log ratios with equal weights. Freeze the 2,000-resample RNG and exact one-sided quantile definition (`1−.05/3`, e.g. linear interpolation); do not resample patches, components or cases as if independently sampled scenes. If any required V/time/ratio is nonfinite or its denominator is zero, mark the screen unresolved/failed, not passed via clipping. A centered Gram-matrix computation is algebraically equivalent and can keep bootstrap work bounded; it must not change the statistic.
3. **Decision:** the additional A condition is `geomean(V_mean/V_random)<1` across all 8 cases. B requires all three predeclared V and V×time ratios, the three upper bounds, six-of-eight per-case wins, ≤1.10 independent total step time, all 8 completed cases and integrity. It is an intersection, not a menu. Single-run timing is noisy; no unplanned retiming to move a near-threshold result across the boundary.
4. **Provenance/access:** bind exact selected colored R2R case IDs, input NPZ/weight/receipt hashes, model/covariance factor and flattened parameter order. The numerical loader should accept only noisy+C and the exact six weight tensors; receipts can establish identities without opening reference, prediction or score files. Keep before/after weight hashes and zero optimizer updates. Raw gradients stay private/local for independent saved-array replay, not GitHub. Preserve the old 0006 logging defect separately; the new gradient operation neither repairs it nor inherits a false all-provenance-pass status.

All 8 cases and all 4096 forward/backward evaluations are required. The 90-second cooperative, 120-second external per-case, and 900-second overall limits are sensible bounded failure conditions; incomplete output is not a reduced successful study. No result-based resume, case replacement, or new data allocation is justified here.

## Historical draft identities

- Minimal design: `a88f4bd8907cac3f55aa39013193b26551eab8e9a30584e39d8d925d5f20b0c6`.
- Mathematical memo: `6fc4b27ff7c927d2430d2cb190b80e57e1980f3032010a7e9be9ebcc8548dddc`.
- Core at inspection: `09c6c92072410ee5f39dcf3aa609363001ad8d9c66e7b456dbcb9f2a39e85465`.

The first memo still contains superseded seed/timing/decision text; the prospective freeze must include root's clarified version, not silently treat that old text as final. No scientific outcome was accessed in this review.
