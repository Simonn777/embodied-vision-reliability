# 0025c: fixed four-point initializer fails the necessary noise gate

**Measured: closed. No robust-sampling stage, real-data campaign or manuscript is admitted.**
The one frozen run completed all 132 manufactured records and 13,332 solve attempts.
Every required useful-hypothesis threshold failed. The angular-only two-point
control was much stronger. The separate frozen numerical audit processed every
record and independently reproduced the failed gate, but its status remains
**FAILED** because three residual comparisons disagreed in one already-invalid
critical diagnostic. The original failed audit is retained; software CI does not
override this scientific audit status.

## Prospectively fixed experiment

[Protocol](../../protocols/0025c-rs-initializer-noise.md) and all production/audit
sources were frozen at `93349f34790496c5c838a2b97f9440a471d6dc68`.
[Source CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39357521012)
passed before the first cohort was generated. Its 37 artificial contract tests
did not generate the target noise cohort. Original preparation snapshots and
test receipts remain in [preparation](../0025c-preparation/).

Eight scene points undergo exact rolling-shutter projection in four declared
motions: generic G, near-identity N, shared constant twist C, and critical E.
Each motion has one clean record and 16 paired P/S realizations. Both noisy regimes
use 0.5-pixel noise and a declared 3% mixture of coherent and independent depth
error. P uses exact manufactured angular inputs; S adds fixed angular-parameter
stress. Neither is measured-IMU evidence. Seed, distributions, all points,
positivity/rank checks and thresholds were fixed before execution.

Each record contains all 70 four-point depth/velocity fits (D4), all 28 two-point
angular-only affine fits (A2), and eight-point D8, depth-free N8, and angular-only
affine A8. These share observations; subset counts are not independent trials.
Invalid outputs remain at 180 degrees in all denominators. No outliers, robust
selection, refiner, renderer or real image is involved. Timing is descriptive.

## Complete necessary-gate outcome

Useful means valid directed translation error at most 5 degrees. Every listed D4
group required at least 144/1,120 useful outputs.

| Motion | P: exact angular inputs | S: angular-input stress | Required |
|---|---:|---:|---:|
| G | 7 / 1,120 | 6 / 1,120 | 144 in each regime |
| N | 12 / 1,120 | 13 / 1,120 | 144 in each regime |
| C | 0 / 1,120 | 0 / 1,120 | 144 in each regime |

The additional necessary headroom comparison also fails in all three S groups.
It requires `D4_useful > 10*A2_useful`; the paired counts are 6 versus 341,
13 versus 357, and 0 versus 115. A2 denominators are 448. The multiplier derives
from the prospectively declared hypothetical inlier rate and unequal subset
counts; it is not a measured robust-speed result. E remains fully reported but
was excluded from the advance gate before outcomes.

Failure-inclusive S median directed errors (degrees):

| Motion | D4 | A2 | D8 | N8 | A8 |
|---|---:|---:|---:|---:|---:|
| G | 99.51 | 2.52 | 24.11 | 173.30 | 3.68 |
| N | 163.81 | 1.97 | 57.65 | 6.48 | 2.06 |
| C | 90.26 | 9.95 | 60.09 | 159.80 | 9.45 |

The [saved-only interpretation](interpretation-01/README.md) aggregates rather
than reruns these fits. Across the 96 primary P/S records, D8 beats A8 in zero,
loses in 94, and ties in two joint failures; useful counts are 2 versus 61.
D8 always attains an algebraic residual no greater than A8 while producing no
lower task error. Most D4 failures are finite, formally valid but wrong estimates,
not rejection artifacts. Inflated saved velocity estimates are consistent with
nuisance parameters absorbing noise; this does not prove that regularization or
trajectory coupling would work. No failed setting is retuned here.

## Audit failure and its limited effect

The original [audit JSON](audit/audit.json) checks all 132/13,332 entries and
independently returns `gate=False`, with exactly three failed comparisons:
`99/99.residual`, `99/99.normal_equation`, `99/99.residual_norm`.
Record 99 / solve 99 is E-clean N8, already unavailable because its rank is 3
rather than the required 8. Normalizing a nearly zero translation in an arbitrary
null vector produces velocity coordinates near 8.54e18. Floating dot products
then suffer extreme cancellation.

[Saved-only forensics](audit-discrepancy.md) reproduce the original einsum
residual bitwise, compare an alternate reduction and exact sums of the stored
binary64 products, and explain the discrepancy. No solver, SVD, generator or
audit rerun occurred. The forensic invocation's ResourceWarning is preserved.
Neither tolerance nor original audit status changed. This diagnostic's residual
does not enter validity or any advance condition, so the independent count-based
closure is unaffected. **This is not an all-numerics-passed result or independent
team replication.**

## Execution and provenance

| Operation | Outer wall seconds | Sampled peak KiB | Exit / status |
|---|---:|---:|---|
| Frozen noise run | 13.104796458 | 49,008 | 0 / completed; utility gate false |
| Frozen independent audit | 18.557059917 | 64,512 | 1 / failed, as disclosed above |

Both outer supervisors report no timeout, surviving descendants or cleanup
errors. Python 3.9.6 / NumPy 2.0.2 ran with four-thread limits, no network, model,
dataset acquisition or new GT access. [Noise receipt](noise/receipt.json),
[audit supervision](audit-supervision/receipt.json), launch records and logs are
retained. Exact runtime arguments are in the launch records; `$WORKSPACE`
replaces only local path prefixes in public copies.

[publication-bindings.json](publication-bindings.json) lists 158 copied or
derived bindings and the all-estimates digest. [all-estimates.csv](all-estimates.csv)
contains every one of the 13,332 attempts, validity reasons, directed errors and
estimated parameters. [noise/all-solves.csv](noise/all-solves.csv) is the original
scalar table. [inputs/](inputs/) retains all 132 exact manufactured inputs and
clean checks, with SHA256 links to the complete original case files. Full local
case files also contain matrices, factorizations and diagnostics and remain
immutable under [noise/seal.json](noise/seal.json); this compact public view does
not claim to contain those full arrays. The frozen sources can reconstruct them.
The [publication review](publication-reviews.json) preserves its initial FAIL
for stale document headings and the subsequent PASS for the heading-only repair.
The reviewer verified all 158 bindings, 132 input derivatives, 13,332 result rows
and seven source hashes; this file-integrity check is distinct from the original
failed numerical audit.
The descriptive/forensic scripts expect that original private directory layout;
copying them into this compact publication does not make them standalone commands.

An additional [primary-source screen](coupled-motion-prior-screen.md) rejects
generic shared-trajectory/acceleration coupling as a new two-point contribution:
Su et al. ICCV 2025 already establishes the relevant asynchronous geometry, and
the two-point translation direction has a depth-free counterpart. This literature
decision admits no follow-up experiment. The overall two-source, matched-cost
improvement goal remains open; no paper is written from these failures.
