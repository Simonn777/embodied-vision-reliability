# Frozen v2 evaluation: no practical motion-residual benefit observed

October 17, 2026. AI-assisted orchestration, measurement, independent recomputation and writing.

The predeclared extra constant-velocity residual did not meet the practical benefit
criterion over native temporal confidence on these fixed tracker coordinates. At
matched 80% coverage, augmented minus native failure risk was +0.169683 percentage points,
with paired 95% bootstrap interval [-0.157781, +0.712631] percentage points.
The interval includes zero. This neither establishes benefit nor demonstrates
general harm. The prespecified practical signal (delta <= -2 pp and CI upper < 0)
is **false**. No method, threshold, endpoint or partition was changed.

## Scope and exposure

This is the first and only frozen v2 evaluator invocation on the existing
10 calibration + 20 evaluation prediction groups. No model construction, weights
loading or inference ran. Ten calibration groups supplied 1,141 visible post-query
opportunities; twenty evaluation groups supplied 2,209. Each video contributes
only its first 64 frames and first two frame-0-visible annotated indices. Those
few fixed points are not representative random samples. All 30 original prediction
sets (3,840 rows) and the 988 prior tracked results/docs/protocols remain unchanged.

Evaluation labels/errors have now been accessed. These videos cannot be called
untouched in future work. This remains an exact-duplicate-group-disjoint exploratory
pilot in a shared object-set/camera domain: scene independence and complete
checkpoint training/model-selection non-exposure are not established. Native,
visibility-head and local-head confidence come from a temporal causal tracker;
none is a strict single-image baseline. The original single-image question remains
unresolved. The expected-distance head was trained for >6 px, while this frozen
endpoint is strict EPE >4 px. No 6 px secondary endpoint was added.

## Fixed matched-coverage comparison and all controls

Every method has 2,209 jointly available evaluation opportunities. The fixed
target ceil(0.8 * 2,209) accepts 1,768, coverage 80.036215%.
Missing coordinates and unavailable joint scores are both 0%; attainable coverage
is 100%. Rows with no acceptance/opportunities would retain null risk.

| Ranking | Accepted failures / 1,768 | Conditional failure risk |
|---|---:|---:|
| native | 4 | 0.226244% |
| augmented | 7 | 0.395928% |
| visibility | 6 | 0.339367% |
| local | 0 | 0.000000% |
| random | 59 | 3.337104% |
| oracle (non-deployable) | 0 | 0.000000% |

Oracle reads GT EPE and is descriptive, never deployable. Random ranks by the
predeclared method/outcome-independent hash tie. All six complete risk-coverage
curves are in the immutable [evaluation result](../results/0002-frozen-evaluation-v2-20261017/evaluation.json);
no new coverage or winning control was chosen from these results.

## Calibration thresholds and operational transfer

Five boundaries were fit once by the frozen evaluation on calibration only, then
applied unchanged to evaluation. Independent audit recomputation verifies those
same boundaries; bootstrap never refits them. All five calibration targets were
reachable and accepted 913 / 1,141 opportunities (coverage 80.017528%).
A boundary is the lexicographic pair (score, original hash tie), not score alone.

| Policy | Score boundary | Full tie boundary | Calibration risk | Eval accepts | Eval coverage | Eval risk |
|---|---:|---|---:|---:|---:|---:|
| native | 0.018163505156090921 | `e2875e32d9c8892ab4bddbb16c065968938b6eacc5403135b3a1ee0793c758aa` | 0.000000% | 1855 | 83.974649% | 0.431267% |
| augmented | 0.07856091778632783 | `368237615109efba53a9ef35470b7a8956e99e7ee7cddb6f002e9ce90586b865` | 0.000000% | 1826 | 82.661838% | 0.492881% |
| visibility | 0.01305009785674126 | `729c60a7eb27eb87e98ee55fab9ba64e838de143efc64aceccc05efa1e9c5247` | 0.438116% | 1833 | 82.978723% | 0.436443% |
| local | 0.0030934482493569786 | `76d3f482b2bd029a21e9f56410ac89847d5f4dc95ef5323e39ef8b5b85def69b` | 0.000000% | 1886 | 85.377999% | 0.212089% |
| random | 0 | `c851d8a9963eaab5f5e22272cc118e96745805374c05811d40273d43e2e067f9` | 5.147864% | 1761 | 79.719330% | 3.350369% |

Native transfers at 83.974649% coverage and 0.431267% risk;
augmented transfers at 82.661838% coverage and 0.492881% risk.
Unequal operational coverage is not a matched-risk superiority comparison.

## Paired uncertainty and diagnostics

The complete allocated 20-group frame was sampled with replacement 2,000 times,
seed 20261018. Every sampled group copy and its original ties remain; each draw
recalculates its own opportunity count and ceil(80% * Ndraw). Calibration
boundaries remain fixed. All three effects have 2,000 defined replicates, meeting
the >=1,900 interval rule. Intervals are paired group percentile 95%, conditional
on this calibration set and capture domain, not independent-scene guarantees.

| Effect (augmented minus native) | Point, pp | 95% interval, pp | Defined |
|---|---:|---|---:|
| matched_risk_difference | +0.169683 | [-0.157781, +0.712631] | 2000 |
| operational_risk_difference | +0.061614 | [+0.000000, +0.220750] | 2000 |
| operational_coverage_difference | -1.312811 | [-5.041834, +3.773254] | 2000 |

Coordinate-only accept-all coverage is 100.000000%, conditional failure
risk 3.259393% (72 / 2,209); estimator failure including missing
is 3.259393%. Coordinate missing and joint-score unavailable rates
are both 0.000000%. Visibility-head accuracy on all 2,520 post-query
rows is 96.111111%; nonfinite head logits would count incorrect. No occluded
coordinate-error endpoint was introduced. Frame 0 is excluded from opportunities.

## Execution and provenance

Actual execution main: `43b64a3936bc05ed05c89894768c630391492cf4`, distinct from the
later publication commit. Freeze input `bbe7428e126a566a62d3c123d7158f120aa66e2b`;
canonical receipt SHA256 `bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`.
Real verify_freeze passed all 29 mandatory committed/current/index-bound inputs
and 16 exact cached dependency versions/origins. Archive 187,291,581 bytes SHA256
`ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc` and existing
checkpoint/upstream source identities were byte-checked before evaluation. No
download_object network fallback was needed; existing caches were required first.
NumPy 2.0.2 remains the disclosed version-matched reconstruction, not historical
byte-exact reproduction. Both normalized UTF-8 RECORD and raw byte hashes are kept.

CPU reported AMD EPYC 9V74 80-Core Processor; Python 3.12.14.
Kept env -i/fixed runtime-fixed:runtime:source:src PYTHONPATH, all five numeric
thread variables=2, two-CPU affinity, 6 GiB AS, external timeout 900 seconds.
Preflight imports torch only to check package/native identity and threads; it
does not construct a model or load weights. The frozen evaluate branch itself
uses NumPy, restricted dataset decoding and stored predictions, not torch/model
inference. Existing source/config/protocol/runner/metrics are unchanged.

| Observation | Elapsed seconds | Peak RSS KiB | Observed exit / raw wait |
|---|---:|---:|---|
| Preflight child | 1.891330699 | 225,884 | 0 / 0 |
| Evaluate runner self measurement | 22.129504583 | 2,551,948 | see parent child observation |
| Evaluate child, outer wait4 | 22.285626334 | 2,551,948 | 0 / 0 |
| Wrapper, parent wait4 | 24.214952574 | 2,551,948 | 0 / 0 |

Original UTC times, exact command/environment, CPU usage, source/config/protocol
hashes, stdout/stderr and separate child/wrapper exit observations are linked in
the [outer receipt](../results/0002-frozen-evaluation-v2-audit-20261017/outer.json) and [supervisor receipt](../results/0002-pilot-execution/evaluation-v2-20261017-supervision/supervisor.json).
Runner result SHA256 `fd2285f4de7f057758f747f91bb07a08f9eec62adc490dfd0c06216dec3fd795`.

## Independent numerical audit

Thirty per-group [review CSVs](../results/0002-frozen-evaluation-v2-audit-20261017/rows) expose original ties, post-query/
visibility/opportunity flags, coordinate/logit validity, raw prediction xy/logits,
EPE/failure and all six scores. They contain no video, weights or full GT xy.
A second permitted restricted decode uses the unchanged frozen row builder to
produce these audit labels/EPE; this is not an independent annotation re-labeling.
Each derived CSV is <1 MiB. The independent implementation imports no evr/candidate
metrics, dataset decoder, torch or model. It uses CSV/math for metrics and causal
score checks, and NumPy only for the identical seeded PCG64 draws. It does not
call frozen evaluate again. Rankings use exact exported score values after the
separate independent raw-history score check; they do not perturb floating ties.

The independent implementation checked 23040 causal-score values and all 3840 ties,
and compared 26566 numerical plus 123 exact result leaves, including thresholds,
all curves, controls/diagnostics and all three bootstrap intervals. Maximum score
difference 0.0; maximum metric difference 0.0. Fixed atol 1e-12, rtol 0.
All 2,000 draws and replicate effects are retained in [draws](../results/0002-frozen-evaluation-v2-audit-20261017/bootstrap-draws.csv)
and [replicates](../results/0002-frozen-evaluation-v2-audit-20261017/bootstrap-replicates.csv). No sampled-copy deduplication or threshold refit.
The independent full result is byte-identical to the original result.

Post-run children have actual parent wait4 exits, source and stdout/stderr hashes.
The same post-run parent closes phase bindings without self-hashes; its own exit
has no further supervisor receipt. Final publication-audit receipt/logs and its
verification output are excluded from their own maps and bound by the final Git
tree. Archived source copies document the original scratch entry points.

One shared software probe/synthetic smoke ran, no install. Work pytest unavailable
(0 tests); public exact-publication-head Research checks CI must pass and its actual
logs be inspected. Synthetic smoke is software validation, not model/real-data
evaluation. No paid APIs/GPU/services/credits, additional downloads/installs, raw
video/weight uploads or account/permission/schedule changes.

## Next step

Retain this negative practical result and these now-inspected evaluation labels.
After independent review/merge, a separately authorized post-hoc diagnosis may
explain differential accepted failures and residual penalties. Any later method
change requires explicit new freezing and genuinely appropriate untouched data
before new evaluation; do not tune on these videos and relabel them untouched.
No SOTA, confirmatory result, real-robot validation, paper reproduction or general
efficacy claim. Keep this PR unmerged for independent parent review.
