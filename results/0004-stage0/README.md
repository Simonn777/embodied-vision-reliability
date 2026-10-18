# 0004 Stage 0: reset benefits are outweighed by harms on this development video

October 18, 2026 (America/New_York). AI-assisted execution and independent saved-output
verification. **The frozen continuation decision is `stop_this_candidate`.**
This is one synthetic development video with eight fixed queries; it supplies no
held-out efficacy, novelty, deployment, population uncertainty or paper result.

## Result

All 250 frames completed in one scientific invocation, with original query indices
0–7. There are 4,488 complete finite prediction rows, 250 encoder calls and 561
refiner calls: keep 250, zero 250, same-state clone 5 and shadow 56. All 13 equality
controls passed with maximum difference 0 and identical recorded next-context
digests. The original descriptors, current-frame features and model buffers passed
the frozen integrity checks. Frame 0 is excluded from scoring.

Of 1,992 post-query rows per main branch, 1,173 are GT-visible. The baseline has
311 visible errors strictly greater than 4 px. The per-frame zero-context branch
repairs 46 of them but introduces 120 errors among previously correct positions:
net repairs minus harms **−74**. These coordinate counts do not depend on predicted
visibility. They do not show that resetting improves tracking overall.

| Fixed threshold | Keep visible errors | Zero visible errors | Keep Jaccard | Zero Jaccard |
|---|---:|---:|---:|---:|
| 1 px | 868 | 928 | 0.155324 | 0.116062 |
| 2 px | 491 | 599 | 0.415602 | 0.333746 |
| 4 px | 311 | 385 | 0.536496 | 0.486542 |
| 8 px | 205 | 279 | 0.593490 | 0.529830 |
| 16 px | 104 | 181 | 0.656176 | 0.581498 |

Mean threshold Jaccard is 0.471418 for keep and 0.409536 for zero; mean visible
EPE is 5.772102 versus 9.189921 px. Predicted-visible fraction is 0.467871 versus
0.492470; visibility accuracy is 0.782631 versus 0.811245. The visibility gain is
reported alongside the worse coordinate/Jaccard results, not selected as success.
These are the protocol's small development diagnostics, not official benchmark scores.

All **56** eight-frame reset-once shadow episodes are retained: **5 beneficial,
13 harmful and 38 neutral**. Their paired loss-count change is −26 over 448
episode-frame opportunities; mean keep-minus-shadow return is **−0.0580357143**.
Beneficial episodes occur for four original queries [1,3,4,7].

| Predeclared continuation requirement | Observed | Meets rule |
|---|---:|---|
| At least 20 baseline visible errors >4 px | 311 | yes |
| At least 5 coordinate repairs | 46 | yes |
| Positive aggregate shadow return | −26/448 | **no** |
| At least 2 queries with beneficial shadows | 4 | yes |

The experiment is not classified undersensitive under its fixed rule. The negative
aggregate return stops this exact candidate from advancing on this dataset. No
threshold, operator, point selection or protocol was changed after observing it.
It does not prove that every possible learned memory policy must fail. Any changed
mechanism requires a dated prospective design; no training or new experiment ran.

## Execution and provenance

Execution checkout: `ddadc5e17a6ceff3c358d3e3f440d80bf3fd0306`.
Scientific source: `e9f5fe18643167f955ead42212be6cd8d6a63d5e`.
Canonical receipt SHA256:
`e14bb4f3f4b4dd91c96fe63b1d28bc9a95636d560854de190a701c25dc34667d`.
Eight frozen inputs matched current/index/HEAD/source/origin-main Git blobs before
execution. The checkpoint, development NPZ and three pinned TAPNet imports were
verified by the unmodified runner. We reused existing cache/dependencies, with no
data/weight/source download, installation or additional paid service.

Scientific UTC interval: 2026-10-18 17:07:59.892211–17:11:46.962379
(13:07:59–13:11:46 America/New_York). CPU: AMD EPYC 9V74; Python 3.12.14,
torch 2.10.0+cpu, NumPy 2.0.2, dm-tree 0.1.9, einshape 1.0.
Five numerical thread variables and torch intra/inter-op threads were 2; affinity
[0,1], address-space limit 6 GiB, external timeout 900 s with kill-after 5 s.

| Measured layer | Wall seconds | Peak RSS KiB | Observed exit / raw wait |
|---|---:|---:|---|
| Runner's own receipt | 227.070124 | 756,272 | completion status; not its own parent observation |
| Wrapper observes timeout/scientific child | 227.698916 | 756,272 | 0 / 0 |
| Supervisor observes wrapper | 228.006450 | 756,272 | 0 / 0 |

Scientific child CPU user/system times were 430.7784 / 16.203074 s. Encoder/refiner
call totals were 117.144356 / 96.418645 s; whole-frame total was 223.841329 s.
These diagnostic costs include counterfactual work and are not a deployed-budget
claim. Each per-frame time is counted once. Raw commands, source hashes, UTC,
stdout/stderr and both observed exits are retained in the receipts. The supervisor
has no further durable parent wait4 receipt. The cache marker remains in place;
[its parsed observation](marker-observation.json) records its hash without rewriting it.

## Independent verification and entry points

- [Frozen-run receipt](run/receipt.json), [raw predictions](run/predictions.csv),
  [all timings](run/timings.json), [official descriptive summary](run/summary.json).
- [Independent checker](check_saved.py), [independently recomputed summary](independent-summary.json),
  [all derived row errors](independent-errors.csv), [comparison](independent-verification.json).
- [Scientific child observation](scientific-child.json), [wrapper exit observation](scientific-outer.json),
  [independent-check execution](independent-check-execution.json), and adjacent raw logs.

Before inference, all **21 pure controller/scorer unittest tests** passed. No
extra model forward was used as an interface test. Independent verification then
used scalar Python loops, `math.hypot`, a scalar stable sigmoid and cached numeric
development labels, without importing the model/controller/official scorer or
repeating inference. It recomputed all thresholds, visibility, Jaccards, per-query
EPE/repair/harm counts, all 56 episode loss sequences and continuation criteria.
All 2,373 numeric leaves matched with maximum absolute difference
3.552713678800501e-15 (predeclared checker atol 1e-10, rtol 0). Checker child took
0.210719 s, peak RSS 34,000 KiB, observed exit/raw 0/0, empty stderr.
Recorded context-digest equalities were checked, not independently reconstructed
from saved tensors. A final byte/provenance audit is saved separately.

That publication-only audit initially matched its own broad private-key-header
regex and raised an assertion. Its source/observation are preserved as attempt1;
the scanner was narrowed to complete key headers, then the saved-file audit passed.
No scientific code, prediction, metric, model call or threshold was changed or retried.

Only cached development video0 was accessed. Original 0002 calibration/evaluation,
PointOdyssey and all earlier results/scientific inputs remain unchanged. No
independent-sequence confidence interval is claimed. Data/model rights remain
those in the existing 0002 audit; no RGB, weights, GT coordinates or third-party
source archive is redistributed here. Public files are this project's code,
derived predictions/errors/summaries and execution evidence. This result branch
is for parent review; no PR, merge, paper or scheduled-task change is performed.
