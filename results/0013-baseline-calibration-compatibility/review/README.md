# Independent saved-result review: 0013 baseline

**The official PoseLib baseline completed all 16 selected development pairs.
This is a baseline measurement, with no new-method improvement.** All maximum
pose errors are below 5°. This cohort therefore supplies no observed failure
above that threshold to motivate a failure-recovery claim.

The compatibility retry was frozen at
`07f535046cb5475453f3cb51508cefb65a6e3ac3`. The
[original failed attempt](../../0013-baseline/) remains separate and ineligible;
it stopped during camera metadata validation before estimation. The retry
preserves authored zero dimensions without changing focal/K checks or solver
settings.

## Verified results

| Metric | Result |
|---|---:|
| Completed / selected | 16 / 16 |
| DMS trapezoidal AUC@5° | 81.1696224778351% |
| DMS trapezoidal AUC@10° | 90.58481123891754% |
| DMS trapezoidal AUC@20° | 95.29240561945878% |
| Maximum-pose-error minimum / median / mean / maximum | 0.126850° / 0.784571° / 1.015216° / 2.358317° |
| Pairs strictly below 1° / 2° / 5° | 9 / 14 / 16 |
| Pairs dominated by translation-direction error | 15 / 16 |
| Complete estimator API calls, summed | 3.098394958 s |
| Per-pair API time, median | 0.1964784585 s |
| Loading and validation, summed | 0.016734792 s |
| Worker subprocess wall times, summed | 5.611949376 s |
| Supervisor wall time | 5.926279750 s |

The supervisor time includes integrity/setup and process overhead; the API time
includes the public estimator call and its internal refinement. These are one
run's measurements, not a repeated timing comparison. Every returned RANSAC
iteration count is 1001. Returned inlier/statistic fields describe the library's
RANSAC state before final Cauchy refinement, not a fresh score of the final pose.

| Pair index | Rotation error (°) | Directed translation error (°) | Maximum (°) |
|---:|---:|---:|---:|
| 0 | 0.391023 | 1.683573 | 1.683573 |
| 1 | 0.319433 | 0.858487 | 0.858487 |
| 2 | 0.448158 | 1.657254 | 1.657254 |
| 3 | 0.366924 | 0.581149 | 0.581149 |
| 4 | 0.233054 | 2.262348 | 2.262348 |
| 5 | 0.382540 | 0.582262 | 0.582262 |
| 6 | 0.036084 | 0.256466 | 0.256466 |
| 7 | 0.219690 | 0.370549 | 0.370549 |
| 8 | 0.073734 | 0.126850 | 0.126850 |
| 9 | 0.536500 | 1.243944 | 1.243944 |
| 10 | 0.437371 | 0.303155 | 0.437371 |
| 11 | 0.486677 | 1.090651 | 1.090651 |
| 12 | 0.420945 | 0.614886 | 0.614886 |
| 13 | 0.601569 | 0.710654 | 0.710654 |
| 14 | 0.419305 | 1.408699 | 1.408699 |
| 15 | 1.014796 | 2.358317 | 2.358317 |

## Audit and interpretation

[The audit JSON](audit.json) binds all 16 final and raw worker records,
process receipts, embedded evaluation predictions, selected-key order,
manifest/protocol/source hashes, frozen source revision, and installed library
binary/initializer hashes. Independent calculations from saved matrices apply
`U @ Vt` to GT rotation and use the **signed** normalized translation dot product,
without absolute value. The largest angle discrepancy is
`3.32e-12` degrees; the largest AUC-fraction discrepancy is `4.97e-14`.

All 16 pair logs contain only their completion record; the supervisor log has
16 pair completions and a successful footer; the evaluation log is empty.
No warning or traceback lines were observed. No failure penalty contributes
to AUC, so all-pair and successful-only metrics coincide.

This review used saved JSON/logs and source/binary hashes only. It did not open
HDF5, repeat its full-file hash, or invoke an estimator. The original dataset
hash check is supported by the saved integrity receipt. Raw evidence is in
[evaluation.json](../evaluation.json), [the prediction seal](../predictions/seal.json),
[the integrity receipt](../predictions/integrity.json), and
[the supervisor log](../supervisor.log).

Sixteen overlapping pairs from one development scene and one seed do not
establish population accuracy, recoverability of the remaining error, novelty,
or superiority to another method. Translation dominance is descriptive and
does not identify a causal failure mechanism. No candidate has been compared;
this result does not revive the failed 0012 scorer or meet a paper-writing gate.
