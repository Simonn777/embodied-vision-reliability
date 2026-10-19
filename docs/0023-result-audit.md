# 0023 independent saved-result audit

2026-10-19. **Verified failure of all four prespecified development conditions. Close this fixed hypothesis; no manuscript or novel improvement is supported.** The wider research objective remains open.

The review used saved receipts, source text, sealed predictions, selected saved GT and independent standard-library arithmetic. It imported no scientific predictor/evaluator, decoded no image/depth/keypoint arrays, and ran no model, native estimator, fit, network acquisition or scientific evaluation. Cache and native-library files were hashed as opaque bytes. Only this review document was written.

## Integrity, coverage and label barrier

- Live GitHub metadata confirms [CI 39342283046](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39342283046) succeeded at exact prediction/evaluation freeze `cb520839515381db125f64ea3bcceb434444a85b`. Input freeze remains `acf502b831ad9e3485ab56549217c8a603d56bf7`.
- All 32 distinct commit/path source checks matched both Git and current bytes. The separate input freeze, unchanged original protocol, technical amendment, old frontend/model-qualification receipt chain and native binary hashes remain bound.
- The prediction receipt, seal, four output hashes/sizes, both successful outer receipts, commands and stdout/stderr hashes match. Both prediction02 and evaluation02 exited 0 without remaining descendants or cleanup errors; their stderr files are empty. All 171 frontend cache file hashes/sizes match their original receipt, without reopening their arrays.
- Every one of the 24 fixed queries has all seven terminal outcomes: 168 total, with 120 completed and 48 exact native-E fallbacks. No query is omitted and no terminal outcome is failed. Native E completed all 24; each of the other six arms completed 16 and fell back on the same eight queries in s00513. Every saved fallback pose equals its fixed native-E pose.
- The GT acquisition receipt binds the complete prediction seal, outer receipt, logs and outputs. Frozen control flow performs the complete barrier before GT acquisition and rechecks it afterward. The saved evidence is consistent with that barrier; this is not an independent OS-level access trace.
- Exactly the three planned pose members were fetched in nine successful range requests: 117,919 response-body bytes. Their hashes, sizes and CRCs match. The audit scanned filenames but converted only the selected anchor plus eight query poses per scene: **27 numeric GT rows**, matching the recorded parsing counts. No unselected pose numbers were parsed.
- Authored W2C quaternions, translations and selected row identities match the saved evaluation. Independent relative-pose and directed-angle calculations reproduce all 168 outcomes; largest angular difference is `7.3165e-12` degrees, below the `1e-7`-degree review tolerance.

## Independently recomputed result

For the frozen DMS convention, the audit used the closed-form integral
`(m*T - sum(errors below T) + e_m/2)/(N*T)` when at least one error is below threshold, otherwise zero; `m` counts errors strictly below `T`, and `e_m` is their largest value. This is separate arithmetic from the production trapezoid implementation. All overall and per-scene summaries, both directed and labelled sign-folded metrics, status counts and medians match. Maximum AUC-fraction discrepancy is `2.2204e-16` (review tolerance `1e-12`).

| Method | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| Native E, reference-filtered query | 52.122880 | 68.629218 | 78.064609 |
| Geometry multiview BA | 61.984444 | 75.643333 | 83.655000 |
| Joint depth-field BA | 57.863271 | 73.582746 | 82.624706 |
| Hessian | 56.549094 | 72.925658 | 82.296162 |
| Deletion scalar | 56.008201 | 72.655211 | 82.160939 |
| Residual scalar | 56.642436 | 72.972329 | 82.319498 |
| Directional candidate | 56.497091 | 72.899656 | 82.283162 |

The directional candidate loses **5.487353 AUC@5 percentage points** and **1.371838 AUC@20 points** to geometry multiview, the strongest control at both thresholds. Its AUC@5 margins against the best control within each scene are 0.000000 (s00513), −9.666212 (s00475) and −4.878036 (s00478). Therefore overall +2 points, positive margins in at least two scenes, worst-scene loss no worse than one point, and AUC@20 loss no worse than half a point all fail. The positive comparison against native E alone does not establish a new contribution.

Prediction02 outer wall time is 298.587250167 s with peak sampled process-group RSS 1,173,520 KiB. Evaluation02 is 7.046813708 s and 340,128 KiB. These are whole-operation costs shared across methods, not matched per-method benchmarks. The original frontend and prediction01 failure remain preserved and charged in the study history.

## Public evidence and limits

The [completed-study publication](../results/0023-development/completed-study/README.md) passed the saved-byte comparison: seven receipt/record copies are identical; four logs differ only by the declared literal local-path replacement; four private output entries match their hashes and sizes. The [public evaluation](../results/0023-development/completed-study/evaluation-public.json) differs from its private original only by removing `rows[*].ground_truth` and adding `publication_note`. Every estimated pose, error, status, aggregate and gate is retained. All 28 [CSV rows](../results/0023-development/completed-study/metrics.csv), their counts and 84 AUC values match. An initial publication-audit snippet used the wrong CSV column name; correcting that read-only lookup to `population` completed the check without modifying results.

| Bound artifact | SHA256 |
|---|---|
| Prediction receipt | `bad670f9bf183f1e964995066ff6c6490e7af3ac321bb5e7975e03572ca09903` |
| Prediction seal | `e6b70e8f9818819929a5b21d7f6bac10b8c134a2aa810877dc89fec384d618ad` |
| Private evaluation | `092aea6e4612d8b1d9d76e891a56db042fe9a5c9179f3e01107dc43a818919b0` |
| Public evaluation derivative | `1abcdf62998438caacb83d473ff5e796da604f71c4e7092109afc2b35fa63fb2` |
| GT acquisition receipt | `24a521191143b1fa45f64e830555811c5b4025bb9fd00c5876fb1e4567c6668b` |

This audit validates the saved evidence, coverage, metric calculation and fixed decision. It does not independently rerun optimization or establish that every fitted objective was globally minimized. Three dependent development places, including a whole-scene fallback, support closing this implementation and protocol; they do not prove a universal impossibility theorem about uncertainty estimation. Do not retune this fixed hypothesis on these 24 queries or describe its native-E comparison as a successful new method.
