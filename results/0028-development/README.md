# 0028 result: fixed normal-offset candidate failed

**No pose improvement or manuscript admission.** The complete 27-image,24-query known-control opportunity test failed three of four prespecified clauses. ANG AUC@5 was **2.695397%**, against **51.000123%** for E (tied by AFFINE): **−48.304727 percentage points**. All three scene margins were negative. This closes the fixed variant without outcome-based changes to grid, losses, bounds, camera controls or selection.

All24 ANG predictions were genuine completed fits; every FULL image/pair comparator was valid. Missing computations do not explain this failure. ANG also trails raw C by2.442834points and FULL by2.459854points at AUC@5. Improvement over the much weaker LS arm alone does not establish a useful contribution.

| Arm | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) | Genuine / fallback / unavailable |
|---|---:|---:|---:|---|
| E | 51.000123 | 67.087784 | 79.148344 | 24 / 0 / 0 |
| A | 6.236841 | 15.601983 | 31.205228 | 24 / 0 / 0 |
| B | 5.098098 | 12.451650 | 29.497576 | 24 / 0 / 0 |
| C | 5.138230 | 12.429056 | 29.458617 | 24 / 0 / 0 |
| LS | 0.000000 | 5.276578 | 18.617436 | 22 / 2 / 0 |
| IRLS | 2.837902 | 10.339291 | 27.674617 | 24 / 0 / 0 |
| ANG | 2.695397 | 7.063840 | 24.551826 | 24 / 0 / 0 |
| FULL | 5.155250 | 12.108517 | 28.213736 | 24 / 0 / 0 |
| NORMAL | 3.967367 | 10.218025 | 21.359737 | 24 / 0 / 0 |
| NORMAL_REFINE | 22.259498 | 37.774875 | 57.500313 | 24 / 0 / 0 |
| AFFINE | 51.000123 | 67.087784 | 79.148344 | 1 / 23 / 0 |

Every denominator is24. AFFINE selected the E incumbent23times; its one selected refined pose produced the same reported AUC as E. This is a fixed-support opportunity test, not a reproduction of a complete official robust-pose system. The current E uses query-only matches and cannot be directly equated with earlier reference-filtered0023 results.

| Development scene | ANG AUC@5 | E AUC@5 | Margin (pp) |
|---|---:|---:|---:|
| s00513 | 0.000000 | 16.336062 | -16.336062 |
| s00475 | 8.086190 | 62.334292 | -54.248102 |
| s00478 | 0.000000 | 81.475769 | -81.475769 |

The comparator-trust clause passes. The +2point AUC@5/coverage clause, scene-consistency clause and AUC@20 noninferiority clause fail. All three places remain development forever; no unseen-source or population-confidence claim is made.

## Execution and independent verification

- Initial source `427d3038399653908803ea6bb62114771e9f655c` passed [CI39371128015](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39371128015), then stopped during a compatibility warning before any raw forward. Its complete failure and warning-ledger repair remain in [qualification](../0028-qualification/README.md).
- Scientific source `e7448319ff9a32fa07436fbd8da53c605fdada74` passed [CI39371479582](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39371479582) before the one complete scientific operation sequence. All104 distinct artificial checks passed before this source.
- Frontend:27/27 raw forwards,42.278514seconds, peak2,064,416KiB. Prediction:24/24 queries,114.506634seconds, peak680,128KiB. Evaluation:9.629114seconds. Independent audit:4.397505seconds. All outer exits were clean, with four CPU threads and fixed memory/time limits.
- Independent saved-output audit passed **2,451 comparisons** across264 pose outcomes, independently recomputing directed errors, DMS trapezoids, coverage and all gate clauses. It did not rerun fitting, native/MSAC scoring, models or original truth extraction.

The [additional saved-evidence review](evidence-review.md) passes635 checks. ANG reduces its own image objective on27/27images but worsens pose error versus raw C on16/24queries; this is a failure of task transfer here, not a unique causal diagnosis.

These are separate stage times using historical cached matching. No matched complete-pipeline speed claim follows. No additional data/weights were downloaded.

## Evidence and scope

The [evaluation](evaluation.json), [independent audit](audit.json),27 image calibration records,24 pair records and four outer receipts retain failures, all pose hypotheses/native attempts, pixel scores, calibration losses and timings. [Manifest](manifest.json) binds transformed public bytes to original private file hashes. Public JSON is compact to reduce repository size.

Images, weights, feature/depth/normal arrays and upstream truth rows remain private. Predicted normal samples and repeated per-point cheirality SVD/depth/residual lists are replaced by explicit hashes; aggregate cheirality counts and every candidate pose remain. No selected branch or failed attempt is removed. Copied `*-seal.json` records bind the **original private operations**, not this transformed public folder; the publication manifest binds this folder.

The next decision must concern a materially different observable/task with a specific mechanism, verified closest prior work, strong controls and a cheap falsifier. This failure is not the user’s requested research success. The main substantial two-source, matched-cost improvement requirement remains unmet, and no paper is drafted.
