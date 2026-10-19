# Independent saved-only review of 0019

**PASS: the known author-loss control produces a real development-set improvement. This is not a new method or a paper-admission result.** Scientific freeze: `542ca5ecee4ef9fc3316de8e6eae0a9d96429406`.

The independent reviewer verifies the original 320-record bank, all 160 new records, source/library/manifest/input declarations, raw/final/process/log hashes, rotated schedule and both outer process receipts. All 160 new calls completed. Every pre-final mask and discrete statistic matches its original record exactly; all **320 floating pre-final statistics are bitwise equal**, with zero normalized discrepancy. Final poses are allowed to change. All 160 worker logs are empty; the two outer logs contain only the expected JSON progress/result records.

Errors were recomputed using scalar angle arithmetic and an independent SVD projection of the original reference rotations saved in 0017. The maximum discrepancy among all six arms and 80 pairs is **4.0502073917e-11 degrees**, below the fixed `1e-7`-degree audit tolerance. Directed and sign-folded summaries, all five 16-pair scene summaries and every material-benefit/harm comparison agree; AUC fraction tolerance is `1e-10`. No pair was dropped.

| Fixed method | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| E normalized Sampson | 71.488081 | 82.362408 | 88.575262 |
| E tangent Sampson | 71.322710 | 82.265689 | 88.525076 |
| Original depth scale | 62.383584 | 77.097927 | 85.932518 |
| Original depth affine | 68.972945 | 80.716066 | 87.351192 |
| Author-loss depth scale | 72.293796 | 83.021898 | 88.999563 |
| Author-loss depth affine | **74.071712** | **83.976009** | **89.450049** |

Affine author loss improves AUC@5 by **5.098767 percentage points** over original affine and **2.583631 points** over E normalized Sampson. It beats E in Florence, Milan and Sagrada, but loses in Lincoln and London. Against E it has 14 material benefits, 5 material harms, 2 rescues below 5 degrees and no crossings from below 5 to at least 5 degrees. Scale author loss improves 9.910212 points over original scale and 0.805715 over E; its London AUC@5 remains 44.420677% versus E's 68.643618%. Thus the aggregate gain does not imply uniform improvement.

| Scene | E AUC@5 (%) | Scale author loss | Affine author loss |
|---|---:|---:|---:|
| Florence | 73.229039 | 76.418904 | 76.039985 |
| Lincoln | 83.845102 | 84.185131 | 82.506712 |
| London | 68.643618 | 44.420677 | 67.510438 |
| Milan | 64.885804 | 75.069985 | 68.106202 |
| Sagrada | 74.544334 | 87.290164 | 82.994450 |

Prediction and evaluation ran once each, both exit 0: outer times **53.605883500 s** and **0.617785042 s**; sampled process-tree peaks 92,480 and 52,976 KiB. Per-worker charged totals sum to 25.881831295 s for scale and 26.750576373 s for affine, including imports/binding/input/output overhead. Native API time sums are 2.180665085 s and 2.918376750 s respectively. Historical timings are not a matched-compute comparison, and sampled RSS can miss peaks.

[audit.json](audit.json) contains hashes, independent arithmetic and the complete scene/comparison tables. [execution.json](execution.json) preserves actual commands, exit codes and separately measured subprocess wall times: three artificial helper tests passed in a 0.074318459 s subprocess; the saved-only audit passed in 0.607352583 s. See [tests.log](tests.log), [audit.log](audit.log), [source](../../../experiments/0019-review/review_saved.py), [sealed predictions](../predictions/seal.json), [evaluation](../evaluation.json), [prediction receipt](../prediction-process.json) and [evaluation receipt](../evaluation-process.json).

No HDF5, images, private native binary or estimator was accessed, and no scientific predictor/evaluator module was imported. Input arrays and coordinate-dependent residual diagnostics remain hash-bound saved evidence, not independently rerun measurements. All five landmark scenes are permanently development data with dependent pairs. The experiment strengthens the required baseline; it does not establish novel research progress or held-out generalization.
