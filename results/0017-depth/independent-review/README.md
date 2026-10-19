# 0017 independent saved-result audit

The first authorized saved-only audit passed on 2026-10-19: all 80 pairs and all 320 arm records were retained. The eight artificial tests passed before the audit. No private HDF5 file, image, native estimator, or production scoring module was opened or imported by the audit.

The independent implementation checks the sealed manifest and source bindings, all 1,281 prediction/raw-worker/process/log/integrity files, fixed arm ordering and options, input agreement across arms, returned-mask encoding, failure classification, and process cost accounting. It independently reconstructs the saved reference rotation projection, scalar rotation and directed/folded translation errors, trapezoidal AUC at 5/10/20 degrees, full-cohort and scene denominators, leave-one-scene-out summaries, selection rules, and prospective decisions.

| Arm or diagnostic | Independently recomputed AUC@5 (%) |
|---|---:|
| Geometric normalized Sampson | 71.488080918 |
| Geometric tangent Sampson | 71.322709510 |
| Known depth scale | 62.383583584 |
| Known depth affine | 68.972945481 |
| External pixel-MSAC selection | 71.520213336 |
| Geometry oracle | 71.640729831 |
| All-arm oracle | 80.843254219 |

The all-arm oracle exceeds the geometry oracle by **9.202524388 percentage points**, satisfying only the predeclared necessary headroom condition. The deployable pixel-MSAC selection gains **0.032132418 points** over normalized Sampson. No new method improvement or manuscript admission is established.

Using the sealed counts, the omission-mechanism condition occurs for the scale arm in Lincoln Memorial and London Bridge; the affine arm meets it only in London Bridge. Thus only the scale arm meets the predeclared two-scene trigger. This is a source-grounded diagnostic association, not proof that omitted residuals caused the pose errors.

The aggregate charged outer-process time is 57.987685621 seconds across all four arms. The selector is charged for all four arms. There were no independently reconstructed oracle identity discrepancies due to floating-point roundoff.

## Limits

Pixel-MSAC values and depth inactivity/reprojection counts are hash-bound saved inputs. Their coordinate-level formulas cannot be recomputed without the private correspondences, which this audit intentionally does not access. Reference poses are independently processed from the saved evaluation JSON rather than reread from the cache. Native library identity is checked against its pinned saved configuration; the binary itself is not reopened.

The artificial fixture initially placed a strict one-degree harm comparison exactly on a floating-point boundary. Only that synthetic fixture was moved away from the boundary before the eight-test pass; real-audit tolerances and decisions were unchanged. The actual saved audit ran once and passed without repair.

## Receipt

```text
python -m unittest discover -s experiments/0017-review -p test_review_saved.py -v
8 tests, OK (reported unittest duration 0.787 s)

python experiments/0017-review/review_saved.py --repo <repo> \
  --manifest configs/0017-development-manifest.json \
  --seal results/0017-depth/predictions/seal.json \
  --evaluation results/0017-depth/evaluation.json \
  --output results/0017-depth/independent-review/audit.json
exit code: 0
tool command wall time: 0.359234459 s
stdout: {"status": "passed", "pair_count": 80, "arm_record_count": 320}

review source SHA256:
529b59d3234d3047ee60ba07406150ca478b5c74e8e8324a24a0a1d283b0aa00
audit.json SHA256:
83cbfcfa0a3be6d480497d93783281aaf3493fdc3d2409861eeb16969f3792ae
```

`audit.json` contains the full recomputed summaries, decisions, cost totals, file hashes, and explicit scope limitations. The review source and artificial tests are under `experiments/0017-review/` and do not modify the frozen production implementation.
