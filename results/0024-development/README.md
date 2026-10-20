# 0024: virtual-rotation offset transfer failed

**CLOSED_FAILED.** The fixed candidate produced usable offsets for 16/27 images
and increased the all-image cross-axis residual by **1.985998%**. All four
prospective conditions failed. This is a predictor-consistency diagnostic, not
a measurement of depth or pose accuracy. No downstream pose experiment or
manuscript is admitted by this result, and the broader research goal is unmet.

| Development place | Usable images | Zero-offset mean error | Transferred-offset mean error | Relative reduction |
|---|---:|---:|---:|---:|
| All | 16/27 | 0.038210684 | 0.038969547 | −1.985998% |
| s00513 | 1/9 | 0.046470384 | 0.046440263 | +0.064816% |
| s00475 | 7/9 | 0.045618648 | 0.047469995 | −4.058311% |
| s00478 | 8/9 | 0.022543019 | 0.022998383 | −2.019977% |

Errors are held-out-grid mean absolute residuals in units of each base depth
map's positive-valid median, averaged over **all 54 image-directions**. Both
methods use identical support and two fitted nuisance parameters per evaluated
virtual view. Unavailable offsets retain their baseline errors; no image or
direction was removed. All three places remain development data forever.

The [frozen protocol](../../protocols/0024-virtual-rotation-falsifier.md) required
at least 19/27 usable images and 5/9 in every place, at least 10% pooled reduction,
at least two improved places, and no place worse by over 2%. Actual coverage
failed, pooled error increased, only one place improved, and two violated the
scene floor. The fixed angle, crop, solver, positivity/rank thresholds and
population will not be tuned on these outputs to rescue this variant.

## Execution and provenance

Source: `5d282a4b8b50094f433c6914931b005e7e16e73a`.
[Exact source CI 39346730569](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39346730569)
passed before launch, including the 40 artificial checks. One supervised
operation completed all 135 planned depth forwards on 27 existing RGB images.
There were zero feature, matcher and native pose calls, zero GT reads, and zero
new downloads or training. Output sealing preceded the aggregate decision.

Complete enclosing operation: **112.994561 seconds**, sampled peak process-group
RSS **2,032,256 KiB**, four CPU threads, clean exit/descendant cleanup and no
recorded offline violation. This includes verification, model loading, all
warps/predictions, analysis and serialization. It is not a comparative speedup
measurement. The Python audit guard is not an OS native-syscall sandbox.

- [Exact diagnostic](diagnostic.json) and [numeric table](metrics.csv).
- [All 27 generated per-image analyses](image-analyses.json), with both axes and failures.
- [Inner receipt](probe-receipt.json), [supervision](supervision-receipt.json),
  [launch and prior CI evidence](launch-receipt.json), and [output seal](private-output-seal.json).
- [Publication transformations/hashes](publication-bindings.json).
- [Independent result audit](../../docs/0024-result-audit.md) and
  [descriptive mechanism diagnosis](../../docs/0024-mechanism-diagnosis.md).
- [Independent recomputed numbers](independent-audit.json),
  [audit execution receipt](audit-execution-receipt.json), and
  [byte-identical independent audit source](../../experiments/0024-virtual-depth/audit_saved.py).

Original depth/mask/intrinsics arrays, original RGB and private full image
records remain local. Public material includes generated summaries, fitted
coefficients, timings and hashes; local path prefixes are explicitly redacted.
No RGB, depth array or GT payload is rehosted. Affine depth equations and
virtual-view consistency have close prior work; the result establishes no new
solver, physical calibration, independent-domain gain or CVPR contribution.
