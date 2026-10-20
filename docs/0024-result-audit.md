# 0024 independent saved-output audit

The independent numerical and provenance audit **passed**. The frozen scientific diagnostic **failed all four gates**. These are different conclusions: the audit confirms the reported negative result, not an accuracy gain or a publishable contribution. The fixed variant is closed; this audit did not change thresholds, supports, views or outcomes.

The audit was prepared before reading probe outputs and executed once after the parent confirmed completion. It imported only the standard library and NumPy, without importing the experiment implementation, Torch or model code. No image decoding, inference, pose fitting or ground-truth access occurred.

## Independently reconstructed result

All 27 images and 54 cross-axis directions remain in the denominator, including unavailable fits and fixed fallbacks. The independently computed zero-offset error was **0.038210683549895416**, versus **0.038969546955092145** for the proposed offset: **−1.985998% relative gain**. Only **16/27** images had usable offsets.

| Development scene | Usable offsets | Relative gain | Strict improvement | Harm-floor pass |
|---|---:|---:|---|---|
| s00475 | 7/9 | −4.058311% | No | No |
| s00478 | 8/9 | −2.019977% | No | No |
| s00513 | 1/9 | +0.064816% | Yes | Yes |

Coverage, pooled gain, scene improvement and scene harm-floor gates all fail. The one positive scene does not satisfy the required two scenes, and its coverage is insufficient. These are predictor-transfer errors, not measured physical-depth or pose errors.

## Scope and numerical checks

Before numeric access, the auditor checked all 54 sealed JSON/NPZ file hashes and sizes, the selected image identities/order, source revisions, predecessor links, outer receipt and logs. It then checked 405 array hashes/shapes/dtypes and reconstructed projection, bilinear sampling, normalized positive-depth support, checkerboard partitions, both shared-offset fits, positivity/rank admission, constrained nuisance fits, held-out errors, fallback accounting and aggregation. Final evidence hashes were checked again after reconstruction.

The independent projection uses componentwise expressions and scalar separable interpolation. Shared offsets use QR nuisance elimination followed by a scalar Frisch–Waugh–Lovell solve, rather than the production five-variable SVD solve. The normalized design spectrum is independently checked. Numerical comparisons used predeclared coefficient tolerances `atol=2e-8, rtol=2e-7` and error tolerances `atol=1e-9, rtol=1e-7`, allowing differing QR/SVD and interpolation arithmetic under the admitted design condition bound. All classifications, counts and gates require exact equality. All **1,133 discrete** and **1,878 floating-point** comparisons passed; the largest discrepancy was **4.33e-6 of its allowed tolerance**.

The audit verified 135 recorded completed depth calls and zero recorded feature, matcher, native-estimator or GT operations. This is a check of recorded counters and five-view records, not independent hardware instrumentation. The probe's outer receipt records exit 0, **112.994561375 seconds**, peak sampled process-group RSS **2,032,256 KiB**, four-thread environment, no termination, no remaining descendants and no cleanup errors. The independent audit itself exited 0 in **2.201095875 seconds**, with empty stderr.

The audit consumes saved canonical intrinsics and rotations; it does not independently redo RGB remapping/canonical focal selection or validate learned inference. It verifies predecessor/source/hash links without re-traversing every installation, model or acquisition payload. Thus it establishes saved-result arithmetic and recorded provenance within this scope, not model correctness or task accuracy.

## Immutable references

Scientific freeze: `5d282a4b8b50094f433c6914931b005e7e16e73a`. The auditor checked the frozen source bytes and the following exact hashes; private paths below are relative to the chat's `work/0024-preparation/` directory. Raw caches and local audit artifacts remain private.

| Artifact | SHA-256 |
|---|---|
| `experiments/0024-virtual-depth/run.py` | `56db02536178762d540bef75feceef2eae8ea6a5fabac1f8d3bde2c1bbe53f1f` |
| `experiments/0024-virtual-depth/math_core.py` | `9e6f18fc1781c39e087ee8f40edfac79963fac5068ff4f99ec9b96b0897ae737` |
| `experiments/0024-virtual-depth/supervise.py` | `3e7124bfccf8642c40d212b62c1bb638fdd094e060e291fa0a8813f2f97f3bcd` |
| `protocols/0024-virtual-rotation-falsifier.md` | `4ae8f943112afa705a072e194e94a424252c282dd6beff54606584310007da3b` |
| Probe seal | `fee81b941fa0514f1e2affe4f66f704dd221595660957f169504e9feebcc598e` |
| Probe diagnostic | `2817d7352468f228bd7694222de35f6a37dc0364af05b7b67cbbfc06c3a8f62d` |
| Probe outer receipt | `09177a19f1676f0275cecb39202572ce3b2c42cbe9b7930b21cf2eef5b36f792` |
| `audit_probe.py` | `49e067fc6391f6794da7a473d37966fcea1697c72c97e5053daf64ab7b2c6128` |
| `audit-01/audit.json` | `edce5102acd602cbcd2b5ade0153f76824f62fdef8dc88d27d23d5feff9087df` |
| `audit-execution-01/receipt.json` | `11c0fdb53a29c1b490e11fb2ff34f29e715bfd17f2ee5563329c9bcea9669662` |

The audit JSON also retains all recomputed image analyses, complete checked source hashes and probe-log hashes. The execution receipt binds the auditor source and its stdout/stderr. Frozen scientific sources, protocol and original results were not modified. Byte-identical copies of the [audit result](../results/0024-development/independent-audit.json), [execution receipt](../results/0024-development/audit-execution-receipt.json), logs and [audit source](../experiments/0024-virtual-depth/audit_saved.py) are now public; their publication does not rerun the scientific probe.
