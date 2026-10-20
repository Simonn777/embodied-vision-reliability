# Full GT-mask/presence memory write: TRAIN diagnostic

All 48 TRAIN anchors and both +1/+3 future contexts completed: 96 native future
forwards, 87 distinct (parent, frame) pairs, and no
image or memory encoder calls, optimization or new data acquisition. The four final-frame
events remain included and flagged. Repeated frames are separate anchor-context
events, not independent observations. All original rows and diagnostics are retained.

Equal-parent baseline J/F/J&F is
93.49997497/96.91835213/95.20916355%.
Cached full GT-mask/presence write replacements give
93.72841701/97.21202301/95.47022001%.
Paired changes are J +0.22844204,
F +0.29367088, J&F +0.26105646 percentage points.

| TRAIN parent | Baseline J (%) | Baseline F (%) | Full GT write J (%) | Full GT write F (%) | ΔJF pp |
|---|---:|---:|---:|---:|---:|
| dog-agility | 93.5960195 | 94.6408482 | 93.6570739 | 94.7262965 | +0.0732514 |
| planes-water | 95.7834186 | 99.2923011 | 95.8576899 | 99.3362105 | +0.0590904 |
| dogs-scale | 92.6308506 | 94.5899514 | 93.2668920 | 95.5023414 | +0.7742157 |
| dog-gooses | 91.9896113 | 99.1503079 | 92.1320123 | 99.2832436 | +0.1376684 |

Each intervention replaces one anchor's spatial memory with the cached native
GT-mask/presence encoded BF16 tensor. Other writes, pointers, positions and native
outputs are fixed. No family projection or magnitude bound is applied. The native
hard-mask encoding path and GT-presence embedding remain explicit confounds;
this is not a pure spatial-locality test.

Saved presence signs agree for 48/48 anchors; that does not
make the GT presence logits or hard-mask encoding identical to the baseline.
The full replacement norm divided by the original rho=.15 pre-BF16 family radius
is defined for 48/48 anchors: mean 0.7154725000321062, maximum 1.9809908434150205.
These are scalar summaries of stored diagnostics, not new tensor measurements.
Norms compare stored BF16 values lifted to float32 with the reference family
bound, not an unobserved prequantization GT tensor.

This is privileged TRAIN target-object, fixed-context horizon-set J/F. It uses
native selected post-presence masks and unchanged pixel metric formulas, but
includes the specified horizon endpoints rather than the official sequence
interior-only aggregate. It is not official all-object DAVIS, causal rollout,
DEV/held-out efficacy, an attainable causal-head result, or an upper bound on
possible improvement. Every parent is retained; no new gate is applied. The
observed change is small relative to the unchanged +3 J&F-point requirement on
each of two independent held-out sources. A small result here does not prove
that every alternative memory intervention must fail.

The unchanged source-only prior note in `docs/local-storage-prior.md` is included
with its original read-scope limitations; the exporter adds no literature claim.

Actual complete outer wall: 67.325347750 seconds; peak group RSS:
1159584 KiB. Inner wall: 67.042041250 seconds.
Inner and row timers overlap the outer cost and must not be added to it.
The original model state and cached histories/positions/pointers were unchanged.
The admission identifies the integration commit; the new diagnostic source is
separately bound by exact hashes, not asserted to have been covered by older CI.

Only UTF-8 source, scope, admission, specs, complete receipts/evaluation and raw
logs are exported. No PT/NPZ/PNG/JPEG, masks, labels, tensors, checkpoint or model
is opened by this publisher or included. Output NPZ hashes remain in receipts;
this export checks their recorded inventory, not their binary contents.

`manifest.json` records original and public byte lengths and SHA256 hashes.
The only transformations are ordered literal path-prefix substitutions:
task root to `$TASK`, then user home to `$USER_HOME`. Embedded original SHA fields
remain unchanged and refer to original bytes, not their sanitized public copies.
No unfavorable field or row is removed. README values are rendered from saved
aggregates only; no metric or scientific computation is rerun.
