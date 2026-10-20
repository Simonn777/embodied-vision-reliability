# Saved teacher target task alignment: TRAIN diagnostic

All 48 TRAIN anchors and both +1/+3 future contexts completed: 96 native future
forwards, 87 distinct (parent, frame) pairs, and no
image-encoder calls, optimization or new data acquisition. The four final-frame
events remain included and flagged. Repeated frames are separate anchor-context
events, not independent observations. All original rows and diagnostics are retained.

Equal-parent baseline J/F/J&F is
93.49997497/96.91835213/95.20916355%.
Saved multistep teacher targets give
93.71958745/97.35394410/95.53676577%.
Paired changes are J +0.21961248,
F +0.43559196, J&F +0.32760222 percentage points.

| TRAIN parent | Baseline J (%) | Baseline F (%) | Teacher J (%) | Teacher F (%) | ΔJF pp |
|---|---:|---:|---:|---:|---:|
| dog-agility | 93.5960195 | 94.6408482 | 93.7291929 | 95.0378061 | +0.2650657 |
| planes-water | 95.7834186 | 99.2923011 | 95.8011807 | 99.2680057 | -0.0032666 |
| dogs-scale | 92.6308506 | 94.5899514 | 93.0747973 | 95.7839677 | +0.8189815 |
| dog-gooses | 91.9896113 | 99.1503079 | 92.2731788 | 99.3259968 | +0.2296282 |

This is privileged TRAIN target-object, fixed-context horizon-set J/F. It uses
native selected post-presence masks and unchanged pixel metric formulas, but
includes the specified horizon endpoints rather than the official sequence
interior-only aggregate. It is not official all-object DAVIS, causal rollout,
DEV/held-out efficacy or evidence that the selected teacher target is learnable.
Positive and negative parent results are all retained. No new gate is applied.

Actual complete outer wall: 67.937473417 seconds; peak group RSS:
1207968 KiB. Inner wall: 67.712706959 seconds.
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
