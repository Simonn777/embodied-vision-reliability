# 0019: author-recommended loss improves development accuracy

The one-setting control completed 160 native calls once, with zero failures and
all pre-final mask/statistic parity checks passing. Affine-depth AUC@5 improved
from 68.972945% to 74.071712%, exceeding fixed geometric E-normalized at 71.488081%.
This is a measured improvement from an existing author-recommended configuration.
It becomes a stronger baseline; no new-method, held-out or paper claim follows.

| Method | AUC5 (%) | AUC10 (%) | AUC20 (%) | Error <5° /80 |
|---|---:|---:|---:|---:|
| e_norm | 71.488081 | 82.362408 | 88.575262 | 72 |
| e_ts | 71.322710 | 82.265689 | 88.525076 | 72 |
| depth_scale | 62.383584 | 77.097927 | 85.932518 | 68 |
| depth_scale_truncated_cauchy | 72.293796 | 83.021898 | 88.999563 | 75 |
| depth_affine | 68.972945 | 80.716066 | 87.351192 | 72 |
| depth_affine_truncated_cauchy | 74.071712 | 83.976009 | 89.450049 | 74 |

Each truncated-Cauchy arm differs from its original arm only in final bundle
loss. Affine gains 5.098767 AUC5 percentage points over its own default and
2.583631 points over E-normalized. Scale gains 9.910212 points over its own
default and 0.805715 over E-normalized. Both retain every pair and improve AUC20.
Against E-normalized, affine has 14 material benefits and five material harms;
scale has 14 each. Neither introduces a new crossing from below 5° to at least
5°; affine rescues two and scale three. These are descriptive paired counts.

| Scene | E-normalized AUC5 | Scale TC AUC5 | Affine TC AUC5 |
|---|---:|---:|---:|
| florence_cathedral | 73.229039 | 76.418904 | 76.039985 |
| lincoln_memorial | 83.845102 | 84.185131 | 82.506712 |
| london_bridge | 68.643618 | 44.420677 | 67.510438 |
| milan_cathedral | 64.885804 | 75.069985 | 68.106202 |
| sagrada_familia | 74.544334 | 87.290164 | 82.994450 |

The scene table shows remaining failures: the scale arm remains much worse on
London. Affine exceeds E-normalized in three scenes and is worse in two. Pooled
all-pair DMS AUC is the frozen primary metric; scene AUCs are separately computed,
not averaged to replace it. No confidence interval treating image pairs as
independent observations is claimed.

Prediction outer wall time was 53.605883500 s (exit 0, sampled peak group RSS
92,480 KiB); saved-label evaluation was 0.617785042 s (exit 0, 52,976 KiB).
Charged worker times sum to 25.881831295 s for scale and 26.750576373 s for affine.
Historical baseline timing and concurrent lightweight source-development activity
do not establish a matched-time improvement. Cached frontend inference is absent
from both comparisons. Exact call and input costs remain in each record.

Scientific freeze: `542ca5ecee4ef9fc3316de8e6eae0a9d96429406`;
[exact CI passed before calls](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39327047725).
All 160 predictions were sealed before one saved-label score. No native retry,
parameter tuning, selected-case deletion or new asset acquisition occurred.
The original binary/data/0017 outputs and failed 0018 diagnosis remain unchanged.

Full [evaluation](evaluation.json), [prediction seal](predictions/seal.json),
[prediction receipt](prediction-process.json) and [evaluation receipt](evaluation-process.json)
are retained. [Independent saved-only review](independent-review/README.md) passes all 160
records and recomputes all six arms across every pair and scene. Maximum angular
discrepancy is 4.0502e-11 degrees; all AUC fractions agree within 1e-10.

The next fixed spatial-calibration-transfer candidate must exceed all ten
prespecified comparators, including these two stronger official arms. Its four
original candidate poses do not change. The unchanged independent-source,
matched complete-pipeline, novelty and replication requirements still precede
any manuscript.
