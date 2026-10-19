# Saved-only characterization of the failed 0015 configuration

The frozen expansion decision remains **FAIL**: full AUC@5 changes
**+0.009558333 percentage points**, below the required +0.25; subset AUC@5
changes **−3.091052179 points**, below the required +2. All16 pairs remain in
both scores. This analysis changes no gate, fit, prediction or score.

[analysis.py](analysis.py) reads only the complete photometric seal, its16
hash-verified JSON records, and [the saved repaired evaluation](../evaluation-repair.json).
It checks ordering, counts, failed-coordinate preservation and evaluation/seal
binding. It imports no estimator, scientific scorer, HDF5 or image decoder.
[summary.json](summary.json) records every input hash, source hash, per-pair
counts/distributions, the191 admitted-query summaries and the original gate/AUC
objects without recomputing them. It was run once with:

```sh
python results/0015-photometric/descriptive/analysis.py
```

## Coverage and failure reasons

Of1,355 selected queries,24 failed affine initialization,1,140 failed the
photometric stage, and191 were admitted: **14.10% of selected queries**.
All191 changed the target coordinate. Exactly191 of the160,000 full-arm
correspondences changed, or **0.119375%**. Failed queries retain their original
coordinates. The initializer failures comprise19 degenerate/reflected affine
maps and5 neighborhoods with fewer than eight inliers.

The1,331 patches passed to the kernel split into93 initial target-patch boundary failures,
1,047 optimized but rejected fits, and191 admitted fits. Their final rejection
reasons are exhaustively partitioned below:

| Final recorded reason | Fits | Fraction of1,140 photometric failures |
|---|---:|---:|
| 30-attempt exhaustion **and** excessive relative GN decrease |987|86.58%|
| 30-attempt exhaustion alone |44|3.86%|
| Excessive relative GN decrease alone |16|1.40%|
| Initial target-patch boundary failure |93|8.16%|

Thus the recorded final reasons predominantly concern the fixed optimization
budget and local stationarity:1,031/1,140 failures reach the attempt cap
(90.44%);1,003/1,140 fail the GN criterion (87.98%). These labels overlap in987
fits. No recorded final rejection is due to normalized-curvature conditioning.
The44 cap-only failures still fail the fixed stopping rule; they are not
retroactively admitted. The saved trajectory cannot establish that additional
iterations would converge to correct correspondences, so this does not authorize
an iteration-budget rescue.

Parameter bounds also reject trial steps, but these events are distinct from
the final failure reason. Among the1,140 failed fits,253 encountered a gain
guard,91 a displacement guard,57 a relative-affine guard,2 a determinant guard,
and1 a target-boundary guard during iteration. These counts overlap; the initial
93 boundary failures have no trial history. There are4,027 rejected gain steps,
1,191 displacement steps,828 relative-affine steps,3 determinant steps and18
target-boundary steps among failed fits, compared with7,582 cost-rejected steps.
These co-occurring events do not identify a causal parameter bound to loosen.

## Admitted patch losses and coordinate updates

The191 admitted patches all reduce their saved photometric objective.
The loss is0.5 times the sum of squared intensity residuals over961 samples;
affine, gain and bias parameters contribute to its decrease as well as target
translation. The reduction does not isolate improved correspondence location.

| Quantity across191 admitted fits | Mean | Median | 10th–90th percentiles |
|---|---:|---:|---:|
| Initial patch loss |8.467211|5.942167|0.788151–18.352306|
| Final patch loss |0.978318|0.459311|0.080471–2.470483|
| Per-patch relative loss reduction |79.3502%|89.4520%|44.7404%–98.8689%|
| Applied target displacement, pixels |0.504424|0.407407|0.115736–1.010539|
| Attempt count |16.1047|14|8–27|

Summed patch loss falls from1,617.237279 to186.858670, a pooled88.4458%
reduction. Applied displacement ranges from0.027163 to3.176924px. These are
conditional descriptive values for the admitted set, not performance measures
for rejected queries or calibrated uncertainty estimates.

## All16 pair outcomes

Angles below are the saved maximum of rotation and directed translation error
in degrees. Change is **refined minus original**, so a negative number improves.
The four unavailable subsets retain180° in both scores; the dash avoids treating
equal penalties as evidence of equal predictions. Pair IDs map to full image
pair names in the JSON. No pair is excluded.

| Pair | Selected | Affine init failed | Photo failed | Admitted | Subset status | Subset original | Subset refined | Subset change | Full change |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
|00|89|0|67|22|available|8.982024|7.098590|−1.883434|+0.000227|
|01|76|0|62|14|available|0.419482|2.114530|+1.695048|−0.000542|
|02|86|1|73|12|available|3.624630|2.545426|−1.079204|−0.000398|
|03|46|0|43|3|unavailable|180.000000|180.000000|—|−0.000126|
|04|130|0|107|23|available|3.461432|3.411989|−0.049443|−0.000882|
|05|92|1|79|12|available|1.172836|1.317974|+0.145138|−0.002247|
|06|138|4|119|15|available|0.447994|1.193719|+0.745725|+0.000391|
|07|95|1|85|9|available|2.977232|1.997811|−0.979421|+0.000150|
|08|106|2|84|20|available|0.902969|0.704018|−0.198950|+0.000503|
|09|81|2|69|10|available|1.401873|5.094804|+3.692931|−0.000072|
|10|31|2|27|2|unavailable|180.000000|180.000000|—|+0.000078|
|11|63|0|60|3|unavailable|180.000000|180.000000|—|−0.004948|
|12|102|6|83|13|available|0.686955|1.782922|+1.095967|+0.000108|
|13|93|5|87|1|unavailable|180.000000|180.000000|—|−0.000046|
|14|79|0|59|20|available|1.542599|1.426560|−0.116039|+0.000218|
|15|48|0|36|12|available|13.158894|2.509574|−10.649321|−0.000123|

Among the12 available subsets,7 improve and5 worsen by more than1e-9degree.
Their descriptive median change is−0.082741° and mean change−0.631750°;
the largest improvement is−10.649321° and largest harm+3.692931°. These summaries
do not replace the failed all16 AUC@5 comparison: mean angular change and a
threshold-limited AUC weight these outcomes differently. The full arm improves
9/16 pairs, below the required10, with very small changes throughout.

These are dependent queries and image pairs from one already-used scene.
The record supports a narrow failure characterization: low admission, frequent
budget/stationarity failures, strong conditional photometric loss reduction,
and no useful pose improvement under the fixed sparse configuration. It does
not establish a predictable bias field, shared-noise covariance, a causal
failure explanation, population significance, novelty or paper readiness.
