# 0020: spatial calibration transfer fails the development gate

The frozen selector completed all 80 pairs once and achieved **71.107107%
AUC@5**, below the strongest fixed comparator, author-recommended affine-depth
truncated Cauchy at **74.071712%**. The difference is **-2.964605 percentage
points**. Random-point transfer and global calibration also outperform the
proposed spatial transfer. This fixed formulation is closed; no fold, weight,
threshold, fit rule, candidate bank or failed-case policy will be retuned on
these outcomes. No manuscript is admitted.

| Method | AUC5 (%) | AUC10 (%) | AUC20 (%) |
|---|---:|---:|---:|
| Proposed spatial transfer | 71.107107 | 82.295167 | 88.584304 |
| Geometric pixel MSAC selector | 71.520213 | 82.371613 | 88.579865 |
| Global affine LS selector | 72.569953 | 83.139548 | 88.999627 |
| Spatial weighting only | 71.538068 | 82.451197 | 88.600401 |
| Random-point transfer | 72.926812 | 83.317978 | 89.088842 |
| Fixed E-normalized | 71.488081 | 82.362408 | 88.575262 |
| Fixed E-ts | 71.322710 | 82.265689 | 88.525076 |
| Fixed depth-scale default | 62.383584 | 77.097927 | 85.932518 |
| Fixed depth-affine default | 68.972945 | 80.716066 | 87.351192 |
| Fixed depth-scale author loss | 72.293796 | 83.021898 | 88.999563 |
| Fixed depth-affine author loss | 74.071712 | 83.976009 | 89.450049 |

All ten mandatory comparators remain in the table. The gate's same aggregate
AUC5 comparator is fixed depth-affine author loss. The proposal exceeds it only
in Florence (one of five scenes, below the required two), and loses 0.865745
AUC20 percentage points against the best AUC20 comparator (above the allowed
0.5-point loss). All three accuracy conditions fail; full coverage and technical
resource checks pass. Every scene and pair is retained in [evaluation.json](evaluation.json).

| Scene | Proposed AUC5 | Strongest aggregate comparator AUC5 |
|---|---:|---:|
| florence_cathedral | 76.699786 | 76.039985 |
| lincoln_memorial | 82.357214 | 82.506712 |
| london_bridge | 61.929662 | 67.510438 |
| milan_cathedral | 66.317848 | 68.106202 |
| sagrada_familia | 74.131357 | 82.994450 |

The proposal selects E-normalized / E-ts / depth-scale / depth-affine on
25 / 19 / 13 / 23 pairs. Against the strongest comparator it has nine material
benefits and twelve material harms under the frozen one-degree-or-5-degree-
crossing definition. Against global affine LS, it changes fourteen choices:
seven strictly better and seven worse, with one material benefit and three
material harms. These are descriptive outcomes, not independent-sample tests.

All 80 spatial partitions are available. The full bank uses exactly 3,200 fits;
2,794 complete and 406 retain full validation cost because the fitted slope is
nonpositive. Unavailable fits by control are 37 global, 40 spatial-weighted
global, 163 random-fold and 166 spatial-fold. None is dropped or retried. These
counts do not identify a causal failure mechanism: actual geometric-support
counts and masses differ between folds, as declared in the protocol.

The unique prediction operation exited zero in **28.911574541 seconds**, with
sampled peak process-group RSS **94,480 KiB**. Evaluation exited zero in
**1.110610208 seconds**, peak **74,992 KiB**. Charged scoring workers total
27.258136580 seconds; generating all four original candidates historically cost
57.987685621 seconds. Cached frontend inference is absent, and historical costs
are not a matched complete-pipeline comparison. No efficiency improvement is
claimed.

Scientific source freeze: `bec376568729c3a158cd9f1279cd00d3e5f1de76`.
[Exact CI39329346633 passed](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39329346633)
before scoring. All choices, scores, fits, partitions and receipts were sealed
before the single saved-label evaluation. No native estimator, new model,
image download or pose-label read inside scoring occurred. The original 0017 and
0019 predictions are unchanged.

[Independent saved-only review](independent-review/README.md) passes all 80
record/process chains, 3,200 fits, choices, partition counts and all 11 methods'
aggregate/scene metrics. It independently confirms all three accuracy conditions
fail. The reviewer verifies saved score arithmetic and forced-invalid mass bounds;
it does not recompute successful-fit depth residuals from raw arrays. Five
artificial reviewer tests pass; the actual audit exited zero in 1.617221875 seconds.

This negative result closes the specific spatial-transfer selector. It does not
establish that all depth transfer is impossible. Further work requires a new,
independently justified source of information and a separate prospective design;
the substantial held-out improvement and novelty requirements remain unchanged.
