# 0031 completed development screen — fixed lookup closed

The fixed initial-pair lookup is closed. The saved prospective continuation clauses
are False, 2/4 parents (required at least 3), and False.
No repair head is admitted by this result. No novelty, whole-mask gain, independent
replication or final two-source +3 J/F evidence is established.

All 96 TRAIN anchor/horizon events, 87 unique future frames and
6144 current crossing queries remain, including fixed fallbacks.
The 96 ranking files were sealed before current TRAIN labels were decoded; all
96 diagnostic files and every candidate field are retained. Four final-frame
events are included. These correlated TRAIN events are not an official benchmark.

| Arm | Pooled joint loss (lower better) | Correctly oriented crossing fraction |
|---|---:|---:|
| paired | 0.4689534505 | 0.6536458333 |
| unpaired_max | 0.4805501302 | 0.6399739583 |
| deranged | 0.4630432129 | 0.6538085938 |
| native | 0.2585652669 | 0.8264973958 |

The decision uses equal-parent/event deltas, not a redefined pooled gate.

| Parent | Events | Empty events | Paired joint-loss delta from native | Paired correct-fraction delta |
|---|---:|---:|---:|---:|
| dog-agility | 24 | 0 | +0.2854003906 | -0.2929687500 |
| planes-water | 24 | 0 | +0.1271158854 | -0.0514322917 |
| dogs-scale | 24 | 0 | +0.3140055339 | -0.2656250000 |
| dog-gooses | 24 | 0 | +0.1150309245 | -0.0813802083 |

The separate TRAIN-selected global output threshold is 0.
All 241 saved native-zero-mask parity checks passed. Fixed DEV J/F/JF deltas
are {"J": 0.0, "F": 0.0, "JF": 0.0} in fraction units, with all 150 interior
frames (154 raw) retained. This is a calibration control, not a contribution.

Attempt01 remains failed: RuntimeWarning: divide by zero encountered in matmul; zero completed ranking
events, before current-label diagnostics. The fresh attempt02 uses explicit
float64 multiply/sum reduction after manufactured qualification; its scientific
contract and original source are unchanged. Exact historical cause is not proven.
The qualification recorded 400/400 warning-producing
matmul attempts and 0 successful matmul calls; a stored zero
max error for zero successful calls is not an accuracy result. The explicit
adapter passed 100 synthetic cases and all seven geometry tests.

| Operation | Status | Complete outer seconds | Peak sampled group RSS KiB |
|---|---|---:|---:|
| calibration | completed | 6.659425541002 | 259360 |
| screen01-failed | failed | 1.995777500037 | 296752 |
| arithmetic-qualification | completed | 1.357631499995 | 207728 |
| screen02 | completed | 8.659152417036 | 306016 |

Outer sum 18.671986958070s includes the failed screen.
The separate original manufactured unit command took 0.142272167024s.
Inner times overlap outer times and are not added again. Prior 0030 cache/model
construction costs are referenced by the bound metadata, not remeasured or free.

This export only reads text. No arrays, media, model, metrics or scientific
pipeline were rerun. JSON is compactly reserialized with every key/value retained
then literal task/home prefixes aliased; original raw and public byte hashes
identify the distinct representations. Source snapshots are deduplicated by
source identity; source origins, tests, failure logs, specs and receipts remain.
Embedded output hashes still address original bytes, as mapped by the manifest.
The two PR67 CI records are prior publication context, not CI for these private
development workers; their worker sources are bound by operation specifications.
Original prospective scope/admission notes are preserved unchanged.
