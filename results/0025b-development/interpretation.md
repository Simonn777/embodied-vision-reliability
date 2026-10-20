# 0025b saved-results interpretation

This is a post-result descriptive reading of the complete fixed diagnostic, not a new gate, fit, method, or pose experiment. Only saved metadata and the 64 case JSON records were read for this note. No images, NPZ arrays, models, or fitting pipelines were executed. All 64 case JSON hashes still match the result seal; prediction/evaluation receipt and outer hashes match the successful independent audit, and all three outer log hashes remain valid.

Freeze: `13d4b3259f8d12a1ba4d5d0de9ff56ec6527cf62`. The first independent audit passed 64 cases / 256 arm records with no discrepancies; maximum numeric difference was `1.0455525334407412e-13`. The unchanged preregistered gate passed. Its conclusion is only permission to propose a separate pose falsifier.

## What passed, and what remained worse

Errors below are percentages of held-out mean AbsRel. Tail columns are quantiles **over case means**, not pooled pixel quantiles. The 64 cases reuse 16 source images and contain only two development sequences; they are not 64 independent scenes.

| Arm | Available | Mean | Median | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| RS row affine | 64/64 | 3.1351 | 2.7684 | 6.0729 | 7.1213 | 9.4071 |
| GS transported affine | 64/64 | 1.6845 | 1.2314 | 2.9358 | 4.3922 | 4.8982 |
| RS middle affine | 63/64 | 3.1012 | 2.6391 | 5.7044 | 7.1228 | 9.3914 |
| RS row scale only | 64/64 | 4.0456 | 3.5637 | 7.5737 | 8.1621 | 9.5735 |

The RS row scene medians were **2.93645% living room / 2.29345% office**, versus GS transported **1.89201% / 1.15471%**. The required median of paired RS-minus-GS differences was **+0.973455 / +1.097963 percentage points**, not the difference between marginal medians. RS was worse in **56/64 cases**, including 28/32 in each sequence. Passing the bounded-deterioration gate therefore does not establish an improvement over GS.

Ten RS row cases had mean error above 5%; none exceeded 10%. The worst was office frame 15, `rs_y_positive`: 9.40710% versus its GS control 2.82063%. The median of each case's held-out pixel P95 was 6.92141% for RS row, and the largest case-level pixel P95 was 23.1260%; these are descriptive tails, not additional success criteria. There were 112,985 held-out evaluations across the reused cases. Common-support coverage ranged from 59.7412% to 99.6094% of each fixed lattice, with mean 86.2061%.

## Row-time semantics and the offset

The alternative middle-frame interpretation remained competitive. On the 63 jointly usable cases, RS-row minus RS-middle had median **+0.0361641 pp** and mean +0.0348835 pp: middle was better in 40 cases, row in 23. The median across cases of the physical row-versus-middle mean relative separation was only **1.08046%**. A two-parameter oracle fit can absorb much of this small separation. These observations do **not** identify a unique physical anchor time for MoGe depth.

The sole unavailable middle arm was **office frame 10, numeric stem 251, `rs_y_positive`**. Its two-column design had rank 2 and singular values `[1.4138065763771404, 0.03392586907873014]`, but the unconstrained training solution was `a=-0.0023212108861602613`, `b=2.441192955028255`. It failed the fixed positive-scale rule; its 1,842 held-out points were not evaluated. The 63-case middle summary is explicitly partial coverage and must not replace the complete primary comparison.

Adding the row-model offset reduced held-out mean AbsRel in **55/64 cases**; the paired scale-only-minus-affine median improvement was **0.307094 pp** (mean 0.910522 pp), with nine regressions. The offset's fitted sign was positive in 33 cases and negative in 31. This supports allowing affine nuisance parameters in the next falsifier, but only as an **oracle GT-assisted model-capacity observation**. It does not show those parameters can be estimated from noisy image matches, nor that a new solver improves pose.

## Consequence for a prospective 0025c

There is no result-based reason to veto a separately frozen, bounded within-readout translation falsifier. There is also no basis to treat translation, true sensor RS, or downstream pose as already qualified. The current experiment used pure rotation, GT-assisted checkerboard calibration, a common smooth-depth support, and no feature matching. Keypoints near discontinuities may occupy exactly the regions excluded by the four-neighbor GT spread test; these results cannot justify dropping such matches or failed cases from a later pose benchmark.

Before a task-level comparison can mean anything, its new design must address depth-dependent visibility/disocclusion in translation rendering, real predicted correspondences rather than GT pairings, and identical measured-rotation/angular-motion information for the candidate and its inertial baselines. The known-R conditional four-point claim must not be presented as solving unknown rotation. RS-row and RS-middle ambiguity should remain a limitation rather than be resolved by selecting whichever interpretation later scores best. Current numerical conditioning evidence and oracle-depth fit errors do not establish noise robustness of the minimal solver.

This note changes no frozen parameter or gate. It establishes neither pose gain nor a new-method contribution and admits no manuscript. Prediction recorded 80 repeated CPU-autocast warnings that explicitly disabled unsupported autocasting; these remain in the original receipt. Evaluation recorded no warnings. Clean outer times were 72.0348 s prediction, 6.79348 s evaluation, and 0.902562 s independent audit.

Evidence hashes: aggregate `e03770d013f10096df878a3b85ef9d7874f62e5af824a9382282bbe2ccd840ec`; result seal `ca463fd267ee15b58c879db95fc84edcb2a41af28dbd9c2adc725d87c7487701`; prediction seal `bf3ceea0d07d485175f15cb217b864ae340d4c5e5abf890d5b079c919bc27a06`; successful audit `9ff75107f211bda01c8a8026413707ebedda48f7a33ca5b99ff00c3f38404282`.
