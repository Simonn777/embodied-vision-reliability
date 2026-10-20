# 0025c admission: necessary initializer-noise falsifier

The [0025b physical depth approximation](../results/0025b-development/README.md)
passed a tolerance test, but did not improve pose and its GS oracle was better.
The four-point affine rolling-shutter system also has poor conditioning and
known critical motions. Before spending on rendering or an IMU-data pipeline,
test whether the initializer survives a small, fixed errors-in-variables probe.

The [prospective protocol](../protocols/0025c-rs-initializer-noise.md) fixes four
manufactured motions, eight physical points, one clean record and 16 paired noise
realizations per motion. P/S share pixels and depth errors; S additionally
perturbs angular input parameters. These are not measured IMU streams. No new
scientific cohort or benchmark solver has run at this source preparation.

Every four-point subset is retained (70 per record), together with all 28
two-point angular-only subsets identified as a stronger small-sample control
before outcomes, alongside the identical
eight-point depth system, an exact-angular-integration depth-free RS system and
an angular-only RS affine control. All have the same observations and angular
inputs. There is no truth-based selection, scorer, refiner, outlier experiment,
renderer, model call, real data acquisition or new checkpoint. Unavailable
solutions count as 180-degree errors. No folded translation-sign metric.

Each of six noncritical motion/input groups must yield at least 144 useful
four-point initializers among 1,120 attempts at a directed five-degree threshold.
At least one perturbed-angular-input motion must also satisfy
`D4_useful_count > 10*A2_useful_count`, the exact integer form of optimistic
useful-proposal headroom over the two-point control. This is only a necessary
screen under a hypothetical later 50% correspondence inlier rate; it is not observed robust-estimation advantage. A failure closes
this fixed four-point initializer under the declared contract. A pass admits
only a separately frozen robust-estimation proposal. Neither outcome admits a
manuscript or completes the research goal.

Sources, complete noise draw order, arithmetic tolerances, positivity, directed
sign handling, controls and failures are fixed before the first execution.
Artificial tests use independent invented data, without generating the target
cohort. Independent source review and exact-commit CI must pass before one
supervised run and saved-output audit. Raw matrices, all attempts, timing and
source bindings are preserved. The two-source held-out improvement gate remains.

The [previous publication](0025b-publication-verification.md) is verified.
