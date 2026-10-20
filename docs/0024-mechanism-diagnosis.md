# 0024: shared virtual-rotation offset did not transfer usefully

The fixed variant is **closed**. All 27 images and 135 depth forwards completed,
but all four decision components failed. The pooled normalized held-out MAE
increased from **0.038210684** to **0.038969547**, a **1.9860% increase**, rather
than the required reduction of at least 10%. Only **16/27** images had usable
offsets, below 19/27; one place also fell below its 5/9 minimum.

| Development place | Usable offsets | Relative MAE reduction (positive is better) |
|---|---:|---:|
| s00475 | 7/9 | −4.05831% |
| s00478 | 8/9 | −2.01998% |
| s00513 | 1/9 | +0.06482% |

Only one place improved, and two violated the 2% maximum-regression condition.
All 54 image-directions remain in the denominator. Twelve directions improved,
20 worsened, and 22 tied exactly through the prescribed unavailable-offset
fallback. Those ties are not successful calibrations. No success-only primary
score, selected-axis result or threshold adjustment is substituted.

## Failure location in the recorded computation

There were **no missing-median, common-support, rank or nonpositive-scale
failures**. Common support ranged from 1,918 to 1,932 of the 4,096 grid cells;
training and evaluation each retained 959–966 cells, above their 256 minimum.
All 54 axis fits passed the five-column excitation threshold and produced
positive scales.

Every one of the **11 unavailable images failed the full-image positivity
guard**: six yaw fits and eight pitch fits failed, with three images failing
both axes. Thus three images failed yaw only and five pitch only. Estimated
offsets would have made more than 1% of the already positive-valid base pixels
nonpositive. In the most extreme recorded axis fit, only 2.58% would remain
positive. This is incompatibility with the prespecified feasibility condition,
not a measurement of true scene depths. Neither dropping those images nor
clipping their corrected depths is an admitted repair.

Column-normalized singular ratios were:

| Axis | Minimum | Median | Maximum |
|---|---:|---:|---:|
| yaw | 0.001464 | 0.005604 | 0.006378 |
| pitch | 0.002070 | 0.007277 | 0.009165 |

All exceed the fixed `1e-5` cutoff; therefore that rank filter did not cause
the coverage failure. This does not prove absence of estimator sensitivity or
errors-in-variables bias. All 108 zero-offset nuisance fits and 64 evaluated
candidate nuisance fits had interior positive scales; their scale-zero or
rank-deficient boundary rules did not trigger.

## What the offsets and transfer errors show

Yaw/pitch offset disagreement, in units of each base map's positive-valid
median, ranged from **0.0683 to 7.0464**, with median **0.4517** across all 27
images. The two offsets had opposite signs in 14 images. Even among the 16
usable images, disagreement had median **0.2246** and maximum **2.5872**. These
are descriptive quantities, not new significance thresholds or uncertainty
estimates.

Both transfer directions worsened on their complete 27-image denominators:
yaw-to-pitch MAE increased **4.0238%**, and pitch-to-yaw increased **0.1830%**.
Consequently the negative result is not solely a coverage penalty: the usable
offsets also failed to improve the prescribed cross-axis prediction on average.
The fixed fallback contributes equal errors, not artificial candidate gains.

The findings reject this procedure's practical premise that these virtual
predictions supply a feasible, useful common scalar correction across axes on
the declared development cohort. They do **not** invalidate the algebraic
affine-depth relation under its exact assumptions. Disagreement and poor
transfer are consistent with axis-dependent or non-affine predictor response,
but this saved-record analysis cannot separate that explanation from
interpolation effects, spatial generalization error, shared base-map error or
OLS attenuation. No uniquely causal attribution is made. In particular, no
depth/pose GT was used and no physical accuracy improvement was measured.

## Scope, cost and disposition

This analysis reads only the saved `probe-01` per-image JSON records and sealed
diagnostic summary. It follows the private pre-outcome interpretation note;
there was no NPZ read, refit, model/native call, image/GT access or parameter
search. Summed per-image processing was **105.9512 s**, including **102.2082 s**
of depth-forward time and **0.1887 s** of numerical analysis. These sums are
constituent timings, not the enclosing operation's complete wall time or a
speed comparison.

The conditional fixed-shift pose API review remains unexecuted. No downstream
pose study, altered crop/angle/mask, new offset combination, subgroup rescue,
or paper is admitted from this result. All three places remain development
data and the failed 0023 study remains closed.
