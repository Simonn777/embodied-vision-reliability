# 0026 complete result and decision

**Measured:** all80 previously used development pairs, five scenes with16 pairs each. Source and protocol were frozen at `b596542f9eee399219ac03fdbcd91b33e8417fd6`; exact source CI39363242000 succeeded before the one prediction operation. No parameter changes or repeat fits followed outcome inspection.

## Accuracy and fixed gate

AUC here is the exact empirical hinge integral, `100 * mean(max(0, 1 - error/tau))`, using the maximum directed rotation/translation angle and the complete80-pair denominator. Old reference poses are recomputed on this metric; historical trapezoidal AUC values are not mixed into the table.

| Method | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| incumbent | 73.623633648 | 83.640200238 | 88.954997682 |
| continued_tc | 73.623633196 | 83.640200012 | 88.954997569 |
| G_local | 74.090760319 | 83.881063860 | 89.075429493 |
| F_local | 74.139670112 | 83.898218470 | 89.084006798 |
| G_lift | 73.588101055 | 83.622433941 | 88.946114534 |
| F_lift | 73.621231914 | 83.638999371 | 88.954397249 |
| F_all | 74.139670112 | 83.898218470 | 89.084006798 |
| G_all | 74.090760319 | 83.881063860 | 89.075429493 |
| historical_e_norm | 70.980845112 | 81.947165172 | 88.051941998 |
| historical_e_ts | 70.802478726 | 81.846385230 | 88.001959862 |
| historical_TC_scale | 71.717198346 | 82.733599173 | 88.507326879 |

The fixed primary is **G_lift**, not G_all. The pooled-best prespecified control is **F_all**. The primary loses0.551569057points at5degrees, improves in only2/5scenes and loses2.250535112points on London Bridge. Its0.137892264point AUC@20 loss passes the tail guard; the other three conditions fail. F_all and F_local agree within1.6e-12AUC@5points. The0.516036464point ordinary F_all improvement over incumbent is below the development target and is not the proposed new mechanism.

| Scene | G_lift minus fixed F_all, AUC@5 points |
|---|---:|
| florence_cathedral | -0.100450745 |
| lincoln_memorial | +0.348548835 |
| london_bridge | -2.250535112 |
| milan_cathedral | -0.756687956 |
| sagrada_familia | +0.001279692 |

## Coverage, resource use and independent check

All560 new slots remain in the records:451 completed,22 failed and87 unavailable. These sum across dependent proposals, not independent samples. Both lift arms have42 unavailable proposals; the candidate union keeps the original incumbent available. No filtered-subset claim is made.

External prediction wall:53.0001145seconds, peak sampled process-group RSS124080KiB. Evaluation:0.541215917seconds. Independent audit:0.532888167seconds. Every external operation exits0 with no surviving observed descendants or cleanup errors. Internal prediction outer also reports all80 complete and exit0.

G_lift charged-core sum is2.063433422seconds; F_all is5.700711379seconds. These include the declared prerequisite and attempt costs but exclude old frontend and historical baseline generation. They are not complete-pipeline or matched-time speed evidence. Common checkpoint time can overlap slot timing and must not be added twice.

The independent checker rehashes686 files and passes14,496 saved comparisons. It recalculates pose angles through atan2 rather than the production acos implementation, all eleven method AUCs, five-scene margins, selectors and charged-time sums. Maximum angle discrepancy is3.802e-11degrees; AUC discrepancy is3.536e-11points; charged sums agree exactly. Original tolerances remain1e-6degrees,1e-5AUCpoints and1e-9seconds.

This audit does not independently rerun optimization, recompute pixel-MSAC from input arrays, or prove global minima. It covers internal sealed process evidence; the separate external receipts and native build bindings are retained alongside it. Agent review is not independent-team replication.

## Decision

**CLOSED_FAILED.** No held-out acquisition, same-outcome tuning, new-method gain, manuscript admission or main-goal completion. A saved-output mechanism diagnosis may motivate a structurally distinct prospective test. All five scenes remain development forever.

[Complete evidence](../results/0026-development/README.md) · [Frozen protocol](../experiments/0026-calibration-moments/protocol.md) · [GitHub PR57](https://github.com/Simonn777/embodied-vision-reliability/pull/57)

[Saved mechanism diagnosis](0026-mechanism-diagnosis.md) identifies physical-rounding unavailability and redundant local minima. Its final alternative observation-model question is an unadmitted hypothesis, not the current selected experiment. [Final evidence review](0026-final-evidence-review.md) independently checks the external receipt and native-file chain.
