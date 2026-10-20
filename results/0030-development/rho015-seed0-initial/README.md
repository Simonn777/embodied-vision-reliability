# First 0030 development configuration: completed evidence snapshot

rho=0.15, seed=0. All recordings are permanently exposed development data.
The teacher completed all 48 TRAIN anchors. Its mean fixed-context surrogate
loss was 0.018459060 before correction and
0.017412603 at its selected state. Selection used TRAIN
loss; this is an optimization diagnostic, not accuracy or generalization.
All 48 selected writes changed after BF16 storage. The three KD heads completed
288 regression updates each; their objectives are target-fitting diagnostics.

The native causal rollout completed 154 DEV frames with
152 exact noninitial native parity checks and no learned writes.
Saved equal-parent J/F/J&F are 94.7651825/89.5892584/92.1772205%.
This covers initial target ID 1 only, excluding first/final frames from the
means: 150 included of 154 raw frames. It is not an official all-object result.
The completed multistep head's own rollout has saved J/F/J&F 94.2441723/89.1234051/91.6837887%, a J&F difference of -0.4934318 percentage points versus native. Parent J&F differences are cat-girl -0.1743853 and upside-down -0.8124782 points.

The completed one_step head's own rollout has saved J/F/J&F 94.5860235/89.3969181/91.9914708%, a J&F difference of -0.1857496 percentage points versus native. Parent J&F differences are cat-girl -0.3086098 and upside-down -0.0628895 points.

The completed gt_memory head's own rollout has saved J/F/J&F 94.8480678/89.9341811/92.3911244%, a J&F difference of +0.2139040 percentage points versus native. Parent J&F differences are cat-girl +0.0295183 and upside-down +0.3982897 points.

Multistep and one-step show no gain. The GT-memory head has a modest positive DEV result; it does not demonstrate a benefit from multistep future optimization and does not meet the paper gate. Direct controls are outside this snapshot; this does not close the whole development campaign.

| Complete operation | Actual outer wall seconds |
|---|---:|
| teacher-rho015-01 | 1162.078428 |
| fit-multistep-rho015-seed0-01 | 6.431160 |
| fit-one_step-rho015-seed0-01 | 4.884038 |
| fit-gt_memory-rho015-seed0-01 | 4.810073 |
| rollout-native-rho015-seed0-01 | 108.974859 |
| evaluate-native-rho015-seed0-01 | 1.067922 |
| rollout-multistep-rho015-seed0-01 | 113.125211 |
| evaluate-multistep-rho015-seed0-01 | 1.096583 |
| rollout-one_step-rho015-seed0-01 | 113.190952 |
| evaluate-one_step-rho015-seed0-01 | 1.099869 |
| rollout-gt_memory-rho015-seed0-01 | 112.935254 |
| evaluate-gt_memory-rho015-seed0-01 | 1.094718 |

These are actual executed operation costs. Inner/anchor timers overlap and must
not be added. Shared baseline cache cost and the original failed evaluation are
referenced in `baseline-bundle-reference.json`; failures are not removed from
development accounting. Complete same-information and cost-matched controls,
further development, and the unchanged two-source held-out +3 J&F gate remain
outstanding.

Each included operation has its original inner receipt, worker/outer specs,
outer receipt and raw logs, with only the literal path-prefix substitutions
declared in `manifest.json`. Original SHA fields are retained unchanged and
refer to original bytes; a redacted public file has its own manifest SHA.
No JSON fields are removed. Source maps are recorded per operation; the
integration note explains the inherited historical expected_commit field.

No PT/NPZ, checkpoint, model, RGB, mask or annotation file is included.
No started receipt is copied. The saved-summary tool does not recalculate
metrics or rerun models. This snapshot claims no novelty or method efficacy.
