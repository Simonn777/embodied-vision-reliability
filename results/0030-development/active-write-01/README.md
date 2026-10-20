# All-active spatial-write direct development result

All three operations and their continuation completed clean before export. Only
the final completed checkpoint was rolled out. rho=.15, seed=0; all six parents
remain permanently development. No additional checkpoint selection or metric run.

| Arm | J % | F % | JF % | ΔJF pp vs native | cat-girl ΔJF pp | upside-down ΔJF pp |
|---|---:|---:|---:|---:|---:|---:|
| native | 94.7651825 | 89.5892584 | 92.1772205 | +0.0000000 | +0.0000000 | +0.0000000 |
| multistep | 94.2441723 | 89.1234051 | 91.6837887 | -0.4934318 | -0.1743853 | -0.8124782 |
| one_step | 94.5860235 | 89.3969181 | 91.9914708 | -0.1857496 | -0.3086098 | -0.0628895 |
| gt_memory | 94.8480678 | 89.9341811 | 92.3911244 | +0.2139040 | +0.0295183 | +0.3982897 |
| direct-cost_matched | 94.0890518 | 88.4860442 | 91.2875480 | -0.8896724 | -0.2872975 | -1.4920474 |
| direct-update_matched | 94.2013303 | 88.6548030 | 91.4280666 | -0.7491538 | -0.1031775 | -1.3951301 |
| direct_all_active | 94.7513294 | 90.0497459 | 92.4005376 | +0.2233172 | +0.4764950 | -0.0298606 |

Saved fixed decision: **close_exact_channel_affine_family**. Exceeds the bound GT-control mean:
True; both parents exceed native: False. Exact reference-file
values, rather than a rounded admission literal, determine this decision.
This is a development continuation rule, not a paper claim or independent-seed
replication. The two-independent-held-out-source +3 J&F-point gate remains unmet.

The fit completed 250 updates. Its complete outer cost is
1152.387684083s against 1168.509588667s (teacher plus
multistep regression). Rollout/evaluation outer costs are
109.803284667s/1.102764542s.
All inner timers overlap these outer costs; do not add them again.
All per-step losses, active write sets and checkpoint hashes are retained in
the fit receipt. They are TRAIN diagnostics, not segmentation accuracy.

This variant differentiates through all consumed noninitial spatial writes,
with complete fixed baseline histories and unchanged pointers. It is not
recurrent BPTT. More causal write contexts receiving gradients are part of the
intervention; a change is not uniquely attributable to one mechanism.
The causal rollout is scored on 150 interior frames of 154 raw frames, one target
per parent, equal-parent DEV J/F. It is not official all-object held-out DAVIS.

Earlier six-arm results, all earlier costs and the original failed evaluator
attempt are preserved by the reference bundle; they are not copied or discarded.
New source/scope/decision documents are separately byte-bound, not claimed to
have appeared in the older fit integration commit's CI.

Only complete UTF-8 receipts, specs, logs, source and saved metric reports are
published. No PT/NPZ/model/media/GT payload is opened or copied. Checkpoint identity
is checked from bound descriptors; binary rehashes are inherited from the clean
dispatcher/rollout/evaluator, not independently repeated by this publisher.
Manifest original hashes address raw bytes; public hashes address the exact
ordered literal task-root→$TASK and home→$USER_HOME substitutions. No copied
JSON field is deleted. Scalar subtraction/decision replay is the only arithmetic.
