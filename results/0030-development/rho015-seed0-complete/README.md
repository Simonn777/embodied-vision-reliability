# Completed first-configuration direct controls and TRAIN-fit diagnostic

rho=0.15 / seed=0; all six parent recordings remain permanently development.
Both direct fit/rollout/evaluation triples, the continuation state and the
TRAIN-only fixed-fit diagnostic completed clean before this directory was created.

| Arm | J (%) | F (%) | J&F (%) | ΔJF pp vs native | cat-girl ΔJF pp | upside-down ΔJF pp |
|---|---:|---:|---:|---:|---:|---:|
| native | 94.7651825 | 89.5892584 | 92.1772205 | +0.0000000 | +0.0000000 | +0.0000000 |
| multistep | 94.2441723 | 89.1234051 | 91.6837887 | -0.4934318 | -0.1743853 | -0.8124782 |
| one_step | 94.5860235 | 89.3969181 | 91.9914708 | -0.1857496 | -0.3086098 | -0.0628895 |
| gt_memory | 94.8480678 | 89.9341811 | 92.3911244 | +0.2139040 | +0.0295183 | +0.3982897 |
| direct-cost_matched | 94.0890518 | 88.4860442 | 91.2875480 | -0.8896724 | -0.2872975 | -1.4920474 |
| direct-update_matched | 94.2013303 | 88.6548030 | 91.4280666 | -0.7491538 | -0.1031775 | -1.3951301 |

Metrics are saved single-target DEV values, with 150 interior frames of 154
raw frames, equal-parent aggregation. No metrics were recomputed during export.
The table is not an official all-object benchmark or held-out paper result.
The unchanged final two-independent-source +3 J&F gate remains outstanding.

The measured teacher-plus-multistep-regression training budget is
1162.078428250 + 6.431160417 = 1168.509588667 seconds.
Cost-matched direct used 247 complete updates
and 1151.319815500 seconds
(within budget). Update-matched direct completed 288 updates and used
1339.648869000 seconds.
These comparisons do not omit rollout/evaluation costs:

| New complete operation | Actual outer wall seconds |
|---|---:|
| fit-direct-cost_matched-rho015-seed0-01 | 1151.319815 |
| rollout-direct-cost_matched-rho015-seed0-01 | 112.568699 |
| evaluate-direct-cost_matched-rho015-seed0-01 | 1.094740 |
| fit-direct-update_matched-rho015-seed0-01 | 1339.648869 |
| rollout-direct-update_matched-rho015-seed0-01 | 113.540880 |
| evaluate-direct-update_matched-rho015-seed0-01 | 1.097797 |
| fixed-fit-01 | 1.340794 |

Same-seed prefix check from saved receipt fields: the first 247 losses
are exactly equal: True; sample-identity mismatch
count: 0. Common checkpoint steps are
[48, 96, 144, 192, 240]; all corresponding saved SHA-256
hashes equal: True. This is deterministic
prefix consistency, **not independent-seed replication**. Checkpoint tensors
were not loaded; final checkpoints at different step counts are not compared.
All mismatch indices and comparison pairs, if any, remain in the summary.

Fixed-fit reevaluates the three final heads against all 48 TRAIN targets:
144 head forwards, no SAM2 call, no fitting update or DEV label access. Its
saved E/E0 and direction summaries are training diagnostics, not accuracy,
and are retained in full rather than substituted for rollout results.

The initial twelve operations and original failed baseline evaluation are
hash-bound references in `initial-bundle-reference.json`; they are not recopied.
Shared cache and historical failure costs remain in `all-arms-summary.json`.
Inner/anchor timers overlap with outer time and must not be added.

Only text receipts/specs/logs/source maps and reporting sources are exported.
No PT/NPZ, model, RGB, mask or GT payload is copied. Original SHA fields continue
to refer to original bytes; public hashes record the literal path redactions.
No completed receipt field is deleted. This package asserts no novelty.
