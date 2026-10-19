# 0019: author-recommended depth refinement control

Prospective development comparison, 2026-10-19. Freeze source and this protocol
in a published commit with successful exact-commit CI before scientific calls.
This strengthens an existing baseline; it does not introduce a new method.

Use every one of the 80 existing 0017 pairs, two depth arms, seed 0 and the
original pinned PoseLib 6a636b1d build. Change exactly the nested bundle setting
from `{}` to `{"loss_type":"TRUNCATED_CAUCHY"}`. All RANSAC settings, input order,
raw camera/depth conventions and thresholds remain those of 0017. The wrapper
still sets the effective final loss width; internal TRUNCATED/25-step LO and
final 100-step budget are unchanged. No warm start, solver repair or new build.
See the source-checked `docs/literature/0019-control-admission.md` for the current
API, effective settings, overwritten initial-pose trap and novelty limits.

Run each of 160 calls once, alternating arm order by pair index, in isolated
workers: one thread, 90 seconds per worker, 1800 seconds per complete operation,
8 GiB sampled memory ceiling. Preserve errors and logs; no automatic retry or
replacement. Charge input, native call, output diagnostics and supervisor costs.
The historical 0017 timings are not contemporaneous matched-time measurements.
Cached frontend inference cost remains unavailable. No new data or model assets.

Before execution verify the complete 0017 prediction seal. Require each new
input hash/calibration binding to agree with its original record. Only the final
loss changes, so the returned pre-final epipolar mask and RANSAC statistics must
agree with 0017: masks/discrete quantities exactly, numeric values within
`1e-12*max(1,abs(a),abs(b))`. A mismatch is an integrity/qualification failure,
not an accuracy outcome. Final raw geometry may change. Native estimator errors
and timeouts remain in the 80-pair denominator at 180 degrees; non-estimator
failures block interpretation. Seal all 160 outcomes before opening any labels.
For this controlled qualification, a returned malformed or nonfinite geometry,
mask or statistics object is an output-integrity failure and blocks the whole
comparison. This strict rule differs from a thrown native exception or timeout;
it is fixed before execution and cannot be relaxed after an outcome.

One separate evaluation reads only the saved 0017 evaluation (SHA256
`66f3f693fa7473aba5371fe473c4f8c9df39a3dd711a32941f28f3e0380eb151`),
after the new complete-cohort seal passes. Do not reopen HDF5 reference poses.
Use the unchanged 0013 directed pose-angle and DMS AUC implementation, retaining
sign-folded translation as secondary. Report AUC@5/10/20, all five scene tables,
all pair deltas, material benefits/harms (at least 1 degree or crossing 5 degrees),
strict-below-5 counts, failures, and costs. Compare each new arm with its own
0017 CAUCHY arm and fixed `e_norm`; retain `e_ts` in the reference table.
Do not select a method using test labels or present an oracle as deployable.

No threshold tuning, selected-scene claim, repeated seed search, confidence
claim from correlated image pairs, or manuscript admission follows. Any gain
belongs to an existing author-recommended configuration and becomes a stronger
control. A distinct candidate must still meet the unchanged substantial-gain,
matched complete-pipeline time, two independent held-out sources, scene-level
uncertainty, harm, ablation and replication requirements. All five scenes remain
development forever. 0018 remains ineligible after its nonfinite-state failure;
0019 neither repairs that diagnostic nor supplies its missing mechanism result.
