# Two-parent development baseline

This package preserves both evaluation attempts. No model was called by either evaluator.

| Parent / equal-parent mean | J (%) | F (%) | J&F (%) | Included / raw frames |
|---|---:|---:|---:|---:|
| cat-girl | 94.1226307 | 90.2994880 | 92.2110594 | 87 / 89 |
| upside-down | 95.4077343 | 88.8790287 | 92.1433815 | 63 / 65 |
| Equal-parent mean | 94.7651825 | 89.5892584 | 92.1772205 | 150 / 154 |

Both recordings have initial IDs 1 and 2; this campaign targets only ID 1, the smallest positive initial ID. These are **object-subset development J&F**, not an official all-object DAVIS benchmark result. The metric implements the referenced official boundary and endpoint rules. First and final frames are excluded from means but retained in the complete raw-frame JSON.

Attempt 01 failed before PNG/NPZ decoding because its new driver expected `signal_attempts` from a different supervisor schema. Its source, failure receipt, logs, and full cost remain intact. Attempt 02 changed only admission to require the actual legacy schema and fixed supervisor/monitor hashes; the J/F implementation is byte-identical between attempts.

Attempt 02 completed with exit 0, 0.715826292 s inner time, 0.931311083 s complete outer time, and 72,960 KiB peak process-group RSS. Both logs are empty, with no remaining descendants or cleanup errors. Attempt 01 cost is separately retained in `summary.json`; it is not removed from development accounting.

The baseline cache completed all six recordings in 458.30076425 s with 4,672,848 KiB peak RSS. This covers all six, whereas the evaluation above covers two DEV parents. Inner and outer durations overlap and must not be summed.

`bindings.json` maps every copied private origin to its byte-identical copy and SHA-256. Absolute paths remain as recorded; no redaction or field truncation was applied. No media, prediction NPZ, Torch cache, or checkpoint is included. `summary.json` is a labeled derivation from the original saved JSON records.

The accompanying campaign source-CI record reports commit `43aef1615ee95d636f5fe9a712740f87ea8b3e17`, completed/success (run `39402751859`). Evaluation driver versions are separately pinned by the two private operation specs and preserved source snapshots; this CI record does not claim that those private drivers were part of the Git commit.
