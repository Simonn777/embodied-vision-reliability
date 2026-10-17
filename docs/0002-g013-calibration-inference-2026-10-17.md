# Frozen-v2 calibration inference: g-013

October 17, 2026. AI-assisted execution, audit and writing. **2/10 calibration
groups inferred; 0/20 evaluation groups.** This is raw prediction collection,
not tracking-error analysis, threshold fitting or a reliability result.

## Scope and provenance

Fetched main `5561cf5ee91ac01a9c419ca3a1c938fd43d32a4e` after PR #6 merged;
this is the **actual execution source**, distinct from this later publication.
Read the repository instructions, state, charter, frozen v2 and
[g-033 publication review](0002-g033-publication-review.md). No active model process,
g-013 cache marker/output or open research PR was found. Read-only task lookup
provides last-run metadata, not a live-execution status API; the earlier run in
this conversation had finished without starting a model. No schedule was changed.
A cache-wide research lock, group lock and exclusive persistent launch marker
protected this attempt across worktrees. An isolated worktree preserved older edits.

The unchanged real gate verified all **29 mandatory files** against input commit,
execution HEAD/current files and fetched main; the canonical committed/published
receipt is unchanged. Frozen input:
`bbe7428e126a566a62d3c123d7158f120aa66e2b`.
Receipt SHA256:
`bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`.
Protocol/config/manifest/source hashes are in the
[preflight](../results/0002-pilot-execution/g-013-20261017/preflight.json).

All16 exact metadata versions and cached module origins passed, including real
NumPy2.0.2 and torch2.10.0+cpu imports/native paths. Fifteen RECORD digests match
historical provenance; NumPy's differs as already disclosed by g-033 recovery.
The same version-matched reconstructed environment was reused, not newly rebuilt
or claimed byte-identical to historical provenance. No dependency/model/data
bytes were downloaded in this step. The conservative cumulative acquisition
ceiling remains **1,482,283,622 bytes <10GiB**, including prior torch dry-run and
installation/replacement events and the NumPy recovery. Git/CI traffic is separate.
Cached checkpoint/archive SHA256 and pinned upstream730cda1c source hashes passed;
actual imported TAPNet hashes are also in the inference receipt. No frozen file,
parameter, score implementation or gate was changed or mocked.

## Single real inference

Only allocated whole group g-013, containing video13, was inferred once: its first
64 consecutive256x256 frames, arrival0, at most the firsttwo frame0-visible original
point indices. Query count was determined by the unchanged runner: **2**,
indices **[0, 1]**. No future-visibility screening/replacement or replay.

```bash
env -i PATH="$PATH" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_frozen_pilot.py infer --cache scratch/boot-cache --group-id g-013 --output results/0002-pilot-predictions/v2-reconstructed/g-013
```

This command **has run**; its output exists. Do not reuse the directory or rerun it.
The exact execution-time orchestration, preflight and parent supervisor sources,
commands, stdout/stderr, UTC times, wait statuses and digests are retained in
[execution](../results/0002-pilot-execution/g-013-20261017/) and
[parent observation](../results/0002-pilot-execution/g-013-20261017-supervision/).
Archived wrapper/supervisor copies document the original `scratch/` entry points;
their archived locations are not runnable entry points. Frozen runner/adapter
code enforces seed20261016, strict weights_only state dict, explicit causal flag,
CPU eval/no gradients/inference_mode. Kept env-i, four-entry PYTHONPATH, all five
numeric thread variables2, affinity[0,1], torch intra/inter-op[2,2], external
900-second timeout and6GiB AS cap. The supervisor independently records the
wrapper exit; each wrapper phase separately records its own child exit/wait4.

| Actual measurement | Value |
|---|---:|
| Inference child start UTC | 2026-10-17T17:38:20.803813+00:00 |
| Inference child end UTC | 2026-10-17T17:39:10.214937+00:00 |
| Inference child exit / raw wait status | 0 / 0 |
| Wrapper exit / raw wait status observed by its parent | 0 / 0 |
| Inference child wall seconds | 49.410802142 |
| Runner wall seconds | 48.772355263 |
| Whole wrapper wall seconds, including preflight | 51.362839581 |
| Supervisor observed elapsed seconds | 51.402631025 |
| First-frame interval seconds | 0.506901694 |
| Subsequent-frame median interval seconds | 0.517579196 |
| Frame intervals summed once seconds | 34.517204113 |
| Peak RSS KiB | 2553240 |
| CSV rows / raw shape | 128 / [64, 2, 4] |
| Finite / nonfinite scalars | 512 / 0 |
| Coordinate-missing / logit-invalid rows | 0 / 0 |

Frame intervals are streaming measurements, not an independently measured full
forward wall time. Hardware: AMD EPYC 9V74 80-Core Processor; Python3.12.14.
At preflight, host MemAvailable=9344732KiB;
cgroup memory.max=8589934592bytes, current=3270541312bytes;
disk free=29259108352bytes. The process still had the stricter
6GiB AS and two-CPU/thread limits. No realtime/all-point guarantee follows.

Raw [CSV](../results/0002-pilot-predictions/v2-reconstructed/g-013/video-013.csv)
SHA256: `8707198875187c1b43c06a07970eb5da72fc745f372f43afd87411daf72c6d5c`.
[Runner receipt](../results/0002-pilot-predictions/v2-reconstructed/g-013/receipt.json)
retains each frame interval, selected indices, model hashes and actual modules.

## Independent checks and limits

A separate standard-library-only verifier checked complete frame×query grids,
point IDs, finite/missing counts, raw hashes, timestamps, once-per-frame latency,
runner/child/parent exits and resources, all29 frozen files and16 recorded version/
origin bindings. It also checked all **37 existing g-033 result/evidence files**
byte-for-byte against execution main. See
[verification](../results/0002-pilot-execution/g-013-20261017/independent-verification.json) and its source. Query selection
is supported by unchanged frozen code/input identity and the runner receipt; this
verification did not decode annotations again, import candidate/model code,
replay inference or calculate GT errors/scores. It verifies evidence consistency,
not external independent replication of the model run.

Work pytest is absent: the saved availability probe reports false/zero tests;
no pytest was installed or Work test pass claimed. The existing synthetic smoke
completed1,240 frames in2.75s (3.075s child wall). Full tests and smoke must pass in
**this publication head's** standard public Research checks CI before review.
Software smoke is not another learned-model/data run.

No other group, g-033 rerun, evaluation, GT error, confidence score, reliability
comparison or calibration-threshold fitting occurred. Shared objects/camera,
unknown scene dependence and unresolved checkpoint exposure still constrain the
future exploratory pilot. No efficacy, paper reproduction or real-robot claim.
No paid service/API/GPU/credits, permission expansion, raw-data/weight upload or
schedule change. Keep this PR unmerged for parent review; next allocated group
is **g-022**, not run here. Preserve all completed outputs.
