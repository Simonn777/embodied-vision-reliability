# Frozen-v2 calibration inference: g-040

October 17, 2026. AI-assisted execution, measurement and verification.
**4/10 calibration groups inferred; 0/20 evaluation groups.** Only raw prediction
collection occurred. No GT errors, confidence scores, thresholds or reliability
comparison were computed.

## Scope and execution identity

Fresh main `3c6b26ac1997216eb60b8509df9abea71ed093d3` contained reviewed PR #8.
This is the actual **execution source**, distinct from the later publication head.
Read repository instructions/state/next actions, charter, frozen v2, g-022 report
and parent review. Visible research processes, worktree outputs and cache
reservations contained no existing g-040 task. This scoped check does not attest
to all prior cloud activity; scheduling metadata is not live execution evidence.
An isolated worktree preserved existing edits. Cache-wide/group locks, an
exclusive persistent marker and a fresh output directory guarded one launch.

The unmodified real gate passed all **29 frozen files** and **16 exact dependency
versions/origins**, including actual NumPy 2.0.2 and torch 2.10.0+cpu/native paths.
Canonical receipt `protocols/0002-freeze-v2.json` was committed and published on
fetched main; its SHA256 remained
`bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`.
Frozen input commit remained `bbe7428e126a566a62d3c123d7158f120aa66e2b`.
[Preflight](../results/0002-pilot-execution/g-040-20261017/preflight.json) records protocol/config/manifest/source hashes,
metadata/module/native paths, cache identities and resource limits. Pinned TAPNet
`730cda1c`, official weights and archive/import hashes matched. No frozen file,
parameter, gate or score implementation changed; no mocks or bypass.

Reused runtime-fixed and all cached inputs, with **zero downloads/installations**.
The prior NumPy version-matched reconstruction and historical normalized RECORD
difference remain disclosed; this is not byte-identical historical environment
reproduction. Other 15 historical normalized RECORD digests match. Acquisition
ceiling remains 1,482,283,622 bytes below 10 GiB, including prior torch dry-run,
install/replacement and NumPy recovery events; Git/CI traffic is separate.

## One real model run

Manifest membership is whole calibration group g-040, video 40. The unchanged
runner inferred its first 64 consecutive 256×256 frames with arrival 0 and at
most the first two original frame-0-visible point indices. Actual query count
was **2**, indices **[0, 1]**. No future-visibility
screening, zero-query substitution or replay occurred.

```bash
env -i PATH="$PATH" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_frozen_pilot.py infer --cache scratch/boot-cache --group-id g-040 --output results/0002-pilot-predictions/v2-reconstructed/g-040
```

**This command has run. Preserve the existing output; do not repeat it.**
[Raw CSV](../results/0002-pilot-predictions/v2-reconstructed/g-040/video-040.csv) and [runner receipt](../results/0002-pilot-predictions/v2-reconstructed/g-040/receipt.json) retain xy,
both logits and every frame interval. CSV SHA256:
`399c64850174219a298641eb277b2f36918e08dac590e1e8e6b25c4da71c3741`.

| Measurement | Actual value |
|---|---:|
| Inference child start UTC | 2026-10-17T18:07:38.429846+00:00 |
| Inference child end UTC | 2026-10-17T18:08:26.263751+00:00 |
| Preflight / inference child exits and raw waits | both 0 / 0 |
| Parent-observed wrapper exit / raw wait | 0 / 0 |
| Inference child wall time, seconds | 47.833605655 |
| Runner wall time, seconds | 47.122065346 |
| Full wrapper wall time, seconds | 49.646681832 |
| Supervisor elapsed time, seconds | 49.686474911 |
| First-frame interval, seconds | 0.533883190 |
| Subsequent-frame median interval, seconds | 0.507835578 |
| Frame intervals summed once, seconds | 33.468414940 |
| Runner peak RSS, KiB | 2,553,160 |
| Rows / raw output shape | 128 / [64, 2, 4] |
| Finite / nonfinite scalars | 512 / 0 |
| Coordinate-missing / logit-invalid rows | 0 / 0 |

Hardware: AMD EPYC 9V74 80-Core Processor; Python 3.12.14. Preflight host MemAvailable was
9,355,220 KiB, cgroup memory.max 8,589,934,592 bytes,
memory.current 3,294,982,144 bytes, and free storage 29,231,644,672 bytes.
Kept the stricter 6 GiB address-space cap, two-CPU affinity, torch intra/inter-op
threads [2, 2], five numeric variables at 2, external timeout 900 seconds,
`env -i`, fixed four-entry PYTHONPATH and seed 20261016. Frozen code enforces strict
`weights_only` state matching, explicit causal convolution, CPU eval/no gradients
and `inference_mode`. Small-clip latency establishes no real-time/all-point claim.
Streaming intervals are not an independent full-forward wall-time measurement.

The [wrapper](../results/0002-pilot-execution/g-040-20261017/outer.json) records timeout-wrapped preflight/inference
children separately; the [supervisor](../results/0002-pilot-execution/g-040-20261017-supervision/)
directly records its wrapper child's exit. UTC times, elapsed/RSS/CPU accounting,
commands, original stdout/stderr and source hashes are retained. Wrapper `wait4`
resource usage can include waited descendants; runner RSS is reported above.
Archived source copies describe original `scratch/` entry points, not runnable
entry points at their archive locations.

## Independent artifact verification

A separate [stdlib verifier](../results/0002-pilot-execution/g-040-20261017/verification-source.py) passed complete grid,
original point IDs, finite/missing counts, file/numeric hashes, once-per-frame
latencies, exits/resources, frozen Git/current bindings and cached byte checks.
It checked all **93 prior-group result/report/review files** against execution
main byte-for-byte, including historical failures. Its
[receipt](../results/0002-pilot-execution/g-040-20261017/independent-verification.json) lists those hashes. No candidate/model
imports or annotation decoding occurred. Frame-0 query selection is supported
by unchanged frozen code/input identity and runner indices, not a second
annotation-derived selection.

Cache checks rehash archive/checkpoint bytes, three upstream files, 16 module
entry files and 16 RECORD files; native paths are checked. They do not hash every
installed package/native file or independently replicate model execution.
RECORD comparisons use the existing universal-newline normalized UTF-8 text
digest. Separate raw-byte digests are retained; unlike the historical g-022
verifier failure, these distinct representations are not directly compared.

Every data/software verification phase runs under a simple
[post-run parent supervisor](../results/0002-pilot-execution/g-040-20261017-postrun/): source hashes, actual child exit/raw wait,
UTC/elapsed/RSS/CPU and raw stdout/stderr are saved separately. The software-check
and independent-verifier children both exited 0 with raw wait 0. No post-run
failure or recovery occurred. The post-run records are outside the verifier's
artifact map to avoid circular hashes; publication review can check their own
source/log bindings. No frozen runner change was needed.

Work pytest remains unavailable: saved probe false, zero tests, no installation
or claimed Work test pass. The existing synthetic smoke passed 1,240 frames in
2.62 seconds (2.842 seconds child wall). Exact publication-head
standard public Research checks CI must also pass. These software checks do not
run the learned model or access tracking outcomes.

No other group, g-033/g-013/g-022 replay, evaluation, GT errors, score or threshold
fit occurred. Shared objects/camera, unknown scene dependencies and unresolved
checkpoint exposure limit the future exploratory pilot. No efficacy, SOTA,
paper-reproduction or real-robot claim. No paid API/GPU/credits, raw-data/weight
upload, account/permission expansion or schedule changes. Next allocation is
**g-039**, not run here.

## Independent publication review

Two separate read-only audits of candidate `b6ffe257` passed; their findings are
consolidated in the [parent review receipt](../results/0002-pilot-execution/g-040-parent-review.json).
They recomputed the complete 64×2 grid, numeric/file hashes and latency
accounting, checked all 29 frozen inputs and 93 unchanged historical files, and
verified source/log/exit bindings for all three post-run checks. All 35 new
result artifacts remain unchanged by this review. The normalized RECORD
semantics also passed six standard-library text-decoding fixtures.

These are independent checks of published artifacts, not a second model run or
a local rehash of cloud cache bytes. Query selection was not re-derived from
annotations. Candidate [CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39057030013)
passed 184 tests in 7.70 seconds and the 1,240-frame synthetic smoke in 3.70
seconds. Final publication and merged main must pass their own checks. No
reliability comparison or real-robot result is established by this collection.
