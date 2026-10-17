# Frozen-v2 calibration inference: g-020

October 17, 2026. AI-assisted execution, measurement and verification.
**6/10 calibration groups inferred; 0/20 evaluation groups.** This step collected
raw predictions only, without real-data errors, confidence scores or thresholds.

## Execution identity and controls

Fresh execution main was `697371746cbe3bd89d37472ec5889cb824db4f70`, containing
reviewed PR #10. It differs from the later publication head. Read instructions,
state, next actions, charter, v2 and the g-039 report/parent review. Visible
research-process, output and reservation checks found no g-020 task; scheduling
history was not used as live execution evidence. An isolated worktree preserved
existing edits; cache-wide/group locks, a persistent exclusive marker and fresh
output directory guarded one launch. No credentials or permissions were changed.

The unmodified real gate passed **29 frozen files and 16 exact dependency
versions/origins**, including actual NumPy 2.0.2, torch 2.10.0+cpu and native paths.
Frozen input: `bbe7428e126a566a62d3c123d7158f120aa66e2b`. Canonical committed and
main-published `protocols/0002-freeze-v2.json` retained SHA256
`bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`.
[Preflight](../results/0002-pilot-execution/g-020-20261017/preflight.json) retains protocol/config/manifest/source hashes,
module/metadata/native origins, pinned TAPNet `730cda1c`, checkpoint/archive
identities and resources. No frozen code, parameter, score or gate changed or
was mocked. Reused runtime-fixed/cache with no downloads or installation.

The prior NumPy reconstruction remains version-matched, not byte-identical to
historical provenance: its normalized RECORD differs; the other 15 match.
Acquisition ceiling remains 1,482,283,622 bytes below 10 GiB, including prior torch
dry-run/install/replacement and NumPy recovery events. Git/CI traffic is separate.

## One measured inference

Manifest identity is whole calibration group g-020, video 20. The unchanged
runner used its first 64 consecutive 256×256 frames, arrival 0 and at most the
first two original frame-0-visible indices. Actual query count was
**2**, indices **[0, 1]**; no future-visibility screening,
zero-query substitution or replay.

```bash
env -i PATH="$PATH" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_frozen_pilot.py infer --cache scratch/boot-cache --group-id g-020 --output results/0002-pilot-predictions/v2-reconstructed/g-020
```

**This command has run. Preserve its output; do not repeat it.**
[Raw CSV](../results/0002-pilot-predictions/v2-reconstructed/g-020/video-020.csv) SHA256:
`147c2a9450526daa9806810d312cbeaf60480b08360c2eb78f94eb9f1cd1a0a2`.
The [runner receipt](../results/0002-pilot-predictions/v2-reconstructed/g-020/receipt.json) retains every frame interval and source identity.

| Measurement | Actual value |
|---|---:|
| Inference child start UTC | 2026-10-17T18:33:16.610560+00:00 |
| Inference child end UTC | 2026-10-17T18:34:11.231486+00:00 |
| Preflight / inference child exits and raw waits | both 0 / 0 |
| Parent-observed wrapper exit / raw wait | 0 / 0 |
| Inference child wall time, seconds | 54.620452908 |
| Runner / wrapper wall time, seconds | 53.911346312 / 56.859585258 |
| Supervisor elapsed time, seconds | 56.902681966 |
| First / later median frame interval, seconds | 0.837475877 / 0.576290600 |
| Frame intervals summed once, seconds | 38.403893087 |
| Runner peak RSS, KiB | 2,553,192 |
| Rows / raw shape / finite scalars | 128 / [64, 2, 4] / 512 |
| Nonfinite scalars / coordinate-missing / logit-invalid rows | 0 / 0 / 0 |

Hardware: AMD EPYC 9V74 80-Core Processor; Python 3.12.14. Preflight MemAvailable was
9,328,520 KiB, cgroup memory.max 8,589,934,592 bytes,
memory.current 3,343,679,488 bytes and free storage 29,202,989,056 bytes.
Kept `env -i`, fixed four-entry PYTHONPATH, five numeric variables at 2, affinity
[0, 1], torch intra/inter-op [2, 2], timeout 900 seconds, 6 GiB AS and seed 20261016.
Frozen code enforces strict `weights_only` state matching, explicit causal
convolution, CPU/eval/no gradients/`inference_mode`. These few points/frames do
not establish real-time/all-query performance. Interval sums are not an independent
full-forward wall-time measurement. Wrapper `wait4` resource usage can include
waited descendants; runner RSS is separate.

## Verification and evidence limits

[Execution](../results/0002-pilot-execution/g-020-20261017/) and
[supervision](../results/0002-pilot-execution/g-020-20261017-supervision/) preserve original
commands, UTC/elapsed/RSS/CPU, child and parent-observed wrapper exits/raw waits,
stdout/stderr and source hashes. Archived sources describe original `scratch/`
entry points and are not runnable at archive locations.

The [independent stdlib verifier](../results/0002-pilot-execution/g-020-20261017/independent-verification.json) passed complete
grid/point IDs, finite/missing counts, file/numeric hashes, once-per-frame timing,
resources/exits and frozen Git/current bindings. All **167 prior result/report/review
files**, including failures, remain byte-identical. Cache verification rehashes
objects, three upstream files, 16 module entries and RECORDs, and checks native
paths. It does not import candidate/model code, decode annotations, hash every
installed file or independently repeat inference. Query selection is supported
by unchanged frozen code/input identity and the runner's indices, not another
annotation-derived selection. RECORD comparisons use universal-newline normalized
UTF-8 text with separately recorded raw-byte digests; no mixed representations.

The three [post-run records](../results/0002-pilot-execution/g-020-20261017-postrun/) save the parent's actual observations of its
software/verifier/publication-audit **children**: exit/raw wait, UTC/elapsed/RSS/CPU,
commands, source hashes and raw stdout/stderr. All child observations are 0/0.
Progressive closure verifies completed/current phase source/log bindings without
receipt self-hashes; these records stay outside the verifier artifact map.
**The post-run parent program's own exit has no further supervisor receipt.**
It is not the child exit field above. No extra supervisor or frozen-runner change
was added, and no post-run failure/recovery occurred.

Work pytest is unavailable: probe false, zero tests, no installation or claimed
Work test pass. Existing toy smoke passed 1,240 frames in 3.08 seconds
(3.346 seconds child wall). Exact publication-head standard public
Research checks CI must also pass; software checks are not learned-model runs.

No other group, old-model replay, evaluation, real-data GT errors, scores or
threshold fit occurred. Shared objects/camera, unknown scene dependencies and
checkpoint exposure remain limitations of the prospective exploratory pilot.
No efficacy, SOTA, paper-reproduction or real-robot claim. No added paid service,
API/GPU/credits, account/permission/schedule change or raw-data/weight upload.
Keep this PR unmerged for independent parent review. Next manifest group
**g-016** was not run.
