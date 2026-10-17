# Frozen-v2 calibration inference: g-039

October 17, 2026. AI-assisted execution, measurement and verification.
**5/10 calibration groups inferred; 0/20 evaluation groups.** This step saved raw
predictions only, without GT errors, confidence scoring or threshold fitting.

## Provenance and scope

Actual execution main was `1f4d14f8b139c4ecc2e55036228262d74e9fd7f9`, fetched after
reviewed PR #9 merged. It is distinct from the later publication head. Read
AGENTS/state/next actions, charter, frozen v2 and the g-040 report/parent review.
Visible research-process, worktree-output and cache-reservation checks found no
g-039 task. They do not attest to all cloud activity or infer live execution from
scheduling history. An isolated worktree, cache-wide/group locks, persistent
exclusive marker and fresh output directory guarded one launch.

No applicable cloud runtime skill was exposed by the fully paginated skill
catalogs. Only non-sensitive system/resource and selected package/path metadata
were collected; no credential values were read or printed and no permissions
changed. See [prelaunch](../results/0002-pilot-execution/g-039-20261017/prelaunch.json) and [preflight](../results/0002-pilot-execution/g-039-20261017/preflight.json).

The unmodified real gate passed all **29 frozen files** and **16 exact dependency
versions/origins**, including actual NumPy 2.0.2, torch 2.10.0+cpu and native paths.
Input commit: `bbe7428e126a566a62d3c123d7158f120aa66e2b`. The canonical committed,
main-published `protocols/0002-freeze-v2.json` retained SHA256
`bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`.
Pinned TAPNet `730cda1c`, actual import hashes, checkpoint and archive matched.
Protocol/config/manifest/source hashes are retained in preflight. No frozen
input, parameter, score implementation or gate changed; no mocks or bypass.

Reused runtime-fixed and cached inputs without downloading or installing
packages/data/weights. The prior NumPy version-matched reconstruction and its
historical normalized RECORD exception remain disclosed; this is not byte-exact
historical environment reproduction. The other 15 historical RECORDs match.
Acquisition ceiling stays 1,482,283,622 bytes below 10 GiB, including prior torch
dry-run/install/replacement and NumPy recovery events; Git/CI traffic is separate.

## Single measured run

Manifest membership is whole calibration group g-039, video 39. The unchanged
runner used the first 64 consecutive 256×256 frames, arrival 0 and at most the
first two original frame-0-visible point indices. Actual query count was
**2**, indices **[0, 1]**; no future visibility selection,
zero-query replacement or replay occurred.

```bash
env -i PATH="$PATH" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_frozen_pilot.py infer --cache scratch/boot-cache --group-id g-039 --output results/0002-pilot-predictions/v2-reconstructed/g-039
```

**This command has run. Preserve the output; do not repeat it.**
[Raw CSV](../results/0002-pilot-predictions/v2-reconstructed/g-039/video-039.csv) SHA256:
`999a427c893754a1d08e455f217b6472fb5d60089fd8af3db47c05e362bcdf9f`.
The [runner receipt](../results/0002-pilot-predictions/v2-reconstructed/g-039/receipt.json) retains xy/logit identities and every interval.

| Measurement | Actual value |
|---|---:|
| Inference child start UTC | 2026-10-17T18:20:53.457673+00:00 |
| Inference child end UTC | 2026-10-17T18:21:41.614975+00:00 |
| Preflight / inference child exits and raw waits | both 0 / 0 |
| Parent-observed wrapper exit / raw wait | 0 / 0 |
| Inference child wall time, seconds | 48.156908932 |
| Runner / wrapper wall time, seconds | 47.530838826 / 50.068811038 |
| Supervisor elapsed time, seconds | 50.110496364 |
| First / subsequent median frame interval, seconds | 0.498307125 / 0.512663093 |
| Frame intervals summed once, seconds | 33.778117613 |
| Runner peak RSS, KiB | 2,553,996 |
| Rows / raw shape / finite scalars | 128 / [64, 2, 4] / 512 |
| Nonfinite scalars / coordinate-missing / logit-invalid rows | 0 / 0 / 0 |

Hardware: AMD EPYC 9V74 80-Core Processor; Python 3.12.14. Preflight host MemAvailable was
9,319,180 KiB, cgroup memory.max 8,589,934,592 bytes,
memory.current 3,326,357,504 bytes and disk free 29,217,443,840 bytes.
Kept `env -i`, fixed four-entry PYTHONPATH, five numeric variables at 2, affinity
[0, 1], torch intra/inter-op [2, 2], external timeout 900 seconds, 6 GiB AS and
seed 20261016. Frozen code enforces CPU/eval/no gradients/`inference_mode`, strict
`weights_only` state matching and explicit causal convolution. Small-clip latency
is not a real-time/all-query guarantee; interval sums are not full-forward wall time.

The [wrapper](../results/0002-pilot-execution/g-039-20261017/outer.json) and
[supervisor](../results/0002-pilot-execution/g-039-20261017-supervision/) separately save
child and parent-observed wrapper exits, raw waits, UTC/elapsed/RSS/CPU, commands,
stdout/stderr and source hashes. Wrapper `wait4` usage can include descendants;
runner RSS is separate. Archived scripts document their original `scratch/`
entry points and are not runnable at their archive locations.

## Verification, software checks and limits

The [independent stdlib report](../results/0002-pilot-execution/g-039-20261017/independent-verification.json) verifies the
complete frame×query grid, original IDs, finite/missing counts, file/numeric
hashes, once-per-frame timing, resources/exits and frozen Git/current bindings.
All **130 prior-group result/report/review files**, including failures, remain
byte-identical. It rehashes cache objects, three upstream files, 16 module entries
and 16 RECORD files, and checks native paths. No candidate/model import, annotation
decode or model replay. Query selection relies on frozen code/input identity and
the runner receipt, not another label-derived selection. These checks do not hash
every installed package/native file or independently replicate inference.

RECORD uses universal-newline normalized UTF-8 text for historical/preflight
comparison; separate raw-byte digests are recorded. No raw/text hash mixing.
Each [post-run check](../results/0002-pilot-execution/g-039-20261017-postrun/) has a simple parent's actual child exit/raw wait,
UTC/elapsed/RSS/CPU, source hashes and raw stdout/stderr. Software and verifier
children exited 0/0. The parent additionally checks source/log bindings for every
closed phase, including its current phase. Final publication audit closes all
three records. Post-run records stay outside the verifier artifact map; no receipt
hashes itself. No post-run failure/recovery or frozen-runner change occurred.

Work pytest remains unavailable: saved probe false/zero tests, no installation
or claimed Work test pass. Existing toy smoke passed 1,240 frames in 3.06 seconds
(3.316 seconds child wall). Exact publication-head standard public
Research checks CI must also pass; software fixtures are not learned-model runs.

No other group, prior-model replay, evaluation, GT error, confidence score or
threshold fit occurred. Shared objects/camera, unknown scene dependencies and
unresolved checkpoint exposure constrain the future exploratory pilot. No efficacy,
SOTA, paper reproduction or real-robot claim. No paid API/GPU/credits/service,
account/permission/schedule change or raw-data/weight upload. Next manifest group
**g-020** was not run.

## Independent publication review

Two separate read-only audits of candidate `0af5092b` passed; the
[parent review receipt](../results/0002-pilot-execution/g-039-parent-review.json)
consolidates their findings. They recomputed the complete 64×2 grid, numeric/file
hashes and timing, checked all 29 frozen inputs and 130 unchanged historical
files, and verified the three post-run checks' source/log/exit bindings. The
progressive binding records include each current phase without hashing a receipt
into itself. All 35 original result files remain unchanged. RECORD normalization
also passed six standard-library text-decoding fixtures.

The review checks published evidence; it does not rerun inference, re-derive
query selection from annotations or locally rehash cloud cache bytes. Post-run
child exits are observed by their parent; that parent's own exit has no further
supervisor record. Candidate [CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39057978375)
passed 184 tests in 11.35 seconds and a 1,240-frame synthetic smoke in 3.13
seconds. Final publication and merged main must pass their own checks. These
prediction-collection results establish no reliability or real-robot outcome.
