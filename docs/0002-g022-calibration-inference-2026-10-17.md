# Frozen-v2 calibration inference: g-022

October 17, 2026. AI-assisted execution, audit and writing.
**3/10 calibration groups inferred; 0/20 evaluation groups.** This step collected
raw predictions only. It did not compute tracking errors, confidence scores,
calibration thresholds or a reliability comparison.

## Source and safeguards

Fresh main was `767aa1600947d8726903a97bb91405f35a8b9a34`, containing the reviewed
PR #7 publication. This is the **execution source**, distinct from the later
publication head. Read AGENTS, state, next actions, charter, frozen v2 and the
g-013 report/parent review. No active research process, g-022 output/marker or
open research PR was found. Task lookup exposes last-run metadata, not live run
status; the earlier run in this conversation had finished without starting a
model. No schedule was changed. An isolated worktree preserved existing edits;
cache-wide and group locks plus a persistent exclusive marker guarded one launch.

The unchanged real gate passed all **29 frozen inputs** against the frozen input
commit, execution HEAD/current files and fetched main. The canonical committed,
main-published receipt SHA256 remained
`bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`;
input commit remained `bbe7428e126a566a62d3c123d7158f120aa66e2b`.
[Preflight](../results/0002-pilot-execution/g-022-20261017/preflight.json) records every input hash, all 16 exact dependency
versions, module origins/entry hashes, metadata paths and native module paths.
Actual NumPy 2.0.2 and torch 2.10.0+cpu imports passed. Cached weights, archive and
pinned TAPNet `730cda1c` source/import hashes matched the freeze. No gate, frozen
source, protocol, score, parameter or dependency provenance was edited or mocked.

Reused runtime-fixed and all cached inputs. No download, installation or recovery
was needed. The prior NumPy version-matched reconstruction remains disclosed:
its historical normalized RECORD digest differs, while the other 15 match.
This is not a claim of byte-identical historical environment reproduction.
The conservative cumulative acquisition ceiling remains **1,482,283,622 bytes,
below 10 GiB**, including prior torch dry-run/install/replacement and NumPy
recovery events. Git/CI traffic is separate. No paid service or permission change.

## One real inference

The frozen manifest assigns whole calibration group g-022 to video 22. Only its
first 64 consecutive 256×256 frames were inferred, with arrival 0 and at most the
first two original frame-0-visible point indices. Actual query count was
**2**, indices **[0, 1]**, selected by the unchanged runner.
No future-visibility screening, zero-query replacement or model replay occurred.

```bash
env -i PATH="$PATH" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_frozen_pilot.py infer --cache scratch/boot-cache --group-id g-022 --output results/0002-pilot-predictions/v2-reconstructed/g-022
```

**This command has run; do not rerun or reuse the output directory.**
[Raw CSV](../results/0002-pilot-predictions/v2-reconstructed/g-022/video-022.csv) and [runner receipt](../results/0002-pilot-predictions/v2-reconstructed/g-022/receipt.json) preserve the
complete grid, xy, both logits and every frame interval. CSV SHA256:
`18b6b7735367d792e98a6d0780aaf0d7338da7f20afbbc96b95cdc3f4b2db278`.

| Measurement | Actual value |
|---|---:|
| Inference child start UTC | 2026-10-17T17:51:59.186449+00:00 |
| Inference child end UTC | 2026-10-17T17:52:57.015728+00:00 |
| Preflight / inference child exit codes | 0 / 0 |
| Preflight / inference raw wait statuses | 0 / 0 |
| Parent-observed wrapper exit / raw wait status | 0 / 0 |
| Inference child wall time, seconds | 57.829069148 |
| Runner wall time, seconds | 56.932276619 |
| Wrapper wall time including preflight, seconds | 59.956238971 |
| Supervisor elapsed time, seconds | 60.005698581 |
| First-frame interval, seconds | 0.596420843 |
| Subsequent-frame median interval, seconds | 0.679082166 |
| Frame intervals summed once, seconds | 42.702031249 |
| Peak runner RSS, KiB | 2,553,116 |
| Rows / raw output shape | 128 / [64, 2, 4] |
| Finite / nonfinite scalars | 512 / 0 |
| Coordinate-missing / logit-invalid rows | 0 / 0 |

Frame intervals describe streaming work, not a separate measurement of full
forward wall time. Hardware: AMD EPYC 9V74 80-Core Processor; Python 3.12.14.
At preflight, host MemAvailable was 9,346,112 KiB;
cgroup memory.max was 8,589,934,592 bytes and memory.current was
3,277,438,976 bytes; disk free space was 29,249,101,824 bytes.
The stricter per-process 6 GiB address-space cap and two-CPU/thread limits applied.
These few points/frames establish no real-time or all-point performance guarantee.

Kept `env -i`, the four-entry fixed PYTHONPATH, five numeric thread variables set
to 2, CPU affinity [0, 1], torch intra/inter-op threads [2, 2], external timeout
900 seconds, seed 20261016, strict `weights_only` state-dict matching, explicit
causal convolution, CPU eval/no gradients and `inference_mode`. The
[wrapper record](../results/0002-pilot-execution/g-022-20261017/outer.json) separately observes each timeout-wrapped child;
the [supervisor](../results/0002-pilot-execution/g-022-20261017-supervision/) directly
observes the wrapper's exit. UTC times, elapsed/CPU/RSS accounting, exact commands,
stdout/stderr and source hashes are retained. `wait4` wrapper resource accounting
can include waited descendants; the runner's own RSS is reported above. Archived
sources document original `scratch/` entry points and are not runnable at their
archived locations.

## Independent verification and software checks

The [standard-library verifier](../results/0002-pilot-execution/g-022-20261017/verification-source.py) reads no annotations
and imports no candidate/model code. It checks grid/point IDs, finite/missing
counts, file/numeric hashes, once-per-frame latency, time/exits/resources and
Git/source bindings. It rehashes cached checkpoint/archive bytes, three upstream
files, 16 module entry files and 16 installed RECORD files; native paths are
checked. This does not hash every installed package file or independently repeat
inference. Query selection relies on frozen code/input identity and the runner
receipt, without another annotation decode. All **65 prior g-033/g-013 result,
report and review files** match execution main byte-for-byte; their hashes are
listed in [verification](../results/0002-pilot-execution/g-022-20261017/independent-verification.json).

An added post-run RECORD check initially exited 1 because it compared raw CRLF
file bytes with the preflight/history digest of universal-newline decoded text.
The [diagnostic](../results/0002-pilot-execution/g-022-20261017/verification-RECORD-diagnostic.json) and failed verifier
source are preserved. All 16 normalized text digests matched the recorded values.
The corrected verifier explicitly records **both raw-byte and normalized-text
digests** and uses the existing text semantics for comparison; it passed.
No runtime bytes, frozen gates or inputs changed, and the model was not restarted.

Work pytest remains unavailable: the saved probe reports false and zero tests;
no pytest was installed or Work test pass claimed. The existing toy smoke passed
1,240 frames in 3.12 seconds (3.398 seconds child wall).
Full tests and smoke must also pass for the exact publication head in standard
public Research checks CI. Software smoke is not learned-model inference.

No other group, evaluation, GT error, score, threshold fit or reliability
comparison was run. Shared objects/camera, unknown scene dependence and unresolved
checkpoint exposure still limit this exploratory pilot. No efficacy, paper
reproduction, SOTA or real-robot claim. No added paid API/GPU/credits, raw-data or
weight upload, account/permission expansion or schedule change. Keep this PR
unmerged for independent parent review. The next allocated group is **g-040**;
it was not run. Preserve all completed outputs and historical failure records.
