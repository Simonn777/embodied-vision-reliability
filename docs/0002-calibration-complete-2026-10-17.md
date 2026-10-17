# Frozen-v2 calibration prediction collection complete

October 17, 2026. AI-assisted execution, measurement, verification and writing.
**10/10 calibration groups collected; 0/20 evaluation groups collected.**
Only 10 of the 30 prescribed group prediction sets exist. No real-data tracking
errors, confidence scores, threshold fitting or reliability comparison ran.

## Serial completion batch

Fresh main was `d6eb151874fbe2afe52651f0049bd2e009afe8d7`, after reviewed PR #11 merged.
This is the actual execution source for all four new groups, distinct from the
later publication head. Read repository instructions/state/next actions, charter,
frozen v2 and the g-020 review. Visible process, output, marker and lock checks
preceded each launch; scheduling history was not live execution evidence.
The four singleton groups ran once in frozen allocation order, each in a fresh
independent process/output directory with cache-wide/group locks and an exclusive
persistent marker. Each real preflight, inference and independent artifact audit
completed before the next launch. The previous six groups were read from existing
receipts and were not replayed.

Every new group passed the unmodified real gate: all **29 frozen inputs and 16
exact dependency versions/origins**, actual NumPy 2.0.2, torch 2.10.0+cpu/native
paths, canonical main-published receipt, pinned source/import/archive/checkpoint
hashes. Input commit `bbe7428e126a566a62d3c123d7158f120aa66e2b`; receipt commit
`fbb6dd9208f6ea95e588f410c05c9ac5b58dcb1c`; canonical receipt SHA256
`bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`.
No frozen runner/config/protocol/score changes, mocks, new downloads or installs.
The historical NumPy reconstruction remains version-matched, not byte-exact;
normalized RECORD and raw-byte hashes are recorded separately. Acquisition ceiling
stays 1,482,283,622 bytes below 10 GiB, including prior torch and NumPy events.

Each group used its first 64 consecutive 256×256 frames, arrival 0 and at most
the first two original frame-0-visible point indices, preserving zero-query rules.
Actual selected indices and output completeness appear below. Kept `env -i`,
PYTHONPATH `scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src`,
five numeric variables at 2, CPU affinity [0, 1], torch intra/inter-op [2, 2],
external timeout 900 seconds, 6 GiB AS and seed 20261016. Frozen code enforces
strict `weights_only` state matching, explicit causal convolution, CPU eval/no
gradients/`inference_mode` and the restricted NumPy loader. No future visibility
screening, zero-query replacement or replay. Exact commands, resource environment,
UTC/elapsed/CPU/RSS, raw waits, stdout/stderr and source hashes are retained in
each group execution/supervision directory. Small prefixes/few points do not
establish real-time/all-query feasibility or accuracy.

## Four new measurements

| Group | Points | Inference child s | Runner s | Wrapper s | Peak RSS KiB | First / later median frame s |
|---|---|---:|---:|---:|---:|---:|
| g-016 | [0, 1] | 46.510177233 | 45.858284290 | 48.399880116 | 2,553,328 | 0.555603973 / 0.479288369 |
| g-032 | [0, 1] | 46.137381181 | 45.534410617 | 47.952475859 | 2,553,332 | 0.606362378 / 0.486504761 |
| g-035 | [0, 1] | 47.626444436 | 47.029284239 | 49.415877146 | 2,552,520 | 0.507343440 / 0.504103178 |
| g-026 | [0, 1] | 47.103831073 | 46.431240773 | 48.881782242 | 2,553,172 | 0.572420861 / 0.498965964 |

Frame intervals are counted once per frame and are not independent full-forward
wall times. Runner peak RSS is separate from parent `wait4` usage, which can
include waited descendants. Each new preflight/inference child and separately
parent-observed wrapper exit/raw wait is 0/0. Full precision is in the
[machine-readable index](../results/0002-calibration-completion/index.json).

## All ten groups

Old rows below transcribe existing receipts, with their original execution SHAs.
The original g-033 wrapper exit lacks a separate parent receipt; its completed
inference child/CSV and historical review remain the evidence, without a new run.

| Group | Origin | Source | Rows / finite scalars | CSV SHA256 |
|---|---|---|---:|---|
| [g-033](../results/0002-pilot-predictions/v2-reconstructed/g-033/receipt.json) | existing | `05c2d7cb` | 128 / 512 | [`22bb58db053c0cb011c72cadee1e0d91012301ff6e15bb3264ae23e39cd9819e`](../results/0002-pilot-predictions/v2-reconstructed/g-033/video-033.csv) |
| [g-013](../results/0002-pilot-predictions/v2-reconstructed/g-013/receipt.json) | existing | `5561cf5e` | 128 / 512 | [`8707198875187c1b43c06a07970eb5da72fc745f372f43afd87411daf72c6d5c`](../results/0002-pilot-predictions/v2-reconstructed/g-013/video-013.csv) |
| [g-022](../results/0002-pilot-predictions/v2-reconstructed/g-022/receipt.json) | existing | `767aa160` | 128 / 512 | [`18b6b7735367d792e98a6d0780aaf0d7338da7f20afbbc96b95cdc3f4b2db278`](../results/0002-pilot-predictions/v2-reconstructed/g-022/video-022.csv) |
| [g-040](../results/0002-pilot-predictions/v2-reconstructed/g-040/receipt.json) | existing | `3c6b26ac` | 128 / 512 | [`399c64850174219a298641eb277b2f36918e08dac590e1e8e6b25c4da71c3741`](../results/0002-pilot-predictions/v2-reconstructed/g-040/video-040.csv) |
| [g-039](../results/0002-pilot-predictions/v2-reconstructed/g-039/receipt.json) | existing | `1f4d14f8` | 128 / 512 | [`999a427c893754a1d08e455f217b6472fb5d60089fd8af3db47c05e362bcdf9f`](../results/0002-pilot-predictions/v2-reconstructed/g-039/video-039.csv) |
| [g-020](../results/0002-pilot-predictions/v2-reconstructed/g-020/receipt.json) | existing | `69737174` | 128 / 512 | [`147c2a9450526daa9806810d312cbeaf60480b08360c2eb78f94eb9f1cd1a0a2`](../results/0002-pilot-predictions/v2-reconstructed/g-020/video-020.csv) |
| [g-016](../results/0002-pilot-predictions/v2-reconstructed/g-016/receipt.json) | new batch | `d6eb1518` | 128 / 512 | [`331416f8db40d559fc1e5fcff1982eaabcc863d6ec66c6f088bbcebaff746c79`](../results/0002-pilot-predictions/v2-reconstructed/g-016/video-016.csv) |
| [g-032](../results/0002-pilot-predictions/v2-reconstructed/g-032/receipt.json) | new batch | `d6eb1518` | 128 / 512 | [`f5bbbffe94f0cce8adb5260dff4e6e0a0deffca2436fb9522b05c73c3df7bf3c`](../results/0002-pilot-predictions/v2-reconstructed/g-032/video-032.csv) |
| [g-035](../results/0002-pilot-predictions/v2-reconstructed/g-035/receipt.json) | new batch | `d6eb1518` | 128 / 512 | [`5c4b9f568999258ad62b743394e20a51c2db90325037860f0e2aaf52027f77e4`](../results/0002-pilot-predictions/v2-reconstructed/g-035/video-035.csv) |
| [g-026](../results/0002-pilot-predictions/v2-reconstructed/g-026/receipt.json) | new batch | `d6eb1518` | 128 / 512 | [`ab23af11347fa6c6a41ca9b5c527cde0e18cf01f494d939acbd1f029dce42da7`](../results/0002-pilot-predictions/v2-reconstructed/g-026/video-026.csv) |

## Verification and publication boundaries

Per-group independent stdlib checks verify complete grids/point IDs, finite/missing
counts, file/numeric hashes, once-per-frame timing, resources/exits and frozen
Git/current/cache bindings without candidate/model imports or annotation decoding.
Cache checks rehash objects, three upstream files, 16 module entries and RECORDs,
and check native paths; they do not hash every installed file or replicate model
execution. Query selection relies on frozen code/input identity and runner indices.
All **204 prior result/report/review files**, including failures, remain byte-identical.
Later launches also preserve earlier batch artifacts. [Baseline](../results/0002-calibration-completion/baseline.json) and each prelaunch receipt retain the hash maps.

The reviewed g-020 wrapper/supervisor/post-run parent code is reused with group,
video, source and artifact bindings. Each group has separate verification and
publication-audit child observations. Unchanged software was checked **once**
under g-016: later audits explicitly reference that shared software receipt.
The final aggregate audit is another phase of the existing g-026 post-run parent,
not a new recursive supervisor. Source/log/child-exit bindings are closed without
receipt self-hashes; post-run records remain outside per-group verifier maps.
Post-run child exits are parent-observed. **The post-run parent programs themselves
have no further supervisor exit receipt.** Archived copies document original
`scratch/` entry points, not runnable archive entry points.

Work pytest probe was unavailable/zero tests; no install or Work test pass is
claimed. The one shared toy smoke passed 1240 frames in 2.6 seconds (2.838 seconds child wall).
Exact publication-head public Research checks CI must also pass. These software
fixtures are not real-data inference or scoring.

Next stage is **20 evaluation-group prediction collections**, in the frozen
manifest order, starting g-031, in future bounded sessions after review/merge.
Do not call the complete evaluator until all 30 prescribed prediction sets exist
and pass validation. No evaluation group ran in this batch; no outcomes informed
changes to frozen methods or parameters. Shared objects/camera, unknown scene
dependence and checkpoint exposure still constrain this exploratory pilot.
No efficacy, SOTA, paper reproduction or real-robot claim; no added paid service,
API/GPU/credits, account/permission/schedule change, raw-data or weight upload.
Keep this combined PR unmerged for independent parent review.
