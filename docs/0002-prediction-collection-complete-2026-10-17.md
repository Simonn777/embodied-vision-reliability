# Frozen-v2 prediction collection complete

October 17, 2026. AI-assisted orchestration, execution, measurement, verification and writing.
**10/10 calibration plus 20/20 evaluation groups: all 30 prescribed prediction sets collected.**
This is prediction collection only. No real-data tracking errors, confidence scores,
threshold fitting, method comparisons or evaluator invocation occurred.

## Execution and controls

Fresh reviewed main `6bdce8259d196a832a52ec0ab289066ecdf4907c` is the actual execution source
for all 20 new evaluation groups, distinct from the later publication commit.
All ten calibration outputs were reused, never replayed. Each evaluation group
is a singleton, but the unchanged runner still infers its complete manifest group.
Allocation order was independently checked and retained without outcome selection.
Before each launch, visible processes, cache-wide/group flocks, persistent launch
markers and worktree output footprints were checked. Scheduling history is not
live execution evidence. Independent bounded processes ran serially; each group
passed its real preflight, inference and post-run artifact audits before the next.

Each real, unmodified freeze gate checked all 29 mandatory Git/current/index/HEAD
inputs and 16 exact dependencies/origins, canonical published receipt, pinned
upstream/checkpoint/archive identities. The frozen runner checked decoded video
and 64-frame prefix identities. Input commit `bbe7428e126a566a62d3c123d7158f120aa66e2b`;
receipt commit `fbb6dd9208f6ea95e588f410c05c9ac5b58dcb1c`; receipt SHA256
`bd9dc9e90980c78050a6f48043117e7d0f53d004c25d324b4ff25148380df05b`.

Every group retained first 64 consecutive 256x256 RGB frames, arrival 0 and at
most the first two original frame-0-visible indices, including the zero-query
rule. Actual queries and missing/nonfinite outputs are recorded, never replaced.
No future visibility screening. Kept env -i, the four fixed PYTHONPATH entries
(runtime-fixed:runtime:source:src under scratch/boot-cache and repository src),
five numeric variables at 2, affinity [0, 1], torch intra/inter-op [2, 2], seed
20261016, external timeout 900 seconds and 6 GiB address-space limit. Strict
weights_only loading/state matching, causal convolution, CPU eval/no gradients/
inference_mode and restricted NumPy decoding remain unchanged. Each receipt
retains commands, source/config/protocol hashes, original UTC/wall/CPU/RSS,
child/wrapper wait statuses, raw xy/two logits, per-frame timing and stdout/stderr.

NumPy 2.0.2 is the prior disclosed version-matched reconstruction, not historical
byte-exact environment reproduction. RECORD text digests normalize universal
newlines; raw-byte digests are separately retained. Existing torch 2.10.0+cpu
and all source/weights/data caches were reused. New download/install bytes: 0.
Prior acquisition ceiling remains 1,482,283,622 bytes (Git/CI traffic separate).

First-gate environment: AMD EPYC 9V74 80-Core Processor, Python 3.12.14.
Memory/storage/cgroup snapshots and per-group environment records are retained in
the machine-readable index and each real preflight receipt.

## Twenty evaluation groups

| Group | Points | Rows / finite scalars | Child wall s | Peak RSS KiB | First / later median frame s |
|---|---|---:|---:|---:|---:|
| g-031 | [0, 1] | 128 / 512 | 50.093573740 | 2,553,852 | 0.640301506 / 0.538333566 |
| g-015 | [0, 1] | 128 / 512 | 46.736641358 | 2,553,252 | 0.530333654 / 0.491286222 |
| g-001 | [0, 1] | 128 / 512 | 48.602505929 | 2,552,232 | 0.587564861 / 0.528608186 |
| g-005 | [0, 1] | 128 / 512 | 48.166087856 | 2,553,340 | 0.564324264 / 0.507085410 |
| g-038 | [0, 1] | 128 / 512 | 46.403710551 | 2,553,800 | 0.537907701 / 0.481793883 |
| g-004 | [0, 1] | 128 / 512 | 49.073086308 | 2,553,104 | 0.526372002 / 0.509389206 |
| g-021 | [0, 1] | 128 / 512 | 50.611271507 | 2,553,160 | 0.626339390 / 0.529834588 |
| g-010 | [0, 1] | 128 / 512 | 46.022736704 | 2,553,248 | 0.506122057 / 0.488418967 |
| g-014 | [0, 1] | 128 / 512 | 47.867375554 | 2,553,800 | 0.568034642 / 0.507587279 |
| g-007 | [0, 1] | 128 / 512 | 47.690456650 | 2,553,092 | 0.502999840 / 0.500246742 |
| g-011 | [0, 1] | 128 / 512 | 48.663157285 | 2,553,024 | 0.594029118 / 0.509736277 |
| g-049 | [0, 1] | 128 / 512 | 50.176856294 | 2,553,256 | 0.720326315 / 0.532600914 |
| g-028 | [0, 1] | 128 / 512 | 48.221343069 | 2,553,148 | 0.572698318 / 0.518362153 |
| g-025 | [0, 1] | 128 / 512 | 49.639692321 | 2,553,108 | 0.615167152 / 0.504007434 |
| g-019 | [0, 1] | 128 / 512 | 51.054465499 | 2,554,092 | 0.552243617 / 0.543921760 |
| g-006 | [0, 1] | 128 / 512 | 50.197169302 | 2,552,380 | 0.560005130 / 0.516350160 |
| g-037 | [0, 1] | 128 / 512 | 48.236158117 | 2,553,336 | 0.506920028 / 0.508318788 |
| g-008 | [0, 1] | 128 / 512 | 52.096595989 | 2,552,964 | 0.581939646 / 0.536232096 |
| g-027 | [0, 1] | 128 / 512 | 49.770825522 | 2,552,284 | 0.588349000 / 0.510755195 |
| g-024 | [0, 1] | 128 / 512 | 49.227646771 | 2,553,092 | 0.516032449 / 0.519309302 |

Full precision, runner/wrapper/supervisor elapsed times and raw CSV hashes are
in the [all-30 index](../results/0002-evaluation-completion/index.json) and
[evaluation-only index](../results/0002-evaluation-completion/evaluation-index.json).
Frame intervals are counted once per frame, not once per query; they omit some
runner overhead and do not establish real-time or full-query feasibility.

## Raw prediction identities

| Group | Partition | Execution source | Rows / finite scalars | CSV SHA256 |
|---|---|---|---:|---|
| [g-033](../results/0002-pilot-predictions/v2-reconstructed/g-033/receipt.json) | calibration | `05c2d7cb` | 128 / 512 | [`22bb58db053c0cb011c72cadee1e0d91012301ff6e15bb3264ae23e39cd9819e`](../results/0002-pilot-predictions/v2-reconstructed/g-033/video-033.csv) |
| [g-013](../results/0002-pilot-predictions/v2-reconstructed/g-013/receipt.json) | calibration | `5561cf5e` | 128 / 512 | [`8707198875187c1b43c06a07970eb5da72fc745f372f43afd87411daf72c6d5c`](../results/0002-pilot-predictions/v2-reconstructed/g-013/video-013.csv) |
| [g-022](../results/0002-pilot-predictions/v2-reconstructed/g-022/receipt.json) | calibration | `767aa160` | 128 / 512 | [`18b6b7735367d792e98a6d0780aaf0d7338da7f20afbbc96b95cdc3f4b2db278`](../results/0002-pilot-predictions/v2-reconstructed/g-022/video-022.csv) |
| [g-040](../results/0002-pilot-predictions/v2-reconstructed/g-040/receipt.json) | calibration | `3c6b26ac` | 128 / 512 | [`399c64850174219a298641eb277b2f36918e08dac590e1e8e6b25c4da71c3741`](../results/0002-pilot-predictions/v2-reconstructed/g-040/video-040.csv) |
| [g-039](../results/0002-pilot-predictions/v2-reconstructed/g-039/receipt.json) | calibration | `1f4d14f8` | 128 / 512 | [`999a427c893754a1d08e455f217b6472fb5d60089fd8af3db47c05e362bcdf9f`](../results/0002-pilot-predictions/v2-reconstructed/g-039/video-039.csv) |
| [g-020](../results/0002-pilot-predictions/v2-reconstructed/g-020/receipt.json) | calibration | `69737174` | 128 / 512 | [`147c2a9450526daa9806810d312cbeaf60480b08360c2eb78f94eb9f1cd1a0a2`](../results/0002-pilot-predictions/v2-reconstructed/g-020/video-020.csv) |
| [g-016](../results/0002-pilot-predictions/v2-reconstructed/g-016/receipt.json) | calibration | `d6eb1518` | 128 / 512 | [`331416f8db40d559fc1e5fcff1982eaabcc863d6ec66c6f088bbcebaff746c79`](../results/0002-pilot-predictions/v2-reconstructed/g-016/video-016.csv) |
| [g-032](../results/0002-pilot-predictions/v2-reconstructed/g-032/receipt.json) | calibration | `d6eb1518` | 128 / 512 | [`f5bbbffe94f0cce8adb5260dff4e6e0a0deffca2436fb9522b05c73c3df7bf3c`](../results/0002-pilot-predictions/v2-reconstructed/g-032/video-032.csv) |
| [g-035](../results/0002-pilot-predictions/v2-reconstructed/g-035/receipt.json) | calibration | `d6eb1518` | 128 / 512 | [`5c4b9f568999258ad62b743394e20a51c2db90325037860f0e2aaf52027f77e4`](../results/0002-pilot-predictions/v2-reconstructed/g-035/video-035.csv) |
| [g-026](../results/0002-pilot-predictions/v2-reconstructed/g-026/receipt.json) | calibration | `d6eb1518` | 128 / 512 | [`ab23af11347fa6c6a41ca9b5c527cde0e18cf01f494d939acbd1f029dce42da7`](../results/0002-pilot-predictions/v2-reconstructed/g-026/video-026.csv) |
| [g-031](../results/0002-pilot-predictions/v2-reconstructed/g-031/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`14f1337b1eb8c5d55d3595c920189ff691e55cd9392b7814780807175a08c609`](../results/0002-pilot-predictions/v2-reconstructed/g-031/video-031.csv) |
| [g-015](../results/0002-pilot-predictions/v2-reconstructed/g-015/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`c5ea796b7cff26ab756f53cd95b0df8d751df4bd150e7733d750dfeb1f4a4cff`](../results/0002-pilot-predictions/v2-reconstructed/g-015/video-015.csv) |
| [g-001](../results/0002-pilot-predictions/v2-reconstructed/g-001/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`5dd61ba8617a43d901794ba4a043809858c8af8495eb3be5b91269168750991c`](../results/0002-pilot-predictions/v2-reconstructed/g-001/video-001.csv) |
| [g-005](../results/0002-pilot-predictions/v2-reconstructed/g-005/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`a1697f3fbadf1c4ccd94b2f62078b0858dff9cd62d42ee2e3a31390b28d8678e`](../results/0002-pilot-predictions/v2-reconstructed/g-005/video-005.csv) |
| [g-038](../results/0002-pilot-predictions/v2-reconstructed/g-038/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`c2483ed0524a55a9a74759c2c6752470b1e1092c4c57f36fcc7f17019805a742`](../results/0002-pilot-predictions/v2-reconstructed/g-038/video-038.csv) |
| [g-004](../results/0002-pilot-predictions/v2-reconstructed/g-004/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`dfee558a2055be6a5f5f40d497593123c86389eafae1a91298235566898036c4`](../results/0002-pilot-predictions/v2-reconstructed/g-004/video-004.csv) |
| [g-021](../results/0002-pilot-predictions/v2-reconstructed/g-021/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`dc9c0a2389c2ed4259413ded234b1faf5409391908b3716c5ceb0af2d4344628`](../results/0002-pilot-predictions/v2-reconstructed/g-021/video-021.csv) |
| [g-010](../results/0002-pilot-predictions/v2-reconstructed/g-010/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`77622a1530432439d9b695a78ec0681642a65b675127bad631665334760e79f8`](../results/0002-pilot-predictions/v2-reconstructed/g-010/video-010.csv) |
| [g-014](../results/0002-pilot-predictions/v2-reconstructed/g-014/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`6675b00412b95f87d0c2dd8a9755a6f51ba62ec5dab90a618a40a227a259cae5`](../results/0002-pilot-predictions/v2-reconstructed/g-014/video-014.csv) |
| [g-007](../results/0002-pilot-predictions/v2-reconstructed/g-007/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`c93e9d00eb9c859e8a4eb13f601fcb7e22993bb3de5478a279a3b0ca0ee3e67a`](../results/0002-pilot-predictions/v2-reconstructed/g-007/video-007.csv) |
| [g-011](../results/0002-pilot-predictions/v2-reconstructed/g-011/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`02dd7397140b0ca6d867033053d6d55aa34c605328560b9d0b64abe0efa8c140`](../results/0002-pilot-predictions/v2-reconstructed/g-011/video-011.csv) |
| [g-049](../results/0002-pilot-predictions/v2-reconstructed/g-049/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`7ef76c4e5d1d0e5d7993f258a8b5e46ee2fd6164e6881d2f31e76b3fdc3a38af`](../results/0002-pilot-predictions/v2-reconstructed/g-049/video-049.csv) |
| [g-028](../results/0002-pilot-predictions/v2-reconstructed/g-028/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`cb61108ffbdcf5b377a1d0f1b05154586f9ed0de1188986eedcb49269a4e0a4e`](../results/0002-pilot-predictions/v2-reconstructed/g-028/video-028.csv) |
| [g-025](../results/0002-pilot-predictions/v2-reconstructed/g-025/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`363cd827a5b472d8ddf9c200eb2790692a27637eefc3f06571a4a619f1f656af`](../results/0002-pilot-predictions/v2-reconstructed/g-025/video-025.csv) |
| [g-019](../results/0002-pilot-predictions/v2-reconstructed/g-019/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`4b81622b2cc58c57002df2094c1f79b7a54039b7c03c5f786eaab8b2f7bdd71a`](../results/0002-pilot-predictions/v2-reconstructed/g-019/video-019.csv) |
| [g-006](../results/0002-pilot-predictions/v2-reconstructed/g-006/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`2f2cc6677588a2ba395b32f1f46bc2eedb634e925475cfc1d43e6ab38ef65e4c`](../results/0002-pilot-predictions/v2-reconstructed/g-006/video-006.csv) |
| [g-037](../results/0002-pilot-predictions/v2-reconstructed/g-037/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`4951a52918e1239c17b9f9f41059fecede4a5a37498078e45a3295d1b5592dd7`](../results/0002-pilot-predictions/v2-reconstructed/g-037/video-037.csv) |
| [g-008](../results/0002-pilot-predictions/v2-reconstructed/g-008/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`d8d7556f02cceb48a156e9334c6c7d3dce5d2ac2dae4197d3ca9066b9e90f794`](../results/0002-pilot-predictions/v2-reconstructed/g-008/video-008.csv) |
| [g-027](../results/0002-pilot-predictions/v2-reconstructed/g-027/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`61831f9e809f406f59129cfdc1c1519e6e2dbee2f8c02a2b544dc61b30b52fb4`](../results/0002-pilot-predictions/v2-reconstructed/g-027/video-027.csv) |
| [g-024](../results/0002-pilot-predictions/v2-reconstructed/g-024/receipt.json) | evaluation | `6bdce825` | 128 / 512 | [`6a04d0a2885b9804cc9def8622460bc57b0b513746fde20f510be4199522ebd0`](../results/0002-pilot-predictions/v2-reconstructed/g-024/video-024.csv) |

## Verification, adjustments and limits

The [baseline](../results/0002-evaluation-completion/baseline.json) covers all
**422 tracked results/docs/protocols**, including prior calibration/development
outputs, failures and reviews. All remain byte-identical. Each later launch also
checks earlier current-batch artifacts. All 100 generated per-group orchestration
scripts were statically checked before model execution for syntax, partition,
group/member, CLI, source/count/next-group, shared-software and phase bindings.
The scientific 29-file freeze and resource limits did not change.

Non-frozen orchestration adjustments bind the reviewed PR #12 scaffold to all
20 evaluation groups and source/baseline/output paths, keep calibration counts
at 10 and derive evaluation progress independently. The derived-text publication
size guard is 1 MiB instead of 100,000 bytes to admit cumulative hash maps;
binary/secret/private-path/raw-input guards remain. A pre-model static generator
assertion initially conflated calibration allocation with inference partition;
its source and observed exit-1 error are preserved in preparation-failure.json
and preparation-v1-source.py. The guard was corrected before any model launched.
No scientific gate failed and no model was retried to repair this static issue.

Independent stdlib checks read receipts/raw CSVs and Git/cache bytes only: complete
frame-query grids, original point IDs, finite/missing counts, numeric/file hashes,
latency counted once, all 29 frozen blobs, provenance, exits/resources and cycle-safe
source/log bindings. They do not import candidate/model code or re-decode labels.
Cache checks rehash 16 module entries and RECORDs, three upstream files and the
archive/checkpoint; native paths are checked, not every installed package byte.
Query selection relies on unchanged frozen code, verified input identity and
saved indices, without a second annotation decode.

Each preflight/inference child and independently parent-observed wrapper exited
0. Separate post-run parents capture actual verification/publication-audit child
exits and logs. The final batch audit uses one additional phase of the existing
g-024 post-run parent; no recursive supervisor. Its own parent program exit has
no further outer receipt. Current final audit receipt/logs and audit output are
excluded from their own hash maps; the final Git tree binds them after closure.
Archived source copies document original scratch entry points, not standalone
runnable archive entry points. Older g-033 lacks a separate parent wrapper-exit
receipt; this historical boundary remains explicit without replay.

Unchanged software had one shared availability probe/toy smoke under g-031;
later audits reference it. Work pytest available: false, Work tests executed: 0.
No installation. Shared toy smoke: 1240 frames in 2.93 seconds;
exact publication-head public Research checks CI must pass and its logs be read.

Next stage, only after independent review/merge and separate authorization, is
the full frozen evaluator on all validated predictions. This session did not
run it, fit thresholds, compare methods, tune parameters or change the freeze.
Shared object assets/camera, unknown scene relationships and unresolved checkpoint
training/model-selection exposure constrain this exploratory pilot. No independent
scene/object/viewpoint generalization, method efficacy, paper reproduction, SOTA
or real-robot result is claimed. Only small original derived artifacts are published;
no videos, weights or private data. No added fees, paid API/GPU/credits, account,
permissions or schedule changes. Keep this combined PR unmerged for parent review.
