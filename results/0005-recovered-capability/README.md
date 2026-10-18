# Native TAPNext++ CPU capability: eight frames completed

The reviewed recovery completed once on October 18, 2026. The unchanged float32
CPU adapter processed development video 0, frames 0–7 / queries 0–7: **8 forwards,
64 point/frame outputs**. No warmup, support query, reset, accuracy scoring or
future-GT access occurred. This establishes short-prefix feasibility only, not
long-clip capability, method improvement, real-video transfer or a paper result.
Operation, checking and writing were AI-assisted.

Execution main: `f0787ab50f5dfc31f8b8a58bdcb165a12d2d83a6`; scientific source:
`a501017ea7e5fe01652f650ae212526a38a9fdf9`; freeze commit:
`ec7e9e3604961d7c80b78f68fb99c55dfb432b4a`. The
[16-input canonical receipt](../../protocols/0005-native-recovery-freeze.json)
SHA256 is `916cdabbdc8d5be7dcb0060ca54216457d982a3bdc564c6509c79195e51966ef`.
The original seven-input freeze remains unchanged. Both freezes passed before
each preparation invocation and again in the native supervisor. Publication
follows execution and has a different commit.

| Actual operation | Exit / raw wait | Wall seconds | Peak RSS KiB |
| --- | --- | --- | --- |
| Verify before dependencies | 0 / 0 | 0.181045 | 14,080 |
| Dependency recovery, external | 0 / 0 | 2.988573 | 290,944 |
| Import child within recovery | 0 / 0 | 2.701883 | 290,944 |
| Verify before checkpoint | 0 / 0 | 0.180372 | 14,080 |
| Checkpoint recovery, external | 0 / 0 | 146.707927 | 22,396 |
| Native invocation, external | 0 / 0 | 55.082073 | 3,727,996 |
| Native child observed by supervisor | 0 / 0 | 53.080108 | 3,727,996 |
| Targeted synthetic software suite | 0 / 0 | 0.579527 | 23,096 |

Native external UTC: 18:31:59.587292–18:32:54.669583. Adapter elapsed was 50.420191
seconds: input checks/imports 6.204599; construction/checkpoint load 23.855715;
inference with validation/artifacts 20.356499. Forward calls totalled 19.415824
seconds: first 2.670397, subsequent seven mean 2.392204. These intervals are
contained in the enclosing measurements and must not be added to them. No warmup
or sustained-throughput estimate was made.

Hardware: AMD EPYC 9V74, Linux x86_64, Python 3.12.14. Principal stages used CPUs
0–3, four numeric threads, 8 GiB AS and `timeout --kill-after=5s 1800`; Torch
intra/inter-op counts were four. Cgroup memory limit was 8 GiB. Native peak RSS
was about 3.555 GiB, a wait4 high-water measurement rather than a whole-host memory
trace. Actual commands, environments, CPU times, source hashes, logs and wait
statuses are in `control/` and `native/outer.json`. Observer parents have no
additional durable outer wait4 receipt; that boundary is explicit.

## Assets, preservation and accounting

The [old failed attempt](../0005-native-capability/README.md) is unchanged.
Exactly five verified source files were copied into a new namespace; the existing
378-file overlay was reused. No installation or dependency download occurred.
The one import-only child passed versions/origins: Torch 2.10.0+cpu, NumPy 2.0.2,
torchvision 0.25.0+cpu, einops 0.8.2, Pillow 13.0.0. NumPy remains the previously
disclosed version-matched reconstruction, not historical byte-exact reproduction.

The sole new payload was the checkpoint's **first GET**, HTTP 200:
**2,532,282,370 bytes**, SHA256
`cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da`.
The frozen URL/revision and normal TLS were used, with no fallback or retry;
no checkpoint partial remained. The historical 4,823,387,101-byte ceiling, prior
9,179,574-byte payload, metadata allowance and reservations carry forward.
**Current conservative ceiling: 7,499,066,782 bytes**, below 10 GiB. Historical
ceiling plus observed old/new payload is 7,364,849,045 bytes, not a remeasurement
of historical traffic. Git traffic is separate. No added paid service was used.

Both complete manifests and ledgers are in `preparation/`. All **407 failed-cache
files**, recorded old markers/runtime identities and **1,651 prior result files**
matched. Saved review checked all 1,832 execution-main tracked files before state
updates. The old 1,769-file baseline is referenced by path/hash, with only 66
changed/new hashes added; it was not copied again. Empty partial persistence
remains unexplained; serial execution does not establish its cause. 0004 remains
stopped; 0003 remains deferred.

## Saved-output checks and retained review errors

`native/run/` contains eight original small NPZs and the adapter receipt.
`control/saved-predictions.csv` contains all 64 derived coordinate/visibility rows;
coordinate logits remain in NPZs. NumPy/stdlib review verified the complete grid,
**33,088 finite float scalars and 64 boolean outputs**, xy/yx reversal,
visibility-logit thresholding and file hashes. Stored state records have steps
1–8, twelve float32 CPU layers per step and a stable query hash. State tensors
were not retained: independent review checks stored schema/finite assertions and
hash consistency, not a recomputation of those tensors. The runtime adapter
checked actual tensors. The reviewer imported no model and decoded no raw dataset.

Two review-only bugs retain their actual exit 1/raw 256 evidence: the checkpoint
checker expected `checkpoint.json` instead of asset ID `checkpoint-256.json`;
the native checker interpreted dotted module IDs as paths. Separate version-2
checker files corrected only those interpretations and passed on saved bytes.
Their original source/log/wait records remain alongside the corrections. No
preparer, import probe, checkpoint GET or model operation was repeated; no frozen
source, parameter, scientific gate or output changed.

One relevant artificial suite passed **24 tests** (0.451 seconds within external
0.579527 seconds), with no real model or network. Parent CI 447 tests/92 subtests
is historical input-review evidence, not this run's local count. All sixteen
recovery inputs and the old freeze remain unchanged.

[Copy manifest](copy-manifest.json) binds **74** original small files to byte-exact
copies, including control/checker sources and failures. No checkpoint, wheel,
upstream source body, video frame, secret or signed URL is published. Next:
independent parent review, then a separately reviewed independent real-video
access/design step. Feasibility alone authorizes no further inference or efficacy
claim.
