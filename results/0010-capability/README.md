# 0010 cloud CPU capability and solver admission

The fixed official CLIP ViT-B/32 probe and artificial StatA checks completed on
the existing free Work cloud CPU. **This is resource and numerical admission,
not classification improvement. No 128-image quality study or manuscript result
is reported here.** The exact correction freeze and CI preceded execution;
see [publication verification](../../docs/0010-cached-admission-publication-verification.md).

| Actual stage | Captured outer wall seconds | Peak RSS KiB | Outcome |
|---|---:|---:|---|
| Cached source/asset/runtime/process guard |2.380593|14,676|PASS|
| Eight-image / ten-prompt probe |8.375169|1,301,056|8 image calls and1 text call completed|
| Saved-feature check |0.944420|64,832|PASS; no model calls|
| Artificial solver functional |1.635440|239,760|20 tests;0 failures/errors/skips|
| Three artificial timing fits |1.639733|263,140|3 states/6 authored executions; admitted|

These enclosing wait4 times differ from child-only or individual-fit times.
The raw command, execution and closed logs are retained under cloud/scratch.
Python3.12.14, Torch2.10.0+cpu and NumPy2.0.2 ran with four intra/inter-op
threads, CPUs0–3 and8GiB; no CUDA or paid service was used.

The [capability receipt](cloud/scratch/0010-capability-output/receipt.json)
binds the original config/source, eight excluded names, ten prompts/token IDs,
12 assets and four saved float32 arrays. Their shapes are8×512(image) and
10×512(text), raw and normalized. Saved-only review found finite outputs and
exact raw-to-normalized agreement, with maximum unit-norm deviation
1.1920928955078125e-7. No classification score was computed.

Using the prescribed setup-plus16-times-eight-image preprocessing/forward
formula, the encoder projection is **25.022365751745383 s≤450 s**.
The [three artificial timings](cloud/scratch/0010-solver-timing/child/receipt.json)
are T63=0.11852241800079355 s, T641=0.08900357699894812 s and
T644=0.06562033699447056 s. The unchanged formula
1.5×(128×T63+2×T641+2×T644) gives **23.220175998132618 s≤360 s**.
Each timing fit includes two authored executions for reference parity; T63
also includes its held-out readout. These projections are not a real-study
runtime guarantee, representative semantic geometry or an accuracy benefit.

The original non-frozen cache-emptiness guard's exit1 remains unchanged. Its
residual partial and three temporary copies were retained, hash/stat checked,
and excluded from model inputs. Their origin remains UNKNOWN. The new guard
recorded14 visible processes,121 successful fd stat/readlink observations and
one explicit disappearing-FD ENOENT race; no permission/other gap or protected
holder was reported. This is a bounded visible-process snapshot, not proof
against hidden, prior or future writers.

One additional **pre-scientific inventory mapping attempt exited1** on the
export alias control/assets.command.json. Its actual tool return and the
[alias correction](cloud/scratch/0010-cached-admission-control/transport-mapping-correction.json)
are retained. Correcting that alias then verified65 old files before the
single guard/probe sequence. It was not a scientific retry. Do not describe
every command in the session as exit0.

Cloud preservation checks report3,046 old tracked files,45 old marker/preflight
records,65 prior exported records and both frozen source sets unchanged. The
parent cannot independently read all historical private cache bytes; that
boundary remains. The new probe marker's161-byte identity is in the postcheck,
but its raw bytes were not exported. Original failures and one-shot state remain.

The parent verified102 exported public files/357,715 bytes and retained
[actual tool returns](parent-tool-observations.json), distinct from assistant
summaries. Three saved NPZ files/264,232 bytes were separately transferred for
private independent arithmetic/hash review and are not rehosted here. The
[copy manifest](cloud/copy-manifest.json) records exact public bytes and omissions.
No image/checkpoint/wheel/overlay is published. New scientific acquisition is0;
previous direct body charges remain456,087,034 bytes; historical all-HTTP stays
UNKNOWN, and control/Git/transport is not a certified wire-total measurement.

[Independent saved-output review](independent-review/README.md) passed769
assertions, including private-array arithmetic, actual command/log closure and
both resource formulas. It supports only the unchanged bounded development
screen; full private-cloud inventories and raw marker bytes remain outside the
local review. An initial local lookup-path failure is preserved separately.

Next is the separately frozen [128-image development screen](../../protocols/0010-real-data-headroom.md),
using32 new image batches, zero extra text calls, five fixed arms and complete
coverage before its one score. These admissions do not relax its improvement
threshold, unsupported-case policy, independent-data requirement or novelty bar.
