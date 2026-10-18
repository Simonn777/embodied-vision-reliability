# 0005 native TAPNext++ CPU capability gate — prospective, not executed at freeze

This prospective operation tests whether a stronger published causal tracker can
be used within the existing no-additional-fees environment. It is not an efficacy
comparison, a new tracking method, a retry of 0004, or a paper result. The 0004
hard-zero candidate stays stopped. Nothing in this document permits old evaluation
tuning, PointOdyssey re-acquisition or a model-state intervention.

## Fixed native configuration

- Official google-deepmind/tapnet source revision
  `730cda1c730877cfedbe01bf87fb1cadb78a565d`, native `TAPNextPP` wrapper.
- Standard 256px checkpoint at HF google/tapnet revision
  `5d3fb48e76c5422841e38501514121e251beabb7`, file `tapnextpp_ckpt.pt`,
  2,532,282,370 bytes, SHA256
  `cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da`.
- CPU float32, no autocast, quantization, compilation, pruning, support queries,
  warmup, extra forward, changed weights or state reset. One model construction
  and official `weights_only=True` checkpoint load; no unsafe-load fallback.
- Existing cached synthetic development video0, frames 0..7, 256x256 RGB,
  original queries 0..7 initialized only from their frame0 coordinates. Source
  bytes may be hashed; no post-query labels are interpreted or used to choose a
  subset. This is an already exposed development clip, never held-out evidence.
- Convert RGB to the native wrapper's documented BGR input. Preserve query order
  and state across all eight frames. Exactly eight native forwards and 64 output
  point-frame rows. A read-only hook can observe those same forwards, not add more.
- Save every output coordinate, visibility/logit and raw coordinate logits without
  rounding. Check all values and the complete returned state for finiteness; save
  state shapes, dtypes, step and hashes after each frame, not the large tensors.

## Separated preparation and scientific attempt

The already completed read-only preflight showed Python3.12.14, torch2.10.0+cpu
and NumPy2.0.2 available, but torchvision/einops, the five specified source files
and new checkpoint absent. The existing BootsTAPIR runtime/cache is read-only.

Preparation uses a separate overlay and asset directory. Bind official source
and wheel URLs, versions, byte lengths and SHA256 in a reviewed configuration.
Only missing exact bytes are fetched once; reuse verified cached new assets.
Use no-dependencies/no-index installation from verified wheels so existing Torch
and NumPy cannot be replaced. Install the fixed Pillow 13.0.0 wheel in this new overlay; record and check its actual import origin and version explicitly. Any missing/unplanned
dependency or failed import ends preparation for review, without model execution.
Do not resolve package upgrades opportunistically or install a CUDA stack.

The conservative historical acquisition ceiling is 4,823,387,101 bytes from the
preserved 0003 ledger, including the previously downloaded PointOdyssey archive
even though that archive is not visible now. Add an explicit 128MiB allowance for
subsequent source/metadata inspection, exact new wheel and checkpoint bytes, and
record every actual acquisition attempt. The total must remain below 10GiB.
No cache-size substitution, old-asset redownload, paid API, cloud purchase or credit
purchase is permitted. An incomplete new transfer remains recorded; no silent
retry or replacement URL is allowed in this operation.

Before model construction, verify the separately published source/config/protocol
freeze and source import locations, all pinned assets/packages and no active
scientific process. Use an exclusive output directory and a new cache-wide 0005
attempt marker; existing marker/output requires inspection, not another launch.

The model child is limited to at most four allowed CPU cores/numerical threads,
8GiB address space and an outer 1,800-second hard timeout (5-second kill grace).
The actual preflight allowed cores 0..8; fix this operation to cores [0,1,2,3]. Record
thread variables, Torch thread settings, cgroup limit, versions and all imports.
Do not extend time/memory or fall back to GPU, MPS or a different model after failure.

## Outputs and decision

Save actual commands, source and input hashes, acquisition and preparation receipts,
constructor/load time, each native forward time, input/output/state-audit overhead,
total wall/CPU time, peak RSS and observed child/outer exit. Flush per-frame raw
outputs and partial receipt. Preserve failure/timeout and the marker without retry.

PASS requires all eight prescribed native outputs, finite full state/outputs,
correct state progression and exactly eight observed forwards within the fixed
resource limits. FAIL or partial completion means that this exact capability
configuration has not passed. Diagnose from saved artifacts before any newly
reviewed version. A successful first eight frames gives only measured startup and
short-prefix feasibility, not sustained throughput, full-video or benchmark success.

No coordinate accuracy, AJ, AJ_RD, improvement, model transfer or manuscript claim
is permitted from this capability gate. Compare future methods only after a separate
development/evaluation protocol with independent real-video groups is frozen.

## Frozen execution and accounting

The canonical `0005-native-capability-freeze.json` binds seven execution inputs
from a preceding source commit. The CLI verifies current files, index, HEAD and
that ancestor's Git blobs. Before either preparation command the cloud caller
runs its verify-only gate and compares the CLI hash to the published commit.
Use the existing Python3.12.14 interpreter and the new sibling `tapnextpp-cache`.
First run preparation stage `dependencies`, inspect its observed exit and bound
receipt, then stage `checkpoint` only if dependency import succeeded. Bound each
preparation invocation externally by 1800 seconds with a five-second kill grace.
Individual GETs have an 840-second deadline and no automatic retry; pip is limited
to 180 seconds and import to 120 seconds. Preserve partial outputs and markers.

Only five pinned upstream leaves may exist in the new source tree. Their parents
are namespace directories with no initializer; reject other entries, symlinks,
preloaded TAPNet or a separately installed regular package before imports. The
full installed overlay is hashed before/after the import gate and before native
execution, including metadata and binaries. An unchanged manifest is provenance,
not proof that every dependency is safe or of model accuracy.

The model supervisor verifies both successful preparation receipts and every
overlay file, then enforces its limits and launches a fresh child. A failed child
returns nonzero; timeout retains the partial receipt. The exclusive scientific
marker binds the canonical freeze and preparation receipts. The parent observes
wait status, CPU/wall/RSS, and stdout/stderr hashes; the cloud tool must separately
retain the parent command's actual exit and bound the whole invocation with
`timeout --kill-after=5s 1800`, including preliminary asset verification. No active scientific process may overlap.

Published capability artifacts include small output arrays and receipts, never
weights, wheels, source copies, private billing records or source video frames.
The acquisition ledger carries the earlier PointOdyssey bytes despite missing
cache. The fixed metadata allowance also covers small prior paper/source/metadata
reads not individually indexed. A successful entire plan reserves 7,499,066,782
bytes including one sentinel byte per asset; actual payload is recorded separately.
No new real-video data is authorized by this capability protocol.

This source/configuration/protocol must pass artificial unit checks and independent
review, then be committed and frozen separately before any acquisition or forward.
