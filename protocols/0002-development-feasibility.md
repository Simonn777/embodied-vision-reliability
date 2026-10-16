# Protocol 0002-D: development feasibility, before real-weight execution

2026-10-16. AI-assisted development plan. Publish this protocol, configuration and
source on the authorized branch before acquiring/decoding data or running inference.
This is not the final evaluation protocol and does not authorize final inference.

Use TAPNet `730cda1c730877cfedbe01bf87fb1cadb78a565d`, official Online BootsTAPIR
checkpoint and RGB-Stacking archive from the existing audit, with exact object
generation, length, server MD5 and locally computed SHA-256. Code/checkpoints are
Apache-2.0; RGB-Stacking video/annotations CC BY 4.0. Keep upstream files and large
bodies in cache, retain notices, and publish original code plus small measurements.

Predeclare the **whole archive video index 0 as development**, including any frames
or labels subsequently inspected. Other indices 1–49 remain unassigned, never
inferred or visually inspected. The pickle container necessarily materializes
their arrays during restricted decoding; disclose this distinction. No evaluation
split, scene independence, or untouched-final status is asserted. Reserve groups
only after provenance/scene auditing. Never move video 0 into final evaluation.

Run acquisition, restricted decoding and inference as separate credential-free
processes with 900-second hard timeouts, a 6 GiB address-space limit and CPU affinity
to two cores. Set all five numerical thread environment variables to 2 and torch
intra/inter-op threads to 2. Record hardware, available memory/storage, packages,
source/protocol/config hashes, first/subsequent latency, wall time and peak RSS.
Total transfers including old dependency attempts/caches remain below 10 GiB.

Minimum upstream inference imports: torch/numpy/dm-tree/einshape, plus their small
runtime dependencies. Do not import official evaluation_datasets (eager training
stack imports). Pin patched **torch 2.10.0+cpu**; GHSA-63cw-57p8-fm3p identifies
<=2.9.1 as affected. Earlier 2.5.1 dry-run/install downloads occurred during setup,
but no checkpoint was loaded with that version; preserve/count them. Load with
weights_only=True on CPU, require strict keys and use_casual_conv=True, disable
gradients, eval/inference mode, deterministic seed 20261016. Never downgrade the
loader on failure. Restricted data unpickling rejects unapproved globals and
extension/persistent references; inspect ZIP sizes/paths and pickle opcodes first.

Inputs stay 256×256 RGB. Select the first two point indices visible at frames 0
and 4, without choosing by model outcomes. Begin with one query and four frames;
then 16 frames if resources permit. Preserve xy tracks and raw occlusion/error
logits. Every arrival group gets construct_initial_causal_state; descriptor input
is the arrival frame with local t=0, absolute arrival governs output masking.
Active groups use positive query_chunk_size=8; empty sets skip model calls.

On the same development clip measure: a full 16-frame run; a run with frames 8+
inverted (prefix through frame 7 unchanged); an 8-frame truncation; simultaneous
two-query batching versus one query; a second query arriving at frame 4 (pre-query
outputs NaN); repeat one-query execution. Report exact max absolute differences,
finite/mask shape checks, atol=1e-4, rtol=0, and failures without retuning tolerance.
Save first cold call and subsequent call latencies separately; no realtime or
whole-dataset feasibility claim follows. Repeat the minimal run in a fresh process
if budget permits; distinguish that from same-process repetition.

No position-error comparison or calibration on remaining videos is run here.
After observing resource/adapter results, draft the future exploratory protocol:
same frozen coordinates, native temporal confidence versus extra motion residual;
strict single-image question remains unresolved. Final code, checkpoint digest,
complete group split, endpoint, matched coverage and practical effect threshold
must be published/frozen for a later session. Unresolved scene relationships mean
an exploratory video-disjoint pilot, not scene-independent confirmation.

Tests must cover query coordinate order/local time, arrival/chunk gates, empty
groups, active masks, restricted deserialization, and numerical comparison gates.
Run the full existing suite and toy smoke, never the full protocol 0001 final run.
Failures and partial progress are publishable evidence. Stop at a reviewable PR;
do not merge in this session.
