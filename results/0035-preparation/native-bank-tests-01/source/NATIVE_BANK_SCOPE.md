# 0035 native bank — implementation and metric addendum

Source preparation only. No model, media, scientific array or annotation has been
read to write this worker. No acquisition or scientific operation is admitted
by this file.

This addendum preserves `native-bank-feasibility.md` unchanged at SHA256
`8a8c9f567f5f3a9fff99c3e9e14b98074485313914a210db41d2c5eb11fa3e83`.
Its statement that VOST255 must be ignored is **superseded**: the pinned author
loader replaces255 with background and the evaluation caller supplies no void
mask. Primary author-compatible evaluation therefore treats255 as background;
any ignore255 sensitivity is separate. See `metric-oracle-review.md`, including
the pinned author source URLs. The prediction worker prompts only contiguous
positive initial IDs1..max, excluding0/255, and reads no later labels.

## Fixed model and operation

`native_bank.py --spec JSON` imports the unchanged, hash-pinned
`0030-memory-content/campaign/common.py` and calls its `load_runtime`/`finish`.
The same qualified CPU float32 SAM2.1 tiny, original configuration/overrides,
offline guard, Python3.12.13, Torch4/1 threads and frozen weights apply.
An external reviewed supervisor must enforce the complete1800s/8GiB operation.
No new backend, batching, autocast, image-size, object-count or history cap.

Production requires exactly these ordered clips:4691_cut_guava86,
817_cut_onion90,9675_fold_clothes201,1934_tear_film69. All446 frames and every
first-mask object are processed, one clip in RAM at a time. Native synchronous
loading holds a complete clip's image tensor. Its largest201-frame tensor is
2.355GiB before model, history and temporary outputs; total feasibility remains
unmeasured. No partial-success evaluation or deadline-based truncation.

Required spec fields: existing common loader fields `repo`, `native_root`,
`native_commit`, `source_sha256`; plus `common_path`, `metrics_path`, `output`,
`data_index:{path,bytes,sha256}`, `max_output_bytes` (positive integer,
prospectively determined for the entire operation). Seed defaults0 as in the reused loader. All
source-map paths are absolute and are rehashed before/after the operation;
the map must include worker, this addendum, unchanged common and shared metrics.
Data index JSON has `sequences`, each exactly
`{sequence,role:"TRAIN",rgb:[descriptors],initial:descriptor}`, optionally
`parent_id`/`recording_id` metadata. Each descriptor contains absolute `path`,
positive `bytes`, SHA256 and ordinal integer `frame`; extra original-stem
metadata is allowed. RGB ordinals are0..N−1 and initial.frame=0. No noninitial
annotation list is accepted. Unknown recording grouping remains unknown.

## Native capture and composition

One ordinary native inference_state per clip; add IDs in ascending order and
propagate once. The native encoder cache is shared, histories stay per object.
The transparent `_track_step` wrapper identifies the object's exact history
dictionary, copies low-bank/quality/presence, checks native selected low/high,
and returns the original result without modification. It never changes
selected masks, pointers, memory encodings or model state. Initial outputs
have one bank entry; ordinary object-frames have three post-presence entries.
All native absent/duplicate/empty outcomes are retained. Positive-presence
empty selected masks do not disable their alternative masks.

Native B comes from actual yielded original-resolution logits. The selected
low stack of shape[objects,1,256,256] is resized with the original predictor
helper and must be bit-identical to that output. Alternative1/2 mean ascending
original decoder-slot order after excluding each object's selected argmax,
not quality ranks. Each alternative stack uses the same[objects,1,256,256]
layout and direct native bilinear original-resolution resize, align_corners
false. No high-to-original double resize or probability/mask interpolation.

Shared `metrics.compose_controls` receives original logits in canonical
[objects,3,H,W] B/M1/M2 order and sorted IDs. The only four output arms are
B, B+M1, B+M2 and B+M1+M2. Per object use the maximum included logit; then
the greatest strictly positive object score wins, exact ties to lowest ID;
otherwise background. Both native overlap flags remain false internally.
This external categorical convention applies to B too, without feedback.
Initial single-bank outputs are duplicated for composition; all four arms
are identical and initial frames are diagnostic, outside author scoring.
Oracle selection is wholly separate and is not implemented here.

## Outputs and preservation

One compressed NPZ per frame, no pickle:

- `low_bank`:float32[O,K,256,256], raw decoder-slot order; K1 initial/K3 later.
- `native_original_logits`:float32[O,H,W], actual yielded B.
- `object_ids`:int64[O]; `selected_slots`:int64[O];
  `alternative_slots`:int64[O,0] initial or[O,2] ordinary.
- `predicted_quality`:float32[O,K]; `object_score_logits`:float32[O].
- `labels`:uint8[4,H,W] in the fixed arm order above.

All original alternative logits are exactly reconstructible from low bank and
slot indices using the same qualified Torch/native resize and O×1 layout.
No high masks or vision features are cached. Receipt rows preserve all frame
identities/slot maps/quality/presence/parity flags and output SHA/size. Counts
record intended/completed image encoder, track step, mask decoder and memory
encoder calls. Expected image encoders446; other calls sum N_i*object_count_i,
including initial pointer-decoder and preflight memory work. Counts are not
an end-to-end speed claim. Timings separately charge loading, native calls,
parity/composition, serialization, state hashes and admission/rechecks.

`bytes_written` sums completed NPZ file sizes; `partial_output_bytes` retains
unfinished NPZ extent on failure. A bounded ZIP writer refuses writes beyond
the remaining cap, with receipt bytes deducted; current receipt and final
seal sizes are also checked against the same total cap. JSON failure receipts
may marginally exceed an exhausted cap to preserve the failure; no new NPZ
write or success seal follows. This is a generated-output limit, not a network
transfer budget. Initial IDs/dimensions and the cap are settled before science.

Failures preserve receipt/current invocation intent, previous outputs and
pending files; no success seal is written. Successful full operation records
immutable model hashes and a seal binding receipt, source/spec/index and every
frame output. No logit/annotation metric is computed by this worker.

For manufactured qualification only, reusable
`initialize_record(output,spec_sha256,source_sha256,max_output_bytes=256MiB)`
(last argument keyword-only) makes a record, and
`execute_sequence(runtime,common,sequence,output,record,*,compose_controls)`
processes one fresh clip directory inside record.output. This helper does
not set scientific eligibility or weaken production's fixed-cohort admission.
It records the complete final native state hash before releasing the clip,
so the parent can compare a second unobserved native propagation. No
qualification or tests have been executed during source preparation.
The state descriptor and tensor hashing format are identical to qualified
`native_fixture.state_hash`; tensor byte memoryviews are streamed in1MiB
chunks to avoid the old hash helper's full-image-tensor `.tobytes()` copy.
