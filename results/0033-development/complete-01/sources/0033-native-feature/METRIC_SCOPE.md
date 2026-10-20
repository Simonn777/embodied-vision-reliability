# 0033 prediction and metric contract

Source preparation only. Prediction and evaluation are separately admitted,
serial, complete operations, each bounded by four CPU threads, 8 GiB sampled
group RSS and 1800 seconds. No partial-population result replaces completion.
These are known supervised baseline controls, not a new method.

## Prediction

Use qualified CPython 3.12.13 with `-B -S -W error predict.py SPEC.json`.
Required fields are `output`, `python`, `campaign_root`, `evaluation_root`,
`guard_source`, `source_sha256`, and `bindings`. The ten binding descriptors
(`path`, `sha256`, `bytes`) are the six inherited
`cache_receipt/cache_outer/cache_operation/cache_worker_spec/mapping/data_index`
and four `fit_receipt/fit_outer/fit_operation/fit_worker_spec`.

The source map includes predict/head/train/SCOPE/ADMISSION, campaign common.py,
the unchanged offline guard and evaluate_jf/evaluate_baseline/evaluate_rollout.
Fit dependencies are rehashed from its saved source map. The old cache and
complete two-arm fit must have clean bound outer receipts. Both heads must be
terminal epoch12 checkpoints from this same fit with identical normalization,
initial-state identity and scientific configuration. Checkpoint state is loaded
strictly, CPU float32, via `weights_only=True`. Native SAM weights are not loaded:
their unchanged identity is inherited from the completed native cache.

The original guard is installed before `site.main()` and Torch imports. Only
the tiny learned heads are constructed; no SAM encoder, decoder, memory or
tracking call occurs. Each cached container supplies finest `vision[0]` with
shape `[65536,1,32]`, reshaped to `[1,32,256,256]`, native selected logits
`[1,1,256,256]` and the native presence scalar. Other serialized cache fields
are not used. There is no RGB/annotation decode.

Predict both visual and logit-only on the same 87 unique fixed TRAIN future
frames and all154 raw DEV frames: 241 frames and482 mask files in one complete
operation. Every frame verifies exact original-resolution native-zero parity;
both zero-initialized head paths verify exact low-logit equality and original
mask parity. Zero-head checks are extra arithmetic and their cost is charged.
Nonpositive native presence, native-empty masks and each DEV frame0 remain
native. Otherwise use the corresponding terminal head output, resized by
bilinear interpolation with `align_corners=False`, threshold zero. No threshold,
checkpoint, frame, arm or candidate selection uses GT.

Receipt schema is `evr0033.native-feature-prediction.v1`: six sequence rows,
each with `predictions: {visual: [...], logit_only: [...]}` and all requested
`frame_records`. Flat `outputs` uses sequence order, then frame order, then
visual/logit-only order. Each NPZ contains exactly original-resolution
`masks: bool[1,H,W]` and `object_ids: int[1]` with the native target ID.
All no-ops, changed-pixel counts, head calls, zero-head parity, per-arm costs
and partial failures are retained. Final head states and all source/input
bytes must remain unchanged.

The complete `seal.json` (`evr0033.native-feature-seal.v1`) binds the final
receipt, source/spec hashes, both arms, `frame_count:241`,
`arm_prediction_count:482` and the exact output inventory. No labels are
decoded before this seal; failures do not create a complete seal.

## Saved-output evaluation

Use qualified `python -B -W error evaluate.py SPEC.json`, with no model/Torch
import. Add four `prediction_receipt/prediction_outer/prediction_operation/
prediction_worker_spec` descriptors to the preceding ten bindings.
`baseline_reference` separately identifies the unchanged published native
DEV evaluation. The source map additionally includes evaluate.py,
METRIC_SCOPE.md and campaign cache_baselines.py. Existing `admit` handles
clean operation provenance; predictor metadata helpers are stdlib-only.

Before the first annotation decode, verify complete cache/fit/prediction
chains, both terminal identities, both482-file inventories and all241 encoded
label hashes. Structurally validate every prediction and all declared native
no-ops. Only then mark the receipt `predictions_sealed_before_gt:true`.
Decode each unique label once and compute native, visual and logit-only J/F.

Use unchanged official-formula original-resolution target-object J/F:
255 treated as background, native initial smallest-positive target ID,
boundary disk `ceil(.008*hypot(H,W))`, no union or resize. All96 TRAIN
anchor×{1,3} event rows are retained, including duplicates,87 unique frames
and four final-frame events. They are descriptive **fit-subset** accuracy;
each parent has24 equally weighted events and the four parents are equal.
All154 raw DEV rows are retained, with only first/last excluded from parent
means (150 interiors); the two parents are equal. Native per-parent and
overall J/F/JF must match the bound reference within absolute1e-12.

Continuation requires all five clauses: visual DEV JF at least .01 above
native; visual JF strictly positive against native on both parents; visual
mean JF strictly greater than logit-only; visual mean J and F each no worse
than native by more than .005. These are fractional units; multiply by100
for absolute percentage points. The clauses and both complete arms remain
visible regardless of outcome. Passing indicates known-baseline capacity,
not novelty or the unchanged two-source +3 official J/F paper gate.

`evaluation.json` contains complete rows, parent/equal-parent means, both
native deltas, visual-minus-logit-only, all clauses, and cache/fit/prediction
outer costs. Its own outer cost is separate. No nested timer is added again,
and the old cache is neither free nor a matched end-to-end timing claim.
