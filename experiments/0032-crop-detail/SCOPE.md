# 0032: one fixed crop-detail baseline

Source preparation, 2026-10-20. No scientific execution is admitted by these
files. This is a known prompt-refinement adaptation and additional-compute
baseline, not a novel method or a reopening of the failed 0031 lookup.
The parent controls the execution specification and admission.

## Fixed observations and outputs

Use the already completed native six-parent cache and the same original RGB.
TRAIN parents are dog-agility (25 frames), planes-water (38), dogs-scale (83),
and dog-gooses (86). For each use the existing 12 anchors
`1 + floor(i*(N-5)/11)`, `i=0,...,11`, and futures `+1,+3`: 96 event references,
87 unique frames. DEV is all 89 cat-girl and all 65 upside-down frames. Thus
the worker produces exactly **241 original-resolution predictions**. DEV
frame zero is an unchanged native copy, not refinement of annotation input.
Native-empty masks are also unchanged no-ops. No frame is selected by error.

The worker never opens an annotation payload, including initial annotations.
It validates the completed mapping and data-index metadata, completed cache
receipt/outer/log chain, RGB, native binary predictions and selected-logit
cache files. Each cached frame `.pt` also serializes native image features and
memory fields; loading that container is unavoidable with this existing
format. The worker uses only `out.pred_masks` and, descriptively,
`out.object_score_logits`; it neither uses its image/memory features nor opens
anchor/GT-memory caches. No future observation enters a refined frame.

Every frame first reconstructs native original-resolution logits using
float32 bilinear `align_corners=False` and verifies exact zero-threshold
parity with its saved native binary mask. A parity failure is technical
failure, not a frame to discard.

## One fixed refinement

For a nonempty, noninitial frame:

1. Compute inclusive foreground extrema. Pad each side by `ceil(0.10*w)`
   and `ceil(0.10*h)`, where w/h include the last foreground pixel. Clip the
   rectangle to image bounds and slice with exclusive right/bottom indices.
2. Choose the foreground pixel maximizing Euclidean distance to background
   after adding a one-pixel zero exterior. OpenCV's precise L2 transform is
   used, with the first row-major maximum breaking ties. Use this single
   positive point in crop coordinates. The box is the full crop's pixel
   centers `[0,0,crop_width-1,crop_height-1]`.
3. Slice native original-resolution logits to the identical crop. Resize to
   256×256 with float32 bilinear `align_corners=False, antialias=True`, then
   clamp to [-32,32]. Supply shape 1×256×256; do not sigmoid or binarize.
4. Use one `SAM2ImagePredictor` around the one existing strictly loaded
   CPU-float32 tiny model, with threshold/hole/sprinkle all zero. Each
   `set_image` independently applies the pinned RGB-to-1024-square transform.
   Invoke `predict` once with point, box and mask prompt, `multimask_output=True`,
   `return_logits=True`, `normalize_coords=True`.
5. Select the greatest of the three predicted IoUs, first native index on
   exact ties. Threshold the returned original-crop logits at zero. Replace
   that crop in the native full mask; leave all pixels outside it untouched.
   An empty refined result is retained. There is no feedback into tracking,
   prompt retry, weight update, tuned mixing, or GT selection.

The pinned image API directly invokes the prompt encoder/mask decoder. It
does not use the video `use_mask_input_as_output_without_sam` bypass. The
worker explicitly forbids `_track_step` and memory-encoder invocation.
The image API does not apply video object-presence gating: its raw object
score is recorded, but cannot change the predeclared selection. Three
candidate crop masks, low logits and predicted IoUs are saved for diagnosis;
they are not a candidate oracle.

## Runtime, receipts and metric boundary

Run, only after admission, with the qualified interpreter:
`python -B -S -W error refine.py SPEC.json`. The spec includes the ordinary
campaign runtime fields (`repo`, `native_root`, `native_commit`, seed 0),
`campaign_root`, `evaluation_root`, fresh `output`, `source_sha256`, and six
descriptor bindings: `cache_receipt`, `cache_outer`, `cache_operation`,
`cache_worker_spec`, `mapping`, `data_index`. Source bindings must include
this worker, this SCOPE, sibling ADMISSION.md, campaign common.py and the three reused evaluation
admission/metric helper files. Runtime source/checkpoint admission and the
offline guard are reused unchanged through `common.load_runtime`.

Each eligible frame costs one additional full image encoder and one mask
decoder call. Hooks require exactly these counts and zero calls for no-ops.
Three candidates arise in that single decoder call. Native weights are
hashed before/after. All costs—including input verification, image decoding,
resizing, writing, and final verification—remain within the outer operation
time; per-frame encoder/decoder times are descriptive. The earlier six-video
cache's entire elapsed cost is retained as an inherited-history reference
(it includes cache and GT-memory preparation), not claimed as a clean
matched complete-pipeline timing. No free baseline or speed claim.

The source proposes one complete operation within 1800 seconds / four CPU
threads / 8 GiB under the existing supervisor; feasibility still needs actual
measurement. The image API path has not been inferred qualified merely from
the earlier video API qualification. Six small manufactured geometry tests
are prepared but not executed by this source-preparation task.

`receipt.json` has schema `evr0032.crop-detail.v1`. Its six `rows` expose
sequence, parent_id, role, original frames, shape, target_id, anchors, and
`predictions` (frame, relative path, bytes, SHA256, kind="prediction"). Each
prediction NPZ contains exactly bool `masks` of shape 1×H×W and integer
`object_ids=[target_id]`. `outputs` is precisely the flattened predictions.
`diagnostics` is the separate candidate-file inventory. Per-frame records
retain prompts, pIoUs, selection, no-op/empty status, parity, call counts and
timings. Failures retain partial receipts and no complete seal.

After all 241 outputs and source/input checks, the completed receipt is
written once, then `seal.json` binds that receipt, both output inventories,
source/spec hashes and count 241. No current GT is decoded by this worker.
Only a separately admitted metric stage may open current annotations after
verifying the complete seal. It evaluates all 96 TRAIN event references and
all 150 interior DEV frames, retaining parent results and native comparisons.

## Interpretation

All six parents remain permanent development. A gain would establish a useful
fixed extra-compute baseline, not generalization, novelty, or manuscript
readiness. This differs from the full SAMRefiner recipe: no feature-adaptive
elastic box, negative point scheme, Gaussian dense prompt, split/merge or IoU
head adaptation. See the prior source-only feasibility note in
`../0031-paired-boundary/crop-refiner-feasibility.md`.

Wrong identity, clipped missing parts, one-point fragmentation, dense-prompt
bias, square-resize distortion, crop seams, and predicted-IoU misranking
remain possible. Comparing improvement specifically attributable to cropping
would require a separate equal-call full-frame control; this single baseline
does not make that causal claim. No padding/point/prompt/selection tuning is
authorized from its outcomes.
