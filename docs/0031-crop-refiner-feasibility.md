# One-crop SAM2 refinement: feasible known baseline, not a new method

2026-10-20. Source-only; no scientific arrays, RGB, annotations, models or
checkpoint bytes read. No implementation or experiment admitted here. The
failed fixed 0031 lookup remains closed.

**Recommendation.** A fixed current-frame crop plus point/box/native-logit
prompts is implementable with the existing tiny weights. It is a plausible
stronger practical baseline because it re-encodes enlarged object detail and
invokes the learned decoder. It is neither a SAMRefiner reproduction nor a
free cached-feature correction. Success would demonstrate useful extra
computation, not innovation.

## Published recipe and adaptation boundary

Read the [SAMRefiner author manuscript, §§3.1–3.3 and §4.1](https://arxiv.org/html/2502.06756v1).
It combines distance-guided positive and negative points, feature-conditioned
elastic boxes, and a Gaussian-style dense prompt; it selects among multiple
masks by predicted IoU. Its optional ++ variant adapts the IoU head.
The paper explicitly warns that naive tight boxes and mask-only prompts can
fail. Its default foundation model is SAM ViT-H. The proposal below instead
uses frozen SAM2.1 tiny, one positive point, fixed padding and native logits.
It omits elastic-box reasoning, Gaussian prompts, negative-point sampling,
split/merge and IoU-head training. Call it a **single-crop SAM2 prompt-refinement
adaptation**, with these differences visible.

Read boundary: author HTML v1 full relevant sections; no final-PDF equivalence
check or author-code reproduction. One web-text request, unmetered text bytes;
no source archive or model/dataset download.

## Exact prospective dataflow

Use the already qualified SAM2 source at
`facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4`
and local `sam2.1_hiera_tiny.pt` (156,008,466 bytes; declared SHA256
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`).
No Hugging Face loader or new weights are needed.

1. **Native input.** Reconstruct original-resolution logits from the cached
   selected `out["pred_masks"]` with bilinear `align_corners=False`, exactly as
   the pinned video output path does. Verify threshold-at-zero parity against
   the saved native binary mask before using it. Do not substitute memory
   features, probabilities or a sign-preserving but rescaled binary mask.
2. **One crop.** From every nonempty native mask obtain inclusive foreground
   extrema. Let width/height include the final pixel; pad each side by
   `ceil(0.10*width)` / `ceil(0.10*height)`, clip the rectangle to the image, and
   slice with exclusive right/bottom bounds. Empty native masks remain native
   no-ops, counted explicitly. These proposed constants must be fixed before
   any baseline execution, not selected on outcome.
3. **Prompts.** Pick the native foreground pixel with maximum Euclidean
   distance to background, including a one-pixel zero exterior; break ties
   row-major. Subtract crop origin for its coordinates and label it positive.
   The box covers the padded crop in pixel-center coordinates:
   `[0,0,crop_width-1,crop_height-1]`. Thus the tight native box is not an
   additional hard localization hint inside the context crop.
4. **Dense logit transform.** Slice the original-resolution native logits to
   exactly the RGB crop, resize to 256×256 with float32 bilinear,
   `align_corners=False, antialias=True`, then clamp to [-32,32].
   Supply shape `1×256×256` as `mask_input`. Antialiased resizing matches
   SAM2Base's dense-prompt resize rule; clamping follows the image predictor's
   returned low-logit range. This is an explicit caller adaptation: the image
   predictor does not itself resize or sigmoid an incoming mask prompt.
5. **Native image API.** Wrap the strictly loaded, eval-mode, CPU-float32
   model in `SAM2ImagePredictor(model, mask_threshold=0,
   max_hole_area=0, max_sprinkle_area=0)`. A video-predictor model inherits the
   same base and can supply this interface; do not change its weights or
   invoke its video state. Call `set_image(crop_rgb_uint8)` once, then
   `predict(point_coords, point_labels, box, mask_input,
   multimask_output=True, return_logits=True, normalize_coords=True)` once.
6. **Selection/output.** Choose the highest predicted-IoU candidate, with
   lowest index on ties; retain all three scores/candidates for diagnosis.
   Threshold its returned crop-resolution logits at zero and replace only
   that crop in a copy of the native full mask. Do not choose via GT, blend
   by a tuned coefficient, rerun prompts, or feed the repaired output back
   into the video stream. A decoder-produced empty mask is an outcome, not a
   reason to silently restore native. Technical failures must be preserved.

## Native transforms and two important API traps

The inspected [image predictor](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py)
calls `forward_image`, prepares the three feature levels and adds
`no_mem_embed` where configured. Its RGB transform converts to [0,1], resizes
directly to **1024×1024** using the installed torchvision bilinear/antialiased
resize, and uses ImageNet means/stds. There is no aspect-preserving letterbox.
The prompt transform divides crop x,y by crop width,height and multiplies by
1024; the prompt encoder then applies its own +0.5 convention. Pass ordinary
crop pixel coordinates and do not add a second resize-center correction.
The native decoder's low logits are resized back to crop H,W with bilinear
`align_corners=False`; hole/sprinkle processing stays off.

**Mask bypass:** the tiny config has
`use_mask_input_as_output_without_sam=True`. Feeding `mask_inputs` through
video `_track_step` can therefore return the provided mask rather than refine
it. The image predictor avoids that branch and directly calls the prompt
encoder and mask decoder. Merely using `add_new_mask` is not this baseline.

**Presence difference:** the image API ignores the decoder's extra object
score output, whereas video `_forward_sam_heads` uses it to hard-gate masks
when absent. Keep this native image behavior explicit; do not silently bolt
on an unqualified presence gate. The predeclared empty-native no-op prevents
spurious image-predictor resurrection in those empty frames, but does not
guarantee correct identity in nonempty ones.

## Causality, cost and limits

The refiner sees current RGB and the current causal native mask/logits. The
latter carry past information indirectly; “current-only” means no extra
reference frame, future frame or direct memory read. Initial identity remains
the native tracker's responsibility. Native memories, pointers, positions and
future emitted masks must be byte-identical to the baseline stream.

Each nonempty processed frame costs **one extra full 1024-square image
encoder and one prompt-decoder call**, yielding three masks in that decoder
call; plus decoding, resizing and prompt construction. Cropping does not
reduce the encoder's token grid. Cached full-frame features cannot replace
crop encoding without changing this baseline. Report incremental time and
baseline-plus-refinement cost, not a matched-cost speed claim. The qualified
tiny CPU runtime makes this plausible, but the image API/crop path still
requires a small manufactured qualification and measured bounds before
claiming ≤1800 s / 8 GiB.

Likely failures: a false-positive native box/point reinforces the wrong
identity; 10% padding cannot recover distant missing parts; clipped crops
can truncate new foreground; native logits may overconstrain correction;
one positive point can prefer one fragment; square resizing changes aspect
ratio; predicted IoU can misrank candidates. A fixed baseline success would
justify comparing later proposed mechanisms against it. Attributing success
specifically to cropping would additionally require an equal-call full-frame
prompt control, not an oracle candidate selection.

Local source checks: image predictor SHA256 `f13e5f9d…32b029ba`;
transforms `ba3a64f4…a6d1146`; SAM2Base `69a46b44…fdc6e41`;
prompt encoder `4965ccb4…8ae1d40d`; mask decoder `ca3523c5…916e099a`.
These were source reads/hashes only. No runtime imports, image decoding or
scientific execution occurred.
