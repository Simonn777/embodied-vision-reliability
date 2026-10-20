# 0035 native-bank feasibility — source only

Read against `DRAFT_SCOPE.md` SHA256 `c36e59a61991dd6db95f68faf158420133cafce2d07a483a481f9474b50b8586`. No imports, model calls, media/label/array reads, acquisition or scientific worker were performed. The proposed population is four complete clips / 446 frames; first-frame object counts and image dimensions remain unknown.

## Smallest reuse route

Use `work/0030-memory-content/campaign/common.py::load_runtime/finish` for the qualified CP312 runtime, offline guard, strict SAM2.1-tiny loading, CPU float32, Torch 4/1 threads, frozen weights and final hash. It builds the ordinary, uncompiled predictor using `experiments/0029-memory-information/native_fixture.py::CONFIG/OVERRIDES`. Keep model math and five overrides unchanged, including hole filling zero and native BF16 compact memory.

Reuse **the transparent `_track_step` observer idea**, not the complete `native_video.run` worker: `experiments/0029-memory-information/native_video.py:348–385` calls the original once, copies its seven-element `sam_outputs`, and returns the original answer untouched. Its capture/parity section at lines454–476 already records the existing three-mask bank without an extra native call.

The 0029 worker is a fixed single-target/eight-query experiment with forks, witnesses and extra controls. The 0030 cache is also single-target (`ids.min()`), saves only winner/maximum quality, keys observations only by frame, and decodes additional TRAIN labels. **Neither loop can be called unchanged for this screen.** Reuse their loader/observer patterns, omitting training labels, forks and feature-cache exports.

Pinned `SAM2VideoPredictor` already provides the requested multi-object implementation:

- One `init_state` per clip; add every positive, non-255 first-mask ID in ascending numeric order with `add_new_mask(..., frame_idx=0)`.
- One ordinary forward `propagate_in_video` iterator, no later prompts or new object IDs.
- `output_dict_per_obj` provides separate causal histories. The loop runs each object at `batch_size=1`.
- `_get_image_feature` caches the current frame's shared encoder output. Its initial frame is warmed once; subsequent objects reuse it. Thus there is one encoder per frame, not one per object, without a custom batched tracker.
- Identify each observer result by `(frame, object_id)`, using the native `output_dict` identity/current object context. The observer has no explicit object-ID argument. A `quality[frame]` map or “one capture per yielded frame” assertion would overwrite/misassociate multi-object results.

## Exact bank, ordering and output geometry

For every ordinary noninitial object-frame, the returned tuple is:

`(low_all, high_all, ious, low_selected, high_selected, obj_ptr, object_score_logits)`.

`low_all` is `[1,3,256,256]`; `high_all` is `[1,3,1024,1024]`; quality is `[1,3]`. Tracking requests multimask because the frozen config enables it and zero points lies in its allowed `[0,1]` range. The returned slots 0/1/2 correspond to the decoder's multimask tokens 1/2/3, not its single-mask token0.

Let `k = argmax(ious[0])`, with first-index tie behavior. Require exact equality between the selected low/high tensors and slot k; selected low must equal the stored native `pred_masks` because filling is disabled. B is the actual yielded native output for that object. M1/M2 are **the remaining returned slot numbers in ascending order**, not quality-ranked alternatives. Record all qualities, k, both alternative slot IDs and presence logits.

Original-resolution logits come **directly from low logits** through the pinned `_get_orig_video_res_output`: bilinear resize to `(video_height, video_width)`, `align_corners=False`. Do not resize the 1024 high logits again, resize a thresholded mask, crop/unpad, or interpolate probabilities. Mask support uses `logit > 0`.

Keep the actual yielded B authoritative. Reconstruct the selected stack in the same native `[object_count,1,256,256]` shape and require its native-helper resize to equal the yielded stack. A bank resized in a different channel/batch arrangement must not create a new unqualified bit-equality assumption; retain any selected-channel difference descriptively. Do not overwrite B with a numerically different bank rendering.

The only real support arms are:

1. B
2. B union M1
3. B union M2
4. B union M1 union M2

The observer never changes the native selected pointer, mask, presence or memory encoding. Unions are output-only. All candidate low/original logits and slot mappings can be saved; intermediate high logits need not be held for the whole video. Do not add an encoder/decoder to recover them.

## Initial, absent and empty cases

- **Initial masks:** `use_mask_input_as_output_without_sam=True` yields one supplied-mask-derived result, dummy quality1 and presence based on the supplied mask. An internal decoder call still computes the object pointer. There is no three-way initial bank to demand or invent. Record `bank_count=1`, no alternatives, and all arms equal native initial output. Its resize roundtrip need not be pixel-identical to the original annotation.
- **Predicted absence:** `_forward_sam_heads` replaces **all three** low logits with `NO_OBJ_SCORE` when object-score logit is not strictly positive. Preserve these already-gated outputs; do not expose the pre-gate decoder masks as extra proposals or invoke a new decoder.
- **Empty B with positive presence:** retain ordinary alternatives. A blanket “native mask empty ⇒ every arm no-op” would erase precisely the potential missing-support case under study.
- Empty or duplicate alternatives naturally produce duplicate arms; keep them. Invalid shape/nonfinite output is a technical error, not a reason to select a different object or omit a frame.

## Multi-object overlap is an unresolved scientific contract

The qualified predictor has `non_overlap_masks=False`; `non_overlap_masks_for_mem_enc=False` is also unchanged. Per-object binary masks can therefore overlap. The normal iterator concatenates per-object selected logits and returns them; it does **not** silently turn them into a unique-ID label map under these settings.

The native optional output rule chooses the highest logit across objects and suppresses losers to at most −10. It does not alter stored per-object tracking history. Applying an equivalent external rule to *every* real arm is possible, but would be a newly explicit output convention. It cannot be described as the previously qualified default.

One coherent possible convention is: for each object take the max logit over its arm's included slots; then choose the largest positive score across objects, ties by ascending original object ID, else background. This is a proposal for the parent to settle, **not an implemented default**. The same composition must apply to B and all unions. A newly added candidate can steal another object's pixels under this rule, so per-object support inclusion does not imply globally monotone accuracy or a valid independently composed oracle.

Accordingly, the parent must fix whether the oracle optimizes joint label-map choices or independent binary object supports, the metric/aggregation weights, and its relationship to the common real-control composition. No new oracle rule is inferred here.

## Counts and bounds

Let object counts be `o1..o4` for the four complete clips of lengths86/90/201/69. With no later interaction, expected image-encoder calls are **446**. Native `_track_step` calls are

`C = 86*o1 + 90*o2 + 201*o3 + 69*o4`.

There are `C − (o1+o2+o3+o4)` ordinary three-bank object-frames, plus one initial single-bank row per object. The initial pointer route still calls the mask decoder; no claim of zero initial decoder cost. Native memory encoding also occurs once per object-frame, including initial preflight. Capture adds **zero encoder/decoder calls**, but copying, resizing, hashing and serialization still cost time.

`num_maskmem=7` limits the attention inputs, **not total retained history storage**. The unchanged synchronous JPEG loader allocates `N×3×1024×1024×4` bytes. The 201-frame clip alone needs2,529,165,312 bytes (2.355GiB) for this image tensor. A compact BF1664×64×64 memory plus float32 selected256×256 logits contributes about0.75MiB per object-frame, before positions/pointers/container overhead, workspaces and output buffers. Keep clips serial and release the completed state; no full-state deep-copy forks and no all-frame candidate tensors in RAM.

Thus neither446frames nor the earlier single-object timing proves the whole four-clip operation fits1800s/8GiB. Actual all-ID counts, dimensions, serialization volume and one complete bounded runtime remain necessary. No timeout splitting, object subset, resolution reduction or history pruning can be silently substituted after a limit failure.

## Concrete items still to settle

1. Common multi-object composition/ties and oracle/aggregation definition.
2. First-mask ID/dimension admission after separately authorized input access; this determines C and memory/storage feasibility.
3. A small multi-object manufactured qualification of `(frame,id)` capture association, initial single-bank handling, selected-stack parity, absence and all four arms. Existing single-object qualification does not prove these new orchestration details.
4. Durable output layout/size bound for all object-frame low/original logits, without new full-feature caching.

VOST255 must remain void for evaluation. No labels beyond the first-frame prompt belong in this prediction operation. The four-clip cohort and metric conventions remain prospective; no claim of independent recording groups, tiny accuracy or method novelty follows.

## Exact source observations

| Read file | SHA256 |
|---|---|
| `work/0030-memory-content/campaign/common.py` | `751355a33b6ed468208d8f5aa92c8b4b72522f534159814928799e56634407c7` |
| `work/0030-memory-content/campaign/cache_baselines.py` | `5d16b4658cef5d451e88f33b3b3f99562a9c782abf199b1e22c1c9c5b0b8f0da` |
| `R/experiments/0029-memory-information/native_video.py` | `4af3e0e5b26327d4ecb95b4644b94e13c257a38c928dbb09f23bd95151001d0a` |
| `R/experiments/0029-memory-information/native_fixture.py` | `4e20a61150407c575eedff41309281923f9a4b57fa3f34609153e645dde7ad5f` |
| installed `sam2/modeling/sam2_base.py` | `69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41` |
| installed `sam2/sam2_video_predictor.py` | `912555920a77f72ded07839efa33fcceecc79f3523abbec783f0be46ee2c55c2` |
| installed `sam2/configs/sam2.1/sam2.1_hiera_t.yaml` | `f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d` |
| installed `sam2/utils/misc.py` | `01600c01c161cd079d7106fb1d4da845cf91aa31ab2bcecaf8cb151b6d6d20a2` |

`R` is `work/embodied-vision-reliability`; installed source is under `work/0029-runtime-02/installation/sources/sam2`. Paths are relative to the shared project directory. The inspected SAM2 source pin is `2b90b9f5ceec907a1c18123530e92e794ad901a4`; hashes above bind the actual local text read.
