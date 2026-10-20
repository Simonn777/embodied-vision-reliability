# Cached native-feature output head: baseline feasibility

Source-only, 2026-10-20. No implementation, specification, tensor/label reads, training, model calls, or new assets. This is a possible known learned baseline, not a crop-method rescue or an innovation claim.

## Decision and exact information contract

**Technically feasible as a small supervised capacity test.** A current-frame head could test whether existing native features support more accurate target masks when read with task-specific supervision. It would not introduce a new visual observation: SAM2 already uses the finest features in its own decoder. A positive result would establish useful supervised readout capacity on this development task, not previously inaccessible information, temporal recovery, or a new refinement principle.

The source confirms these available fields without opening cached tensors:

- `campaign/cache_baselines.py` saves each frame's `vision`, native compact `out`, and scalar pIoU. `vision[0]` is the flattened, projected finest image feature: `[65536,1,32]`, reshaped to `[1,32,256,256]`. It is **image-only**, not the memory-conditioned decoder embedding. The already declared feature layout is also documented in `0031-paired-boundary/SCOPE.md`.
- `out["pred_masks"]` is the selected, post-presence native low-resolution logit map, `[1,1,256,256]`. `out["object_score_logits"]` preserves the presence decision. These are native baseline outputs, not the former teacher or modified-memory rollouts.
- Pinned `SAM2Base.forward_image` applies the mask decoder's `conv_s0/conv_s1` before caching; `_prepare_backbone_features` flattens these maps. `MaskDecoder.predict_masks` already adds the finest feature to its upscaled embedding. Therefore reusing it cannot be described as adding a previously unused high-resolution input to SAM2.

Using a head **only on emitted masks** leaves native history, pointers, memory writes, and future native inputs unchanged. A sequential pass over the complete baseline cache is consequently the appropriate causal output-only policy, provided no future-frame statistic enters the head. It is not repaired-mask recurrent rollout or BPTT.

## One simple architecture, if later admitted

Let `F` be the 32-channel finest feature and `l` the native low-resolution logit. Use a single small head:

`hθ([F, sigmoid(l)]) = Conv3×3(32→1)(GELU(Conv1×1(33→32)([F,sigmoid(l)])))`

Emit `l + hθ` at 256², followed by the exact native original-resolution bilinear interpolation and threshold zero. Initialize the last convolution to zero, so the pre-training output is native. This head has **1,377 trainable parameters** and approximately **88.1 million convolution MACs per frame**, excluding interpolation, loss, activation, and backward work. These are architecture counts, not measured CPU timing.

The bounded sigmoid input avoids feeding the absence sentinel as an extreme feature value; the **residual skip retains the original logits**. For native `object_score_logits <= 0`, emit the native mask unchanged rather than allowing the head to override presence. This deliberately cannot repair false absence. Keep frame 0's supplied annotation/native output unchanged. No current GT enters the forward inputs, masking policy, threshold, or deployment selection.

Use ordinary dense BCE plus soft Dice supervision on **original-resolution interpolated logits**, with coefficients fixed before training. Nearest-neighbor downsampling of the target to 256² would discard some of the boundary information under investigation. TRAIN annotations supply supervision only; all two-DEV outputs must be produced without current labels. Four TRAIN recordings are four parents, not millions of independent examples.

This architecture deliberately retains the native output grid. It can move boundaries and exploit encoded local context, but it cannot establish recovery of arbitrary sub-grid structures absent from the cached feature representation. A negative result would reject this small readout/training configuration, not prove that no richer decoder or original-RGB refiner could work.

## Meaningful controls and interpretation

The strongest relevant **same-input learned control is ordinary direct supervised refinement itself**: the exact head above, frozen native features, same target logits, same labels, same budget, and no privileged teacher, special error labels, or future loss. Any later proposed special supervision or representation method must beat that control, not just bilinear native output. At this baseline stage there is no need to invent a second, algebraically equivalent “ordinary refinement” arm.

To attribute gains specifically to image detail rather than learned mask cleanup, the informative companion is a **logit-only information ablation**: same head dimensions and residual path, but the 32 feature channels are fixed to zero in training and inference. This is intentionally not a same-information comparator; it tests the value of `F` beyond target-mask shape. If both improve similarly, the evidence supports learned mask regularization/calibration rather than visual-detail recovery. Native threshold zero remains the deployment reference. A global threshold control alone is weaker than this learned local mask-only control.

An optional direct-replacement output using the same small head and inputs would chiefly test the residual optimization bias, not new information. It is not required to make this a capacity baseline, and should not become an architecture sweep selected on DEV. Full SAM-REF/SAM2Refiner have richer inputs/modules and are not fairly reproduced by assigning their names to this head.

## Closest primary overlap

- **SAM-REF, CVPR 2025**, [author v4 §§3.2–3.3 and §3.6](https://arxiv.org/html/2408.11535v4): a decoupled learned refiner combines image, prompts, coarse mask, and encoder features; an error head controls replacement. This directly precedes supervised output refinement conditioned on a coarse target mask and SAM features. Our restricted cached-feature head omits raw RGB, explicit interaction prompts, error-map gating, and the local stage. Those are budget restrictions, not a novelty argument.
- **SAM2Refiner, ICCV 2025**, [author v1 §§3.2.1–3.3](https://arxiv.org/html/2502.09660v1): freezes SAM2 while training added modules, combining object information and hierarchical image features for higher-resolution masks, with video integration. Its enlarged local image encodings and richer decoder differ materially in cost and inputs. This is strong precedent for the broad family; no-extra-encoder is a useful constraint to measure, not by itself an innovation.
- **SegFix, ECCV 2020**, [official paper pp. 5–8, §§3.1–3.3, Eq. 1](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf): learns image-derived boundary locations and directions toward reliable interior labels, then transfers coarse labels. It demonstrates supervised boundary correction without changing the underlying segmenter. The actual method has its own image backbone and structured offset supervision; a native-feature residual head is neither an exact SegFix reproduction nor a new model-agnostic offset method.

These known components do not invalidate an empirical baseline. They do rule out presenting “a tiny supervised head improves frozen SAM2 details” alone as a new contribution. A useful baseline could establish the practical headroom that a distinct later mechanism must explain and exceed.

## Cost and target-conditioning pitfalls

One float32 finest feature is 8 MiB; logits are 0.25 MiB. Streaming one frame/minibatch avoids retaining all feature graphs or loading SAM2. The existing checkpoint format contains additional native tensors, so actual disk-read/deserialization costs exceed those minimums. Head activations and parameter state are small relative to 8 GiB, but CPU forward/backward time, original-resolution loss, cache I/O, and complete-operation overhead remain **unmeasured**. No claim that a useful training run fits 1,800 seconds follows from MAC counts alone.

The target logit is the only object-conditioning signal in this minimal head. Image-only features cannot specify which of several similar objects is intended. If native logits attach to the wrong object, this local head may merely sharpen the wrong mask; if native presence is false, the preserved gate forces a no-op. It should not be judged as an identity-recovery method. Avoid sequence IDs, per-video learned embeddings, future statistics, GT-defined crop/boundary selection, and per-DEV thresholds. Feature normalization, if added, must use TRAIN-only constants or a declared per-frame operation consistently at inference.

Keep all fixed TRAIN frames/events and their duplicate weighting explicit, and all DEV frames with unchanged official endpoint handling. Report J and F separately, every parent, native-parity checks, absent-frame behavior, training fit versus DEV performance, and added full cost. Two exposed DEV parents do not establish driving relevance, broad generalization, or the final two-source goal.

## Read boundary

Local source inspection only: `campaign/cache_baselines.py` SHA256 `5d16b4658cef5d451e88f33b3b3f99562a9c782abf199b1e22c1c9c5b0b8f0da`; pinned `sam2_base.py` `69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41`; `mask_decoder.py` `ca3523c58365574faddf1bfb54f374e4b4beba05127185ea2b17d22b916e099a`; predictor storage/interpolation source. Existing local prior notes were reused. Web work was one two-query discovery batch and scoped reads/lookups of the three primary works above. Web text bytes are unavailable; no separate source, dataset, model, or package payload was acquired. No scientific array or label was opened.
