# Post-0029: cached-feature learning feasibility (source-only)

Prepared 2026-10-20. This is an interface/resource assessment, not implementation admission, a training result, or a novelty claim. I read only pinned source/configuration and existing helper source; no model imports, image/mask/NPZ reads, training, inference, new downloads, or timing tests. The newly proposed action is a **causal spatial memory-content residual**, not the closed D1/D3 deletion rule.

**Conclusion.** The installed SAM2.1 tiny exposes the required tensors and differentiable submodules. Cached-feature mask-decoder training and a small memory residual head are implementable without another model dependency. However, optimizing a memory target through future masks still requires backward through the four-layer memory attention and decoder. Neither its CPU backward memory nor its ability to complete a useful training schedule in 1,800 seconds is measured. A one-future-frame artificial gradient/resource qualification is the missing executable prerequisite; a fixed heuristic succeeding is not a logical prerequisite for learning.

## Pinned interfaces and cache boundaries

Source root is `work/0029-runtime-02/installation/sources/sam2`, official SAM2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`. The actual config is `sam2/configs/sam2.1/sam2.1_hiera_t.yaml`, not the cached DAM config. It sets image size 1024, feature dimension 256, seven spatial-memory slots, memory dimension 64, four memory-attention layers, high-resolution decoder features, object pointers, and uncompiled Hiera tiny. The existing qualified operation uses CPU float32, four threads, evaluation behavior, and BF16 stored memories.

| Tensor, one image/object | Shape | Float32 storage |
|---|---|---:|
| Frozen low-resolution image feature | `1×256×64×64` | 4 MiB |
| Projected high-resolution features | `1×32×256×256`, `1×64×128×128` | 8 + 4 MiB |
| Memory-conditioned current feature `pix_feat` | `1×256×64×64` | 4 MiB |
| Spatial memory content | `1×64×64×64` | 1 MiB (0.5 MiB when stored BF16) |
| Object pointer | `1×256` | 1 KiB |
| All four raw decoder masks | `1×4×256×256` | 1 MiB |

`SAM2Base.forward_image` (lines 467–479) runs the image encoder **and** decoder-owned `conv_s0/conv_s1` projections. `_prepare_backbone_features` flattens these into sequence tensors. Caching the projected 16 MiB/image bundle requires freezing those two projections; updating them would require caching the unprojected features instead (84 MiB/image). For 386 frames, these are approximately 6.03 GiB versus 31.66 GiB of float32 cache, excluding position embeddings and metadata. They must be streamed from disk, not all resident under an 8 GiB process cap. Common positional encodings can be stored once. There is no existing complete image-feature cache to assume: predictor `_get_image_feature` (704–735) retains only the most recent frame.

`_track_step` returns `(current_out, sam_outputs, high_res_features, pix_feat)`. The last tensor is already object/history-conditioned. It is useful for a fixed-baseline-history decoder control, but is **not** a reusable image-only embedding when the learned student changes its own memory. For the content-corrector study, cache frozen image features and detached baseline histories for training contexts; recompute memory-conditioned features after each proposed content change. Evaluate the learned student on its own causal rollout, not a future teacher's or baseline's cached conditioned features.

The direct differentiable decoder API is `sam_mask_decoder.predict_masks(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, repeat_image=False, high_res_features=...)`. It yields four raw masks, four IoU predictions, four 256-dimensional mask tokens, and an object-score logit. The public `forward` selects one or three masks. `_forward_sam_heads` then applies a hard object-present gate and predicted-IoU argmax. A training loss must declare which raw logits/visibility loss it uses and preserve the intended inference policy; a hard absent-object branch can otherwise eliminate mask gradients. Feeding a current target GT mask through native `track_step` is invalid supervision: `use_mask_input_as_output_without_sam=true` bypasses the decoder for mask prompts. GT belongs in the loss, except for the admitted initial object mask.

## Exact spatial-memory intervention

`_encode_new_memory` (678–727) converts the low-resolution image feature plus predicted mask into `maskmem_features`, adds the native no-object spatial embedding, and returns positional encoding separately. `_encode_memory_in_output` (789–819) stores it in `current_out`. The narrow insertion is:

`M_t_corrected = M_t_native + delta_theta(causal_context_t)`

replacing only `current_out['maskmem_features']` **after** that native encoder/no-object term and **before** predictor `_run_single_frame_inference` converts it to BF16 for storage. Keep the slot/frame index, positional encoding, object pointer, predicted mask, native quality, and selection logic unchanged. A 64→64 single convolution has 4,160 parameters for 1×1 or 36,928 for 3×3 including biases. Those counts concern the head, not its expensive downstream training graph. Extra context channels require an explicitly larger head and identical controls.

Future `_prepare_memory_conditioned_features` flattens selected memories to `4096×B×64`, concatenates them with pointer tokens, and applies memory attention. There is no explicit detach on that path. Freezing model parameters permits gradients to a memory-input leaf; wrapping the call in `no_grad`/`inference_mode` does not. Existing native operations and public predictor entry points run in inference mode, so training needs ordinary tensors and direct grad-enabled submodule calls. Cache reload or cloning outside inference mode is needed; merely detaching an inference tensor is not sufficient assurance. No weights/source on disk need modification.

Required narrow guards before scientific training: zero residual reproduces the unchanged rollout; native BF16 storage/readback remains in the forward path (or any different precision is declared and equally controlled); finite gradients reach only the residual/head; parameter, pointer, positional-encoding, sibling-state and original-cache hashes remain unchanged. Keep the frozen backbone/attention/decoder in evaluation behavior while enabling gradients: whole-model `.train()` enables dropout and changes temporal/pointer filtering. Teacher and student must use the same declared hard-mask/quality policy or the same explicitly qualified differentiable surrogate. RoPE's existing slice assignment and native casts need actual backward qualification, not a source-only promise.

## Teacher, controls, and the one-step equivalence

A bounded offline teacher could optimize a current memory residual using one or a fixed small number of later **training** mask losses, with a declared trust region/regularizer. The causal head predicts that residual from information available at time t. Future labels/context must never enter the deployed head inputs. A minimum-norm or spatially constrained target matters because a 262,144-dimensional memory has many directions invisible to a few mask losses; arbitrary optimized feature deltas need not be learnable or transferable.

If one teacher step is `delta* = -eta grad_delta L(0)` and the student fits squared delta error, then at a zero-output student initialization the parameter gradient is `2 eta J_theta^T grad_delta L(0)`: it is proportional to the gradient of **same-head direct future-loss training**. A one-step gradient target alone is not a distinct mechanism. Multiple teacher steps within a fixed trust region can differ from direct parameter optimization, and this equivalence does **not** veto that prototype. Generic feature L2 distillation alone does not establish its novelty or benefit.

The smallest useful controls are:

1. Unchanged causal SAM2; retain its native initial-mask supervision.
2. Same spatial head trained directly through the same future mask loss, with identical labeled frames, context/horizon, regularizer and inference behavior. This is the essential control for teacher optimization/amortization.
3. Same head with ordinary delta/feature-target distillation from the same multi-step teacher; ordinary logit KD from the same teacher is a separate cheap output-level control. A new representation objective cannot receive a better teacher or additional masks uniquely.
4. A dense-supervised decoder/head-only control using the same admitted human masks and causal conditioning, to identify gains due simply to supervision/capacity. Its fixed-history training scope must be stated.
5. Forward-only versus future-conditioned teacher, or forward/backward teacher, with equal annotation access. Reverse tracking needs a later conditioning mask: a GT endpoint is extra supervision; a forward-predicted endpoint is a different, noisier teacher. Do not select/fuse teachers with evaluation GT.

Teacher gain is useful information about the learning target, not a requirement that a hand-coded causal rule already achieves it. Conversely, noncausal teacher performance is not deployed student performance. A student can only learn the component of the future-dependent correction predictable from causal context. Six recordings remain six physical training units, regardless of the number of frames/crops. The currently acquired initial and query masks do not silently admit every training mask; dense supervision and any reuse of the exposed 48 development query labels need an explicit new training policy, with untouched validation parents kept separate.

## Resource verdict and cheapest measurement still needed

For cached-context decoder training, minibatch one avoids image-encoder backward and is plausibly below 8 GiB, but has not been measured. For memory-target optimization, **small head size does not imply cheap training**. At a full seven-memory context, each of four cross-attention layers has 4,096 queries and about 28,672 spatial keys. The two attention matrix products alone cost roughly `2×4096×28672×256 ≈ 60 billion` multiply-accumulates per layer, before projections, image self-attention and MLPs. An explicitly materialized float32 score matrix is about 448 MiB/layer; actual SDPA allocation depends on the existing CPU backend. These are architectural estimates, not observed allocation/time. Frozen parameters reduce gradient/optimizer storage but do not remove derivatives through those operations.

One local teacher step needs one future-context forward plus backward to the selected memory; K teacher steps require approximately K such evaluations, with fresh graphs rather than retaining all steps. Plain target-distillation minibatches after teacher construction only train the small head. Direct-future-loss control pays the future forward/backward per update. Charge teacher construction, cache creation, training and own-rollout evaluation separately and completely; neither wall time nor the number of useful updates under 1,800 seconds can be inferred from prior inference qualification.

The next permissible feasibility measurement, if separately admitted, is one manufactured full-size context and one future frame: zero-residual parity, finite nonzero memory/head gradient, no frozen-weight changes, exact output policy, peak RSS and forward/backward time at four CPU threads. Only then choose a complete, prospectively bounded training operation. This is an API/resource check, not a dense guarantee, an accuracy experiment, or a requirement to rescue the closed deletion assay. No full-sequence backpropagation or encoder fine-tuning is justified by the present source evidence.

## Source record

All hashes below were computed from the installed text, without importing it. Official source links use the pinned commit, e.g. [SAM2Base](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py), [MaskDecoder](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/mask_decoder.py), [memory attention](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/memory_attention.py), and [tiny configuration](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_t.yaml). Links were not fetched in this review.

| Path relative to installed SAM2 root | SHA256 |
|---|---|
| `sam2/modeling/sam2_base.py` | `69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41` |
| `sam2/modeling/memory_encoder.py` | `73f7089ae5fdacdfcaaf3deca1ca6d9f84d89e7deb9cf97589622a9ef17ba42f` |
| `sam2/modeling/memory_attention.py` | `07358cb7f58ec3788e88ff4e6415f8a9d628a3963529d6e2c915494a5347c8e2` |
| `sam2/modeling/sam/mask_decoder.py` | `ca3523c58365574faddf1bfb54f374e4b4beba05127185ea2b17d22b916e099a` |
| `sam2/modeling/sam/transformer.py` | `cda19052331e775190ce8b1159efcb828a4632d931a6d8bd4d47109f121782f1` |
| `sam2/sam2_video_predictor.py` | `912555920a77f72ded07839efa33fcceecc79f3523abbec783f0be46ee2c55c2` |
| `sam2/configs/sam2.1/sam2.1_hiera_t.yaml` | `f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d` |

Existing `Prep/native_fixture.py` read for loader/runtime conventions: `4e20a61150407c575eedff41309281923f9a4b57fa3f34609153e645dde7ad5f`. The installed source tree does not contain the upstream training package; `SAM2Base.forward` explicitly raises and refers to `SAM2Train`. A small direct-submodule training loop is a new implementation to qualify, not an already-qualified native training API. No existing scientific file was modified.
