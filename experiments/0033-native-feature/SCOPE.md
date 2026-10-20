# 0033 native-feature supervised baseline — fixed fit scope

Source preparation only. No scientific execution follows from this file. This is an ordinary directly supervised output head and a logit-only information ablation, not a new algorithm, a crop retry, or a memory-learning claim. Root owns admission, specifications, supervision and execution.

## Inputs and exact split

The six existing metadata bindings are `cache_receipt`, `cache_outer`, `cache_operation`, `cache_worker_spec`, `mapping`, and `data_index`, each an absolute path/bytes/SHA256 descriptor. The native cache must have its original completed clean legacy outer, 386 frames, six recordings and unchanged model hash. The mapping and data index identify original labels and actual target IDs. Only metadata describing all six recordings is inspected.

Fitting opens **228 noninitial TRAIN frames**: dog-agility 1–24, planes-water 1–37, dogs-scale 1–82, dog-gooses 1–85. There is no initial-frame annotation or feature read, no DEV payload read and no protected-data access. Target IDs are inherited from the admitted native cache's smallest-positive-initial-ID convention; they are not guessed to be 1. Label 255 is background for this target-object path.

Every selected tensor file, native prediction NPZ and encoded annotation is hash/size checked before the first tensor or label decode. All 228 reconstructed original-resolution native masks must exactly equal saved native NPZ masks **before any TRAIN label decode or optimization**. `torch.load` uses CPU and `weights_only=True`. The program keeps only copied finest features, native low logits and boolean targets; other deserialized pyramid, memory and pointer tensors are released immediately.

## Observations, head and normalization

Source-bound `head.py` supplies `NativeFeatureHead(mean32,std32,zero_features=False)` and `original_logits(low,size_hw)`. Features are projected image-only native `vision[0]`, reshaped from `[65536,1,32]` to float32 `[1,32,256,256]`; logits are native post-presence `[1,1,256,256]`. Both inputs detach in the head.

The 1,377 parameters are Conv1×1(33→32), GELU and Conv3×3(32→1,padding1). Inputs concatenate normalized features and sigmoid(native logits); output is `native + 4*tanh(residual)`. The final convolution starts at exact zero. No new encoder/decoder call or SAM model import occurs. Native history, pointers, memory and future native observations are unchanged.

All 228 feature maps, including no-op frames, contribute equally per pixel to TRAIN population channel statistics. Per-frame float64 variance/mean are combined using the centered-moment merge formula; population variance divides by `228*65536`. The mean and standard deviation are converted to float32, with standard deviation floored at 1e−6. `normalization.json` records raw float64 standard deviation and actual buffer values. No DEV normalization is permitted.

The `visual` and `logit_only` arms have identical initial state, statistics, architecture and ordered frame attempts. The latter zeros all 32 channels **after normalization**, leaving the sigmoid-logit channel unchanged. It is an information ablation, not an equal-information competing method.

## Fixed optimization and no-op accounting

Both arms train sequentially in **one complete operation**. CPU float32, seed0, 12 full epochs, one frame per batch, Adam(lr1e−3, weight_decay1e−4, betas=(.9,.999), eps1e−8, foreach=False, fused=False). A single `random.Random(0)` generates twelve fresh shuffles of indices 0–227; both arms replay these exact orders. The second arm receives an exact copy of the first arm's initial state.

For original-resolution logit `z` and binary target `y`, loss is:

`mean(BCEWithLogits(z,y)) + 1 − (2 sum(sigmoid(z)y)+1)/(sum(sigmoid(z))+sum(y)+1)`.

Interpolation is bilinear, `align_corners=False`, directly from 256² to original H,W. There is no GT downsampling, GT crop selection, boundary subsampling, class reweighting, clipping, learning-rate rescue, early stopping, checkpoint selection or auxiliary loss. The training objective weights frames equally, so parent weights are 24/228, 37/228, 82/228 and 85/228. Reported epoch losses are **online pre-update losses**, not terminal-checkpoint errors.

Native presence ≤0 **or** an empty original native mask forces a no-op. Such frames remain in each epoch and every later evaluation. There is no head call, backward or optimizer step on these attempts; their fixed native BCE+Dice is included in the 228-frame epoch mean. Active-only loss is reported separately. Finite gradients, parameters and optimizer state are required, but neither nonzero gradient in every layer nor loss improvement is required. With a zero final layer, early-layer first-step gradients can legitimately be zero.

Only `visual-terminal.pt` and `logit_only-terminal.pt` are saved for deployment after their respective twelfth complete epochs. If the operation fails, any already-written first-arm checkpoint and partial logs remain, but the **whole fit is failed**, and neither is admitted as a successful subset.

## Record and source contract

CLI: qualified CPython3.12.13 `python -B -S -W error train.py SPEC.json`. Spec keys used: `output`, `qualified_python`, `campaign_root`, `evaluation_root`, `guard_source`, `seed` (exactly 0), `source_sha256`, and the exact six `bindings`. Scientific hyperparameters are constants in the source; no specification overrides exist. No executable spec is created by this preparation.

The exact ten source bindings are this `train.py`, `head.py`, `SCOPE.md`, `ADMISSION.md`; campaign `common.py` and `cache_baselines.py`; evaluator `evaluate_jf.py`, `evaluate_baseline.py`, `evaluate_rollout.py`; and the existing offline guard `qualify.py`. `common.py` is imported only after its bytes are checked, for serialization/start helpers; `common.load_runtime` is never called. The guard installs before `site.main()` and science-library imports. Torch must be 2.8.0; environment and Torch intra-op threads are 4, inter-op 1, OpenCV serial.

Receipt schema `evr0033.native-feature-fit.v1` includes `config`, `parents`, 228 `train_frames` rows, shared `initial_state_sha256`, `normalization`, `orders`, `training_log`, `outputs`, `arms`, runtime/warnings/counts/costs, and input/source fingerprints. Each completed arm contains epochs12, optimizer-step and attempted-frame counts, online epoch summaries, a terminal checkpoint descriptor, final logical state hash and wall time. Both initial logical state hashes must match.

Each checkpoint is `evr0033.native-feature-checkpoint.v1` with `arm`, complete `state_dict`, `normalization:{mean,std}` CPU float32 vectors of length32, shared `config`, epochs12, optimizer-step count and `initial_state_sha256`. The normalizer is also in state buffers; downstream strict loading should compare both representations to `normalization.json`. Each terminal descriptor appears exactly in the receipt's flat output inventory.

All 5,472 frame attempts append to `training-log.jsonl`; receipt updates every24 attempts/epoch, not a full log rewrite per attempt. Loss components, no-op reason, gradient norm/nonzero count, identity/order, step count and step wall time are retained. Partial logs and failures remain. Final checks rehash all used source, specification, metadata, logs inherited from admission, and selected input payloads. Successful status requires both complete arms and no offline attempts.

The whole operation, including imports, loading, parity, normalization, labels, both fits, output serialization and final hashes, is limited to 1,800 seconds, four CPU threads and 8GiB group RSS under the parent supervisor. No performance estimate is an actual qualification. The later complete DEV output-only pass and parent-equal J/F evaluation are separate admitted operations; they must preserve native initial/absent/empty outputs and never use DEV labels for head selection.
