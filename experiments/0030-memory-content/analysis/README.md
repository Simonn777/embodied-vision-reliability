# Fixed-final-checkpoint target-fit analysis — prepared, not executed

Run `qualified-python -B -W error fixed_checkpoint_error.py SPEC_JSON` only after active model operations finish, under the existing supervisor with four threads and a fresh output. Normal site initialization is required for installed Torch; no SAM2 module, weight, future mask, media, or DEV label is loaded.

The JSON spec has:

- `output`: fresh absolute directory.
- `family_root`: absolute residual-family directory.
- `source_sha256`: exact absolute paths and hashes of this script, `family.py`, and `head.py`.
- `targets`, `teacher_receipt`: descriptors `{path, sha256, bytes}`.
- `checkpoints`: exactly three ordered entries, `multistep`, `one_step`, `gt_memory`, each with `arm`, `checkpoint` descriptor and completed regression `receipt` descriptor. The checkpoint must be the final recorded 288-update state.

For all 48 saved TRAIN targets and each final head, the analysis computes `E=(qhat-q*)ᵀG(qhat-q*)/B²`, zero-predictor `E0=q*ᵀGq*/B²`, predicted normalized energy, and Gram cosine. Zero-radius errors are zero; zero-target error ratios and zero-vector directions remain undefined rather than being discarded. Parent error means and aggregate E/E0 keep all 48 examples. There are exactly 12 per parent, so the aggregate equals the equal-parent mean.

This is a final fixed-checkpoint check, unlike the saved changing-head online training losses. It adds no constant fit, new training arm, checkpoint selection, or method change. Good TRAIN fit does not establish generalization; poor fit does not prove inherent unpredictability.

The receipt distinguishes 144 tiny-head forward calls from zero SAM2 calls and zero fitting updates. `model_calls: 0` means no frozen segmentation model was invoked; `head_forward_calls` explicitly counts the learned MLP evaluations. Full wall time and any failure are retained. No run or import has occurred during preparation.
