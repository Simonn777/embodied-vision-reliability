# 0030 manufactured memory-gradient probe — prepared, not executed

This is a single-context API/resource qualification. It does not read DAVIS, masks from any recording, scientific predictions, or evaluation GT. No training campaign, teacher target bank, method efficacy, or final residual architecture is admitted by a pass.

## Fixed check

`probe.py` reuses the qualified 0029 installation/source admission, exact `native_fixture.CONFIG`/`OVERRIDES`, `strict_state` and local `weights_only=True` SAM2 checkpoint loading. It uses the original `q.fixture(np)` texture and original nine-frame JPEG construction: 384×512, right shift 12 pixels/frame, zero padding, quality95/subsampling0, initial rectangle y118:182/x207:271. The target mask is that rectangle translated by 84 pixels at manufactured frame7. No SuperPoint/LightGlue model is instantiated.

Native baseline emits frames0–7 with the video length still9. Frame7 is the sole future frame being probed relative to the modified frame6 write. All seven spatial-memory slots are present: conditioning0 and nonconditioning1–6. Object pointers are `[0,6,5,4,3,2,1]`, split into28 tokens; the unchanged fixture carrier hook checks their actual order, features and positions. Image features/history are cloned outside inference mode into ordinary tensors, and every replay uses the same cached frame7 image features. Image-encoder calls must remain8, with four future replays and two backward passes.

1. Frozen/eval/CPU-float32 model; parameter `requires_grad=False`. Reuse four CPU threads/inter-op1/OpenCV serial behavior. Zero-residual float32→nativeBF16 write must reproduce both raw decoder and selected native output exactly under `no_grad`.
2. A full spatial-memory float32 leaf passes through the BF16 write/read convention and native future attention/decoder. Grad-enabled raw masks/quality/presence and native selected logits must agree with baseline at **atol1e-4, rtol1e-5**. Maximum absolute/relative differences and selected-mask binary differences are recorded. The leaf gradient must be finite and nonzero overall.
3. A zero-initialized 64→64 single1×1 convolution (4,160 parameters), residual `0.01*tanh(head(memory))`, performs one SGD step at lr0.001 on the same fixed future context. Require finite, nonzero overall head gradient, changed finite head weights, and no model-parameter gradients. Record how many memory elements actually change after BF16 rounding; zero changed elements is descriptive, not evidence of useful deployment. This head/bound/step is only a manufactured API test.
4. Hash model state_dict, original native state, ordinary history/cache/pointers before/after. Recheck exact admitted installation inventory, native provenance files, frozen source files and the private script bytes. No network/process/input attempt is permitted after the existing offline guard is installed.

The loss is **always explicitly a surrogate**: BCE on the raw pre-presence-gate mask selected by the native predicted-IoU argmax, plus BCE of the native object-presence logit against manufactured presence1. The native presence decision is recorded; it is never relabeled as an equivalent derivative of the hard-gated segmentation output. BF16 backward is PyTorch's cast surrogate, not the mathematical derivative of rounding. A fourth, fixed no_grad replay reports post-step native-logit/binary-mask and pre-presence-logit changes separately from coefficient change and BF16 storage change. No post-update accuracy/loss improvement is required or claimed. The model remains in eval mode throughout; only the separate head is trainable.

## Interface and launch preparation

CLI:

```
QUALIFIED_PYTHON -S -B -W error probe.py \
  --repo-root REPOSITORY --expected-commit REVIEWED_REPO_COMMIT \
  --native-root PRIVATE/work/0029-runtime-02 \
  --native-commit 37b54982bcc3b22b56a5817b6881f22489eba8d5 \
  --source-sha256 REVIEWED_PROBE_SHA256 --output-dir FRESH_PRIVATE_OUTPUT
```

Use the existing `experiments/0029-memory-information/supervise_exit_checked.py` with a concrete copy of `spec-template.json`; fill the repository/parent paths and reviewed repository commit. The template is intentionally not runnable. Its command must use `installation/venv/bin/python`, `-S`, and the exact reviewed script hash. When executing from inside the repository, the script also binds its own bytes with h.frozen at that commit. Do not change the old runtime, source or qualification receipts. The outer operation has a complete1,800-second limit, summed8GiB RSS cap, four-thread environment, retained stdout/stderr and no retry. Complete inner receipt alone is not success if the outer operation fails.

After parent source review and explicit execution authorization, the launcher is the existing supervisor's `python SUPERVISOR CONCRETE_SPEC.json`. Do not launch `probe.py` unsupervised or bypass a failed admission. Before real execution, preserve reviewed script/template/instructions with the concrete spec in a fresh operation folder. A failed first run retains all files/logs; any repair is separately recorded.

Output: `receipt.json` (`evr0030.gradient-probe.v1`), nine manufactured JPEGs, and one manufactured-mask NPZ. Receipt includes stage wall times (not summed into another misleading total), native loading/provenance, all call/carrier traces, warning/exception retention, raw-surrogate loss components, parity, gradient norms/counts, optimizer-step statistics and exact before/after hashes. Complete wall/RSS/cleanup come from the existing outer receipt. Failure remains `eligible=false`; no success-only subset result exists.

Only AST parsing and in-memory Python compilation are permitted in current preparation. `static-check.json` records that check; it is not a native or autograd test. No full new supervision or auditing framework is introduced.
