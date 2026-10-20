# Manufactured gradient probe: source-only review

2026-10-20. Reviewed final `probe.py` SHA-256 `5d8414398a715c3d8e5bd81f4c5fb1daa6490142a5b378f1a0c271daf0bed6dd`, its README/spec template, and the relevant pinned native decoder, tracking, memory-carrier and tensor-hash implementations. No probe/model import, execution, scientific data or GT access occurred in this review.

**No concrete source/API blocker found for the bounded manufactured check.**

- The decoder hook observes **`mask_decoder.forward`**, after its native multimask exclusion/dynamic-stability selection. The surrogate takes predicted-IoU argmax within that returned bank, before the later object-presence gate. It does not incorrectly take argmax over all four `predict_masks` candidates. Raw surrogate and gated native output remain distinct.
- History and image features are cloned into ordinary tensors outside `inference_mode`. The replaced frame6 memory remains in the declared seven-spatial-memory/seven-pointer frame7 context, with BF16 storage retained. Pointer/position carriers are checked against the original native formula.
- Frozen model parameters remain in eval behavior with gradients disabled. The memory leaf and separate zero-initialized head receive gradients; the requirement is finite nonzero **overall** head gradient, not nonzero gradient in every layer.
- Surrogate losses are checked finite before entering the JSON record. Zero/no-grad replay requires exact parity; the two zero-update grad replays use the prospective stated tolerance. The fourth, fixed post-step replay records changes without incorrectly demanding baseline equality or loss improvement.
- Eight image-encoder calls and four future replays agree with the fixture and counters. Head coefficient change, BF16 storage change, raw-logit change and native-mask change are reported separately. No post-update accuracy claim follows from any one of them.
- The existing supervisor template supplies four threads, 8 GiB and a complete 1,800-second limit, fresh private outputs and retained failures. It contains the final script hash; concrete paths/commit still must be filled before launch. The inner source/runtime/post-state checks complement, but do not replace, a clean outer receipt.

This review does not prove that native backward succeeds, that the step survives quantization, or that the operation fits its resource limits. Those are precisely the manufactured probe's questions. It qualifies neither the proposed 128-coefficient family, longer-horizon BPTT, a useful training schedule, nor segmentation efficacy. No new audit framework or extra prerequisite experiment is requested.
