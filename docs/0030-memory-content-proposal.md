# 0030 prospective problem card: amortized correction of spatial memory content

2026-10-20. **Development proposal, not a new-method claim, efficacy result, or manuscript.** Existing 0029 results and closure stay unchanged. This card authorizes no final-test access. Ordinary development is already within the user's research request; remaining requirements are concrete scientific and resource definitions, not a new permission request.

## Question and changed operator

In initial-mask-conditioned causal VOS, can a small residual network learn a useful memory-write representation from a multistep offline optimizer of future segmentation loss, beyond the **same network trained directly on that loss** and a current-GT-mask memory-denoising target?

The deployed model receives only the initial target annotation and RGB through the current time. Freeze the qualified SAM2.1-tiny encoder, memory attention, mask decoder and memory encoder. After the current native mask has been emitted and its spatial memory `m_t` encoded, replace only `maskmem_features` by `m_t + R_theta(current causal inputs)`. Keep the native object pointer, positional encoding, memory ages, mask selection and capacity unchanged. The correction affects subsequent frames. This changes the stored representation rather than selecting from the old B/D1/D3 bank or choosing an entry to evict.

Candidate causal inputs: native spatial memory, resized current mask probability, current object score and predicted IoU. Use a small spatial residual head (1x1 bottleneck, depthwise 3x3, 1x1 output), with a zero-initialized output layer and a norm-bounded residual. Exact shapes, numerical conventions, bound and training schedule must be fixed after the narrow artificial gradient check and before real target construction.

Initial training scope, if qualified: fixed-context differentiation. Hold other detached baseline memory/pointer tensors constant, replace only the selected memory, and recompute each later attention/decoder output. Teacher and direct control share those contexts and the exact gradient surrogate. This is not recurrent BPTT; its effect must later survive the student's own full causal rollout. Native BF16 casts use PyTorch surrogate derivatives, not a mathematical derivative of a quantizer.

Training teacher: optimize a bounded additive correction to a TRAIN frame's memory tensor for several steps, using later TRAIN masks and a frozen downstream model. The candidate is ordinary regression of this optimized correction into the causal residual head. It is **not** a claim that distillation itself, privileged training, memory refinement, or multistep optimization is new. The potential empirical insight is whether task-optimized latent writes transfer better than correcting the current mask representation or directly optimizing the same causal head.

## Necessary distinctions and falsifiers

1. An unprojected one-step target `delta* = -eta grad_delta L(0)` with squared-target regression produces, at a zero-output head and with identical contexts/loss/derivative conventions, a parameter gradient proportional to direct future-loss training. This local result does not imply identical later optimization or equivalence after active projection. That special case is a control, not a distinct contribution.
2. A multistep teacher may find corrections unavailable to a causal student, exploit future labels, or overfit hidden-state directions. Teacher loss reduction is therefore insufficient. Selection must use the student's own complete causal DEV rollout, regenerating every changed subsequent memory.
3. The correction can preserve or harm identity. Current-mask confidence, teacher-target error and frozen-context future loss are diagnostic quantities, not substitutes for sequence J&F. No claim that future information is observable at deployment.
4. If gains vanish against direct same-head future-loss training or GT-memory denoising, this proposed use of a teacher has not established an advantage. The broader user goal remains unmet even if an artificial gradient check passes.

## Controls and equal information

All learned arms use identical TRAIN/DEV groups, available annotations, causal inputs, architecture, residual bound and selection opportunity:

- Native causal SAM2.1-tiny with no residual.
- Same head trained directly on the downstream annotated-frame segmentation objective; report both update-matched and complete-training-cost-matched comparisons. Do not call frozen-context differentiation full recurrent BPTT.
- Same head regressing the difference between native predicted-mask memory and current-GT-mask memory, using the same allowed training masks.
- Same head regressing a one-step future-loss correction (equivalence control).
- Candidate same head regressing a multistep bounded future-loss correction.

The deployment inference budget is identical among learned arms. Teacher construction, feature extraction, fitting, validation and failures are charged separately and together. The direct baseline must not be starved of updates merely because the candidate spends more on teacher construction. Any comparison to published OOKD, EdgeTAM, Cutie or SAM2RL must distinguish an adapted objective/control from an actual reproduction.

## Development, final evaluation and resources

Use only the six permanent DAVIS development recordings for an initial parent-disjoint 4/2 TRAIN/DEV pilot, assigned by a recorded hash rule before teacher construction. Both portions remain development because all six have already informed this project. Derived clips, objects, features and teacher targets inherit the parent assignment. Existing 48 query masks do not provide complete-sequence J&F. A separate small acquisition record must bind the missing full development masks; reuse existing bytes without redownload. No new or protected recording is selected because of the 0029 errors.

This is an iterative development campaign: at most three documented candidate configurations, equal selection opportunity for strong controls, and a second fixed-seed comparison for the selected configuration. Bugs and training iteration are logged; failed runs remain. Do not create a new publication/auditor pipeline for every update. Before real fitting, record exact sampling, losses, optimizer, run/total budget, failure handling and selection criteria in one campaign file. Cache only frozen image computations; memory-conditioned state must be regenerated when writes change.

The final evidence gate remains at least +3 absolute official J&F points on **each of two independent held-out sources**, positive paired recording-level 95% intervals, neither J nor F reduced more than 0.5 point, prespecified strong same-information controls at a common complete-pipeline time budget, ablations and independent replication. VOST original-recording grouping remains unresolved. No final payload or annotation access is allocated here.

Keep four CPU threads, summed RSS <=8 GiB, <=1800 seconds per complete operation, no fees, and the conservative acquisition balance 1,289,618,314 bytes until a new explicit debit. MPS training is unqualified. New training feasibility is not implied by prior inference timings.

## Immediate engineering check, before any real training

Reuse the qualified runtime and supervisor. Use only manufactured RGB/masks and the already cached checkpoint. Verify that a leaf spatial-memory tensor receives finite, nonzero gradients through a future attention/decoder call outside `inference_mode`, frozen model parameters remain unchanged, zero residual preserves native output within the declared native numerical convention, and a tiny head can execute a complete optimizer step. Measure the complete operation and peak RSS. A failure should result in a narrow recorded repair or feasibility stop, not scientific scoring. No new runtime, benchmark data, full teacher generation or final-test access is part of this check.

The source-only methodological review permits this minimal probe and identifies the shared gradient context, explicit BF16 surrogate and same-head direct-control comparison as necessary definitions. Exact check implementation remains pending. In particular, the candidate is not admitted on an assumption of novelty or feasible backward computation.
