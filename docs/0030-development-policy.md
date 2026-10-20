# Post-0029: separate development from confirmatory testing

2026-10-20. Source/document review only: `post0029-research-selection.md`, the current 0029 closure/state/actions, and the existing topic-level gate. No new payload, labels, arrays, model, experiment, or repository edit. The feasibility of training on the installed runtime has not been measured.

**Recommendation:** retain the failed historical tests, but stop treating every future development run as a one-shot confirmatory study. A distinct learned method needs an explicitly iterative training/development phase. Freeze its final evaluation only after development. Provenance protects the interpretation; producing more provenance is not the scientific endpoint.

## What remains binding, and what should change

The B/D1/D3 instantaneous deletion variant remains closed. Its 0.0768-point admissible custom-J headroom and failed signed selector cannot be relabelled, tuned into a successful historical test, or used to justify held-out expansion. Its six recordings remain development permanently.

That bank's ceiling does **not** bound masks produced by a different trained causal model. Neither 0029's one-attempt rule nor its prohibition on rescue tuning is a universal ban on training, loss development, debugging, or repeated validation of a genuinely different method. Future-conditioned training supervision changes the learning contract, although it adds no future observation at deployment.

The selection memo correctly asks for a concrete bottleneck, strong controls, and resource feasibility. Two of its proposed prerequisites are too strong if generalized:

- A fixed cheap heuristic must not have to succeed before a learned extractor may be built. Extracting the signal can be the learning problem itself.
- A separate frozen opportunity measurement before every implementation can become an endless prerequisite. A tiny end-to-end trainable baseline can itself be the cheapest opportunity measurement. Its failure should guide a bounded development campaign, not automatically demand a new protocol/runtime.

For a newly scoped development project, ordinary authorized bug fixes and train/dev iterations need no additional permission ritual. Log changes and runs; do not rewrite failed attempts or use protected test outcomes to choose them. This does not authorize executing anything in the present source-only task.

## Minimum problem card: six concrete answers

One page is sufficient; unknowns become explicit engineering checks, not speculative certainty.

1. **Task and causal contract.** Semi-supervised VOS with the initial target mask and RGB through time \(t\); output the current mask without future frames or extra deployment annotation. The endpoint is full-sequence official J&F, not the old 48-query J.
2. **Failure and transferable information.** For example, target reappearance lets an offline teacher revise an earlier occlusion/distractor interpretation. State what cue remains available to the student before reappearance. If identical causal histories admit different identities, the student cannot recover the realized future; it can only learn a conditional predictor. Distillation is not a guarantee against that ambiguity.
3. **One trainable operator.** Specify exactly which small module changes, its causal inputs, parameters, target, loss, and how its output affects subsequent state. A residual memory-write adapter is one possible card, not an established new method. “Future teacher → causal student” alone describes ordinary privileged-information distillation, not the contribution.
4. **Equal-supervision controls.** Unmodified causal SAM2; the same trainable module with ordinary available-label supervision; causal-teacher distillation; and plain distillation from the **same offline teacher**. All learned controls receive the same permissible labels, teacher targets, training budget and selection opportunity. If a future endpoint mask comes from GT, disclose it as added training supervision and give it to the relevant controls. An offline teacher is a diagnostic upper-information comparator, never a causal deployment baseline.
5. **Bounded implementation and cost.** Name the cached invariant features, trainable boundary, data units and complete training-run definition. Use a small fixed campaign budget rather than creating successive one-shot assays. A reasonable initial budget is one implementation and at most three documented candidate configurations, with matched baseline tuning; repeat the selected comparison with another fixed seed before final testing. Close or re-scope explicitly when the budget is exhausted.
6. **Selection and falsifier.** Choose by causal development J&F, with every declared recording retained. Failure to beat plain same-teacher distillation, or improvement disappearing in the student's own rollout, refutes the proposed extra mechanism at this stage. Teacher agreement, training loss and isolated mask corrections are insufficient. No novelty proof is required before this prototype, and no novelty claim follows from it.

## Smallest defensible split plan

Use original-recording groups, including all derived clips, objects, augmentations and teacher targets.

- **Initial TRAIN/DEV pilot:** allocate the six already-exposed DAVIS parents by a fixed metadata-only rule to four training and two internal development parents before generating teacher outputs. This is development infrastructure testing, not fresh generalization evidence: all six have already informed research decisions. Do not move the two development recordings into fitting during model selection.
- **Labels:** the currently acquired initial masks and 48 query masks are not complete full-sequence J&F ground truth. They can support permitted training/sanity checks, but cannot silently become an official J&F validation set. Obtain a small, explicitly budgeted, fully labelled development block if needed; record its role before outcomes.
- **Teacher boundary:** future RGB and any allowed extra masks are used only to construct training targets on TRAIN. Feature normalization, teacher calibration and pseudo-label filtering are also fitted there. Student checkpoint/hyperparameter selection uses causal DEV rollouts. Any offline teacher assessment on DEV is separately labelled diagnostic, never a deployed student result or training target.
- **Protected final tests:** reserve two independent data sources with original recordings disjoint from TRAIN/DEV and known overlap with pretrained assets documented. Distinct mirrors or source-derived datasets do not count twice. DAVIS and VOST are only prospective names: VOST's original-recording grouping remains unresolved. Establish grouping and licensed complete metric coverage before spending on final payloads. Lock test membership and keep their labels/results out of all architecture, threshold and checkpoint decisions.

The final standard is unchanged: at least **+3 absolute official J&F points on each of two independent held-out sources**, positive paired recording-level 95% intervals, neither J nor F down more than 0.5 point, and the strongest prespecified eligible controls at a common complete-pipeline budget. Failed/technical cases and all costs remain visible. If final results inform a redesign, those tests become development for the redesign; they cannot remain its untouched evidence.

## Keep the engineering proportional

Reuse the qualified runtime, supervisors and metrics. A training campaign needs source/config/seed/split/checkpoint hashes, costs and a short append-only run table; it does not need a new independent auditor, publication package and prerequisite paper screen for each gradient run. Review changed interfaces and meaningful tests once; repeat full saved-output auditing at the final frozen comparison.

Cache only computations invariant to the trained parameters and history. In the pinned SAM2 source, projected decoder features depend on `conv_s0/conv_s1`, and memory-conditioned features depend on prior state. Freeze those projections when caching them; regenerate affected temporal state when the student changes memory. The native inference-mode video API is not automatically a differentiable training path. Resolve the narrow gradient boundary with a small training smoke check before scaling.

Retain the stated limits: four CPU threads, 8 GiB and 1,800 seconds per complete operation, and a conservative **1.289 GB** acquisition balance supplied for this review. This note allocates none of it. Measure a complete small training job, not just a cached forward; include teacher construction, feature extraction, training, selection, failures and final inference in separately identified costs. Do not divide an over-budget single operation merely to evade its limit. If even a useful small adapter and its strong controls do not fit, this is a feasibility stop—not a reason for further provenance-only stages.
