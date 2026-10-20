# 0029: can sparse identity evidence identify a useful memory intervention?

October 20, 2026. **Hypothesis and prospective success standard, not an executable protocol, new method claim or measured improvement.** No video/mask payload, checkpoint or prediction has been inspected. This question does not reopen the rejected historical-replay proposal.

## Problem and closest precedents

A memory can strongly change a video mask while helping recognition; a stable prediction can still follow a distractor. The candidate question is whether initial-mask labels, transferred by a separate image matcher, can tell these cases apart when comparing otherwise identical memory interventions.

[Exploiting Recollection Effects, IVC 2023](https://doi.org/10.1016/j.imavis.2023.104866) already uses predictions with and without memory features to manage storage. Its publisher abstract/introduction were checked; the complete formula and code were unavailable. Thus generic leave-memory-out influence is occupied, and exact non-equivalence of the proposed criterion remains unresolved. [Counterfactual Memory Auditing, September 2026 preprint, §§III-A–III-D](https://arxiv.org/html/2609.27247v1) also distinguishes a memory's influence from its benefit in robot policies. That conceptual distinction is not our invention.

[DAM4SAM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.pdf), with [extended method §§3.1–3.2.2](https://arxiv.org/html/2509.13864v1) reviewed, already selects distractor-resolving anchors using mask alternatives and quality/area stability. [SENTRY §3.2](https://arxiv.org/html/2606.24449v2) reasons over backward target and neighbor tracks. These are strong comparison requirements. Combining matching and memory intervention is useful only if a distinct, effective decision survives them and simpler same-evidence controls.

## Observable and falsifiable claim

At time t, use only causal RGB, an immutable initial binary target mask, a fixed SAM2 state, and fixed SuperPoint/LightGlue initial-to-current correspondences. Let P and N be transferred positive and negative points, selected without consulting any current predicted mask. For current probabilities p define

`L(p) = −0.5 mean(P, log p) −0.5 mean(N, log(1−p))`.

For a noninitial memory entry m, compare the unchanged prediction with a prediction excluding m. The signed removal score is `L(unchanged) − L(without m)`. A positive score predicts useful removal. Numerical clipping, point filtering, evidence minimum and deterministic ties must be fixed in the execution protocol.

The intervention must remove the declared spatial-memory/pointer bundle, preserve other selected entries and temporal positions, and prevent automatic backfill. It measures the entry's current marginal effect. It does not identify sole historical responsibility: other memories can retain its errors, and attention effects interact.

**Claim to test:** this signed score selects interventions with better full-mask accuracy than unsigned output change on the same candidate predictions. It is not enough to lower sparse point loss.

An initial complete binary mask supplies both positive and negative labels. Earlier deliberation unnecessarily restricted negatives to other annotated objects. Under exact correspondence to the same currently visible material point and invariant target membership, an initial background label transfers as well. The revised proposal may use eroded foreground and the exterior of a dilated initial mask; single-object videos are not automatically excluded. These assumptions are not guaranteed by LightGlue. Occlusion, false matches and changing material membership can break them.

Even correct sparse labels do not guarantee dense improvement: removal may fix every sampled point while adding a large false-positive region elsewhere. Maximizing a score over several removals can further exaggerate noisy proxy gains. Missing either witness class must produce no-op and remain in the denominator.

## Smallest useful diagnostic

Use one unchanged causal baseline stream, a fixed frame schedule and two predetermined noninitial memory entries. Fork only the current prediction, then discard each probe; do not mutate later baseline states. Freeze the cohort and complete implementation before outcomes. Evaluate all scheduled cases, including missing matches and failures.

Compare signed selection, unsigned change, predicted mask quality, fixed/random removal and no-op on identical probe outputs. Also compare the same witness score selecting existing decoder alternatives, the same points supplied directly as positive/negative prompts, and current correction after recent-memory reset. Record full-mask accuracy, oracle candidate headroom, selector regret, coverage and total computation separately. Oracle selection is diagnostic only. If direct prompting or reset explains the benefit, attribution has not earned a role. Fixed-state probe gains do not establish a streaming tracking gain.

## Prospective topic-level success gate

Before a manuscript, a frozen streaming method must improve mean **J&F by at least 3 absolute percentage points on each of two independent held-out video sources**, with original recordings disjoint from development and positive paired recording-level 95% confidence intervals. Neither J nor F may decline by more than 0.5 point. Compare the strongest prespecified eligible controls, including the same-evidence prompt/reset controls, at a common complete-pipeline time budget fixed before results. Count matching, encoding, intervention, failed attempts and storage; report median and tail latency. A faster weak baseline alone cannot satisfy this comparison.

Closest-prior differentiation, mechanism ablations and independent reproduction remain necessary. A small diagnostic, a known-operation gain or a software check does not meet this gate. Exact assets/runtime, data grouping, numeric diagnostic gates and complete protocol/source freeze remain unresolved prerequisites; this document authorizes no scientific execution.
