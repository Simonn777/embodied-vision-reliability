# Future credit for causal memory content: bounded prior and operator check

2026-10-20. Source-only note. No scientific arrays, labels, model, training, inference, data acquisition, or repository edit. The fixed 0029 deletion experiment remains closed. This note neither allocates a new experiment nor establishes a method contribution.

**Disposition:** learning memory decisions from later segmentation quality is already a direct published/preprint operation. Learning continuous memory content is also established. The narrower remaining question is whether **amortizing a multi-step, future-loss-optimized memory write** improves generalization or training efficiency over the **identical causal content head trained directly through the same future losses**. The bounded search did not find that exact combination in the inspected methods; this is not a novelty proof. It merits a precise feasibility check, not a large implementation or a claim that future credit is new.

Integration note: the [development policy](0030-development-policy.md) governs later prototypes. This source review's suggested sequential falsifier is not a universal ban on logged TRAIN/DEV iteration or a requirement for an additional one-shot study before every implementation. A frozen-context teacher/control may be tested if named accurately; it is not the full rollout objective written below. Both require the student's own causal rollout for development assessment.

## Exact closest coverage

**SAM3RL / SAM2RL.** [SAM3RL: Memory Control via Reinforcement Learning for Visual Object Tracking, arXiv:2610.19931v1, 14 October 2026](https://arxiv.org/html/2610.19931v1) is a preprint; §§3.1–3.2 and Appendices A, B, D, E were read. Appending the current memory creates eight entries; a seven-way action rejects that write or evicts one non-initial entry. Its return uses future IoU relative to the default policy; actual PPO training uses gamma=1 and GAE lambda=1. The deployed actor is causal. The paper also trains this controller on frozen SAM2.1-L, so this is not merely a SAM3 analogy. It does not optimize spatial memory values. Its stated training is 24 hours on four H200 GPUs with 256 GB host memory, despite a 3.23M-parameter actor. Appendix B promises code release; no official released implementation was verified in this bounded check. This is the direct future-utility baseline, not an admissible local training recipe.

The preliminary [SAM2RL: Towards Reinforcement Learning Memory Control in Segment Anything Model 2, arXiv:2507.08548v1, July 2025](https://arxiv.org/html/2507.08548v1), full short method/evaluation text read, already optimized the same kind of admission/eviction through IoU reward. It deliberately fitted separate agents to individual videos, providing an upper-bound study rather than a generalizing policy. Do not use that oracle-like experiment as deployable evidence.

**QDMN.** [Learning Quality-aware Dynamic Memory for Video Object Segmentation, ECCV 2022; arXiv:2207.07922v1](https://arxiv.org/html/2207.07922v1), §§3.2–3.3 and implementation section read. Its quality model learns squared error to the **current predicted mask's GT IoU**. Admission uses a quality threshold; eviction combines quality and recency. Sequence segmentation supervision is present, but the inspected quality target is not counterfactual downstream utility of the write. This is a strong current-quality proxy control, not an exact future-credit target.

**Continuous writes are not new.** [Learning Dynamic Memory Networks for Object Tracking, ECCV 2018; arXiv:1803.07268v1](https://arxiv.org/html/1803.07268v1), §§3.3–3.6 and training text read, learns differentiable read/write/erase operations, with an LSTM controller and gated retrieved-template residual. Training uses 16-frame clips. The inspected HTML does not give a sufficiently explicit loss formula to attribute a particular per-write future-gradient target to it.

**MemSAM is the important content-level comparison.** [MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation, CVPR 2024, official repository](https://github.com/dengxl0520/MemSAM) and official source were read; the final paper was available only as search-indexed method excerpts, not a downloaded/full-read PDF. In [`Mem.encode_value2`](https://raw.githubusercontent.com/dengxl0520/MemSAM/main/models/segment_anything_memsam/modeling/mem.py), the optional learned foreground-reinforcing module transforms the embedding using the predicted mask before value encoding. [`ForegroundReinforcingModule`](https://raw.githubusercontent.com/dengxl0520/MemSAM/main/models/segment_anything_memsam/modeling/mem_modules.py) concatenates mask/features, applies convolutions and channel softmax, then mask modulation. [`train_video.py`](https://raw.githubusercontent.com/dengxl0520/MemSAM/main/train_video.py) backpropagates sequence segmentation loss, using endpoint supervision in semi-supervised mode. Those inspected paths contain no per-sample optimized memory tensor teacher or target-distillation stage. They do establish learned content reinforcement; the repository's mutable `main` was inspected, not pinned or acquired as an executable artifact.

A related earlier operation is [Adaptive ROI Generation for Video Object Segmentation Using Reinforcement Learning, arXiv:1909.12482v1](https://arxiv.org/html/1909.12482v1), §§3.1–3.4 and training sections read. It learns positive/negative pseudo-label ROI thresholds for online adaptation, with IoU-derived reward and actor–critic future return. Thus “choose today's update for later segmentation” is occupied beyond memory-slot selection. It is not the proposed frozen-decoder spatial-write teacher.

The previous source note already verified OOKD, MobileVOS and EdgeTAM as identity/relational/feature-distillation controls. Their names do not replace a matched implementation: compare the particular targets and equal information, not incomparable published training budgets. A search-only HFVOS abstract also mentions history/future memory; its causal/training interface was not verified and is not used to assert exact overlap or clearance.

## One precise candidate and its mathematical limitation

Let \(B_t\) be the causally obtained bank and \(m_t\) its new spatial memory, before storage. Let \(x_t\) denote only available current/past features and confidences. Keep pointer production, insertion schedule, decoder, and the rest of the tracker fixed. A compact residual head writes

\[
  \widetilde m_t=m_t+D\!\left(h_\theta(x_t,B_t,m_t)\right),
\]

where \(D\) declares the same residual family for every learned control. A full \(64\times64\times64\) teacher delta versus an unrelated smaller student would confound the target with representational capacity. The clean first definition optimizes the teacher within the student's permitted residual family.

For a training-only future window, define

\[
 L_t(u)=\sum_{h=1}^{H}\ell\!\left(
  \widehat y_{t+h}\big[\operatorname{rollout}(B_t,m_t+D(u))\big],
  y_{t+h}\right),\qquad
 u_t^\star=\operatorname{Optimize}_{u\in\mathcal U}L_t(u).
\]

The rollout must re-encode downstream memories affected by the intervention. Reusing baseline future memories silently removes part of the proposed effect. Future images/masks enter only this training target, never the deployed head. The residual family, any constraint/penalty, horizon, teacher iteration budget and target normalization remain to be fixed prospectively; this note selects none.

The potential contribution is **amortized optimization of useful memory content**, not additional test-time evidence. A future-loss gradient is ordinary backpropagation through time. In particular, for a one-step target \(u^\star=-\eta g_t\), \(g_t=\nabla_u L_t(0)\), the loss

\[
 \tfrac12\|h_\theta-u^\star\|^2
\]

has gradient \(\eta J_\theta^\top g_t\) at \(h_\theta=0\). Direct future-loss training has gradient \(J_\theta^\top g_t\) there. This exact local equality rules out claiming a new credit-assignment principle merely from regressing the gradient. Multiple teacher optimization steps can change the target geometry and optimization behavior; whether that helps is the open empirical question.

Optimized targets need not be unique or causally predictable. Two identical prefixes can have incompatible futures. Squared target regression then averages conditional targets, which need not minimize expected future segmentation loss. Large teacher deltas can also exploit the frozen decoder in a way a causal head cannot reproduce robustly. These are failure modes, not impossibility claims about learning useful distributional regularities.

## Minimum discriminating controls and falsifier

All learned arms need the same videos/labels, sampled horizons, causal head, residual family, initial state distribution, decision rate, frozen backbone and deployment interface.

1. **Direct future-loss BPTT through the identical head.** This is the decisive comparison. Count teacher-target optimization, training and caching in the distillation budget; give the direct arm a matched total compute budget as well as a common-data/common-update comparison.
2. **Current-mask denoising / GT-mask memory encoding target.** Use the same head and full permitted training annotations. This distinguishes improved current-mask representation from specifically downstream usefulness.
3. **Memory-feature and relational KD.** Same head/task loss, with the already-established feature/relationship targets, rather than granting the proposed arm extra teacher capacity or labels.
4. **Native FIFO, current-quality admission, and learned slot utility.** They isolate content modification from write scheduling. SAM2RL-style learned eviction is the relevant strong family; a cheaper adaptation must be named as such, not claimed to reproduce the published model.
5. **One-step versus optimized targets.** A necessary diagnostic if the claimed increment is target optimization rather than ordinary gradient learning.

A bounded future experiment should first ask whether the optimized write improves actual downstream masks over both zero residual and GT-mask-encoded memory within the declared residual family. The teacher result is an upper bound only. If that opportunity exists, the causal student must beat direct BPTT and the same-information target controls on **parent-disjoint validation videos**, not just reduce target MSE. Failure at any of those distinctions closes that candidate without increasing horizon, teacher capacity or selecting favorable occlusions after outcomes.

A small positive result would justify a larger separately frozen validation; it would not establish novelty or the standing substantial two-source result. Report whole-sequence quality and predeclared occlusion/reappearance strata together. A lower training cost at genuinely matched quality could also be a useful result, but requires complete preprocessing/target-generation accounting and a defined quality-equivalence criterion.

## Feasibility and source questions before implementation

The memory tensor itself is about 1 MiB in float32; this says little about unrolled backward activations. Frozen encoder features may be cached, but policy-dependent memory outputs cannot simply be reused across interventions. Multiple teacher optimization steps can dominate the training budget. Existing CPU inference qualification does not qualify autograd, ≤8 GiB training, or MPS.

The first engineering issue is the exact differentiable path: inference wrappers can disable gradients, while mask selection, thresholding and memory/pointer decisions introduce discrete branches. A training surrogate with fixed branches must be named and checked against actual causal rollouts; it cannot be called the exact native future gradient by assumption. No such qualification occurred here.

Existing SAM2.1-tiny weights could avoid new model downloads. The parent-supplied acquisition balance remains **1.289 GB**; this note establishes neither a training-data allocation nor adequate untouched validation coverage. Local four-CPU, 8 GiB, 1800-second jobs and unqualified MPS remain binding feasibility constraints. The user's research authorization is not in question; execution still needs a concrete prospective scope, resource check and frozen comparison.

## Request log and read bounds

This follow-up used web textual tools only. Search strings:

- `QDMN quality aware dynamic memory network video object segmentation memory quality training objective`
- `video object segmentation memory management reinforcement learning future reward memory update policy`
- `"SAM3RL" code github`
- `"Learning Dynamic Memory Networks for Object Tracking" "loss"`
- `MemSAM memory reinforcement spatial memory feature SAM training video segmentation`
- `video object segmentation future loss optimize memory content distillation meta optimization`
- `"MemSAM" arxiv`
- `site:github.com/dengxl0520/MemSAM "reinforcement"`

Successful primary text reads: SAM3RL v1 HTML/abstract and the sections identified above; QDMN v1; SAM2RL2025 v1; Adaptive ROI v1; MemTrack v1; MemSAM official README, `train_video.py`, `modeling/mem.py`, `modeling/mem_modules.py`. Targeted section opens/finds revisited these pages. A GitHub models-directory click failed; raw official text files above were subsequently read. MemSAM paper and the HFVOS hit were indexed excerpts/abstract only. No explicit PDF request, repository archive, checkpoint, dataset, scientific output or package was downloaded.

Web-tool response-body byte counts are unavailable. No zero-byte claim or exact network debit is made. The search is bounded and does not certify absence of a closer 2026 method.
