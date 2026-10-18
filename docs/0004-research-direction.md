# 0004: When does resetting tracker state help?

**Decision date:** 2026-10-18. **Status: hypothesis and development plan.**
This AI-assisted direction has no demonstrated improvement, established novelty,
or manuscript-ready result. Contributions follow [the repository policy](CONTRIBUTIONS.md).

## Why change direction?

Experiment 0002 filtered saved predictions using temporal residuals. Its
[post-hoc diagnosis](analysis/0002-motion-residual-posthoc.md) found that correct
high-curvature predictions were displaced by smoother wrong predictions. The
observed baseline headroom was also insufficient for its unchanged improvement
target. Those results and inspected evaluation groups remain historical evidence;
they will not become an untouched test set again.

0004 changes the question from identifying risky outputs to choosing an operation
that improves subsequent tracking. An error warning does not imply that deleting
history helps: useful motion or appearance memory may be lost as well.

## Falsifiable claim

For a frozen point tracker, a small predictor of **short-term-state reset benefit**
can improve causal tracking more than predicting current error, at the same
deployed computation budget. The proposed training target is the difference
between retain-state and reset-state continuations from the **same pre-action
state**, with identical original query, weights, current image features and
future input stream. Only the declared state component changes.

Future annotated frames may supply training/development outcome labels; deployed
decisions must use information available at the decision time. Both branches use
the same fixed continuation policy when constructing these labels. Any shadow
probe used at deployment is charged to computation and latency. This is an
intervention in an algorithm, not evidence of physical causality in a scene.

The claim fails if resets provide negligible achievable benefit, benefit cannot
be predicted beyond confidence-only controls, or gains disappear with a stronger
tracker, independent data, or matched cost. A pilot containing too few errors or
reappearances is **undersensitive**, not evidence for or against transfer.

## Closest work checked

Read depth describes the actual review; selected full-text sections are not a
claim to have read every page. Versions below were checked on 2026-10-18.

| Work, authors | Year / version / status | Read depth and consequence |
|---|---|---|
| [Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](https://arxiv.org/html/2603.12217v1) — Görkay Aydemir, Fatma Güney, Weidi Xie | 2026; arXiv v1, March 12; CVPR 2026 | Methods §§4.1–4.5 and selected training/experiment sections. Learned reliability, candidate selection and pseudo-label distillation already exist. The verifier uses unmasked temporal attention. [Code](https://github.com/gorkaydemir/track_on), MIT. |
| [Track-On2: Enhancing Online Point Tracking with Memory](https://arxiv.org/html/2509.19115v2) — Görkay Aydemir, Weidi Xie, Fatma Güney | arXiv v2, March 16, 2026; v1 September 2025. Author repository labels TPAMI 2026; final IEEE version not reviewed | Selected prediction/memory sections and §IV-E7. FIFO, write intervals and short/long memory partitions are already compared. [Code](https://github.com/gorkaydemir/track_on), MIT. |
| [ReTracker: Exploring Image Matching for Robust Online Any Point Tracking](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.html) — Dongli Tan, Xingyi He, Sida Peng, Yiqing Gong, Xing Zhu, Jiaming Sun, Ruizhen Hu, Yujun Shen, Hujun Bao, Xiaowei Zhou | ICCV 2025 accepted version, pp. 4306–4316 | Abstract, selected training/dataset text and supplement §§1.5–1.6. Recovery and confidence-gated historical patches are established. [Code](https://github.com/re-tracker/ReTracker), Apache-2.0. |
| [TAPNext++: What's Next for Tracking Any Point (TAP)?](https://arxiv.org/html/2604.10582v1) — Sebastian Jung, Artem Zholus, Martin Sundermeyer, Carl Doersch, Ross Goroshin, David Joseph Tan, Sarath Chandar, Rudolph Triebel, Federico Tombari | 2026; arXiv v1, April 12; [author page](https://tap-next-plus-plus.github.io/) identifies CVPR 2026 **Findings**, not main track | Selected method, evaluation and state-analysis sections. Long-horizon causal tracking, re-entry training and **AJ_RD** already exist. [Code](https://github.com/google-deepmind/tapnet), Apache-2.0. |
| [SAM3RL: Memory Control via Reinforcement Learning for Visual Object Tracking](https://arxiv.org/html/2610.19931v1) — Tomáš Čížek, Matej Straka, Illia Volkov, Matěj Kripner, Ervin Macić, Jiri Matas, Klara Janouskova, Martin Schmid | 2026; arXiv v1, October 14; **preprint**, no acceptance verified | §§3–3.2 and selected analysis/training appendix. A frozen tracker, fixed conditioning frame and lightweight policy optimizing future memory-action reward already exist. Paper CC BY 4.0; released-code license not established. |

**Novelty remains unknown.** In particular, SAM3RL rules out a broad claim that
learning memory actions for future tracking reward is new. The narrower candidate
is paired supervision of a specified reset intervention, rather than current
reliability scoring or PPO-based single-entry eviction. That distinction must be
useful and survive further literature review; combining existing components or
making their controller smaller does not establish novelty.

## Development first, paper only after evidence

Stage 0 is a bounded **250-frame, eight-query development test** with the existing
tracker. It checks whether the intervention is isolated and whether there is
enough beneficial and harmful reset behavior to justify training. It is not
independent-data, modern-model, or transfer evidence. Inadequate error/recovery
events require a more discriminating development cohort, not a weakened success
criterion. No Stage 0 result is claimed here.

Essential controls are native never-reset, always-reset, periodic,
budget-matched random, confidence-triggered, and appearance-disagreement-triggered
reset; the same small predictor trained on current error rather than action
benefit; and a comparable return-based memory-control baseline motivated by
SAM3RL. An action oracle is only a diagnostic upper bound.

Before independent evaluation, freeze the method, sequence/scene splits,
endpoints, interventions and the following **provisional paper gate**:

- Include at least one modern causal tracker beyond BootsTAPIR and **two real
  video datasets**, with independent held-out sequence/scene groups. Document
  pretrained-model exposure separately from our split independence.
- Target at least **+2 absolute AJ points and +5 AJ_RD points** over the strongest
  applicable control on each dataset. Use the existing AJ_RD definition; report
  paired 95% confidence intervals clustered by independent sequence/scene, with
  positive lower bounds for the primary gains. Preserve all failure strata;
  allow no more than 0.5 AJ point loss on always-visible and rapid-motion strata.
- Limit **end-to-end deployed cost to at most 1.25×** the native tracker on the
  same device, resolution, queries and sequences. Include preprocessing, feature
  extraction, controller/probes, state management and postprocessing; also report
  latency distribution, peak memory, and separate training/label-generation cost.
  Freeze this ceiling before evaluation rather than choosing it after results.
- Reproduce the comparison from pinned inputs and code. Show that gains persist
  beyond one old model or a few selected failures, and quantify harms on visible
  points and legitimate rapid motion.

These are research continuation thresholds, not predicted outcomes or conference
acceptance criteria. CPU/MPS feasibility must be measured; published GPU speed
does not establish it. Work remains within the no-additional-fees constraint.
Only reproducible results that survive these controls justify writing a paper.
