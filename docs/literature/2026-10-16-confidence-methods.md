# Confidence, visibility, and causal tracking

2026-10-16 · AI-assisted literature/code review for protocol 0002. No weights/data downloaded or inference run. Protocol 0001 unchanged. Findings below are source evidence; recommendations are interpretations.

## S1 — TAPIR

**TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement** — Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, Andrew Zisserman. ICCV 2023. [arXiv v2](https://arxiv.org/abs/2306.08637v2), 2023-08-30.

Read [full-text](https://arxiv.org/html/2306.08637v2) §§3–6 and Appendix B.4/C.1; not the whole paper. Separate heads predict occlusion and thresholded localization error. Refinement updates both scores and coordinates temporally. Original TSM features require future frames; C.1 also uses whole-segment position normalization. The released variant differs (§6), including its backbone and six-pixel error threshold. §5 describes DAVIS-informed model selection.

Paper license: [arXiv non-exclusive distribution](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html), not Apache-2.0.

## S2 — CoTracker3

**CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos** — Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht. [arXiv v1](https://arxiv.org/abs/2410.11831v1), 2024-10-15; [ICCV 2025 record](https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html) verified through indexed official metadata (spells Makarov “Yuri”). Conference-PDF equivalence unchecked.

Read [full-text](https://arxiv.org/html/2410.11831v1) §§3.2–3.3, §4 evaluation protocol/4.1 and Appendix A. Confidence targets localization within 12 pixels; visibility has separate supervision. Temporal attention refines both. Online processing uses overlapping windows with half-window advances. Evaluation combines visibility/confidence and distinguishes visible/occluded localization where annotations permit.

Paper license: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## S3–S4 — official code inspected

- **S3 TAPNet**, commit `730cda1c730877cfedbe01bf87fb1cadb78a565d`: [causal PyTorch notebook](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/colabs/torch_causal_tapir_demo.ipynb) uses `causal_bootstapir_checkpoint.pt`: **Online BootsTAPIR**, not original TAPIR. It carries per-frame state and thresholds `(1−sigmoid(occlusion))×(1−sigmoid(expected_dist))`. [Model documentation](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/torch/tapir_model.py#L423-L431) identifies `expected_dist` as a logit, not pixel distance. [README](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/README.md#L244-L259) declares Apache-2.0 for non-TAPVid-3D software and linked checkpoints; dataset rights differ.
- **S4 CoTracker**, commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`: [online predictor](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/predictor.py#L212-L309) defaults to window 16, step 8; returns Boolean `(visibility×confidence)>0.6`, although its underlying model returns both channels. [LICENSE.md](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md): CC BY-NC 4.0, with [README exceptions](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/README.md#L334-L336). Individual checkpoint licensing/checksums unchecked.

## Interpretation for protocol 0002 — proposed, not adopted

1. Keep identical frozen coordinates. Compare raw native visibility, localization confidence, their product, and temporal augmentation. Refined confidence is a **temporal baseline**. An initializer's score targets different coordinates; do not silently pair it with refined outputs.
2. “First query” and “online” do not establish immediate causality. Specify input cutoff, emission latency, revisions, query initialization and support points. Require unchanged emitted predictions/scores after altering future frames. Preserve raw scores before Boolean wrappers discard ranking information.
3. Protocol 0001's rejected predictions still enter motion history; stable wrong correspondence can become consistent. Test abrupt correct motion, occlusion/reappearance and identity switches separately on development data. Appearance anchoring remains an untested hypothesis.
4. Predefine resize coordinates, failure threshold and coverage denominator. Separate visibility errors from localization risk; score occluded locations only where valid annotations exist. Native AJ/OA supplement selective-risk analysis. Upstream confidence thresholds need not match our endpoint.
5. Freeze checkpoint, support-query policy and sequence-level splits/calibration before testing; bootstrap sequences. Previously used benchmarks are not completely unseen upstream. A fitted error predictor is a future stronger control, absent from 0001.

Next: inspect one causal adapter's score interface and prefix behavior. M5 CPU/MPS feasibility, checkpoint integrity and ranking quality remain unmeasured. These findings establish controls, not novelty or a positive result.
