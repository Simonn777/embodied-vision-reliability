# One remaining information question: target motion outside its predicted mask

Status: conceptual proposal only. No scientific arrays, labels, saved outputs, model execution, or new payloads were accessed. The 0030/31/32 outcomes below were supplied by the parent. Their fixed variants remain closed.

**Recommendation:** consider one small TRAIN information screen for *target-versus-background motion evidence at pixels omitted by SAM2*. This is materially different information from the three failed interventions, but **it is not presently a defensible new-method claim**. Its strongest direct prior must be a control, not an omitted citation.

## The causal hypothesis

0030 changed stored feature amplitudes; 0031 compared initial-frame appearance pairs; 0032 supplied a larger current-image crop to the same frozen model. None directly asked whether a currently excluded pixel moves with the target rather than its neighbouring background.

Suppose a textured target rim is absent from both previous and current native masks. Transporting the previous mask cannot restore it. Nevertheless, its RGB displacement can agree with a motion estimated from the target interior and disagree with motion estimated from the adjacent exterior. That supplies a physical membership cue not present in the transported binary label. This is the specific opportunity—not “temporal consistency” by itself.

At a current pixel \(p\), let \(W_F,W_B\) be backward local motion hypotheses estimated from disjoint, native-mask-defined interior/exterior supports. A possible evidence statistic is
\[
S(p)=D(I_t(p),I_{t-1}(W_B(p)))
     -D(I_t(p),I_{t-1}(W_F(p))).
\]
Positive \(S\) favours target motion. The evaluated boundary pixels must not also fit the motion hypotheses. There is no new sensor and no claim that this evidence is statistically independent of SAM2. The difference is an explicit RGB temporal measurement that can support **persistent omissions**, not another choice among SAM2 mask candidates.

Motion equality, poor texture, aperture ambiguity, deformation, blur, illumination changes and occlusion can destroy this information. A target may also be behind the adjacent surface. Therefore target identity must never be equated with “front layer”, and an unavailable/ambiguous observation must leave the native prediction unchanged. Good forward/backward agreement is a reliability check, not proof of correct correspondence.

## Cheapest discriminative check before any training

Use the existing four TRAIN parents and all fixed 48-anchor × {1,3} events, retaining duplicate-event accounting and parent means. Only current and already observed RGB/native outputs enter scoring. No new encoder, checkpoint, future frame, GT-selected location, or head is needed.

Before execution, fix a bounded, sparse local RGB-motion estimator and one target-independent boundary sampling rule. Estimate competing motions away from the evaluation band; score held-out band pixels. Retain out-of-image, textureless, occluded and failed-fit cases in the denominator with a native no-op. A sparse pyramidal image-registration implementation using the installed OpenCV is a plausible CPU implementation, **not yet a qualified runtime or a frozen algorithm**.

The useful discriminator is whether motion evidence correctly admits *currently excluded* target pixels **without admitting more background**, compared with:

1. Native logits at the same pixels.
2. Ordinary previous-native-mask warping and fixed equal-weight fusion using exactly the same computed motion evidence and cost.
3. The same local two-motion comparison with its native-layer prior/coupling retained—the reduced localized-layer baseline below. This is the strongest mechanistic control.

Report failure-inclusive signed correction utility, coverage and full-mask J/F for a single predeclared no-training correction, as well as parent results. A GT breakdown of persistent omissions versus transient errors is descriptive only; it must not select eligible pixels. No oracle-selected correction counts as success.

**Do not train merely because the motion score has positive AUROC on a selected subset.** Continue only if the actual fixed correction beats native and ordinary mask propagation overall and in at least three of four TRAIN parents, without a loss on the fourth. A tiny gain would still not justify an expensive head: its measured full-frame improvement must plausibly support the existing substantial final research gate. Freeze a numerical effect-size requirement before any run. If signal is absent, mostly unavailable, or fully explained by the localized-layer control, stop this proposed direction instead of tuning flow/geometry on these events.

## Closest prior and the actual remaining difference

- **Semantic Optical Flow, CVPR 2016:** [Sevilla-Lara et al., *Optical Flow with Semantic Segmentation and Localized Layers*](https://arxiv.org/html/1603.03911v2), §§3.1–3.2, Eqs. 1–6, read in full. It already estimates local foreground/background motion jointly with segmentation, uses visibility-aware photometric evidence and a segmentation coupling term, and initializes motion away from uncertain boundaries. This directly occupies the core physical mechanism. Replacing its semantic mask by SAM2 is not an invention.
- **FlowVOS**, author preprint v1, 20 November 2021: [§§3.1–3.4](https://arxiv.org/html/2111.10621), read in full. Its learned object-conditioned warp transports the previous mask, with mask/foreground-image losses and decoder fusion. It is the explicit detail-preserving temporal-propagation precedent. The narrow distinction proposed here is measuring membership outside both native masks, rather than simply propagating their foreground support.
- **Motion-Guided Cascaded Refinement Network, CVPR 2018:** [author-hosted paper](https://www.ntu.edu.sg/docs/librariesprovider106/publications/image-processing/motion-guided-cascaded-refinement-network-for-video-object-segmentation.pdf?sfvrsn=559dd620_2). The retrieved abstract/introduction explicitly combines differential motion segmentation with learned refinement. I did not complete a method-equation read for this third paper and do not claim an exact implementation comparison.

Thus a positive screen would establish an attainable **information advantage over the failed actions and simple mask transport**, not novelty over localized motion segmentation. A scientifically defensible successor would still need an actual improvement over that strong same-information baseline; there is no such evidence now. If the parent requires a ready novel mechanism rather than a tightly scoped information check, my answer is **none established under the current constraints**.

## Read/request boundary

Read the existing private `0030-memory-content/output-repair-prior.md`, `paired-boundary-question.md`, and `0032-crop-detail/detail-prior.md`; no result files were opened. Four search queries in two batches; primary web-text method reads as above. Two direct CVF opens returned 403; author arXiv/official-mirror text resolved the relevant prior. No local PDFs, repositories, weights or datasets were downloaded. Web-tool body bytes are not exposed and are not claimed as a precisely metered zero.
