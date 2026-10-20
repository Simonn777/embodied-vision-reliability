# Target-conditioned relations after 0033: a comparative question, not a diagnosis

Source-only, 2026-10-20. No scientific arrays, masks, labels or outcome files were opened; no implementation or experiment was prepared. The parent supplied the completed 0033 results: visual fit-subset +0.6513 J&F points, DEV −0.2785, with opposite parent signs; logit-only DEV −0.2071. The fixed 0033 recipe remains closed.

**Judgment:** a target-relative representation is a reasonable *known-baseline comparison*. The potentially useful question is whether local feature relations add transferable correction evidence **beyond ordinary foreground/background similarity conditioning**. Neither the result nor the papers establish that absolute channels caused the failure. At present there is no established new-method contribution.

## One hypothesis and its limits

The absolute 32-channel head can learn associations between particular feature directions and corrections. A target-relative representation instead asks whether a pixel resembles the current target interior or its surrounding non-target region, and whether its neighbours form a coherent appearance transition. Those meanings may transfer across recordings even when the target appearance changes. This uses existing observations with a different inductive bias, not additional test-time information.

For unit feature vectors \(q_p\), let \(\mu_F,\mu_B\) be averages over fixed, target-independent-rule supports inside/outside the **native** mask. Keep evaluated boundary pixels outside those prototype supports. Candidate relative cues are
\[
s_F(p)=q_p^\top\mu_F,\qquad s_B(p)=q_p^\top\mu_B,\qquad
G_{ij}(p)=q_{p+\delta_i}^{\top}q_{p+\delta_j}.
\]
The single question is whether the local Gram cues improve cross-recording correction on top of \((\ell,s_F,s_B)\), where \(\ell\) is the native logit. Foreground/background conditioning supplies polarity; Gram alone cannot distinguish a target/background label exchange.

“Rotation-invariant” here means a **common orthogonal change of feature-channel basis**, not image rotation. These dot products are unchanged when every vector, including reference vectors, receives the same \(Q\). This is not invariance to arbitrary channel rescaling, nonlinear domain shift, independent per-frame rotations or contaminated prototypes. Per-channel normalization does not commute with arbitrary \(Q\); the normalization/metric must be specified before making any invariance claim.

There is also no evidence that different recordings actually induce orthogonal changes of the encoder basis. A global basis rotation can be absorbed by the first layer of an absolute-feature head. Such a synthetic check verifies algebra, not the cause of poor DEV transfer. Similarity can discard useful absolute semantics, average away distinct target parts, reproduce the native mask's mistakes, or mistake an internal texture edge for a target boundary. These are substantive no-go risks.

## Three closest primary precedents

1. **PerSAM, ICLR 2024.** [Official conference record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4f44e5c82dc7b0ba4472083e5dedb08e-Abstract-Conference.html); [author v2, §3.2, Eqs. 3–10](https://arxiv.org/html/2305.03048v2). L2-normalized reference-foreground/query cosine maps are averaged, then used for point selection and attention guidance; the pooled reference feature also conditions decoder tokens. Target-conditioned similarity in a frozen SAM is already explicit. Its full pipeline is not just a two-prototype residual head. Read: actual §3.2–3.3 HTML; the ICLR PDF was blocked by a browser challenge, so final-version equivalence was not audited.

2. **FEELVOS, CVPR 2019.** [Official record](https://openaccess.thecvf.com/content_CVPR_2019/html/Voigtlaender_FEELVOS_Fast_End-To-End_Embedding_Learning_for_Video_Object_Segmentation_CVPR_2019_paper.html); [author v2, §3, Eqs. 1–4](https://arxiv.org/html/1902.09513v2). It uses object/background nearest-neighbour embedding distances from the initial frame and local previous-frame matches, together with previous probabilities and backbone features, in a shared dynamic segmentation head. Thus “similarity as a soft cue, corrected by a learned head” is directly known. Unlike the proposed narrow test, it trains the embedding and retains absolute backbone features. Read: full method/training/inference section.

3. **SAM2Refiner, ICCV 2025.** [Official record](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_Towards_Fine-grained_Interactive_Segmentation_in_Images_and_Videos_ICCV_2025_paper.html); [author v1, §§3.2.2–3.3](https://arxiv.org/html/2502.09660v1). Dense positive/negative/mask prompts retarget object embeddings before hierarchical detail refinement; previous mask features support video inference. Target identity conditioning plus high-resolution refinement is occupied. The retrieved method does not impose a channel-basis-invariant local-relation bottleneck. Read: actual modules/training/inference, not an implementation audit.

The remaining difference is therefore a **restricted representation and transfer comparison**, not prototype conditioning itself. A positive result could justify studying that interaction; absence of an exact matching paper among these three is not novelty proof.

## Cheap TRAIN cross-parent falsifier

Before another convolutional-head campaign, prospectively fix a small readout and compare representations with four leave-one-TRAIN-parent-out folds. Fit any statistics/calibration only on three parents; construct held-parent prototypes from its native predictions, never its current GT. Use the already fixed 96 TRAIN horizon events and original-resolution labels for scoring, retaining all events and failures. These are exposed development recordings, not a new final test.

Required controls are: native/logit-only; absolute-channel readout; ordinary target-conditioned \((\ell,s_F,s_B)\); and that same conditioning plus local Gram. Use the same label budget, parent weighting, prediction sites, correction bound and comparable readout capacity. A capped object/background nearest-neighbour distance-map variant is the strongest **same-feature FEELVOS-style control** against a weak single-mean prototype. Do not call it an exact FEELVOS reproduction.

Prefer a fixed small linear/regularized readout for this screen rather than another spatial network. Include a prototype-swap or fixed within-frame support-permutation negative control preserving the native logits and supports' cardinalities; this tests whether target association matters rather than merely supplying extra smooth channels. Freeze all sampling, supports, missing-cue handling and regularization before execution. No DEV retuning.

The discriminative outcome is out-of-parent, failure-inclusive mask J/F, not fitted loss or pixel AUROC alone. The relational variant must beat both ordinary conditioning and the absolute-channel control on the parent-equal mean and at least three of four held parents, with no material harmed parent; define the effect-size/no-harm thresholds prospectively. If ordinary conditioning explains the benefit, call it a known baseline. If only same-parent fitting improves, or the matched conditioning control is as good, reject the proposed local-relation increment. Even a positive screen would not establish channel bias as the unique cause of 0033 failure, or meet the research's two-source efficacy gate.

## Scope and request log

Three targeted queries in one web-search batch; three primary papers only. PerSAM v3 URL returned 404; the unversioned author HTML identified itself as v2 (4 October 2023). FEELVOS v2 is 8 April 2019; SAM2Refiner v1 is 12 February 2025. Method sections were read through web text; no source archives, weights, datasets or local PDFs were acquired. Web response-body byte counts are unavailable. No new method, executor, specification, training or data access is admitted by this note.
