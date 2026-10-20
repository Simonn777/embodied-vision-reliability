# Narrow detail-refinement prior check

2026-10-20. Source-only; three method reads. No 0032 outputs, arrays, labels,
model runs, or executable-source changes. This note neither admits a successor
experiment nor declares novelty from an absent search hit.

**Assessment:** enlarging detail, deriving prompts from a coarse mask,
selecting by predicted IoU, and learning where to replace coarse predictions
all have direct precedent. The current fixed 0032 adaptation is appropriately
a practical baseline. A useful empirical result alone would not establish a
new refinement mechanism.

## 1. SAMRefiner — ICLR 2025

[Official conference record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html);
[author full text v1, §§3.1–3.3, Fig. 3, Eqs. 1–4](https://arxiv.org/html/2502.06756v1).

Read the actual prompt and ranking definitions. Positive points maximize
distance to background; negative points use the converse distance inside the
foreground bounding box. CEBox uses a foreground feature mean and surrounding
feature similarity to decide directional box expansion, in 10% steps. This is
context-dependent box prompting, not the fixed crop-and-reencode operation in
0032. Its Gaussian dense prompt replaces a brittle binary input. The base
method selects the highest predicted IoU among three masks. The ++ variant
changes selection through a small IoU-head adapter with pairwise hinge loss,
using coarse-mask agreement as training supervision. It does not change mask
generation.

Thus EDT points, coarse-mask prompt construction, and pIoU selection are
explicitly occupied. This is primarily image-mask postprocessing, not proof
of causal temporal fusion. Read boundary: author v1 method text, conference
venue record; no code replication or final-PDF equivalence audit.

## 2. SAM2Refiner — ICCV 2025

Yao et al., *Towards Fine-grained Interactive Segmentation in Images and Videos*.
[Official record](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_Towards_Fine-grained_Interactive_Segmentation_in_Images_and_Videos_ICCV_2025_paper.html);
[author v1, §§3.2.1–3.3, Eqs. 1–3](https://arxiv.org/html/2502.09660v1).

Actual method: encode the full image plus four enlarged nonoverlapping
quadrants. Pool/reassemble local features and cross-attend with global queries,
`G_ref = CrossAttn(G,L,L)`. Prompt retargeting fuses a spatial prompt map
(positive clicks, negative clicks, mask) with enhanced object embeddings.
A UNet-like decoder combines object and hierarchical image features, then
upsamples to 1024². It trains added modules using final and intermediate
BCE+Dice supervision while freezing SAM2. Section 3.3 explicitly connects
the modules to previous-frame mask-feature memory for video inference.

This directly covers high-resolution local/global detail fusion combined with
temporal object information. It is not an object-bbox crop, an output-only
frozen decoder, or a same-cost reproduction of 0032. Read boundary: full
relevant author v1 methods and training/inference; venue verified separately.
The conference PDF was indexed but not independently reconciled to v1.

## 3. SAM-REF — CVPR 2025

Yu et al., *Introducing Image-Prompt Synergy during Interaction for Detail
Enhancement in the Segment Anything Model*.
[Official record](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_SAM-REF_Introducing_Image-Prompt_Synergy_during_Interaction_for_Detail_Enhancement_in_CVPR_2025_paper.html);
[author v4, §§3.2–3.6, Eqs. 1–6](https://arxiv.org/html/2408.11535v4).

The global refiner fuses RGB, positive/negative prompt maps, coarse logits,
and encoder features. Its learned error map gates replacement:
`M_out = sigmoid(E) * M_new + (1-sigmoid(E)) * M_coarse`.
Error heads receive GT-derived error supervision. The local stage selects a
prompt-containing connected mask-difference region, expands its box, RoIAligns
logits/features, and applies another gated refinement. A ratio of successive
predicted error-map areas chooses whether the local result is pasted onto
the global result or the preceding interactive mask.

This is a strong direct precedent for object-conditioned detail repair and
learned error/uncertainty-style gating. Its “previous” mask is a previous
interaction, not automatically a previous video frame. Read boundary:
v4 full method; earlier v1 was consulted then superseded. Conference PDF
opens failed in the web tool. No independent author implementation audit.

## Consequences for interpretation

The three mechanisms above are different controls, not interchangeable names.
SAM2Refiner supplies the explicit temporal-memory connection; SAM-REF supplies
learned gated replacement and local patch selection; SAMRefiner supplies
automatic coarse-mask prompts and pIoU selection. An error-map gate is not,
without calibration evidence, a calibrated uncertainty estimate.

0032 combines fixed known operations with an extra encoder call. Its outcome
cannot isolate crop enlargement from prompt changes, decoder rerunning, or
candidate selection. Nor can a failure exclude trained high-resolution
refinement. Any later contribution would need a specific intervention and
matched-information/cost controls beyond these generic ingredients; this note
does not choose or authorize one.

## Read/request boundary

Web-text only: four small search batches (9 queries total), followed by scoped
method opens of the three papers above. Additional search results were used
only to identify the closest third method; no fourth method was analyzed.
CVPR 2025/2026 searches found SAM-REF as the closest directly read detail
refiner; this is not an exhaustive 2026 clearance. Two SAM-REF conference-PDF
opens and one broad CVPR-2026 index open returned tool errors; author HTML v4
resolved the method read. No local PDF, dataset, checkpoint, or source-archive
download. Web-text transfer sizes are unavailable, not reported as zero.
