# Initial-mask-only output adaptation: closest controls

Source-only review, 2026-10-20. Exactly three method families read. No arrays, annotations, models, code execution, or payload acquisition. Web text/PDF extraction access is unmetered; no separately downloaded paper/source payload. This note defines prior overlap and comparison requirements, not an admitted experiment.

**Conclusion:** fitting an instance-specific segmentation function on one annotated frame and then freezing it is established supervised one-shot adaptation. Freezing SAM and learning only its output combination is also directly established. A small spatial residual head on frozen SAM2 features would be a different parameterization, but neither the first-mask supervision nor the freeze-after-fit schedule is a new information contract. It remains a legitimate strong control; known components alone do not establish either efficacy or impossibility.

## Three primary methods

### PerSAM / PerSAM-F — ICLR 2024

Read the [final paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/4f44e5c82dc7b0ba4472083e5dedb08e-Paper-Conference.pdf), §§2.2–2.3, Eq.11, Appendix C.1–C.2.

PerSAM uses the reference image/mask to construct target features, location prompts and decoder guidance; it is training-free. PerSAM-F instead freezes SAM and learns **two scalar weights** combining its three mask scales, supervised by the single reference mask. Its personalized-image implementation uses 1,000 epochs, batch one, Dice plus focal loss and AdamW. The DAVIS adaptation uses the first frame for 800 epochs; its published video configuration also uses DINOv2 for localization and previous-frame box prompts. Thus copying only two-weight fitting onto SAM2 would be an adaptation of this control, not full PerSAM-F reproduction.

The quoted roughly ten-second fit is on **one A100**, not evidence of our CPU runtime. Its output mixing is less expressive than a spatial residual head, but directly occupies the “frozen SAM + initial annotation + small learned output correction” claim. It does not train on future annotated frames.

### OSVOS — CVPR 2017

Read [author manuscript v3](https://arxiv.org/html/1611.05198v3), §§3.1–3.2 and §4 Timing; [final conference record](https://openaccess.thecvf.com/content_cvpr_2017/html/Caelles_One-Shot_Video_Object_CVPR_2017_paper.html). The final PDF endpoint did not return usable text.

OSVOS starts from ImageNet initialization, trains a foreground/background parent FCN on DAVIS training masks, then fine-tunes the network on the test sequence's first annotated image. The resulting weights segment later frames independently. It is network adaptation, not a frozen-backbone residual head, and its offline supervision must be disclosed in any reproduction.

The paper explores **10 seconds to 10 minutes of per-sequence fitting**, separately reporting 102 ms per 480p frame. These are author-reported timings, not CPU feasibility evidence; the inspected manuscript passages did not establish a hardware-comparable budget. Its direct relevance is the first-label-only training-and-freeze paradigm. A head-only version economizes that paradigm rather than introducing it.

### CAV-SAM — ICCV 2025

Read [author manuscript v1](https://arxiv.org/html/2508.07759v1), §§3.1–3.4 and §4.2; verified [final conference record](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Correspondence_as_Video_Test-Time_Adaption_on_SAM2_for_Reference_Segmentation_ICCV_2025_paper.html). Final-PDF extraction failed.

Its TTGA component fine-tunes **only the SAM2 image encoder's FPN neck**, using one labeled reference image and corresponding augmentations. Prototype similarities receive augmented-mask and cyclic BCE supervision (Eqs.3–8); the resulting pseudo-masks become SAM2 prompts. It specifies SAM2 tiny and **100 adaptation steps per reference**. The complete method additionally fits diffusion LoRA for 200 steps, uses 20-step DDIM inversion and a nine-frame semantic transition sequence. Therefore its full cost/information use is much larger than cached-feature output refinement; no comparable CPU wall time was verified.

This establishes single-reference supervised SAM2 adaptation, but not the precise residual-after-native-output operator. Also, its task is reference/class segmentation using a target-conditioned pseudo-video, not our causal same-instance rollout. A TTGA-only causal adaptation would need its own name and measured accounting.

## A material native-input mismatch

The already installed SAM2 source was inspected without importing it:

- `sam2/modeling/sam2_base.py`, `_use_mask_as_output`, around lines 420–467; SHA256 `69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41`.
- `sam2/sam2_video_predictor.py`, `add_new_mask`, around lines 320–375; SHA256 `912555920a77f72ded07839efa33fcceecc79f3523abbec783f0be46ee2c55c2`.

The native initial mask is not an ordinary erroneous decoder prediction. `add_new_mask` resizes the supplied annotation to model resolution and thresholds it; `_use_mask_as_output` makes logits `20*mask−10`, then antialiased bilinear downsamples them. Presence follows whether the supplied mask contains foreground; IoU is a dummy one. The decoder can still supply the object pointer, which does not change this mask construction.

Consequently, fitting a residual from that initial low-resolution mask to the original annotation can learn reconstruction of an annotation's resize/threshold errors. Later masks instead contain model confidence, semantic and localization errors. Low initial training loss cannot establish correction of those errors. This is an input-distribution mismatch, not a proof that a feature-conditioned head cannot generalize. Replacing the first input with an ordinary decoder prediction or synthetic corruption would be a distinct declared training protocol, not a silent implementation detail.

## What a future comparison must distinguish

The inexpensive faithful comparison concept is **first-mask supervised adaptation**, retaining native history and evaluating only the emitted-mask change. Native output, a same-feature ordinary supervised residual head, and a PerSAM-F-style output-mixture control separate extra spatial capacity from the known personalization recipe. A feature-only foreground classifier is also a useful same-label ablation if the proposed residual merely learns target appearance. Multi-mask generation and fitting costs must be charged; a SAM2 mixture port must not be called exact PerSAM-F.

All such arms should receive exactly the same first annotation, with fitting schedules/losses selected without future DEV labels and frozen before later frames. The fully measured cost includes initial fitting as well as rollout. Full OSVOS or CAV-SAM are prior/context controls, not automatically affordable compulsory reproductions; omitting them for budget does not justify claiming their broad mechanism as new.

No claim is made that these three reads exhaust prior art or that a spatial head is an exact reproduction of any one paper. The presently defensible designation is a **generic one-shot personalization control**, with unresolved transfer from annotation-derived initial logits to native future errors.
