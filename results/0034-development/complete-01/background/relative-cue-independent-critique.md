# Independent critique: original-resolution RGB versus feature Gram cues

2026-10-20. Source/text review only. Read `target-conditioning-source-question.md` and three primary works below; no image, label, cache array, prediction outcome, model, code, specification or admission was created/read/run for this task. No payload or local paper download. This is a recommendation for an information-availability comparison, not a method or novelty claim.

**Recommend one direction: a target-conditioned, original-resolution local RGB readout.** Give it priority over another finest-feature Gram head. Its justification is a concrete, testable observation difference at the correction interface: actual pixel colors can distinguish positions within the interpolated native output grid. The first question is whether this information produces transferable corrections beyond ordinary RGB-guided refinement—not whether a new name for relative features sounds more general.

## What changes, and what does not

The existing finest `F[32,256,256]` is already a native decoder input. A Gram map computed from it is a deterministic re-expression, potentially with useful regularization, but adds no observation. For normalized features,

`||q_i − q_j||² = 2 − 2 q_iᵀq_j`.

Thus many proposed “Gram relation” cues are ordinary similarity/distance cues. Channel-basis invariance alone does not establish a domain-transfer mechanism: the encoder uses one fixed basis, and an absolute-feature readout can absorb a common basis rotation. Neither point rules out empirical benefits from the inductive bias.

Original-resolution RGB also comes from the same image that SAM2 saw; **it is not a new sensor, view, or temporal observation**. The narrower distinction is that the cheap readout receives color samples directly at the final pixel locations instead of only a cached representation and interpolated logits. There can be two nearby pixels with similar native logits but different color affiliation to the target interior/background. A color-conditioned output can distinguish them without rerunning the encoder. Whether `F` already encodes their relevant distinction remains an empirical question: a 256² spatial grid with 32 channels is not proof of information loss.

A useful falsifiable statement is:

> Holding target conditioning, readout, supervision and native history fixed, original-resolution color relationships improve out-of-parent correction compared with the same relationships computed from RGB downsampled to 256² and then interpolated back, and compared with ordinary same-RGB foreground/background affinity refinement.

The first comparison tests a resolution-specific source of useful evidence. The second prevents an ordinary bilateral/nearest-neighbour improvement from being mislabeled a new relational mechanism. Passing only the first establishes available fine-detail information, not a new algorithm.

## Restricted readout worth testing

Use native logits and the native mask to define a fixed boundary band plus local, sufficiently interior foreground/background reference supports. Held-parent supports must use its **native prediction**, never its current GT. Exclude the query neighborhood from its own reference support. Empty supports, absent targets and other prespecified invalid cases return the native output and remain in the full denominator.

At each permitted output pixel, supply a small shared readout with: native logit; foreground/background color distances or affinities; and a few fixed-offset RGB differences that describe the local transition. The readout predicts a bounded logit correction at the original pixel grid. No extra image encoder, identity template optimization, frame-specific fitting, or history update is needed for this screen.

Do not make a single global mean color the only serious comparator. A target can contain several colors, and “background” is highly multimodal. A fixed-cap, locally restricted foreground/background nearest-neighbour or kernel-affinity control is harder to defeat and uses the same evidence. Reference cardinalities, radius, temperature, band, correction bound and missing-support policy must be fixed before execution. Low-level color distances also require an explicit color space; normalized RGB dot products discard intensity information. Calling sRGB interpolation a physical compositing model would be unjustified.

This is deliberately a **known-baseline family**. A gain would support further investigation of the observation/readout contract. Without a substantive additional mechanism and stronger efficacy evidence, it is not a paper contribution.

## Closest primary overlaps and mandatory controls

**1. Krähenbühl and Koltun, “Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials,” NIPS 2011.** Read §§2–4, especially Eqs. 1–4 and Algorithm 1 in the [official proceedings paper](https://proceedings.neurips.cc/paper/4296-efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials.pdf). Its appearance kernel combines spatial distance and RGB distance, alongside spatial smoothing; mean-field updates combine these affinities with unary predictions. This directly occupies generic image-guided sharpening. A native-logit unary plus same-RGB affinity refinement is a mandatory control. A truncated local bilateral implementation must be named as such, not presented as an exact fully connected CRF reproduction. Its hyperparameters receive only the same permitted TRAIN fitting.

**2. Yuan et al., “SegFix: Model-Agnostic Boundary Refinement for Segmentation,” ECCV 2020.** Read §§3.1–3.3, Eq. 1 and the discussion on page 9 in the [final ECVA paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf). It predicts boundaries and directions toward interior pixels, then transfers their labels. Reliable-interior-to-uncertain-boundary reasoning is already explicit. A cheap interior-label transport control is relevant, but is not the full trained SegFix architecture. The shared critical assumption is interior correctness: erosion does not certify target ownership. A wrong native object, missing thin structure or contaminated reference can make refinement confidently wrong. Report this limitation without selecting only correctly seeded cases.

**3. Forte and Pitié, “F, B, Alpha Matting,” author preprint.** Read §1 and §§3.1–3.3 in [arXiv:2003.07711](https://arxiv.org/pdf/2003.07711); retrieved text identifies v1, 17 March 2020, so no final-publication equivalence is asserted. RGB, foreground/background guidance, and fine-detail reconstruction are established ideas; the method jointly estimates colors and alpha and uses compositing consistency. However, a native-derived pseudo-trimap is not a correct user trimap, DAVIS provides binary instance masks rather than alpha mattes, and the full pretrained matting network adds different supervision and compute. It is a strong conceptual prior, not a fair lightweight drop-in unless those differences are explicitly admitted. A same-support local color-mixture/affinity baseline should precede any claim of new matting-like inference.

## Four-TRAIN leave-one-parent-out falsifier

Use the already declared 96 TRAIN events, 24 per parent, with repeated events weighted explicitly and all duplicates confined to their parent fold. No DEV or protected-source selection. Fit readout parameters/statistics on three parents and predict the fourth; repeat for all four. Native state and the prediction-derived support rules remain identical across arms. Parent-equal original-resolution J/F and each held-parent difference are primary; pixel ranking, fit loss and boundary-only scores are secondary diagnostics.

The minimum informative comparison is:

1. Native/logit-only and the existing absolute finest-feature readout as reference controls.
2. Ordinary same-RGB target affinity/nearest-neighbour refinement, plus a declared RGB-guided unary/bilateral control.
3. The proposed small local RGB relation readout, paired with its **identical downsampled-then-upsampled RGB version**.

A matched direct RGB/logit patch readout is also necessary if superiority of the *relative representation* is claimed. Otherwise an improvement could come from providing RGB to any supervised head. A prototype-only win is insufficient when the stronger multi-color affinity control matches it. If a feature-Gram arm is retained, it is a comparator under the same correction sites and training allowance, not the primary new-evidence story.

Before seeing fold outputs, fix a practical J/F increment and no-harm tolerance. A credible necessary screen should require positive parent-equal gain over both native and the strongest same-input control, gains on at least three of four held parents, and no material harmed parent. This note intentionally does not create numerical admission thresholds; the design owner must freeze them once, together with the finite readout/parameter budget. Four exposed parents provide development evidence, not an independent confidence claim.

The specifically diagnostic outcomes are:

- **Original RGB beats its degraded version and same-RGB known controls:** evidence that the restricted readout can use additional spatial detail; investigate why ordinary affinity fails before proposing novelty.
- **Original RGB helps but ordinary affinity is as good:** useful known refinement baseline; no demonstrated relational increment.
- **Fine RGB and degraded RGB are equivalent:** no support for a subgrid-detail explanation at this readout/budget.
- **Only in-parent fitting improves, or one parent dominates:** no evidence of transferable correction; do not select a favorable class, band or fold after the fact.

## Main ways the hypothesis can fail

An RGB edge need not be the target boundary: shadows, fur texture, reflections and background texture supply misleading edges, while camouflage and motion blur erase real ones. Native-based references can perpetuate identity errors; current-frame color has no independent object-identity oracle. Fine structures can have no safe native interior at all. The DAVIS boundary metric also tolerates a scale-dependent distance, so subpixel-looking improvements can produce little J/F gain. These are reasons to measure full masks, not reasons to assume the test will fail.

No timing guarantee follows from this source review. Local supports and fixed offsets can avoid an all-pixel quadratic affinity matrix and stream frames, but readout fitting and full-resolution evaluation still require measured complete costs. No model/package acquisition is needed merely to state the comparison.

**There is no driving or embodied-performance claim.** Generic DAVIS target-mask refinement does not measure ego-motion robustness, object tracking identity, dynamic-agent localization, occupancy or control. Even a useful LOPO result would remain a small, exposed VOS development result.

## Read/access scope

One three-query primary-source search batch, then method-text reads of the three papers above. An arXiv HTML request for the matting paper failed; its PDF text was read through the web tool instead. No local PDFs, repositories, weights or data were downloaded. Web text response bytes are not available as a metered transfer ledger. PerSAM/FEELVOS/SAM2Refiner claims in the tracking note were not independently re-audited here; no fourth primary work was added.
