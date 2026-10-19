# Focused prior check: oblique-view coupling for factorized point coordinates

2026-10-19 UTC. AI-assisted screening and mathematical proposal. **Developmental admission: YES, after source, implementation, protocol and fixture review. No CVPR novelty, empirical improvement or full posterior reconstruction is claimed.** This document reports a primary-source review; no new model inference or asset acquisition occurred in that review. Pure implementation fixtures followed separately.

## Question and a more precise candidate

For a frozen causal tracker with separate coordinate logits, can one oblique reparameterization of the same RGB sequence help recover the pairing between plausible x and y values, beyond ordinary same-compute test-time ensembling?

Distinguish the question from “rotate images and average predictions.” The candidate should preserve the original one-dimensional marginals and alter only their coupling. This exposes a specific mechanism, a clear control and a plausible failure mode. Establishing novelty conclusively is not a prerequisite for a bounded pilot; the point of the pilot is to determine whether the mechanism occurs in the cached model at all.

## Closest inspected primary precedents

**SimCC / SimDR.** Li et al., arXiv [2107.03332v3](https://arxiv.org/html/2107.03332v3), 5 July 2022 (v1 7 July 2021); official author repository reports ECCV 2022 Oral. Read §§2–3 and §5. Independent axis classification, subpixel bins and label smoothing are established. More importantly, §5 explicitly identifies ambiguity in matching candidate x/y coordinates for bottom-up multi-person pose, suggesting extra embeddings. §2 describes Yin et al.'s earlier co-attention coupling of 1D heatmaps, but that underlying paper was not read. Thus neither factorization nor its matching ambiguity is new. SimCC's [official inference code](https://github.com/leeyegy/SimCC/blob/main/lib/core/function.py) implements horizontal flip testing: `validate_simdr` averages aligned marginal probabilities; `validate_sa_simdr` averages logits before softmax. This is a direct same-image TTA baseline. The mutable main code was inspected, not revision-frozen.

**DARK.** Zhang et al., [1910.06278v1](https://arxiv.org/html/1910.06278v1), 14 October 2019; CVPR 2020 according to the SimCC bibliography. Read the decoding method, Eqs. 3–11. It uses a local Gaussian/log-heatmap approximation and derivative-based subpixel mode correction, with smoothing. This is not missing-coupling recovery, but precludes presenting a more careful local mode decoder alone as new.

**Flowing ConvNets.** Pfister, Charles and Zisserman, [1506.02897v2](https://arxiv.org/html/1506.02897v2), 8 November 2015. Read §2.3 and the corresponding failure discussion. Full 2D heatmaps from neighboring video frames are aligned using optical flow and pooled, including learned weighting. Thus spatial distribution fusion and resolving false modes with other observations are established. These inputs are different time observations, can include future frames, and already contain 2D heatmaps. Our proposed inputs are different rasterizations of exactly the same observed past/current frames and expose only 1D coordinate logits.

**TAPNext.** Zholus et al., [2504.05579v1](https://arxiv.org/html/2504.05579v1), 8 April 2025, ICCV 2025 primary proceedings search record. Read §3 and Appendices C–D. Appendix D is the exact factorized-head reference: independent 256-bin coordinates, truncated soft argmax around the marginal argmax with radius 20, and coordinate cross-entropy plus Huber losses. Eq. in Appendix D uses Huber weights 0.1 per coordinate and classification weights 1.0, applied at each layer. Visibility has a separate BCE loss. Native support queries and distribution-based uncertainty already exist. We must compare against the shipped truncation rule, not global expectation.

**TAPNext++ unresolved recurrent-state analysis.** Jung et al., [2604.10582v1](https://arxiv.org/html/2604.10582v1), Appendix D. Repeated static inputs evolve the state; the better model's eigenvalue distribution does not substantially shift, leaving regulation elsewhere in the network as a proposed explanation. This is a real mechanistic gap but does not by itself support resets or redundant-frame skipping. Those interventions are not the present candidate. The primary CVF search record identifies CVPR **Findings** 2026; the direct proceedings-page open failed.

No inspected source directly implements the proposed fixed-marginal coupling from an oblique rasterization. This bounded review is not an exhaustive priority claim. In particular, the two targeted projection/tomography searches returned mostly irrelevant results; their lack of a hit is weak negative evidence.

## Exact pinned inference source and practical consequence

Repository metadata pins Google DeepMind tapnet commit `730cda1c730877cfedbe01bf87fb1cadb78a565d`. The following official text sources were inspected at that revision; no Python module was imported:

| Source | Inspected behavior | Expected SHA256 recorded in existing local adapter |
|---|---|---|
| [tapnext/tapnext_torch.py](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapnext/tapnext_torch.py#L225) | `prediction_heads`, physical source lines 225–246: returns 512 raw logits, two 256-bin chunks; multiplies logits by 0.5; masks ±20 around each argmax; renormalizes; expectation plus 0.5. `forward` returns raw logits with tracks/state. | `48072289b22439267b3a51eb4cde5e2f81f38c840b6c95b659d60032a07c691a` |
| [tapnextpp/votsp2026/model.py](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapnextpp/votsp2026/model.py#L158) | physical source lines 158–169 discard raw logits and reverse model `[y,x]` into display `[x,y]`; visibility is `vis_logits > 0`. | `e1219e2341c33e0f9124fe02d09a521ac7c91bc27505fcbebbf05741f825e46b` |
| [tapnextpp/votsp2026/utils.py](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapnextpp/votsp2026/utils.py#L55) | Coordinate helpers at physical source lines 55–121 scale display/model positions and construct `[t,y,x]` queries. Preprocessing uses BGR→RGB, bilinear resize with `align_corners=False`, and normalization to [−1,1]. | `0336c526dc2b91244c3e5b803f84b65c6323ede138340a4107441a6b9c9fdbf8` |

Hashes above were read from existing repository metadata. The parent subsequently reported matching live-cloud rehashes and supplied the corrected physical-file line ranges above; this worker did not independently read or hash the cloud cache. Browser-extracted line numbers were not reliable physical source line numbers and have been corrected. Local reference: `scripts/tapnext_native_adapter.py` lines 30–34. Expected cloud sources are under `/workspace/scratch/a349d03a0e0a/tapnextpp-recovery/source/`; existence must be rechecked before use. Source headers state Apache-2.0.

A new diagnostic wrapper can retain already returned logits without altering checkpoint inference. The original high-level wrapper does not expose them. Pixel-center offset and the y/x reversal are scientifically consequential: inverse rotation must act on the actual display coordinate convention. A clone/control must reproduce the original wrapper before any transformed run. Raw logits are obtainable, but existing saved coordinate outputs cannot reconstruct them.

## Falsifiable mechanism, using standard optimization honestly

Let p_x and p_y be full native marginals derived from the two raw-logit chunks using the native inverse temperature. Keep native readout as a separate baseline. Let p_u and p_v be the corresponding marginals from a fixed non-axis-aligned image transform R, with correctly transformed initial queries and an independent causal state. Define the transport polytope

    U(p_x,p_y) = { Q >= 0 : sum_y Q(x,y)=p_x(x), sum_x Q(x,y)=p_y(y) }.

Choose the coupling

    Q* = argmin_{Q in U} KL(Q || p_x p_y)
                        + E_Q[ -log p_u((R[x,y])_u) - log p_v((R[x,y])_v) ].

This is familiar entropy-regularized transport / iterative proportional fitting; do not rename the optimizer as a new algorithm. The candidate distinction is what evidence determines pairing while native marginal confidence is preserved. Choose the readout now: **joint MAP followed by the conditional mean in a square of ±20 native bins around that joint mode**, with a deterministic lexicographic tie rule. Apply the identical joint readout to the unconstrained product and additive-mixture maps. This reduces to the shipped independent-axis truncation for the native independent coupling, apart from floating-point implementation tolerance. Do not switch after outcomes to a utility-maximizing decoder optimized for the TAP metric. Such a decoder is a separate potential study, not a rescue of this pilot. All geometric and numerical details still need to be frozen before predictions. **A global mean cannot show a gain:** fixed marginals force its x/y expectation to equal the original full-marginal expectations. This is an essential implementation sanity check, not an optional ablation.

For an axis-preserving flip, 90-degree rotation or independent axis scaling/translation, the projected cost separates into a function of x plus a function of y. Its expectation is constant throughout U; the unique positive-marginal entropy optimum is the independent coupling p_x p_y. Hence these operations cannot add coupling under this objective even when their predicted marginal values differ. Oblique rotation yields a nonseparable cost and can change the pairing. This gives a sharp algebraic negative control.

Analytic support: equal mass on (−a,−a),(a,a) and equal mass on (−a,a),(a,−a) have identical original marginals, while 45-degree projected marginals differ. This proves extra projected outputs could carry the missing information. It does not prove that a neural tracker is equivariant, that calibrated uncertainty is available, or that four marginals uniquely identify every joint distribution.

## Equal-compute comparisons and a bounded pilot decision

A meaningful pilot would freeze one small development-only cohort, the transformation, its scale-matched rendering control, all readouts and one attempt before inspecting new predictions. Metadata-audited RGB-Stacking reserve groups with no current visible inference record could supply this cohort; permanently designate them as development at admission, keep other groups untouched, and never relabel already exposed evaluation groups. The candidate reference is the 0-degree shrink view paired with the same-scale 45-degree view; native full resolution is a separate comparator, not the source of a scale-mismatched marginal claim. Reuse inference outputs across all decoding comparisons; never repeat forwards for a preferred score.

1. Native full-resolution tracker and actual shipped truncation/visibility rule.
2. Original plus oblique view: average inverse-warped native coordinate outputs; select the sharper view by a frozen native-entropy rule; unconstrained joint product-of-experts; additive mixture of inverse-warped product maps. These use the same logits and encoder calls as the candidate. Both additive and multiplicative map fusion matter.
3. Proposed fixed-marginal coupling, with exactly the same observation and query budget.
4. A same-scale unrotated rendering to identify resolution/padding effects, and an axis-preserving view as the coupling negative control. The latter tests the algebra, not an expectation of zero total TTA benefit.

Recommended primary quality endpoint is the standard average fraction within 1, 2, 4, 8 and 16 pixels, compared with the strongest predeclared non-oracle control. Report coordinate corrections and harms, full-coverage error, EPE and per-sequence outcomes. Use unchanged per-view recurrent states: all fusion is readout-only, with no feedback to future inference. Keep visibility fixed across competing decoders when testing coordinate quality; no abstention can masquerade as recovery. A positive preliminary signal requires improvement over the strongest same-logit comparator, not just the native single-view output. If a better rotated single view fully explains the gain, the coupling hypothesis fails. If the true target falls outside useful support of a native marginal, preserving that marginal can prevent recovery; this is a designed limitation rather than a reason to relax constraints after scoring.

The main feasibility risk is representation error: oblique resampling may make p_u/p_v worse or inconsistent with native marginals. No independent cameras are added, so no new physical viewpoint, occluded surface, stereo depth or epistemically independent sensor observation appears. The extra information is only what the fixed network reveals under another coordinate parameterization. Calling it triangulation or a calibrated posterior would overstate the evidence.

Admission is for a bounded developmental pilot, not a positive manuscript claim. First freeze and review the source, implementation, exact canvas/coordinate transforms, outside-canvas likelihood convention, positivity floor, optimizer tolerance, solver hard cap, ties and readout, then run meaningful analytic fixtures. A cheap analytic implementation test can precede neural inference. A tiny model pilot need not settle all prior art, but it must answer whether the claimed pairing mechanism has headroom in this model. Broad generalization requires later independent scenes and cannot be established by transformed copies of exposed RGB-Stacking videos.

## Search and access ledger

Exactly four additional primary-domain queries:

1. `site.arxiv.org SimCC SimDR independent coordinate classification x y heatmap rotated test time augmentation joint distribution`
2. `site.openaccess.thecvf.com coordinate classification rotated heatmap fusion test time augmentation joint x y pose estimation`
3. `site.arxiv.org pose estimation coordinate representation "projection" "rotation" "distribution" SimCC`
4. `site.arxiv.org "heatmap" "tomographic" OR "Radon" "pose"`

Source opening followed links/known primary IDs, without further search queries. Initial guessed HTML versions 2107.03332v4 and 1910.06278v3 returned 404; the metadata pages then established the real versions v3 and v1, whose HTML was read successfully. No claim relies on the failed versions. SimCC source is mutable main; freeze a commit if adopting its actual implementation later. No independent-camera fusion paper was fully read in this bounded pass; comparison to that broad category is a task-definition distinction rather than an exhaustive prior claim.
