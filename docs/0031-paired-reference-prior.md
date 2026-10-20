# Paired initial-reference matching: nearest priors and exact reduction

2026-10-20. Source/math only. Inputs to this review were the existing `0030-memory-content/output-repair-prior.md` and `paired-boundary-question.md`; no scientific arrays, masks, model calls or results were inspected. Four primary works were read, not a survey.

**Assessment.** The proposed score is ordinary nearest-neighbor matching on a concatenated two-point descriptor. Its possible contribution is the empirical value of retaining initial foreground/background *co-occurrence*, rather than a new matching operation. The inspected papers establish strong matching and figure/ground precedents but do not establish an identical initial-boundary-pair implementation. It would be inaccurate either to claim a new matcher or to call the whole proposed SAM2 experiment a faithful reproduction of one of these papers.

## Exact reduction and what remains testable

For separately unit-normalized descriptors, let `z=[x;y]`, `p_k=[u_k;v_k]`, and `n_k=[v_k;u_k]`. The proposed quantities satisfy

```
P+ = max_k z · p_k = 2 - 0.5 min_k ||z-p_k||²
P- = max_k z · n_k = 2 - 0.5 min_k ||z-n_k||²
S  = P+ - P-      = 0.5 (min_k ||z-n_k||² - min_k ||z-p_k||²).
```

Thus the primitive is two-class exemplar nearest-neighbor classification with a signed distance margin. The two orientations may select different exemplars. It is not a learned metric, a new conditional probability, or a maximum over per-pair signed differences.

Independent-max matching instead uses all combinations `[u_k;v_j]`. Restricting this Cartesian-product bank to observed pairs can reject a conjunction whose two individually plausible sides never occurred together. That association is additional information relative to descriptor marginals, but not additional sensory input relative to the full annotated initial frame. It can also discard legitimate combinations after background replacement or deformation. Whole-pair permutation is exactly irrelevant; outside-only permutation changes the associations without changing either marginal. These are algebraic observations, not measured utility.

## Four primary comparisons

| Primary source and actual method read | Closest overlap and precise difference |
|---|---|
| Hu, Huang and Schwing, **VideoMatch: Matching based Video Object Segmentation**, ECCV 2018. [Author manuscript v1, §§3.1–3.3](https://arxiv.org/html/1809.01123v1). | Builds first-frame foreground/background feature banks; computes cosine similarities and averages the top K separately for each bank, then obtains foreground probabilities. It also updates banks online. This directly precedes reference-conditioned FG/BG matching, with a richer exemplar bank than a mean prototype. Its displayed equations do not preserve one foreground/background boundary-pair index. Changing to frozen SAM2 features and K=1 is an adaptation of this matching family, not an exact reproduction of its trained pipeline. |
| Voigtlaender et al., **FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation**, CVPR 2019. [Author manuscript, §3, Eqs. 1–4 and Dynamic Segmentation Head](https://arxiv.org/html/1902.09513). | Uses nearest-neighbor distance to each first-frame object bank, treating background as another object, and local previous-frame matching. A segmentation head combines these distance maps with backbone features and previous predictions. Therefore reference matching used as a soft repair cue is already established. Its per-pixel classwise lookup differs from the two-point bank, but the latter remains the same nearest-neighbor primitive in a larger descriptor space. |
| Lukežič, Matas and Kristan, **D3S—A Discriminative Single Shot Segmentation Tracker**, CVPR 2020. [Author manuscript v2, §§3.1–3.3, Eqs. 1–2](https://arxiv.org/html/1911.08862v2). | Initial target features and nearby background features form separate banks; normalized similarities and top-K averaging yield similarity/posterior maps. These are combined with a discriminative localization pathway and multiscale refinement. Hence “local background examples plus target features plus a refinement head” is also occupied. Its invariant pathway intentionally discards spatial relationships; the inspected equations do not bind the two banks through an initial boundary-pair index. |
| Maire, Narihira and Yu, **Affinity CNN: Learning Pixel-Centric Pairwise Relations for Figure/Ground Embedding**, CVPR 2016. [Final CVF PDF, §2, Fig. 3 and Eqs. 7–13](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Maire_Affinity_CNN_Learning_CVPR_2016_paper.pdf). | Explicitly separates boundary existence from figure/ground ordering conditional on a boundary, and integrates pairwise relations into consistent grouping/layering. This is a direct precedent for directed pair relations and a warning against treating orientation as boundary existence. It predicts generic image relations with a trained CNN and global inference, not nearest-neighbor retrieval of initial target-boundary exemplars. |

**Terminology correction.** Here positive means “target side versus non-target side.” It does not necessarily mean “front surface versus occluded surface”: the target can itself be behind an occluder. Initial segmentation masks do not supply physical occlusion-order labels. Call the diagnostic **target-mask transition orientation**, rather than claiming physical boundary ownership. A positive orientation margin also does not establish that the current pair straddles the target; both endpoints can be background.

## Implication for the already proposed small test

The independent-max marginal baseline is essential; mean prototypes alone are too weak to isolate co-occurrence. Actual pairing and outside-only derangement must use the same endpoints, bank size, candidate crossings and tie rules. Concatenated-descriptor nearest neighbor is an algebraically identical implementation check, not an additional scientific competitor. Established top-K exemplar matching is a meaningful later robustness control if a complete refiner is pursued; this note does not add a tuning grid to the current bounded assay.

If actual pairing helps over both independent-max marginals and the fixed derangement on the complete fixed TRAIN cohort, the supported conclusion is only that this reference association carries useful ranking information in those frozen features. Whole-mask consistency, causal J/F improvement and novelty remain separate. A later head must still beat same-capacity current-only and unpaired-reference heads. If pairing adds nothing, relabeling the same score as conditional matching, boundary ownership or a prototype module would not rescue the claim.

The current work is best described as a **diagnostic adaptation of known exemplar matching with a specific reference-bank constraint**. No inspected source proves that exact constraint absent from the literature; no evidence here justifies a new-method claim. Equally, reduction to a known primitive does not by itself invalidate an experimentally useful composition.

## Read limits and access accounting

The VideoMatch and FEELVOS CVF PDF requests returned 403; their author HTML method sections were successfully read, without asserting byte identity to final proceedings. D3S HTML reports v2 dated 14 April 2020. The Affinity CNN final PDF text was read through §2; missing extracted later pages mean §3 implementation was **not** audited. Conference identities were checked against the respective official CVF records. No source code, model, dataset or standalone PDF payload was downloaded to disk. Four targeted discovery queries plus same-document method lookups were used; web-reader response bytes are unmetered. No exhaustive current-literature or code-equivalence claim is made.
