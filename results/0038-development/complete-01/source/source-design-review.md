# Single-write late merging: a bounded hypothesis, not a new-method claim

2026-10-20. Private source/design review only. No implementation, model import, scientific array access, experiment, new allocation or admission. D35–D37's four VOST TRAIN clips are already exposed development; their recording independence is unresolved. This note does not reopen the failed D37 recipe.

**Assessment.** A well-defined operator remains: evaluate complete alternative-conditioned memory-attention stacks for ONE recent write, then merge their outputs before one decoder. This is a testable choice of merge location. It is not a calibrated Bayesian marginal, a correlation-preserving state, or presently an established innovation. The principal reason to test it would be a substantial advantage over early merging, ordinary output ensembling and a comparable-cost multi-path tracker—not merely a difference between two attention formulas.

## 1. Exact object and statistical scope

Let a single common causal parent at time \(s\) generate the native candidate masks \(m_{sj}\), tokens \(u_{sj}\), quality estimates and object-presence logit. Define each complete candidate write

\[
 W_{sj}=(\operatorname{MemEnc}(I_s,m_{sj}),\;
             \operatorname{Ptr}(u_{sj},o_s),\;\text{native positional metadata}).
\]

The candidates are decoder alternatives, not samples from an identified posterior. They may describe nested granularity, complementary parts, the same object, or a distractor. In particular, D35's union benefit does not support treating them as mutually exclusive true material states. An index \(J\) with weights \(\pi_j\) is an **algorithmic mixture over decoder alternatives**. The minimal proposal uses uniform weights over the returned native candidate slots; no pIoU calibration, outcome-selected temperature, uncertainty threshold or learned router is assumed. Duplicate slots are retained and their multiplicity disclosed.

For current image features \(Q_t\), let \(C_t\) contain all normally selected history except write \(s=t-1\), including the other object pointers. Let \(\Phi\) be the entire unchanged four-layer memory-attention stack, including each self-attention, cross-attention, residual, normalization and FFN, and its final normalization. Let \(D\) be the native decoder with its current-image high-resolution inputs:

\[
 Z_{tj}=\Phi(Q_t,C_t\cup W_{s j}),\qquad
 Z_t^{late}=\sum_j\pi_j Z_{tj},\qquad
 \widehat m_t=D(Z_t^{late}).
\]

The proposed merge is **after the complete stack**, not independently after every cross-attention sublayer. Each branch therefore keeps one candidate assignment through the stack. Once the mean is formed, branch variance and joint dependence are discarded. In general \(D(\sum\pi Z_j)\ne\sum\pi D(Z_j)\). Averaging post-normalized features can itself create an off-manifold input to the pretrained decoder. No accuracy monotonicity follows.

This narrowly tests: *does delaying the merge until after nonlinear memory conditioning improve complete causal tracking enough to justify its cost?* It does not test whether a posterior or physical identity distribution has been recovered.

## 2. What is and is not different from token pooling

For one attention head and one fixed query, write the common-history unnormalized numerator/partition as \(N_C,Z_C\), and candidate \(j\)'s as \(N_j,Z_j\). Separate complete-history attention gives

\[
 r_j=\frac{N_C+N_j}{Z_C+Z_j},\qquad
 r_{\rm mix}=\sum_j\pi_j r_j.
\]

A fair flat-token control includes the common history ONCE and adds \(\log\pi_j\) to candidate-\(j\) token logits. It gives

\[
 r_{\rm flat}=\frac{N_C+\sum_j\pi_jN_j}{Z_C+\sum_j\pi_jZ_j}
             =\sum_j\beta_jr_j,\quad
 \beta_j=\frac{\pi_j(Z_C+Z_j)}{Z_C+\sum_l\pi_lZ_l}.
\]

Thus the distinction is removal of partition-dependent branch reweighting, not access to new evidence. Flat pooling without the prior-mass correction is an unfair weak control: duplicating an identical candidate slot \(K\) times increases its mass relative to common history. With common value 0, candidate value 1, and all logits 0, native/separate mixing returns \(1/2\), whereas three unweighted candidate copies return \(3/4\).

The displayed identity concerns one attention calculation with a common query. It does not identify complete-stack flat pooling with complete-stack branch mixing: later branch queries diverge. Replacing every CA output by a mean before its FFN is a third, layerwise approximation and must be named separately. Likewise, attending to candidate-only tokens and adding a separately normalized common-history output is incorrect for the stated operator.

## 3. Pointer compatibility and a complete causal policy

The installed tiny configuration enables candidate-specific output tokens and an MLP pointer projection. Native code selects the mask token by pIoU and applies its object-presence/no-object rule to the projected pointer. An alternative spatial memory must therefore travel with the corresponding projected, presence-handled pointer. Reusing the winner's pointer for every candidate would be an explicit spatial-only ablation, not a coherent full-write branch. The same applies to spatial no-object embeddings, BF16 storage, actual temporal distances, pointer-token splitting and exclusion from spatial RoPE. Do not average tokens before the nonlinear pointer MLP and call that an average of pointers.

A fully specified minimal deployment policy is:

1. Initial supplied memory remains unchanged. At each later frame, retain the actual native candidate bank generated by that arm's single common decoder state.
2. Only the immediately preceding write is branched for the next frame's read. All older spatial memories and pointers use the canonical candidate chosen at their creation by the unchanged native pIoU rule. No retroactive selection by current or future labels occurs.
3. Run the complete stack separately for each recent candidate, merge once, and run the native decoder once. Its output defines the current prediction and next candidate bank. All branches use identical frame indices and native memory-selection rules; no backfill or extra conditioning masks.
4. If native presence suppresses all candidates, retain that native no-object behavior. Do not invent a visible branch, switch to union after inspecting an error, or omit the frame.

This is a one-write approximation with immediate historical collapse. At \(t+1\), histories are not the Cartesian product of alternatives at \(t-1,t\). Exact joint marginalization over several uncertain writes would generally require their joint assignment and branch-specific downstream states; it is not obtained by independently averaging their reads. Maintaining coherent paths across time moves toward SAM2Long. Shared parent state and cached image encoding are implementation reuse, not an additional mechanism.

D35 saved logits alone cannot support this experiment: they lack the candidate mask tokens, full memory encodings and conditioned image features. New native causal inference would be needed, with all costs charged.

## 4. Minimum useful comparison and cost interpretation

For a merge-location comparison, freeze the same incoming history, candidate masks/tokens, positional metadata, weights and current image features. This isolates the operator. Such matched-context differences are diagnostic only: their sum cannot establish the effect of recurrent deployment. The decisive comparison must use complete causal rollouts, where every arm receives the same RGB/initial mask but owns its evolving state. Its internally generated candidates may then differ; that is a consequence of the policy, not unequal external observations.

The necessary controls are:

| Control | Required distinction |
|---|---|
| Native winner and fixed candidate union/soft-mask write | Known strong choices; define the synthetic-mask pointer explicitly. Using the native winner's pointer is a stated control convention, not a claim that the union has its own decoder token. |
| Early encoded-feature mean | Encode all candidates first, average stored spatial features and projected pointers, then use one native stack/decoder. Same alternatives and uniform weights. |
| Mass-corrected flat tokens | Common history once; candidate-token log-prior correction above; native pointer/spatial layout preserved. |
| Ordinary ambiguity-weighted early gate | Same bank and raw pIoUs, a fixed declared softmax weighting with no tuning; not called a reproduction of trained UAMP. It tests whether an ordinary quality gate suffices. |
| Late full-stack mean (candidate) | Three stack evaluations and one common decoder, as defined above. |
| Ordinary decoder-output ensemble | Same complete branches and weights, but merge native presence-handled output probabilities after branch decoders. This is a stronger, slightly more expensive conventional ensemble; any continuing state/pointer convention must be declared. |
| SAM2Long | Actual multi-path control with the same checkpoint and initial supervision; separately report a strictly causal prefix-output variant if official final-path selection revises past outputs. Never silently call that variant the unchanged official method. |

A layerwise separately normalized CA mixture is a useful mechanistic ablation only if the late-stack candidate has a meaningful signal; it need not create another prerequisite diagnostic.

One image encoding may be shared. Count every candidate memory encoding, pointer projection, attention stack, decoder, retained state, feature copy and output operation. Three branches do not establish equal wall cost: late merging uses three stacks/one decoder; a three-path tree may use more decoding/encoding and different history management. Compare full method quality/cost curves or an explicitly fixed complete-cost envelope, not just branch counts. Report the stronger ensemble even if it costs slightly more; report its actual cost. If published methods cannot be reproduced within available resources, record the unresolved comparison rather than asserting superiority to them.

No source-only analysis proves that all these arms fit CPU4/8 GiB/1800 seconds. The previous native runtime is not a bound on three repeated stacks plus all controls. Any admitted plan must retain complete-video operations and account for shared and arm-specific costs; it must not meet the limit by reporting successful clips only.

## 5. Sharp proposed continuation, without a new oracle ladder

There is no need for an additional GT-assisted opportunity ceiling. If this operator is selected for development, use the complete four exposed TRAIN clips and all their scored positions, not only disagreement/occlusion frames. Keep the pinned author-compatible J/Jtr primary policy and separate void-ignore sensitivity. This is development, with no held-out or recording-independent inference.

A proposed threshold high enough to justify another research step is: **at least +3.0 absolute J AND +3.0 Jtr points over the strongest complete, same-observation comparator in each metric**, including the best simple merge/ensemble and the qualified comparable-cost SAM2Long control; positive Jtr change on at least 3/4 clips; and no clip J or Jtr more than 0.5 point below native. Require full finite coverage and the declared resource bound. This is a proposal for a future scope, not an adopted amendment or a result. A small average benefit, a win only against uncorrected flat concatenation, or a better loss/feature distance does not pass it. Failure would reject this exact one-write uniform late-merge recipe, not prove that all ensemble or multi-state tracking is impossible.

The final research gate remains unchanged: +3 official J&F points on EACH of two independent held-out sources, positive paired recording-level 95% intervals, neither J nor F down more than 0.5 point, strong same-information controls at fixed common complete cost, ablations, an actual innovation distinction and independent replication. VOST J/Jtr and four exposed clips cannot substitute for it.

## 6. Closest primary precedents and actual reading scope

- **SAM2Long**, §§3.1–3.3: complete spatial-memory/pointer pathways, cumulative quality scores and pruning. It preserves alternative histories where this proposal collapses them. Read the primary HTML methods, not its implementation in this task. [ICCV paper / arXiv v3](https://arxiv.org/html/2410.16268v3).
- **UAMP**, §3.3 prose: uncertainty-weighted fusion of memory readout and previous features plus memory selection/updating already covers broad uncertainty-aware propagation. This note does not claim exact equivalence; displayed equation images were not fully transcribed. [PLOS One primary article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353156).
- **Competitive Memory Readout**, §3.2 Eqs1–3: target/competitor log-sum-exp scores gate target logits before normalization. These are distractor competitors, not full candidate-write branches. It still prevents presenting “query-dependent memory competition” as an unoccupied concept. Read the primary technical-report methods. [CMR](https://arxiv.org/html/2608.22064v1).
- **MA-SAM2**, §2.2: context-aware candidate-history pruning and occlusion-resilient memory already use alternative masks and accumulated branch scores. Its described selection differs from averaging independently conditioned outputs. Read official PDF method pp3–6; no implementation reproduction. [MICCAI2025 paper](https://papers.miccai.org/miccai-2025/paper/2634_paper.pdf).
- **BAM / Mixture of Attention**, §3.4 and §4.1: weighted sums of separately evaluated attention experts, including all-expert soft routing, are established operators. Different SAM memory data with shared parameters is a narrower application distinction, not a new mixture primitive. Primary PDF checked after the peer's source pointer. [NeurIPS2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/665bb142d4b9f55660cb89bb56a66fe1-Paper-Conference.pdf).
- **Training-Free-VOS via Streaming-data Mixture Models**: the official abstract already describes a dynamic ensemble of spatially/temporally coherent feature mixtures. Only the abstract was read here; the PDF link fetch failed, so no exact attention-equivalence claim is made. [NeurIPS2023 primary abstract](https://papers.neurips.cc/paper_files/paper/2023/hash/240cc9ac4789351653d13cfcba4ee85c-Abstract-Conference.html).

These bounded checks find close operations and application precedents, not a universal novelty proof or a proof of duplication. The remaining research question is empirical advantage of a narrowly specified merge location. **Recommendation: keep it as a defensible but low-confidence development hypothesis; do not label it “correlation-preserving memory,” claim innovation, or infer a GO from algebra alone.**

Local source read, without import: `sam2/modeling/memory_attention.py` lines55–99,147–165 (SHA256 `07358cb7f58ec3788e88ff4e6415f8a9d628a3963529d6e2c915494a5347c8e2`); `sam2_base.py` lines339–403,587–648 (`69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41`); `sam/mask_decoder.py` lines136–166 (`ca3523c58365574faddf1bfb54f374e4b4beba05127185ea2b17d22b916e099a`); `configs/sam2.1/sam2.1_hiera_t.yaml` (`f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d`). All are under the existing qualified installation; no source bytes were edited.
