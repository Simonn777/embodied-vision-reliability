# 0009 — fixed-endpoint permutation tails: bounded prior-art and null audit

Date: 2026-10-18. Status: **no direct vision duplicate established; novelty UNKNOWN; real-match significance claim NOT justified yet.** This is literature/mathematical review, not an experiment, protocol approval, or evidence of performance. The rejected moment-only z ranker is not revived.

## Decision

The narrower idea differs mathematically from the ORSA formulas inspected: condition on both observed coordinate sets and count the support of a *uniform random bijection*, rather than draw independent spatially uniform points. But exact rook/hit counting, component multiplication, and complete-board closed forms are classical. Calling their assembly a new inference method would overclaim. A potentially distinctive vision contribution would require a defensible conditional null for the actual matching pipeline, useful coverage under bounded computation, and better geometry/false-alarm behavior against established a-contrario baselines. None is established here.

**Reject now:** “exact calibrated significance after SIFT/descriptor matching,” “first geometric use of permutation tests,” or “new rook-polynomial algorithm.” **Not directly killed by the inspected sources:** fixed-coordinate, fixed-H upper-tail counting as a deliberately stipulated combinatorial null / hypothesis-ranking research question. A small artificial correctness test could assess implementation; it could not certify novelty, image-matching calibration, or practical benefit.

## Closest primary sources actually inspected

1. **Lionel Moisan, Pierre Moulon, Pascal Monasse — _Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers_, IPOL 2 (2012), 56–73; published 2012-05-19.** [Official article](https://www.ipol.im/pub/art/2012/mmm-oh/), [current manuscript](https://www.ipol.im/pub/art/2012/mmm-oh/article.pdf). Read abstract, §§1.1–1.3, equations (1)–(3), not the complete algorithm/experiments. The site records a 2022-01-01 editorial-information revision. §1.2 combines uniform spatial points and random coupling; §1.3 uses disk-area probability πr²/(w′h′). Its NFA multiplies a power of an independent correspondence probability by combinatorial test factors and discounts the four fitting correspondences. This is not an exact law over the observed endpoint permutation orbit. “Random coupling” alone must not be quoted as proving equivalence. Article header says CC BY-NC-SA; no source code was acquired or reused.

2. **Pierre Moulon, Pascal Monasse, Renaud Marlet — _Adaptive Structure from Motion with a contrario model estimations_, ACCV 2012 author manuscript.** [Author-hosted PDF](https://imagine.enpc.fr/~monasse/Callisto/pdf/ACCV2012.pdf). Read title/abstract and §4.1–4.2, equations (1)–(2), table and rigidity definitions; not full results. The null explicitly uses pairs of independent points uniformly distributed in their images. The generic NFA uses `(error^d × α0)^(k−Nsample)` and model/subset multiplicities; for homography α0=π/image-area. This directly rules out presenting accidental-support correction, model significance, or NFA-based hypothesis selection as new. It does not show the proposed fixed-marginal permutation distribution. Publication-year/venue are identified by this official ACCV manuscript; no implementation or license audit was performed.

3. **Nicolas Noury — _Mise en correspondance A Contrario de points d’intérêt sous contraintes géométrique et photométrique_, Université Henri Poincaré–Nancy 1 doctoral thesis, defended 2011-10-13.** [Official university PDF](https://docnum.univ-lorraine.fr/public/SCD_T_2011_0069_NOURY.pdf). Read selected §§4.1.2, 4.2.1–4.2.3, 4.3.3, 4.4 and bibliographic entries; not the full thesis. Its joint photometric/geometric null assumes mutually independent distances and uses `fD(δD)^k fG(δG)^(k−7)`. NFA counts possible endpoint subsets and `k!` matchings. §4.4 adapts this to homographies with disk-area geometry. Thus joint descriptor-aware significance and matching combinatorics are substantial prior art, not fresh concepts. Permutation matrices in its Softassign discussion describe matching optimization, not an exact permutation-tail test. No rook/hit-number computation was found in the read sections. Its “colored rigidity” attribution points back to Moisan–Stival 2004; that original paper was not separately method-read in this audit.

4. **Feryal Alayont, Nicholas Krzywonos — _Rook Polynomials in Three and Higher Dimensions_, author-hosted preprint marked “To appear in Involve.”** [Primary author PDF](https://faculty.gvsu.edu/alayontf/notes/rook_polynomials_higher_dimensions_preprint.pdf). Exact preprint date/version not stated in inspected pages; do not invent one. Read introduction and §1 two-dimensional foundations, not higher-dimensional results. This explicitly reviews restricted permutations, `r_k = choose(a,k) choose(b,k) k!` for complete boards, and multiplication for boards sharing neither rows nor columns. It attributes the underlying theory to Kaplansky–Riordan 1946. The supplied component strategy is therefore old mathematics, independently of any unknown vision application. No code reused.

## Counting identity and what components do — direct derivation

Let A be the fixed n×n binary compatibility board for fixed H and tolerance. For π uniform on S_n, let T=Σ_i A[i,π(i)]. With r_k the number of size-k matchings in A, counting the ways a permutation extends a chosen k-matching gives

`E[choose(T,k)] = r_k (n−k)! / n!`,

and hence

`E[z^T] = Σ_k r_k (n−k)! (z−1)^k / n!`.

Thus the integer count for T=t is

`h_t = Σ_{k=t}^n (−1)^(k−t) choose(k,t) r_k (n−k)!`,

with tail `Σ_{t≥t_obs} h_t / n!`. Equivalently the count-generating polynomial is `per(J+(z−1)A)`. These identities are elementary classical counting, not a novelty claim. Isolated vertices still contribute to the global n! space.

Multiply/convolve **rook coefficients** over disjoint row-and-column components, then apply the global factorial transform. Do **not** convolve component hit distributions as if independent: one global permutation can pair vertices across components. Integer arithmetic avoids cancellation-induced negative probabilities; require nonnegative integer h_t, sum h_t=n!, and agreement with tiny exhaustive permutations. Large difficult components must return explicit unsupported, with coverage reported for the entire prespecified cohort. Do not substitute a moment bound, normal tail, or Monte Carlo zero as an “exact p.”

## The scientific bottleneck is exchangeability, not factorial precision

The following are this audit’s deductions, not claims that the papers proved them.

- A valid conditional permutation p-value requires the observed pairing to be uniform over the stated orbit under its null, conditional on everything held fixed. Fixing coordinate marginals alone does not create that property. Descriptor neighborhoods depend on content and location; ratio tests, mutual nearest neighbors, candidate pruning, repeated texture, and assignment rules can make eligible pairs unequal or forbidden. A wrong geometric model does not imply descriptor-selected pairs are random bijections.
- Even if all selected endpoints are distinct, the matching-selection event can restrict the permutation orbit. Duplicated target indices violate the bijection model outright unless preprocessing defines the tested sample differently. Restricted/weighted matching nulls require their own justified sampling measure and counting scheme, not the same n! denominator.
- A concrete logical counterexample is a fixed descriptor-eligibility board allowing only one permutation. Conditional on that selection event its probability is one; a full-S_n tail can nevertheless be tiny. This shows that an exact combinatorial answer can target the wrong conditional experiment. It is not a measured failure on real images.
- Using disjoint proposal and verification pairs can remove direct minimal-sample support bias. It does not by itself fix descriptor-selection dependence. H and the tolerance must be fixed relative to the tested pairing, or the full selection must be included in the test. Searching many H/tolerances requires a justified multiplicity correction or valid selection-inclusive randomization. A fixed-H p-value is not automatically the selected-model p-value.
- Smaller tail probability is neither posterior probability that H is correct nor lower transfer error. Statistically unusual isolated support can be wrong, and a correct transformation on a repeated/ambiguous structure may be unsurprising under this null. Task accuracy and false-alarm calibration must be assessed separately.

## Bounded search and remaining uncertainty

Exactly six search queries were issued, then only primary-document opens/finds and one official-page/PDF retrieval:

1. `homography "permutation" "a contrario"`
2. `"ORSA" "fixed" "points" probability`
3. `"matching" "rook polynomial" "vision"`
4. `"permutation test" "correspondences" homography`
5. `"a contrario" "homography" "null hypothesis"`
6. `"rook polynomial" "permutation" distribution matching`

The scan did not establish a vision paper implementing exactly this fixed-endpoint rook-polynomial upper tail. It was not an exhaustive 2026 literature review and cannot establish absence. In particular, original Moisan–Stival 2004, other assignment/permanent statistical tests, and later conditional/randomization matching work remain incompletely checked. No downloaded dataset, model, package, image sample, real labels, or experiment was used; repository unchanged. New direct shell retrieval comprised the official IPOL HTML (20,261 bytes) and manuscript (3,983,987 bytes; SHA256 `7044f6d1eaf2bcf5f32c7e9f289fe5845bcf5d31221a2c1ccc5711431cf3c9d7`), parsed in memory; web-tool transport bytes are unavailable. These are literature reads, not data acquisition.
