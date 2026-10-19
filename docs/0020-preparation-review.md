# Independent preparation review: 0020

Reviewed 2026-10-19, before any 0020 scientific input read or scoring call. **No remaining mathematical or implementation blocker was found for the single bounded development diagnostic**, subject to the published source freeze and its passing exact-commit CI. This is permission to test a falsifiable fixed formulation, not evidence of efficacy or novelty. No HDF5, images, native estimator or new scientific outputs were accessed during this review.

## Mathematical checks

The four-parameter design is the correct axial-depth consistency equation: `R(c1 h1) + t_unit = c2 h2`. Unit translation removes the otherwise arbitrary metric scale. If raw depth is `d_j = a_j Z_j + b_j` with positive `a_j` and the true translation has length `T`, an exact correct-pose solution is `alpha_j = sigma_j/(a_j T)` and `beta_j = (mu_j-b_j)/(a_j T)`. Generic rank four identifies this calibration; rank-deficient data do not. Independent positive affine changes to either input depth preserve normalized depths mathematically. Full-input normalization therefore changes coordinates, without making the candidate pose statistically independent of the validation points.

The implementation forms the declared weighted 3-by-4 blocks and computes `V (U^T b / singular_values)` with the strict declared rank threshold. It enforces positive slopes without silently switching to a constrained optimizer. Both projection directions use the same fitted depths and unit gauge. Nonpositive source depths, invalid transformed points and invalid projections retain their full capped directional cost. Failed fits preserve their entire validation mass. The 2px geometric component ports the frozen 0017 operation order and verifies the sealed aggregate before fitting. No native mask or label enters calibration or candidate choice.

The spatial/random comparison controls nominal fold sizes and the spatial weighting measure. After intersection with candidate-dependent geometric support, actual training counts, masses and rank failures can still differ. The final protocol explicitly records this confound; a score gain alone cannot identify regional transfer as its cause. Near-rank-deficient fits can also produce sensitive coefficients despite passing a machine-precision rank test. Their singular values and failures remain visible; this limitation does not justify changing the predeclared threshold after results. Empty partitions and all-failed calibration preserve coverage. The protocol now accurately distinguishes real-arithmetic equivalence of `G+1` to `G` from possible floating-point ties.

## Implementation and comparison checks

The complete original four-pose bank is verified before allowed input access. The scoring worker makes no new pose estimate. All five selectors use this same bank. The two stronger 0019 author-loss arms are fixed benchmark outcomes only; neither can become a new selected candidate. The evaluator retains all 80 pairs, all five scenes and all ten comparators. It chooses one aggregate AUC@5 comparator with the declared tie order, uses that same comparator for scene comparisons, and separately uses the best comparator AUC@20 for the tail-loss guard. The 2-point / two-scene / 0.5-point conditions match the protocol.

The full score seal and completed outer prediction receipt are required before saved labels are decoded and are checked again afterward. Worker and cohort resource receipts are mandatory. Candidate generation charges all four original candidates; observed historical costs cannot establish matched-compute efficacy. Mathematical unavailability keeps full cost, whereas a technical failure blocks the cohort.

Two preparation defects were corrected before freezing: generic `ValueError` was removed from mathematical-unavailability catches so unexpected implementation faults propagate; saved-output validation now reconstructs `G`, each `G+D` score and all fit counters from the required fit records. Exact global / four-fold coverage and empty-partition placeholders are checked. These changes do not change the scoring algorithm, candidates or thresholds.

Independent execution passed all **13 manufactured core tests**, including exact affine recovery against a direct reference, signed/physical cases, invariance, rank failure, fixed denominators, input/sealed-score barriers and unexpected-error propagation. After the output-validation fix, the reviewer separately ran its **three targeted artificial regressions**, all passing: mutually rewritten hybrid fields, missing/reordered fits or forged counters, and unavailable-partition placeholders. These are software checks, not research measurements. The coordinating agent separately records the final complete 33-test execution and its actual process evidence in [0020 preparation results](../results/0020-preparation/).

Reviewed bindings:

| Source | SHA256 |
|---|---|
| Protocol | `1217cc6ab435d2a15da8aaa4b6c7ee8999b5a3fe3fdfd931324eb0c98170b5e3` |
| score.py | `e875f663c93fcc69d1e8cb314df32f548e5f564a4be58d1eeb3caf85d86dd3e1` |
| common.py | `614d4f81bd28b54f4072bb7ca1d0dab6e36ece2932ceb28fe4ee858b1fcff03a` |
| predict_pair.py | `339c6070d2dd1e6304f185ac21b43c4d329fe63ca94957c396554b1c4d115109` |
| run_diagnostic.py | `10f05a43ded3433241f5c9be355d3f0b94c5d5cbc9fb70b4bcd0619978fb54a1` |
| evaluate_saved.py | `000b3207373eab4bf56663b06fc216fd5b3e54996fe52a0d754fed1e92e4137a` |
| run_operation.py | `272bc02a7d0387b6f60bab1657302567a47654497a77ad5eb84661c22ad4ff4a` |

The result must still beat the stronger known controls under the fixed gate. Even a pass would only admit subsequent development: official method comparisons, a support-matched mechanism test and independent scene/source evaluation remain necessary before claiming a contribution. A failure closes this fixed formulation without tuning on these 80 labels.
