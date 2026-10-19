# 0023 saved-result mechanism diagnosis: closed

The frozen directional-instability hypothesis failed all four development gates. This is an eligible negative result, not an incomplete evaluation or a paper-quality improvement. There will be no efficacy retuning on these 24 queries.

Scope: saved `prediction-02` scene records, saved `evaluation-02` errors/metrics, and operation receipts only. No fitting, model/native calls, images, NPZ inputs, HTTP, or new GT reads were performed. Raw `ground_truth` subobjects in the saved evaluation were skipped lexically before JSON decoding. The private compact receipt is `work/0023-preparation/mechanism-diagnosis-02.json`; it records input hashes and all scene/arm aggregates. Prediction/evaluation freeze: `cb520839515381db125f64ea3bcceb434444a85b`; the original acquisition/frontend remain at `acf502b831ad9e3485ab56549217c8a603d56bf7` under the disclosed arithmetic-order repair/reuse protocol.

## Outcome against the prespecified controls

All 24 queries remain in every metric, with directed translation and max(rotation, translation) error. Eight results per multiview arm are native fallbacks. There are no excluded failed places or queries.

| Method | AUC@5 (%) | AUC@20 (%) |
|---|---:|---:|
| Geometry multiview | 61.9844 | 83.6550 |
| Joint field | 57.8633 | 82.6247 |
| Residual scalar | 56.6424 | 82.3195 |
| Hessian | 56.5491 | 82.2962 |
| **Directional** | **56.4971** | **82.2832** |
| Scalar deletion | 56.0082 | 82.1609 |
| Native E, reference-filtered query | 52.1229 | 78.0646 |

Directional loses 5.4874 AUC@5 points against the best control, rather than gaining the required 2 points. Its margins against the best control in s00475/s00478/s00513 are −9.6662/−4.8780/0 points: zero positive places, and two violations of the −1 point scene floor. Its AUC@20 deficit is 1.3718 points, exceeding the permitted 0.5. All four gate components therefore fail.

The 4.3742 point improvement over native E alone is insufficient: the prespecified geometry control gains more with the same reference/query population. Directional is also below Hessian (−0.0520), joint field (−1.3662), and residual scalar (−0.1453). Its +0.4889 over scalar deletion does not establish the required contribution. Against geometry it improves 2 queries, worsens 14, and ties the 8 common fallbacks; these counts are descriptive, not independent-sample inference.

## Coverage and what the fallback does—and does not—explain

In s00513, full reference initialization fails with the saved reason `PnP insufficient inliers: B1`. Seven of eight deletion initializations fail for the same reason; deleting C3 succeeds. The complete reference and geometric-reference paths are unavailable, so all eight queries in every multiview arm fall back to exactly the same native-E reference-filtered-query result. Native calls themselves return successfully; an initialization inlier gate fails. This is not evidence of missing depth being the cause merely because the downstream fallback message also mentions depth-scale availability.

The common fallback explains limited method coverage (16/24 queries execute the update), but cannot explain directional's relative deficit: the common eight results tie, while both active places lose against geometry. The successful C3 deletion demonstrates sensitivity of this particular initializer/graph pipeline; it does not identify the physical cause or authorize selecting that fold as a replacement.

## Solver state and inactive observations

All 19 successful field-reference fits (2 complete references and 17 deletion references) hit the frozen 30 accepted-step limit. Neither of the two geometric-reference fits converges. Every geometry and joint-field query start also hits that limit. Finite last accepted iterates were admissible under the protocol and remain included.

| Query arm | Starts attempted | Converged starts | Converged selected results / 16 | Recorded query-detail seconds |
|---|---:|---:|---:|---:|
| Geometry multiview | 32 | 0 | 0 | 180.906 |
| Joint field | 32 | 0 | 0 | 60.164 |
| Hessian | 32 | 24 | 13 | 1.038 |
| Residual scalar | 32 | 24 | 13 | 1.060 |
| Scalar deletion | 32 | 9 | 5 | 1.586 |
| Directional | 32 | 20 | 10 | 1.413 |

Across references and queries, 213 solver attempts make 5,758 objective evaluations and 5,454 accepted steps; 136 terminate at the accepted-step limit and 77 satisfy `ftol`. Thus deletion variability includes effects of bounded optimization, graph deletion and reinitialization. These records do not isolate statistical depth uncertainty. Nonconvergence is a limitation, not an excuse to relax the gate: geometry remains the strongest result despite its own limit, and no counterfactual establishes that more iterations would rescue directional.

In the complete field references, finite capped / invalid / total projection-block counts are 1,106 / 149 / 7,856 (s00475) and 371 / 7 / 8,038 (s00478). The corresponding geometric references have 88 / 43 / 5,735 and 24 / 18 / 5,557 observations. For selected joint-field query fits, inactive fractions range 13.11–15.15% and 4.28–4.83%; selected geometric fits range 2.26–2.50% and 0.69–0.76%. Field edge-projection blocks and geometric track observations are different populations, so these fractions are not a paired causal comparison. Finite capped and invalid counts are disjoint; both retain full cap cost and zero IRLS information. Conditional-arm per-block counts were not saved and were not reconstructed.

Every active reference marginal has numerical rank 17, the shape projector has rank 8, and all returned active-arm observability tests pass. Scalar deletion inflation ranges 154.3–2,656.6 in s00475 and 7.26–82.68 in s00478; residual inflation is respectively 1.3727 and 1.0. These show that deletion-based inflation was present, not that it was calibrated or aligned with actual pose error. Directional versus Hessian changes AUC@5 by +0.9723 and −1.3443 in the two active places, with no overall benefit.

## Cost and limits of interpretation

There are 432 actual native calls (51 relative, 381 absolute), all returned, totaling 0.1774 seconds inside the enclosing timers. Full/deletion field-reference processing costs 32.658 seconds, of which deletion passes cost 28.662 seconds. The additional geometric-reference solves cost 11.597 seconds. All three scenes together take 293.850 seconds. These nested timings must not be added to their parent timers or counted twice.

Supervised frontend and eligible prediction wall times are 103.603 and 298.587 seconds: 402.190 seconds combined, with sampled peaks of 2,142,832 and 1,173,520 KiB. Acquisition and evaluation cost another 75.975 and 7.047 seconds; the preserved pre-fit technical failure `prediction-01` separately costs 1.410 seconds. The experiment stays within the resource limits. Query-detail timers include both starts, but omit shared reference construction and some covariance/validation work. They cannot establish a standalone or matched-wall-time speed advantage. In particular, directional's small conditional-solve time does not make its eight deletion fits free, and those fits do not produce an overall improvement over Hessian.

## Claims closed by this result

The tested combination does not deliver the prespecified useful pose-accuracy improvement, cross-place consistency, scene-loss protection, or AUC@20 protection. It also provides no evidence that directional deletion inflation improves upon the ordinary Hessian control in aggregate. The records do **not** prove that all spatial depth correction is impossible, that invalid blocks caused the deficit, that depth alone caused the geometry gap, or that a differently converged optimizer would succeed. No matched interventions establish those causal statements. This three-place development graph is neither a held-out result nor an official Map-free benchmark track, and supplies no CVPR-result or SOTA claim.
