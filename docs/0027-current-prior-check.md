# 0027 bounded current primary-source check

Read-only check, 2026-10-19 local date. Three search queries, followed by primary arXiv HTML/abstract reading. No paper/model/data download, solver call, test, cohort or result inspection. This supplements, rather than replaces, the detailed ECCV 2020/ECCV 2024/MADPose review already recorded in `affine-tangent-prior.md`.

**Finding:** this search found no 2025–2026 primary paper explicitly solving the exact combination of known intrinsics, one affine correspondence, two axial depth gradients, two independent additive depth shifts and a relative depth scale. That is a bounded retrieval observation, **not a novelty or priority proof**. The strongest confirmed overlap is more specific than merely “both use depth”: RePoseD already uses the same three-equation, six-monomial, quartic calibration elimination.

## Exact search scope

Queries were:

1. `"affine correspondence" "depth" "shift" 2025 2026`
2. `"one affine correspondence" "monocular" 2025 2026`
3. `"relative pose" "depth gradients" "shift"`

Results largely returned the already known 2020/2024 methods and MADPose. Search-engine crawl/publication-age labels were not treated as paper publication dates. Nonprimary aggregation and generated hypothesis pages were not used as evidence of a method's existence or coverage. No broader claim about all 2026 literature is warranted by these three queries.

## Primary sources actually checked

| Source and version/status | Reading depth this pass; exact coverage |
|---|---|
| Ding et al., **RePoseD: Efficient Relative Pose Estimation With Known Depth Information**, ICCV 2025; [arXiv 2501.07742v3](https://arxiv.org/html/2501.07742v3), revised 3 April 2025; [version metadata](https://arxiv.org/abs/2501.07742) | Read abstract, related work, §3 and §3.1, Eqs.1–5 and the elimination following Eq.5. The calibrated model has two shifts and relative scale, estimated from **three point correspondences with depths**. It eliminates translation by differences and rotation by preserved lengths. Its 3×6 monomial system expresses three dependent monomials as quadratics in one shift; their product identity gives a quartic. These inspected equations do not take an affine map or depth gradients as inputs. This is direct algebraic precedent for 0027, not an exact one-AC input solver. |
| Yu et al., **Relative Pose Estimation through Affine Corrections of Monocular Depth Priors**, CVPR 2025 Highlight; [arXiv 2501.05446v3 metadata/abstract](https://arxiv.org/abs/2501.05446), revised 8 April 2025 | Rechecked current version and abstract only; method-level reading remains in the earlier note. The abstract explicitly covers independent affine depth ambiguities and calibrated/uncalibrated solvers plus a hybrid epipolar pipeline. This check supplies no evidence of a later derivative-input extension; it also does not establish that no such extension exists elsewhere. |
| Sun et al., **Learning Affine Correspondences by Integrating Geometric Constraints**, CVPR 2025; [arXiv 2504.04834v2](https://arxiv.org/html/2504.04834v2), revised 10 April 2025; [version/venue metadata](https://arxiv.org/abs/2504.04834) | Read abstract/introduction and §4.4 relative-pose setup. It learns affine correspondences from dense matches and geometry-based supervision. Its reported pose experiment uses essential-matrix estimation and affine GC-RANSAC. The inspected pose setup does not jointly solve two monocular-depth shifts and scale from one AC. It remains an important stronger affine frontend, not evidence that a local sparse-neighbor derivative is a free independent observation. |
| Barath, **Rolling Shutter Relative Pose Estimation Made Practical**, [arXiv 2606.26863v1](https://arxiv.org/html/2606.26863v1), submitted 25 June 2026; [version metadata](https://arxiv.org/abs/2606.26863) | Read abstract, §3 row-pose model and the solver portion of §4. It derives row-corrected affine epipolar constraints and a seven-AC solver for pose and readout motion, eliminating twelve rolling-shutter unknowns. No monocular depth or additive depth-calibration input appears in the inspected formulation. The arXiv metadata checked here does not declare a conference acceptance; describe it as a preprint. |

## Consequence for the proposed contribution

An accurate description remains **a candidate shift-aware differential initializer**, whose information is the measured affine map and depth derivatives. Do not claim first use of one AC with depth, differential lifting, rotation-free Gram constraints, a quartic, or the affine-depth calibration model.

Together with the recorded differential-limit derivation, RePoseD's current equations make the AC-to-three-point baseline especially strong. A result must distinguish the exact tangent construction from finite-step conversion, ordinary physical triples, and later full-nine-point denoising at honest input and compute cost. The current search neither vetoes that empirical possibility nor establishes it. No additional implementation or experiment is admitted by this note.

Web-tool transfer sizes are not exposed and were not invented or added as zero-byte receipts. No payload was saved by this check.
