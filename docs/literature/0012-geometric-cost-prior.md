# Geometry / driving literature admission review

Date: 2026-10-19. Scope: source reading, public code/metadata inspection, HTTP HEAD only for datasets. No model checkpoint or dataset body downloaded; no inference, fitting or outcome inspection performed. This is a candidate review, not a result or a novelty certificate.

## Decision

Prefer a bounded falsification of **moment-assisted certification of truncated geometric costs** over another confidence/distillation system. It is CPU-feasible with existing public correspondence caches, but the novelty is narrow and the useful task improvement is unproven. Do not admit the broader claim “geometric bounds make RANSAC exact and fast”: Space-Partitioning RANSAC already does conservative cell rejection with preserved accuracy. Generic multi-teacher tracker verification and lightweight coarse/fine uncertainty mixtures also have direct recent prior art.

The practical task is two-view camera relative pose, a component of SLAM and embodied navigation. MegaDepth is Internet-photo geometry, not an autonomous-driving dataset. Results there cannot establish driving or robot-control improvements.

## Candidate 1: provenance-conditioned geometric confidence / distillation

Proposed mechanism: distinguish many dense matches inherited from a shared wrong coarse assignment from independent evidence, then validate pose proposals on groups withheld from fitting. A small student could learn a downstream pose-utility target rather than copy teacher confidence.

Usable information would have to be *actual coarse assignment lineage or cross-view evidence*, not a spatial grid given a new name. A groupwise robust score alone does not create evidence for the correct pose when wrong correspondences coherently support another pose.

Closest primary prior art:

- [Track-On-R, CVPR 2026](https://arxiv.org/abs/2603.12217) already uses a learned verifier to select per-frame predictions among trackers and generates pseudo-labels for real-world adaptation. The [authors' project](https://kuis-ai.github.io/track_on_r/) describes local visual and temporal cues. Therefore multi-teacher selection plus distillation is directly occupied.
- [CoRe, arXiv 2608.08685, August 2026](https://arxiv.org/html/2608.08685v1), §§3.2–3.3 and Appendix B, uses a nine-parameter calibrated coarse/fine Laplace mixture and posterior-weighted geometric refitting. Generic post-hoc mixture confidence and geometry refit is occupied. Its per-match formulation does not itself establish a shared latent assignment model, but absence of that feature is not a priority proof.
- [VSAC, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ivashechkin_VSAC_Efficient_and_Accurate_Estimator_for_H_and_F_ICCV_2021_paper.pdf), §2, explicitly discounts dependent nearby inliers and evaluates independent support. GroupSAC and graph-cut RANSAC are further necessary comparisons.
- [Learning Correspondence Uncertainty via Differentiable Nonlinear Least Squares, CVPR 2023](https://arxiv.org/abs/2305.09527) already learns feature-position covariances through camera-pose optimization. This rules out calling “pose-aware uncertainty” new by itself.
- [TAPNext++, CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Jung_TAPNext_Whats_Next_for_Tracking_Any_Point_TAP_CVPRF_2026_paper.html) addresses long sequences and redetection. Its venue must be labeled Findings.

Admission verdict: **do not implement now**. DMS caches have no demonstrated coarse-lineage fields, and a source-code read cannot prove the HDF5 has them. Their loader uses coordinates, camera calibration, poses, names and optional confidence. Producing genuine lineage for this hypothesis requires a new matcher extraction operation and independent development images. Spatial clustering alone is not the stated information source. The old 0009 exact-permutation-tail candidate remains stopped; a different score over the same stipulated synthetic null would not answer this new question.

If revisited, first show on development scenes that actual lineage predicts correlated error beyond spatial distance, confidence and VSAC independence; then show ≥3 percentage-point pose AUC@5 improvement over the strongest comparator on disjoint scenes, with a positive scene-level confidence interval and no downstream degradation. Confidence calibration metrics alone cannot pass.

## Candidate 2: moment-assisted certification of truncated geometric costs

Known starting point: [DMS, CVPR 2025](https://arxiv.org/html/2506.02893v1), §3.2, replaces each correspondence cluster with second-moment information, approximates the Sampson denominator using a representative match, and moves truncation outside the cluster sum under an all-inlier/all-outlier assumption. The latter two approximations need not preserve candidate ranking. Its paper reports already-small average accuracy losses; this is a serious headroom warning, not evidence of a large opportunity.

The proposed object is the exact robust objective for a fixed finite candidate set:

`F(E) = sum_i min(n_i(E) / d_i(E), tau^2)`.

For each hierarchical 4D correspondence cell, retain count, geometric bounds, and `M = sum_i a_i a_i^T`, where `a_i = x_i ⊗ y_i`. Then `Q(E) = e^T M e = sum_i n_i(E)` is exact up to floating-point arithmetic. Pose-dependent conservative intervals for the denominator and individual residual give lower and upper bounds on the aggregate truncated cost. Refine only cells whose uncertainty can change which candidate wins; exact leaf evaluation is the fallback.

One candidate bound, to be proved and independently checked before any data:

- Suppose every denominator is in `[dmin,dmax]`, with `dmin > 0`, and every nonnegative residual `z_i=n_i/d_i` is in `[l,u]`.
- Bound the residual sum by `Slo=Q/dmax` and `Shi=Q/dmin`, intersected with `[n*l,n*u]`.
- If `u <= t`, where `t=tau^2`, use `[Slo,Shi]`.
- If `l >= t`, the truncated cluster cost equals `n*t`.
- If `l < t < u`, let `alpha=(t-l)/(u-l)`. Concavity of `min(z,t)` gives the secant lower bound `L=alpha*Slo+n*l*(1-alpha)` and `U=min(Shi,n*t)`.
- A denominator interval containing zero must trigger safe refinement or a separately proved limiting bound. Numerical guards must preserve containment through directed/outward rounding; a tolerance comparison is not a mathematical certificate.

This formula is a prospective derivation, not a published or experimentally verified result. It may be a routine combination. Certification only identifies the minimum of the chosen score among evaluated candidates; it does not certify the true camera pose, global geometric optimality, or better accuracy.

The closest direct overlap is [Space-Partitioning RANSAC, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920715.pdf), §§3–4. It conservatively bounds cell mappings, removes impossible inliers and rejects candidates through support bounds, including F/E and thresholded scoring. The potentially different element here is bounding the **nonconstant sum of truncated costs inside inlier/mixed cells with moments**, rather than only conservative membership/count culling. Plain exact early rejection, a hierarchy, or the term “certificate” is insufficient novelty.

[SupeRANSAC, arXiv 2506.04803](https://arxiv.org/abs/2506.04803) and [its implementation](https://github.com/danini/superansac) are essential modern comparisons. They demonstrate that implementation and solver configuration can dominate conclusions. The proposal must be compared against optimized dense scoring, DMS, SP-RANSAC, SP-RANSAC plus DMS where compatible, VSAC, and MAGSAC++ / SupeRANSAC under matched proposal streams and then matched wall time. Do not compare Python candidate code with a deliberately weak Python baseline while ignoring optimized C++ implementations.

Why it might improve quality: preserved scores plus saved evaluation time could allow more useful pose hypotheses under an actual fixed latency budget. Why it might fail: bounds may be loose for epipolar degeneracies; tree construction and memory traffic may dominate; ranking errors may be rare; extra hypotheses may not improve the best pose. Equal accuracy with a small isolated kernel speedup does not meet the user's substantial-result gate.

## Public asset metadata: verified, not downloaded

The official [DMS repository](https://github.com/jastermark/DMS) default branch is `master`, not `main`. The source is MIT; its dataset script links the following public files. HTTPS HEAD returned status 200 and `Accept-Ranges: bytes` on all six URLs below. ETags are server metadata, not content hashes.

| Asset | Bytes | Purpose |
|---|---:|---|
| [MegaDepth DKM matches](https://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_dkm.h5) | 332,175,160 | Coordinates / geometry cache |
| [MegaDepth RoMa matches](https://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_roma.h5) | 332,175,160 | Second matcher |
| [ScanNet DKM matches](https://vision.maths.lth.se/viktor/posebench/relative/scannet1500_dkm.h5) | 322,764,856 | Indoor confirmation candidate |
| [ScanNet RoMa matches](https://vision.maths.lth.se/viktor/posebench/relative/scannet1500_roma.h5) | 322,764,856 | Second matcher |
| [MegaDepth images](https://vision.maths.lth.se/viktor/posebench/relative/megadepth1500-images.zip) | 623,957,998 | Not needed for coordinate-only solver |
| [ScanNet images](https://vision.maths.lth.se/viktor/posebench/relative/scannet1500-images.zip) | 264,001,670 | Not needed for coordinate-only solver |

Four match caches total **1,309,880,032 bytes**, about 1.22 GiB. No teacher weights are needed to evaluate geometry from cached matches. Peak memory should be kept small by processing one image pair at a time. Actual CPU installation, solver availability, integrity, pair schema, calibration conventions and scene IDs are unverified.

Sources inspected: [MegaDepth script](https://github.com/jastermark/DMS/blob/master/data/download_megadepth.sh), [ScanNet script](https://github.com/jastermark/DMS/blob/master/data/download_scannet.sh), [loader](https://github.com/jastermark/DMS/blob/master/src/dms/dataloader.py). The public directory also lists sparse SIFT / SuperPoint+LightGlue caches and ASpanFormer caches, useful as potential independent matcher controls. No guarantees about additional fields follow from file names.

Licenses: [MegaDepth's official page](https://research.cs.cornell.edu/megadepth/) states CC BY 4.0 for derived dataset assets and separate licenses for original images. [ScanNet's official repository](https://github.com/ScanNet/ScanNet) says dataset use is governed by its Terms of Use, distinct from the code's MIT license. The ScanNet TOS PDF fetch timed out in this review; no claim is made that access/redistribution requirements are satisfied. Public-hosted derived caches are not automatically relicensed by DMS's software license. Store only derived aggregate results in GitHub until applicable terms are verified.

## Prospective smallest useful experiment

1. Mathematical/implementation phase: establish a valid bound and exact fallback; challenge containment with degenerate denominators, low parallax, high dynamic range and mixed cells. This is correctness evidence, not a CVPR result.
2. Data admission: inspect metadata first and freeze exact scene IDs and checksums. Prefer a separate official development/training source. If a whole MegaDepth1500 scene is deliberately consumed as development, mark the entire physical scene development permanently and do not later report a clean full-standard-benchmark result. Pair-level random splits are inadequate; shared image IDs and scenes must stay in one partition. Verify rather than assume how many scenes exist.
3. Development diagnostic: use a fixed finite hypothesis stream, same solver and thresholds. Compare exact dense scores, DMS approximate scores, SP-style culling, and the new intervals. Measure whether approximation changes winners and whether any such changes improve true pose. GT is used only in the scorer, never to form bounds or choose cells.
4. Kill early if exact scoring's oracle advantage over the strongest approximate scorer is below 2 AUC@5 points at the prespecified development budget, if setup-inclusive runtime is worse, or if bounds fail containment once. Do not rescue by choosing only the pairs where approximation fails.
5. If the diagnostic has meaningful headroom, freeze one algorithm and compare matched-time pipelines on untouched scenes and a second dataset. The final protocol sets a manuscript gate of ≥2 AUC@5 percentage points on each of two independent held-out sources, scene-cluster confidence interval lower bound >0, median and tail runtime within the fixed budget, and stable gains for two matchers. These are prospective thresholds, not measured achievements or guarantees. A different justified threshold may be set only before data outcomes are inspected.

No new benchmark was allocated by this memo. No result, novelty, or paper-readiness is claimed. The recommended next action is a small admission test with an explicit stop rule, not a tracker-training framework or manuscript draft.

Subsequent decision: the prospective protocol selected candidate2 and2-point matched-time AUC@5 as the later outcome gate before data values were read. The original memo's3-point suggestion was not a frozen protocol or observed result. Metadata allocation is in configs/0012-development-manifest.json.
