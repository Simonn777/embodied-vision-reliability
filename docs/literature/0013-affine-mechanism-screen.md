# 0013: screening a dense-to-affine pose mechanism

Scope: targeted primary-source reading only. No correspondence, confidence,
calibration or pose arrays were read; no experiment or installation occurred.
This note does not admit a new method. The next registered activity is a
baseline-only pose-headroom diagnostic, controlled by its separate protocol.

## Decision and exact overlap

Combining existing components can produce a useful research contribution.
However, the claim that converting dense neighborhoods to affine constraints,
adding spectral uncertainty and using a two-affine-correspondence solver is
itself new is not supported. The increment over ordinary local fitting and
existing uncertainty handling must be defined and demonstrated.

| Primary source and reading depth | What is already established |
|---|---|
| [Sun et al., Learning Affine Correspondences by Integrating Geometric Constraints, CVPR 2025](https://arxiv.org/html/2504.04834v1), method §§3.2–3.4 read in full-text HTML | Dense matches locate corresponding **32×32 RGB patches**. A second learned module estimates orientation, scale and residual affine shape. The described method does not obtain affine maps by least-squares fitting neighboring warp coordinates. This is a distinction in extraction, not proof of novelty for local fitting. Code was not audited. |
| [Minh and Hajder, Affine Transformation from Fundamental Matrix and Two Directions, VISAPP 2020](https://www.scitepress.org/PublishedPapers/2020/91544/91544.pdf), publisher-indexed §3.2 and bibliographic text read; direct PDF opening failed | The section explicitly fits a six-parameter affine transformation from optical-flow correspondences in a disk around a point, using a linear least-squares system. This directly precedes training-free neighborhood fitting. The paper's other contributions involving a known fundamental matrix must not be confused with this local fitting operation. DOI: 10.5220/0009154408190826. |
| [Barath et al., Making Affine Correspondences Work in Camera Geometry Computation, ECCV 2020](https://arxiv.org/html/2007.10032v1), §§3, 4.4–4.5 and uncertainty supplement read | Symmetric intensity-based affine refinement, parameter covariance, uncertainty propagation through minimal solvers, degeneracy checks and point-based local optimization are covered. Essential-matrix uncertainty preemption was excluded from their experiments because per-model computation was too expensive. |
| [Astermark et al., Dense Match Summarization for Faster Two-view Estimation, CVPR 2025](https://arxiv.org/html/2506.02893v1), §§3–4.4 read | Cluster summaries retain geometric constraints beyond a representative point. CCA means center sampling, center scoring and approximate-summary refinement. The authors also tested estimation from one summary and report generally worse results, cautioning against assuming fewer sampled groups ensure better estimates. |
| [G²SR: Geometric Methods for Fast and Memory-Efficient Gaussian-based Surface Reconstruction, 2026 preprint](https://arxiv.org/html/2607.14470v1), local affine fitting around equations 5–6 read | A local affine map is fitted to optical-flow supports by least squares, with a small sigma-point approximation. This is a further extraction precedent in surface reconstruction, not evidence of two-view pose accuracy. |

## What the moments do and do not add

For centered source and target coordinates, let Cxx, Cyx and Cyy denote their
unnormalized scatter matrices. The ordinary local least-squares solution and
its residual scatter are

    A_hat = Cyx Cxx^{-1}
    S = Cyy - Cyx Cxx^{-1} Cxy.

The inverse requires adequate rank; a pseudoinverse does not make an
unobserved direction reliable. With our row-major feature convention

    a = [y0*x0, y0*x1, y0, y1*x0, y1*x1, y1, x0, x1, 1],
    G = sum a a^T,

the submatrix with zero-based indices [2,5,6,7,8] contains the first and second
moments of [y0,y1,x0,x1,1]. It therefore determines these local fits. Here G is
the Gram matrix; DMS's paper also uses a factor M satisfying M^T M=G.

The fitted affine map supplies neighborhood deformation information relative
to a **center-only** representation. It adds no independent observations
relative to G or the original matched coordinates. A gain could arise from a
useful local smoothness assumption, denoising or a different hypothesis
generator. Adding affine constraints and their fitting points as independent
measurements would double-count the same evidence.

Small eigenvalues of Cxx diagnose weakly sampled directions, not the full
uncertainty of a learned warp. Coherent errors and neighborhoods crossing
occlusion/depth boundaries can violate ordinary regression assumptions.
Duplicating identical observations leaves the fitted A unchanged and does
not add independent information, although an IID covariance estimate may
incorrectly become more confident. K, R and t alone also do not uniquely
specify a ground-truth local affine map: local surface geometry remains
unknown.

## Potential increment and its required evidence

A possible later question is whether correlation-aware local differential
constraints can remain useful near ambiguous or mixed neighborhoods without
becoming falsely confident as match density increases. This is a research
question, not an established novelty or an authorized implementation.

A proposed combination must show a concrete increment over plain local
least-squares plus an existing affine solver and uncertainty treatment. At a
minimum, comparisons must include full PoseLib, DMS CCA, ordinary local
LS-plus-2AC, the same neighborhood's point-only fit, and relevant uncertainty
controls. Input access, local optimization and computational budgets must be
matched. Separating center information, local smoothing, affine constraints
and uncertainty handling is necessary to identify the source of any gain.

The intended benefit must be better pose estimates on the prespecified
evaluation, not merely fewer hypotheses, smaller residuals or a confidence
score. If ordinary local fitting or existing uncertainty treatment explains
the gain, the claimed additional mechanism has not been demonstrated. If a
distinct treatment does improve those strong controls, using known components
does not by itself invalidate that contribution. Neither absent search hits
nor this bounded source review establish priority.
