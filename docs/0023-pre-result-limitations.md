# 0023 limitations recorded before pose results

2026-10-19. Recorded before any 0023 pose fit or ground-truth access. At this
point prediction01 had failed only at canonical-cache calibration verification,
as preserved in [the technical amendment](0023-cache-verification-repair.md).
Acquisition and frontend completion establish usable inputs, not pose accuracy.
This note changes no frozen scientific source, population, control, gate or
fallback in [the protocol](../protocols/0023-mapfree-directional-instability.md).
It makes no efficacy or manuscript-admission claim.

## Closest verified primary sources and narrow remaining hypothesis

* **Learning a Depth Covariance Function, CVPR 2023**, Section 5.2, Eq.12,
  already incorporates spatially correlated depth priors into joint camera,
  structure and per-image scale optimization. Its covariance is learned from
  images; reference-group deletion is not its estimator.
  [Official paper](https://openaccess.thecvf.com/content/CVPR2023/html/Dexheimer_Learning_a_Depth_Covariance_Function_CVPR_2023_paper.html);
  [verified full text](https://arxiv.org/html/2303.12157).
* **Scalix, August 2026 preprint**, Eq.4, explicitly models cross-pixel
  covariance from shared scale uncertainty. Section V-C jointly marginalizes
  pose and scale information. Correlated depth uncertainty, pose coupling and
  Schur marginalization are therefore not contributions of 0023.
  [Verified full text](https://arxiv.org/html/2608.17553v1).
* **Marginalized Bundle Adjustment, 3DV 2026**, Sections 3.1–3.2, already
  combines monocular depth, affine depth calibration and robust multi-view
  camera estimation. Its residual-distribution marginalization differs from
  Schur elimination; the shared word must not obscure that distinction.
  [Verified full text](https://arxiv.org/html/2602.18906v1);
  [primary venue record](https://research.google/pubs/marginalized-bundle-adjustment-multi-view-camera-pose-from-monocular-depth-estimates/).

The remaining testable hypothesis is specifically **reference-only deletion
instability projected onto query-supported non-affine depth functionals, while
retaining the joint pose/depth covariance**, compared with the same observations
in joint BA and scalar controls. The examined full texts do not establish an
identical construction, but this limited search is not proof of novelty.
Deletion, covariance inflation and quotient projection are established tools;
their combination supplies no new observations.

## Counterexamples and interpretation limits

1. A stable depth bias shared by every reference can yield negligible deletion
   variation while still damaging the query estimate.
2. Removing a critical reference can change graph conditioning, connectivity or
   the attained optimization basin even with correct depths. Eight overlapping
   deletion folds are not eight independent samples; each query endpoint has
   only three direct reference neighbours. Variation alone does not identify
   depth-model misspecification.
3. Removing global affine depth directions does not prove those directions are
   irrelevant to pose. Query depth-functional visibility is not pose
   observability, and reference null directions cannot acquire information
   merely through covariance inflation.
4. In a correctly specified linear-Gaussian problem, exact joint estimation
   already uses the same observations efficiently. Inflation has no universal
   accuracy advantage. A gain must be established for a particular failure
   mechanism, not attributed to additional information.

## Nonstationary reference states and the omitted linear term

The frozen study accepts finite last-accepted states under a fixed optimization
budget; it does not require stationarity. Its compressed reference priors are
centred at those states and constructed from Jacobian information. They do not
retain a reference gradient offset. This is a limitation of interpreting that
prior as an exact local marginalization, not a change to the frozen algorithm.

For a local least-squares model, nuisance elimination gives
`||P (r + J_keep delta)||^2`, with `P = I - J_nuisance J_nuisance^+`.
Keeping only `||P J_keep delta||^2` omits the linear term unless the reduced
gradient vanishes. Scalix explicitly retains the residual offset in **PDF page
5, Section V-C, Eq.17**, and eliminates both the Hessian and right-hand side in
**Eq.19–20**.
[Exact primary source](https://arxiv.org/pdf/2608.17553v1#page=5).
Consequently, even a development-gate pass would not alone distinguish useful
misspecification covariance from altered anchoring around a nonstationary state.

## Descriptive audit after a completed, eligible run

Saved field summaries contain states, objective values, block diagnostics and
termination information, but not residual vectors or full Jacobians. Saved
information spectra also do not determine a gradient. JSON alone therefore
cannot establish stationarity.

After a complete prediction seal and successful outer operation, a separately
recorded descriptive audit can reconstruct the frozen inputs, retained edges and
saved states, and evaluate `FieldProblem.local(state)` without refitting. It can
first check the saved objective and block counts, then report the true objective
gradient `2 J_exact.T @ r` and the reduced linear term under the frozen IRLS
information approximation. The exact radial residual embedding and the IRLS
Jacobian must not be mixed to form a spurious gradient. Rank/null-space effects,
cap boundaries and proxy curvature remain explicit limitations. This requires
neither new model/native calls nor GT and creates no replacement predictions.

Such an audit cannot change the 24-query gate, exclude outcomes or justify
retuning on these queries. Any subsequent method validation needs a separately
frozen untouched-data comparison of offset-retaining marginalization and
directional versus scalar inflation, with the same observations and complete
cost accounting. A current pass admits that investigation only; a current gate
failure remains closed under the protocol.
