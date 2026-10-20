# 0026 completed evidence review

**Fixed development variant failed; saved-output audit passed.** This review only read completed files and recomputed file hashes. It did not rerun the estimator, evaluator, auditor, objective fits, native library, or original scientific arrays.

The frozen `G_lift` union achieved exact empirical directed AUC@5 of **73.5881010546%**, versus **74.1396701117%** for the strongest predeclared control, `F_all`: **−0.5515690571 percentage points**. Only two of five scenes improved; London lost 2.2505351115 points. The pooled +2-point, three-positive-scenes, and no-scene-loss-over-one-point clauses failed. AUC@20 loss was 0.1378922643 points and passed its 0.5-point limit. Thus the complete gate failed, with no manuscript or goal-completion admission.

All 80 pairs, 560 new slots, 640 raw incumbent/proposal entries, and 11 evaluated methods were retained. In particular, the `G_lift` proposal completed for 35 pairs, failed for three, and was unavailable for 42; the union selected that proposal on 16 pairs and otherwise selected the fixed incumbent/continued-TC alternatives. This is not an evaluation on a successful subset. `G_all` and `F_all` are numerically indistinguishable from their corresponding local-start unions at the aggregate precision relevant here; the lift produced no established aggregate gain.

The separately implemented saved auditor passed 14,496 comparisons, rehashed 686 consumed files, checked 320 historical records, and reported zero failures. Maximum differences were 3.802e−11 degrees, 3.536e−11 AUC percentage points, and zero seconds for reconstructed charged component sums, within the prospectively fixed tolerances. Historical controls were recomputed under this study's same exact empirical AUC rather than imported as headline numbers from earlier metric implementations.

I independently checked 376 file/source/receipt bindings in this review: all 80 final/partial/process/log groups, the prediction job, all 20 current and frozen source files, all three operation specs, output hashes, and stdout/stderr hashes. All matched. Source commit was `b596542f9eee399219ac03fdbcd91b33e8417fd6`; saved CI receipt identifies successful run `39363242000`. Native library and build-receipt bytes also matched the job's pinned hashes. The three external operations completed with exit 0, no termination, no surviving descendants, no cleanup error, and empty stderr. Prediction/evaluation/audit outer wall times were 53.0001145 / 0.541215917 / 0.532888167 seconds; corresponding peak sampled process-group RSS was 124080 / 62912 / 66096 KiB.

Principal output bindings:

- Prediction seal: `f295b3f240d9af080148a4680c81d42098d25579ac01117e1dd499d78d980736`.
- Evaluation: `a45edff1176723122a89c0e5d709ff464f13417f1aacf91418b2e376ad8d2617`.
- Independent audit: `ea23b9ce9a4f46eb74a505adb61b9d4d6ce32bd018bb62830917ba933eada029`.
- Auditor source: `93e32dc24cf80f6473c07f63fe9821b9a904349fad83aff21fe39422b461952f`.

The scope remains development evidence from already exposed scenes, not held-out generalization or an independent-team replication. The auditor reconstructs score-based selections from saved scores and independently recomputes pose metrics; it does not independently recompute pixel scores or optimizer solutions from original correspondences. Charged prototype components are not a matched complete-pipeline speed comparison; checkpoint timings must not be double-counted as disjoint extra compute. Neither the numerical relaxation diagnostics nor this audit is a rigorous global certificate or evidence of novelty. The fixed 0026 efficacy claim should close as failed, while preserving the exact algebra and all outcomes without promoting an ablation after seeing results.
