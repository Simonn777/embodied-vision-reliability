# Independent saved-output review of 0015

The predefined expansion gate **failed**. Full AUC@5 increased only **0.009558333 percentage points** (required ≥0.25); same-index subset AUC@5 decreased **3.091052179 points** (required ≥2 gain). Full pose error improved in **9/16** pairs (required ≥10), and worsened in 7. Coverage, 12 usable subsets and the AUC@20 guard passed. This is not a material gain, a novelty result or a manuscript admission.

| Method | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| Full original | 81.169622478 | 90.584811239 | 95.292405619 |
| Full refined | 81.179180811 | 90.589590405 | 95.294795203 |
| Same subset, original | 43.967891463 | 55.544366385 | 64.937664723 |
| Same subset, refined | 40.876839284 | 57.719611465 | 66.359805733 |

All metrics use all16 pairs. Pairs3,10,11,13 have fewer than8 admitted queries, so both subset arms receive180° error; these pairs remain in the denominator. Subset AUC@10/20 improves, but does not override the frozen AUC@5 gate. All original/full-refined errors are below5°.

The audit independently checked both16-record seals, 32 process receipts, 16 original baseline prediction hashes, source/configuration/calibration/library and inlier-mask bindings, admitted coordinate/change accounting, pose validity, identical subset indices and returned-inlier maps. All13 scientific source files match prediction freeze `057a5ea1527d3e205987a2780a512d01ed490a87`. It used saved JSON, source and logs only: no repository scientific module imports, HDF5/image reads, estimator calls, or evaluator rerun. The actual cached arrays and image pixels were not independently re-decoded.

Of1355 selected queries,1331 initialized;191 were admitted and all191 changed target coordinates. There were24 initialization failures and1140 fit failures; failed-query targets stayed unchanged in the saved replacement records. Zero admitted queries were unchanged or zero-update fits. The191 replacements represent **0.119375%** of160000 full-arm observations. Admitted displacement ranges0.027163–3.176924px (median0.407407px). There were exactly40 successful new native calls:16 full,12 original-subset,12 refined-subset. No retry or failed native call was recorded.

The photometric gate rejected1031 queries for iteration exhaustion and1003 for excessive predicted remaining decrease; these reasons overlap. It rejected93 for target patch bounds. Low admission is a limitation of this fixed configuration; these observations do not authorize retrospective gate relaxation or claim that all photometric information is useless.

Timing totals are8.584815s for the16 photometric workers (including5.930790s fitting,0.497191s image reading and0.251299s selection),3.070500s for full-refined native calls,0.039219s original-subset calls and0.038131s refined-subset calls. The preserved full-original native calls total3.098395s. All32 new worker processes took22.686077s outer supervisor time, including controls and orchestration. These scopes are different; this is not a matched end-to-end speed claim. Maximum sampled worker RSS was114144KiB. Prediction logs contain ordinary completion records and no warnings/tracebacks.

The original scorer failed with `KeyError: call_seconds` when reporting baseline time. The original baseline schema stores it at `timing_seconds.estimator_call`; the artificial fixture and prefreeze independent review missed that mismatch. Original prediction/scorer files remain unchanged. A separate reporting-only scorer was frozen at `6d1f377bc5744a934bee8197914118b7c4241ca9` and executed once from saved outputs, taking3.062927s with an empty log. No observation, pose, metric formula or gate was changed. See the [original failure](../evaluation.log), [failure receipt](../evaluation.process.json) and [repair protocol](../../../protocols/0015-evaluation-metadata-repair.md).

Independent angle recomputation uses projected GT rotation and **signed** translation; AUC uses the frozen left-threshold trapezoid convention. Maximum difference from the repaired scorer is3.318e−12degrees and1.606e−12AUC percentage points. No substantive binding or arithmetic discrepancy was found.

- [Photometric seal](../photometric/seal.json): `fbab3146703c6dba2222a0dac11a234ec96d131030fea1cdad63595854842c48`
- [Geometric seal](../geometric/seal.json): `8f92c1a73e402c1299183c2cfcaf2e8a236d5c2b41dcaae4b1ea09c6fce0ea9d`
- [Repaired evaluation](../evaluation-repair.json): `9badd27ccca63f3c83347cff9731590d3d3f6820cedd4515bf010cf1e3a38cea`
- Repair source SHA256: `d4f9ed15ede9382890aba29d82ac072e920f27fcab663c6c911fd7104f922cf3`
- [Audit JSON](audit.json) contains all16 per-pair errors/counts, complete checked-file hashes, costs and gate arithmetic.

A separate **post-hoc descriptive** [epipolar check](../epipolar-review/analysis.json) was also verified independently for all191 admitted points. Using the original source coordinate and fixed GT pose, the perpendicular target-pixel distance to its GT epipolar line increases in mean from0.536246830px to0.596308882px; equal-pair mean increases from0.546650395px to0.607451459px. There are89 improved and102 worsened points, with4/16 lower pair means. Independent normalized-ray/cross-product arithmetic agrees with the saved pixel-F calculation within1.61e-12px. This measures one-dimensional epipolar compatibility conditioned on the possibly noisy source point, **not true correspondence error** or independent evidence for selecting a new method. It does not alter the prospective gate.

The reused, dependent single-scene cohort, sparse replacement budget and fixed asymmetric patch convention limit inference. This result does not support expanding this exact configuration or writing a contribution paper. No further experiment or same-configuration tuning was performed by this audit.
