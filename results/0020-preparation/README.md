# 0020 preparation: spatial transfer of depth calibration

This package freezes a four-pose selector whose depth calibration is fitted on
three source-image regions and evaluated on the omitted region. It tests whether
regional transfer contributes information beyond geometric scoring, global
affine-depth fitting, spatial density weighting and random-point cross-fitting.
All four candidate poses already used the full correspondence set. Only nuisance
calibration is held out, and the five source scenes remain development forever.

The fixed candidate bank contains the original 0017 poses. The improved 0019
truncated-Cauchy poses are mandatory fixed benchmark comparators, never additional
selectable candidates. Together with four same-bank controls and the four fixed
original methods, there are ten comparators. Admission requires at least two
AUC@5 percentage points above the strongest, positive gains in at least two
scenes against that same comparator, and at most half a point AUC@20 loss against
the strongest AUC@20 comparator. Failure closes this fixed formulation.

The [protocol](../../protocols/0020-spatial-depth-transfer.md) specifies every fit,
partition, score, failure rule, comparison and resource limit. No native estimator,
new model, image, optimization or asset download is used. All scoring records and
a complete outer-process receipt must validate before saved reference labels are
decoded. Mathematical degeneracy retains its full saturated validation cost;
technical failure blocks the cohort.

This is prospective qualification, not a measured scientific result. The required
independent-source evaluation, official strong baselines, matched complete-pipeline
costs, replication and novelty evidence remain prerequisites for a manuscript.

All 33 combined manufactured tests passed (actual exit 0; observed process wall
time 1.334321417 seconds). The [receipt](artificial-test-receipt.json),
[complete log](artificial-tests.log), [source bindings](source-bindings.json) and
[independent preparation review](../../docs/0020-preparation-review.md) are retained.
