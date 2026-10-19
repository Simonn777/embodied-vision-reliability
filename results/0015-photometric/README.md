# 0015: sparse photometric correction does not pass the expansion gate

**FAIL.** Full AUC@5 changes from81.169622478% to81.179180811%, only
**+0.009558333 percentage points** (required≥0.25). The same-index subset
changes from43.967891463% to40.876839284% (**−3.091052179 points**, required≥+2).
Full pose error improves9/16 pairs (required≥10). Preserve all results; do not
expand or tune this sparse configuration, propagate its corrections, or write
a method paper from it. The broader research objective remains active.

| Arm | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| Full original |81.169622478|90.584811239|95.292405619|
| Full refined |81.179180811|90.589590405|95.294795203|
| Same subset, original |43.967891463|55.544366385|64.937664723|
| Same subset, refined |40.876839284|57.719611465|66.359805733|

All16 pairs remain in every denominator. Four pairs have fewer than8 admitted
points and retain180degree error in both subset arms. Gains at AUC@10/20 do not
replace the frozen primary AUC@5 criterion. These reused pairs share images in
one development scene; no held-out or population claim follows.

Of1355 selected queries,1331 initialize and191 pass the photometric criteria.
All191 change the target coordinate: only0.119375% of160000 full observations.
There are24 initialization failures and1140 photometric failures; every failure
retains original coordinates. Forty native calls finish once:16 full and12 per
subset arm. The low admission and sparse budget limit the conclusion; failure
of this configuration is not a proof that all image information is unhelpful.

The [saved descriptive analysis](descriptive/README.md) records that1031 fits
reach the30-attempt cap and1003 fail the local GN criterion (987 overlap), with
93 initial target-boundary failures. Admitted patch loss drops88.4458% in total,
yet task accuracy does not improve materially. A separately labeled
[post-hoc epipolar check](epipolar-review/analysis.json) finds target-line distance
mean0.536247px→0.596309px,89 points improved/102 worsened,4/16 lower pair means.
It measures conditional epipolar compatibility, not true correspondence error.
Neither result authorizes loosening gates, selecting favorable pairs or tuning
iterations to rescue this outcome.

Scientific source freeze `057a5ea1527d3e205987a2780a512d01ed490a87` preceded all32
workers. Both complete prediction seals preceded evaluation. The outer operation
took22.686077seconds; photometric workers total8.584815seconds, including5.930790
fitting. Full-refined native calls total3.070500seconds, versus3.098395seconds
for the earlier original baseline; the additional image work is not free.
Maximum sampled RSS was114144KiB. This is not a matched end-to-end speed claim.

The first evaluator failed on absent top-level `call_seconds`; the actual baseline
field is `timing_seconds.estimator_call`. Its source, process record and traceback
remain preserved. A separate [metadata-only repair](../../protocols/0015-evaluation-metadata-repair.md)
was frozen at `6d1f377bc5744a934bee8197914118b7c4241ca9` before one saved-only score,
which exited0 in3.062927seconds. No scientific worker, metric, coordinate or gate
was changed or rerun. The original artificial fixture and prefreeze review missed
this schema mismatch; the new fixture checks the actual nested representation.

[Independent review](independent-review/README.md) verifies all source/process
bindings and reproduces every pose metric. Maximum differences are3.318e−12degrees
and1.606e−12AUC percentage points. The separate epipolar derivation agrees within
1.61e−12px. Photos are private and are not redistributed.

## Evidence

- [Frozen scientific protocol](../../protocols/0015-photometric-information-development.md)
- [Photo seal](photometric/seal.json) and [pose seal](geometric/seal.json)
- [Repaired evaluation](evaluation-repair.json) and [process receipt](evaluation-repair.process.json)
- [Supervisor receipt](supervisor.process.json) and [full log](supervisor.log)
- [First evaluator failure](evaluation.log), [receipt](evaluation.process.json), [path-normalization hashes](evaluation-log-normalization.json)
- [Independent audit](independent-review/audit.json)
- [Prior-work and identifiability screen](../../docs/literature/0015-photometric-followup-prior.md)

Any further study must justify a distinct information source or falsifiable
mechanism. Generic joint patch/pose refinement is already covered by JET and
featuremetric work; residual propagation overlaps with EpicFlow and ArgMatch.
No follow-up scientific operation or manuscript is admitted by this failure.
