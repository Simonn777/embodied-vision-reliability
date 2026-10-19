# 0011: oblique coordinate coupling — negative development result

**Decision: NO_GO.** The fixed-marginal candidate did not improve tracking quality.
All three predeclared quality gates failed; all four numerical gates passed.
No manuscript or method-improvement claim follows. Substantial AI assistance is
disclosed in [CONTRIBUTIONS](../../docs/CONTRIBUTIONS.md).

## Complete comparison

Quality averages strict distance hits at 1/2/4/8/16 px, within each group then
equally across the two groups. It is not TAP-Vid AJ. Eligible rows are subsequent
observations with non-occluded GT: 87 in g-036 and 53 in g-002. Every method has
100% coverage of those same 140 rows; query rows are excluded from quality.
No predicted confidence/visibility filters the evaluation.

| Method | g-036 quality | g-002 quality | Macro quality | Macro EPE(px) |
|---|---:|---:|---:|---:|
| original_native | 62.0690% | 64.1509% | 63.1100% | 11.74824 |
| reference_native | 62.2989% | 64.5283% | 63.4136% | 12.54499 |
| oblique_native | 69.8851% | 58.4906% | 64.1878% | 9.42900 |
| axis_control_native | 57.0115% | 58.8679% | 57.9397% | 14.40514 |
| reference_independent | 62.2989% | 64.5283% | 63.4136% | 12.54499 |
| coordinate_average | 66.2069% | 60.3774% | 63.2921% | 10.68808 |
| entropy_select | 63.6782% | 61.8868% | 62.7825% | 12.58809 |
| product_joint | 66.4368% | 61.1321% | 63.7844% | 12.44367 |
| mixture_joint | 63.9080% | 60.7547% | 62.3314% | 12.40180 |
| oblique_coupling | 62.5287% | 63.3962% | 62.9625% | 12.59547 |
| axis_coupling | 62.2989% | 64.5283% | 63.4136% | 12.54499 |

The candidate is **1.22533 percentage points worse** than the strongest macro
control, oblique-native (exact difference -113/9222). Its EPE is 12.59547 px
versus 9.42900 px. Against the strongest control in each group, quality differences
are -7.35632 pp and -1.13208 pp. At 4 px, comparison with macro-best oblique-native
gives 3 corrections and 11 harms in total (g-036: 1/10; g-002: 2/1). These counts
are descriptive paired totals, not independent statistical replications.

[Complete frozen-score output](results.json) includes all thresholds, medians,
occluded/all-point EPE and paired comparisons. [All 256 case errors](case-errors.json)
retain query rows and occluded points as well as the 140 primary eligible rows.

## Execution and independent checks

The [published freeze](../../docs/0011-freeze-publication-verification.md) preceded
preparation and one run of each phase. Source commit a288f65; freeze 1951ae7;
merged main a0b0616. Newly allocated development groups g-036/g-002 each used
16 sparse observations at original frames 0, 16, ..., 240 and eight initial queries.
Four independent views produced 128 total forwards,256 point/frame rows and eleven
decoded methods. A deployed candidate would require two views; the other two
are experimental controls. This is sparse-observation, causal tracking.

| Scientific phase | Wall(s) | Peak RSS(KiB) | Exit |
|---|---:|---:|---:|
| Preparation |23.515624|2,568,120|0|
| g-036 prediction |167.109087|3,733,668|0|
| g-002 prediction |153.888267|3,735,396|0|
| Saved-output derivation |11.847668|39,084|0|
| Formal score |0.155039|33,596|0|

All phases used the existing cloud CPU under four-core/four-thread/8GiB limits.
Independent metrics review passed 14,935 checks including 3,997 numerical
comparisons, maximum difference 1.4211e-14. Separate positive-domain IPF replay
passed all 256 points/5,632 coordinate comparisons, maximum difference 2.2738e-13 px.
Neither reviewer called the model or imported the frozen solver/scorer. The
numerical reviewer did not read RGB, GT or quality scores; the metrics reviewer
used private saved GT. This agreement validates a negative result.

## Evidence and honest limits

The [archive](cloud-bundle/bundle.tar.gz) contains 101 text/source/log files
(2,745,997 uncompressed bytes); [manifest](cloud-bundle/manifest.json) records
all member hashes and the sole large historical preexecution-inventory omission.
Archive SHA256:24826b2aadff7f45fc51789ef0e72523809e7d1f18db86ad619ef1f63fc3dea6.
The parent independently retrieved and verified every member. Raw RGB/GT/logit
NPZ and checkpoint weights remain in the original cloud and are not in this bundle.
The two readable JSON files above are byte-exact copies of archived score outputs.

The bundle wrappers describe their historical packaging state, including
`published:false`; they are preserved unchanged. Subsequent publication is recorded
separately. Two read-only summary commands failed after successful derivation.
The second command printed too much, its transport was truncated, and no full
original terminal log was archived. The explicit provenance note preserves that
gap; saved scientific arrays and the closed phase logs are unaffected.

The cloud postcheck preserved 174 original study files,24 frozen inputs,3,393
older tracked files,53 cache metadata records and52 prior control/marker records.
Large cache preservation uses unchanged stat metadata, not a fresh parent byte
comparison. Parent [actual command observations](independent-review/parent-tool-observations.json)
separately establish scientific command exits.

Two simulated groups share the RGB-Stacking source; scene and checkpoint-training
independence are unknown. No population CI, real-robot result or novel optimizer
is claimed. The [protocol](../../protocols/0011-oblique-coupling-development.md)
and original results are unchanged. Do not repeat or tune this failed configuration.
No new model/data downloads, installations, paid APIs or added fees were used.
The broader research objective remains open; [next-direction review](../../docs/literature/post-0011-direction-review.md)
does not yet admit another experiment or paper.
