# Complete memory-readout comparison: negative result

The primary `late_mean` readout reaches **44.5832299 J / 24.3306200 Jtr%** on the four exposed VOST TRAIN clips. Its changes relative to native are **+0.2507305 / +0.2138975 percentage points**. It is **6.7847371 J / 7.6512308 Jtr points below** the strongest included control, the sealed D37 fixed negative veto. Three of four frozen continuation clauses fail. Close this exact uniform/latest-write/full-stack shadow-readout recipe without weight, threshold, layer, seed, cohort or resolution rescue.

This is a technically complete, audited negative development experiment. It does **not** establish a novel method, a substantial gain, held-out transfer, a full recurrent tracking policy or readiness for a manuscript. The final research goal remains unmet.

## All nine arms

Scores below are percentages; changes are absolute percentage points. Primary scoring follows the pinned VOST author implementation: label255 is background. All four clips have equal weight, with the frozen438 interior J positions and110 tail Jtr positions. The separate void-ignore sensitivity preserves the same population and also fails three of four clauses. Jtr is not boundary F; no official J&F result is claimed.

| Arm | Primary J | Primary Jtr | Void-ignore J | Void-ignore Jtr |
|---|---:|---:|---:|---:|
| Native B | 44.3324994 | 24.1167225 | 45.6111333 | 24.4760536 |
| B+M1 | 48.5356317 | 31.1076415 | 49.6980230 | 31.6221495 |
| B+M2 | 46.0317472 | 25.3822545 | 47.5831386 | 25.8448033 |
| Union B+M1+M2 | 48.6004894 | 31.3383185 | 49.8047423 | 31.9481137 |
| D37 fixed negative veto | 51.3679670 | 31.9818508 | 52.8473341 | 32.6730818 |
| Early memory mean | 44.6408066 | 24.3704179 | 45.9238432 | 24.7149496 |
| Flat complete histories | 44.6374043 | 24.4753531 | 45.8714902 | 24.8393767 |
| **Late full-stack mean, primary** | **44.5832299** | **24.3306200** | **45.8539758** | **24.6827204** |
| Output-logit ensemble | 44.4232187 | 24.1644101 | 45.7021271 | 24.4910689 |

The primary also loses to the early and flat readouts on both aggregate endpoints. Relative to union it changes J/Jtr by −4.0172595/−7.0076985 points. D35 controls were reproduced and matched exactly; D37 fixed negative veto is the pre-existing sealed numerical reference, not a rerun of its training or flow pipeline.

## All four clips

Every raw position remains in the artifacts. No clip was removed after observing results. Each cell is J / Jtr in percent under the primary policy; all nine per-clip and per-frame scores are in [evaluation.json](complete-01/evaluation.json).

| Clip | Raw frames | Native | Union | Fixed negative veto | Primary late mean |
|---|---:|---:|---:|---:|---:|
| 4691_cut_guava | 86 | 43.726073 / 12.006843 | 54.468378 / 22.178485 | 54.468414 / 22.178485 | 44.008153 / 12.386267 |
| 817_cut_onion | 90 | 35.328639 / 18.102813 | 19.206603 / 16.150963 | 30.300033 / 18.726142 | 35.581928 / 17.661872 |
| 9675_fold_clothes | 201 | 59.099082 / 46.071685 | 81.041529 / 66.652790 | 81.042825 / 66.652790 | 59.607295 / 47.029035 |
| 1934_tear_film | 69 | 39.176203 / 20.285549 | 39.685448 / 20.371036 | 39.660596 / 20.369986 | 39.135544 / 20.245306 |

| Frozen continuation clause | Outcome |
|---|---|
| At least +3 J points over every other arm | Fail: −6.7847371 against the strongest control |
| At least +3 Jtr points over every other arm | Fail: −7.6512308 against the strongest control |
| Positive Jtr change over union on at least3/4 clips | Fail: only onion improves,1/4 |
| No clip loses more than0.5 J or Jtr points against native | Pass |

The four clips are permanently exposed development data. Original-recording independence has not been established, so no recording-level confidence interval or generalization claim is inferred from446 frame observations.

## What actually ran

The [frozen complete protocol](../../../experiments/0038-late-memory-readout/FULL_COMPARISON.md) specifies the scientific comparison. Its historical ungranted/1800s launch status was superseded only by the user's [one-operation resource authorization](../../../experiments/0038-late-memory-readout/RESOURCE_EXCEPTION.md). Source commit is `fc12968bb8776cb453f6d0a07e7648afa3957111`; exact-source CI39489941451 succeeded before launch. All47 manufactured tests pass. The failed compressed/batched optimization remains excluded and preserved in [its own result](../../0038-preparation/optimized-qualification-01/README.md).

At each eligible position, the current native history is retained and only the latest noninitial memory write is branched into the three native candidate writes. The four side readouts never write back to the native state. The qualified original implementation uses full attention stacks and selected-native reuse, with no identity shortcut. Early BF16 write averaging and late FP32 stack averaging also differ in rounding; this experiment cannot uniquely isolate a correlation-preservation mechanism.

One complete operation performs admission, model loading, prediction for all446 positions, immutable prediction sealing, scoring under both policies and shutdown. Every original-resolution native prediction matches D35 exactly, and all four complete native final states match. All current predictions are sealed before numerical evaluation-label decoding. No extra acquisition, paid service or GPU execution occurs.

The whole-operation outer monitor records **2515.293119708 seconds (41.921552 minutes)** and peak sampled process-group RSS **4,949,600 KiB**, clean exit0, no timeout, no residual descendants and no cleanup error. This includes all four new readouts jointly and is not per-arm latency or a speedup claim. The one3600s exception is consumed; later operations retain the1800s default.

Logical extra work across438 eligible positions is876 alternative memory encodes,1752 real attention stacks and2190 SAM heads. There are1752 capture-only dispatches,438 selected-native stack/head reuses and zero identity fast paths. Four new logit/mask outputs are saved per position; current archive bytes written are389,398,263. These generated private artifacts are not additional downloaded data and are not uploaded.

Historical costs remain separate: the D35 native prediction used686.546369958 seconds, and D37's complete saved-native comparison used126.866632125 seconds. The current D38 operation reruns native prediction and does not reuse those historical prediction wall times as a current inference measurement. A comparable-budget recurrent SAM2Long baseline has not been executed. No efficiency advantage is established by comparing these different pipelines.

## Independent scoring implementation

The [review source](source/independent-review-01-source.py) was frozen before D38 accuracy inspection. It does not import production metrics or readout code. It independently decodes all446 sealed original-resolution packed masks and their pinned label pixels, recomputes intersection/union under both255 policies, checks the438/110 populations, equal-clip means, all nine aggregates and the frozen decision. Its three manufactured pixel fixtures cover background255, ignored255 and empty unions.

The [review](independent-review-01/review.json) passes **31,128 checks, zero errors, maximum score difference0**. Its separate clean outer costs3.718287709 seconds and160,064 KiB, with no model or network call. It verifies all98 bound source files, relevant seals, native parity/state records and artifact hashes. Older controls remain sealed scalar references. The review does not replay the neural decoder or interpolation and is a separate implementation by the same AI assistant, not independent researcher replication.

The immutable worker receipt correctly says it cannot itself certify clean external exit. The [post-exit decision](final-decision-01.json) binds the finished scientific outer and finished review, admits the complete technical evidence, and records the failed scientific continuation separately. No old receipt is rewritten to imply success.

## Evidence and reproducibility

- [Complete specification](complete-spec-01.json), [outer specification](complete-outer-spec-01.json), [launch admission](launch-admission-01.json) and [source CI](source-ci-01.json).
- [Worker receipt](complete-01/receipt.json), [prediction seal](complete-01/prediction-seal.json), [full evaluation](complete-01/evaluation.json), [outer receipt](complete-supervision-01/receipt.json) and complete stdout/stderr.
- [Review specification](review-spec-01.json), [review receipt](independent-review-01/receipt.json), [review outer](review-supervision-01/receipt.json) and full review report.
- [Manifest](manifest.json):45 bound text records/source snapshots, with both private raw and public transformed SHA256/byte counts. Only literal workspace prefixes are replaced by `$PRIVATE`/`$REPOSITORY`; source snapshots are byte-exact. No media, weight, NPZ, credential or private-data payload is published.

The complete source/runtime/reference closure is recoverable from the pinned repository revision and specifications, but the text bundle alone is not a runnable copy of upstream assets. Obtain licensed upstream inputs separately and satisfy the resource admission contract; the consumed one-time exception is not reusable permission.

The final two-independent-held-out-source substantial J&F gate, uncertainty, component-loss limits, strong complete-cost controls, ablations, novelty distinction and independent replication remain unmet. No manuscript is started.
