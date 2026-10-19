# 0012c: native verification failed the speed gate

The primary hybrid reduced residual evaluations but was **6.45× slower than partial exact scoring** and **4.54× slower than full scoring**. This result does not justify pose integration, a paper claim, or completion of the research goal.

The implementation and [prospective protocol](../../protocols/0012c-native-scoring.md) were frozen at commit `ff306864820de0228453f049321306617f8ae049` before this single development replay. It reused the original 16 pairs from scene 0015, with 10,000 correspondences and 64 saved hypotheses per pair. Each of five methods ran three fixed cyclic timing repetitions. The native C++17 kernel is single-threaded, compiled with `-O3`, `-fno-fast-math` and `-ffp-contract=off`.

## Five-method result

Primary time is the sum of the 16 per-pair median outer wrapper-call times, including input checks and native setup, plus shuffle preparation charged to modes 1, 3 and 4. Residual counts represent one call per pair; they are identical across the three repetitions. Repeated leaf and fallback evaluations are included.

| Mode | Method | Residual evaluations | Primary time (ms) | Mode 4 / method time |
|---:|---|---:|---:|---:|
| 0 | Full exact scoring | 10,240,000 | 18.543251 | 4.538999 |
| 1 | Original-order partial exact, block 256 | 5,413,920 | 13.052085 | 6.448609 |
| 2 | Moment bounds with full exact fallback | 2,956,863 | 83.793917 | 1.004462 |
| 3 | Matched-order partial exact | 4,696,864 | 73.316543 | 1.148005 |
| 4 | Moment bounds with partial exact fallback, primary | 2,018,367 | 84.167793 | 1.000000 |

Mode 3 pays for the same coarse-bound incumbent and hypothesis ordering as the bound methods, without bound rejection or refinement. Mode 4 uses 62.72% fewer residual evaluations than mode 1 and 57.03% fewer than mode 3. That work reduction does not translate into a runtime gain: mode 4 is slower than modes 0, 1 and 3 on every pair.

The predeclared speed gate required mode 4 to be at least **1.5× faster than each of modes 0, 1 and 3**, or a time ratio at most two thirds. All three aggregate comparisons fail. The result is `numerical_speed_gate_passed: false`.

## Numerical checks

All **240 timed calls** returned the saved reference winner index and satisfied the declared score tolerance, `1e-8 + 1e-12 * abs(reference)`. The largest native/NumPy score difference was `7.275957614183426e-11`.

There were **zero observed violations** in 131,072 initial cell checks and 1,024 aggregate checks. These checks do not certify every refined descendant or every floating-point input. Native and NumPy reductions are not claimed to be bit-identical. No held-out confirmation or pose-accuracy evaluation was performed.

## Where the measured time goes

The following additive breakdown selects the actual median-primary call for each pair and sums its components. This avoids adding independently selected phase medians from different repetitions.

| Mode 4 component | Time (ms) |
|---|---:|
| Native setup | 44.248539 |
| Native scoring | 38.228333 |
| Outer wrapper/destruction gap | 0.357046 |
| Charged shuffle preparation | 1.333875 |
| **Total** | **84.167793** |

Setup includes input checks, matrix normalization, spatial splitting/sorting, moment construction and initial partitioning. Scoring includes coarse bounds, refinement, residual evaluation and selection. The saved timers do **not** isolate sorting from moment construction, or coarse-bound evaluation from other scoring costs.

The independently computed phase-median sums in the summary are also valid: 44.132205 ms setup and 38.140208 ms scoring. They are not an additive decomposition of the primary median total.

Even eliminating the entire selected-call setup phase would leave 39.919254 ms, still slower than the partial baseline. Meeting the 1.5× gate against that baseline requires at most 8.701390 ms in total. Changing only the tree data structure is therefore insufficient evidence for a useful next implementation; both preparation and bound-evaluation costs need to be addressed prospectively.

## Evidence

- [Complete JSONL, all pairs and all timed calls](diagnostic.jsonl), SHA256 `b2ee2c1af070f79e2a454f142886a1c692245beb00170581543c3beb20b8e857`.
- [Machine-readable summary](summary.json) and [all-pair, all-mode table](all-pairs-and-modes.csv).
- [Execution record](execution.json) and [run log](run.log).
- [Summary script](../../experiments/0012-score-bounds/summarize_native.py), [native implementation](../../experiments/0012-score-bounds/native_score.cpp), and [replay driver](../../experiments/0012-score-bounds/run_native.py).
- [Original saved hypotheses and scores](../0012-development/diagnostic.jsonl), SHA256 `c5f36cc3cad8598238d48ba5debb663cb03cf8e4def4b73510734fb44ddbd683`.

The JSONL start record retains source/build hashes and the timing convention. Public build-receipt paths are normalized; the raw local receipt remains separate. The 16 pairs reuse one development scene and some images. Three timing repetitions on that workload are not independent confirmation. This directory records a failed speed test without reclassifying it as research success.
