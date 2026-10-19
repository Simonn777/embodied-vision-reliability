# 0012d development: lazy grid screening failed the speed gate

**The real-development speed gate failed.** The primary method reduced exact residual evaluations by 20.08% against random-order partial scoring, but was **31.21% slower**. It was slower than both partial-scoring controls on all 16 pairs. The favorable artificial result did not establish a useful real-data speedup.

This result does not admit pose-estimation integration, a paper claim, or further tuning of this bound-screening family in this research run. The research objective remains unmet.

## Frozen workload and five-method result

The conditional replay ran once, with exit zero, from revision `fb68ded6667e94a1c5db09c1db90734991b3750a`, after the [artificial admission and serialization repair](../synthetic-serialization-retry/README.md). It retained all original 16 development pairs, 10,000 correspondences and 64 saved hypotheses per pair. No hypothesis resampling or pose evaluation occurred.

All five methods ran three fixed cyclic repetitions per pair. Primary times below sum the per-pair median outer wrapper-call times, including input checks and required native preparation. Measured shuffle setup is charged to mode 1. Residual totals count one call per pair; counters are identical across its three repetitions.

| Mode | Method | Exact residual evaluations | Primary time (ms) | Mode 4 / method time |
|---:|---|---:|---:|---:|
| 0 | Full exact | 10,240,000 | 18.543418 | 0.924438 |
| 1 | Random-order partial, block 256 | 5,413,920 | 13.064877 | 1.312087 |
| 2 | Grid-order partial | 5,452,576 | 14.714626 | 1.164980 |
| 3 | Lazy geometry plus grid-order partial | 5,445,664 | 16.121164 | 1.063338 |
| 4 | Lazy geometry and Cantelli plus partial, primary | 4,326,688 | 17.142249 | 1.000000 |

The [predeclared gate](../../../protocols/0012d-lazy-grid-screening.md) required mode 4 to be at least **1.5× faster than every control**, corresponding to each ratio being at most two thirds. All four aggregate comparisons fail. The small 7.56% time reduction against full scoring does not satisfy that gate or outperform the stronger partial baselines.

Mode 4 is faster than full scoring on 11/16 pairs and geometry screening on 4/16. It is faster than neither partial control on any pair. No pair reaches the 1.5× threshold against any control. All pair-level results are retained in the [complete table](all-pairs-and-modes.csv).

## Correctness and identity checks

All **240 timed calls** return the required saved winner index and satisfy the declared score tolerance, `1e-8 + 1e-12 * abs(reference)`. The maximum native/NumPy score difference is `7.275957614183426e-11`. There are **zero observed violations** in 69,376 eligible-cell lower-bound checks and 2,048 aggregate checks, covering geometry-only and combined lower bounds.

Independent saved-result review verified every pair's schedule, medians, counters, winner and score; the original pair keys, hypotheses and reference scores are unchanged. Source, protocol, artificial-admission, reference, build-receipt and binary hashes matched their recorded identities. Numerical checks remain observed evidence, not a formal floating-point certificate.

- Native grid source: `8b00a7e01a8e9e149881419c7299b1aa903da35c7273cb08b8365a4e2a639814`.
- Included frozen residual kernel: `106851f8b1dcb939d2fdaa41430b967bc4fa4d95056ace9805a2ea75bd46fd8d`.
- Replay driver: `f2472384e8a0fa57c7557dbaeb4e98151e83c7686e168ab2ecda325851dd4617`.
- Unchanged native binary: `198c3a8c8841570c99c467323ea40f9abdddb24564d30922e29940138764e388`.

The JSONL start record retains the complete source and build identities. The kernel uses one thread and the same strict compiler settings as artificial admission.

## Artificial versus real behavior

The deliberately favorable one-cell fixture gave a candidate/grid-partial time ratio of **0.330928**. The real-development aggregate ratio is **1.164980**. This comparison uses the grid-order control, so it is unaffected by the artificial random-control RNG startup anomaly.

On real pairs, mode 4 constructs moments for **535 of 542 eligible cells**, covering **151,593 of 160,000 points** (94.75%). It makes 21,865 geometry calls and 21,746 Cantelli calls. Most moment construction therefore remains necessary in the actual execution, while residual savings are only 20.08% against random partial and 20.65% against grid partial. These counters describe the observed tradeoff; they do not isolate the runtime of every bound kernel.

The additive profile below selects each pair's actual median-primary call, rather than combining phase medians from potentially different repetitions:

| Mode 4 component | Time (ms) |
|---|---:|
| Native setup | 1.284834 |
| Native scoring, including lazy moments and bounds | 15.514832 |
| Outer wrapper/destruction gap | 0.342583 |
| **Total** | **17.142249** |

Setup became much cheaper than in 0012c, but the remaining scoring work still costs more than the entire random-partial baseline. Its 1.5× gate would require a total at most 8.709918 ms. The passing artificial case is retained as a favorable-case observation, not substituted for this failed development test.

## Evidence and stopping decision

- [Complete JSONL with all 240 calls](diagnostic.jsonl), SHA256 `0e7d8d74be78e39f674cd65d604291459b23108689b930dd4c1a9b58bd61f8a2`.
- [Machine-readable summary](summary.json), [all-pair/mode table](all-pairs-and-modes.csv), and [summary script](../../../experiments/0012-score-bounds/summarize_grid.py).
- [Execution record](execution.json) and [run log](run.log).
- [Artificial retry evidence](../synthetic-serialization-retry/README.md) and [preserved original serialization failure](../synthetic/README.md).

This is repeated exploratory development on one scene, not held-out confirmation. No poses, labels or held-out coordinates were evaluated. The independent audit used saved results and hash-only source/build reads, without data reads or a scientific rerun. The decision is to stop this method family here rather than relax the gate, select favorable pairs, or continue same-family tuning. No official pose integration follows from this result.
