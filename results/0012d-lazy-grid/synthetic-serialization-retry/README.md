# 0012d: artificial cost admission passed after a serialization repair

This complete retry passes the fixed artificial cost gate. It admits only the already specified development replay; it does not establish real-data acceleration, pose improvement, novelty, held-out confirmation or paper readiness.

The repair was frozen at `fb68ded6667e94a1c5db09c1db90734991b3750a`. The [original attempt](../synthetic/README.md) failed while serializing a NumPy boolean after its timing calls. Its incomplete JSON and error log remain preserved and are not an admissible result. The documented retry changed serialization and its checks, keeping the fixture, native binary, timing schedule and thresholds unchanged.

## Complete recorded result

The fixed favorable fixture has 10,000 correlated points in one grid cell and 64 models. Model zero has score 5,000; each challenger has score 10,000. All **45 timed calls**, nine per mode in the prescribed cyclic order, and all **five disclosed warmup calls** select model zero with score 5,000. All 128 cell and 128 aggregate lower-bound checks report zero violations.

| Mode | Method | Median primary time (ms) | Candidate / method time |
|---:|---|---:|---:|
| 0 | Full exact | 1.152333 | 0.208996 |
| 1 | Random-order partial | 9.509916 | 0.025324 |
| 2 | Grid-order partial | 0.727750 | 0.330928 |
| 3 | Geometry screening plus partial | 0.740834 | 0.325084 |
| 4 | Lazy geometry and Cantelli plus partial | 0.240833 | 1.000000 |

Every required ratio is below two thirds, so the unchanged 1.5× admission gate passes against all four controls. Mode 4 builds one cached moment summary for 10,000 points, makes 63 geometric and 63 Cantelli calls, and evaluates 10,000 exact residuals. Modes 1–3 each evaluate 332,560 residuals; full scoring evaluates 640,000. These counters are identical across repetitions.

**Random-order timing caveat:** mode 1 includes a separately measured **8.808583 ms** interval covering RNG construction and permutation generation, charged to each call as prescribed. This unusually large startup charge dominates its reported median. Its outer-call median without that charge is **0.701333 ms**, giving a descriptive candidate ratio of **0.343393**. This does not replace the declared gate; it shows that the pass is not dependent on the large startup charge. The mode 1 ratio must not be presented as a general 39.5× speedup. The three other controls also pass independently of that charge.

## Identity, evidence and limits

The native library SHA256 is `198c3a8c8841570c99c467323ea40f9abdddb24564d30922e29940138764e388`. Independent checking matched it across the original incomplete report, preservation record, build receipt, retry report and current binary bytes. All recorded source and protocol hashes also matched. The original incomplete JSON retains SHA256 `62ef022dda5a88f1df5ed2b1ea4f3fcef29bebd5548f151594822c3c54475c1b`, including its unfinished trailing bytes.

- [Complete retry report with all 45 timings](diagnostic.json), SHA256 `143a02b6d086e0a55ca4cb613d3c8cbb970bc9dcc333be80443d43401ea61541`.
- [Execution record](execution.json), recording exit zero, and [run log](run.log).
- [Original failure preservation record](../synthetic/preservation.json) and [incomplete original JSON](../synthetic/diagnostic.json).
- [Frozen prospective protocol and repair scope](../../../protocols/0012d-lazy-grid-screening.md).

The fixture deliberately makes signed moments useful while geometric intervals remain weak. It tests overhead feasibility in that favorable case, not the distribution of real matches. Numerical checks are observed containment evidence, not a formal floating-point certificate. No real coordinates were read during this result audit, and no timing was rerun. The conditional development replay must retain its separate frozen speed, correctness and scope gates.
