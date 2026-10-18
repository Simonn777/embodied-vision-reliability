# 0008 publication verified

2026-10-18. The failed frozen training-quality result and independent review are
published through [PR 33](https://github.com/Simonn777/embodied-vision-reliability/pull/33).
The parent checked the remote PR identity, exact commit CI, merge and main CI.

| Check | Observed result |
|---|---|
| Reviewed PR head | `2f1a67063144042b89c0614803c8e1acac1a98ca` |
| PR CI | [39166937335](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39166937335), success at that head |
| PR tests | 576 passed, 31 skipped, 156 subtests passed in 21.05 s; 1,240 smoke frames in 3.38 s |
| Merge | `a915553fd63573df28661868a7c573e7fe2d3b76`, 2026-10-18 21:16:11 UTC |
| Main CI | [39167082741](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39167082741), success at the merge |
| Main tests | 576 passed, 31 skipped, 156 subtests passed in 24.33 s; 1,240 smoke frames in 3.78 s |

The CI environment lacks Torch; its 31 skips are disclosed, not counted as
tensor execution. The separate pre-execution cloud tests ran 51 tests with zero
skips. Both sets check software, not a new quality result. The frozen scientific
decision remains **stop this candidate**. Original result receipts and the
parent-review layer are preserved; their statements that publication was pending
describe the time those immutable artifacts were recorded. This later document
records completion without rewriting them.

No training, inference, formal score or private-array replay was repeated during
PR handling. No paid service, new data or model acquisition occurred.
