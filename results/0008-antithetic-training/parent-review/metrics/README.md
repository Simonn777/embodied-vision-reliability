# Independent parent review of 0008 public metrics

**Public arithmetic and receipt-link review: PASS. Scientific decision: STOP.**
No blocking arithmetic or gate inconsistency was found. This review reads the
published Git objects at `70bc57fd83634a8df66aecf072ea9b526e3af043`, not mutable
new result files. All 13 frozen inputs also match their scientific-source,
canonical-freeze, execution-main and current-worktree bytes.

| Recomputed criterion | 500 updates | 5-second training endpoint |
|---|---:|---:|
| Mean paired gain versus per-case best control (dB) | -0.000955546016498 | -0.330065287859 |
| Positive image-token means out of four | 1 | 0 |
| Worst image-token mean (dB) | -0.419613749262 | -1.08928174622 |
| Pooled first-difference error ratio | 1.02592786169 | 1.04786258441 |

At both endpoints, the frozen +0.5 dB, at-least-three-positive-image-means and
no-image-worse-than −0.2 dB gates fail. First-difference gates pass. The wall
endpoint allocated-cost ratio is **0.9957494786484813**, passing the ≤1.10 gate;
this does not override the failed quality requirements. No paper or independent
scene generalization claim is supported.

The reviewer reconstructed every pixel MSE/PSNR from the 64 published block SSEs,
verified all 4,096 block bounds/counts, checked the directional and pooled
first-difference arithmetic, reselected the fixed-order per-case oracle controls,
and recomputed both endpoint decisions and all cost totals. It cross-checked
public score rows against the 32 receipts, saved execution/log identities,
original-output hash declarations and timing table.

There are **32 shared trajectories and 64 correlated endpoint outputs**, not 64
independently timed training runs. Actual optimizer updates total **98,994**;
64 inference receipts total **48,000 forward calls** and **2,333,811,200 normal
scalar draws**. Case outer walls sum to **260.0609781540006 s**, the inner study
clock is **260.10785466699963 s**, and the actual archived study outer wall is
**260.1861582160018 s**. Endpoint allocated outer costs subtract only later
training plus the other endpoint's separately measured write/inference/save
work. Their sums are not additive consumed compute. Every 5-second snapshot
completed in **[5.000254870999925, 5.0034228559998155] s**.

## Scope and reproducibility

This is an independent **public-summary algebra** review, with standard-library
code only. The private pixel/reference/prediction/weight arrays are unavailable
locally and were not opened; pixel-level correctness cannot be newly established
from non-reconstructable SSE summaries. The separate archived cloud saved-array
replay reports 45,770 checks / 5,287 numeric comparisons and maximum discrepancy
7.105427357601002e-15. It is identified as an archived cloud result, not claimed
as a raw-array replay performed by this parent reviewer. Actual cloud tool-return
observations and full publication provenance are separately reviewed.

Run from any local clone containing the published commit:

```sh
python3 review_public_metrics.py --repo REPOSITORY --report NEW_REPORT.json
```

The script uses Git read commands and creates only the requested exclusive report.
It imports no production code, NumPy or Torch, calls no model, contacts no cloud,
and acquires no data. Actual local execution returned **exit 0** on Python 3.9.6,
in **1.623496709 s**, with **89,135 checks / 1,609 numeric comparisons** and maximum
absolute difference **1.0658141036401503e-14** (rtol 1e-10 / atol 1e-12).
AI-assisted review; no institutional endorsement or manuscript claim.
