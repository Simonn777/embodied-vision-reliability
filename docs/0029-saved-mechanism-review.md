# Attempt 02: saved scalar mechanism review

**Disposition: the fixed 0029 variant is closed.** Its technically complete
development run failed the frozen opportunity gate. This review reads only
`work/0029-native-development-02/evaluation/receipt.json` (606,359 bytes;
SHA256 `1d8ae1ffb4f1e17200e1f8e30d80906dd742591fdf6a0c6d73457247a1392819`).
No prediction arrays, GT masks, models or other scientific files were opened;
no fitting, selection changes or reruns were performed. This is descriptive
interpretation of saved scalars, separate from the independent audit.

The receipt declares commit `098dee64878b5ef33edbac12223f86da176e091e`,
completed/eligible, 48 evaluated queries from six videos, and zero model
calls in evaluation. Its recorded scientific source hashes are:

- `native_video.py`: `4af3e0e5b26327d4ecb95b4644b94e13c257a38c928dbb09f23bd95151001d0a`
- `diagnostic_core.py`: `4c76bdb492bef0507b4322be67a2a157ec34c420bfacb874388b1f7facef0c34`
- `evaluate_queries.py`: `272ae306b0f80573d3350d892cefbc13cf9df5a06d89b6188ee9c86584c5237f`

These bindings are reported from the receipt, not re-established by opening
source or upstream artifacts in this review. Historical attempt 01 remains
technically ineligible; none of its failures or costs are erased.

## What failed

All 48 rows remain in the denominator. Sparse-evidence coverage passed:
38/48 actionable queries, with at least one in all six videos. The signed
selector's pooled foreground J was **93.923162%**, versus baseline
**93.935668%** and the strongest pooled control, `max_predicted_iou`,
**93.970921%**. Thus it lost **0.012507 percentage points (pp)** to baseline
and **0.047759 pp** to that control.

Even the nondeployable best of the admissible baseline/D1/D3 candidates
reached only **94.012489%**: **0.076821 pp** above baseline, versus the
prespecified **2 pp** information-opportunity requirement. Allowing deletion
at the ten nonactionable queries raises the descriptive unrestricted oracle
gain to only **0.125031 pp**. Missing witness coverage therefore does not,
by itself, account for the much larger unmet opportunity requirement.
Neither oracle is a deployable selection result.

The gate's coverage and bounded per-video harm clauses passed; candidate
information, pooled selection and four-positive-video clauses failed.
Signed selection beat the strongest control for **0/6 videos**. This is not
an all-clauses failure, and the worst margin **−0.229683 pp** is not a
2.29683 pp loss.

Each video below has exactly eight evaluated queries. Margins and oracle
gains are J percentage points, not relative percentages.

| Video | Actionable | Baseline J % | Admissible deletion oracle gain | Signed − baseline | Strongest per-video control | Signed − that control |
|---|---:|---:|---:|---:|---|---:|
| dogs-scale | 3 | 92.186300 | +0.066472 | −0.013516 | max_unsigned_change | −0.079988 |
| upside-down | 8 | 95.567608 | +0.201413 | −0.012006 | max_predicted_iou | −0.199871 |
| dog-agility | 8 | 93.934808 | +0.060609 | −0.023735 | always_D3 | −0.070738 |
| cat-girl | 8 | 94.753082 | +0.080107 | −0.026759 | always_D3 | −0.054005 |
| planes-water | 8 | 95.775013 | +0.011738 | −0.002398 | always_D3 | −0.011710 |
| dog-gooses | 3 | 91.397199 | +0.040586 | +0.003374 | reset_direct | −0.229683 |

## Mechanism observations

**The interventions were not empty operations.** All 96 D1/D3 candidate
masks differed from baseline. The signed selector chose D1 on 25 queries,
D3 on three, and baseline on 20. Those 20 final-mask no-ops comprise the ten
nonactionable queries plus ten actionable queries where the signed rule
retained baseline. All 28 actions changed the final mask. Thus the result
cannot be explained by every branch silently returning the original mask.
Small possible gains also do not establish that memory is generally useless:
they describe these two fixed deletions and these six high-baseline videos.

**The signed sparse loss did not reliably order whole-mask J.** Across the
38 actionable queries, D1's loss-gain/J-gain signs agreed on 17 and disagreed
on 21; D3 agreed on 19 and disagreed on 19. Among the 28 selected actions,
all of which reduced the signed witness loss by construction, **13 improved
J and 15 worsened J**. The remaining ten actionable selections were exact
loss/J no-ops. These are exact-sign, paired descriptive counts; small effects
and correlated queries preclude treating them as an independent statistical
test of a general causal claim.

No selected action caused the prespecified strictly greater than 0.05 J
drop. The largest observed signed-action drop was **0.0040394021 J
(0.403940 pp)** at cat-girl frame 88. Absence of a large harmful event is
not evidence of a useful positive effect.

**Transferred labels were imperfect, particularly positives.** All retained
labels were known/nonvoid in the saved evaluation: 0/2,308 void labels.
Positive precision was **664/772 = 86.0104%**; negative precision was
**1,522/1,536 = 99.0885%**. These are retained-point totals, not equal-video
averages or independent samples.

| Video | Correct/retained positive | Correct/retained negative | Signed actions improving / worsening J |
|---|---:|---:|---:|
| dogs-scale | 34/42 | 254/256 | 1 / 1 |
| upside-down | 198/206 | 250/256 | 2 / 1 |
| dog-agility | 18/77 | 256/256 | 3 / 5 |
| cat-girl | 149/170 | 252/256 | 4 / 3 |
| planes-water | 245/256 | 254/256 | 2 / 4 |
| dog-gooses | 20/21 | 256/256 | 1 / 1 |

Incorrect positives occurred in 33/48 queries, including 31/38 actionable
ones; incorrect negatives occurred in 10/48, including 8/38 actionable
ones. This makes witness contamination a plausible limitation, not an
isolated causal explanation. Five actionable queries had entirely correct,
nonvoid retained labels in both classes; four led to signed actions, and one
of those still reduced whole-mask J. Correct sparse membership labels alone
therefore did not guarantee the correct direction for the unobserved mask.
The record cannot distinguish feature mismatch, occlusion, changed material
membership, spatial support or calibration errors as unique causes.

The other fixed uses of the same sparse evidence were also below baseline:
direct prompt J **93.541852%**, signed prompt **93.698850%**, direct reset
**86.918558%**, signed reset **92.906901%**. The deployed native-bank selector
scored **91.412392%**; its nondeployable admissible/unrestricted oracle mean
gains were **0.300947/0.737028 pp**. These controls do not justify promoting a
post hoc bank, new rule or altered threshold after the fixed gate failed.

## Cost and limits of interpretation

The six saved attempt-02 video outer times sum to **882.052503 s**
(14.700875 min); maximum sampled group RSS was **4,986,784 KiB**.
Recorded completed calls total 386 baseline/image-encoder frames and 606
track steps: 386 baseline + 48 verification + 96 deletion + 76 prompt/reset.
Native mask-bank capture added zero calls. Query-only named stage totals
(e.g. 268.417035 s method forks and 102.534114 s verification) are components
or overlapping descriptions, not extra wall times to add to the outer total.
The receipt explicitly excludes separately bound frontend/acquisition
costs from that total; historical failed-attempt costs remain separately
preserved. No matched complete-pipeline speed claim follows.

This is a custom fixed-query foreground-J development diagnostic, not
official DAVIS J&F, an independently held-out two-source result, or evidence
that all counterfactual memory diagnostics are impossible. It specifically
rejects proceeding with the frozen D1/D3 signed-loss variant on the basis of
this probe: the candidate family had insufficient useful headroom, and the
deployed selector did not recover a positive advantage over fixed controls.
No threshold, query, sequence, memory-age, witness or candidate-bank rescue
is proposed from these outcomes.
