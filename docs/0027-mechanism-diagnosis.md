# 0027 saved-result mechanism diagnosis

The fixed manufactured study **failed gates 2 and 4 and is closed**. This note only summarizes the sealed outputs; no estimator, model, fit, root computation or parameter search was rerun. A correct generic formula on exact inputs did not establish the required noisy advantage over the prescribed controls. No epsilon, noise level or scene geometry is changed in response.

## Evidence and denominators

Frozen commit: `7c6144492f6f07247fb603db4bb79e9d1344a853`. The cohort has 138 rows: 4 exact-clean, 4 fitted-clean, 128 noisy (8 groups of 16), and 2 descriptive cases P/F. Every derivative record reports completed. The saved independent audit passed 41,707 checks over 552 arm-rows and 1,762 eligible candidates; it did not rerun solvers or prove root completeness.

All hashes below were rechecked from saved bytes for this diagnosis, relative to `work/0027-development/`:

| Artifact | SHA256 |
|---|---|
| `evaluation-01.json` | `d5b69e1e3119e5ba047aad50d07559b0c7fab4f9765e5757a1effa912fa79f06` |
| `predictions-01/predictions.jsonl` | `5e2e32b210574dc6fa9762bd503a8b9c7317316b8f500964ad38b7ce8fa1a264` |
| `generation-01/inputs.jsonl` | `333db6c91001dd479868cdd07d99f4ce88582619f6695525c58558cbcb678a66` |
| `generation-01/truth.jsonl` | `92ce67b6b64ba1237fb9f6892048652bf3e04bcc5b91b2609f541a69d21a3ec7` |
| `audit-01.json` | `b4094a33c0d9c2769c264233188d9d1f2dac981e33d2181f4b1a3a0ef1d189c7` |

T is the proposed shifted differential solver; V pools four virtual-point step sizes; D8 pools eight physical-point triples; S pools the scale-only SVD and ordered-frame constructions. Selection uses the frozen all-nine bidirectional capped reprojection score, then generation order. Task error is max(rotation, directed translation) in degrees; unavailable outputs incur 180 degrees. Utility AUC5 here is `100*mean(max(0,1-error/5))`, the empirical step-CDF convention, **not** the DMS trapezoid metric.

## Exact algebra passes; finite-neighborhood inputs already change the answer

T's four exact-clean task errors are between `8.15e-15` and `2.02e-13` degrees. With noiseless but fitted derivatives, its task error becomes **0.307362 degrees for G** and **0.066338 degrees for M**, in both calibration regimes. G is rotation-dominated; M is translation-dominated. D8 remains at at most `1.42e-11` degrees on these eight clean rows. Thus there is a finite-neighborhood approximation effect before stochastic perturbations; the saved comparison does not isolate which fitted derivative contributes most.

V has a small exact-clean discrepancy (G about 0.024456 degrees, M about 0.031140 degrees), and fitted-clean errors about 0.331201/0.095633 degrees. This is consistent with its finite virtual-point approximation; it does not justify a new epsilon search. S is exact in the two scale-only exact rows but has 15.507/35.414-degree errors in the shifted exact rows. Handling additive shifts is useful relative to that restricted model, but is already present in the V/D8 controls.

## Noise yields do not support a distinct useful advantage

Each table entry is the selected useful count out of **16**, using the frozen 5-degree threshold:

| Group | T | V | D8 | S |
|---|---:|---:|---:|---:|
| G scale-only L | 16 | 16 | 16 | 16 |
| G scale-only H | 4 | 4 | 2 | 2 |
| G shifted L | 16 | 16 | 16 | 0 |
| G shifted H | 4 | 3 | 2 | 0 |
| M scale-only L | 8 | 8 | 3 | 6 |
| M scale-only H | 0 | 1 | 0 | 0 |
| M shifted L | 8 | 8 | 5 | 0 |
| M shifted H | 1 | 1 | 0 | 0 |

Gate 2 fails G-shifted-H (4/16 versus 8 required), M-shifted-L (8/16 versus 12), and M-shifted-H (1/16 versus 8). Gate 4 fails its improvement clause: T gains only 1 useful row over the stronger V/D8 control in G-shifted-H and 0 in M-shifted-H, below the required gain of 2 in at least one case. Gates 1 and 3 pass; this does not override either failure.

Across all 128 noisy rows, T and V each yield **57/128** useful outputs and their usefulness decisions agree on **126/128**. Among 120 rows where both return eligible outputs, the median absolute difference in task error is 0.023403 degrees (maximum 2.343183). These are descriptive similarities under the fixed finite cohort, not statistical equivalence or a universal numerical identity. Noisy AUC5 is T **29.238564**, V **29.474902**, D8 **20.095113**, S **11.940439** percent.

## Root availability and selection are different issues

T's 128 noisy attempts produce 512 reported quartic roots: 260 complex rejections, 131 corrected-depth rejections, and 121 locally admissible roots. Seven rows have no admissible root; one additional row (93) fails all-nine corrected-depth positivity. The remaining **120 rows each contain exactly one eligible T candidate**. T has 57 selected-useful and 57 oracle-useful rows, so changing the selection among its saved candidates cannot repair its yield failure.

For comparison, V has 57 selected-useful versus 62 oracle-useful rows, D8 44 versus 51, and S 24 versus 24, all out of 128. These oracle counts only describe retained-candidate headroom; they are not deployable results and are never used for admission. In M-shifted-H even the retained V/D8 oracles reach only 1/16 and 0/16. Root-completeness claims are unavailable: the pinned native V/D8 root-compaction limitation remains documented in the evaluation and audit. That limitation weakens broad control comparisons, but does not remove T's independent gate-2 failure or create a positive result.

## P/F delimit this branch, not all possible solvers

In P, T explicitly returns `singular_M_generic_branch` (reported M rank 2, condition about `5.72e17`). V returns 0.020771-degree error and D8 approximately `7.66e-12` degrees. This is a generic-branch coverage limitation, not evidence that the pose is unidentifiable.

In F, T returns `vanished_or_numerically_unresolved_polynomial`; coefficient magnitudes are at most `3.54e-14` against the predeclared `8.49e-11` cancellation guard. V has no retained candidates and D8's two retained candidates are ineligible. S returns 6.828462-degree error. The predeclared flat-gradient example concerns calibration gauge freedom; this one record neither proves universal pose impossibility nor shows a successful gauge-invariant pose recovery. It does not authorize a post-outcome alternative elimination branch.

## Cost and conclusion

The 138-row prediction outer receipt reports **13.715273 seconds**, peak descendant RSS **39,632 KiB**, exit 0; generation/evaluation/audit also complete. Recorded arm solver-plus-scoring sums are T **0.136232 s**, S **0.396884 s**, V **1.124290 s**, D8 **3.639621 s**, with 1,656 native calls entered. Common process sums are 12.641143 s; derivative preparation sums 0.008431 s. These intervals overlap and must not be added. The implementation explicitly includes argument preparation and progress callbacks in solver intervals and all callbacks in process time; there are differing numbers of calls and serialization overhead. These are execution-accounting observations, not an isolated solver speed benchmark or end-to-end vision speedup.

The evidence supports algebraic correctness in four generic exact cases and identifies approximation, noisy-root feasibility and branch coverage limits. It rejects the prescribed useful-yield and comparative-advantage requirements. It does not uniquely attribute noise sensitivity to one derivative, prove impossibility of the wider model, establish real-image performance, or justify a paper-level advance. The fixed proposal remains closed without efficacy retuning.
