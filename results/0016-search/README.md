# 0016 result: tested search leaves no useful candidate-selection headroom

**NO_GO for the proposed ordinal selector on this four-candidate bank.**
The strongest current deployed full-four selector has AUC@5=78.2293102381%.
The best current four-candidate oracle reaches78.2587614824%: only
**0.0294512443percentage points** above it, far below the predeclared2pp necessary
condition. An oracle is not deployable; its small optimistic gap is sufficient
to reject this particular selection opportunity. No method improvement,
new contribution, independent generalization or manuscript result is claimed.

## Execution and bindings

The complete scientific source/configuration freeze
`055ea70bdfba2a7a3c284063ab91dd1ecb624c27` was published in
[PR45](https://github.com/Simonn777/embodied-vision-reliability/pull/45) before the
run. [Exact-head CI39312932830](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39312932830)
succeeded. Local56 artificial tests and six manufactured native calls preceded
execution. See [protocol](../../protocols/0016-search-headroom.md),
[allocation](../../configs/0016-development-manifest.json) and
[library preparation](../0016-preparation/README.md).

All64allocated development pairs,192workers and768native calls completed once,
with zero estimator failures or retries. The outer prediction operation exited0
in227.078419958seconds; sampled maximum worker RSS43,040KiB. The192records were
sealed before the evaluator opened any selected R/t. The single evaluator
exited0 in0.570092167seconds with empty stdout/stderr. No prediction was rerun.
All64pairs remain in every reported metric; no hard pair was replaced.

- Prediction seal SHA256:
  `e6313a51f074b0b0d9dc97cb12b4ccebc6d897a3388f8722bb192118a2794f7e`.
- Evaluation SHA256:
  `d3e87cc01742e2cd02f36b055d5f92ab67e6764edd173c380bd144a0da778dc5`.
- [Complete scores, candidate poses and choices](evaluation.json),
  [prediction seal](predictions/seal.json), [actual outer prediction receipt](prediction.process.json),
  [evaluator receipt](evaluation.process.json).

## Accuracy and compute

All values below use directed translation angle, max(rotation,translation), and
the full64-pair denominator. Sign-folded secondary metrics and every fixed-budget
prefix remain in evaluation.json. The common selector rescores final poses with
pixel F-Sampson plus cheirality; native pre-refinement scores never select seeds.

| Backend | Seed0 AUC@5 % | Full4 AUC@5 % | Oracle4 AUC@5 % | Full4 AUC@10 % | Full4 AUC@20 % |
|---|---:|---:|---:|---:|---:|
| Official2.0.5 normalized |77.841095599|77.838846439|77.856982388|87.497584558|93.694950296|
| Current3.0.0 normalized |78.228357935|78.229310238|78.258761482|87.693598118|93.793270650|
| Same current3.0.0 TS |77.206555321|77.194269424|77.218281131|86.393881828|92.561092075|

| Backend | All64 seed0 charged time, s | All64 full-four charged time, s | Full-four native calls only, s |
|---|---:|---:|---:|
|2.0.5 normalized|13.754562386|54.325587510|53.787468373|
|3.0.0 normalized|13.873699625|55.238035830|54.707264910|
|3.0.0 TS|22.502700588|88.396200166|87.867625833|

Full4 costs include all four candidates even when an earlier seed wins. Its AUC@5
change versus seed0 is−0.002249159pp,+0.000952303pp,and−0.012285898pp,
respectively. The small version change is an official-baseline observation, not
our contribution; old/current builds and default final losses differ. Norm/TS
share one current binary. Charged work excludes imports/JSON/supervision and all
cached feature extraction. This is not a complete-pipeline speed comparison.
Budget-prefix replay is offline, not an interruptible deadline implementation.
The selected candidate's completion timestamp is not the total selector cost.

## What the result does and does not imply

Current normalized seed0 has61/64errors below5degrees,62/64below10degrees, and
all64below20degrees. Translation dominates rotation in58/64pairs. Its three errors
above5degrees are at pair indices6,54,61; the corresponding translation errors
are5.236061,15.974229,10.080019degrees. Across four seeds their maximum translation
dispersions are about0.000000854,0.021948342,and0.000001207degrees. Thus the
tested restarts do not produce substantially better alternatives even for these
larger errors. The result does **not** identify input bias as their cause.

Current TS introduces an additional large error at index4 (rotation18.925255deg)
and is worse overall here. That is a disclosed strong-control outcome, not a
reason to compare a new method only against the weaker arm. Median candidate
translation dispersion is near numerical angle resolution in all three banks.

Stop spending on further seeds or ordinal selection of this bank. A later
proposal must introduce useful additional information or genuinely different
hypotheses, and compare against current normalized PoseLib and relevant strong
depth methods. Generic RePoseD/MADPose calibration or hybrid fitting is already
known. Qualifying an additional cached-depth input is a possible next diagnostic,
not an admitted novel method or an automatic continuation of the failed selector.

These64pairs share a single landmark and may share photographs. No scene-level
population interval or held-out transfer claim is supported. Scene0022 remains
numerically unread; no other dataset, image or model inference ran. The research
goal remains active, and the substantial two-source accuracy/time/novelty gate
required before writing a paper is unchanged.

[Independent saved-only review](independent-review/README.md) verifies all192 worker/process bindings, all768 candidate errors, all deployed choices and complete64-pair metrics. Maximum angular disagreement is6.96e-11degrees. Twenty oracle-only seed identities differ under scalar reduction within3.30e-11degrees; the exact stored-error argmin remains correct and the failed gate is unchanged. The initial strict audit failure and audit-only clarification are preserved. No production source or scientific result changed; seven artificial audit tests pass.
