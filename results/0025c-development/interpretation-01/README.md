# Saved-only interpretation of the completed 0025c noise falsifier

**Decision unchanged: the four-point initializer is closed under the frozen contract. No robust-stage or rendered-image experiment is supported by these results.** This note only aggregates the 132 sealed case JSON files and their 13,332 saved solves. It imports no production solver, performs no fit/SVD, generates no perturbation, and reads no model, image or real dataset. It is descriptive analysis after the result, not an amended gate or manuscript.

The six D4 gate groups contain respectively G/P 7, G/S 6, N/P 12, N/S 13, C/P 0 and C/S 0 useful hypotheses out of 1,120; every group required 144. The known lower-sample A2 control also dominates the prospective sampling comparison. Subsets share observations and are not independent evidence.

## Eight points do not reveal a useful unrestricted-depth alternative

The table reports saved **failure-inclusive median directed translation error** in degrees for the angular-input-stress regime S. All eight-point arms see the same noisy observations and supplied angular inputs.

| Case | D8: affine depth + free readout velocities | N8: depth-free + free readout velocities | A8: angular-only affine | D8 useful /16 | A8 useful /16 |
|---|---:|---:|---:|---:|---:|
| G | 24.11 | 173.30 | 3.68 | 0 | 12 |
| N | 57.65 | 6.48 | 2.06 | 1 | 13 |
| C | 60.09 | 159.80 | 9.45 | 0 | 5 |

Across G/N/C and both P/S regimes, D8 beats A8 in **0/96**, ties in 2/96 (both failed), and loses in 94/96. Its 2/96 useful outputs contrast with A8's 61/96. D8 can beat N8—for example 12/16 C/S cases—but this is comparison against another unstable arm, not useful headroom over the strongest control. In critical E, D8 beats A8 only 4/16 in each regime; this does not reopen an excluded critical group or rescue a failed gate.

## The failure is predominantly wrong finite solutions

Across the six primary D4 groups, 4,328/6,720 outputs pass all physical/numerical checks, but **4,290 of these have error above 5°**. The other 2,392 fail affine-scale/depth positivity. Only five invalid outputs have a saved raw direction within 5°; only 111 failures are specifically positive on the fitting subset but not on the full pool. Thus the observed deficit is not explained by the full-pool positivity check. No constraint was relaxed for this description.

D8 has 84 valid outputs among the 96 primary noisy cases; 82 of those are wrong by over 5°. Its 12 invalid cases have no hidden raw useful directions. N8 has 18 unresolved-cheirality failures and 54 wrong-but-valid outputs among its 96 cases; 32 valid directions exceed 90°. Cheirality majority is a physical selection rule, not assurance that the selected noisy model is correct. Folding the angle would conceal part of the task failure and was not done.

The G/N/C D4 and D8 noisy matrices all retain rank 12 even at the fixed weak-rank diagnostic threshold. Their saved task-null maps are zero because no nullspace exists at that threshold; this is **not** a bound on finite errors-in-variables sensitivity. Clean D4 solutions are accurate (maximum G/N/C errors below 4e−9°), while their clean equilibrated condition-number medians are approximately 1,369, 1,736 and 1,130. Clean N8/C is especially sensitive: condition number about 5.08 million despite exact clean recovery. These values depend on the declared column equilibration and are not universal physical condition numbers.

## What the saved nuisance estimates support—and do not prove

D8's saved algebraic residual is no greater than A8's in all 96 primary noisy cases (median residual-norm ratio 0.773), yet its task error is never lower. D4 commonly interpolates its 12 equations to residual norm around 1e−14 with a wrong direction. In S, valid D4 estimates have median maximum readout speed about 291, 1,675 and 157 input-depth-units/s for G/N/C; the actual manufactured maximum is about 0.969. D8 medians are about 82, 502 and 91. This comparison is meaningful for the affine arms because they retain the fixed `a1=1` gauge; absolute N8 speed or translation magnitude is not comparable because its scale is arbitrary.

The scale-invariant `0.04*max(||v1||,||v2||)/||t||` also inflates: D8/S medians are 4.09, 32.84 and 6.69 against truth 0.0754, 0.0754 and 0.3335. A8 accepts a small clean approximation error (about 1.03°, 0.82°, 0.79°) and performs much better with the declared noise. This pattern is consistent with free velocity/offset parameters absorbing observation error. It does not isolate the separate effects of depth noise and pixel noise, prove a particular regularizer works, or establish a new motion-model contribution. P already fails with exact angular inputs, so S's small angular perturbation is not necessary for the observed failure.

Critical E illustrates why parameter rank and task rank differ. Clean D8 has rank 9, yet its saved translation-functional null norm is only 1.10e−15: the rejected freedom is nuisance in that configuration. Most clean D4 subsets have the same property; one additional subset degeneracy reaches translation. N8/E has a genuinely nonzero task-null map. Under perturbations, numerical rank can increase without creating trustworthy information. These facts explain the conservative rejection policy; they do not authorize selecting a new estimator or threshold using these outcomes.

The remaining research question would require a substantively justified source of motion information or a separately defended physical model, with independent priors and fresh prospective tests. Simply using eight points, weakening positivity, adding more seeds, or selecting a favorable subset is unsupported here. The present saved results cannot establish whether any proposed trajectory coupling, sensor prior, or robust pipeline would succeed.

## Reproducibility of this description

`../interpret_saved.py` SHA256 `03a25a8791689f72edaffbe30d982a3a39676d91b61ecae78b001cffd831d714` rechecks the sealed hash of each consumed case and writes `summary.json` SHA256 `02cf08f2cba163a95ce44fc71005813960c90424f90cd0c315101e02c07a1302`. It completed once in 0.9955 seconds; the local receipt records zero fits, new perturbations, model calls and production imports. The JSON preserves every case hash and all group decompositions. Its nearest-rank descriptive p90 is explicitly distinct from the frozen scientific metrics. It is not a replacement for the separate frozen independent numerical audit.
