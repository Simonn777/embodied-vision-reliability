# Whole-mask selection is insufficient for the declared Jtr target

The best possible GT-assisted choice among the four existing real arms reaches **55.0372302% J / 33.1352595% Jtr**. Against the strongest fixed arm (union), these are **+6.4367408 / +1.7969411 pp**. J clears the necessary 3-point condition; Jtr does not. A more elaborate selector limited to these same four complete outputs cannot exceed this bound on these saved predictions. This closes that action class as sufficient for the declared target; it does not fail the earlier positive support screen or prove any finer action succeeds.

| Primary action | J (%) | Jtr (%) |
|---|---:|---:|
| Native B | 44.3324994 | 24.1167225 |
| Strongest fixed real arm: union | 48.6004894 | 31.3383185 |
| Per-clip constant GT-assisted choice, separately optimized per metric | 52.8782426 | 32.1928043 |
| Per-frame GT-assisted choice of four whole arms | 55.0372302 | 33.1352595 |
| Privileged pixel support O | 60.4942916 | 35.6775156 |

The per-clip constant J and Jtr columns can use different arms; they are separate bounds, not one common deployable policy. The per-frame choice uses the same raw frame J criterion and fixed tie order for both metrics. O additionally removes false-positive pixels and assigns true identities, so its remaining **5.4570614 J / 2.5422560 Jtr pp** above whole-arm selection is not available merely by choosing another existing mask.

| Complete clip | Best-per-frame J (%) | Best-per-frame Jtr (%) |
|---|---:|---:|
| 4691_cut_guava | 60.5924711 | 22.5258786 |
| 817_cut_onion | 38.4439184 | 22.9894863 |
| 9675_fold_clothes | 81.3668462 | 66.6545959 |
| 1934_tear_film | 39.7456851 | 20.3710773 |

Void-ignore sensitivity gives **56.7930966 / 33.8228215%** for the per-frame bound, only **+6.9883543 / +1.8747078 pp** over its strongest fixed arm; it also fails the Jtr clause. No sensitivity result rescues the primary decision.

The complete calculation uses all446 already exposed frame rows of the fixed four single-object clips. Primary means use438 interior and110 tail positions, with equal sequence-object weighting. All six original arm vectors, choices, constant-arm choices, per-clip summaries and both policies are retained. No media, model, new labels or pixel metrics were read or computed. The source and exact original evaluation JSON were hashed before/after. One clean complete supervised operation finished in **0.190451334s**, peak sampled RSS **4,464 KiB**, exit0, no residual descendants or cleanup errors. Four manufactured tests and exact-source CI passed before execution.

Independent exact-rational [saved-text review](independent-review-01/review.json) passes 1,158 checks with no errors; maximum numerical difference is 1.11e-16. It confirms all populations, ties, both failed Jtr clauses and the source/receipt chain, without importing the producer or reading arrays. Its separate clean operation costs 0.185388458s at 4,576 KiB. The [final decision](final-decision-01.json) rejects training a four-arm selector for this declared target. This is development action-class diagnosis on already exposed data; no novelty, learned transfer, independent recording confidence interval, official benchmark result or manuscript gate is established. D35's actual fixed-union improvement and support opportunity remain valid. A future method needs a richer action or a defensible change to future causal tracking state, with native and union plus strong same-information controls. No successor model, training, acquisition or manuscript is admitted by this result.
