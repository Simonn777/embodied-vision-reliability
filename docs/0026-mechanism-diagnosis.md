# 0026 saved-output mechanism diagnosis

2026-10-19. **The fixed lifted-seed variant failed and is closed.** This note
reads the 80 sealed pair JSON records, saved evaluation and saved audit only.
It does not read scientific inputs, recompute an objective, refit, run native
code, change a threshold, or choose a new method using the labels.

## Result and evidence identity

Primary `G_lift` AUC@5 is **73.58810105463826%**, versus
**74.1396701117367%** for the frozen best control `F_all`: **−0.5515690570984475 pp**.
Only two of five scene differences are positive. London loses 2.2505351115392784
pp. The AUC@20-loss clause passes (loss 0.1378922642746261 pp); the other three
clauses fail. No pair was dropped. This is exposed development, not held-out
evidence or a manuscript result.

All 80 pair JSON byte hashes were checked against `predictions-01/seal.json`
before aggregation. Their recorded frozen commit is
`b596542f9eee399219ac03fdbcd91b33e8417fd6`. Relevant SHA256 identities:

| Artifact | SHA256 |
|---|---|
| `work/0026-development/predictions-01/seal.json` | `f295b3f240d9af080148a4680c81d42098d25579ac01117e1dd499d78d980736` |
| `work/0026-development/evaluation-01.json` | `a45edff1176723122a89c0e5d709ff464f13417f1aacf91418b2e376ad8d2617` |
| `work/0026-development/audit-01/audit.json` | `ea23b9ce9a4f46eb74a505adb61b9d4d6ce32bd018bb62830917ba933eada029` |
| frozen `core.py` | `f2bfc0ac6a0fa995f4cfbdb7059396dc3f7eb53fbcfbde72576920432053fdd3` |
| frozen `lift_relaxation.py` | `6f14df204ffca7cfa85c107b91fa83f81bff1e70a04981861d33b0d30487017e` |
| frozen `moments.py` | `3b96ee5d9928b961db68be7ebab51a6fd840f715197a868982c8633c88bedef4` |
| frozen `calibration_controls.py` | `1def821931043a560e45ee7e93d403f5ec3c2b96e8d074aad01de159600fa8b9` |
| frozen native library | `00418b002b005e664e75198c45e4a318d576a2ad04d8d0cb7eb7788affc18621` |

The seal contains the full 20-file source map, including evaluator and protocol.
The separate audit reports `passed`, 14,496 comparisons and source binding
passed; its scope is saved bindings/selection/cost sums/pose metrics, not an
independent rerun of scientific objectives.

## What failed, specifically

**Physical recovery, rather than conic convergence, removes 42/80 starts.**
All 80 relaxation solves report completed, with all nine barrier stages reporting
convergence. Exactly 42 physical roundings report
`nonpositive_corrected_depth`. Using the saved lifted variables and saved support
minima, 26 violate both views, 10 only the source, and 6 only the target. These
are the signs of `m1+b` and `a*m2+c` for the declared recovered parameters;
no scientific arrays were needed. The unavailable counts by scene, in frozen
order Florence/Lincoln/London/Milan/Sagrada, are **9/5/14/8/6**. This establishes
a mismatch between this relaxation/rounding and its physical seed domain. It
does not prove that a different relaxation would improve pose, and is not an
invitation to repair this variant on the same outcomes.

**The surviving lift usually reproduces the local-control solution.** For
descriptive comparison only, define matching saved objective values as
`abs(c1-c2) <= 1e-8*max(1,abs(c1),abs(c2))`. All **38/38** completed F-lift
costs match the best completed F-local cost. For G-lift, **33/35** do. Pair 63
has lower G-lift cost (8329.805933 versus 8799.704626); pair 70 has higher cost
(7964.719007 versus 7185.113373). This tolerance is a reporting convention,
not a gate, filter or reselection rule. Matching cost alone does not prove
identical minima.

More decisively, after the unchanged TC stage and original selection, the maximum
per-pair saved pose-error difference between `F_all` and `F_local` is
**2.9964e−12 degrees**; between `G_all` and `G_local` it is **9.3193e−13 degrees**.
F-lift wins the exact floating-point selector in 9 F-all records, and G-lift in
5 G-all records, but those winner names do not represent new effective accuracy.
`F_all−F_local` AUC@5 is about 1.5e−12 pp and `G_all−G_local` about 2.3e−13 pp.
The strongest evidence against the proposed mechanism is this absence of
incremental value over the same-objective local starts, not merely missing seeds.

| New attempt | Completed | Failed | Unavailable |
|---|---:|---:|---:|
| continuation | 79 | 0 | 1 |
| G neutral | 73 | 7 | 0 |
| G incumbent | 71 | 8 | 1 |
| G lift | 35 | 3 | 42 |
| F neutral | 78 | 2 | 0 |
| F incumbent | 77 | 2 | 1 |
| F lift | 38 | 0 | 42 |

There are 394 local-optimizer starts, 372 completed and 22 failed. Recorded
failures are 12 nonfinite/nonpositive represented physical trials, 8 abnormal
line-search terminations, 1 iteration-limit termination and 1 nonfinite distance
cost. Pair 37's incumbent has nonpositive supported corrected depth, making its
continuation and incumbent starts unavailable; the original pose remains in the
unions. Support sizes are 621–2048 (median 1996.5), so no support-under-four
fallback occurred.

## Positivity boundaries and final refinement

The records exhibit specific near-boundary calibrations, not proof that all
failures are caused by cloud collapse. Pair 35 G-incumbent finishes at
`a=3.8216546548587566e−11` and minimum corrected source depth
`1.2308820629414186e−11`, with positive raw G cost 199708.27965. Pair 55
F-incumbent has `a=9.87756835502247e−13`, but corrected minima are 1.208018 and
3.482793, and F cost is 8.234090. In the latter, the target depth signal is
almost removed by scale while the positive intercept remains; calling this
complete point-cloud collapse would be misleading. No completed local objective
has a negative reported raw cost. These examples support a boundary/calibration
ambiguity concern; their causal effect on the overall gate is not identified.

There are **451 native calls**. Of 79 continuations, **78 take zero LM iterations**
and one takes four; the saved baseline was already stationary under this call's
stopping rules. Among all calls, **39 reach the 100-iteration cap**. Saved final
native cost never exceeds its corresponding initial cost. This does not certify
convergence: below-cap termination can have several reasons, and the trace here
does not expose every accepted step.

Ten completed proposal outputs in four pairs have `physical_admissible=false`:
G neutral/incumbent for pairs 10,14,39, and F neutral/incumbent for pairs 10,43.
Their negative-depth counts were retained as required. No completed lift output
has that flag. Thus post-native positivity violation is observed but is not a
direct explanation of primary lift failure. The experiment did not intervene
on this behavior, so no causal rescue claim follows.

## The final score is not pose accuracy

The primary chooses G-lift in 16 pairs. Relative to the incumbent, using a
**descriptive** 1e−6-degree equality tolerance across all 80, its error is lower
in 4, higher in 6 and unchanged in 70. Examples are concrete:

| Pair | Incumbent MSAC → G-lift MSAC | Saved max-angle error |
|---|---|---|
| 20 | 299.109475 → 298.842852 | 0.381015° → 0.587358° |
| 72 | 342.063047 → 340.017695 | 0.619568° → 0.782261° |
| 31 | 254.187292 → 245.882349 | 1.286946° → 1.111895° |

Lower all-match MSAC can accompany either improvement or harm. This is a
documented ranking mismatch, not proof that another score would generalize or
that GT-based selection is a deployable solution. Pair 63 also shows why lower
G cost is insufficient: its lift proposal ends near the incumbent error, while
the local proposal has lower final MSAC but higher error. Pair 70's distinct
G-lift solution is not selected over the incumbent.

Recorded core component sums are 2.063433 s for G-lift, 4.305308 s for F-local
and 5.700711 s for F-all. Moments and relaxation total 0.179909 s and 0.420441 s.
These are the original component-accounting numbers, not complete-pipeline
timings: total worker walls sum to 43.132730 s, core wall to 11.544884 s, with
5.808325 s of progress callbacks reported separately and partly included in slot
walls. There is no matched-time speed claim. The redundant lift increases
reported F-all cost by 1.395403 s over F-local without meaningful accuracy gain.

## One different question, not an admitted method

An alternative to more initializations is **separating correspondence validity
from view-specific depth validity during refinement**. A match may be a valid
same-point geometric observation while one view's predicted depth is corrupted.
A candidate mechanism would retain its epipolar factor while treating that
view's depth factor as belonging to a spatially shared corruption state, instead
of forcing all supported depths into one calibrated rigid cloud. This changes
the observation model and the final optimum, not the seed or the number of
restarts. It would predict gains specifically when a localized depth failure
coexists with otherwise consistent two-view geometry.

**Novelty and utility are unknown; admission is not recommended yet.** Independent
robust losses, latent outlier mixtures, hybrid estimators and spatial models are
known ingredients. RePoseD/MADPose and recent robust-depth refinement must first
be checked for precisely this view-specific grouped treatment. It must beat
native TC, independent per-factor robustification, a spatial-weight-only control
and the strongest F-local control; merely suppressing difficult regions is not
a contribution. Previous spatial-field/cross-validation failures remain closed.

The smallest useful risk assessment would use the already available matched
pixels/depths to ask whether depth contradictions within an epipolar-consistent
support actually share reproducible **view-specific** regions, beyond geometry
error and independent point outliers. The current aggregate JSON cannot answer
that, so this note does not assert such a pattern exists. That assessment needs
a separately frozen, GT-free input diagnostic, no new frontend, and a stopping
rule: if group structure is absent or an independent robust-depth control explains
it, reject this direction before building an estimator. No such diagnostic has
been run or authorized here. The saved result supports closing seed-generation;
it does not establish this alternative mechanism.
