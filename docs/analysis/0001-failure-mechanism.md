# Two frames of warning, then a stable wrong track

**October 16, 2026 · Exploratory analysis of saved experiment 0001 results.**
AI-assisted analysis; [contribution statement](../CONTRIBUTIONS.md). No estimator,
score, calibration threshold, or original result was changed. These test sequences
were already inspected. This is a mechanism analysis, not a new held-out experiment.

The temporal gate notices the jump to a distractor, then loses the evidence of
that jump as its two-position history fills with wrong predictions. A second
effect matters: its clean-calibrated threshold is more permissive on the static
score scale. Together, these explain why it accepts every wrong prediction after
the transition in this constructed case.

## What the saved measurements show

The independent units are **40 base sequences**, each evaluated in four paired
conditions. Counts below pool frames to describe those sequences; they are not
counts of independent trials. Failure means a position error strictly greater
than 0.025 of the image diagonal, as fixed in [protocol 0001](../../protocols/0001-synthetic-localization.md).

| Condition and frames | All frames | Missing | Static accepted / failures among accepted | Temporal accepted / failures among accepted |
|---|---:|---:|---:|---:|
| Appearance, jump and next frame (8–9) | 80 | 0 | 62 / 62 | 0 / undefined risk |
| Appearance, stable distractor (10–31) | 880 | 0 | 682 / 682 | 880 / 880 |
| Abrupt motion, turn (16) | 40 | 0 | 35 / 0 | 0 / undefined risk |
| Occlusion, partial/full mask (8–23) | 640 | 160 | 0 / undefined risk | 0 / undefined risk |
| Occlusion, first two clear frames (24–25) | 80 | 0 | 72 / 0 | 0 / undefined risk |

The 8–9 versus 10–31 appearance split and the 24–25 recovery detail are **post-hoc
diagnostic windows** motivated by the scorer's two-position memory, not
preregistered success endpoints. The complete per-frame table and all intervention
phases are retained in [analysis artifacts](../../results/0001-failure-analysis/),
including the random and accept-all controls. Undefined selected risk is not zero.

## Why the wrong track becomes consistent

The renderer simultaneously fades the target and adds a disk with the original
color and radius. The estimator thresholds color, then chooses the largest
connected component. It has no later identity association or learned visual
representation. From frame 8 onward, it chooses the distractor in every test
sequence. Across the 960 affected frames, every position error exceeds the fixed
tolerance.

For consecutive valid predictions, the causal extrapolation is
`p_hat[t] = 2 * p[t-1] - p[t-2]`. Once the last two raw predictions both equal the
stationary distractor position `q`, the next extrapolation and detection also
equal `q`; the residual is exactly zero. This happens at frame 10 in all 40
sequences. Rejected predictions still update history, as the frozen protocol
requires. Thus the gate does not retain the transition warning as an identity
warning. This algebra also permits a wrong constant-velocity trajectory to have
zero residual; **only the stationary case was measured here**.

An additional threshold effect explains the acceptance counts. The saved clean
calibration thresholds are:

| Policy | Threshold score | Calibration acceptance |
|---|---:|---:|
| Static | 0.03456659227905431 | 496 / 620 |
| Temporal | 0.07305208098769775 | 496 / 620 |

Tie keys remain part of the policy. In the stable-distractor phase, the two scores
are numerically equal, but their thresholds differ. The distractor's static score
reaches a maximum of approximately 0.054334 in this sample: 31 of 40 sequences pass the static gate,
and all 40 pass the temporal gate. Adding a nonnegative residual therefore does
**not** imply a subset of static acceptances when thresholds are calibrated
separately. Equal calibration coverage does not ensure equal coverage under shift.
Changing either threshold after seeing this outcome would be a new exploratory
comparison, not a repair of the original result.

Sequence 120 was selected for visualization in the original protocol, before
the full run. Its saved trace is a concrete example:

| Frame | Predicted point | Normalized position error | Static score | Temporal score | Static / temporal acceptance |
|---|---|---:|---:|---:|---|
| 8 | (18, 18) | 0.38671 | 0.04591 | 7.51513 | reject / reject |
| 9 | (18, 18) | 0.39153 | 0.04591 | 7.41302 | reject / reject |
| 10 | (18, 18) | 0.39634 | 0.04591 | 0.04591 | reject / accept |
| 31 | (18, 18) | 0.49748 | 0.04591 | 0.04591 | reject / accept |

## Why a correct track can be rejected

At the deliberate turn in frame 16, all 40 raw positions remain within the
localization tolerance. Static accepts 35; temporal rejects all 40 because the
constant-velocity prediction is inconsistent with the new motion. By frame 17,
temporal accepts 39. This is sensitivity to a motion-model violation, not evidence
that the image estimator failed.

In the occlusion condition, both gates reject every frame during the mask.
There are 160 missing detections, kept in the coverage denominator, and 412 large
errors among the 480 remaining detections. For those 480 partial-mask detections,
the observed area is only 21.24–36.36% of its initial value. The area term alone
is at least 1.01160, already far above the static threshold of 0.03457. This
explains why static rejection works on this conspicuous shape change. After the
mask disappears, all 80
predictions in frames 24–25 are accurate but temporal rejects them; its history
still contains biased partial-mask centroids. Static accepts 72. Over the whole
condition, temporal accepts 459/1,240 versus static 522/1,240, with zero observed
failures among either set of accepted predictions.

Both gates also reject the 68 partial-mask predictions still within tolerance.
For the 80 clear recovery frames, raw positions, static scores, and errors exactly
match the paired clean-condition records, while the motion history differs. The
clean temporal gate accepts 68 of those same 80 frame indices. This paired check
isolates the history effect within the existing construction.

## Uncertainty and limits

The original paired sequence bootstrap gives an occlusion coverage difference
(temporal minus static) of **−5.08 percentage points**, 95% interval
**[−7.18, −2.33]**. For appearance, the difference is **+12.02 points**,
**[+2.66, +22.43]**; selected failure increases by **2.07 points**,
**[−0.77, +5.68]**. These original intervals condition on the fixed calibration
sample and compare different accepted coverages; they do not establish a
matched-coverage advantage or a general harm from temporal information.

The supplementary analysis also resamples whole sequences, paired across gates,
for its phase summaries: 1,000 draws with seed 20261018, holding the original
calibration thresholds fixed. Its intervals are exploratory: selecting phases after
seeing a result is not accounted for by a percentile bootstrap. Degenerate
intervals from all-zero or all-one observed outcomes do not certify population
certainty. Neither calculation accounts for a new renderer, a new calibration
sample, or natural video.

Three limits separate the observations from a general research claim:

1. **The estimator is intentionally fragile.** Color matching causes this
   identity failure; it is not evidence that a frozen learned tracker would make
   the same errors.
2. **The score measures shape and local motion.** Its area/covariance features
   are not probabilities of correct identity. A stable wrong track exposes an
   information limitation of this particular score, not of all causal methods.
3. **The generator makes the case unusually clean.** The integer-centered,
   stationary disk has fixed area and isotropic shape; fading and distractor
   insertion are bundled. There is no factorial evidence attributing failure to
   either intervention alone, nor an estimate of its prevalence in real video.

## Decision for the next study

Keep experiment 0001 frozen and move to the dataset/estimator audit in
[NEXT_ACTIONS](../../NEXT_ACTIONS.md). A new external-data protocol must distinguish
visibility, localization error, identity ambiguity, and output availability;
audit whether a learned confidence head already uses temporal or future context;
and compare rejection scores on the same predictions with matched coverage or a
prespecified utility. The [methods reading note](../literature/2026-10-16-confidence-methods.md)
records the relevant primary-source checks and their reading depth.

Candidate explanations for a later development-only study include distractor
motion, fading without a distractor, a distractor without fading, and persistent
identity evidence. These are proposed controls, not implemented improvements.
Any revised score needs new, untouched final sequences and a committed protocol.

## Reproduce this analysis without rerunning the estimator

```bash
python scripts/analyze_0001_failures.py --output scratch/0001-failure-analysis
```

The analysis checks the original artifact hashes and uses their saved acceptance
flags. It writes a separate output directory, records its own source hash and
execution provenance, and refuses to overwrite an existing output. No new images,
weights, or datasets are needed. See the [analysis provenance](../../results/0001-failure-analysis/provenance.json)
for the actual execution and the [original provenance](../../results/0001-pilot/provenance.json)
for the frozen experiment.

Validation: 107 repository tests passed locally. A second execution of this
analysis matched all six deterministic output files byte for byte; another AI
worker independently counted the saved rows and reviewed the mechanism. The
[verification record](../../results/0001-failure-analysis/verification.json) keeps
these checks separate from the original experiment's reproduction. Analysis
source was uncommitted during execution and is identified by its saved hashes;
this status is recorded explicitly.
