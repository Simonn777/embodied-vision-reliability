# 0034 saved-scalar metric contract

Source preparation only. `metrics.py` uses the Python standard library and never
opens files, imports workers, loads arrays, decodes labels or computes per-mask
J/F. Root's model-free label driver owns complete prediction admission, original-
resolution mask/label validation and the qualified per-mask J/F implementation.
That driver treats label 255 as background for the inherited single target;
this helper cannot independently verify pixel-level conventions.

## API and validation

`build_report(rows, fit_failures)` returns a JSON-compatible report with schema
`evr0034.relative-rgb-metrics.v1`. Inputs are lists. Each of the 232 row dictionaries
must contain `sequence`, integer `frame`, integer `frames`, positive integer
`target_id` other than 255, and `arms`. Target ID must stay fixed within a parent.
Rows must cover exactly frames `0..N-1` for dog-agility25, planes-water38,
dogs-scale83 and dog-gooses86. Missing/duplicate/foreign frames, missing arms or
malformed values raise `ValueError`; incomplete inputs cannot create an eligible
report. Additional row/arm diagnostic fields and the complete `fit_failures`
list are deep-copied unchanged. All input values must be finite JSON-compatible
data; the helper does not mutate its arguments.

The exact arm order is `native`, `logit`, `absolute`, `prototype`, `nn`,
`ordinary_relations`, `bilateral`, `signed_no_gram`, `signed_gram`,
`displaced_gram`, `rgb256`. Every arm supplies finite numeric J, F and JF in
[0,1], with JF exactly `(J+F)/2` in the producing arithmetic. Booleans are not
metrics. No validation or decision epsilon is added.

## Aggregation and preserved populations

The returned top-level `rows` retains all 232 original records, ordered by the
declared parent order and frame. `coverage` reports 232 raw frames, 228
noninitial frames, 224 primary interior frames, four parents, eleven arms and
2,552 raw arm rows. Initial and final rows remain available even though primary
means exclude them.

`primary.parents` contains every arm's mean J/F/JF and its difference from native
for each parent, with interior counts 23/36/81/84. `primary.mean` is the mean of
four parent means, not a pooled-frame average. Every mean uses `math.fsum`.
`primary.delta_from_native` retains all arms and all three components. Metrics
remain on [0,1]; multiply differences by 100 to report percentage points.

`secondary.events` expands the old twelve anchors
`1 + i*(N-5)//11`, `i=0..11`, with horizons 1 and 3. It preserves all 96 events,
87 distinct `(sequence,frame)` identities, four final-frame events and every
duplicate. Each event includes saved arm metrics and its anchor/horizon;
complete extra frame diagnostics remain in top-level `rows`. Secondary means
use 24 events per parent and then four equal parents. They are descriptive and
never enter continuation.

## Fixed decision

The primary candidate is `signed_gram`. The strongest pooled named control is
the maximum primary JF among `native`, `logit`, `absolute`, `prototype`, `nn`,
`ordinary_relations`, `signed_no_gram`, `bilateral`; exact ties select the first
in that order. This is a diagnostic comparator, not per-parent GT-driven
deployment selection.

All eleven returned `decision.clauses` must be true:

1. Mean JF minus native is at least .01.
2. Mean JF minus the strongest pooled named control is at least .0025.
3. JF strictly beats ordinary relations on at least three parents.
4. JF strictly beats signed-no-Gram on at least three parents.
5. Each parent's JF minus native is at least −.005.
6. Mean J minus native is at least −.005.
7. Mean F minus native is at least −.005.
8. Mean JF strictly exceeds displaced-Gram.
9. Mean JF minus RGB256 is at least .001.
10. `fit_failures` is empty.
11. The complete fixed population is present.

Comparisons use exact `>=`/`>` on the aggregated stored numbers without relaxed
tolerances. `decision` preserves the chosen control, all parent margins,
positive-parent counts, all clause outcomes and `passed`. Its action is
`development_continuation_only` on a pass, otherwise
`close_fixed_signed_rgb_gram_recipe`. Declared numerical-fit failures remain
visible and invalidate continuation even when native fallback supplied complete,
finite metrics. Ordinary finite iteration exhaustion must not be mislabeled as
a numerical failure by the caller.

This is four-parent exposed-development cross-fitting, not an official complete
benchmark or four independent external replications. No bootstrap confidence
claim is made. Goal/manuscript flags remain false even on a pass; the existing
two-independent-held-out-source +3 official J&F gate is unchanged.
