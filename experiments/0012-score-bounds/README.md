# 0012 numerical score-bound prototype

This directory contains a NumPy implementation and synthetic correctness tests.
It contains no tracking model or acquired data. Test success does not establish
novelty, useful task accuracy, speedup, or floating-point formal certification.

The target is the summed MSAC-truncated Sampson score. `tau2` is the **squared**
threshold in the same coordinate system as the inputs. A zero denominator,
including 0/0, contributes `tau2` in every path. The zero matrix is accepted
under that same convention. Finite nonzero matrices are normalized by their
largest absolute coefficient. No essential/fundamental rank constraint is
imposed by the scorer.

`MatchTree(xy1, xy2, leaf_size=32)` constructs a reusable four-dimensional
median tree. Each cell retains its correspondence indices, feature rows,
second moment, row extrema, and spatial box. A second absolute-product moment
supports the numerical cancellation guard; its memory and build cost must be
counted. Full matrices and per-node rows are stored for clarity; this is not a
memory-optimized implementation.

```python
from score_bounds import MatchTree, cell_bounds, exact_msac

tree = MatchTree(xy1, xy2, leaf_size=32)
cells = tree.partition(128)  # deterministic shared starting partition
interval = cell_bounds(cells[0], E, tau2=1.0)
bounded = tree.score_bounds(E, 1.0, initial_cells=128, max_refinements=256)
winner = tree.best_hypothesis(models, 1.0, initial_cells=128,
                              max_refinements=256, mode="moments")
control = tree.best_hypothesis(models, 1.0, initial_cells=128,
                               max_refinements=256, mode="culling")
```

`mode="moments"` combines quadratic-moment, row-box, denominator and mixed-cell
secant bounds. `mode="culling"` only bounds certified capped outliers; ambiguous
cells retain `[0, n*tau2]` until subdivision or direct residual evaluation.
The two modes share a partition and refinement policy. This is a spatial-culling
control, not a reproduction or claimed equivalence to an external implementation.

Adaptive refinement replaces a parent with its children, prioritizing interval
width. Leaves can be evaluated directly. `best_hypothesis` computes a full-score
incumbent, prunes only with a strict guarded lower-bound comparison, and directly
scores unresolved hypotheses. Ties use the smaller original index.
`rank_hypotheses` additionally supports complete ordering: disjoint interval
components are ordered directly, and overlapping components receive full scores.

Counters include visited bound nodes, directly evaluated residuals, fallback
models, and refinements. Direct leaf work and a subsequent full-model fallback
are both charged if both occur. Time the complete caller, including normalization,
tree construction/amortization, partition construction, bookkeeping, and exact
fallback; do not convert a lower residual count into a speedup claim.

Numerical safeguards use outward padding, absolute product-sum error scales for
near-zero quadratic moments, expanded box radii, denominator-zero handling, and
conservative score-cap enclosures. Upper bounds may slightly exceed `n*tau2`
because a reference NumPy reduction can round above that product. These guards
are tested engineering allowances, not a proof for every float64 input or BLAS.
Extreme coordinate magnitudes that overflow stored moments/reference residuals
raise an error rather than silently dropping correspondences.

Run the small synthetic test suite with:

```sh
python -m pytest -q experiments/0012-score-bounds/test_score_bounds.py
```

Validation on 2026-10-19: **12 tests passed**. Tests cover random cells/refinement,
duplicates/empty inputs, zero matrices and denominators, cancellation around
large coordinates, truncation thresholds, extreme hypothesis scales, a strictly
tighter mixed-cell secant example, partition coverage, strict pruning, and
deterministic exact fallback ordering. The first run exposed a one-ULP cap
clamping defect; the implementation now encloses reference reduction rounding,
and tests keep zero test-side containment tolerance.

Independent review then exposed a distinct extreme-cancellation defect: a
finite stored moment could produce a NaN quadratic query and an infinite error
scale, which nonnegative clipping hid. Raw values and intermediate intervals are
now checked before clipping; nonfinite query arithmetic returns the full-cap
interval. The exact reported `1e77` counterexample is a regression test.
Independent synthetic recheck passed at source SHA256
`392bbdee15f6e6ebf3faa00d95c6e133ec780d4fbbba5511cc013bf781d64a4e`.
