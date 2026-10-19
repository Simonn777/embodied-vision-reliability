# Lazy-grid diagnostic

This is the separately frozen0012d computational hypothesis; the failed0012c
source is included unchanged for common residual and partial-scoring controls.
It is neither a complete pose estimator nor an implementation of official DMS.
The [protocol](../../protocols/0012d-lazy-grid-screening.md) and
[derivation](../../docs/0012d-lazy-grid.md) define the scientific limits.

`native_grid.cpp` exports `ng_select` and `ng_validate`, while retaining the
included `ns_scores` for exact-score validation. `NGResult` has the original
`NSResult` as its first member, followed by six uint64 counters: moment cells,
moment points, grid cells, eligible cells, geometry calls and Cantelli calls.
Moment creation is lazy and is therefore charged to the scoring phase; total
outer time includes every phase. `nodes` counts screened cell visits, and
`refinements` is zero because the grid is fixed. All fresh selection calls own
their own geometry and moments.

Validation returns `NSValidation`. It checks both geometry-only and combined
lower bounds on each eligible cell, plus both aggregate lower bounds for every
model. Thus64 models produce128 aggregate checks; a workload with no eligible
cells can legitimately have zero cell checks. This is lower-bound validation,
so `max_upper_violation` remains zero. `first_cell` is a first-seen grid index,
with -1 indicating an aggregate violation; the field does not identify which
of the two lower bounds failed. Invalid floating arithmetic cannot become a
positive certificate.

Build with `python experiments/0012-score-bounds/grid_backend.py --build`.
The printed local build receipt identifies the library and both native source
hashes. Build files remain outside the repository in `work/native-grid/`.
Set `GRID_LIBRARY` to that receipt's library path, then run
`python -m unittest discover -s experiments/0012-score-bounds -p test_grid_backend.py`.
Use the current Python environment with NumPy; no new package is needed for
these artificial tests. Use one native/BLAS thread for the timing diagnostic.

After the independently reviewed source/protocol freeze, run
`run_grid_synthetic.py --build-receipt ... --output ...` once. It retains45
timed calls,5 untimed loading calls, exact scores, containment checks, hashes
and the admission decision. Each call starts with fresh needed setup. The
fixture deliberately benefits signed moments; success only establishes cost
headroom in this constructed situation.

`run_grid.py` refuses the real development replay unless the artificial
admission passed and the protocol, source and actual binary hashes still agree.
Supply `--data`, `--manifest`, `--reference`, `--build-receipt`,
`--synthetic-admission` and a new `--output` path. This driver only reads the
already allocated16 pairs' x1/x2 arrays. It cannot infer pose accuracy from
score agreement. Reused development results never become independent evidence.
