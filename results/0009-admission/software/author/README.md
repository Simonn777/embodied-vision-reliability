# Exact permutation-tail correctness audit

2026-10-18. **Mathematical formula and artificial software correctness: PASS.** This does not approve geometric performance, a real-data experiment, novelty, or real-match p-value calibration. No images, real correspondences, labels, model, package installation, network, or repository edits were used by this audit.

## Formula

Fix an n×n binary board A. For a uniform permutation π of n columns let T=Σ_i A[i,π(i)], and let r_j count size-j nonattacking rook placements on A. Counting permutations extending each j-matching yields

`Σ_t choose(t,j) h_t = r_j (n−j)!`,

where h_t counts permutations with exactly t hits. Expanding `E[z^T]` in powers of `(z−1)` gives

`h_t = Σ_(j≥t) (−1)^(j−t) choose(j,t) r_j (n−j)!`.

For integer K≥1, summing over t≥K and using

`Σ_(t=K)^j (−1)^(j−t) choose(j,t) = (−1)^(j−K) choose(j−1,K−1)`

proves the requested tail count

`N_K = Σ_(j=K)^n (−1)^(j−K) choose(j−1,K−1) r_j (n−j)!`.

Divide by n!. All arithmetic in the candidate uses Python integers; no normal approximation, floating cancellation, Monte Carlo estimate, or invented zero p-value occurs. The Fraction property is exact; the stored numerator/denominator are unreduced permutation counts.

Degeneracies: K≤0 gives numerator n!, K>n gives zero, an empty n=0 board has one empty permutation, an empty-edge board has T=0, and an all-one board has T=n. No variance division is involved. The API intentionally checks computational support even for trivial K, so its supported-board domain does not vary with the observed count. Threshold must be a built-in integer, excluding bool.

## Implementation and scope

`exact_tail.py` is a single stdlib-only candidate. API:

```python
result = exact_permutation_tail(board, K, operation_limit=2_000_000)
# result.numerator, result.denominator, result.probability (Fraction)
# result.rook_coefficients, result.components, result.operations
```

Input is a square built-in list/tuple of built-in binary int/bool entries. Each nonisolated row/column connected component is found in the bipartite graph. For a complete a×b component, its coefficient is `choose(a,j) choose(b,j) j!`. Otherwise a subset DP tracks occupied vertices on the smaller side while visiting each larger-side vertex once. Each visit either skips that vertex or matches it to one free neighboring small-side vertex. This enumerates every partial matching exactly once. Aggregating masks by population gives component rook coefficients.

Multiply component rook polynomials, then use the **global** factorial formula. Isolated vertices contribute through n!, not artificial one-by-one components. Local permutation hit distributions cannot be multiplied: for A=diag(1,1), P(T≥2)=1/2, not (1/2)². This counterexample is an executed artificial test.

Fixed computational domain:

- n≤512; larger n explicitly unsupported.
- Complete bicliques use the closed form regardless of the smaller side.
- Every other component must have smaller side≤12.
- A deterministic budget, default 2,000,000 charged work units per call, refuses *before* the next charged operation. Validation cells/rows, graph vertex/edge visits, DP transitions/aggregations, polynomial coefficient products, factorial-table multiplications, and tail terms are charged. Refusal reports used units and returns no result.

This work-unit budget is **not** a count of machine instructions or measured CPU/RSS/time guarantee: Python sorting, bit operations, dictionary implementation and big-integer internals are not separately metered. Fixed n and small-side bounds additionally constrain those costs. The n≤512 bound implies final permutation counts at most 512!; intermediate rook and transformed terms are larger in some cases but still bounded by n-dependent finite combinatorial quantities. No empirical runtime extrapolation is claimed.

Ignoring bit complexity, graph construction costs O(n²), a noncomplete component costs O(b·s·2^s), s≤12 and b the larger side, and DP state memory is O(2^s). Straight component convolution is at most O(n²) coefficient products. Board adjacency memory is O(n²); coefficient and integer memory are bounded by the fixed n cap. A supported component can still exceed the overall work budget.

`UnsupportedBoard` covers component/dimension refusal; `OperationBudgetExceeded` is its subclass. Invalid inputs raise TypeError/ValueError. An integer range invariant failure raises ArithmeticError. None yields a p-value. A later study must preserve the original case denominator and mark candidate-method failure if any required hypothesis is unsupported; silently optimizing over only successfully computed hypotheses would change the method. This audit did not establish actual geometric component coverage.

## Actual artificial verification

Executed once, with warning-as-error:

```sh
cd /tmp/evr-0009-tail-audit
python3 -W error -m unittest test_exact_tail -v
```

`execution.json`, `stdout.log` and `stderr.log` record exit 0; **11 tests passed**, zero skipped, Python 3.9.6, external wall time 0.314829958 s (unittest reports 0.273 s). This time describes these correctness fixtures only.

Coverage is deterministic from the executed test source:

- Every binary board for n=0,1,2,3 (531 boards).
- 80 fixed-seed artificial boards each for n=4,5,6 (240 boards); every permutation enumerated for each board.
- Zero/all-one/permutation boards, disconnected rectangular components, isolated vertices, row/column reindexing, and all integer thresholds −1 through n+1 on enumerated boards.
- Across `check_board` calls: 791 board instances, 76,286 permutation visits, and 5,218 exact tail comparisons. These are software coverage counts, not independent scientific samples. 563 rook vectors additionally compared with separately enumerated row subsets and distinct-column assignments.
- A noncomplete 12×13 component compared with the closed form for a complete rectangle minus one edge; a noncomplete connected 13×13 cycle refused; a complete 13×13 board accepted.
- Invalid/ragged/nonbinary/NaN/type inputs rejected; an exact-budget call succeeds and one-less-budget call refuses at the stated used count; unsupported exceptions contain no probability.

The exhaustive reference uses direct `itertools.permutations`, not the production rook/tail formula. The direct rook reference independently enumerates partial assignments. Tests are original artificial constructions; no geometry accuracy comparison was run.

## Remaining blockers for research claims

This computes an exact law only for a fixed board under the stipulated uniform bijection null. Descriptor screening or many-to-one matches need not satisfy that null. H/tolerance selection and multiplicity also require justification; holding coordinates fixed does not establish exchangeability. An exact tail is not the probability that a homography is correct. Rook theory is classical; this audit confers no novelty. Actual image component support, full pipeline CPU feasibility, comparison with ORSA, geometric quality, and false-alarm calibration remain untested. Independent code review is still appropriate before freezing this candidate.
