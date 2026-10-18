# Independent exact-tail software review

2026-10-18. **APPROVE for its bounded artificial-mathematics scope. No blocking implementation defect found.** This is not approval of a geometric experiment, a real-match probability claim, or scientific novelty. No capacity failure occurred in this review.

Reviewed `/tmp/evr-0009-tail-audit/exact_tail.py`, SHA256 `cfef2d31b6aaf6a0e700c847eea62a23e3fe29d4c959febb7a367877ab1356e0`; author test SHA256 `c670f3425634ce180c5630e29406a7128a67ab3c45aeb1fa2ad7b5333e1d1c02`. Original source, tests, execution receipt, stdout and stderr files remained byte-identical. Their recorded source/log hashes match the actual files.

## Findings

- **Rook coefficients are correct.** Lines 121–126 recognize genuinely complete bicliques by component edge count and use the rectangular closed form. Lines 132–159 mask the smaller side and count both skipped vertices and every available one-edge assignment. Coefficients remain exact Python integers. The 12-vertex bound applies to the smaller side; a large complete biclique legitimately takes the closed-form path.
- **The global tail is combined correctly.** Lines 179–188 convolve rook polynomials, not component hit distributions. Isolated vertices remain in the original n. Lines 189–203 use global factorials and exact integer inclusion–exclusion. Empty boards, K≤0 and K>n have consistent counts, and the Fraction property preserves an exact ratio.
- **Unsupported means unsupported.** Lines 61–62 reject n>512; lines 127–131 reject a noncomplete component whose smaller side exceeds 12. These checks still apply for trivial thresholds. Refusal exceptions carry reasons/operation counts, but no fabricated probability or numerator. No approximation, case deletion or silent fallback is present.
- **Operation accounting is deterministic for the defined units.** The budget is checked before each instrumented unit. Independent tests of every positive sub-budget below successful small-board totals observed refusal exactly at the limit. Structural maximums also bound adjacency storage and DP state count (at most 2^12 masks).

**Resource limitation, not a blocker:** the counter is an algorithmic work budget, not a hard CPU-time or byte quota. Sorting, container allocation, combination/factorial internals, integer bit operations and Fraction's reduction are not individually charged. MAX_N=512 bounds their sizes, but a future resource-controlled runner still needs its own wall-clock/memory limits. No maximum-size runtime or geometry throughput was measured. The name of the author's “hard deterministic” budget test should be interpreted as a hard limit on these defined work units only.

The implementation does not repair the mathematical ranking limitations already documented in `evr-0009-exact-tail-audit.md`: exact permutation tails need not prefer broad true dense support, and descriptor-induced assignments need not be conditionally uniform. No calibrated p-value or universal selector guarantee follows from these passes.

## Actual independent checks

`python3 -W error /tmp/evr-0009-tail-independent-check.py` exited **0**, using Python 3.9.6 and only the standard library. The separate oracle directly enumerates row subsets/injective column assignments for rook coefficients and full n! permutations for tails. It does not call the candidate's private component or DP functions.

- **579 matrices:** every binary n≤3 matrix, plus 16 fixed-seed artificial matrices at each of n=4,5,6.
- **579 independently enumerated rook coefficient vectors**, all equal.
- **3,547 exact tail comparisons** over K=−1 through n+1, all equal.
- **263 individual too-small operation budgets** refused explicitly; **five positive exact-budget runs** passed, plus the empty zero-work board passed at the minimum legal budget of one.
- **Five 13×13 cycle refusal checks**, at K=−1,0,1,13,14, all raised unsupported without a probability. These are resource-refusal fixtures, not n>6 probability enumeration or performance tests.

The original suite was also independently rerun with `python3 -W error -m unittest test_exact_tail -v` from the author's prototype directory: **11 tests passed**, test-reported 0.274 seconds, observed process wall time 0.311044 seconds, exit **0**. This includes the smaller-side-12 boundary and complete-biclique-over-12 cases. It was not a benchmark.

Evidence files:

- `/tmp/evr-0009-tail-independent-check.py`
- `/tmp/evr-0009-tail-independent-check.json`
- `/tmp/evr-0009-tail-independent-author-tests.json`
- `/tmp/evr-0009-tail-independent-author-tests.stdout.log`
- `/tmp/evr-0009-tail-independent-author-tests.stderr.log`

No real data, geometric performance run, network access, installation, model execution or repository edit was involved. The prototype itself was not modified.
