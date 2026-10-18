# 0009 exact-tail software verification

2026-10-18. **Artificial arithmetic/software checks passed.** These are not geometry research samples, a performance comparison, or evidence of innovation. The candidate computes an exact count under a stipulated uniform-permutation null; descriptor-selected real correspondences need not satisfy that null.

- [Author archive](author/README.md): unchanged prototype, tests, execution receipt and logs. The original run passed 11 tests. Its 791 board instances and 5,218 tail comparisons are deterministic software-coverage counts, including every binary board through n=3 and fixed-seed boards at n=4–6.
- [Independent original review](independent/original-review.md), [check result](independent/original-check.json), and [author-suite rerun receipt](independent/original-author-tests.json): preserved verbatim. The separate oracle checked 579 matrices, 579 rook vectors, 3,547 tail counts, 263 insufficient budgets, and five unsupported-component fixtures. It also independently reran the author's 11 tests.
- [Portable checker](replay/check_exact_tail.py) and [new actual replay receipt](replay/execution.json): the original checker was changed only to accept source/output CLI paths, bind the reviewed source hash before import, and refuse report overwrites. Enumeration, seeds, comparisons and failure tests are unchanged. This packaging replay passed the same 579/3,547 checks with exit 0; it does not overwrite or replace either earlier run.

The original checker SHA256 is `4b41c0391ca9dd0654e9b1673f5d1db14389ba91ef605429f39929f8bb1d3f7e`; the portable version is `df7bb3931e666453041e2b0e5ce81d939ef6216a1c2e73e8cc175f71cc4421a7`. The archived source and [repository source](../../../src/evr/exact_assignment_tail.py) are byte-identical. The repository test changes only the import path. [Bindings](bindings.json) records those relationships and the 13 byte-exact copies. [Manifest](manifest.json) covers this package and its public review, excluding the manifest itself.

To repeat the independent artificial checks from a repository checkout, use a fresh output path:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  results/0009-admission/software/replay/check_exact_tail.py \
  --source src/evr/exact_assignment_tail.py \
  --output /tmp/0009-independent-replay-new.json
```

The original `/tmp` paths in historical receipts/scripts are retained as execution provenance, not required current locations. The author suite can be repeated from `author/` with `python3 -W error -m unittest test_exact_tail -v`.

The 2,000,000-operation limit counts defined algorithmic work units; it is **not a wall-clock or memory limit**. Any future experiment needs separate process limits and must record unsupported cases rather than discard them or substitute probabilities. Real graph coverage, geometry accuracy, exchangeability, selection corrections and novelty remain unresolved. See the [public independent review](../../../docs/0009-tail-independent-review.md). Substantial AI assistance is described in [CONTRIBUTIONS](../../../docs/CONTRIBUTIONS.md).
