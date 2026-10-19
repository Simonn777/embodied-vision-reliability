# 0012b: useful residual-work reduction; runtime gain not established

The exploratory follow-up completed all16 previously used development pairs and
the identical1,024 saved hypotheses at frozen source `71a336a0fcb3372b1229b0a9e7e5bdf06b8e0c52`.
No new pairs, labels or hypotheses were selected. No ground-truth pose was read.

| Method | Direct residual evaluations | Exact winner agreement | Setup + selection seconds |
|---|---:|---:|---:|
| Added signed mean/variance bound | 2,956,863 | 16/16 | 25.83199 |
| Ordinary block256 partial scoring | 5,413,920 | 16/16 | 0.57849 |
| Original full scoring (historical) | 10,240,000 | 16/16 | 0.33550 |
| Original moment bound (historical) | 10,200,974 | 16/16 | 25.84403 |

The new method reduces residual evaluations **45.3841% versus partial scoring**
and **71.1244% versus full scoring**. Every pair improves residual work versus
partial scoring; the candidate/partial ratio ranges0.2350–0.9099 (median0.5684).
Fourteen pairs individually meet the20% saving target; the prospective gate was
aggregate. Fifteen meet50% saving versus full scoring. Candidate pruning avoids
full scores for734 hypotheses; partial scoring prunes more models, but spends
more residual work before deciding. Model counts alone are not a work measure.

The aggregate numerical gate passes: ratios0.546159 and0.288756 are below0.8 and
0.5, every winner is correct, and all131,072 checked initial cells plus1,024
aggregate intervals contain the reference. The added bound increases123,532
initial-cell lower bounds. All saved proposals and reference scores are unchanged.
Independent saved-only arithmetic/source review agrees.

**This is a mechanism result, not a runtime or pose-quality result.** The Python
candidate remains44.65times slower than partial scoring. Bound evaluations,
tree preparation and fallback must be charged; residual reductions are not
speedups. Historical timing is not a concurrent controlled comparison. This is
one dependent scene reused for development, with no generalization estimate or
held-out confirmation. Numerical guards are not formal floating certification.

The run exited0 in48.19seconds; peak child RSS58,064,896 bytes. Retained evidence:
[all pairs](all-pairs.csv), [raw JSONL](diagnostic.jsonl), [summary](summary.json),
[execution](execution.json), [log](run.log). Recompute summaries with:

The frozen reference again emitted three types of NumPy matmul warnings; all
explicit finite checks passed, but the warning cause remains unresolved. The
public log changes only local absolute workspace prefixes; the original is
retained locally. No warnings or measurements are omitted.

```sh
python experiments/0012-score-bounds/summarize_mean_variance.py
```

The permitted next step is a bounded same-backend native comparison with direct
and partial scoring, including fresh setup. Only if measured time improves does
calibrated pose integration become justified. The paper-quality gate remains
unmet, and no manuscript has been started.
