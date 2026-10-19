# 0012 initial diagnostic: negligible computational benefit

All16 preselected development pairs completed once at source revision
`040de5f4a2c2c9b50b01bbbe1b168c85e705710c`; each has10,000 matches and64 valid
eight-point hypotheses. This is a numerical study of truncated Sampson scoring,
not an evaluation of relative-pose accuracy or an official DMS reproduction.

| Scorer | Direct residual evaluations | Selected exact winner | Total scoring + setup seconds |
|---|---:|---:|---:|
| Full vectorized MSAC | 10,240,000 | 16/16 | 0.33550 |
| Spatial-culling control | 10,210,000 | 16/16 | 21.89948 |
| Moment-bound candidate | 10,200,974 | 16/16 | 25.84403 |
| DMS-form surrogate | Not directly comparable | 14/16 | 1.02975 |

The candidate saved9,026 residual evaluations against culling, **0.0884%**.
Only one pair saved work; seven required extra work and eight were unchanged.
Seven hypotheses were pruned versus three for culling. This technically satisfies
the prospective nonzero aggregate-saving check, but is **insufficient practical
benefit**: the candidate was18.0% slower than this culling control and77.0times
slower than vectorized direct scoring. No speed, accuracy or paper-readiness
claim follows. Python timings do not predict optimized C++ performance, and
culling is overcharged for moment-tree construction that it does not need.

No recorded violations occurred in131,072 initial-cell checks per mode and
1,024 whole-score intervals. This does not prove floating-point certification
or independently inspect every adaptive refinement cell. DMS-form disagreements
are pairs4 and6; these concern this surrogate's ranking, not official DMS pose
quality. All16 pairs belong to one development scene and are dependent; no
generalization confidence interval is appropriate. Ground-truth poses and the
other scene's coordinates were not read. Official calibrated DMS additionally
uses cheirality checks and five-point hypotheses, which this diagnostic does not.

The entire run exited0 in63.18seconds with57,573,376 bytes peak child RSS on
macOS. [Raw measurements](diagnostic.jsonl), [all-pair table](all-pairs.csv),
[machine-readable summary](summary.json), [execution receipt](execution.json)
and [run log](run.log) are retained. Recompute the summary without accessing data:

```sh
python experiments/0012-score-bounds/summarize_development.py
```

Next scientific question: the spatial box discards correlation between paired
coordinates. The existing moment already contains signed residual mean and
variance, which may yield a substantially tighter deterministic tail bound.
Any such follow-up is exploratory on the same development cohort, preserves
these results and needs independent mathematical review before execution. No
extra confirmation data is allocated and no manuscript has started.
