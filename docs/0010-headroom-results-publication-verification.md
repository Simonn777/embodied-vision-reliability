# Failed headroom result: publication verified

[PR39](https://github.com/Simonn777/embodied-vision-reliability/pull/39)
published source `1df077ed4c06a18b8bdd247c29f4ae78e730b7c2` and merged as
`d9cb569441f9131694433e137eaad9b6471b01e9` on October19 UTC
(October18 America/New_York). The parent verified the remote PR/main identities,
read both completed CI logs, and found no tree difference between head and merge.

| Check | PR head | Merged main |
|---|---:|---:|
| Repository tests |649 passed,31 historical skips,193 subtests;25.40s|same counts;24.50s|
| Capability fake tests |54;0.152s|54;0.151s|
| Solver fake tests |10;0.016s|10;0.016s|
| Guard fake tests |11;0.166s|11;0.167s|
| Headroom predictor fake tests |15;1.577s|15;1.574s|
| Headroom driver fake tests |17;6.330s|17;6.217s|
| Synthetic smoke |1,240 frames;3.86s|1,240 frames;3.80s|

[PR CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39182254754)
and [main CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39182366051)
both succeeded. These software checks did not repeat the real-image experiment.
The independent saved-output review and original cloud archive are linked from
the [result](../results/0010-headroom/README.md).

Publication confirms the recorded **failure**: exactLOO61/128 versus zero-shot65/128,
with7 corrections and11 harms. It does not change the frozen NO_GO, establish
novelty, or activate conditional follow-ups. All historical failures, private
cloud caches, original outputs and once-only markers remain intact. No model,
data or package acquisition, paid service, or scientific retry was needed.
