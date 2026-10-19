# Cached first-probe correction: published before execution

[PR 37](https://github.com/Simonn777/embodied-vision-reliability/pull/37)
merged as `3b874f53b44fb2b3a622caf624a1c0ffe2e802c4` on October 19 UTC
(October 18 America/New_York). The parent verified its tree matches receipt head
`9fda4201231e7bdda2547e673db0bb45723f27d5` with no content difference.

The source commit is `4f3eedf1c614dd2b7ef95c1366994b601e273a09`. Its direct child
adds only [the canonical receipt](../protocols/0010-cached-probe-solver-freeze.json),
SHA256 `1f745cc649436664ca257ebdf10640921918121894452ab02e26036d56bcca3f`,
binding 18 inputs. Original capability receipt/source remain unchanged; its
cloud checkout stays on the original main so its 25 historical inputs still agree.

Exact [receipt-head CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39179854354)
and [merged-main CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39179957104)
completed successfully before dispatch. Parent read both actual job outputs:

| Check | Receipt head | Merged main |
|---|---:|---:|
| Repository tests |649 passed, 31 historical skips, 193 subtests; 25.10 s|same counts; 24.95 s|
| Capability artificial tests |54; 0.148 s|54; 0.148 s|
| Solver orchestration fake tests |10; 0.015 s|10; 0.017 s|
| Cache/runtime guard fake tests |11; 0.169 s|11; 0.165 s|
| Synthetic smoke pipeline |passed, 1,240 frames|passed, 1,240 frames; 3.69 s|

CI did not run CLIP, the 20 Torch fixtures, the three measured solver fits or
classification evaluation. The source-only commit was not separately claimed
to have CI; the verified receipt head contains that exact source tree plus receipt.
This publication admits only the bounded steps in the correction protocol,
not the separate prospective 128-image quality study or a manuscript.
