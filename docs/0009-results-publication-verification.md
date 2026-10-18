# 0009 result publication verification

2026-10-18. [PR 35](https://github.com/Simonn777/embodied-vision-reliability/pull/35)
merged as `6351a8c3eae12008dbac0a1e01e047de6e3b54ab`. The remote main was
fetched and the local main fast-forwarded to that exact commit. No science was
repeated for publication. The scientific decision remains **NO_GO**.

The cloud records branch was fetched at
`73d235fee6810e0a53249937a55d40d7c65a5add`, then cherry-picked as
`c4c89585d4272f081319ae55e9241900611a3c52`. Parent review and root documentation
were added at `0de78bc0c3fa172f6295ae64ea8a3e081845d221`.
The [independent review](../results/0009-stage0/parent-review/README.md) passed
15,780 checks of preserved public bytes, saved outcomes and actual tool records.
Its scope and unavailable private-cache observations remain explicit.

| Exact revision | CI run | Result |
|---|---|---|
| PR head `0de78bc0` | [39175262113](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39175262113) | Success: 649 passed, 31 skipped, 193 subtests in 24.40 s; 1,240 smoke frames in 2.36 s |
| Merged main `6351a8c3` | [39175347791](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39175347791) | Success: 649 passed, 31 skipped, 193 subtests in 19.68 s; 1,240 smoke frames in 2.01 s |

The 31 skips are historical Torch-dependent tests in CI's environment, not
passing tensor checks. The seven Linux supervisor tests ran. The smoke frames
exercise the existing 0001 output pipeline and are not new 0009 quality evidence.
Earlier HTTP accounting and the 0006 stdout defect remain unchanged.
