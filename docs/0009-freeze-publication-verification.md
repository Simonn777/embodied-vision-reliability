# 0009 freeze publication verification

2026-10-18. PR [34](https://github.com/Simonn777/embodied-vision-reliability/pull/34)
merged as `cd144b90d337a172a892b56d87cd794524605753`.
The canonical receipt-only commit is
`17f417efcccf85047e0935569f87f94b27705e82`, immediately after source
`4b11f1de4d6fcc9bef45c8eee824ff8e7241abc0`. It binds15 exact files.
Receipt SHA256: `2485a844f95c47ee420617d0464be7ce8bf5e747b8539b70f2450df2306d6631`.
Local verify-only passed both at the receipt head and on merged main.

## Actual CI

| Revision | Run | Result | Tests | Smoke |
|---|---|---|---|---|
| Source4b11f1d | [39169826835](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39169826835) | success | 649 passed,31 skipped,193 subtests;24.28s | 1,240 frames;2.96s |
| Receipt17f417e | [39169935934](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39169935934) | success | 649 passed,31 skipped,193 subtests;24.53s | 1,240 frames;3.74s |
| Maincd144b9 | [39170031333](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39170031333) | success | 649 passed,31 skipped,193 subtests;24.79s | 1,240 frames;3.73s |

The31 skips are historical Torch-dependent tests in CI's environment. The seven
0009 Linux supervisor tests ran. The smoke is the existing0001 pipeline; it is
not a0009 quality result. Exact head/status and Git remote were checked directly.

After these checks, the original authorized Work cloud received one execution
request for the frozen256-case screen. No outcome is asserted by this publication
receipt. No dataset/model/package download or paid service is required.
