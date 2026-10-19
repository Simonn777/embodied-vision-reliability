# 0011 result publication verified

[PR41](https://github.com/Simonn777/embodied-vision-reliability/pull/41) merged as
`6ee2638a0934f8cbff03fd545157a0ddd5159769` on October 19, 2026 at 01:54:35 UTC
(October 18 in New York). Exact PR head was
`999c0efb48d2878acbcf15cf74e248a28c8a4c20`; parent Git comparison confirms no tree
difference between the tested head and merge.

The parent directly verified both CI runs:

| Scope | Run | Result | Main tests | Smoke |
|---|---|---|---|---|
| Exact PR head | [39187363844](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39187363844) | success |763 passed, 31 historical skips, 272 subtests; 30.34s|1,240 frames; 3.05s|
| Exact main merge | [39187476005](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39187476005) | success |763 passed, 31 historical skips, 272 subtests; 29.73s|1,240 frames; 3.65s|

Both also passed the 54/10/11/15/17 artificial helper suites. The historical
Torch-related skips remain disclosed; CI is not an independent native model
rerun. The original cloud completed 114 artificial tests before science, then
both eight-case reviewer self-tests and the separate saved-result reviews.

Remote commit 0302bdab supplied the four text-bundle wrappers/archive. Parent
fetched those objects and verified all 101 archive members before publication.
[Independent arithmetic and provenance review](../results/0011-oblique/independent-review/README.md)
confirms the complete comparison and known log/privacy boundaries. All 24 frozen
inputs still match source commit a288f65 and pre-execution merge a0b0616.

This closes publication of **NO_GO**, not the owner's research goal. No further
inference, formal scoring, model/data download, installation or paid service was
used for this result publication. GitHub CI used its normal software-test setup.
The known missing/truncated read-only summary terminal log remains explicitly
disclosed. A manuscript remains unwarranted.
