# 0025b physical-depth qualification publication verification

The full frozen diagnostic, all 256 arm records and independent arithmetic audit
were published in [PR #55](https://github.com/Simonn777/embodied-vision-reliability/pull/55),
merged at 2026-10-20 01:48:04 UTC (October 19 local research date).

- Scientific source freeze: `13d4b3259f8d12a1ba4d5d0de9ff56ec6527cf62`;
  [CI 39353648975](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39353648975)
  passed before acquisition and every scientific operation.
- Reviewed result head: `d54305dad91f999abe38c4fd4b51ef21eee3c1ba`;
  [CI 39355286895](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39355286895)
  passed before the merge bound to that exact head.
- Verified merge/main: `f71264a480a6d3a90887c54fb759696372292a3a`;
  [CI 39355433237](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39355433237)
  completed successfully. Live PR/run metadata and fetched main confirmed these identities.

The tolerance test passed, but the GS oracle was better in 56/64 cases. This is
qualification of an approximation with GT-assisted calibration, not improved
pose, a deployable method, independent-team replication or manuscript admission.
The two-source held-out improvement goal remains unmet.
