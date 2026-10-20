# 0024 closed-result publication verification

The frozen hypothesis, failed one-shot result and independent reconstruction are
published in [PR #53](https://github.com/Simonn777/embodied-vision-reliability/pull/53),
merged at 2026-10-20 00:31:34 UTC (October 19 local research date).

- Scientific source: `5d282a4b8b50094f433c6914931b005e7e16e73a`;
  [CI 39346730569](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39346730569)
  passed before the single 135-forward execution.
- Audited result head: `ee989bab4060605fd1e54ca87935af08ded97b82`;
  [CI 39347857463](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39347857463)
  passed before merge, which was bound to this exact head.
- Verified remote main: `a33c468fd69f4a7c4aae144db9a70bacdef79db8`;
  [CI 39348163527](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39348163527)
  completed successfully.

Live GitHub metadata and a fetch verified the identities. The complete result
remains **CLOSED_FAILED**: 16/27 usable offsets, 1.985998% increased cross-axis
residual, all four prospective conditions failed. CI and publication success do
not establish scientific progress or manuscript eligibility. No pose/GT test
or rescue tuning followed this probe.
