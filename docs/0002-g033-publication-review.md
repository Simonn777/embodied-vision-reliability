# Independent publication review: g-033

October 17, 2026. AI-assisted review of published candidate
`1c359bc574add90d6f6ee7db7a6ea191e9bf03a3`. Two separate read-only reviews checked
the raw artifacts and dependency provenance. No blocking discrepancy was found.
The [machine-readable audit](../results/0002-pilot-execution/g-033-parent-review.json)
records the checks, hashes and evidence boundaries.

The CSV contains the complete 64-frame by two-query grid: 128 unique rows,
point indices 0 and 1, and 512 finite coordinate/logit values. Its SHA-256 is
`22bb58db053c0cb011c72cadee1e0d91012301ff6e15bb3264ae23e39cd9819e`.
Each frame's latency agrees across query rows and with the runner receipt;
duplicated latency fields were counted once. Recorded intervals sum to
32.783010926 seconds. The inference subprocess took 49.286824886 seconds;
the complete wrapper, including its separate preflight, took 51.180634659 seconds.
Both runtime records report a peak RSS of 2,553,156 KiB (2.435 GiB).

All 29 frozen inputs match the frozen input commit, actual execution commit
`05c2d7cb4ec1f81b6bf080252deb69bd9a72a319`, and reviewed publication commit.
The publication commit is not the execution source. The v2 receipt is unchanged.
The 12 original failure-evidence files are byte-identical to failure checkpoint
`c585733510ed144be1305ca36eebc22f613dce62`; 20 recorded artifact digests also match.

All 16 recorded dependency versions and module origins are consistent. Fifteen
installed RECORD digests match history; NumPy's differs and is explicitly
reported as an exact-version environment reconstruction. Download, installation
and acquisition totals agree. This review did not download the wheel, model or
archive, rehash the cloud installation, or claim byte-identical environment
reproduction. Cloud measurements establish those external-byte identities;
the independent review checks their published record consistency.

## Exit-code clarification

`outer.json` records `wait4` results for the timeout-wrapped preflight and inference
children. Both successful-run child exit codes are zero. The wrapper itself
records `completed`; its parent's observation of the wrapper exit was not saved
as a separate public artifact. Child exit zero, a completed runner receipt and
the complete CSV support the inference-completion claim.

The original failure verification field `outer_exit_code` refers to the failed
preflight child's exit code, not a separately recorded wrapper exit. The earlier
note's shorthand “outer exit” should be read with this distinction. Original
records remain unchanged. No new execution was made to fill this evidence gap.

## Validation and next step

[Research checks 39010321341](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39010321341)
tested the candidate's merge preview into main `05c2d7cb`. Its job log confirms
184 tests passed in 7.40 seconds and the 1,240-frame synthetic smoke check passed
in 4.89 seconds. Final documentation/publication checks are attached to PR #6
and its eventual main commit; these software checks do not evaluate g-033.

The reviewer read no ground-truth coordinates and computed no tracking errors,
confidence scores, calibration thresholds or reliability comparison. Query
selection relies on the unchanged frozen runner and verified input identities;
annotations were not independently decoded again. No model replay occurred.

Progress remains **1/10 calibration groups inferred, 0/20 evaluation groups**.
The next allocated calibration group is g-013. Use the existing g-033 predictions
and preserve both failure and recovery records. Environment restoration was
within the user's existing request to complete this group without extra fees;
references to subsequent authorization in the execution notes describe the
orchestrating instructions, not a separate new human approval.
