# Independent PR20 path-map correction

Reviewed candidate `2a13cc36bedc87dc14a6e64b6de10e503ccdb0ab` against base `eb4769c23f5947e5789dbea82dfb8072ddf23726`. This is an AI-assisted publication review, not a scientific execution.

The original `results/0003-cache-readiness-20261018/verification.json` uses absolute runtime-workspace keys for its 17 artifact references. The verifier scans files while excluding its own output, then creates that output with `str(Path)` keys. The newly emitted absolute paths therefore escape the scanner. Its recorded PASS does not establish clean output-path handling.

[the normalized manifest](../results/0003-cache-readiness-20261018/independent-normalized-manifest.json) supplies all 17 repository-relative references. Every digest matches the corresponding candidate Git blob; no artifact content or digest changed, and no original published file was edited. The original verification JSON is separately bound by its SHA256. The new manifest excludes its own hash so a later publication commit or enclosing bundle can bind it.

The readiness parent's recorded executed-source hash differs from its published, redacted source. The publication note says a cache-location literal was replaced with an environment lookup. Because the exact executed snapshot is unpublished, this review cannot independently verify that this was the only source difference. Gate-command, frozen-source, log and child-exit evidence remains separately checkable; the published script must not be described as byte-identical to the executed parent.

The saved record is a cache-visibility blocker: the publication gate passed, but no scientific runner main was invoked by the recorded command. Missing cache and markers do not establish cross-host inactivity or authorize marker reconstruction. The first scanner failure has a narrative record, not a separately captured verifier exit; final verification PASS records in-process assertions. Neither substitutes for geometry results.

The accompanying independent review JSON uses `scientific_blockers: []` to mean no additional scientific-claim or evidence-review defect was found in the recorded blocker. It does not mean `execution_ready`: the unavailable original cache and markers still block the real one-shot operation.

This addition performed only Git-blob/hash inspection: no tests, gate, data/cache access, model or geometry rerun. Original evidence remains unchanged.

Normalized manifest SHA256: `413c9277ab51b798aca1213ac2c24cb9320ccf22b028198a53edf53a3faa0b9e`.


## Offline follow-up and limits of the parent review

The subsequent candidate `c80c1d769c2e6b512c1c0b6c8eba09aaa52a2dae` adds a
[recovery report](0003-cache-recovery-2026-10-18.md), its observation JSON and
append-only state/action/note sections. It reports that the requested original-path
lookup was rejected before a shell started. PID, exit, wait, timing and RSS fields
are correctly null. Current cache, mounts, processes and markers remain unknown.
Its recorded clock is a later compilation time, not a measured launch timestamp.

The parent inspected the cloud agent's statements and GitHub publication actions,
but the thread reader did not expose the raw failed execution-tool response.
This review checks the report's internal consistency, frozen-source preservation,
Git publication and CI; it does not independently attest the runtime rejection or
global network/billing state. No real-data output was added or verified. The cloud
report cannot establish deletion of the original cache or authorize a retry.

Root independently read candidate CI 39143856993 logs: merge reference cbf94a5
contains c80c1d7 on eb4769c, with 320 tests and 4 subtests passing in 8.46 seconds
and 1,240 synthetic frames in 2.94 seconds. These standard GitHub-hosted software
checks do not establish Work runtime availability or a successful geometry check.
Final publication checks are recorded separately in
[PR20](https://github.com/Simonn777/embodied-vision-reliability/pull/20).
