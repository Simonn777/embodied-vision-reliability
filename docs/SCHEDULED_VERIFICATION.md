# Scheduled Work execution and migration

Verified on October 16, 2026. This is an AI-assisted operations receipt, not an experimental finding.

## Active schedule

The owner authorized migration to ChatGPT Work cloud within existing account entitlements, with no additional fees. The saved task runs in the same Work conversation every day at **09:00 America/New_York**, through **October 15, 2027** inclusive. The settings interface confirmed Eastern Time, daily repetition, that end date, and “Same chat for every run.” After validation, the local Codex heartbeat was paused; both superseded cloud candidates remain paused. There is one enabled research schedule. The next regular run shown at verification was **October 17, 2026, 09:00 EDT**.

The task uses cloud execution and the remote repository for persistent research state; it does not require the owner's Mac or desktop Codex to remain open. It retains the single-repository connector scope. The separately billed API workflow remains `disabled_manually` and `CLOUD_RESEARCH_ENABLED=false`. Do not buy credits, enable automatic reload, call paid APIs, or switch to paid compute if account entitlements are exhausted.

## Actual non-interactive trial

The native scheduler's **Run now** operation created a distinct non-interactive Work run on October 16 at 12:15:49 EDT. The run finished at 12:22:26 EDT. This was an immediate test of the saved automation, not the next day's clock-triggered run. No manual research message was substituted for it.

The run read remote main `14b3e5fd83190a1be9e8941d2fd2770acca485ca`, audited pinned official RGB-Stacking and Online BootsTAPIR sources and object headers, and published [PR #2](https://github.com/Simonn777/embodied-vision-reliability/pull/2). It did **not** download or evaluate the external dataset or model. The [audit](data-model-audit-2026-10-16.md) documents the remaining CPU, serialization, scene-grouping, causal-prefix and baseline checks.

Actual cloud-shell command outputs were inspected independently of the final answer. After an initial missing-pytest failure and installation of the declared dependencies, Linux x86_64 / Python 3.12.14 completed:

- `python -m pytest -q`: **107 passed in 2.57s**.
- `python scripts/run_experiment.py --smoke --output scratch/scheduled-audit-smoke --protocol protocols/0001-synthetic-localization.md`: **1,240 scored frames in 3.92s**.

Both commands had a 900-second hard timeout, a 6 GiB address-space limit and five numerical-library thread settings at 2. The [raw smoke provenance](cloud-capability/2026-10-16-research-smoke-provenance.json) records hashes and environment. Its shallow-history `protocol_commit` field is not the original protocol freeze; the unchanged protocol SHA-256 and the clarification in the [dated note](../notes/2026-10-16.md) preserve that distinction. Smoke outputs are infrastructure checks, not new final-test evidence.

## Independently checked publication

PR #2's final head was `b97d2c2c23c394eb14c19931ca556106887e344c`. [Research checks 38931256480](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38931256480), [job 117075516672](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38931256480/job/117075516672), passed before merge. Decoded logs show **107 tests in 3.38s** and **1,240 smoke frames in 3.81s** on the separate GitHub-hosted runner. CI tested merge ref `d8fcba8f8b35577cfa7f8dfbe8ea71a83cca1d7f`, combining the reviewed head and unchanged base.

The cloud run merged with `expected_head_sha` equal to that head and `merge_method="merge"`. Remote main was independently confirmed as [9d415308](https://github.com/Simonn777/embodied-vision-reliability/commit/9d415308d2eae55d94f972ced969563e3a862ea4). Only after execution, tests, CI and publication were verified did the settings session pause the old local heartbeat and retain the validated replacement as the sole daily research schedule.

## Limits and continuation

The successful Run now invocation establishes non-interactive tool access and publication in this Work configuration. The next wall-clock invocation has not occurred yet; this receipt does not prove a full year of uninterrupted execution or guarantee future quota, tool availability or platform uptime. Do not backfill missed activity or describe a queued/failed run as completed research.

Continue with protocol 0002's development feasibility stage in [NEXT_ACTIONS.md](../NEXT_ACTIONS.md). Read current remote state and active PRs before work, avoid duplicate experiments, preserve negative results, and publish only substantive changes after meaningful tests and review. The final scheduled session should assemble the annual record; do not extend the program beyond October 15, 2027 without renewed authorization.
