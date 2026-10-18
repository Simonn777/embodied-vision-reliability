# Shell restored; required PointOdyssey cache not found in the visible search scope

October 18, 2026, America/New_York. AI-assisted infrastructure observation.
The fixed 3-frame / 2-point geometry operation **did not run** in this follow-up.

## What changed

After the owner restored the original workspace connection, commands dispatched
through the original Work conversation executed successfully. The parent inspected
the actual command records, not just the cloud assistant's completion message.
This supersedes the current-runtime portion of the earlier
[offline observation](0003-cache-recovery-2026-10-18.md), which remains unchanged
as historical evidence. It does not establish why the earlier direct executor
route failed or why the required files are not visible now.

The first command returned the expected project working directory, but `stat`
reported `No such file or directory` for both `source-history` and the exact
`bootstapir-cache/0003-pointodyssey` directory (exit 1). A second command confirmed
that the project root, cache parent and `capability-repo` exist; `findmnt` reported
the root overlay filesystem (exit 0). A working shell alone is insufficient to
establish cache identity or reconcile previous execution state.

## Bounded existing-file search

The [observation](../results/0003-restored-shell-inventory-20261018/observation.json)
retains all three unmodified command/output records, with individual SHA-256
digests, tool exits and timing limitations. No data contents were opened.

| Observation | Result |
| --- | --- |
| Scope | Only the original project root and the known prior scheduled-project root |
| Traversal limits | 20,000 entries, directories opened through depth 6, 15-second traversal guard |
| Exclusions | No symlink descent; skip `.git`, installed runtimes, dependency and Python cache directories listed in the source |
| Actual traversal | 1,302 entries; 278 directories; no count/time guard triggered; no reported errors |
| Search-process elapsed time | 0.05493815399950108 seconds, including metadata listings and two Git checks |
| Tool result | Exit 0; execution-tool duration 422 ms, not child CPU time |
| Targets | Exact PointOdyssey cache/NPZ, prior two markers, new operation marker/output, and named archive/recovery candidates |
| Matches / recovery candidates | 0 / 0 within that scope |

The depth boundary and named-directory exclusions still apply when the output
says `truncated=false`. The source limits directory traversal, not the whole
command: the direct cache listing separately caps at 200 entries and each Git
subprocess at five seconds. No deeper or unrelated private-root search occurred.

The visible original root contains capability, CPU-development, pilot and g033
directories. `capability-repo` reports HEAD
`14b3e5fd83190a1be9e8941d2fd2770acca485ca`; its existing modified and untracked
files were preserved. The scheduled root has only `.codex` as a direct directory.
These are observations about the visible view, **not proof of a rollback, cache
deletion, backup availability or the cause of the problem**.

## Consequences for the frozen operation

There is no new geometry error, agreement count, camera qualification or model
result. The one-shot state across workspaces remains **UNKNOWN**, so the missing
marker cannot be treated as permission to retry. The frozen scientific gate and
runner were not invoked, and no lock or marker was created. The semantic-read
carry remains **22,910,156 bytes**, with no additional charge from this metadata
check. No new downloads, installation, inference, old-test tuning, threshold
change, paid service or schedule change occurred.

All earlier failures, the original rigid threshold
`9.5367431640625e-7`, the 0.5-pixel agreement rule and the published 45-input freeze
remain intact. The source-history conclusion and INCONCLUSIVE float16 proposal
have not been rerun or strengthened by this infrastructure work.

Continuation requires recovery of the **existing** selected annotation and both
original markers from a supported workspace view or an identifiable existing
backup. Their frozen sizes/hashes, shared locks, processes and any new-operation
marker/output must then be reconciled before the unchanged gate and bounded
runner may execute once. This search identified no restore candidate or supported
restore interface; platform-level recoverability remains unresolved. Do not
redownload or synthesize markers to bypass that prerequisite.

Publication review checks the transcription and preservation of existing files.
Normal repository CI validates software only; it cannot establish that the
PointOdyssey cache has returned or that this geometry operation completed.
