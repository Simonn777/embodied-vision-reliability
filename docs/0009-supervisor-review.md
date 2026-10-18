# 0009 external supervisor review and bounded repair

2026-10-18. Reviewed the parent's new wrapper before freeze. Changed only
`scripts/supervise_0009.py` and added `tests/test_0009_supervisor.py`. No stage,
science, network, asset, installation, commit or other repository file modified.
The wrapper initially came from the parent; this reviewer authored the repairs
below. Its source is now a jointly implemented artifact, not an untouched
independent endorsement.

Pre-edit wrapper SHA256:
`a0eafb6f495982ee936c2d5678f8ced30e0a3a7a33dc8398d38044e8000c0ea1`.
Original bytes retained at `/tmp/evr-0009-supervisor-before.py`.

Final wrapper SHA256:
`2d306a8813d36c207eb276a187a298864965bffbe79c1dd463d9482f3bf7d77f`.
Final test SHA256:
`a0007e42467955462e60d04651f4271bdb09c4586bebb2775603b820d51491f6`.

## Findings fixed

1. **Launch failure lost its receipt.** Popen was outside the protected region.
   Now launch errors retain command/log/receipt files, safe exception type and
   failure stage; supervisor exit125, child exit/wait/RSS=null. No invented child
   measurement or full exception message is recorded. Existing-directory refusal
   remains before launch and cannot overwrite a previous attempt.
2. **SIGTERM could orphan the scientific child.** The old process only handled
   Python exceptions. The wrapper now handles SIGTERM/SIGINT, kills the child's
   process group, wait4-reaps it, records the actual child signal status and maps
   the supervisor interrupt to143/130. The launch interval blocks these signals
   until Popen provides a usable PID; the child restores the original mask before
   exec. Original signal handlers/mask are restored. Cleanup ignores a second
   termination signal to finish reaping. Surviving same-group descendants are
   terminated even after a normally exited leader.
3. **BaseException was swallowed and every fault became an interrupt.** Ordinary
   launch/wait faults now return125 with their safe type/stage, while an unusual
   BaseException is cleaned up, archived, then re-raised unchanged. In that case
   the wrapper does not invent a final outer exit code: its receipt field is null.
   Missing reap evidence also cannot become success.
4. Deadline validation rejects NaN/infinity/bool and invalid overrides; explicit
   affinity rejects duplicates, bool/noninteger CPUs, unavailable CPUs and >4 CPUs.
   CLI still offers no timeout override and builds only the fixed collector argv.

## Actual local validation

Command: `python3 -W error -m unittest discover -s tests -p test_0009_supervisor.py -v`.
Actual exit0 on Python3.9.6/Darwin. **15 discovered, 8 passed, 7 explicitly skipped**.
Do not describe this as Linux resource/timeout execution. Portable mocked fault
checks cover launch evidence, invalid limits/affinity, non-Linux refusal, wait
failure, SIGTERM cleanup/handler restoration, and preserving SystemExit after
cleanup/receipt. These do not call the collector or generate geometric cases.

Seven Linux-only artificial subprocess checks are ready for CI/cloud preflight:
exit0 with exact stream hashes; exit7; actual affinity/thread environment/RLIMIT_AS
and RLIMIT_CORE/signal-mask/process-group checks; sleeping child+grandchild hard
timeout; no-overwrite; missing executable; SIGTERM of a separate supervisor with
actual child reaping. They use stdlib toy processes only, no scientific input.
Linux completion of these checks remains required before a claimed execution-safe
run. Local command/log hashes and wall time are recorded in
`/tmp/evr-0009-supervisor-binding-test.json` with sibling stdout/stderr logs.

## Evidence boundaries

The wrapper enforces CPU affinity, thread environment and per-process 8GiB
address-space limit; RLIMIT_AS is not a process-tree aggregate RSS limit. The
existing cloud's 8GiB cgroup remains the aggregate constraint. wait4 measures the
leader and descendants it actually waited for, not arbitrary system processes.
The fixed collector is not expected to detach into another process session.
The deadline loop governs the launched collector; a kernel-level uninterruptible
operation, unavailable filesystem, SIGKILL of the supervisor or a pathological
Popen/pre-exec stall cannot be made receipt-complete by this Python wrapper.
These are not reasons to claim a completed result after missing evidence.
No numerical-science claim follows from these tests. The parent should re-read
this small repair and require the Linux tests before freezing/executing.

## Parent-requested lock binding follow-up

After the original repair, the parent requested one additional evidence binding.
`bind_collection` writes a fresh `collection-binding.json` after child reaping,
with the raw execution receipt SHA and the regular, non-symlink selection-lock
SHA/size. It never rewrites the execution receipt. A missing or unsafe lock is
explicit null; child exit0 with no lock makes the actual CLI return125. Existing
nonzero exits are preserved. The independent saved reviewer consumes this
separate binding. Two additional artificial tests cover hash identity, receipt
preservation, exclusive creation, missing/symlink locks, and preserving exit7/124.
The final hashes and 15-test counts above include this follow-up; the original
13-test evidence remains separately preserved. No scientific input was read.
