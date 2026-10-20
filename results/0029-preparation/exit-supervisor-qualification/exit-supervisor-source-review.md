# Exit-checked supervisor: independent source review

Verdict: **no blocking source defect identified for the prospective technical replication**. This is a source-only review, not an independent execution qualification.

Reviewed bytes:

- `supervise_exit_checked.py`: `8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8`
- `test_supervise_exit_checked.py`: `bab32c2d2c3dc23326a0c95fe643f70ee2f3fbde495c31f2dc5b265bac921b8a`
- Reused repository `experiments/0025c-rs-noise/supervise.py`: `4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd`

The decisive ordering is correct: lines 126–140 retain the pre-poll sample, observe child exit through `poll`, and obtain a fresh sample before classifying remaining descendants. A descendant in that fresh sample still causes failure and group cleanup; the correction does not waive a genuine observed survivor. Final sampling can independently reject survivors. Nonzero exit, sampling exceptions, cleanup errors, residual PIDs, sampled RSS excess, or timeout prevent `completed`.

The receipt preserves the observed exit time, sample intervals and full triggering rows, declared/observed process group, signal attempts and outcomes, and cleanup/final samples. Known process groups remain available to the unchanged monitor after parent exit. Cleanup refuses to signal the supervisor's own group. Four-thread environment requirements, sampled 8 GiB group RSS, maximum 1800-second operation limit, fresh output directories, and log/output hashes remain explicit.

The two manufactured tests exercise distinct cases: a truthful pre-exit snapshot returned after parent/descendant completion, and a live child remaining after parent exit. They preserve process receipts, logs, and the forced ordering witness. The first test correctly limits its implication to the constructed history.

Scope and timing: the reviewer read and hashed sources only; no imports, process tests, models, scientific data, or GT were executed/read. **This final review was recorded after the parent reported its first manufactured qualification had already passed both tests** (outer wall 0.436426208 s, 31,456 KiB, exit 0, clean). Those execution figures are parent-reported here, not independently re-audited. This document must not be described as completed before that qualification.

The monitor still samples PID/PPID/PGID/RSS, without process start-time identity or STAT. It does not provide continuous process-tree observation, instantaneous resource enforcement, or proof against every possible escaped/reused process identity. The recorded exit time is observation time, not kernel exit time. Source/commit/spec admission remains the dispatcher's separate responsibility. None of these inherited limits proves the cause of historical `cat-girl`; its original failed outer receipt and ineligible status remain unchanged.
