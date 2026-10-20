# Portable supervisor test adaptation: source-only review

Verdict: **no blocking defect identified in the test-only adaptation**.

Compared current repository test `e6c35c3c71b8bcc0167fc0c9843cc1cb7240f095a7e9fe487e320b796b0e5bba` with private `test_supervise_exit_checked_portable.py`, SHA-256 `2202236d6ba047c47dae078d3b7d34c9a137ebc4fcb1017450892e3f5d4a8688`. The production supervisor remains byte-identical at `8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8`.

The original two real process-lifecycle test bodies and assertions remain intact. The Darwin guard limits their operating-system qualification claim to the actual native runtime platform; it does not alter production process validation or convert the failed Linux CI into a success. Class-specific retained artifact directories avoid collisions when both real and controlled tests run.

The added portable cases invoke the actual `run_spec` function while substituting a deterministic process/monitor boundary. They explicitly require the event prefix `snapshot → poll → snapshot`, consume the complete prescribed sample sequence, and preserve an event/receipt proof:

- The stale case supplies a descendant only in the pre-poll sample, then empty post-exit/final samples. It requires completed status and no signals.
- The live-residual case supplies no descendant before polling but a descendant in the fresh post-exit sample. It requires failure, that exact triggering snapshot, and recorded `SIGSTOP`/`SIGKILL` to the child group even though the final sample is empty.

These opposite histories test the new decision rule rather than merely asserting a mock return value. Popen, process sampling and signaling are mocked; accidental real `subprocess.run` calls are rejected. Thread environment and new-session arguments are checked. The fake cases do not qualify Linux process-table parsing, actual signal delivery, or real cleanup timing.

The reported Linux CI failure is only established as generic process-row validation failure; its offending row was not preserved. This change neither diagnoses that row nor claims a Linux production-monitor repair. Preserve the original failure and the previous Darwin qualification separately.

Review scope: source/diff/hash reads only. No test, module import, model, scientific data, network, or repository mutation was performed by this reviewer.
