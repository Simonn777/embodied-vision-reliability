# Test-only platform scope correction

CI run 39389319650 reported two failures in the real process lifecycle
tests. Its leak assertion received
`ValueError: invalid process identity/RSS sample`, rather than the expected
residual-child classification; its race assertion reported zero forced
snapshots. The offending process-table row was not saved. Neither a specific
PID/PGID value nor the underlying OS cause can be inferred from this log.
The failure log and executed CI test are copied unchanged under
`exit-supervisor-ci-failed-sources/` (log SHA256
`8a9f04fc31f896d4a0dadf38bc274bd471660946b1d7cdd7e07d81910617afd9`;
test SHA256 `e6c35c3c71b8bcc0167fc0c9843cc1cb7240f095a7e9fe487e320b796b0e5bba`).

The private alternative `test_supervise_exit_checked_portable.py` preserves
the two real lifecycle cases, explicitly scoped to Darwin. This is the
platform of the qualified native runtime and the already preserved first
successful process tests. This does not qualify the monitor on Linux.

Two additional controlled fake-process cases call the unchanged
`supervise_exit_checked.run_spec`: one supplies a stale pre-poll child row
followed by an empty fresh post-exit sample; the other supplies a live
post-exit row and checks failure plus stop/kill attempt recording. Popen,
signal calls and process-group queries are mocked; any real subprocess.run
is forbidden. The ordered snapshot/poll events and all receipts/logs are
retained. These cases test the decision logic without asserting any OS
process-table semantics.

Alternative test SHA256:
`2202236d6ba047c47dae078d3b7d34c9a137ebc4fcb1017450892e3f5d4a8688`.
Production supervisor remains
`8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8`;
neither it nor the old monitor is changed.

Pending parent review and authorization, the first local invocation from P
is qualified Python `-B -m unittest discover -s work/0029-preparation -p
test_supervise_exit_checked_portable.py -v`, with all five thread variables
set to 4 and `EVR_EXIT_TEST_OUTPUT` set to a fresh private case directory.
Darwin should run four cases; Linux CI should run the two controlled cases
and explicitly skip the two real lifecycle cases. No invocation has been
performed while preparing this alternative.
