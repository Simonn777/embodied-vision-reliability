# Saved-only first qualification review

The first two manufactured process tests passed once: unittest 0.277 s;
outer 0.43642620800528675 s, 31,456 KiB, exit 0, no termination, cleanup
errors or remaining descendants. Outer receipt SHA256:
`b247381fc0a1dd9bacacdc954d4a290e4521f385a8191abf2474a2c149049ef8`.
All saved stdout/stderr hashes in the three receipts were rechecked.

The race fixture retained a truthful pre-poll sample with parent 57033 and
child 57036. The fixture explicitly allowed both to exit before returning
that sample. The fresh post-exit sample was empty; the new supervisor
completed without sending signals. The saved proof records
`old_predicate_would_fail: true`. This demonstrates the possible ordering
fault on a manufactured process history, not the missing historical
cat-girl trigger.

The genuine-leak fixture retained child 57029, reparented to PID 1 but still
in the supervised process group, in the fresh post-exit sample. It failed
with `descendants_remain_after_fresh_exit_sample`; SIGSTOP and SIGKILL were
recorded as sent. Its final remaining list was empty. Thus a live leak is
still rejected, and an empty post-cleanup list does not erase the failure.

Executed sources remain in `exit-supervisor-tests-01/sources/`:
supervisor `8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8`;
tests `bab32c2d2c3dc23326a0c95fe643f70ee2f3fbde495c31f2dc5b265bac921b8a`.

The current private test file has only location/output plumbing changes for
later CI: repository detection supports both private and published layouts
(or `EVR_EXIT_TEST_REPO`); the default retained fresh artifact directory is
outside the repository; `EVR_EXIT_TEST_OUTPUT` still overrides it. The two
process scripts, assertions, timing caps and child four-thread environment
are unchanged. Parent CI's two-thread environment does not change the
manufactured child specification, which explicitly sets all five keys to 4.
Current test SHA256:
`e6c35c3c71b8bcc0167fc0c9843cc1cb7240f095a7e9fe487e320b796b0e5bba`.
This location-only revision has not been executed here; root will include it
in the integrated suite. Production supervisor bytes are unchanged.
