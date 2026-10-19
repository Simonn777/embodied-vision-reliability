# Incomplete artificial timing record — not admissible

The first invocation after freeze c2c48eddfb38626abddcae438b39269328a63ce0
exited1 while serializing a NumPy boolean. The in-memory loop finished, but
`diagnostic.json` truncates inside the first timed entry and is intentionally
preserved as invalid JSON. Later timings and complete validation metadata
were not saved and cannot be recovered. Its early `admission_passed: true`
header is not a complete or usable gate decision. Warmup times are not
substituted for the missing formal record.

The [log](run.log) retains the traceback with this local workspace prefix
replaced by `<workspace>`; the untouched log is retained in local work storage.
The [execution record](execution.json) records the actual failure. No real
coordinate read or development replay was admitted.

The [prospective repair amendment](../../../protocols/0012d-lazy-grid-screening.md)
permits one explicitly disclosed technical retry after converting agreement
values to built-in bool and testing complete serialization. It uses the exact
same native binary, fixture and gates, and a new output directory. This is not
a claim that the originally intended one-invocation protocol was achieved.
