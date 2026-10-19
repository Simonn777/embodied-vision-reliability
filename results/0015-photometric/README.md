# 0015 predictions complete; first report failed

Scientific freeze: `057a5ea1527d3e205987a2780a512d01ed490a87`.
All16 photometric and16 geometric workers completed once in22.686077seconds.
Both complete prediction seals preceded evaluation. No worker is rerun.

The first evaluator failed on an absent top-level baseline `call_seconds`
reporting field. Correct schema is `timing_seconds.estimator_call`. No score
file was written; the failure and original source remain preserved. The
[metadata-only repair](../../protocols/0015-evaluation-metadata-repair.md)
keeps metrics, gates, predictions and source seals unchanged. Accuracy is not
claimed before that separately frozen saved-output operation.

[Supervisor receipt](supervisor.process.json), [full process log](supervisor.log),
[photo seal](photometric/seal.json), [pose seal](geometric/seal.json),
[failed evaluator receipt](evaluation.process.json), [traceback](evaluation.log),
[private-prefix normalization hashes](evaluation-log-normalization.json).
