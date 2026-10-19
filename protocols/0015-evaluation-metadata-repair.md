# 0015 saved-score metadata repair

After freeze057a5ea1527d3e205987a2780a512d01ed490a87, the32 scientific
worker processes completed once and both16-pair prediction seals were complete.
The first evaluator exited1 with KeyError `call_seconds`, after verifying both
seals and opening the already-known0013 label file. It computed the first row's
errors in memory, but wrote no evaluation file. The process receipt and normalized
traceback preserve this failure; the original private-path traceback is retained
locally with a hash link. No scientific worker failed or is rerun.

The reporting-only bug is that the original0013 record stores API time at
`timing_seconds.estimator_call`, not top-level `call_seconds`. The artificial
cohort fixture incorrectly supplied the latter field, and independent prefreeze
review did not catch that schema error. Correct the fixture to reflect actual
schema and test the new reporting path. Do not weaken prediction validation.

Keep the failed evaluator source unchanged because it is part of the scientific
source seal. A separate `evaluate_saved_repair.py` copies that scorer with only
the timing lookup corrected, plus explicit repair description and its own source
SHA256 in the output. Its imports, seal/source checks, label hash, metric formulas,
coverage penalties and decision thresholds remain identical. It does not accept
new input observations, run estimation, alter a point, omit a pair or optimize any
setting. Publish this repair before one new saved-output scoring invocation,
writing `evaluation-repair.json` and distinct process/log files. The first failed
attempt and all original prediction files/seals remain intact.
