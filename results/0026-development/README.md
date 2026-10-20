# 0026 development result: gate failed

The frozen `G_lift` selector scored **73.5881010546 AUC@5**, versus
**74.1396701117** for the strongest prescribed control, `F_all`: a difference
of **−0.5515690571 percentage points**. The development gate failed. This frozen
variant is closed; these results do not establish a new-method gain, admit a
manuscript, or complete the research goal.

| Predeclared clause | Observed | Pass |
|---|---:|---|
| AUC@5 gain at least 2 pp | −0.551569 pp | No |
| Positive gain in at least 3 of 5 scenes | 2 of 5 | No |
| No scene loses more than 1 pp | London Bridge: −2.250535 pp | No |
| AUC@20 loss at most 0.5 pp | 0.137892 pp | Yes |

All **80 previously exposed development pairs** remain in the evaluation.
The metric is the exact empirical directed pose-error AUC in percent; invalid
estimates receive 180 degrees. This is not the older trapezoidal summary or an
untouched-test result. Failed and unavailable new proposals remain recorded;
the frozen selector rules retain their incumbent fallback. See the complete
[evaluation](evaluation-01.json), including every method, scene, selection,
proposal status and charged-time summary.

The source freeze was `b596542f9eee399219ac03fdbcd91b33e8417fd6`.
The saved [CI receipt](source-ci.json) records a successful
[exact-head run](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39363242000)
before scientific execution. The [prediction seal](predictions-01/seal.json)
and [inner outer-receipt](predictions-01/outer-receipt.json) record all 80 completed
pair workers. All 80 final records, 80 partial checkpoints, 80 process receipts,
80 logs, the job record and seal are retained, including intermediate states.

| Supervised operation | Exit | Outer wall seconds | Peak sampled group RSS (KiB) |
|---|---:|---:|---:|
| [Prediction](prediction-operation-01/receipt.json) | 0 | 53.000115 | 124,080 |
| [Evaluation](evaluation-operation-01/receipt.json) | 0 | 0.541216 | 62,912 |
| [Independent saved-output audit](audit-operation-01/receipt.json) | 0 | 0.532888 | 66,096 |

Each operation receipt records no termination, no remaining descendant PIDs and
no cleanup errors. Specifications and stdout/stderr logs are included. These
are operation timings, not a claimed matched-cost speed advantage.

The [audit](audit-01/audit.json) passed **14,496 comparisons with zero failures**,
covering bindings, selector decisions, component sums and pose metrics. It
rehashed 686 files and did not rerun the scientific-array pipeline or objective
fits. It is an independently implemented check within the same AI-assisted
project, not an independent research-team replication. The exact helper copies
are [auditor](helper-sources/audit_saved_outputs.py) and
[supervisor](helper-sources/0026-supervise.py).

## Publication integrity

[publication-bindings.json](publication-bindings.json) records every copied
file's private-relative origin, original SHA256/size, public SHA256/size and
transformation. Files without private paths are byte-identical. The only
permitted textual transformations replace the absolute project prefix with
`<project>`, then the user-home prefix with `<user-home>`. Original receipts,
seals, logs and source snapshots retain their embedded **original** hashes;
those hashes are not silently replaced with hashes of redacted copies. Use the
publication manifest to verify the public counterparts.

Private originals remain unchanged. This package contains no image, depth or
match cache, model, compiled binary, or private dataset payload. Packaging did
not repeat inference, optimization, scoring or the numerical audit. The
[frozen protocol](../../experiments/0026-calibration-moments/protocol.md) and
[preparation evidence](../0026-preparation/README.md) remain separate.
