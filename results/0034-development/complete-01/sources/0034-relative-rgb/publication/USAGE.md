# 0034 explicit text publication

Prepared source only. The parent supplies an explicit file specification and
executes this helper after all three stages complete with clean outer receipts.
The helper imports only the standard library. It does not import workers, open
feature/prediction arrays, fit models, recalculate metrics, or modify the repository.

```sh
python3 -B work/0034-relative-rgb/publication/publish.py PUBLICATION_SPEC.json
```

The operation `kind` strings are exactly the worker's stage strings:
`prepare`, `fit_predict`, and `evaluate`. Each requires its own seal.

```json
{
  "output": "/absolute/task/work/0034-relative-rgb/publication-complete-01",
  "files": [
    {"path": "/absolute/original.json", "published_path": "run/original.json",
     "sha256": "EXACT_ORIGINAL_SHA256", "bytes": 123}
  ],
  "operations": [
    {"kind": "prepare", "receipt": "/absolute/prepare/receipt.json",
     "outer": "/absolute/prepare-supervision/receipt.json",
     "worker_spec": "/absolute/prepare-spec.json",
     "operation_spec": "/absolute/prepare-outer-spec.json",
     "seal": "/absolute/prepare/seal.json"},
    {"kind": "fit_predict", "receipt": "/absolute/fit-predict/receipt.json",
     "outer": "/absolute/fit-predict-supervision/receipt.json",
     "worker_spec": "/absolute/fit-predict-spec.json",
     "operation_spec": "/absolute/fit-predict-outer-spec.json",
     "seal": "/absolute/fit-predict/seal.json"},
    {"kind": "evaluate", "receipt": "/absolute/evaluate/receipt.json",
     "outer": "/absolute/evaluate-supervision/receipt.json",
     "worker_spec": "/absolute/evaluate-spec.json",
     "operation_spec": "/absolute/evaluate-outer-spec.json",
     "seal": "/absolute/evaluate/seal.json"}
  ],
  "evaluation_report": "/absolute/evaluate/evaluation.json"
}
```

Every guard artifact, both original logs beside each outer receipt, and all 15
executed source files must occur in `files`. Include the tests and manufactured
qualification receipts/logs, local admission/scopes, this publisher and usage,
all operation specifications/receipts/seals, and the complete evaluation report.
The parent can include a prepared result README and other explicit text notes.
Do not recursively collect directories or upstream installation sources.

Every JSON output named by a stage receipt is mandatory: all 40 individual
terminal models, all 40 complete optimizer traces, `terminal-models.json`, and
`evaluation.json`. The full receipts preserve all 232 prepared feature
descriptors, all four folds' normalization and training identities, fit weights
and failures, and all 232 prediction descriptors representing 2,552 masks.
Failed fit attempts and native fallbacks remain intact; `completed_fits=40`
means 40 finished attempts, not necessarily 40 successful fits. No optimizer
closures, partial failure evidence, coefficients, timings or diagnostic fields
are removed.

Only explicit UTF-8 `.json`, `.jsonl`, `.py`, `.md`, `.txt` and `.log` files are
accepted. Binary NPZ/PT, model weights, images, masks and feature payloads are
never opened or copied. Binary output hashes and sizes are checked for inventory
structure and retained as recorded by the completed workers; this exporter does
not independently rehash binary payloads. Existing 0030 input payload metadata
need not be duplicated: the exact six inherited bindings and historical cache
cost reference remain in the stage receipts/specs.

The output must be a fresh direct child of `0034-relative-rgb`. Repeated original
paths are copied once to the first specified destination and explicitly listed
as deduplicated inputs. Distinct paths remain distinct even if their bytes match.
No output directory is created until the completion, source, cross-stage
binding, inventory and privacy checks pass.

Original formatting and every JSON key/value/row are preserved. The only text
transform replaces the absolute task workspace prefix with `${WORKSPACE}`.
JSON/JSONL is validated without reserialization; no floats are rounded.
Remaining absolute macOS home paths, NUL bytes, private-key markers and common
credential-like strings cause failure. Each manifest entry contains original
and published SHA256/byte counts and the exact transform. Internal receipt
hashes continue to refer to the original bytes, not the sanitized copies.
All originals are rechecked before writing, and every published file is read
back for byte equality. Preserve execution stdout/stderr and exit status
separately when the parent runs the export.

Required saved coverage is 232 raw frames, 40 named fit attempts, 80 individual
terminal/trace JSONs, one terminal aggregate, 232 prediction archives and 2,552
arm masks. The report retains all 232 metric rows, 224 primary interiors and
96 descriptive secondary events with 87 unique frames and four final-frame
events. The helper checks these records and their bindings; it does not
recompute the J/F means, gate clauses or selected controls.

A failed scientific continuation gate is publishable. A clean complete operation
with retained fit failures is also publishable, with continuation false.
Incomplete or unclean operations are rejected. The manifest records each
whole-operation wall time and peak RSS separately; nested fit/closure timings
are preserved but not added again. The supplied README must state the saved
development outcome without an official-benchmark, novelty or paper-success
claim. This export is a text/provenance check, not an independent scientific
audit or another experiment.
