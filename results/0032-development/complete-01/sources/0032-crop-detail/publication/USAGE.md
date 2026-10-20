# 0032 explicit text publication

Prepared only. Root supplies the final explicit specification and runs this once
after both prediction and evaluation have completed cleanly. No imports of model,
metric or scientific worker code occur.

```sh
python3 -B work/0032-crop-detail/publication/publish.py PUBLICATION_SPEC.json
```

Specification:

```json
{
  "output": "/absolute/task/work/0032-crop-detail/publication-complete-01",
  "files": [
    {"path": "/absolute/original.json", "published_path": "run/original.json",
     "sha256": "EXACT_ORIGINAL_SHA256", "bytes": 123}
  ],
  "operations": [
    {"kind": "prediction", "receipt": "/absolute/prediction/receipt.json",
     "outer": "/absolute/prediction-supervision/receipt.json",
     "worker_spec": "/absolute/prediction-spec.json",
     "operation_spec": "/absolute/prediction-outer-spec.json"},
    {"kind": "evaluation", "receipt": "/absolute/evaluation/receipt.json",
     "outer": "/absolute/evaluation-supervision/receipt.json",
     "worker_spec": "/absolute/evaluation-spec.json",
     "operation_spec": "/absolute/evaluation-outer-spec.json"}
  ],
  "prediction_seal": "/absolute/prediction/seal.json",
  "evaluation_report": "/absolute/evaluation/evaluation.json"
}
```

Every guard path and both logs beside each outer receipt must occur in `files`.
Include local ADMISSION/SCOPE/METRIC_SCOPE, refine/evaluate/test sources, this
publisher/usage, manufactured-test receipts/logs, both operation specs and
receipts/logs, prediction seal and complete evaluation report. A generated
results README may be an explicit file too. Include all completed prediction
receipt fields; do not hand-trim its 241 frame records. Repeated original paths
are copied once to their first listed destination and declared in the manifest.
Only explicit `.json`, `.py`, `.md`, `.txt` and `.log` files are accepted.

The output must be a fresh direct child of `0032-crop-detail`; the repository is
never modified. No recursive collection or reference traversal occurs.
Upstream installation source is excluded; its hashes/pins remain in specs.
Already published 0030 input metadata need not be duplicated. NPZ/PT, masks,
images, model weights and other binary payloads are forbidden.

The original text is preserved byte-for-byte except literal substitution of the
absolute task workspace prefix with `${WORKSPACE}`. JSON is not compacted or
rewritten, and all fields/rows are retained. Remaining absolute macOS home paths, NUL bytes,
private-key markers and common credential-like strings cause failure. Original
and published SHA256/bytes and the exact transform are recorded per file.
Internal original hashes still refer to original bytes, not sanitized copies.
The manifest records the full sanitized publication spec and publisher hash.
Output bytes are read back and checked after writing.

This checks saved completion and provenance, not model correctness or numerical
metrics. It neither reopens payloads nor creates a new science/audit stage.
Preserve execution stdout/stderr, exit status and source/spec hashes separately.
