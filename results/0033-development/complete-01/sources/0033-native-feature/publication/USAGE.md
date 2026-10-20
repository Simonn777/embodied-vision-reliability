# 0033 explicit text publication

Prepared only. Root supplies the final explicit specification and runs this once
after fit, prediction and evaluation have completed cleanly. No imports of model,
metric or scientific worker code occur.

```sh
python3 -B work/0033-native-feature/publication/publish.py PUBLICATION_SPEC.json
```

Specification:

```json
{
  "output": "/absolute/task/work/0033-native-feature/publication-complete-01",
  "files": [
    {"path": "/absolute/original.json", "published_path": "run/original.json",
     "sha256": "EXACT_ORIGINAL_SHA256", "bytes": 123}
  ],
  "operations": [
    {"kind": "fit", "receipt": "/absolute/fit/receipt.json",
     "outer": "/absolute/fit-supervision/receipt.json",
     "worker_spec": "/absolute/fit-spec.json",
     "operation_spec": "/absolute/fit-outer-spec.json"},
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
Include local ADMISSION/SCOPE/METRIC_SCOPE, head/train/predict/evaluate/test sources,
this publisher/usage, manufactured-test receipts/logs, all three operation specs
and receipts/logs, prediction seal and complete evaluation report. Also include
the exact `normalization`, `orders` and `training_log` text files named in the fit
receipt, with their original hashes and byte counts. A generated
results README may be an explicit file too. Include all completed prediction
receipt fields; do not hand-trim its 241 frame records or 482 arm-output
descriptors. Preserve all 5,472 JSONL training attempts, including every loss,
gradient and timing field. Both terminal checkpoint hashes and sizes remain in
the fit/prediction receipts and manifest; checkpoint payloads are never opened
or copied. Repeated original paths
are copied once to their first listed destination and declared in the manifest.
Only explicit `.json`, `.jsonl`, `.py`, `.md`, `.txt` and `.log` files are accepted.

The output must be a fresh direct child of `0033-native-feature`; the repository is
never modified. No recursive collection or reference traversal occurs.
Upstream installation source is excluded; its hashes/pins remain in specs.
Already published 0030 input metadata need not be duplicated. NPZ/PT, masks,
images, model weights and other binary payloads are forbidden.

The original text is preserved byte-for-byte except literal substitution of the
absolute task workspace prefix with `${WORKSPACE}`. JSON is not compacted or
rewritten, and all fields/rows are retained. JSONL lines are parsed for validity
and complete fixed training identities, but are not reserialized. Remaining absolute macOS home paths, NUL bytes,
private-key markers and common credential-like strings cause failure. Original
and published SHA256/bytes and the exact transform are recorded per file.
Internal original hashes still refer to original bytes, not sanitized copies.
The manifest records the full sanitized publication spec and publisher hash.
Output bytes are read back and checked after writing.

Required coverage is 228 fit frames and 5,472 two-arm attempts, 241 unique
prediction frames and 482 arm predictions, all 96 TRAIN metric events (87
unique future frames), and 154 raw DEV frames with 150 interior metric frames.
TRAIN metrics describe a fit-subset horizon population. Separate whole-operation
fit, prediction and evaluation costs are retained; nested timings are not added
again. A continuation failure is publishable and is preserved without rescue,
selection or modification. The supplied README must describe saved results as
a known supervised capacity baseline, with no method or paper-success claim.

This checks saved completion and provenance, not model correctness or numerical
metrics. It neither reopens payloads nor creates a new science/audit stage.
Preserve execution stdout/stderr, exit status and source/spec hashes separately.
