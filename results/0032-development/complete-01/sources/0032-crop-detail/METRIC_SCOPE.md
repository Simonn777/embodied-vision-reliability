# Fixed crop baseline: saved-output metric stage

Prepared source only, 2026-10-20. No model, image encoder, SAM2 import, fitting, threshold choice, crop choice or hyperparameter selection occurs here. Root supplies and admits the execution specification later. No scientific inputs or outputs were opened to prepare this worker.

CLI: qualified CPython 3.12.13, normal site initialization, `python -B -W error evaluate.py SPEC.json`. Four thread environment variables must each equal4; OpenCV is explicitly serial. The complete operation receives its own existing exit-checked supervisor, 1800-second/8-GiB envelope and fresh output directory. There is no successful-subset report.

## Specification and input contract

Required specification keys: `output`, `python`, `campaign_root`, `evaluation_root`, `source_sha256`, `bindings`, `baseline_reference`. The exact source map contains this worker, this scope, sibling `ADMISSION.md`, `refine.py`, `SCOPE.md`, campaign `common.py` and `cache_baselines.py`, and unchanged `evaluate_jf.py`, `evaluate_baseline.py`, `evaluate_rollout.py`. Each absolute filename maps to SHA256. The specification itself is hashed and checked again at completion.

`bindings` has exactly ten `{path,sha256,bytes}` descriptors: the six existing `cache_receipt/cache_outer/cache_operation/cache_worker_spec/mapping/data_index`, and `refinement_receipt/refinement_outer/refinement_operation/refinement_worker_spec`. `baseline_reference`, separately, binds the original published native baseline `evaluation.json`; it supplies existing DEV means for parity checking, never candidate selection. No new execution specification is created by this worker.

The existing `evaluate_rollout.admit` verifies the clean old native-cache supervisor and clean new refinement supervisor, spec/command/receipt/log bindings and limits. Refinement must use the same six metadata descriptors, have unchanged model hashes matching cache, no offline attempts, zero current-label decodes, and all241 outputs. Its source map is rehashed without importing the model or refinement code.

Before any annotation decoding, the worker requires sibling producer `seal.json` with schema `evr0032.crop-detail-seal.v1`, `complete:true`, `receipt_sha256`, `source_sha256`, `spec_sha256`, `frame_count:241`, `outputs` and `diagnostics`. These must equal the completed producer receipt. The seal already exists on entry; the metric worker does not construct it retroactively. All generated outputs/diagnostics, all241 corresponding native predictions, and all241 encoded annotation files are hash-checked first. Every prediction is then structurally decoded with `allow_pickle=False`; both DEV frame0 masks must equal native exactly. Only afterward is the metric receipt marked `predictions_sealed_before_gt=true` and any label decoded. Source/input fingerprints are checked again at the end.

Prediction NPZs contain exactly `masks:bool[1,H,W]` and integer `object_ids:[target_id]`, at the original480p dimensions. The six rows, original frame counts, roles, anchors, shapes and target IDs must match the admitted native cache and index. Paths must remain within the producer/cache output roots. The target is the cache's actual initial smallest-positive object ID, excluding255; no object union or reassignment.

## Metrics and full population

Reuse the unchanged official-formula `evaluate_jf.jf` and `_means`: binary IoU; official boundary construction and Euclidean disk radius `ceil(.008*hypot(H,W))`;255 is background in this semisupervised path. Values and differences use `[0,1]`; multiply by100 for percentage points.

- TRAIN: `dog-agility`, `planes-water`, `dogs-scale`, `dog-gooses`. Each has12 fixed anchors `1+i*(N−5)//11` and horizons1/3. Evaluate87 distinct frames once, then retain all96 horizon events, including duplicates and four final-frame events. Exactly24 events receive equal weight within each parent; average the four parent means equally. This is privileged development horizon-set reporting, not full-video TRAIN benchmark scoring.
- DEV: `cat-girl`, `upside-down`. Retain all154 raw frames and their J/F/JF. Exclude each sequence's first and last frames only from its mean, leaving150 interiors. Average two parent means equally. Native, refined and their deltas are exposed for every raw frame and parent.

Native DEV parent J/F/JF and equal-parent means must reproduce the source-bound published native reference with absolute tolerance `1e-12` and no relative tolerance. No alternative interpolation, mask threshold or original-size rescaling occurs in this stage.

The fixed ADMISSION continuation criteria are evaluated on DEV: equal-parent JF gain≥.01, strictly positive JF gain on both parents, equal-parent J loss≤.005 and F loss≤.005. All clauses are reported individually. Passing would support continuation around a stronger **known crop baseline**, not novelty, official all-object DAVIS performance or the two-source paper gate.

`evaluation.json` retains all TRAIN events, all DEV frames, parent/overall summaries, continuation clauses and separately referenced native-cache/refinement outer costs. `receipt.json` records source/spec/input hashes, label-attempt/completion counts, complete coverage, zero model calls and metric wall time. Nested times are never summed into a latency advantage; metric outer wall/RSS remain in its own supervisor receipt.
