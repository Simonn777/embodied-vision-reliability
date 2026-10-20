# Causal rollout evaluator, prepared only

`evaluate_rollout.py SPEC_JSON` uses the unchanged `evaluate_jf.py`. It imports `sha`, `require`, and `save` from `evaluate_baseline.py`, so all three files belong in `source_sha256`.

The spec has a fresh absolute `output`, the three-file `source_sha256` mapping, and descriptor-valued `bindings` (`path`, `sha256`, `bytes`) with these ten keys:

- `rollout_receipt`, `rollout_outer`, `rollout_operation`, `rollout_worker_spec`
- `cache_receipt`, `cache_outer`, `cache_operation`, `cache_worker_spec`
- `mapping`, `data_index`

The rollout requires the new exit-checked supervisor; the baseline cache requires its actual legacy supervisor. The two complete DEV recordings must remain ordered `cat-girl`, `upside-down`, with all 154 predictions. Cache metadata fixes target ID, dimensions, and full label mapping. The rollout checkpoint, complete prediction inventory, operation receipts/specs/logs, metadata, and evaluator sources are checked. No Torch/model imports occur.

Outputs are `evaluation.json` (all raw frames plus the 150 interior-frame, equal-parent summaries) and `receipt.json` (complete-operation timing, bindings, `model_calls: 0`, and failure retention). Both initial and final raw metrics remain saved. The existing qualified runtime runs normally with `-B -W error`; it must not use `-S`, because NumPy/cv2/PIL are required.

This driver is source-only preparation. It has not been executed or imported, and no rollout output or new label was accessed while the teacher was active.
