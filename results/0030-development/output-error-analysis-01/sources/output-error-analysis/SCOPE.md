# Native DEV output-error location — prepared, not executed

This is a descriptive saved-output analysis, not a model intervention, selected-subset result, gain claim, new training experiment or final-data access. Run only after the active-fit operation ends. Use the already qualified Python/NumPy/cv2/PIL runtime; there is no Torch import, model loading, RGB access, inference, encoder, optimizer or network call.

Bind the completed **native** own-rollout, its completed evaluation and their original operation/outer receipts/logs plus the complete data mapping/index and baseline cache receipts. Reuse unchanged `evaluate_rollout.admit` for the two exit-checked operations and the actual legacy cache operation. Verify all 154 prediction and annotation descriptors before decoding; then decode exactly the 150 official interior frames: cat-girl 1–87 and upside-down 1–63. First/last frames remain excluded by the existing metric, not selected by error. Every interior frame and both parents remain in all summaries, including empty or perfect masks.

Reuse the unchanged metric's PNG/NPZ readers, target ID, original resolution, `boundary_map`, `boundary_radius` and `jf`. Treat 255 as background as in the existing official semisupervised convention. Recomputed per-frame J/F/JF, per-parent means and equal-parent cohort mean must equal the already saved native evaluation **exactly**; mismatch fails the operation. No alternate threshold, resized mask or object matching is introduced.

For each frame, report TP/FP/FN/TN and error mass in four disjoint bins relative to the official GT boundary: within radius `r = ceil(.008*hypot(H,W))`; outside r but within `2*r`; beyond `2*r`; and undefined because no GT boundary exists. Use the same exact Euclidean disk dilation, with zero exterior. The last bin is essential: an empty/full-image GT mask can have no boundary, and its errors must not silently count as measured far-boundary errors. All four bins partition every FP/FN pixel. Report raw numerators; a frame with no region error has null error fractions rather than invented evidence.

Also report boundary precision, recall and F at r, and at 2r as a **descriptive tolerance diagnostic**, not a changed official F. Follow the existing official source's empty-boundary conventions: prediction boundary empty / GT nonempty gives P=1,R=0; prediction nonempty / GT empty gives P=0,R=1; both empty gives P=R=1. At r the reconstructed F must equal the metric helper. Parent precision/recall/F summaries are means of per-frame values; do not compute a new harmonic mean of averaged precision and recall and call it official F.

Report exact logical flags rather than a fitted failure classifier: target absent, prediction empty, both empty, complete target miss (GT nonempty, TP=0), empty prediction with target present, disjoint nonempty prediction/target, foreground prediction when target absent, and exact mask. Flags may overlap and do not identify identity swaps or causes. Keep per-frame rows, each parent's metrics/error counts, and separately labeled all-frame pixel sums and frame averages. Pixel totals are size/duration-weighted descriptive statistics, not independent samples or a statistical gain test. Boundary concentration does not imply that a causal repair can observe or correct those pixels.

Invocation:

`QUALIFIED_PYTHON -B -W error analyze.py SPEC.json`

Do **not** add `-S`: the qualified site packages provide the array/image metric dependencies. Use the existing complete-operation supervisor/four-thread environment; this worker disables cv2 internal threads. No execution while model training is active.

Spec fields: fresh absolute `output`, `evaluation_root` pointing to the existing metric source directory, `source_sha256`, and descriptor-valued `bindings`. Source bindings must contain this script and unchanged `evaluate_jf.py`, `evaluate_baseline.py`, `evaluate_rollout.py`; the parent may also bind this scope and the already saved official metric source.

Bindings are exactly the native evaluator's original ten keys (`cache_receipt`, `cache_outer`, `cache_operation`, `cache_worker_spec`, `rollout_receipt`, `rollout_outer`, `rollout_operation`, `rollout_worker_spec`, `mapping`, `data_index`) plus `evaluation_receipt`, `evaluation_outer`, `evaluation_operation`, `evaluation_worker_spec`, and `evaluation` (the existing evaluation.json). Each is `{path,sha256,bytes}`. The evaluator's saved bindings must match the same ten descriptors; its native arm, 154/150 coverage and report hash must agree.

Outputs are `analysis.json` and a partial-preserving `receipt.json`, schema `evr0030.native-output-error-analysis.v1`, with exact saved-metric consistency, all 150 rows, source/input hashes and inner elapsed time. The outer supervisor supplies complete wall/RSS/exit evidence. No new framework, scientific selection, threshold optimization or conclusion that output repair will help is added.
