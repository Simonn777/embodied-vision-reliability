# Exact query deletion did not improve real-image classification

**NO_GO under the unchanged prospective rule.** On the fixed 128 EuroSAT
development images, exact leave-one-out (LOO) StatA classified 61 correctly,
versus 65 for the strongest fixed control, zero-shot CLIP. This is a net loss
of four images (−3.125 percentage points), not the required gain of eight.
Both batches lost accuracy. Stop this candidate for this cohort; do not tune
prompts, anchors, selection or runtime limits to rescue it. No manuscript follows.

## Complete results

| Fixed method | Correct /128 | Accuracy |
|---|---:|---:|
| Zero-shot CLIP |65|50.78125%|
| Native StatA α=1 |62|48.43750%|
| Native StatA α=4 |52|40.62500%|
| Exact LOO α=1 |61|47.65625%|
| OTHER-out α=1 |63|49.21875%|

| LOO compared with | Corrections | Harms | Net | Batch 1 / Batch 2 net |
|---|---:|---:|---:|---|
| Zero-shot |7|11|−4|−1 / −3|
| Native α=1 |2|3|−1|−1 / 0|
| Native α=4 |10|1|+9|+10 / −1|
| OTHER-out |1|3|−2|−2 / 0|

The +9 comparison against α=4 does not pass the fixed rule: that control is
weaker than zero-shot, native α=1 and OTHER-out. All 640 predictions are valid;
no case was dropped. There were no singleton or doubleton examples in the
declared within-batch occupancy strata, so singleton safety is unassessed.
Zero-shot made 63 errors and disagreed with LOO on 37 images. The frozen
low-headroom exception does not apply. [Full score](cloud/scratch/0010-headroom-output/scores.json).

## What actually ran

[PR38 and exact CI](../../docs/0010-headroom-freeze-publication-verification.md)
preceded the single run. Main `49c0c0b22f89e5d395130dec63bafad9377ba5c5`
and [the canonical receipt](../../protocols/0010-real-data-headroom-freeze.json)
bind the fixed selection, five arms, code and actual saved inputs.

The existing cloud CPU completed 128 images in 32 encoder calls, zero text
calls, 132 fitted states and 264 authored parity executions. All predictions
were sealed before one complete-cohort score. The scientific supervisor
observed exit 0, **28.043947 seconds** wall and **1,321,436 KiB** peak RSS;
four CPU cores/threads, 8 GiB address-space cap and a 900-second hard limit.
There was no timeout. Internal encoding took 6.892998 seconds; fitted states
including parity took 12.244390 seconds. This is an expensive headroom
reference, not an efficient deployment method.

The parent directly retrieved [actual tool returns](parent-tool-observations.json),
including the sole scientific launch, independently of the cloud's final prose.
The launch tool reported 27,978 ms and exit 0; the native supervisor's wall above
is its separate measurement. Cloud saved-output arithmetic and preservation
review exited 0, with 11,441 checks. It performed no model, solver or production
scorer rerun. [Execution](cloud/scratch/0010-headroom-supervision/execution.json)
and [saved review](cloud/scratch/0010-headroom-control/saved-review.json).

Three auxiliary failures remain explicit: the initial inventory command exited
2; the first preflight exited 1 when a runtime path outside the workspace was
treated as workspace-relative; and the first export checker exited 1 because
its privacy pattern matched its own source literal. Corrected inventory path
handling and URL checking did not change the scientific code or repeat science.
Both original and corrected scripts and recorded failures remain in the archive.

The [independent parent review](independent-review/REVIEW.md) independently
reconstructs all640 argmax predictions, every paired/batch/occupancy table and
the unchanged NO_GO decision. Source/selection/seal and actual-tool closure
also pass. Its portable reviewer does not import the production scorer.

## Evidence and limits

The parent verified all 37 exported text files (405,907 bytes) and the transfer's
per-file, JSON and gzip hashes. Gzip SHA256:
`94d360c8bad2c33fd7eff7255d9c04fa60d703619aeb445d8eb8bd09bc4d8439`.
The [copy manifest](cloud/copy-manifest.json) identifies every original file.
The preflight public summary is an explicitly labeled derivative; the complete
2,060,917-byte inventory remains in the cloud with a recorded hash. Preservation
of 9,818 old files is supported by that cloud check, not a new local byte check
of every private file. The launch-observation JSON is an assistant transcription;
the parent tool record is the independent evidence of the actual return.

Private feature arrays remain cached and were not exported or published.
Their hashes, shapes and cloud normalization checks are saved. No image,
checkpoint, wheel, new dataset or weight was downloaded for this operation.
No paid service was used. Direct preparation response bytes remain 456,087,034;
the historical all-HTTP total remains UNKNOWN.

This is development evidence from two computational batches, not independently
verified scenes or an unseen population estimate. CLIP pretraining exposure is
unknown. LOO and OTHER-out have different readout peer sets; the result does not
identify a pure self-influence mechanism. Generic self-exclusion has prior art.
The conditional common-peer and validation follow-up is **not activated** by
this failed gate. Preserve all results and used-sample exclusions while the
broader research program evaluates a distinct, justified hypothesis.

AI assistance contributed the design, implementation and review; see
[contributions](../../docs/CONTRIBUTIONS.md). No institutional endorsement or
conference-level improvement is claimed.
