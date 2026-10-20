# Fixed crop-detail baseline: complete negative result

The fixed known-method adaptation is closed by its prospective rule. It is not a new method, substantial improvement, official all-object DAVIS result, or manuscript. The research goal remains unachieved.

## Complete results

| Population | Native J&F (%) | Crop J&F (%) | Change (absolute pp) |
|---|---:|---:|---:|
| TRAIN event mean | 95.2091636 | 94.5445103 | -0.6646533 |
| dog-agility | 94.1184338 | 94.2963098 | +0.1778760 |
| planes-water | 97.5378598 | 95.8803292 | -1.6575306 |
| dogs-scale | 93.6104010 | 91.7861939 | -1.8242071 |
| dog-gooses | 95.5699596 | 96.2152080 | +0.6452485 |
| DEV interior mean | 92.1772205 | 88.3961688 | -3.7810516 |
| cat-girl | 92.2110594 | 86.4367073 | -5.7743521 |
| upside-down | 92.1433815 | 90.3556304 | -1.7877512 |

DEV mean J changes -1.1874663 pp and F changes -6.3746370 pp. All four continuation clauses fail: +1 pp mean J&F, both parents positive, J loss at most0.5 pp, F loss at most0.5 pp. No padding, prompt, threshold or selection tuning is used to rescue this fixed recipe.

## What ran

A frozen SAM2.1 tiny video baseline supplies its current native mask and logits. One 10%-padded target crop is encoded at1024, prompted with the EDT-max positive point, full crop box and resized native logits. The highest predicted-IoU mask is stitched back in original coordinates. Native history and pointers are untouched; image-API object-presence behavior is preserved and reported. This is a simplified SAMRefiner-inspired adaptation, not a faithful reproduction of SAMRefiner, SAM2Refiner or SAM-REF.

All87 distinct TRAIN future frames and154 raw DEV frames were predicted and sealed before any current annotation decode. The two DEV initial outputs are preserved native copies. All241 native zero-logit masks match saved native predictions exactly. Metrics include all96 fixed TRAIN horizon events (duplicates retained, four equal parents, including4 final-frame events), and all150 DEV interiors (two equal parents); all154 raw DEV rows remain. Published native DEV means and each parent reproduce within1e-12.

There are 239 added image-encoder calls and 239 mask-decoder calls, zero video-tracking/memory-encoding calls and zero current-label decodes in the producer. Model state SHA remains 8132009e1c015fadda53e7e72db024e111d256e6ee8b67f396ee7ba744418b29; no network attempts. Both operations exit cleanly.

## Costs and evidence

Prediction outer wall time is 145.796326875s, peak group RSS 1,384,160KiB. Metric outer wall time is 1.975912333s, peak 78,672KiB. Six manufactured geometry tests passed before execution (0.211948125s separate process). The inherited native-cache operation took458.300764250s and included cache/GT-memory preparation; it is not free, nor a clean matched end-to-end timing. No speed claim or equal-cost accuracy claim is made.

The public text bundle preserves all241 per-frame inference records, all96 TRAIN metric events, all154 DEV raw metric rows, all parents, source/spec/output hashes, seal, logs, costs and decision. Binary predictions and three-candidate diagnostics remain local and are represented by their full hash inventories; no dataset pixels, pretrained weights or private environment are published. Paths are literal aliases, not executable locations in a bare clone. Reproducing scientific inference requires reconstructing the declared workspace, qualified pinned runtime and authorized dataset/cache inputs.

The exact fixed recipe is closed. A different mechanism needs a concrete measurable information advantage and strong same-information controls. Learned error gates, crop detail fusion and image/temporal refinement already have close precedents. These source comparisons and this negative result do not establish an invention. All six parents remain exposed development; the unchanged two-independent-held-out-source +3 official J&F gate, recording-level uncertainty, component-loss limits, cost-matched controls, ablations and independent replication still precede any manuscript.
