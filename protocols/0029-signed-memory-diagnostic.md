# 0029: small signed-memory development diagnostic

2026-10-20. **Prospective development protocol; no outcome has been observed.** The scientific cohort and decisions were fixed using cached metadata before scientific image/mask access. This tests a necessary current-frame information question, not a streaming repair policy or a new contribution.

**Pre-data revision:** replace the initial arbitrary half-query coverage requirement by the logically necessary video coverage below, and permit a separate complete ≤1,800-second operation per video. The original private proposal is retained as `work/0029-preparation/diagnostic-design-v1-before-evidence-correction.md`, SHA256 `ac2451b456f0836909395d6886fa2f682c08422f5cdfaf48057f809b62d55e7a`. No data, model execution, labels, schedule shortening or outcome informed this correction.

**Question:** can causal labels transferred from the supplied initial mask select a beneficial *current* memory deletion better than the same-bank ranking controls, ordinary mask alternatives, or using those identical labels directly as prompts? A correct sparse label need not predict dense mask improvement. Generic delayed repair is already covered by Det-SAM2v2; memory recollection, distractor memories, and candidate verification also have close priors. Passing this diagnostic would not settle those overlaps.

## Fixed cohort and 48 queries

Use exactly the six parent-hash candidates in the hash-bound private metadata file `work/0029-screening/video-cohort-plan.json` (SHA256 `f4962c6c8b3ca0806ee4bb751bc6bb6ad74de97771aed26ebd33106b078fc5d1`): dogs-scale (83 frames), upside-down (65), dog-agility (25), cat-girl (89), planes-water (38), dog-gooses (86). These are **386 complete native frames**, not chosen occlusion events. All paired RGB/mask members total 55,846,006 compressed bytes; the selected access manifest includes only386RGB+6initial masks+48query masks, totaling54,390,568 compressed bytes. If subsequently admitted, all six original-video parents become permanent development.

For each video, choose the smallest positive non-void instance ID present in the first annotation. Supply its complete binary first-frame mask; other IDs and background are negative, void is unknown. No size, texture, matchability, occlusion, or prediction-based object substitution. An invalid/missing initial mask is a recorded cohort failure, not permission to replace a video. This is a one-object-per-video diagnostic, not the full DAVIS benchmark.

For a video with `N` frames indexed from zero, query

`t_j = 4 + floor((2*j*(N-5)+7)/14), j=0,...,7`.

This is exact integer round-half-up over `[4,N−1]`: eight unique times per selected video, **48 queries**. Run the baseline sequentially on every native frame, including intervening frames. Never select a query using GT, predicted IoU, witness availability, or apparent failure. DAVIS val, VOST, and all other parents stay untouched. No metadata established that these six contain informative occlusions; a negative result is about this fixed diagnostic.

## First inexpensive stop: external evidence before SAM2

Compute SuperPoint/LightGlue only for the six initial images and 48 query images, with existing pinned weights, the existing deterministic CPU settings, maximum 2,048 keypoints, and longest-side 512 preprocessing. Use the extractor's documented restoration to original-image coordinates. Cache each image extraction once; match initial-to-query directly, without SAM masks, descriptor updates, future frames, or chained pseudo-labels. Use native accepted matches without an additional tuned confidence threshold.

Initial positive keypoints must lie in a 3×3-square (radius 1) erosion of the selected target; negatives must lie outside its 3×3-square (radius 1) dilation, excluding void. Distances are in original 480p-image pixels; label lookup uses nearest pixel, round-half-up. This is a boundary exclusion rule, not a visibility certificate. Label source and target coordinates are retained.

Resolve opposite-label collisions at the same rounded current pixel by discarding the entire collision; for same-label collisions retain highest match confidence, then original match index. Per class, stratify surviving current points in an 8×8 image grid, order each cell by confidence then original index, and take round-robin row-major cells until at most 32 points are retained. The same retained points are used by **every** selector/prompt control.

A query is actionable only with at least four positive and four negative retained points. Otherwise every evidence-based rule returns baseline; the query stays in the 48-query denominator. Four points per class is a fixed sparse-evidence redundancy floor, not a guarantee of correct or spatially representative labels; the grid selection spreads available evidence but imposes no additional occupied-cell quota. Freeze the evidence manifest before any SAM candidate comparison. **Stop before the SAM2 sequence run only if fewer than four of six videos have at least one actionable query.** With no actionable query a video's signed method is exactly baseline, so the later requirement of positive gain in four videos versus controls that include baseline is impossible. Do not require half the queries, four queries per video, or a minimum action frequency: rare catastrophic repairs remain testable. No subsequent GT decoding is needed for this early stop.

## One baseline history, small fixed fork bank

Use the pinned SAM2.1 tiny checkpoint and official tiny configuration in CPU float32, four threads, native image size 1024, no compilation. Keep the initial conditioning memory immutable. CUDA-only cleanup must either have a qualified exact CPU route or be declared disabled identically for every arm before execution. No silent provider fallback.

At each fixed query, reuse its image features and the *same pre-query baseline state* for:

1. baseline `B`;
2. `D1`: remove only the non-conditioning frame `t−1`;
3. `D3`: remove only the non-conditioning frame `t−3`.

Filter the one keyed entry in a shallow copied output dictionary. This removes that frame from both spatial-memory and object-pointer assembly; all other tensors, frame indices, `num_frames`, temporal positions and configuration remain identical. No backfill, re-encoding, re-ranking, or deletion of the initial conditioning frame. If a fixed key is absent or not actually consumed, retain the resulting no-op and record its selected-token trace; do not replace it with a more influential memory.

Call the original `_run_single_frame_inference(..., run_mem_encoder=False)` for forks. Discard all fork states, masks and pointers after recording; only `B` advances the video. Direct prompt and reset controls below are also discarded. Thus the estimand is the direct current-state intervention, not erasing all historical effects of a memory.

Read-only source check supports this narrow route: official SAM2 pin `2b90b9f5ceec907a1c18123530e92e794ad901a4`, [`SAM2Base._prepare_memory_conditioned_features`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L497) uses fixed stride slots and separate pointer time differences; with only one initial conditioning frame there is no alternate conditioning-frame backfill. [`_forward_sam_heads`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L257) already returns mask alternatives and predicted IoUs, whereas `track_step` discards that part of the tuple. Capture these already-computed outputs read-only. Artificial qualification must establish baseline parity, unchanged state/tensors, exact consumed-frame traces, and prompt coordinate conversion before scientific input.

## Selectors and same-evidence controls

Interpolate each recorded high-resolution logit at the fixed witness coordinates. Use the same `align_corners=False` pixel-center mapping for all masks; do not score thresholded masks or resize one arm differently. For positive and negative sets `P,N`,

`L(l) = 0.5*mean_P softplus(-l) + 0.5*mean_N softplus(l)`.

The **signed-deletion rule** selects the lower-loss of `D1,D3` only when its improvement over `B` exceeds `1e−6` nats; otherwise `B`. Ties use order `B,D1,D3`. This numerical tolerance is fixed before data, not a learned confidence gate.

From the identical `B,D1,D3` bank, record:

- maximum predicted IoU (native corresponding score; fixed tie order);
- maximum unsigned mask change `1−IoU(mask,B)`, explicitly **not** claimed to reproduce ExRE;
- always `D1`, always `D3`;
- deterministic random deletion using SHA256 of the fixed string `evr0029:random-deletion:2026-10-20`, sequence name, object ID and frame index with NUL separators; final digest bit chooses `D1/D3`;
- the same random choice only on the signed rule's action occasions. This separates ranking from a favorable no-op policy.

These matched comparisons all return `B` at non-actionable queries; also report the two ungated fixed-age branches descriptively. No GT chooses a deployed branch.

Mandatory same-evidence alternatives, all at the same 48 times:

1. **Native mask-bank ranking:** rank the native baseline SAM alternatives plus `B` with the identical signed loss and no-op margin. Capture alternatives already produced by the pinned configuration; qualification must establish their actual number. If there is only one, predeclare and charge a separate official three-mask decoder call before freezing—do not label a singleton a strong diversity control.
2. **Direct corrective prompt:** feed all the same selected points and labels using the official video-predictor coordinate contract, full baseline memory, and the baseline's current low-resolution mask logits clamped to `[-32,32]`, as in [`add_new_points_or_box`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L161). No current GT prompt.
3. **Reset + prompt:** retain only the original conditioning memory; drop all non-conditioning spatial memories/pointers, supply the identical points, no prior current-mask logits. Keep original time coordinates. This is a declared recent-history reset, not a fresh GT initialization.

For each prompt control report both its direct output and its own signed-loss choice between that output and `B`; this prevents weakening the comparator by forcing a bad correction. Prompt count is the full same ≤64-point evidence set, not a smaller convenience subset. Non-actionable queries remain baseline without a prompt call.

All branches use the same current image encoding. Maximum planned work is 54 SuperPoint extractions, 48 LightGlue matches, 386 SAM image encodings/baseline frame passes, 96 deletion passes and 96 prompt passes, plus48 full-history no-op verification passes. The native qualification measured three original same-history alternatives; capture those outputs from the original baseline without an extra decoder call. Charge verification separately and retain total outer cost; the previous conditional allowance of up to48 additional mask-bank calls is unused. No DAM4SAM full-video loop is hidden in this budget. A qualified DAM4SAM comparator and exact ExRE comparison remain required before a later memory-specific efficacy claim; this necessary screen does not replace them.

## Seal first; evaluate 48 outcomes, not successful subsets

Lock initial-mask choice, witnesses, all candidate/control masks and scores, chosen indices, call counts, runtime, configuration and input/source hashes before decoding any noninitial annotation. Then decode only the 48 fixed query masks. Initial supervision is the only pre-prediction GT exception. No GT-based correspondence filtering, candidate filtering, threshold changes, reranking, or mask repair.

Primary screen metric is foreground **J/IoU**, threshold logit zero, ignore official void pixels; empty/empty gives one, exactly one empty gives zero. Average eight queries within each video, then six videos equally. This is a custom current-frame development metric, not official full-sequence DAVIS J&F and not completion of the old pose AUC gate. Report every case and six video means. Technical incomplete prediction coverage is ineligible; ordinary no-witness/no-action outcomes remain baseline, not dropped.

Before seeing labels, fix these stopping rules:

1. **Candidate information:** admissible deletion oracle gain must be at least **0.02 J** on all 48 queries. Its bank is exactly `B,D1,D3`, with oracle equal to `B` at non-actionable queries. Also report unrestricted deletion-bank headroom separately. Neither oracle may deploy a mask.
2. **Useful signed selection:** signed deletion must exceed the best *fixed named control* above, including no-op, prompt variants and native-bank ranking, by at least **0.02 J pooled**; its gain versus the best named control in each video must be positive in at least **four of six** videos, with no video's gain below **−0.02 J**.

Interpret the failures separately. Insufficient witness-video coverage or admissible oracle headroom means **this assay cannot support the proposed gain**; it does not establish that signed ranking fails when useful candidates and evidence exist. Adequate oracle headroom followed by failure of the signed-control comparison rejects the **fixed tested selector** on this cohort. Either outcome closes this fixed development variant without authorizing cohort expansion or same-cohort threshold, memory-age, witness or object retuning. Neither is a universal method-impossibility claim. A pass only admits a separately frozen longer-horizon test, not a paper or final validation. Both the early video-coverage test and oracle bound are necessary relative to the fixed continuation criteria, not statements about all possible repairs.

Descriptive checks must include transferred positive/negative precision after GT; loss-gain versus J-gain sign agreement; deletion actions that reduce J by >0.05, their magnitude and worst case; ordinary native-mask headroom; both witness-class counts; all no-ops; per-video and pooled results. None changes the primary population. Sparse-label correctness does not imply full-frame correctness; correlated query frames are not 48 independent samples, so no frame-level significance or bootstrap claim.

## Resource and interpretation limits

Each admitted operation would have a hard four-CPU-thread, 8 GiB, 1,800-second bound, no paid/cloud GPU, no retries. A separate operation may process each **complete video**, including its fixed eight queries and all controls; all six complete operations are required before aggregate GT evaluation. This does not require all 386 frames to fit one 30-minute operation. Verify manufactured CPU/runtime feasibility before allocation; a runtime failure never permits shortening a sequence, omitting a control, or evaluating only completed videos. Stream images and retain only necessary baseline memories, pointer history and query outputs. The native runtime/checkpoint acquisition is already complete under the separately retained 300,000,000-byte reservation. The fixed DAVIS phases below have their own prospective reservation.

This design can fail cheaply at witness availability, then decisively at fixed-bank headroom or selection. If prompting performs equally well, the evidence supports transferred supervision rather than a memory-specific mechanism. If native mask diversity performs equally well, deletion may be an expensive candidate generator. Even a pass cannot establish closed-loop stability, dense guarantees, independence of matcher errors, novelty relative to prior recollection methods, or generalization. The 12 DAVIS validation parents remain untouched, and VOST original-recording grouping remains unresolved; two independent final-source validation contracts and their task-specific gate still require a later prospective decision.


## Explicit numerical conventions


- “3-pixel square” means a **3×3 kernel, radius 1**, including the center. This
  supersedes a separate author's original radius-3 interpretation. Outside-image
  target pixels are false. Positives lie in target erosion; negatives are outside
  target dilation and are not void.
- Initial labels and collision pixels use `floor(x+0.5), floor(y+0.5)`. Continuous
  coordinates must first satisfy finite `0<=x<W, 0<=y<H`; a rounded pixel outside
  the image is discarded, not clipped.
- Grid cells use the **continuous** current coordinates, `floor(8*x/W)` and
  `floor(8*y/H)`, with the cell indices bounded to 0…7 after coordinate validity.
  Opposite-label pixel collisions discard all involved matches. Same-label
  collisions retain highest confidence, then the original native match-row
  index. Each class is sampled round-robin through row-major cells, each cell
  ordered by descending confidence then match-row index, up to 32 per class.
- Sampling uses `(x+0.5)*W_logit/W_original-0.5`, independently for y, and bilinear
  edge replication. It samples logits first, then applies stable softplus.
  It does not interpolate probabilities or thresholded masks.
- Random deletion hashes UTF-8 NUL-separated salt, sequence, decimal object ID,
  decimal frame index. Final digest bit 0 means D1; bit 1 means D3.
- The signed margin is strictly greater than `1e-6` nats. Other exact ties follow
  declared order. Native alternatives use their supplied original order after B.
  All evidence-dependent rules return B when either class has fewer than four
  retained points. Descriptive ungated D1/D3 remain separate.
- Technical query failure keeps its predetermined row and makes the full
  48-query evaluation ineligible. It is not assigned a fabricated J and does not
  produce successful-subset means. Ordinary no-witness rows are valid B outcomes.


## Staged execution admission

Only the first evidence stage may run under its first complete source freeze: acquire the fixed386 RGB files and6 initial masks into separate roots, then compute all54 SP extractions/48 LG matches and freeze their evidence coverage. Require the completed nine-frame native qualification, immutable installation inventory and clean outer receipts before ML imports. Every operation uses the existing four-thread/8GiB/1800s supervisor. Acquisition roots are disjoint; the first-stage wrapper verifies all392files before and after use even though matching decodes only54RGBs. No query GT phase is available: acquire_davis.py rejects evaluator_gt before HTTP.

The complete scientific SAM predictor, all same-evidence controls and all-six prediction-seal verifier must undergo a later prospective source freeze and exact CI before the SAM stage or evaluator-GT acquisition. This staged preparation does not authorize a missing/partial comparator, label access, or a successful-subset result. The fixed cohort, query schedule, evidence rule and numerical gates above remain unchanged across these implementation stages.

The standalone native CPU qualification uses9 artificial frames, not these videos. If its measured cost does not fit a valid complete-video operation, do not shorten a sequence or omit controls. A new resource/platform qualification may be proposed before any scientific predictions, while preserving this fixed cohort.

## Prospective pointer-position arithmetic clarification

Before any scientific image/mask access, the native runtime audit established that identical pointer temporal coordinates can produce different last bits under the authored float32 linear projection when the pointer batch size changes. An isolated calculation exactly reproduced both saved full-history/reset hashes, with maximum difference9.5367431640625e-07. This is not a tolerance adjustment or scientific outcome.

For every deletion/control trace, require exact consumed frame IDs, unchanged retained temporal distances and normalization denominator, and exact equality of actual attention positions to the original native formula evaluated at that branch's actual batch size. Retained spatial feature/position blocks and pointer feature blocks remain bitwise checked. Record cross-batch pointer-position hash differences descriptively; do not demand a batch-independent floating result that the original implementation does not guarantee. Original9-frame qualification and its results stay unchanged. This clarification also covers D1/D3 batch-size changes beyond the one7-to6 case already qualified.


## Acquisition allocation before media access

Reserve **60,000,000 bytes** for the exact [440-member inventory](../configs/0029-davis-development.json), retaining every earlier hold and giving a conservative planning balance of **1,289,868,314 bytes** from the previous1,349,868,314. These are planning bounds; provider-mediated web bytes are not fully metered. No refund from the larger runtime or metadata holds is taken. The selected compressed bodies total54,390,568bytes; local-header/maximum-extra overhead bounds the selected response bodies by56,225,868bytes. Phase caps are57,000,000RGB,100,000initial-supervision and500,000query-GT bytes, all within the reservation. No retries, whole-archive fallback, model downloads or extra media are authorized.

The first stage allows only386RGB+6initial masks (392files). The48query masks remain unavailable until a separately reviewed all-six prediction-seal verifier is committed and passes exact CI. Source metadata, ZIP extent, ETag, central-directory records, member CRC/size and output SHA256 are checked. DAVIS and external-source license restrictions continue to apply; no scientific media are published in this repository.
