# 0031: preserved reference-pair information, TRAIN-only

Source preparation, 2026-10-20. No execution, scientific array reads, training,
new assets or model calls are authorized by this document. The preceding 0030
channel-affine family is closed; this changes the question to local output
transition evidence, without modifying memory or masks.

## Fixed observation and geometry

Use the existing four TRAIN parents, in order `dog-agility`, `planes-water`,
`dogs-scale`, `dog-gooses`. For N frames, anchors are
`1 + floor(i*(N-5)/11), i=0..11`, with horizons 1 and 3: 96 events, including
the four final-frame events. Overlapping future frames remain repeated events,
not independent observations. No DEV or final-test label is decoded.

Use only `vision[0]` from the already sealed native frame cache: the finest
image-only feature map, 256×256×32. It is not the memory-conditioned embedding.
The native 1024 square image transform implies the original-to-feature
pixel-center coordinate `(x+.5)*256/W-.5`, similarly for y. Bilinear
interpolation uses border replication within this feature grid, followed by
float64 unit normalization. Exact zero descriptors are unavailable.
Nonfinite cached features are a technical failure.

Let `r=ceil(.008*hypot(H,W))`, the existing DAVIS tolerance radius in original
pixels. Enumerate every horizontal and vertical adjacent-pixel target-mask
transition; its center is halfway between pixel centers and its axis points
from target to non-target. Initial-mask reference endpoints are center±r.
They must be in-image and nearest-pixel labels must actually be target and
non-target, with neither endpoint 255. The reference cap is 128.

Current query centers come from **native predicted masks only**. The cap is
64 per event. Both caps use 8×8 original-image cells, row-major round robin;
inside each cell preserve horizontal-before-vertical, then y,x order.
Initial geometric eligibility precedes its cap; zero descriptors never cause
refilling. No current-GT or feature-quality criterion selects query centers.

At every current query, consider offsets `[0,-r,+r,-2r,+2r]` along its outward
axis, always with the same endpoint spacing 2r. Reject out-of-image endpoints;
never clip them. Label lookup is `floor(xy+.5)`. Half-pixel endpoint ties
therefore have the same declared rounding in every arm.

## Four fixed rankings and failure handling

For normalized endpoint descriptors x,y and initial pairs u,v:

- **paired:** `max_k(x·u_k+y·v_k) - max_k(x·v_k+y·u_k)`.
- **unpaired_max:** independently maximize each of the four side similarities
  before taking the forward-minus-reverse difference. Mean prototypes alone
  would be a weaker control.
- **deranged:** identical paired formula after permuting only outside
  descriptors. Sort retained reference indices by SHA256 of
  `evr0031-pairing-v1\0sequence\0index`, then rotate that order by one. This is
  a deterministic single-cycle derangement, preserving both marginals and K.
- **native:** offset zero.

Maximize score; exact ties use the declared offset order. The two maxima in
the paired formula may choose different reference indices. Both winners and
compatibility gaps are saved. This is ordinary nearest-neighbor matching in
concatenated descriptor space, not a new matching primitive.

Fewer than two usable reference pairs, or invalid zero-offset descriptors,
forces all three scored arms to native with an explicit reason. Other invalid
candidates are retained but cannot win. Empty predicted boundaries remain
events with zero queries; absolute query means are null and intervention
deltas are zero. Technical corruption aborts with a partial receipt, rather
than silently dropping an event.

## Evaluation after the complete ranking barrier

Verify source/metadata and every selected encoded cache, prediction and label
hash before decoding. Initial labels are permitted reference inputs. Save all
96 rankings and their hash manifest **before** decoding any corresponding
current TRAIN label.

For every candidate, retain endpoint state: correct target→non-target,
reversed, both target, both background, or out-of-image. Current void 255 is
background as in the existing metric path, with counts exposed. No GT-based
straddling filter is permitted.

Find the nearest GT target→non-target crossing on the **same scanline and
orientation**. Report signed displacement in outward-axis units and absolute
distance capped at 2r, normalized to [0,1]. If there is no such crossing,
signed distance is null and the capped loss is 1; that query is not excluded.
This is a scanline localization diagnostic, not DAVIS F or Euclidean
nearest-boundary distance. Also report joint loss: this localization loss when
endpoint orientation is correct, otherwise 1. A wide pair that straddles a
boundary does not erase center localization error. Save signed and absolute
chosen displacement from the native center, and noncrossing/reversal rates.
All five candidate metrics are diagnostic; GT never changes selection.

Primary presentation is mean intervention-minus-native difference across
queries within each event, then all 24 events in each parent, then four equal
parents. Empty events contribute zero difference. Separately report pooled
crossing means and raw counts, which weight long boundaries more. Negative
loss deltas are improvements; positive correct-orientation deltas are
improvements. The pairing question requires comparing paired to **both**
unpaired and deranged, not merely to native. No automatic research-admission
gate or confidence interval is claimed from four exposed TRAIN parents.

## Run contract

One serial operation under the existing exit-checked supervisor: at most
1800 s, four CPU threads and 8 GiB descendants RSS. Command:

`QUALIFIED_CP312 -B -S -W error screen.py SPEC.json`

The JSON supplies fresh `output`, `qualified_python`, `evaluation_root`,
`guard_source`, exact `contract`, `source_sha256`, and six descriptors under
`bindings`: `cache_receipt`, `cache_outer`, `cache_operation`,
`cache_worker_spec`, `mapping`, `data_index`. Descriptors have path, bytes,
SHA256. Use the existing qualified runtime with NumPy/Torch/PIL/cv2; Torch
only deserializes CPU tensors with `weights_only=True`. No SAM2 package,
checkpoint or model is loaded. Existing offline guard is installed before
site packages; all five thread environment keys must be 4. Source hashes bind
this script, this scope, the guard and the three unchanged evaluation helpers.
The cache's original successful operation/log binding is rechecked by the
existing `evaluate_rollout.admit(..., legacy=True)` helper.

Per-frame temporary feature arrays are released; no complete-video feature
stack is retained. Work is bounded by 128 references × 64 queries × five
candidates × 96 events, plus cached-file reads/hashes. Actual runtime and peak
RSS must be measured, not assumed. Outputs preserve reference geometry,
permutation, every candidate score, selection and failure reason; ranking
seal; all current-label diagnostics; aggregate analysis and durable receipt.
No repaired full mask, decoder, optimizer, model training, gains claim or
novelty claim is produced. The parent-owned `ADMISSION.md` fixes the prospective
continuation conditions and is also a required source-hash binding. The script
does not automatically change research admission; root applies that unchanged
document to the saved complete summary.
