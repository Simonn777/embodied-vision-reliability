# 0003 prospective timing interpretation addendum v1

October 17, 2026. AI-assisted prospective clarification, before any project 0003
candidate cohort GT arrays or predictions have been accessed. This addendum
**supersedes v1's GT-quality timing only**. Every published v1 input and canonical
receipt remains byte-identical and historical. No effect target, score, sample
budget, family split, geometry tolerance or resource/archive limit changes.
This is not post-hoc tuning or an execution/final-cohort freeze.

## Why timing needs clarification

V1's pre-freeze >=95% held-out GT support requirement conflicts with quarantining
geometry-inspected families and deferring GT diagnostics until predictions are
frozen. It could quarantine every candidate. Before a final cohort freeze, use
schema/identity/provenance metadata, quarantined development geometry/convention
verification, and indispensable frame-0 query fields only. These fields can
include frame-0 visibility/GT2D and the frame-0 3D/camera fields indispensable for
the frozen query-validity rule; they do not permit broader held-out trajectory
quality inspection. Record such access as **FRAME0_GT_ACCESSED**, with family,
point/field IDs. Do not call those data GT-unaccessed. Broad geometry/RGB/GT audit
still quarantines its entire scene/story/session/multiview/asset family. The whole
sample here is conservatively audit/development while family relations are unknown.

## Prospective order

1. Check source/schema/identity/family metadata and quarantine all development
   geometry/convention material. No held-out GT beyond indispensable frame 0.
2. Publish a separate final cohort/pipeline freeze with complete family/point/
   window allocation, identities, source/runtime, manifest and hashes. It must
   bind v1 PLUS this addendum and implement their explicit pre/post distinction.
   Allocation includes zero-query/zero-opportunity families, with no outcome-based
   drops/refills. Pre-inference GT quality is **PENDING_POST_PREDICTION**, not passed.
3. Collect all prescribed predictions, preserve every failure/missing outcome,
   and freeze all prediction hashes BEFORE held-out GT support/reprojection
   adjudication. This addendum does not authorize that inference; separate user
   authorization and a reviewed final pipeline/cohort are still required.
4. Only then adjudicate unchanged family GT gates: >=95% valid source-visible
   post-query GT support, and >=99.9% of finite visible projection checks within
   0.5 original raster pixel under the development-verified convention. No fitting
   or convention switches on held-out data. Required-family gate failure makes
   continuation **INCONCLUSIVE**, preserving all allocated families and zero-query
   units. Undefined quality support is not a pass. No drop/refill/refit/resampling.
5. Keep the original primary definition: finite-GT2D, source-visible post-query
   opportunities, with missing predictions counted as failures; report GT-invalid
   and all-opportunity coverage separately. Invalid GT3D/history anchors make
   only the diagnostic undefined. Do not pick new anchors, remove primary rows
   because their diagnostic is invalid, or change denominator definitions.

The 20% relative continuation goal, 20 independent evaluation families / 5 native-
error-bearing families / 20 native matched failures and paired 95% support rule
remain v1 planning choices. Old labels/results/scales/thresholds stay untouched.
V1 `readiness()` is a **legacy design diagnostic with no execution authority**;
its prechecked-GT fields must not silently gate future inference. The separately
frozen future pipeline must implement `GT_QUALITY_PENDING_POST_PREDICTION` before
inference and the post-prediction adjudication/inconclusive rule afterwards.
`motion_phases.py` is a small typed phase-consistency helper, not an experiment.

## Binding

Publish this addendum, its machine-readable timing config and phase helper/tests
in a real input commit first; then publish a separate canonical
`0003-timing-addendum-freeze-v1.json` receipt. Its full mandatory hash map includes
the unchanged canonical v1 design receipt, v1 protocol/config/helper, this addendum,
config and phase helper/tests. The receipt excludes itself; no self SHA/commit
claim. All eight blobs/current/index/HEAD and the designated published branch
must match. The new receipt does not replace v1 or pretend a cohort is ready.
