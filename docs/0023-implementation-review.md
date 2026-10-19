# 0023 independent implementation review

This review covers source, manufactured arrays/fake interfaces, and saved 0022
manufactured qualification outputs. The reviewer acquired no Map-free image,
calibration or pose payload, ran no model or native estimator, and changed no
scientific outcome. No source-level blocker remains in the reviewed version to
the bounded prospective study **after its exact implementation/protocol freeze
and CI**. This is not evidence of accuracy, novelty, held-out generalization or
paper readiness.

## Geometry and shared evidence

The fixed track union preserves reference components when adding one query;
every deletion rebuilds from that deletion's raw reference graph. The same
retained query support reaches every arm and native E. The native control is
therefore displayed as `native_E_reference_filtered_query`, not a strictly
query-only estimate. Geometry deduplicates track observations while the field
objective retains directed edge blocks; the protocol discloses that weighting
difference.

The geometry information check uses final-state IRLS derivatives, zero
information for capped/invalid blocks, independent three-column point
elimination, and a narrow pose SVD. No dense all-points matrix is constructed.
Its rank floor is based on the original pose block. Final rotation and
translation-direction null ratios are normalized separately; translation-length
nulls can remain harmless. Recentring the information chart avoids attributing
a singular optimizer chart to geometry. Manufactured tests cover a generic
observable scene, collinear support, capped query observations, a length-only
null, and an extreme baseline exposing dilution in the earlier combined ratio.
These remain numerical local checks with fixed units and thresholds.

Insufficient valid anchor depth now gives geometry a common positive unit-
baseline rescaling of all translations and points. Reprojections are unchanged;
field fitting remains unavailable when its depth-unit alignment cannot be
established. The valid anchor-depth LS path is unchanged. Malformed values
declared valid remain technical failures. This prevents depth availability
alone from disqualifying the geometry control.

## Execution and truth barrier

The review found and closed three concrete integration/scope defects:

- Frontend postchecking initially attempted Git subprocesses after installing
  the permanent offline guard. The first check still verifies Git bytes; the
  postcheck now rehashes the already admitted source map and receipt/log chain
  without any subprocess exemption.
- Prediction now rechecks its admitted frontend supervision receipt and both
  logs before sealing, alongside caches, scientific sources and native bytes.
- Evaluation originally parsed every pose row and trusted only frontend receipt
  hashes. It now converts numeric GT only for the nine selected anchor/query
  names per place, and independently reuses the verified predictor's complete
  frontend/acquisition/qualification/native admission before any GT request.
  Other pose rows are inspected only for filenames and duplicates. The 27
  numeric rows are a subset of the protocol's admitted 51 image identities.

Evaluation requires all 24 ordered queries, all seven terminal outcomes, exact
output/seal/source hashes and a successful outer prediction operation. Failure
or tampering blocks GT acquisition. Completed frontend/model/input chains and
the native binary are revalidated; the same barrier is checked again after
scoring. The separate GT phase admits only three exact pose members, nine
bounded requests and the declared 2MiB body budget. The frontend's GT prohibition
is not relaxed.

The pose convention uses the pinned [official Map-free README](https://github.com/nianticlabs/map-free-reloc/blob/c3290ec5fe992f816b6cf55686af7200ddd204dc/README.md):
`frame_path qw qx qy qz tx ty tz`, world-to-camera transforms. Accordingly
`R_rel=R_query R_anchor^T` and `t_rel=t_query-R_rel t_anchor`. Translation angle
is directed; sign folding appears only in a labelled secondary metric. Failed
estimates receive 180 degrees and every method retains denominator24/eight per
scene. Exact native fallback identity is checked. The sorted-error trapezoidal
AUC uses the previously frozen DMS left-search convention, including threshold
ties. The gate compares the candidate with all six controls: at least +2pp
overall AUC5, positive against the best control in at least two scenes, no scene
below −1pp, and no more than .5pp AUC20 loss against the best overall control.

## Observed checks and qualifications

The reviewer ran eight fake prediction tests successfully (tool `e0ddcf`,
unittest0.075s) and thirteen evaluator tests with warnings treated as errors
(tool `43ff72`, unittest0.149s). These exercise complete seals, failure and
interruption preservation, source/native/log changes, the pre-GT barrier,
directed angles, AUC ties, complete denominators and fake ZIP transport. Native,
model and scientific-array calls were not executed. Geometry's thirteen-test
review run passed before the later narrow depth-scale change; that change's
author then passed its two relevant artificial tests and its source was
independently reviewed. Initial test-fixture failures were corrected before
freeze, without changing a scientific fixture or threshold.

The final combined [artificial-suite receipt](../results/0023-development/artificial-checks-01.json)
records all126 tests passing under `-W error`, exit0, wall4.3184525s, with
unchanged before/after source hashes. The reviewer independently checked all29
bound source/configuration/workflow hashes and its log SHA256
`58315e2dce15abe568fb0bae3c42ff5fb8a17a9ebe503673a2622014d734afbd`.
This includes the final geometry scale-availability behavior and six study
driver tests; no duplicate combined run was performed.

An independent saved-only audit of [0022 qualification05](../results/0022-preparation/runtime-attempt-05/README.md)
passed once (tool `c6eb37`, exit0): 16 public/private publication entries, ten
frozen source bindings, six NPZ files and 29 shape/dtype/raw-array hashes all
agreed. The successful outer operation took23.377363625s and sampled peak group
RSS2,129,712KiB. It produced two complete depth outputs,84/135 features and73
matches, retained two CPU-autocast warnings, and recorded no prohibited offline
operation. Earlier failed attempts remain preserved. This establishes a usable
manufactured frontend pipeline, not pose accuracy.

Reviewed source SHA256s (subsequent source changes require focused re-review):

| File | SHA256 |
| --- | --- |
| evaluate.py | 58f7588af87780a07fda7c1555ae08ca1910469b3d78a5df8d63b0c82663f503 |
| test_evaluate.py | 320050465d88b4d1700a6f295c228daec31edfe672908f5e0fc5cd53c0c73f93 |
| predict.py | b0d1dc0f6811f751edde5831d5fe4ab95faee27885ac070669c9585bd2180c83 |
| test_predict.py | 843b70c272b7cfac2cbcc0914998c695ecbd9e22c1dd7e23214b6b8f2d82edd1 |
| geometry_ba.py | e8b6fa7bc789819aac769ccf521af944202a345600606a78077aa2ee5179a013 |
| test_geometry_ba.py | f1d50ceb8cd6ea266953fa68513d1b247e4aa9abd08aae6547a6dd5c6e7da76f |
| study.py | fbbe24e40d17cf1e655e51dc1a22ca726a806eb18c2e0702ee5fff83e9ba2c34 |
| frontend.py | 5076452d2e69645745eb68675a40bb47473ec4a4a28db87cf6f9652229614bf7 |
| acquire_operation.py | fab653e3ed1c0e32cb0a97eac40ef0dbc7ff3df3894da57bf9d4dfac2ba38281 |
| protocols/0023-mapfree-directional-instability.md | c4f5c0cf02e31b990f7aa54bb740cde4dfbaa43c46bca1f260ffbc6bc320673e |

## Technical amendment review, before prediction02

Prediction01 stopped before fitting or GT because the adapter principal-point
recomputation differed from the frozen producer by one ULP. The exact-operation-
order repair retains strict equality; the original successful frontend bytes and
all earlier receipts remain unchanged. The predictor now records an explicit
input-production freeze, and evaluation reconstructs it from the sealed command
while verifying the original complete input chain independently of new predictor
sources. This amendment document is itself source-bound by prediction.

The independent evidence reviewer checked the adapter fix and its one-ULP
rejection case; a separate source reviewer checked predictor/evaluator binding.
No barrier regression was found. Nine public raw receipts and six path-only
redacted logs independently match their originals and publication hashes.

Root artificial-checks03 passed all132 cases in4.554402875s, exit0, with30
source/config/document hashes unchanged. Checks02 is preserved: its runner
omitted the existing SciPy runtime path, producing26 import errors. Correcting
the test command required no source change; scientific execution already adds
that exact qualified runtime. No real fitting or GT evaluation ran in these
checks. New exact-commit CI remains required before prediction02.
