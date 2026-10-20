# 0029 second-stage integration review

**Source-only verdict: no blocking interface or scientific-barrier discrepancy found.** This review neither runs qualification nor admits scientific execution. Exact-source CI, successful manufactured qualification with its saved-output review, and clean supervised operations remain required.

Reviewed the eight public production files and the additional second-stage protocol. The existing first-stage acquisition, evidence wrapper/frontend, arithmetic core, manifest and protocol have no diff from `c09e240009869402e2b97a55af42617eea3c6405`. No media, NPZ, model or query-GT values were loaded; no tests were repeated and no public source was edited.

The interface chain is consistent:

1. `run_native_video_qualification` and `run_video.qualification_contract` use identical qualification source sets, command arguments, nested record names and manufactured artifact paths. The latter checks the clean outer receipt, 25-frame/63-call manufactured record, saved worker inventory and pure technical verifier. Manufactured identity is explicitly not DAVIS evidence.
2. `run_video` derives the immutable first-stage inputs from the old evidence command and uses a separate new source commit. The worker input schema matches both callers. All six videos retain their fixed eight queries. The admitted 38 actionable rows imply 76 prompt/reset calls and 606 tracking calls; qualification is separately charged.
3. The video operation writes its completed receipt before `seal.json`; the seal binds that exact receipt and all prediction files. `seal_queries` requires each video outer specification's `result_file` to be **`seal.json`**, checks its output hash, then follows the seal's receipt hash and opaque inventory. It also checks all 13 saved scalar selections without GT. The six-video proof, common evidence/qualification chain, 386 baseline frames and all 48 queries must pass before admission is returned.
4. `acquire_query_gt` runs the complete verifier in the original acquisition function's first callback, before `RangeReader` or any HTTP request. Later callbacks and postchecks rehash source, proof, receipts, predictions, runtime inventory, metadata and input directory membership. Only the fixed 48 members and unchanged 500,000-byte phase cap are admitted. It does not decode annotations; the old acquisition default remains closed.
5. `run_evaluation` repeats whole-cohort admission and binds the GT acquisition to the same deterministic admission. It loads and verifies **all 48** prediction populations, NPZ/raw-array hashes, shapes, dtypes, finite values and allowed selections before decoding the first query mask. The pure evaluator consumes sealed choices, retains no-action rows, and implements the fixed deletion oracle and named-control gates. Post-evaluation checks are hash/inventory operations, so they remain compatible with the installed offline guard.
6. Complete per-video wall/RSS values are read from the bound raw outer receipts, not from the hash-only binding object. Query timings are separately labeled partial/overlapping descriptions; frontend, acquisition and evaluation charges are not silently added twice or represented as free.

Pre-GT finite-array assurance in the standard-library sealer is correctly described as inherited from the frozen worker plus unchanged bytes. Independent numeric NPZ validation occurs in the evaluator before annotation decoding. The proposed later saved-output audit remains separate from this source review.

| Source | SHA256 |
|---|---|
| `native_video.py` | `4af3e0e5b26327d4ecb95b4644b94e13c257a38c928dbb09f23bd95151001d0a` |
| `qualify_native_video.py` | `dc40d641d353d13af7bf0198c853c4b8c548ec5d10ec7cf09ba683292cf6de59` |
| `run_native_video_qualification.py` | `abd5a0912aae8c8f759c3128cef6cb81ab4357053f426373be9aac1c42c223d8` |
| `run_video.py` | `2521aed76de6a769df27d449498907589fea862f8f3fb378940c007bc9b0b8da` |
| `seal_queries.py` | `ce570071dbbf6446500b66dbf6aaffb752e1ec10bb848b9a257d61214a83584f` |
| `acquire_query_gt.py` | `67ced391f8f911bb93554d35d8ce53d0df028758037b4001e61121dd60e0edc4` |
| `evaluate_queries.py` | `272ae306b0f80573d3350d892cefbc13cf9df5a06d89b6188ee9c86584c5237f` |
| `run_evaluation.py` | `2ed77267f20e2e14fa164503d2926a37c12e0f898b07926134dc7a855f9d7ef1` |
| `0029-native-video-comparison.md` | `4ab0bbb656d3177b9a40bbc7f22b78c3fe3d10249096482f90cb671a2f1d1fbe` |
