# Independent review of 0002 development artifacts

Reviewed repository head: `10936637e4605b3875f52e8aca75e30c49987326`.

This AI-assisted artifact review independently recalculated results from published Git objects using Python's standard library. It did not execute candidate experiment code, rerun the model, download the original archive or weights, or modify repository files. The primary, recording-complete artifacts are in `results/0002-preflight-recording/`; `results/0002-preflight/` preserves the original runs.

## Identity inventory and allocation

- Checked all 50 archive indices and their 150 declared array identities, including recorded shapes and dtypes.
- Verified the complete inventory of 12,500 indexed RGB-frame hashes and its manifest digest. Independently reconstructed connected components from shared exact frame, prefix and full-video hashes: **50 singleton groups, with no cross-video exact-frame matches**.
- Independently reproduced the prespecified allocation hashes and partition assignments: **development 1, calibration 10, evaluation 20, reserve 19**. Video 0 remains development; the first calibration group in allocation order is `g-033`.
- The original and recording-complete identity manifests and frame inventories are byte-identical. The metadata receipt retains its original execution provenance; it is not a second metadata execution.

## Raw outputs, timing and replay

Checked all six CSVs across the original and recording-complete runs: two 64-frame/two-query batches, two corresponding fresh-process repeats, and two 16-frame/single-query checks. Each two-query CSV contains all 128 distinct frame/query rows and 512 finite output scalars; each single-query CSV contains all 16 rows and 64 finite scalars.

CSV digests, complete grids, per-frame latency agreement across queries, first-frame times, subsequent-frame medians and summed frame times match the corresponding receipts. Summed frame times do not exceed recorded subrun wall time.

| Recording-complete subrun | Wall time (seconds) | First frame (seconds) | Subsequent median (seconds) |
|---|---:|---:|---:|
| Two queries, 64 frames | 39.337324552 | 0.724249448 | 0.585713988 |
| Fresh-process two-query repeat | 39.450014091 | 0.650206525 | 0.586655604 |
| One query, 16 frames | 10.349778104 | 0.530384722 | 0.673953370 |

- Recomputed maximum absolute difference for each 64-frame batch/repeat pair: **0**.
- Recomputed single-query versus two-query common-prefix maximum absolute difference: **0.00002765655517578125**, within the fixed `atol=0.0001`, `rtol=0` criterion.
- All 32 saved point/frame rows of the earlier 16-frame two-query batch exactly match the corresponding 64-frame prefix. Original and recording-complete 64-frame batches also match numerically exactly. Timing differences are preserved rather than treated as numerical-output differences.
- Verified all **30 receipt-listed source-file hashes** against the exact bytes at their recorded source commits (`db513434` or `0773f28`). Both commits are ancestors of the reviewed head. Manifest and raw-output hashes also match their receipt bindings.

## Evidence boundary and freeze status

No new inconsistency was found in these published development artifacts. The graph reconstruction is independent computation over the **published hash inventory**: the original archive was not downloaded and its 150 arrays were not independently rehashed in this review. The cloud-side array rehash verification remains a separately reported, same-host check. Replay agreement is same-data reproducibility, not independent-data validation.

Exact-duplicate exclusion does not establish scene independence or exclude near-duplicates. Shared triplet-4 assets and the front-left camera, unknown scene relationships, and unresolved upstream training/model-selection exposure limit generalization claims. No held-out inference or efficacy conclusion is asserted by this review.

The initial review found a P1 in freeze verification: caller-supplied current-file hashes were not bound to the claimed input commit. The subsequent v2 fix and independent verification are recorded below. No model rerun was needed for this provenance repair.

## Addendum: v2 candidate Git and receipt audit

Independently checked the actual v2 receipt at `fbb6dd9208f6ea95e588f410c05c9ac5b58dcb1c`, binding input commit `bbe7428e126a566a62d3c123d7158f120aa66e2b`. This addendum used only standard-library calculations and Git reads; it did not execute the verifier, tests or model.

- All **29 mandatory input SHA256 values** match regular-file blobs at the specified input commit and reviewed HEAD, the index, and current file bytes. The map exactly matches the fixed declarations in the input source, inspected without executing that source.
- All **six prerequisite receipt bindings** agree with the input map. All **16 exact dependency versions** agree with the fixed source declarations, dependency metadata receipt and requirement files. This checks recorded dependency identity, not an independent installation of the cloud runtime.
- All **six source digests in the v2 gate-validation receipt** match the specified input Git blobs and current files. Its recorded base is the input commit's parent. The two recording-complete development receipts retain the recorded model-execution commit and upstream-module/checkpoint identities bound by v2. The earlier **30 historical source-file digests** were rechecked against their own execution commits; historical source versions were not relabeled as newly executed code.
- The receipt commit's sole parent is the input commit, and its sole change is adding `protocols/0002-freeze-v2.json`. That receipt is absent from the input tree and input map. This establishes the non-circular, input-first publication order.
- The v1 receipt, protocol and configuration remain byte-identical to their published versions at `53563e2f62cc689ae6f879185403b35bec57af76`. All earlier result files are unchanged; the only new result file in this revision is the provenance-gate validation record.
- Parsed scientific configuration values are unchanged; differences are the version number and explicit v2 protocol/receipt path bindings, plus formatting. The protocol body from **Question and scope** onward is byte-identical between v1 and v2. Estimator, identity-grouping and metric implementations are unchanged; this revision updates the provenance gate and its entry-point bindings.

**Candidate-chain verdict: no inconsistency found in these Git and hash checks.** At this review, local `origin/main` was still `2702d96680d86a432c3e42b16132b0663d561ccb`, so v2 was not yet eligible for execution under its main-publication gate. Merge/publication verification and the independent positive/negative gate tests are separate checks for the parent reviewer to record. This addendum does not claim held-out inference, remote attestation or a model rerun.

## Independent gate tests and final review

A separate standard-library harness exercised the unchanged v2 verification
function bodies in temporary Git repositories. All 73 checks behaved as
expected: two legitimate publication chains were accepted and 71 invalid cases
were rejected. The original old-commit/new-hash counterexample is now rejected.
Checks also covered missing mandatory entries, empty maps, edited/staged/
untracked/unpublished receipts, alternate bound paths, circular freezes and
modified current inputs. Different JSON formatting is accepted when its exact
bytes are committed and published.

Installed-package version queries were **mocked to the frozen versions** to
isolate Git/content verification. This is not a local CPU dependency installation
check or a second model run. See the [case record](../results/0002-preflight/parent-gate-review.json).
The actual repository correctly rejected execution before main contained v2.

The v2 candidate has no remaining identified review blocker. The independent
scientific review also found the protocol and metric implementation consistent;
this review does not establish method efficacy. GitHub Research checks on
`fbb6dd9208f6ea95e588f410c05c9ac5b58dcb1c` passed [run 38966088707](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38966088707):
184 tests in 8.07 seconds and the existing 1,240-frame toy smoke in 4.04 seconds.
The cloud repair session separately recorded 184 tests in 7.08 seconds and smoke
in 3.10 seconds. Publication will preserve the input and receipt commits via a
merge commit. Future inference must fetch and verify the main-published freeze.
