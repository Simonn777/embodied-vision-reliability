# 0009 frozen synthetic stage0: NO_GO

This exact candidate is stopped. The single authorized collection recorded all 256 cases; 244 tail cases completed and 12 were unsupported. This is a synthetic, oracle-covered translation-ranking development screen, not evidence about real images, a complete geometric estimator, novelty, or a manuscript.

| Family (64 cases each) | raw | msac | cell | excess | ratio | tail |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 64 | 64 | 64 | 64 | 58 | 64 |
| clustered_outlier | 64 | 64 | 64 | 64 | 63 | 64 |
| compact_true | 64 | 64 | 64 | 64 | 51 | 59 |
| sparse_isolated | 63 | 63 | 12 | 63 | 4 | 56 |

The primary clustered-outlier gain over the strongest aggregate cheap control (`raw`, 64/64) is **0**, below the frozen required 8. Primary headroom is insufficient (64 > 56). Solver coverage fails. The paired-harm gate fails: compact_true harms versus raw/msac/cell/excess/ratio are 5/5/5/5/4; sparse_isolated harms are 7/7/0/7/0; uniform harms are all 0. The frozen limit is 2 per comparison. Resource and independent saved-result replay pass, but do not override these failures.

## Saved-case failure attribution

All 12 raw-success → tail-failure cases overlap exactly with the 12 unsupported cases; see [the complete lookup](saved-case-description.json) for each first unsupported H, canonical ID, matrix, reason and component. The unsupported reason is a non-complete component with smaller side greater than 12. Across 8,192 H slots, 7,985 tail calculations completed, 12 were unsupported and 195 remained unattempted after those stops. No failure was retried. All cheap controls have 8,192 recorded scores.

There is also one **completed but incorrect** tail selection, `sparse_isolated-23`: canonical H 13, translation (128,128), saved transfer error 181.0193359837562 pixels. Raw and all other methods also selected H 13 incorrectly. It is neither a solver failure nor a raw-success paired harm. Aggregate counts alone are not used to assign failure causes.

## Execution and evidence

- Execution HEAD `cd144b90d337a172a892b56d87cd794524605753`; source `4b11f1de4d6fcc9bef45c8eee824ff8e7241abc0`; separate freeze `17f417efcccf85047e0935569f87f94b27705e82` (15 inputs). Canonical receipt SHA256 `2485a844f95c47ee420617d0464be7ce8bf5e747b8539b70f2450df2306d6631`.
- Python 3.12.14, cached NumPy 2.0.2, Linux CPU, affinity [0,1,2,3], four numeric threads, actual 8 GiB cgroup and per-process address-space cap. Frozen outer deadline 900 s; case cap 2.5 s.
- 73 targeted artificial tests passed with zero skips; all seven Linux process/resource/termination tests ran. Saved checker self-test passed three groups.
- Frozen supervisor observed collection exit 0/raw wait 0, wall 19.198018779999984 s, user CPU 19.347988 s, system CPU 0.143313 s, peak RSS 34,656 KiB, no timeout. The additional parent observed wrapper exit 0/raw wait 0 and wall 19.234951837002882 s.
- Separate evaluation exit 0 (0.8873839939988102 s); independent replay exit 0 (9.645058818001417 s), PASS for arithmetic/resource evidence and **NO_GO** for the candidate. Replay checked 1,549,368 items, including 227,998 numerical comparisons; maximum absolute difference 2.842170943040401e-14. It did not regenerate RNG or recompute tail DP.
- Preflight, verify-only, tests, self-test, collection, evaluation, independent review and postcheck actual exits/raw waits are all 0. Exact commands, UTC, closed logs and parent wait4 records are in `control/`; frozen child supervision is in `supervision/`. Exit 0 means the declared operation recorded its results, not that scientific gates passed.
- Postcheck preserved all 2,892 prior tracked files (2,672 result/protocol files), all 15 scientific inputs/canonical receipt, and 40,664 historical/cache regular files. Full unrelated private path/hash inventory maps are omitted from `control/pre-execution.summary.json`; the original inventory hash is retained. `postcheck.json` and `compact-execution.json` are unchanged original summaries.

## Archive and copy boundaries

`collection.tar.gz` contains the complete **original synthetic** collection: all 256 case records, all labels, raw synthetic coordinates, candidate matrices and recorded scores/tails/failures, planned cases, collection receipt and selection lock (515 files). No collection bytes were regenerated. The sorted deterministic USTAR/gzip archive has mtime 0; every extracted member was read back and matched its original byte length and SHA256 in [collection-file-manifest.json](collection-file-manifest.json).

To inspect: verify the archive SHA256 from the manifest, then `tar -xzf collection.tar.gz` in a new directory. Verify each extracted `collection/<relative-path>` against the per-file manifest. The original selection lock remains byte-exact inside the archive. Evaluator output is `evaluation/evaluation.json`; independent review is `control/review.json`.

`copy-manifest.json` distinguishes byte-exact copies from the explicitly labeled pre-execution summary derivative. Cloud project/runtime absolute paths are intentionally retained as execution-location evidence; no credentials, signed URLs, upstream datasets, images, weights or unrelated private inventory contents are published. `artifact-hashes.json` excludes itself to avoid circular hashes; source controls and the archive have their own original SHA bindings. The wrapper parent has no recursively added outer exit observer.

AI assisted this research, execution and documentation. New acquired asset bytes: **0**; no additional fees. The older asset ceiling record remains 7,532,690,072 bytes; HPatches metadata has 163,286 measured 206 bytes and an unresolved redirect-body gap, so no verified updated total ceiling is claimed. 0006's stdout evidence gap remains OPEN; 0003 remains deferred; older stopped candidates and results are unchanged. No parameter rescue, additional experiment or manuscript follows from this result.
