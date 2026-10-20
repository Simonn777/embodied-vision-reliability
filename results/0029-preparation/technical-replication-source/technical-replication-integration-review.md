# All-six technical replication: source-only integration review

Verdict: **no blocking defect identified in the reviewed admission adaptation**. No tests, imports, model operations, media/NPZ decoding, query-GT access, or scientific execution were performed by this reviewer.

Verified current private source hashes under `technical-replication-02/`:

| File | SHA-256 |
|---|---|
| run_video.py | cbebe19b2708835e0a48a4951f2f7d037c287f6740361702de93e2b92ec21b78 |
| seal_queries.py | 10ce2701348fb4fc31c1aea7eac7bb041431c7dcce366c4bdaacd386c8e39fd2 |
| run_evaluation.py | 39117b1b72407cf297f36216cf305335be81d2917a486ae5a4d06674429741cd |
| audit_queries_saved.py | 900862d5fd657a649089d97a282267f523914259d200426242be5d1ebd059aed |
| acquire_query_gt.py | 67ced391f8f911bb93554d35d8ce53d0df028758037b4001e61121dd60e0edc4 |

The six fixed video identities must each have a distinct spec/output, one shared new commit, the replication marker, v2 operation/seal schemas, and exact command/source/output bindings. The old ec94 commit is expressly rejected as the new prediction commit. Existing output directories are refused by the worker wrapper. Original v1 predictions and receipts cannot be admitted as new primary members.

The original ec94 manufactured qualification is accepted only at its explicit commit, with its existing clean outer/output/input/runtime chain. Current scientific files are independently required to equal the original commit's bytes. The c09 evidence, all input inventories, runtime/model bindings, and source fingerprints retain their original checks; this does not reinterpret the failed historical video outer.

`new_outer_contract` binds the actual new supervisor and unchanged monitor hashes to the new commit, verifies command/spec/result/log hashes and resource fields, and requires completed exit 0 with no termination, signals, cleanup errors or residuals. It requires a fresh post-exit empty sample followed by a final empty sample with consistent observed timing. Legacy evidence/qualification continue through their original outer contract.

The GT wrapper remains byte-identical. Its first pre-HTTP callback performs complete all-six verification; later callbacks and postcheck rehash the admitted context and runtime. Evaluation again verifies all six and the GT operation before decoding. The saved auditor also verifies the new evaluation outer before GT decoding. No successful-video subset is introduced.

Direct diffs against the preserved originals show that `run_evaluation.py` and `audit_queries_saved.py` each change only their outer-verifier call; metric/selection/audit arithmetic is unchanged. Complete-video costs still come directly from hash-bound original outer receipts, with query-stage durations labelled overlapping descriptions. Historical failed-attempt costs and shared preparation accounting remain a separate publication/dispatcher responsibility.

This review concerns the proposed private source bytes, not a claim that future dispatch, exact-source CI, or any real operation has already passed. The immutable failed attempt remains ineligible, and its historical process cause remains unresolved.
