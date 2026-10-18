# 0005 supervisor review and preparation-gate repair

**Ready for parent review; no remaining demonstrated blocker in this scope.** This is source and synthetic-fixture validation, not permission to execute an unfrozen plan or proof of cloud CPU feasibility. Author's final preparation schema was confirmed stable; its final helper SHA256 is `099be211cb8129737e9a8737d25108bc9c4c3ecfe0944c07fd5852e08f369910`.

## Reproduced issue, then authorized repair

The initial `verify_preparation` accepted two minimal hand-authored `completed` receipts plus one arbitrary overlay file, with no asset records, installation/import child evidence, ledger or RECORD. This was actually reproduced offline and retained in `initial-gap.json`; `supervisor-before.py` preserves the examined source.

At the parent's explicit instruction I edited only `scripts/run_tapnext_capability.py` and added `tests/test_native_capability_supervisor.py`. The exact seven-input freeze function was left unchanged. The strengthened gate now requires:

- both completed stage/schema/config/source bindings and the checkpoint→dependency receipt hash;
- installation/import child JSON equal to the embedded records, actual exit 0/raw wait 0, no parent termination, finite bounded elapsed time, closed timestamps and matching stdout/stderr file hashes;
- exactly the configured stage asset list, successful downloaded/reused states, source/config binding, expected observed byte/hash/count semantics, identical individual asset-record JSON, and real file lengths/hashes;
- immutable per-stage ledger snapshots with exact historical+metadata base, total plan, resource limit, accumulated reservations, actual observed payload, and matching final current ledger;
- complete non-symlink overlay file map, including RECORD, and manifest provenance/hash references.

These checks establish consistency of the retained evidence. They are not signatures and cannot defeat deliberate coordinated fabrication of every file. Actual execution still needs trusted dispatch, raw parent observations and published freeze review.

## Actual validation

`python3 tests/test_native_capability_supervisor.py`: **15 unittest methods passed**, actual subprocess exit **0**, Python3.9.6, measured parent wall **3.984646 s**. This used only temporary Git repositories and tiny generated asset files. Fake adapters did not import runtime/model packages.

Tests covered a valid exact seven-file source/index/HEAD/current chain; all seven worktree and staged mutations; HEAD-only mutation; current/index canonical receipt mutation; missing/extra freeze map entries; invalid commit; and a valid old ancestor paired with new HEAD/current hashes. The latter rejects specifically at the source-commit content comparison.

Preparation tests reject incomplete `completed` claims, failed/unbound stages, altered or nonzero/terminated child receipts, altered child logs, missing/failed/partial/unbound asset records, replaced payloads, ledger count/budget/reservation/current mutations, and changed/extra/symlink overlay entries. Both downloaded and reused asset statuses have a valid synthetic fixture.

An actual temporary CLI `--verify-only` process succeeds with its entire asset cache absent and creates no output. A frozen fake adapter returning `failed` yields actual child process exit **1**. An existing attempt marker rejects before output creation, platform/affinity checks or any child execution. The real adapter's exclusive `marker.open("x")` remains the constructor race barrier.

## Static supervisor checks and limits

Checkpoint path matches preparation's `weights/tapnextpp_ckpt.pt`. Child gets provenance containing freeze SHA, source commit and both preparation receipt hashes. Parent source requires Linux, fixed allowed affinity0–3, 8GiB RLIMIT_AS, five thread variables=4, GNU `timeout --kill-after=5s 1800`, and captures raw wait4 status, exit, wall/CPU/RSS and log hashes. Real adapter also enforces resource/thread/source/asset gates and creates its exclusive marker before model construction. The parent observes the timeout process; this is explicitly labelled in its receipt.

No actual Linux affinity, RLIMIT, GNU timeout, package imports, checkpoint loading or forward was tested here. No network, real asset/GT or cloud access occurred. The supplied canonical receipt SHA must come from the parent's independently confirmed publication; this function verifies Git/current/index binding but does not query a remote host. Config/protocol were still being finalized during review, so this report is not a final published-freeze attestation. No commit or push was made.

Final edited file hashes:

- `scripts/run_tapnext_capability.py`: `3475e043dd81efc843ba948b0a3c298b01627503681ed6c9a1c6a94d38f2c5a5`
- `tests/test_native_capability_supervisor.py`: `02d58dd9b629e92ca3b9c8631e23a77fe405329c460e462c32aee57f0b2425ed`
