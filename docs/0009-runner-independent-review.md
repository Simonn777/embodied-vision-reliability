# 0009 runner independent review

2026-10-18. **APPROVE FOR THE SINGLE BOUNDED TOY SCREEN AFTER CANONICAL FREEZE; no remaining implementation blocker found.** This approves an implementation/design boundary, not a quality result, resource-feasibility result, novelty claim, or practical geometry method. Reviewed the prospective protocol, draft-author critique, final `/tmp/evr-0009-screen-implementation` files, and the current repository supervisor. No repository source was edited by this reviewer.

## Correctness and protocol correspondence

The fixed ten-stream seed derivation, four generators, k schedule, one common shift, bounded per-pair Gaussian rejection, target permutation in the clustered family, 16 cloned base IDs/two copies, and H/GT-independent two-endpoint dedup agree with the written plan. This is static inspection; no protocol seed or scientific generator output was evaluated. All 32 translations retain canonical IDs and a shared shuffled tie order. True labels and identity GT are returned separately and never passed to `score_inputs` or control selection. The collector can hold and save generated labels, so this is an auditable function/stage boundary rather than OS-level secrecy; no metric call is made during collection.

All 32 cheap-control records and their selections precede tail calls. Exact `Fraction` comparisons implement integer rational ordering for tails and ratio, and excess uses `n*N-S`; tails receive native binary lists. A tail tie cannot use support as a secondary rule. Any first or later unsupported tail discards the candidate for the entire case while preserving completed controls; timeout before controls is incomplete, after controls is a tail failure. Invalid probabilities/software errors invalidate study completion. The alarm wraps seed setup, generation, dedup, geometry, hashing, controls, tails and selection; serialization is outside the per-case timer but inside the global supervisor. Component timings are shared work, with no standalone raw/MSAC or speedup claim.

The separate evaluator consumes a completed selection lock, validates its file inventory/hashes and source binding, then joins labels. It checks the fixed identity GT/canonical ID 4, exact candidate pool, full ordered 32-H control record, all five control selections, and valid first-in-order ties. Incomplete/tail-failed records retain fixed denominators. Completion requires every case and every tail; strongest-primary comparison is aggregate across one fixed baseline, and every one of the 15 paired harm limits independently blocks eligibility. The evaluator deliberately returns only `ELIGIBLE_PENDING_EXTERNAL_REVIEW`, never final GO, pending resources and independent saved-result checks.

## Issues identified and closed before freeze

1. Earlier evaluator schema checks could accept duplicated hypothesis IDs, missing control selections, or altered identity GT. Final code rejects these; no outcomes existed when fixed.
2. Earlier normal Python imports could execute a timestamp-valid compiled cache despite hashing the source file. Final loader compiles the rehashed frozen raw bytes in dependency order, then checks every loaded `evr` origin. A deliberate stale-cache fixture passes by executing the intended source.
3. Earlier supervisor did not implement the selection-lock binding required by the lock text. Current `bind_collection` writes a separate binding of the already-closed execution receipt and raw selection-lock bytes after reaping; child-zero plus absent/unsafe lock becomes failure. This binding is distinct from the unmodified execution receipt.

## Actual artificial verification

- Final author test file, independently run with `PYTHONDONTWRITEBYTECODE=1 python3 -W error /tmp/evr-0009-screen-implementation/tests/test_geometric_stage0.py`: **28 tests passed, zero skips, 2.373 s**. Only tiny explicit/FakeRNG fixtures, artificial Git freezes and cache substitution were used.
- Independent `/tmp/evr-0009-runner-extra-tests.py`: **5 tests passed, zero skips, 0.002 s**. They check exact ratio ordering when floats collapse; no larger-support tie fallback; second-H refusal nullifies the whole candidate; changed locked records fail before GT evaluation; and each of all 15 harm limits separately forces NO-GO.
- Current repository `tests/test_0009_supervisor.py`: **15 discovered, 8 passed and 7 Linux-only tests skipped, 0.012 s**. This macOS review does not attest actual Linux kill/affinity/RSS behavior. That remains an execution/receipt check in the approved cloud environment.

These are software tests, not scientific cases or an estimate that the 256-case screen will finish. The final source/config/protocol still must be published and canonically frozen before the one authorized run. Fresh-output refusal prevents accidental overwrite, but does not prove that no separate output directory was ever run; one-run history remains an independently checked operational constraint. Do not silently retry a resource or coverage failure. No data/model/network access or research inference occurred here.

## Final reviewed SHA256 values

- `src/evr/geometric_stage0.py`: `8f77312bc7d293e714fc72628f1d6bb06c2453dd60316e16f67a6738d3d9abec`
- `src/evr/geometric_stage0_evaluation.py`: `b9bb98b2f098f42223d80507d047a81068cf42cf3b801cd860f313986c095039`
- `scripts/run_geometric_stage0.py`: `87aea3b6f9dcee50a5a469ee95ff11e4e58cc283587c081e6fdcc4833942b2e0`
- `tests/test_geometric_stage0.py`: `cc815329292b27aeb4e762ad0b0096c3f75d3d69fc9c60d67c2b84b51a79f4df`
- `configs/0009-geometric-stage0.json`: `71d6d31332968beea2a56b39e5b379bbfc3fea981e7fa8a89cdd0d152b331f93`
- `scripts/supervise_0009.py`: `2d306a8813d36c207eb276a187a298864965bffbe79c1dd463d9482f3bf7d77f`
- `tests/test_0009_supervisor.py`: `a0007e42467955462e60d04651f4271bdb09c4586bebb2775603b820d51491f6`
- `protocols/0009-geometric-stage0.md`: `5f7f543c3f2793b93933c115514a1e39f283e383b659fd1889119b7b5a07b563`

The first five files are the delivered `/tmp` implementation; the supervisor, supervisor tests and protocol are the current repository files. The unchanged geometry/tail helpers were reviewed separately in `/tmp/evr-0009-core-protocol-review.md`.
