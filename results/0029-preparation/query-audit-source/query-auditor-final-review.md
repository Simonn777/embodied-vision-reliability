# 0029 saved-query auditor: final source delta review

2026-10-20. **No remaining blocker found in the reviewed delta.** This is source review before query-GT access, not a completed numerical result audit.

Reviewed private files:

- `audit_queries_saved.py`: SHA256 `6bb7b6297679d8ee770e58253998986d10aa9e5642f09c1f977df3ce1085af02`.
- `test_audit_queries_saved.py`: SHA256 `7d08f1c29641b9ac254dbbeffab6f2a5fb1d1731fb14ea163e2cea9ad0bbc632`.

Both current hashes match the affected-test receipt. The original source snapshot and six-test history remain under `query-audit-tests-01`; the three affected tests are separately retained under `query-audit-tests-02`. I read that receipt and the tests; I did not execute them. The latter receipt records exit 0, unchanged sources, and 0.26884462498128414 seconds outer wall time.

## Resolved issues

1. **Exact decisions from independently rounded scalars.** The original auditor could reject a legitimate near-tie or exact-zero descriptive sign because its scalar interpolation/softplus/reduction differs in floating arithmetic from the producer. The revised preflight independently checks every candidate loss and unsigned change at the unchanged `atol=rtol=1e-9`, fails before continuing if those comparisons fail, then independently replays the exact fixed selector on the verified saved scalars. Its own raw recomputed losses, choices, gains and sign differences remain visible in `scalar_replay_audit`. Selected-pose/mask metrics continue to evaluate the actual sealed choice. This does not replace the deployed choice or alter scientific gates.

2. **Reduction order at aggregate ties/boundaries.** Independent aggregation now performs `fsum(query_J)/8` within each video and `fsum(video_means)/6` for pooling, matching the declared producer convention without importing its metric implementation. Exact best-control names and Boolean gate checks therefore no longer compare different division orders.

The added near-tie test explicitly covers different recomputed versus saved choices and a zero versus tiny nonzero descriptive loss-gain sign. Existing affected tests cover all 48 rows, rejection before annotation indexing, equal-video aggregation and gate clauses.

The extra receipt correction is accurate: `prediction_admission` now stores the actual verified admission, already checked against the freshly reconstructed admission in `admission()`, rather than the prospective proof object.

## Preserved scope

The delta does not change provenance admission, the all-six clean-operation checks, the pre-GT array/selection barrier, annotation identities, J/void/empty conventions, oracle membership, named controls, thresholds, denominators or numerical comparison tolerances. Frozen sealer and GT-contract code remain reused only for provenance; arithmetic remains independently implemented. Source, model, matching, witness-construction and temporal-carrier execution are not independently repeated by this auditor.

No production file, protocol, original result or test history was edited during this review. No imports, tests, models, media, NPZ arrays, scientific outputs or GT were accessed. The actual saved-output audit still requires parent authorization after all six prediction operations and GT/evaluation operations complete cleanly.
