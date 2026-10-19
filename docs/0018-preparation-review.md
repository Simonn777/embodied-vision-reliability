# 0018 preparation review

Reviewed 2026-10-19 before the scientific replay. This receipt records source inspection and artificial checks by the independent preparation reviewer. It does not certify a scientific result or replace the whole-cohort parity/trace checks.

## Supervision and build preparation

The replay supervisor's 14 artificial tests passed, including complete 160-case coverage, raw-gauge/discrete parity, incomplete traces, timeouts/crashes, immutable outputs, hash tampering, safe paths, dynamic serialized-trace budget, and watchdog cleanup. Current source binding explicitly includes `trace0018.h.in`, `trace_pybind.cc.in`, `qualify_native.py`, `qualify_observer.cpp`, and `run_operation.py`. Adding these binding entries changes provenance coverage, not the estimator's numerical settings.

Read-only builder review identified inherited compiler flags, subprocess launch/kill-race receipt handling, deadline scope, and shared-dependency inventory as issues. The builder owner corrected these before execution. Seven mocked tests passed, covering launch failure, group kill/wait races, watchdog failure, pre-launch deadline, dependency mutation, missing/empty dependency roots, and effective compiler flags differing by more than allowed source/build-directory relocation. These tests did not extract an archive, compile code, or invoke any estimator. The actual build and native qualification are separately owned and receipted; they were not executed by this reviewer.

The outer operation wrapper was also reviewed. Its owner corrected tracked/index cleanliness checks, the final elapsed-time check, and cleanup of workers that use separate process groups. The final source stops the controller, refreshes descendant groups, kills observed isolated worker groups on failure even if the controller has already exited, waits for its direct child, and reports any remaining worker groups. An additional mocked probe forced the controller to exit immediately after an RSS violation: the isolated worker still received `SIGKILL`, the direct child was waited, and the failed receipt retained the RSS reason with no remaining groups. This probe launched no real processes. Group RSS is sampled, not a kernel-enforced instantaneous memory ceiling.

## Independent reconstruction barrier review

Static inspection confirms that `diagnose_trace.run` verifies the entire 160-case replay seal before loading observations; rechecks full opaque asset hashes and exact input/calibration diagnostics; reconstructs all 160 cases; then repeats the seal/source/input checks before loading any saved reference pose. It requires the correct all-point versus final-mask support, raw translation gauge, native loss/scaling and accepted-state block costs. Rejected nonfinite trials remain explicit records. Stage ancestry distinguishes discarded or temporarily adopted LOs from the final returned model's lineage.

The diagnosis author's 23 artificial tests passed at the initial review. After the prospective angle-reporting correction, all 24 diagnosis tests passed (1.018 s). Four additional independent probes are now persisted in `test_independent_barrier.py` and passed again (0.798 s):

| Injected failure | Completed reconstructions retained | Saved-label reads |
|---|---:|---:|
| Second complete seal verification fails | 160 | 0 |
| Source binding changes after reconstruction | 160 | 0 |
| Serialized output exceeds prospective cap | 0 | 0 |
| Last trace path attempts parent traversal | 159 | 0 |

All four preserve an ineligible operation receipt, create no reconstruction seal or evaluation, and write no label artifacts. Fixtures are manufactured arrays and opaque text files; no scientific HDF5, image, depth/pose value, or native library is accessed by these tests.

The angle-reporting correction was independently reviewed before scientific replay. Raw geometry/statistical parity remains mandatory at its original componentwise tolerance; the native-cost tolerance is unchanged. Applying that cost tolerance to angles was inappropriate near the `acos` endpoints: an artificial `1e-14` matrix perturbation satisfies raw-state parity but can produce an angular difference above `1e-7` degrees. Final angle differences and frozen/recomputed AUC are now recorded descriptively. The test also confirms a `1e-5` raw-translation mismatch is rejected. This change neither alters an estimate nor relaxes the mechanism gate.

## Reviewed source identities

SHA256 values below identify the reviewed files, not a final Git freeze. The parent performs the final complete test run and publication freeze.

| File under `experiments/0018-depth-trace/` | SHA256 |
|---|---|
| `common.py` | `f8f9d22d4b1db777f743910c90606ac941982bf3d59459b051ceee5819faf4a7` |
| `predict_pair.py` | `7220525e58564d53f80aed247060146712e95a3925d34274d617af020803a03a` |
| `run_diagnostic.py` | `54b68d98e24401eb845832a7b751ffb6095389ef45f8d52076047f284e498769` |
| `build_instrumented.py` | `23d06ecce77bd63cdebee3ce74ec458b02ce713dedb75086cd2ae0047ed16b08` |
| `run_operation.py` | `7a34e2495a2244b94a16d336a81567818e628f4d60bb29640499bf2f774a8211` |
| `test_runner.py` | `825e2a4cf07ea00f89e31463137221515737a6888d9c2c6152ff14e243767027` |
| `test_build_instrumented.py` | `f823c5ef4da1c18f5d700249a416c80310853247840e0094f3705de692c724e6` |
| `test_independent_barrier.py` | `ccaaa6860f07950adba85feff93ff5c64a547d0e0f9374c583caa6a80a914299` |
| `diagnose_trace.py` | `ef9dd1da6fbc094d54fb92504af93dbf63517920469cefa2754d05f1332205ae` |
| `test_diagnose_trace.py` | `5a72417132950887979d22292b1ba92c3b91d424ea7e1ab2ac6ad11bde870fd6` |

Limitations: mocked process and barrier tests establish control-flow behavior, not numerical parity on the 80 scientific pairs. Numerical reconstruction tolerances are cross-implementation engineering tolerances, not formal floating-point proofs. This diagnostic is on permanently exposed development scenes. An observed support/harm association would only admit a separately frozen intervention; it would not establish causality, held-out improvement, novelty, or paper readiness. No production sources were changed during this independent review.
