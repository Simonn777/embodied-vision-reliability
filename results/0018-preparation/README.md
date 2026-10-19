# 0018 preparation: observer qualified, scientific replay not yet run

2026-10-19. This is preparation for the [prospective trace protocol](../../protocols/0018-depth-lo-trace.md).
No method improvement, scientific160case trace result, or manuscript is claimed.

## Isolated build

The original pinned PoseLib source archive was extracted afresh. All original
source hashes matched the preserved0016 tree. The deterministic observer patch
changed seven files and added one header; the full patch and before/after hashes
are in [instrumentation.json](instrumentation.json). Original source and reused
CMake/Eigen/pybind files were verified unchanged after building. Effective
compiler flags matched0016 byte-for-byte after replacing only source/build path
prefixes. Two jobs, Release/O3, same AppleClang21/SDK27, no fastmath or dependency
upgrade. Build76.933039959s, sampled peak summed process RSS1,602,864KiB; sampling
can miss short peaks. Instrument/configure/build all exited0 on the first attempt.

The separate binary is2,673,976bytes, SHA256
`35d03b03304ebb6249b9139c027ab27eca5bd5a859294324786634e067e337b4`.
The [library config](../../configs/0018-libraries.json) binds both this artifact
and the unchanged original0017 binary. Binaries are local; reproducible source
and build commands are published. Private paths in archived commands/logs were
replaced with placeholders. [Archive index](archive-index.json) distinguishes
original local hashes from hashes of the public redacted copies; a raw log hash
inside a receipt must not be confused with the public redacted log's hash.

## Artificial qualification

[Native qualification](native-qualification.json) executed24 API calls across
three generated48point fixtures and two depth arms: original binary, observer
disabled, observer enabled, and capacity-two overflow. All six cases passed
the original-binary tolerance, exact enabled/disabled/overflow value equality,
unchanged sample/proposal checksums, and independent native-cost reconstruction.
This is not evidence that all real cases will pass the required parity barrier.

The separate [C++ control result](observer-qualification.log) passed nine checks:
accepted steps; zero gradient; small step; relative-cost exit; rejected-step count
reuse; TRUNCATED zero weights; buffer overflow; nonfinite rejected trial; and
literal visibility loss/gain. The last check directly exercises refiner branch
counts. It does not assert that every generated end-to-end fixture contains an
accepted visibility-changing LM step. Original callbacks remain unmodified.

Before scientific freeze, review identified an unnecessary additional angle
gate: acos near its endpoints can magnify tiny component differences. Mandatory
raw-state parity remains unchanged at1e-12; final angle differences are reported
descriptively, without reusing the unrelated cost tolerance. The original
qualification receipts remain intact. [Saved synthetic trace recheck](saved-artificial-recheck.json)
reconstructed all six existing traces under the corrected evaluator; every
reconstruction value was identical except elapsed time. It made zero native
calls and read zero scientific inputs. Current evaluator SHA256 is
`ef9dd1da6fbc094d54fb92504af93dbf63517920469cefa2754d05f1332205ae`.

The complete local artificial suite passed49tests in2.91s, including four independently written whole-cohort label-barrier probes. Separate agents reviewed the source; this qualifies diagnostics, not efficacy.

## Prospective execution

After published-source CI succeeds, run the frozen outer wrapper once in each
mode, passing the exact verified commit to `--expected-commit`. Prediction uses
the inherited0017manifest/assets, all80pairs and both depth arms. Diagnosis is
allowed only after all160 outputs pass raw-state/mask/stat parity, and saved
labels are opened only after all160 independent cost reconstructions pass.
The wrapper records actual child exit, full wall time and sampled group RSS;
both operations retain their own1800s cap. No result-based rerun is admitted.

The [resource screen](../../docs/literature/0018-small-heldout-plan.md) separately
verified a range-based Map-free archive directory, with3,945,798bytes transferred
and no image, calibration, pose or model payload read. All65 scenes remain
unallocated. [Transfer ledger](resource-screen/transfer-ledger.json) reserves the
full4MiB for the directory and separate2MiB source/API allowance, leaving2,764,780,692bytes. A second independent source and inference
runtime still need qualification. [Official stronger controls](../../docs/literature/0018-strong-controls.md)
must be addressed before any future novelty or performance claim.
