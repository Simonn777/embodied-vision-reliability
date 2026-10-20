# 0027 final saved-evidence review

**Evidence integrity: PASS. Frozen scientific advancement gate: FAIL.**

This review rehashed the saved evidence and read the completed predictions, evaluation, and independent audit. It did not rerun a solver, model, fixture generator, numerical audit, or scientific metric computation. The additional verification was a single standard-library-only metadata/hash/count check: 259 checks, zero failures, process exit 0.

## Provenance and execution

The 17 bound source files match both current bytes and the bytes in frozen commit `7c6144492f6f07247fb603db4bb79e9d1344a853`. The saved exact-commit CI record reports run `39366809207` successful. Generation, prediction, evaluation, and audit agree on the source bindings and expected commit.

The review checked all four operation specs, normalized commands, resource settings, helper hashes, dispatcher hash, logs, output hashes, and predecessor bindings. All four operations completed with exit 0, no termination reason, no remaining descendant PIDs, and no cleanup errors. Each stderr log is empty; saved prediction warnings are empty.

| Operation | Outer wall time, seconds | Peak sampled group RSS, KiB |
|---|---:|---:|
| Generation | 0.554165292 | 33,888 |
| Prediction | 13.715273292 | 39,632 |
| Evaluation | 0.439198208 | 86,192 |
| Independent numerical audit | 0.803000208 | 102,880 |

These are observed outer timings and sampled memory measurements. Each operation had a 30-second timeout, an 8-GiB group RSS limit, and the recorded thread environment set to four.

Key SHA-256 bindings:

| Artifact | SHA-256 |
|---|---|
| Generation manifest | `e97763da38f5d11512415e71a4bc53908469b4588938cb35b4e8948794295f71` |
| Prediction seal | `1feef85e7f0f2477367098add178b684639f68370fd0ff1edc7d9b0a8f31c48d` |
| Evaluation | `d5b69e1e3119e5ba047aad50d07559b0c7fab4f9765e5757a1effa912fa79f06` |
| Independent numerical audit | `b4094a33c0d9c2769c264233188d9d1f2dac981e33d2181f4b1a3a0ef1d189c7` |
| Dispatcher | `819310a8f975552e4d40c9e9743ac67930c472b76953997e8c7665a90b00a9c7` |
| Supervisor | `7c7798fe76756ace692fe9d9a83755d758690c62f0a6cc7f11af655dd5c60efe` |
| Reused monitor | `4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd` |
| Pinned native binary | `3228d488ca11f966378b6b7657ee6ebc5716da474c7ae88fccff7ab433a6981d` |

Truth was only hashed opaquely during this final chain review. The completed frozen evaluator and auditor had already enforced their pre-truth barriers. The auditor's deliberately limited outer-proof scope is complemented here by rehashing the actual specs, helpers, generation chain, and dispatcher.

## Coverage and numerical audit

All 138 prediction rows are present, ordered, and complete. There are 2,070 recorded method attempts, 1,656 native invocations, and 552 method-by-row outcomes.

| Method | Attempts | Returned candidates | Eligible candidates | Rows with a selected candidate |
|---|---:|---:|---:|---:|
| T: shifted tangent | 138 | 129 | 128 | 128 |
| S: scale-only union | 276 | 276 | 276 | 138 |
| V: virtual-point union | 552 | 518 | 514 | 129 |
| D8: physical triples | 1,104 | 851 | 844 | 137 |

The frozen independent numerical audit passed all 41,707 comparisons, including 1,762 eligible candidates and all 552 method outcomes. Its maximum saved discrepancies were `1.4210854715202004e-14` degrees for angles, `1.1102230246251565e-16` for scores, and `1.4210854715202004e-14` percentage points for AUC. The predeclared tolerances were respectively `1e-6` degrees, `1e-10` absolute score, and `1e-5` AUC percentage points.

This is an independent implementation check using a shared numerical environment, not independent-team replication. This final review did not execute that numerical audit again.

## Fixed decision

Clauses 1 and 3 passed; clauses 2 and 4 failed.

- **Clause 1, clean recovery:** all four exact G/M tangent cases passed the fixed `1e-4`-degree threshold.
- **Clause 2, absolute shifted-depth utility:** three of four required groups failed:

| Group | Useful T outcomes / 16 | Required |
|---|---:|---:|
| G, low noise | 16 | 12 |
| G, high noise | 4 | 8 |
| M, low noise | 8 | 12 |
| M, high noise | 1 | 8 |

- **Clause 3, scale-only and offset controls:** passed under the frozen count rules.
- **Clause 4, point controls:** the noninferiority part passed, but the required improvement of at least two useful outcomes over the stronger of V/D8 did not. T improved by one outcome for shifted G/high and zero for shifted M/high.

For context only, over the 128 noisy rows, failure-inclusive empirical step-CDF AUC@5 was T `29.23856375507189%`, S `11.940439211207519%`, V `29.474902125124324%`, and D8 `20.095112719661376%`. These are the frozen empirical AUC values, not the historical trapezoidal DMS convention. They do not replace the failed advancement clauses.

## Interpretation and limits

The exact branch recovers the fixed clean examples, but the noisy neighborhood-derived tangent inputs do not meet the necessary task-utility threshold. The required advantage over the virtual-point control also is absent. The proper conclusion is to close this frozen candidate configuration. This does not prove that every possible tangent-and-depth method is impossible.

The cohort uses manufactured local geometry and perturbations, two fixed utility geometries, sixteen noise draws per group with paired reuse across methods and noise levels, and the same nine observations for fitting local derivatives and scoring roots. It contains no real-image benchmark, learned-gradient validation, RANSAC study, or common nonlinear refinement. It establishes neither population generalization nor a CVPR-ready method gain.

The predeclared pinned native root-compaction limitation remains relevant to V/D8: source analysis identified a conditional accepted-root storage defect, but this study does not establish which study calls triggered it or certify root completeness. No after-outcome repair or replacement was made. Crucially, T already fails the absolute utility clause independently of that comparator limitation. A corrected or independently qualified alternative native control and the outstanding nonminimal F-all comparison would have been required before any advancement from a passing screen; neither is claimed here.

The records support truthful publication of this failed, bounded diagnostic. They do not support manuscript preparation or a claim of localization improvement.
