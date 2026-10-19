# 0021 manufactured mathematics — first fixed run

Nine artificial tests passed on the first execution. The fixture definitions, formulas and tolerances were written before running; none were changed afterward. This checks algebra and a manufactured rank witness only. It is not a pose-accuracy experiment, evidence of shared errors in real depth maps, or a new-method result.

| Check | Observed result |
|---|---:|
| Generic feature matrix rank | 6 of 6 |
| Generic five-coordinate pose block rank | 5 of 5 |
| Generic joint Jacobian, unit translation | 11 of 11 |
| Generic joint Jacobian, free translation | 11 of 12; one scale gauge |
| Generic unit-joint smallest singular value | 0.0012781294702375568 |
| Plane feature / unit-joint ranks | 5 / 10 |
| Known plane null: reference Jacobian residual norm | 1.1559039315265975e-16 |
| Known plane null: query-depth change after removing global scale and shift | 0.9834433057467553 before multiplying by fixed epsilon 0.1 |
| Free-translation scale-gauge Jacobian residual norm | 2.9618593647713695e-16 |
| Finite scale change: maximum projection difference | 1.6653345369377348e-16 |
| Additive depth shift after projecting out reference-pose derivatives | 0.09804274993716422 |
| Pure-rotation depth-Jacobian norm | 2.2579835797354353e-16 |
| Largest independent finite-difference discrepancy | 1.1135936617279185e-10 |

The generic ranks are unchanged at all three prospectively fixed relative thresholds, 1e-12, 1e-10 and 1e-8. Rank calculations use the analytic Jacobians exclusively. Finite differences are only an independent derivative check.

The plane perturbation preserves reference depths/projections and positive depth/slope on the fixed support, while remaining visible in the distinct query pixels' depth shape after the permitted affine quotient. Thus positive-depth constraints do not remove this particular extrapolation ambiguity. Removing additive shift in that quotient is an intentional transfer convention, even though it is not a reference geometric gauge.

No query pose was fitted, optimized, projected, or compared with a label. Query-depth ambiguity is not a measurement of pose deterioration. All reference/query coordinates are distinct within the plane scene; the separate generic scene is a different manufactured fixture.

`mathematics.json` contains the construction, coefficients, matrices, singular values, null vectors and all reported checks. `tests.log` preserves actual test output and derivative discrepancies. `receipt.json` includes source hashes, environment versions, child-process exit 0, child wall time 0.069892416 s, runner wall time 0.072960958 s, and unchanged-source verification. `tool-execution-receipt.json` separately transcribes the actual tool's exit/status output, without conflating its timing with the runner's monotonic measurement.

There were no failed or discarded runs, no random fixture search, no dependencies installed and no scientific assets read. Independent review is separate from this execution receipt.
