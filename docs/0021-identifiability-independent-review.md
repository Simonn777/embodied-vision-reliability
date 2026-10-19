# Independent review of the 0021 manufactured mathematics

Reviewed 2026-10-19. Scope: source inspection and independent calculations from the saved artificial fixtures/matrices only. No real cache arrays, images, labels, inference, native pose estimation or query-pose fitting. The author's test suite and experiment were not rerun. No experiment source or original output was changed.

**No mathematical or saved-record blocker was found.** This admits the reported algebra check, not a scientific experiment or a new method. The source and [saved results](../results/0021-identifiability/README.md) appropriately distinguish a numerical local-rank witness, an exact planar counterexample and actual pose accuracy.

The independently checked Jacobian is for `pi(R (f x) + t)`, with `f = Phi theta`, features `[d,u*d,v*d,1,u,v]`, left rotation increments, and unit-translation tangent increments. Its three blocks are `Jpi (R x) Phi`, `-Jpi skew(R (f x))`, and `Jpi B`. Rotation does not differentiate the added translation. Free translation has the true scale gauge `(delta theta, delta rotation, delta t) = (theta,0,t)`. Additive depth shift is generally observable; removing it from query-depth shape is an explicitly chosen transfer convention.

An independent NumPy calculation, without importing the author's module, reconstructed the saved analytic unit Jacobian exactly. All six reported feature/pose/joint matrices' singular values and ranks agreed at each fixed relative threshold, `1e-12`, `1e-10`, and `1e-8`. The check exited 0.

| Independent check | Result |
|---|---:|
| Generic feature / pose / unit-joint / free-joint ranks | 6 / 5 / 11 / 11 |
| Maximum difference from saved analytic unit Jacobian | 0 |
| Generic unit-joint condition number | 2184.5362531888727 |
| Plane feature / unit-joint ranks | 5 / 10 |
| Known planar feature-null residual norm | 3.510833468576701e-16 |
| Known planar joint-Jacobian-null residual norm | 1.1559039315265975e-16 |
| Free-translation metric-scale gauge residual norm | 2.9618593647713695e-16 |
| Query null change after independently removing `[f_query,1]` | 0.9834433057467548 |
| Actual query change for the fixed epsilon 0.1 | 0.09834433057467548 |

The condition number depends on parameter units and scaling; full rank does not establish stable calibration. The planar construction preserves reference depths exactly, while distinct off-plane query pixels retain a non-affine depth-shape ambiguity. This is an admissible counterexample even with positive corrected depths and slopes. It is not a measurement of query pose deterioration. The observed finite-difference discrepancies in the author's saved log are at most `1.1135936617279185e-10`; those tests verify derivatives and are not used to determine rank.

All four source hashes matched the unchanged-source receipt:

| File under `experiments/0021-identifiability/` | SHA256 |
|---|---|
| `math_checks.py` | `5b5be06cd27a5ed83fdaccfa606a2c9703a3c026d60e9f0da7b93714c01fe880` |
| `test_checks.py` | `c9c205f1528dc0abf71a76a115b9aa5c8058b3d2bc2c967057f39c2ebe8fed5c` |
| `run_checks.py` | `43e06bb79146e59b40d5596ccf2c2e7f23b7c10f48df6f9ec2702706f5bbb6ce` |
| `README.md` | `a2c9dd7a2207098b173c896c9fc61a431b1b04bbed4e86753b5794fef8cfda1a` |

Saved `mathematics.json` SHA256 is `1354bebf7e69099ffed2772a5dbe86b6ac65b6ee65e00e2cbc711bc86dbca5ec`; `tests.log` SHA256 is `56c0b15b814ba35b44703c20851e40a92a6ae123acdfbfe416b40ba1ef50469d`. Both matched the receipt. The log reports nine tests passing, and the runner/process receipts preserve exit 0. Their separate timing observations are not efficiency evidence.

This does not resolve the [hosted-cache provenance limitation](0021-metadata-independent-review.md), establish shared real depth errors, or distinguish a new estimator from depth-constrained bundle adjustment. A future multi-reference model must retain each edge's positive baseline scale: unit-normalizing every reference translation does not put all triangulated depths in one metric gauge. Any transfer benefit must be compared with joint inference using the same field, observations, priors and charged computation.
