# 0025c: post-outcome residual discrepancy, original audit remains failed

The frozen audit under commit `93349f34790496c5c838a2b97f9440a471d6dc68` completed all 132 records / 13,332 solves and independently obtained `gate=False`, but failed three declared numerical comparisons. Its original failed JSON, exit 1, logs and tolerances remain unchanged. This note does not convert that audit into a pass.

All three reported failures (`residual`, `normal_equation`, `residual_norm`) concern record 99, solve 99: **E / clean / N8**, already `unavailable` because numerical rank is **3**, whereas N8 requires 8. Its reported error is **180°**. The six-dimensional nullspace permits a right-singular vector with almost no translation: the saved raw translation norm is `1.155056320713823e-16`. Normalizing that arbitrary vector by `8.657586492250020e15` produces velocity coordinates as large as `8.539971332116705e18`. This does not constitute a physically useful estimate.

A separately recorded, saved-only forensic invocation evaluated the existing A and x with `einsum`, `@`, and exact rational sums of their binary64 values. No SVD, solver, fixture generator, original audit or model was rerun. The saved x exactly matches its declared normalization; the saved residual and normal-equation vector reproduce bitwise with the original `einsum` reduction. Individual residual dot products sum absolute terms of approximately `3.08e13`–`5.64e15` to residuals of approximately `0.02`–`0.79`: cancellation ratios are about `1.4e15`–`1.5e16`.

| Computation on the same saved A and x | Residual Euclidean norm |
| --- | ---: |
| Original saved / reproduced einsum | 1.2844748502283494 |
| Matrix-vector `@` reduction | 1.3150333564374745 |
| Exact binary64-product sums, rounded only for this norm | 1.2504678011456873 |

For example, row 0 is −0.5 by einsum, −0.8438455929758725 by `@`, and approximately −0.47690973439138135 by exact summation. Both floating reductions lose substantial absolute accuracy here. This explains the discrepancy; it does not justify relaxing the frozen tolerance. The audit-wide maximum absolute difference of 16 is a global summary and should not be described as this residual difference, whose maximum saved-versus-`@` component difference in the forensic calculation is approximately 0.34703.

The discrepancy cannot change the declared scientific decision. Rank alone makes this row unavailable, independent of its residual diagnostic or cheirality vote. E-clean N8 has one retained failure, zero AUC@5 and 180° median; it is not in any advance condition. Independent counting from the saved 13,332-row CSV exactly reproduces the original gate records:

| Required D4 group | Useful / 1,120 | Required |
| --- | ---: | ---: |
| G/P | 7 | 144 |
| G/S | 6 | 144 |
| N/P | 12 | 144 |
| N/S | 13 | 144 |
| C/P | 0 | 144 |
| C/S | 0 | 144 |

All three additional S-group conditions fail as well: D4 counts are 6, 13, 0 versus A2 counts 341, 357, 115, while the fixed condition requires `D4_count > 10*A2_count`. No residual value enters these count-based decisions. Thus this fixed four-point initializer is closed by its necessary gate independently of the unresolved clean-audit status. There is no robust-pose improvement, manuscript admission or permission for a larger rendering study.

Evidence is in `residual-forensics-01/analysis.json`, `residual-forensics-execution-01/{receipt.json,stdout.log,stderr.log}`, and `forensic_saved_residual.py` (SHA256 `a9e9794d39e464c03ffbd134fc3ea5099e2c3699c665a4b22e743f233018ff79`). The one forensic invocation exited 0 in 0.105120375 s, with original input hashes unchanged. Its preserved stderr contains a Python `ResourceWarning` about an unclosed CSV file handle; it contains no numerical exception. That warning was not suppressed and the invocation was not repeated. Exact rational arithmetic here diagnoses only the stored floating-point products, not the original ideal physical solution or a new experiment.
