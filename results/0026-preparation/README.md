# 0026 preparation: artificial qualification only

The first five qualification suites passed **44 tests**: 9 moment-arithmetic,
10 lifted-relaxation, 5 calibration-gradient, 3 optimizer-API, and 17 pipeline tests.
The separate direct-native bridge passed **8 tests**, for **52 passed tests**
across these first test executions. These are manufactured/fake inputs, not
real-pair pose results, a novelty proof, or scientific-run admission.

| First test execution | Tests | Exit | Outer wall seconds | Evidence |
|---|---:|---:|---:|---|
| Moment arithmetic | 9 | 0 | 0.131763 | [receipt](arithmetic-tests-01/receipt.json), [log](arithmetic-tests-01/stderr.log) |
| Lifted relaxation | 10 | 0 | 0.238258 | [receipt](lift-tests-01/receipt.json), [log](lift-tests-01/stderr.log) |
| Calibration controls | 5 | 0 | 0.242047 | [receipt](controls-tests-01/receipt.json), [log](controls-tests-01/stderr.log) |
| Optimizer API | 3 | 0 | 0.785448 | [receipt](optimizer-tests-01/receipt.json), [log](optimizer-tests-01/stderr.log) |
| Pipeline and gates | 17 | 0 | 0.513385 | [receipt](pipeline-tests-01/receipt.json), [log](pipeline-tests-01/stderr.log) |
| Native bridge | 8 | 0 | 0.769232 | [receipt](native-tests-01/receipt.json), [log](native-tests-01/stderr.log) |

The **first native build failed** on three unused parameters in an upstream
PoseLib header under `-Wextra -Werror`. Its [failed receipt](native-build-01/receipt.json),
[compiler log](native-build-01/stderr.log), and [outer source snapshots](native-build-operation-01/sources/)
are preserved. Before a second build, the dependency include roots were marked
`-isystem`; the new bridge retained its own `-Wall -Wextra -Werror`. Upstream
source/archive, algorithms, fixtures and test tolerances were unchanged.
The [second build](native-build-02/receipt.json) completed; its
[outer receipt](native-build-operation-02/receipt.json) records 1.143492 seconds.
The first bridge test execution then passed all 8 tests, including five native
calls on manufactured correspondences and seven pre-native invalid-seed rejections.

The private library is 858,360 bytes, SHA256
`00418b002b005e664e75198c45e4a318d576a2ad04d8d0cb7eb7788affc18621`.
**No library, model, dataset, compiled object, or other binary is uploaded here.**
The bridge notice and original qualification source snapshots are retained.
See the [native qualification note](native-qualification-01.md).

[publication-bindings.json](publication-bindings.json) maps each private-relative
origin and original byte hash/size to its public hash/size. Copies without private
paths are byte-identical. Other UTF-8 text uses only literal path substitutions:
the absolute project prefix becomes `<project>`, then the user-home prefix becomes
`<user-home>`. Original receipts and embedded log/source hashes continue to refer
to the immutable **private original bytes**; the binding manifest supplies hashes
for their public redacted counterparts. Redaction does not recompute experimental
statistics or rewrite historical source fingerprints.

No scientific arrays, GT, images or models were used in these qualification
operations. One manufactured optimizer case and native fixture checks are not
task-accuracy evidence. The [prospective protocol](../../experiments/0026-calibration-moments/protocol.md)
requires a separate exact-source freeze and passing CI before any new 80-pair
scientific operation. The [admission note](../../docs/0026-admission.md) states the
prior-art and evidence limits. No paper or goal completion is claimed.
