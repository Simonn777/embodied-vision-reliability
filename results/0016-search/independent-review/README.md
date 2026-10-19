# Independent saved-only review: 0016

**The diagnostic is complete and its predeclared 2 pp search-opportunity condition fails.** The strongest current deployed selector has AUC@5 **78.2293102381%**. The current normalized backend's nondeployable best-of-four oracle reaches **78.2587614824%**, leaving only **0.02945124425 pp**. This is not an accuracy contribution, evidence that input bias causes the remaining errors, or permission to begin depth-based candidate selection under the existing protocol.

The one scientific run used freeze `055ea70bdfba2a7a3c284063ab91dd1ecb624c27`; its [freeze receipt](../freeze.json) records PR45 and successful CI run39312932830. This review made no HDF5, image, model, or estimator access and imported no experiment/evaluation modules. It read saved JSON/logs and source files only as bytes for hashing.

## Recomputed results

All values below are percentages over **all 64 allocated pairs** in one development scene. Every one of the192 workers and768 native candidates completed; there were no native-failure candidates. All deployed selector identities agree exactly with the independent saved-score selection, including fixed seed ties and inclusive budget boundaries.

| Backend | Seed0 AUC@5 | Full4 AUC@5 | Full4 AUC@10 | Full4 AUC@20 | Nondeployable oracle AUC@5 |
|---|---:|---:|---:|---:|---:|
| Old normalized2.0.5 | 77.8410955985 | 77.8388464390 | 87.4975845582 | 93.6949502955 | 77.8569823881 |
| Current normalized3.0.0 | 78.2283579351 | 78.2293102381 | 87.6935981178 | 93.7932706498 | 78.2587614824 |
| Current TS3.0.0 | 77.2065553215 | 77.1942694239 | 86.3938818276 | 92.5610920745 | 77.2182811313 |

Full4 minus seed0 AUC@5 is −0.00224915947 pp, +0.00095230305 pp, and −0.01228589758 pp respectively. The current normalized oracle exceeds current TS full4 by1.0644920585 pp, but exceeds the stronger current normalized full4 by only0.02945124425 pp. Neither current oracle exceeds **both** current deployed selectors by2 pp. The old-versus-current comparison combines version, defaults and build changes; only current normalized-versus-current TS isolates the TS option in the same build.

| Offline budget (s/pair) | Old normalized coverage | Current normalized coverage | Current TS coverage |
|---|---:|---:|---:|
| 0.1 | 1/64 | 1/64 | 0/64 |
| 0.25 | 40/64 | 41/64 | 7/64 |
| 0.5 | 64/64 | 64/64 | 63/64 |
| 1.0 | 64/64 | 64/64 | 64/64 |
| 2.0 | 64/64 | 64/64 | 64/64 |

All directed and sign-folded summaries, error vectors, threshold counts, coverage counts, candidate dispersions and candidate error ranges were checked. Unavailable budget selections retain180° penalties in the64-pair denominator. Complete tables are in [audit.json](audit.json), checked against [evaluation.json](../evaluation.json).

## Measured cost

The following totals charge all four candidates on all64 pairs, including input preparation, native estimation, validation and scoring. They do **not** substitute the winning candidate's earlier completion time for the cost of searching the full bank.

| Backend | Input (s) | Native calls (s) | Validation/scoring (s) | Full charged total (s) | Sum of worker process walls (s) |
|---|---:|---:|---:|---:|---:|
| Old normalized | 0.130335468 | 53.787468373 | 0.407783669 | 54.325587510 | 63.896876497 |
| Current normalized | 0.125249327 | 54.707264910 | 0.405521593 | 55.238035830 | 64.461395416 |
| Current TS | 0.124953791 | 87.867625833 | 0.403620542 | 88.396200166 | 98.154449999 |

All charged work totals197.959823506 s. The [outer prediction receipt](../prediction.process.json) reports227.078419958 s and exit0; the supervisor's internal seal timer is226.929354916 s. The separate [evaluation receipt](../evaluation.process.json) reports0.570092167 s and exit0. Worker logs contained no warning/traceback lines and the evaluation log is empty. Imports, serialization and process overhead are outside charged time but inside process/outer wall time. Cached feature matching is not measured. Offline prefix costs in the audit use the **last terminal candidate completed within the budget**; this does not constitute an interruptible real-time implementation or zero work when no candidate completed.

## Numerical discrepancies and binding evidence

The first audit correctly stopped on a strict oracle-seed identity mismatch. Independent scalar reductions change the best seed for20 numerically tied nondeployable oracles:14 old normalized,6 current normalized,0 TS. Their maximum independent error regret is3.29809859978e−11 degrees. The exact argmin of the original stored error values is correct in all192 rows. An audit-only clarification now checks both facts and discloses every discrepancy; it preserves the original3e−6° angle tolerance and all deployed selector checks. See [the preserved audit failure/clarification receipt](audit-compatibility-note.json). No frozen prediction, evaluator, metric, threshold or source was changed, and no scientific run was repeated. Seven artificial tests cover the independent audit and this case.

The largest difference across all independently recomputed candidate angle fields is6.95375614729e−11 degrees. SVD reference projection, translation normalization, directed angles, sign-folded angles and DMS threshold-tie/trapezoid AUC were independently implemented. Scalar-best and frozen-oracle AUC agree within the declared numerical tolerance; both give the same failed opportunity decision.

- Prediction seal SHA256: `e6313a51f074b0b0d9dc97cb12b4ccebc6d897a3388f8722bb192118a2794f7e`.
- Evaluation SHA256: `d3e87cc01742e2cd02f36b055d5f92ab67e6764edd173c380bd144a0da778dc5`.
- Audit SHA256: `71275b7cf49066cb9c6331c17808f2deaa747cbd060359afc017fcff3719aaac`.
- All10 declared source/config hashes,192 prediction hashes,192 process receipt hashes, library pins, candidate options and cumulative cost identities passed. The full manifest excludes the previously exposed16 pairs and preserves scene0022.

Residual-score arithmetic itself was **not** recomputed: saved artifacts do not contain the10000 correspondence arrays. The review verifies score copying, bounds, identities and selection use. Saved reference values and allocation declarations are internally verified; original dataset bytes and the complete root-name inventory were not reread. One scene with shared photographs provides no independent-scene generalization or population confidence interval. This failed diagnostic does not satisfy the research goal or justify a manuscript.
