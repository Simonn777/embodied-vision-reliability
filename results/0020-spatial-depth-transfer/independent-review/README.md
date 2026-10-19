# Independent saved-only review: 0020

**Evidence audit PASS; scientific development gate FAIL. Close this fixed formulation.** The reviewer reproduced all three failed quality conditions from the complete saved cohort, without changing the scoring rules or rerunning fits. Scientific freeze: `bec376568729c3a158cd9f1279cd00d3e5f1de76`.

All **80 pairs**, four original candidates, five selectors, 11 evaluated methods and five 16-pair scene summaries are accounted for. Source/library/manifest/bank declarations, raw/final/process/log hashes, complete seals and both successful outer process receipts agree. No worker log contains output. The two 0019 author-loss arms remain benchmark comparators, never selectable candidates.

The review independently checks spatial partition membership/count consistency, derives the deterministic random partition from the saved key, reconstructs weights and validation masses, verifies support-count/mass conservation, checks successful-fit admission, and reconstructs all work counters. There are **3,200 SVD fits: 2,794 completed and 406 unavailable**, all recorded as nonpositive fitted slopes. Failed fits retain full validation mass. Saved depth scores satisfy the forced full-cost lower bounds that can be derived from directional counts and weights. All five selected identities reproduce the declared exact score argmin and fixed tie order.

| Method | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| Pixel MSAC selector | 71.520213 | 82.371613 | 88.579865 |
| Global affine LS selector | 72.569953 | 83.139548 | 88.999627 |
| Spatial weight only | 71.538068 | 82.451197 | 88.600401 |
| Random-point transfer | 72.926812 | 83.317978 | 89.088842 |
| **Spatial transfer (primary)** | **71.107107** | **82.295167** | **88.584304** |
| Fixed E normalized Sampson | 71.488081 | 82.362408 | 88.575262 |
| Fixed E tangent Sampson | 71.322710 | 82.265689 | 88.525076 |
| Fixed original depth scale | 62.383584 | 77.097927 | 85.932518 |
| Fixed original depth affine | 68.972945 | 80.716066 | 87.351192 |
| Fixed author-loss depth scale | 72.293796 | 83.021898 | 88.999563 |
| Fixed author-loss depth affine | 74.071712 | 83.976009 | 89.450049 |

The strongest aggregate comparator is fixed author-loss depth affine. The primary is **2.964605294303 percentage points worse at AUC@5**, versus a required gain of at least 2 points. It wins only in Florence, versus at least two required scenes. Its AUC@20 is **0.865745119877 points worse**, exceeding the permitted 0.5-point loss. It also trails random-point transfer and the geometric-only selector. Coverage and resource checks pass; they do not rescue the quality gate.

| Scene | Primary AUC@5 (%) | Random-point transfer | Fixed affine author loss |
|---|---:|---:|---:|
| Florence | 76.699786 | 76.919521 | 76.039985 |
| Lincoln | 82.357214 | 85.234806 | 82.506712 |
| London | 61.929662 | 67.967918 | 67.510438 |
| Milan | 66.317848 | 66.535078 | 68.106202 |
| Sagrada | 74.131357 | 74.131357 | 82.994450 |

Prediction and evaluation each ran once, exit 0: **28.911574541 s / 94,480 KiB** and **1.110610208 s / 74,992 KiB** outer wall time / sampled peak process-tree RSS. Summed charged scoring-worker time is 27.258136580 s, including overhead; historical generation of all four original candidates costs 57.987685621 s. No selected-candidate-only cost substitution is used. These historical times are not a matched-compute claim, and RSS sampling can miss peaks.

[audit.json](audit.json) retains the full hashes, 80-pair/11-method arithmetic, fit counters, forced-cost bounds, scene summaries, comparisons and gate. [execution.json](execution.json) records the actual independent commands: five artificial tests passed (exit 0; subprocess wall 0.090654500 s), then one saved-only audit passed (exit 0; 1.617221875 s). Logs: [tests.log](tests.log), [audit.log](audit.log). Raw evidence: [prediction seal](../predictions/seal.json), [evaluation](../evaluation.json), [prediction receipt](../prediction-process.json), [evaluation receipt](../evaluation-process.json). Reviewer: [source](../../../experiments/0020-review/review_saved.py).

This is a saved-evidence, selection and metric audit. It reads no HDF5, private coordinate arrays, images or native estimator and imports no production evaluator. **Successful-fit depth residuals and coefficients are not independently recomputed**: the frozen logs contain aggregate depth scores and directional counts, not per-point residuals. Spatial IDs are checked internally, but their original coordinate medians cannot be rederived without private arrays. AUC calculations use the fixed, previously audited 0017/0019 saved per-pair errors, with `1e-10` fraction tolerance; mass checks use `1e-12` tolerance.

All candidates had seen the full pair, and all five scenes remain permanently development data. This failure closes the declared spatial-transfer formulation; it does not prove that every use of depth or spatial structure fails, and it is not a paper result or an achieved research goal.
