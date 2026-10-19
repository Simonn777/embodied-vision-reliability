# Independent saved-result review

All 16 pairs completed, but the predeclared interpretation gate **failed (15/16)**. Pair 13 has `N_objective_increased`: 2212.084830040944 → 2212.0848300409443, exactly one binary64 ULP (4.547473508864641e-13, relative 2.06e-16). N accepted no step on that pair; all 32 own-objective GN stationarity tests passed and no branch exhausted its budget. This numerical-size increase does not show material convergence deterioration, but the frozen strict gate remains failed. No tolerance change or pair exclusion was used.

Freeze: `e5ed651b874cfc6286a6f46940f093470f5b8770`. Independently verified all 16 prediction hashes, baseline pose/mask/calibration/library bindings, and all eight frozen source/protocol hashes. Recomputed directed translation angles, GT rotation projection and trapezoidal AUC from the already-saved 0013 ground truth; maximum angle disagreement 3.32e-12 degrees and AUC disagreement 2.6e-12 percentage points. No HDF5, correspondence access or estimator rerun occurred. Stored objectives/GN quantities were checked for arithmetic consistency; their residual-level calculation was not rerun.

| Arm | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| original | 81.169622478 | 90.584811239 | 95.292405619 |
| native | 81.169622478 | 90.584811239 | 95.292405619 |
| N | 81.169622844 | 90.584811422 | 95.292405711 |
| P | 81.176133020 | 90.588066510 | 95.294033255 |

P changes AUC@5 by only **+0.006510542 percentage points** versus original/native and +0.006510176 versus N. It improves 9 pair errors and worsens 7; mean signed change is -0.000336404 degrees, largest improvement -0.003919568 degrees and largest worsening 0.002586650 degrees. All four arms remain below 5 degrees on every pair. Native takes zero refinement iterations on all 16; N accepts one step in total, while P lowers its own different objective on all 16. Lowering that objective does not establish better ground-truth accuracy.

| Phase | Sum over 16 (ms) | Per-pair median (ms) |
|---|---:|---:|
| native | 4.751417 | 0.286979 |
| N | 58.260625 | 3.548916 |
| P | 454.022376 | 23.631729 |
| objectives | 390.833125 | 24.276312 |

Repaired supervisor wall time: 4.104904 s; subprocess wall sum: 3.831695 s. These phases have different implementations/work and are not a speedup benchmark. The [preserved first failure](../../0014-pixel-noise/predictions/seal.json) completed one additional native call (0.453500 ms) before N failed, so **17 native continuation calls** occurred across attempts. Its N invocation also consumed 0.647125 ms. All repaired subprocesses exited 0; all pair/supervisor logs contain only expected completion JSON, and the evaluation log is empty.

This known-control diagnostic on 16 reused pairs from one scene supplies **no material accuracy improvement, novel method, causal noise attribution, population inference or paper admission**. Shared images and prior development exposure preclude treating these pairs as independent held-out evidence. Keep the failed gate and tiny mixed accuracy changes visible; do not retune convergence on this cohort to turn the gate green.

Evidence: [complete independent arithmetic and all 16 rows](audit.json), [evaluation](../evaluation.json), [seal and prediction hashes](../predictions/seal.json), [supervisor log](../supervisor.log), [empty evaluation log](../evaluation.log), [original saved ground truth](../../0013-baseline-calibration-compatibility/evaluation.json).
