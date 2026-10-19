# Native diagnostic implementation

This is a single-thread C++17 scoring experiment on the already used development
workload, governed by [protocol0012c](../../protocols/0012c-native-scoring.md).
It is not an implementation of complete relative pose, official DMS or SP-RANSAC.
No additional models, training, packages or data are required.

`native_score.cpp` exports full-score validation, initial-cell containment checks
and five selectors through a C ABI. `native_backend.py` builds and loads the
shared library with ctypes. Build products and the local build receipt go outside
Git into the chat's `work/native/` directory. A public receipt copy normalizes
local absolute paths while retaining source/library hashes and compiler options.

```sh
python experiments/0012-score-bounds/native_backend.py --build
python -m pytest -q experiments/0012-score-bounds/test_native_backend.py
```

Modes0/1 are direct and ordinary partial scoring. Mode2 is the original bound
selector with full fallback. Mode3 is ordinary partial scoring after the same
moment-derived incumbent/order, charging its entire tree/coarse-bound work.
Mode4 replaces full fallback with block-partial scoring and is the prospectively
declared primary candidate. Every mode shares the same residual kernel. Modes
using a permutation validate it in a linear native pass; other modes do not pay
unused permutation work. Candidate trees and statistics are rebuilt per call.

Moments are accumulated once in leaves and combined upward, storing45 symmetric
entries and their absolute-product guards. Already computed initial cell bounds
are retained for refinement; only new child evaluations count again. This makes
node counts differ from the Python implementation's repeated work. Direct leaf
scores and later partial fallback both count if both occur. No result is treated
as cached merely because a related computation occurred earlier.

No fast-math or fused contraction is enabled. Guards remain numerical engineering,
not a formal floating-point certificate. Native scalar score reduction can differ
slightly from NumPy; the protocol fixes an absolute/relative comparison tolerance
but requires exact winner indices. Containment validation adds no test tolerance.
Synthetic tests also cover invalid inputs, overflow, degenerate denominators,
scaled/tied matrices, multilevel cells and deterministic counter behavior.

`run_native.py` replays exactly the saved16-pair/64-hypothesis workload. It records
all five modes and three cyclic-order repetitions, timing fresh preparation as
well as scoring. References and containment checks are outside timed selection;
references never enter the selector. The first real run must follow a source
freeze and independent review. Speed and task-quality claims depend on measured
results and subsequent independent experiments, not successful compilation.
