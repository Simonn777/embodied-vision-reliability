# Official DMS integration facts

Read-only inspection, 2026-10-19. DMS commit
`7d5b228de9b6fab460f43e466fb0ed8565ffe137`; PoseLib `v2.0.4`.
No data values read, packages installed, inference run, or repository files changed
during this inspection. This note records integration requirements, not results.

- **The published main arm is CCA:** center sampling, center scoring, approximate
  final refinement. DDD and CCC are the dense and representative controls. Thus
  reversals of our DMS-form approximate scoring are not direct evidence of a
  failure in published CCA. Keep CCA as a complete pipeline comparator; approximate
  scoring is an additional ablation. [Benchmark](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/run_benchmarks.ipynb),
  [dispatch](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/estimation/__init__.py).
- **Calibrated pose uses five-point solutions**, not our eight-point fundamental
  matrices. Both dense and approximate C++ scoring also check cheirality, capping
  geometric inliers that fail it. Plain Sampson lower bounds remain conservative
  for that score, but plain Sampson upper bounds need not. Preserve cheirality via
  exact fallback or separately justified bounds before claiming equivalent pose
  selection. [Estimator and scoring](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/cpp/src/ransac.cc).
- **Coordinate convention:** calibrate by the supplied PINHOLE intrinsics, then
  use threshold `1 / mean(fx1, fy1, fx2, fy2)` for MegaDepth. The current raw-pixel
  F objective is not interchangeable with that normalized E objective. Official
  KMeans uses calibrated 4D coordinates and seed 0. Its representative minimizes
  the sum of the distances in the two images. [Clustering](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/run_clustering.py).
- **Matrix interface:** official `M` is a Cholesky/QR square-root factor with
  `M.T @ M = A.T @ A`, not our Gram matrix. The binding permutes its columns from
  Python row-major to Eigen column-major vectorization. Never pass our moment
  matrix directly to `pysumopt`. [Summary construction](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/clusters.py),
  [permutation](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/cpp/pybind/helpers.h).
- **Sampling and optimization:** the same seed does not align dense and center
  populations. Scores also affect LO triggers and adaptive stopping. DMS LO uses
  25 iterations. Use identical fixed sample streams for causal tests; retain full
  native pipelines for accuracy-time comparison. [RANSAC loop](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/cpp/src/ransac_impl.h).
- **Timing:** the official solver timers omit separately performed clustering,
  summary construction, and some refinement preparation. Record all preparation,
  conversions, solver and refinement costs plus their total. A Python candidate
  versus a C++ reference cannot establish an efficiency gain. [Wrappers](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/estimation/robust.py),
  [refinement preparation](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/estimation/refinement.py).
- **Metrics:** DMS takes max(rotation angle, translation-direction angle), without
  an absolute-dot sign correction; AUC is returned as a fraction. Freeze this
  convention and preserve failures. Average per-seed AUC, not averaged-error AUC.
  [Metrics](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/evaluation/metrics.py).

## Feasible build path

The arm64 host has clang++ and the macOS SDK. CMake/Ninja were absent from PATH;
Eigen/Ceres were absent from standard Homebrew prefixes; the research venv lacked
PoseLib, pybind11, scipy and cv2. Stock DMS requires C++17, Eigen 3.4 and exact
Ceres 2.2, includes malformed requirements pins, and lacks the pybind11 checkout
expected by CMake. Its default LM is custom; making optional Ceres code conditional
is plausible but has not been implemented or parity-checked.
[CMake](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/cpp/CMakeLists.txt),
[requirements](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/requirements.txt).

PoseLib 2.0.4 has a CPython3.9 macOS-arm64 wheel of 781,414 bytes, verified from
PyPI metadata only; NumPy2.0.2 compatibility still needs a synthetic check.
It exposes `relpose_5pt`, `estimate_relative_pose`, and `refine_relative_pose`.
This is a small possible future admission for a strong compiled dense reference,
not a Python score callback facility. A genuine speed comparison still requires
one compiled estimator with controlled scoring, sampling, LO and refinement.
[Bindings](https://github.com/PoseLib/PoseLib/blob/v2.0.4/pybind/pyposelib.cc),
[wheel](https://pypi.org/project/poselib/2.0.4/).

Before this build work, finish the limited same-development numerical diagnostic.
Only if it survives, freeze a five-point headroom diagnostic on the selected
development pairs with calibration available to estimation and R/t restricted to
evaluation. Do not consume the held-out scene or call this confirmation.
