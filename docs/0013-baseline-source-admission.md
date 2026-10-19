# 0013: source admission for a baseline-only pose diagnostic

Source/metadata inspection on 2026-10-19. This source-inspection step downloaded
no package artifact and read no calibration, ground-truth, or new correspondence
values. Package acquisition/import checks have a separate parent-run receipt.
This document proposes a bounded measurement of the
existing baseline; it reports no pose result or method improvement.

The next useful measurement is one fixed official calibrated-pose baseline on
the same 16 already selected development pairs from scene 0015. It is separate
from failed experiment 0012: none of its candidate scoring code enters this
baseline. A separate frozen protocol must authorize the first camera/R/t reads.
Scene 0022 remains untouched. Sixteen pairs and one seed are descriptive
development evidence, not a population accuracy estimate or a SOTA comparison.

## Version, artifact, and installation scope

Both the official GitHub latest-release endpoint and PyPI metadata currently
identify **PoseLib 2.0.5**. The exact tag is `v2.0.5`, resolving to commit
`7e9f5f53372e43f89655040d4dfc4a00e5ace11c`. The GitHub release was published
2025-08-12; the matching wheel was uploaded 2025-08-27. Use this release rather
than the earlier proposed 2.0.4. [Official release](https://github.com/PoseLib/PoseLib/releases/tag/v2.0.5),
[tag reference](https://api.github.com/repos/PoseLib/PoseLib/git/ref/tags/v2.0.5),
[PyPI metadata](https://pypi.org/pypi/poselib/2.0.5/json).

- Artifact: `poselib-2.0.5-cp39-cp39-macosx_11_0_arm64.whl`.
- Size: **812,666 bytes**; SHA-256:
  `43a2275d8a336e1d6c41557d2c59366b0f8d1716ff14468602143f9af8bb2d59`.
- [Exact wheel URL](https://files.pythonhosted.org/packages/a4/a3/38a001b037aeecd9ea53d115b0932d52cbdac6f9a64b104d3a7af6412ff9/poselib-2.0.5-cp39-cp39-macosx_11_0_arm64.whl).
- Python requirement: `>=3.8`; runtime dependency: `numpy`.
  `pytest>=8.3.5` is only for the optional `test` extra.
- License: [BSD 3-Clause](https://github.com/PoseLib/PoseLib/blob/v2.0.5/LICENSE).

The wheel avoids a local CMake/Eigen/Ceres build. Its import/API compatibility
with the existing Python 3.9 / NumPy 2.0.2 environment still requires an isolated
synthetic check after acquisition is authorized. Preserve the wheel hash, module
version and loaded-module path in the local receipt; do not upgrade NumPy or
install unrelated dependencies implicitly.

## Official API and frozen numerical settings

Call `poselib.estimate_relative_pose(x1, x2, camera1, camera2,
ransac_options, bundle_options, initial_pose=None)`. Supply every existing match
in original order as raw pixel coordinates. Camera dictionaries contain the
author's model, dimensions and parameters; for `PINHOLE`, parameters are
`[fx, fy, cx, cy]`. Do not pre-calibrate points and also pass the actual camera.
[Python bindings](https://github.com/PoseLib/PoseLib/blob/v2.0.5/pybind/pyposelib.cc#L468-L505),
[camera projection](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/misc/colmap_models.cc).

The protocol should cross-check each supplied `K1/K2` against its authored
camera dictionary, fail closed on inconsistency, and seal all 16 prediction
records before a separate evaluator first reads `R/t`. Camera/K validation
does not require ground truth. Do not read confidence or matching-score arrays
for this fixed baseline.

Preserve official defaults explicitly: RANSAC `min_iterations=1000`,
`max_iterations=100000`, `dyn_num_trials_mult=3.0`, `success_prob=0.9999`,
`max_epipolar_error=1.0`, `seed=0`, `progressive_sampling=False`,
`max_prosac_iterations=100000`, `real_focal_check=False`. No initial pose,
confidence filtering, subsampling, seed search, or threshold tuning. Final
refinement uses `CAUCHY`, `max_iterations=100`, `loss_scale=0.5`,
`gradient_tol=1e-10`, `step_tol=1e-8`, `initial_lambda=1e-3`,
`min_lambda=1e-10`, `max_lambda=1e10`, `verbose=False`.
The README's gradient tolerance differs; the tagged implementation is the
authority. The Python wrapper sets the default loss scale to half the RANSAC
threshold, unlike a standalone `BundleOptions()` dictionary's scale of 1.
[Default types](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/types.h#L35-L75),
[option parsing](https://github.com/PoseLib/PoseLib/blob/v2.0.5/pybind/helpers.h#L28-L65).

PoseLib internally unprojects each image separately. With
`f_j=(fx_j+fy_j)/2`, its normalized threshold is
`0.5*(1/f_1+1/f_2)` for a one-pixel input threshold; the loss scale is normalized
the same way. Record this value after authorized camera access. This differs
from DMS's `1/mean(fx1,fy1,fx2,fy2)` convention, so this is not a DMS DDD
reproduction. [PoseLib normalization](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust.cc#L169-L206),
[DMS wrappers](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/estimation/robust.py).

This baseline already generates five-point poses, scores Sampson MSAC with
cheirality, runs 25-step truncated-loss local refinement, and performs final
Cauchy refinement. Returned `info` and its inlier mask are library RANSAC
outputs, not re-evaluated statistics of the final returned pose. In particular,
the final refinement need not refresh score/inlier-ratio fields. Preserve these
fields with that label. [Relative-pose estimator](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/estimators/relative_pose.cc#L39-L72),
[RANSAC final refinement](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/ransac_impl.h#L125-L135),
[cheirality-aware scoring](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/utils.cc).

## Evaluation and resource limits

Match the DMS evaluation convention: project the stored ground-truth rotation
to `U @ Vt` by SVD, normalize ground-truth translation, then use degrees:
`e_R=acos(clip((trace(R_est @ R_gt.T)-1)/2,-1,1))`,
`e_t=acos(clip(dot(t_est,t_gt)/(norm(t_est)*norm(t_gt)),-1,1))`, and
`e=max(e_R,e_t)`. There is **no absolute dot product** or translation-sign
minimization. This is the image-1 to image-2 convention
`X2=R*X1+t`, `E=[t]_x R`. Preserve invalid/zero translations as failures.
Report every pair's rotation, translation and max errors, plus DMS trapezoidal
AUC at 5/10/20 degrees as fractions. Keep the entire 16-pair denominator;
report missing estimates and resource failures separately.
[Author loader](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/dataloader.py#L87-L116),
[author metrics](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/evaluation/metrics.py).

Keep the official 100,000-iteration maximum: it is an iteration limit, not a
wall-clock guarantee. A prospective external limit of 90 seconds per isolated
pair process, sequential execution, and 30 minutes total bounds resource use
without weakening the estimator after seeing its results. Fix native/BLAS
thread environment values to one for this diagnostic. Time camera/input
preparation, the complete public API call, and total execution separately.
If a call times out, retain its timeout status and consumed time; do not rerun
with 10,000 iterations and call it the same baseline. Timeouts alone are not
evidence of a geometric mechanism. The exact watchdog policy belongs in the
new protocol, before the first calibration/ground-truth read.

## Interpretation before seeing errors

The diagnostic can establish whether these selected pairs contain actual pose
failures worth investigating. It cannot establish novelty, recoverability,
held-out improvement, or superiority to DMS CCA. A later candidate needs an
identified source of missing information and strong complete-pipeline controls,
including DMS CCA, with a fresh frozen evaluation plan.

Do not predeclare a generic plane or rotation fix: classic
[DEGENSAC](https://cmp.felk.cvut.cz/~matas/papers/chum-degen-cvpr05.pdf) already uses
plane-and-parallax reasoning, and
[C2P, CVPR 2024](https://arxiv.org/html/2312.05995v2) addresses calibrated pose
constraints and near-pure-rotation behavior. Under exact pure rotation,
epipolar reweighting cannot identify translation from these observations.
Failures must first be measured and causally distinguished; they are not yet
an admitted research mechanism.
